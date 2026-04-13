"""
Oracle KV injection probe for BEAM.

This script assumes the BEAM contexts have already been stored by a fresh
benchmark run. It finds stored chunks that lexically overlap the gold answer,
fetches those KV blocks, injects them directly, and prints the answer.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvmemory.adapters.hf_adapter import HFAdapter
from kvmemory.core.injector import inject_and_generate
from kvmemory.storage.kv_store import KVStore
from kvmemory.storage.vector_db import VectorDB
from kvmemory.utils.model_id import sanitize_model_id
from tests.beam_eval import (
    BEAMQuestion,
    QUESTION_TYPES,
    _apply_chat_template,
    load_beam_hf,
    load_beam_jsonl,
    score_answer,
)


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do",
    "for", "from", "how", "i", "in", "is", "it", "me", "my", "of", "on",
    "or", "over", "such", "that", "the", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "with", "you", "your",
}


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def _lexical_score(question: str, gold: str, chunk_text: str) -> float:
    chunk_counts = Counter(_tokens(chunk_text))
    if not chunk_counts:
        return 0.0

    gold_terms = set(_tokens(gold))
    question_terms = set(_tokens(question))
    chunk_terms = set(chunk_counts)

    gold_overlap = len(gold_terms & chunk_terms) / max(len(gold_terms), 1)
    question_overlap = len(question_terms & chunk_terms) / max(len(question_terms), 1)

    score = 3.0 * gold_overlap + question_overlap

    gold_norm = " ".join(_tokens(gold))
    chunk_norm = " ".join(_tokens(chunk_text))
    if gold_norm and gold_norm in chunk_norm:
        score += 5.0

    # Extra nudge for chunks that contain distinctive answer terms more than once.
    score += sum(min(chunk_counts[t], 3) for t in gold_terms & chunk_terms) * 0.03
    return score


def _load_questions(args) -> list:
    if args.run_id:
        run_id = _resolve_run_id(args.obs_dir, args.run_id)
        questions = _load_questions_from_run(args.obs_dir, run_id)
        return questions[: args.n] if args.n else questions
    if args.dataset:
        questions = load_beam_jsonl(args.dataset)
        return questions[: args.n] if args.n else questions
    return load_beam_hf(
        scale=args.scale,
        question_types=QUESTION_TYPES,
        max_questions=args.n,
    )


def _resolve_run_id(obs_dir: str, run_id: str) -> str:
    if run_id != "latest":
        return run_id
    events_dir = Path(obs_dir) / "events"
    event_files = sorted(events_dir.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not event_files:
        raise RuntimeError(f"No event files found under {events_dir}")
    return event_files[0].stem


def _load_questions_from_run(obs_dir: str, run_id: str) -> list[BEAMQuestion]:
    path = Path(obs_dir) / "events" / f"{run_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    questions: dict[str, dict] = {}
    order: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("benchmark") != "kv_memory":
            continue
        question_id = event.get("question_id")
        if not question_id:
            continue
        if question_id not in questions:
            questions[question_id] = {"question_id": question_id}
            order.append(question_id)
        record = questions[question_id]
        record.setdefault("question_type", event.get("question_type") or "unknown")
        if event.get("type") == "question_started":
            record["question"] = event.get("question") or ""
        elif event.get("type") == "score_done":
            record["gold_answer"] = event.get("gold_answer") or ""

    loaded = []
    for question_id in order:
        record = questions[question_id]
        if record.get("question") and record.get("gold_answer"):
            loaded.append(
                BEAMQuestion(
                    question_id=record["question_id"],
                    question_type=record.get("question_type") or "unknown",
                    context="",
                    question=record["question"],
                    gold_answer=record["gold_answer"],
                )
            )
    if not loaded:
        raise RuntimeError(f"No kv_memory question/gold pairs found in run {run_id}")
    return loaded


def _session_id_from_run(obs_dir: str, run_id: str) -> Optional[str]:
    path = Path(obs_dir) / "runs" / f"{run_id}.json"
    if path.exists():
        run = json.loads(path.read_text(encoding="utf-8"))
        session_id = (run.get("metadata") or {}).get("session_id")
        if session_id:
            return session_id
    return None


def _load_points(vector_db: VectorDB, collection_name: str, session_id: Optional[str]) -> list[dict]:
    points: list[dict] = []
    offset = None
    while True:
        page = vector_db.scroll_points(
            collection_name=collection_name,
            limit=256,
            offset=offset,
            with_vectors=False,
            session_id=session_id,
        )
        points.extend(page["points"])
        offset = page.get("next_offset")
        if not offset:
            return points


def _select_oracle_blocks(
    *,
    question,
    points: list[dict],
    max_blocks: int,
    token_budget: int,
) -> tuple[list[str], list[tuple[float, dict]]]:
    scored: list[tuple[float, dict]] = []
    for point in points:
        payload = point.get("payload") or {}
        score = _lexical_score(question.question, question.gold_answer, payload.get("chunk_text") or "")
        if score > 0:
            scored.append((score, point))

    scored.sort(
        key=lambda item: (
            item[0],
            (item[1].get("payload") or {}).get("token_count") or 0,
        ),
        reverse=True,
    )

    selected: list[dict] = []
    tokens_used = 0
    for score, point in scored:
        token_count = (point.get("payload") or {}).get("token_count") or 0
        if token_count <= 0 or tokens_used + token_count > token_budget:
            continue
        selected.append(point)
        tokens_used += token_count
        if len(selected) >= max_blocks:
            break

    selected.sort(key=lambda point: (point.get("payload") or {}).get("created_at") or 0.0)
    selected_ids = [point["id"] for point in selected]
    selected_set = set(selected_ids)
    selected_scored = [(score, point) for score, point in scored if point["id"] in selected_set]
    selected_scored.sort(key=lambda item: (item[1].get("payload") or {}).get("created_at") or 0.0)
    return selected_ids, selected_scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle KV injection probe for BEAM")
    parser.add_argument("--dataset", help="Path to local BEAM JSONL file")
    parser.add_argument("--hf", action="store_true", help="Load BEAM from HuggingFace")
    parser.add_argument("--run-id", help="Load questions/gold answers from an observability run id, or 'latest'")
    parser.add_argument("--obs-dir", default=".kvmem_obs")
    parser.add_argument("--scale", default="100K", choices=["100K", "500K", "1M"])
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--collection", help="Qdrant collection name. Defaults to sanitized model id.")
    parser.add_argument("--session-id", help="Optional session_id filter for stored blocks")
    parser.add_argument("--blob-store-path", default="./kv_store")
    parser.add_argument("--qdrant-url", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument("--max-blocks", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=2000)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    if not args.dataset and not args.hf and not args.run_id:
        parser.error("Specify --run-id, --hf, or --dataset")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    questions = _load_questions(args)
    if not questions:
        raise RuntimeError("No BEAM questions loaded")

    model_id = sanitize_model_id(args.model)
    collection_name = args.collection or model_id
    if args.run_id and args.run_id != "latest":
        resolved_run_id = args.run_id
    elif args.run_id:
        resolved_run_id = _resolve_run_id(args.obs_dir, args.run_id)
    else:
        resolved_run_id = None
    if args.session_id is None and resolved_run_id:
        args.session_id = _session_id_from_run(args.obs_dir, resolved_run_id)

    vector_db = VectorDB(url=args.qdrant_url, port=args.qdrant_port)
    kv_store = KVStore(args.blob_store_path)
    points = _load_points(vector_db, collection_name, args.session_id)
    if not points:
        raise RuntimeError(f"No stored points found in collection {collection_name!r}")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"torch_dtype": dtype_map[args.dtype]}
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if device != "cuda":
        model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapter = HFAdapter(model, tokenizer)

    generation_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
    records = []

    for question in questions:
        selected_ids, selected_scored = _select_oracle_blocks(
            question=question,
            points=points,
            max_blocks=args.max_blocks,
            token_budget=args.token_budget,
        )
        blocks = kv_store.fetch(selected_ids, model_id=model_id)

        prompt = f"{question.question}\nAnswer concisely."
        if hasattr(tokenizer, "apply_chat_template"):
            generation_tokens = _apply_chat_template(
                tokenizer,
                [{"role": "user", "content": prompt}],
            )
        else:
            generation_tokens = tokenizer.encode(prompt)

        output = inject_and_generate(
            adapter=adapter,
            blocks=blocks,
            current_tokens=generation_tokens,
            generation_kwargs=generation_kwargs,
        )
        answer = tokenizer.decode(
            output.sequences[0][len(generation_tokens):],
            skip_special_tokens=True,
        )
        em, f1 = score_answer(answer, question.gold_answer)

        print("\n" + "=" * 90)
        print(f"{question.question_id} [{question.question_type}]")
        print(f"Q: {question.question}")
        print(f"Gold: {question.gold_answer}")
        print(f"Oracle blocks: {len(blocks)} / {len(selected_ids)} requested")
        for rank, (score, point) in enumerate(selected_scored, start=1):
            payload = point.get("payload") or {}
            text = (payload.get("chunk_text") or "").replace("\n", " ")
            print(
                f"  {rank:02d}. score={score:.3f} tokens={payload.get('token_count')} "
                f"id={point['id']} text={text[:240]}"
            )
        print(f"Pred: {answer}")
        print(f"EM={em:.0f} F1={f1:.3f}")

        records.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "gold_answer": question.gold_answer,
                "predicted_answer": answer,
                "em_score": em,
                "f1_score": f1,
                "selected_ids": selected_ids,
                "selected_chunks": [
                    {
                        "id": point["id"],
                        "score": score,
                        "token_count": (point.get("payload") or {}).get("token_count"),
                        "chunk_text": (point.get("payload") or {}).get("chunk_text"),
                    }
                    for score, point in selected_scored
                ],
            }
        )

    if args.output:
        Path(args.output).write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"\nSaved oracle results to {args.output}")


if __name__ == "__main__":
    main()
