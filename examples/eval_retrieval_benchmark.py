from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kv_memory import ConversationMemoryEngine, MemoryConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality for kv-memory")
    parser.add_argument("--dataset", choices=["synthetic", "locomo"], default="synthetic")
    parser.add_argument("--locomo-id", default="Rabbidon/locomo10-flat")
    parser.add_argument("--split", default="train[:4]")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--max-questions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--prefer-hf-embeddings", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def make_synthetic_case(index: int, rng: random.Random) -> dict:
    favorite_color = rng.choice(["teal", "purple", "orange", "indigo", "green"])
    city = rng.choice(["Bengaluru", "Hyderabad", "Mumbai", "Pune", "Chennai"])
    deployment = rng.choice(["single A100", "dual L40S", "single H100", "4xA10"])

    turns: list[tuple[str, str]] = [
        ("user", f"My preferred UI color is {favorite_color}."),
        ("assistant", f"Recorded UI color preference: {favorite_color}."),
        ("user", f"We currently deploy on {deployment} infrastructure."),
        ("assistant", f"Stored deployment environment: {deployment}."),
        ("user", f"I am based in {city}."),
        ("assistant", f"Noted location as {city}."),
    ]

    distractors = [
        "Let us discuss API pagination strategy for logs.",
        "Reminder: monitor token throughput and p95 latency.",
        "We should split retrieval and generation traces.",
        "Please keep model temperature low for determinism.",
        "Think about cache invalidation policies.",
        "Add retries for flaky network calls.",
        "Consider async batching for embed requests.",
        "Add dashboard cards for memory hit rate.",
        "Use a bounded queue for background workers.",
        "Tune beam size only after quality checks.",
    ]

    for text in distractors:
        turns.append(("user", text))
        turns.append(("assistant", "Acknowledged."))

    query = "What did I say about my preferred UI color and deployment setup?"
    answer_markers = [favorite_color, deployment]

    return {
        "id": f"syn-{index}",
        "turns": turns,
        "query": query,
        "answer_markers": answer_markers,
    }


def evaluate_case(case: dict, engine: ConversationMemoryEngine, top_k: int, rng: random.Random) -> dict:
    engine.turns.clear()
    engine.store.clear()

    for role, content in case["turns"]:
        engine.add_turn(role=role, content=content, timestamp=time.time())

    units = engine.rebuild_memory()
    scored = engine.retrieve_with_scores(case["query"], top_k=top_k)
    retrieved = [item.unit for item in scored]

    def contains_answer(unit_text: str) -> bool:
        text_low = unit_text.lower()
        return all(marker.lower() in text_low for marker in case["answer_markers"])

    hit_at_k = any(contains_answer(unit.text) for unit in retrieved)

    rr = 0.0
    for rank, unit in enumerate(retrieved, start=1):
        if contains_answer(unit.text):
            rr = 1.0 / rank
            break

    random_pick = rng.sample(units, k=min(top_k, len(units))) if units else []
    random_hit = any(contains_answer(unit.text) for unit in random_pick)

    return {
        "hit_at_k": hit_at_k,
        "mrr": rr,
        "random_hit_at_k": random_hit,
        "num_units": len(units),
    }


def run_synthetic(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    config = MemoryConfig(
        window_size=args.window_size,
        stride=args.stride,
        top_k=args.top_k,
        prefer_hf_embeddings=args.prefer_hf_embeddings,
        embedding_allow_download=args.allow_download,
    )
    engine = ConversationMemoryEngine(config=config)

    results = []
    for index in range(args.num_samples):
        case = make_synthetic_case(index, rng)
        results.append(evaluate_case(case, engine, args.top_k, rng))

    hit_rate = sum(1 for result in results if result["hit_at_k"]) / len(results)
    random_hit_rate = sum(1 for result in results if result["random_hit_at_k"]) / len(results)
    mrr = statistics.mean(result["mrr"] for result in results)
    avg_units = statistics.mean(result["num_units"] for result in results)

    print("Dataset: synthetic")
    print(f"Samples: {len(results)}")
    print(f"Config: window={args.window_size} stride={args.stride} top_k={args.top_k} prefer_hf={args.prefer_hf_embeddings}")
    print(f"Hit@{args.top_k}: {hit_rate:.4f}")
    print(f"MRR@{args.top_k}: {mrr:.4f}")
    print(f"Random Hit@{args.top_k} baseline: {random_hit_rate:.4f}")
    print(f"Avg memory units per sample: {avg_units:.2f}")


def run_locomo(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except Exception as exc:
        print(f"LoCoMo eval unavailable: datasets import failed: {exc}")
        return

    try:
        dataset = load_dataset(args.locomo_id, split=args.split)
    except Exception as exc:
        print(f"LoCoMo eval unavailable: failed to load dataset '{args.locomo_id}': {exc}")
        print("Tip: this environment may be offline; run synthetic eval for local validation.")
        return

    rng = random.Random(args.seed)
    config = MemoryConfig(
        window_size=args.window_size,
        stride=args.stride,
        top_k=args.top_k,
        prefer_hf_embeddings=args.prefer_hf_embeddings,
        embedding_allow_download=args.allow_download,
    )
    engine = ConversationMemoryEngine(config=config)

    def parse_json_field(value):
        if isinstance(value, str):
            return json.loads(value)
        return value

    def session_sort_key(name: str) -> int:
        # session_12 -> 12
        return int(name.split("_")[-1])

    def flatten_conversation(conversation_obj: dict) -> tuple[list[tuple[str, str]], list[str]]:
        turns: list[tuple[str, str]] = []
        dia_ids: list[str] = []

        session_keys = [
            key for key, value in conversation_obj.items() if key.startswith("session_") and isinstance(value, list)
        ]
        for session_key in sorted(session_keys, key=session_sort_key):
            for turn in conversation_obj.get(session_key, []):
                speaker = str(turn.get("speaker", "user"))
                text = str(turn.get("text", "")).strip()
                if not text:
                    continue
                dia_id = str(turn.get("dia_id", "")).strip()
                turns.append((speaker, text))
                dia_ids.append(dia_id)
        return turns, dia_ids

    total_questions = 0
    hit_count = 0
    random_hit_count = 0
    recent_hit_count = 0
    reciprocal_ranks: list[float] = []
    evaluated_rows = 0

    for row in dataset:
        if total_questions >= args.max_questions:
            break

        try:
            conversation_obj = parse_json_field(row["conversation"])
            qa_list = parse_json_field(row["qa"])
        except Exception:
            continue

        if not isinstance(conversation_obj, dict) or not isinstance(qa_list, list):
            continue

        turns, dia_ids = flatten_conversation(conversation_obj)
        if not turns:
            continue

        engine.turns.clear()
        engine.store.clear()

        now = time.time()
        for index, (speaker, text) in enumerate(turns):
            engine.add_turn(
                role=speaker,
                content=text,
                timestamp=now + index,
                metadata={"dia_id": dia_ids[index]},
            )

        units = engine.rebuild_memory()
        if not units:
            continue

        def unit_dia_ids(unit) -> set[str]:
            start = max(0, unit.turn_start)
            end = min(len(dia_ids) - 1, unit.turn_end)
            if end < start:
                return set()
            return {dia_ids[i] for i in range(start, end + 1) if dia_ids[i]}

        for qa in qa_list:
            if total_questions >= args.max_questions:
                break

            question = str(qa.get("question", "")).strip()
            evidence = qa.get("evidence", [])
            if not question or not isinstance(evidence, list) or not evidence:
                continue

            evidence_set = {str(item).strip() for item in evidence if str(item).strip()}
            if not evidence_set:
                continue

            scored = engine.retrieve_with_scores(question, top_k=args.top_k)
            retrieved_units = [item.unit for item in scored]
            retrieved_sets = [unit_dia_ids(unit) for unit in retrieved_units]

            hit = any(bool(item_set.intersection(evidence_set)) for item_set in retrieved_sets)
            if hit:
                hit_count += 1

            rr = 0.0
            for rank, item_set in enumerate(retrieved_sets, start=1):
                if item_set.intersection(evidence_set):
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)

            random_units = rng.sample(units, k=min(args.top_k, len(units)))
            random_sets = [unit_dia_ids(unit) for unit in random_units]
            if any(bool(item_set.intersection(evidence_set)) for item_set in random_sets):
                random_hit_count += 1

            recent_units = units[-min(args.top_k, len(units)) :]
            recent_sets = [unit_dia_ids(unit) for unit in recent_units]
            if any(bool(item_set.intersection(evidence_set)) for item_set in recent_sets):
                recent_hit_count += 1

            total_questions += 1

        evaluated_rows += 1

    if total_questions == 0:
        print("LoCoMo eval did not find valid QA examples with evidence.")
        return

    hit_rate = hit_count / total_questions
    random_hit_rate = random_hit_count / total_questions
    recent_hit_rate = recent_hit_count / total_questions
    mrr = statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    print("Dataset: locomo")
    print(f"Source: {args.locomo_id} split={args.split}")
    print(f"Rows evaluated: {evaluated_rows}")
    print(f"Questions evaluated: {total_questions}")
    print(f"Config: window={args.window_size} stride={args.stride} top_k={args.top_k} prefer_hf={args.prefer_hf_embeddings}")
    print(f"Hit@{args.top_k} (evidence overlap): {hit_rate:.4f}")
    print(f"MRR@{args.top_k}: {mrr:.4f}")
    print(f"Random Hit@{args.top_k} baseline: {random_hit_rate:.4f}")
    print(f"Recent-only Hit@{args.top_k} baseline: {recent_hit_rate:.4f}")


def main() -> None:
    args = parse_args()
    if args.dataset == "locomo":
        run_locomo(args)
        return
    run_synthetic(args)


if __name__ == "__main__":
    main()
