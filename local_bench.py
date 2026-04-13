"""
local_bench.py — Same benchmark as modal_bench.py but runs locally (no Modal).

Model : Qwen2.5-7B-Instruct (loaded from local path)
GPU   : tested on A4500 (20 GB VRAM), float16

Usage:
    python local_bench.py            # all 10 questions
    python local_bench.py --quick    # first 2 questions only (smoke test)
    python local_bench.py --clean    # delete stale kv_store blobs, then run all
    python local_bench.py --quick --clean
"""

import argparse
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvmemory.config import KVMemoryConfig
from kvmemory.adapters.hf_adapter import HFAdapter
from kvmemory.memory import KVMemory
from kvmemory.core.retrieval import (
    stage1_coarse,
    stage2_rerank_mmr,
    _default_layer_weights,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "models/Qwen2.5-7B-Instruct"   # local path — no HF download
RETRIEVAL_LAYERS = [8, 16, 24]             # Qwen2.5-7B has 28 layers (0-indexed)
STALE_KV_DIR = Path("kv_store")           # leftover blobs from prior runs

# (fact, question, answer_keywords)
BENCH = [
    (
        "Marie Curie was born on November 7, 1867, in Warsaw, Poland. "
        "She was the first woman to win a Nobel Prize.",
        "In what city was Marie Curie born?",
        ["warsaw"],
    ),
    (
        "The Amazon River is approximately 6,400 kilometers long, "
        "making it the longest river in South America.",
        "How long is the Amazon River?",
        ["6400", "6,400"],
    ),
    (
        "Python programming language was created by Guido van Rossum "
        "and first released in 1991.",
        "Who created the Python programming language?",
        ["guido", "rossum"],
    ),
    (
        "The human body has 206 bones in total. "
        "The femur, located in the thigh, is the longest bone.",
        "How many bones does the human body have?",
        ["206"],
    ),
    (
        "Mount Everest, located in the Himalayas on the border of Nepal and Tibet, "
        "stands at 8,849 meters above sea level.",
        "How tall is Mount Everest in meters?",
        ["8849", "8,849"],
    ),
    (
        "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
        "What is the speed of light in km/s?",
        ["299", "299,792", "299792"],
    ),
    (
        "William Shakespeare was born in Stratford-upon-Avon, England, in April 1564. "
        "He wrote 37 plays and 154 sonnets.",
        "In which town was William Shakespeare born?",
        ["stratford", "avon"],
    ),
    (
        "The Great Barrier Reef, located off the coast of Queensland, Australia, "
        "is the world's largest coral reef system, stretching over 2,300 kilometers.",
        "In which country is the Great Barrier Reef located?",
        ["australia"],
    ),
    (
        "Albert Einstein published his special theory of relativity in 1905. "
        "His famous equation E=mc² describes the equivalence of mass and energy.",
        "What year did Einstein publish special relativity?",
        ["1905"],
    ),
    (
        "The Sahara Desert covers approximately 9.2 million square kilometers, "
        "making it the largest hot desert on Earth.",
        "How many square kilometers does the Sahara Desert cover?",
        ["9.2", "9,200,000"],
    ),
]

GEN_KWARGS = {"max_new_tokens": 40}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def score(answer: str, keywords: list) -> bool:
    a = answer.lower()
    return any(k.lower() in a for k in keywords)


def clean(text: str, prompt: str) -> str:
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


def check_stale_blobs(do_clean: bool) -> None:
    if not STALE_KV_DIR.exists():
        return
    pkls = list(STALE_KV_DIR.rglob("*.pkl"))
    if not pkls:
        return
    if do_clean:
        shutil.rmtree(STALE_KV_DIR)
        print(f"  [clean] Removed {len(pkls)} stale .pkl files from {STALE_KV_DIR}/\n")
    else:
        print(
            f"  [warn] {len(pkls)} stale .pkl files found in {STALE_KV_DIR}/ "
            f"(from prior runs — harmless, but run with --clean to remove)\n"
        )


def build_memory(adapter, bench, final_top_k: int, min_relevance: float) -> KVMemory:
    tmp = tempfile.mkdtemp()
    cfg = KVMemoryConfig(
        model_id="bench",
        retrieval_layers=RETRIEVAL_LAYERS,
        qdrant_url=":memory:",
        blob_store_path=tmp,
        importance_threshold=0.0,
        dedup_threshold=1.01,
        final_top_k=final_top_k,
        min_relevance=min_relevance,
    )
    mem = KVMemory(adapter=adapter, config=cfg)

    async def _store():
        for fact, _, _ in bench:
            await mem.store(fact, session_id="bench", explicit_signal=1.0)
        await mem.drain_writes(timeout=120)

    asyncio.run(_store())
    return mem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="KV-memory local benchmark")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run on first 2 questions only (smoke test)",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete stale kv_store blobs from prior runs before starting",
    )
    args = parser.parse_args()

    bench = BENCH[:2] if args.quick else BENCH
    mode  = "QUICK (2 questions)" if args.quick else f"FULL ({len(bench)} questions)"

    check_stale_blobs(args.clean)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: no CUDA found, running on CPU — will be very slow")

    print(f"Model: {MODEL_ID}  |  Mode: {mode}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    adapter = HFAdapter(model, tokenizer)
    n = len(bench)

    # -----------------------------------------------------------------------
    # BASELINE
    # -----------------------------------------------------------------------
    print("=" * 64)
    print("BASELINE — raw generate, no memory")
    print("=" * 64)
    baseline_scores = []
    for fact, question, keywords in bench:
        tokens = tokenizer.encode(question)
        out = adapter.generate(tokens, GEN_KWARGS)
        answer = clean(out.text, question)
        ok = score(answer, keywords)
        baseline_scores.append(ok)
        print(f"  {'✓' if ok else '✗'}  {question}")
        if not ok:
            print(f"       → {answer[:90]}")

    baseline_correct = sum(baseline_scores)
    print(f"\n  Score: {baseline_correct}/{n} ({100 * baseline_correct // n}%)\n")

    # -----------------------------------------------------------------------
    # RETRIEVAL DIAGNOSTICS
    # -----------------------------------------------------------------------
    print("=" * 64)
    print("RETRIEVAL DIAGNOSTICS  (top_k=10, no min_relevance)")
    print("Shows which stored fact is retrieved for each question")
    print("=" * 64)
    diag_mem = build_memory(adapter, bench, final_top_k=10, min_relevance=0.0)
    layer_weights = _default_layer_weights(RETRIEVAL_LAYERS)

    async def _diagnose():
        for fact, question, keywords in bench:
            q_tokens = tokenizer.encode(question)
            sf = {
                "model_id": "bench",
                "session_id": "bench",
                "agent_id": None,
                "retrieve_shared": False,
            }
            cand_ids, qvecs = await stage1_coarse(
                query_tokens=q_tokens,
                adapter=adapter,
                config=diag_mem.config,
                vector_db=diag_mem.vector_db,
                session_filter=sf,
            )
            candidates = diag_mem.vector_db.fetch_with_vectors("bench", cand_ids)
            scored = []
            for c in candidates:
                sim = sum(
                    w * float(
                        np.dot(
                            qvecs[l],
                            np.array(c["vector"][f"layer_{l}"], dtype=np.float32),
                        )
                    )
                    for l, w in layer_weights.items()
                    if f"layer_{l}" in (c["vector"] or {})
                )
                text = (c["payload"] or {}).get("chunk_text", "")[:60]
                scored.append((sim, text))
            scored.sort(reverse=True)

            print(f"\n  Q: {question}")
            for rank, (sim, txt) in enumerate(scored[:3], 1):
                print(f"    #{rank} sim={sim:.3f}  '{txt}...'")

    asyncio.run(_diagnose())

    # -----------------------------------------------------------------------
    # SWEEP
    # -----------------------------------------------------------------------
    configs = [
        ("top_k=10, no cutoff",   10, 0.0),
        ("top_k=3,  no cutoff",    3, 0.0),
        ("top_k=1,  no cutoff",    1, 0.0),
        ("top_k=3,  min_rel=0.1",  3, 0.1),
        ("top_k=3,  min_rel=0.2",  3, 0.2),
    ]

    all_results = {}
    for label, top_k, min_rel in configs:
        mem = build_memory(adapter, bench, final_top_k=top_k, min_relevance=min_rel)

        async def _eval(mem=mem):
            results = []
            for fact, question, keywords in bench:
                out = await mem.generate(
                    prompt=question,
                    session_id="bench",
                    generation_kwargs=GEN_KWARGS,
                )
                answer = clean(out.text, question)
                results.append(score(answer, keywords))
            return results

        scores = asyncio.run(_eval())
        correct = sum(scores)
        all_results[label] = (correct, scores)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 64)
    print("RESULTS SUMMARY")
    print("=" * 64)
    print(f"  {'Config':<28}  {'Score':>6}  {'vs Baseline':>12}")
    print(f"  {'-'*28}  {'-'*6}  {'-'*12}")
    print(f"  {'Baseline (no memory)':<28}  {baseline_correct}/{n} ({100*baseline_correct//n:2d}%)")

    best_correct = -1
    best_label = ""
    for label, (correct, _) in all_results.items():
        delta = correct - baseline_correct
        sign = "+" if delta >= 0 else ""
        print(f"  {label:<28}  {correct}/{n} ({100*correct//n:2d}%)  {sign}{delta:>11}")
        if correct > best_correct:
            best_correct = correct
            best_label = label

    print(f"\n  Best config: {best_label}  ({best_correct}/{n})")
    best_scores = all_results[best_label][1]

    helped     = [bench[i][1] for i in range(n) if best_scores[i] and not baseline_scores[i]]
    hurt       = [bench[i][1] for i in range(n) if not best_scores[i] and baseline_scores[i]]
    both_wrong = [bench[i][1] for i in range(n) if not best_scores[i] and not baseline_scores[i]]

    if helped:
        print(f"\n  Memory HELPED ({len(helped)}):")
        for q in helped:
            print(f"    + {q}")
    if hurt:
        print(f"\n  Memory HURT ({len(hurt)}):")
        for q in hurt:
            print(f"    - {q}")
    if both_wrong:
        print(f"\n  Both wrong ({len(both_wrong)}):")
        for q in both_wrong:
            print(f"    ? {q}")

    verdict = "WORKS" if best_correct > baseline_correct else (
        "NEUTRAL" if best_correct == baseline_correct else "BROKEN"
    )
    print(f"\n  Overall verdict: KV Memory is {verdict}\n")


if __name__ == "__main__":
    main()
