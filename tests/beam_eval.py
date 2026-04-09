"""
beam_eval.py — BEAM benchmark runner. Phase 3.

Measures:
  - Answer accuracy per question type (IE, multi-hop, knowledge update, temporal)
  - Prefill token count per turn (cost reduction metric)
  - Latency breakdown (stage1, stage2, fetch, inject, generate)

Baselines:
  - RAG (re-prefill retrieved text every turn)
  - Sliding window context

Target: beat RAG on information extraction and multi-hop reasoning categories.

Usage:
    python -m tests.beam_eval --model meta-llama/Llama-3-8B --dataset beam_v1.jsonl

Phase 3 target: implement when Phase 2 benchmarks confirm basic system correctness.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# BEAM question types
# ------------------------------------------------------------------

QUESTION_TYPES = ["information_extraction", "multi_hop", "knowledge_update", "temporal"]


@dataclass
class BEAMQuestion:
    question_id: str
    question_type: str
    context: str
    question: str
    gold_answer: str


@dataclass
class BEAMResult:
    question_id: str
    question_type: str
    predicted_answer: str
    gold_answer: str
    correct: bool
    prefill_tokens: int
    latency_ms: float
    stage1_ms: float = 0.0
    stage2_ms: float = 0.0
    fetch_ms: float = 0.0
    inject_ms: float = 0.0
    generate_ms: float = 0.0


@dataclass
class BEAMMetrics:
    accuracy_by_type: dict[str, float] = field(default_factory=dict)
    overall_accuracy: float = 0.0
    avg_prefill_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    prefill_reduction_pct: float = 0.0  # vs RAG baseline


# ------------------------------------------------------------------
# Dataset loader
# ------------------------------------------------------------------

def load_beam_dataset(path: str) -> list[BEAMQuestion]:
    """
    Load BEAM dataset from JSONL file.
    Expected format per line:
        {"id": "...", "type": "...", "context": "...", "question": "...", "answer": "..."}
    """
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            questions.append(BEAMQuestion(
                question_id=d["id"],
                question_type=d.get("type", "unknown"),
                context=d["context"],
                question=d["question"],
                gold_answer=d["answer"],
            ))
    return questions


# ------------------------------------------------------------------
# Answer correctness
# ------------------------------------------------------------------

def is_correct(predicted: str, gold: str) -> bool:
    """
    Loose string match: predicted contains gold answer (case-insensitive).
    Phase 3: replace with F1/EM for production accuracy measurement.
    """
    return gold.lower().strip() in predicted.lower().strip()


# ------------------------------------------------------------------
# Latency instrumented generation
# ------------------------------------------------------------------

async def run_kv_memory_eval(
    questions: list[BEAMQuestion],
    memory,
    session_id: str = "beam_eval",
) -> list[BEAMResult]:
    """
    Run KVMemory system on BEAM questions.
    Instruments latency at each pipeline stage.
    """
    results = []

    for q in questions:
        # Store context first
        await memory.store(
            text=q.context,
            session_id=session_id,
            explicit_signal=1.0,
        )

        # Timed generation
        t0 = time.perf_counter()
        out = await memory.generate(
            prompt=q.question,
            session_id=session_id,
            generation_kwargs={"max_new_tokens": 50, "do_sample": False},
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        results.append(BEAMResult(
            question_id=q.question_id,
            question_type=q.question_type,
            predicted_answer=out.text,
            gold_answer=q.gold_answer,
            correct=is_correct(out.text, q.gold_answer),
            prefill_tokens=len(memory.adapter.tokenizer.encode(q.question)),
            latency_ms=latency_ms,
        ))

    return results


async def run_rag_baseline(
    questions: list[BEAMQuestion],
    adapter,
    generation_kwargs: Optional[dict] = None,
) -> list[BEAMResult]:
    """
    RAG baseline: prepend full context as text prefix every turn.
    Measures cost of re-prefilling context on each query.
    """
    if generation_kwargs is None:
        generation_kwargs = {"max_new_tokens": 50, "do_sample": False}

    results = []
    for q in questions:
        context_tokens = adapter.tokenizer.encode(q.context)
        query_tokens = adapter.tokenizer.encode(q.question)
        combined = context_tokens + query_tokens

        t0 = time.perf_counter()
        out = adapter.inject_and_generate([], combined, generation_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000

        results.append(BEAMResult(
            question_id=q.question_id,
            question_type=q.question_type,
            predicted_answer=out.text,
            gold_answer=q.gold_answer,
            correct=is_correct(out.text, q.gold_answer),
            prefill_tokens=len(combined),  # RAG pays full prefill every turn
            latency_ms=latency_ms,
        ))

    return results


# ------------------------------------------------------------------
# Metrics computation
# ------------------------------------------------------------------

def compute_metrics(results: list[BEAMResult]) -> BEAMMetrics:
    if not results:
        return BEAMMetrics()

    by_type: dict[str, list[bool]] = {}
    for r in results:
        by_type.setdefault(r.question_type, []).append(r.correct)

    accuracy_by_type = {qt: sum(v) / len(v) for qt, v in by_type.items()}
    overall = sum(r.correct for r in results) / len(results)
    avg_prefill = sum(r.prefill_tokens for r in results) / len(results)
    avg_latency = sum(r.latency_ms for r in results) / len(results)

    return BEAMMetrics(
        accuracy_by_type=accuracy_by_type,
        overall_accuracy=overall,
        avg_prefill_tokens=avg_prefill,
        avg_latency_ms=avg_latency,
    )


def print_comparison(
    kv_metrics: BEAMMetrics,
    rag_metrics: BEAMMetrics,
    label_kv: str = "KV Memory",
    label_rag: str = "RAG Baseline",
) -> None:
    print(f"\n{'=' * 60}")
    print(f"BEAM Benchmark Results")
    print(f"{'=' * 60}")
    print(f"{'Metric':<35} {label_kv:>12} {label_rag:>12}")
    print(f"{'-' * 60}")
    print(f"{'Overall Accuracy':<35} {kv_metrics.overall_accuracy:>11.1%} {rag_metrics.overall_accuracy:>11.1%}")
    print()

    for qt in QUESTION_TYPES:
        kv_acc = kv_metrics.accuracy_by_type.get(qt, float("nan"))
        rag_acc = rag_metrics.accuracy_by_type.get(qt, float("nan"))
        label = qt.replace("_", " ").title()
        print(f"  {label:<33} {kv_acc:>11.1%} {rag_acc:>11.1%}")

    print()
    print(f"{'Avg Prefill Tokens':<35} {kv_metrics.avg_prefill_tokens:>11.0f} {rag_metrics.avg_prefill_tokens:>11.0f}")

    if rag_metrics.avg_prefill_tokens > 0:
        reduction = 1 - kv_metrics.avg_prefill_tokens / rag_metrics.avg_prefill_tokens
        print(f"{'Prefill Reduction':<35} {reduction:>11.1%}")

    print(f"{'Avg Latency (ms)':<35} {kv_metrics.avg_latency_ms:>11.0f} {rag_metrics.avg_latency_ms:>11.0f}")
    print(f"{'=' * 60}")


# ------------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------------

async def main(args) -> None:
    logging.basicConfig(level=logging.INFO)

    questions = load_beam_dataset(args.dataset)
    logger.info("Loaded %d BEAM questions", len(questions))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvmemory import KVMemory, KVMemoryConfig
    from kvmemory.adapters.hf_adapter import HFAdapter

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    adapter = HFAdapter(model, tokenizer)

    config = KVMemoryConfig(
        model_id=args.model.replace("/", "_"),
        retrieval_layers=args.retrieval_layers,
        token_budget=args.token_budget,
    )
    memory = KVMemory(adapter=adapter, config=config)

    logger.info("Running KV Memory evaluation...")
    kv_results = await run_kv_memory_eval(questions, memory)

    logger.info("Running RAG baseline...")
    rag_results = await run_rag_baseline(questions, adapter)

    kv_metrics = compute_metrics(kv_results)
    rag_metrics = compute_metrics(rag_results)

    print_comparison(kv_metrics, rag_metrics)

    if args.output:
        import json as _json
        with open(args.output, "w") as f:
            _json.dump({
                "kv_memory": {
                    "accuracy_by_type": kv_metrics.accuracy_by_type,
                    "overall_accuracy": kv_metrics.overall_accuracy,
                    "avg_prefill_tokens": kv_metrics.avg_prefill_tokens,
                    "avg_latency_ms": kv_metrics.avg_latency_ms,
                },
                "rag_baseline": {
                    "accuracy_by_type": rag_metrics.accuracy_by_type,
                    "overall_accuracy": rag_metrics.overall_accuracy,
                    "avg_prefill_tokens": rag_metrics.avg_prefill_tokens,
                    "avg_latency_ms": rag_metrics.avg_latency_ms,
                },
            }, f, indent=2)
        logger.info("Results saved to %s", args.output)

    await memory.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEAM benchmark runner for KV Memory System")
    parser.add_argument("--model", required=True, help="HuggingFace model name or path")
    parser.add_argument("--dataset", required=True, help="Path to BEAM dataset JSONL file")
    parser.add_argument(
        "--retrieval-layers",
        type=int,
        nargs="+",
        default=[8, 16, 24],
        help="Layers to use for retrieval (default: 8 16 24)",
    )
    parser.add_argument("--token-budget", type=int, default=2000)
    parser.add_argument("--output", help="Path to save JSON results")
    args = parser.parse_args()

    asyncio.run(main(args))
