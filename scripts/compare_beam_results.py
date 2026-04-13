"""
Compare two BEAM result JSON files.

Usage:
    python scripts/compare_beam_results.py results_beam_local_100K_n5.json results_beam_local_100K_n5_qk.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_kv_run(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "kv_memory" in payload:
        kv = payload["kv_memory"]
        return kv.get("metrics", {}), kv.get("results", [])
    if "kv_metrics" in payload and "kv_results" in payload:
        return payload.get("kv_metrics", {}), payload.get("kv_results", [])
    raise ValueError(f"{path} does not look like a BEAM result JSON")


def _by_id(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["question_id"]): row for row in results}


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "-"


def _short(text: str, limit: int = 120) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _diag(row: dict[str, Any]) -> str:
    diag = row.get("retrieval_diagnostics") or {}
    if not diag:
        return "diag: none"
    parts = [
        f"gold_stage1={diag.get('gold_in_stage1')}",
        f"gold_selected={diag.get('gold_in_selected')}",
        f"best_gold_stage1={diag.get('best_gold_stage1_rank')}",
        f"best_gold_rerank={diag.get('best_gold_rerank_rank')}",
    ]
    return ", ".join(parts)


def _first_selected_preview(row: dict[str, Any]) -> str:
    chunks = row.get("retrieved_chunks") or []
    return _short(chunks[0], 160) if chunks else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two BEAM KV-memory result JSONs")
    parser.add_argument("before", type=Path, help="Baseline/old result JSON")
    parser.add_argument("after", type=Path, help="Experiment/new result JSON")
    parser.add_argument("--before-label", default="before")
    parser.add_argument("--after-label", default="after")
    parser.add_argument("--top", type=int, default=20, help="Max changed rows to print")
    args = parser.parse_args()

    before_metrics, before_results = _load_kv_run(args.before)
    after_metrics, after_results = _load_kv_run(args.after)
    before_by_id = _by_id(before_results)
    after_by_id = _by_id(after_results)
    shared_ids = [qid for qid in before_by_id if qid in after_by_id]

    print("BEAM KV comparison")
    print(f"  {args.before_label}: {args.before}")
    print(f"  {args.after_label}:  {args.after}")
    print(f"  shared questions: {len(shared_ids)}")
    print()
    print("Metrics")
    for key in ("overall_em", "overall_correct_rate", "overall_f1", "avg_prefill_tokens", "avg_latency_ms"):
        before_val = before_metrics.get(key)
        after_val = after_metrics.get(key)
        if key.startswith("overall"):
            before_s = _pct(before_val)
            after_s = _pct(after_val)
        else:
            before_s = "-" if before_val is None else f"{float(before_val):.1f}"
            after_s = "-" if after_val is None else f"{float(after_val):.1f}"
        print(f"  {key}: {before_s} -> {after_s}")

    changed = []
    for qid in shared_ids:
        before = before_by_id[qid]
        after = after_by_id[qid]
        before_pass = bool(before.get("correct"))
        after_pass = bool(after.get("correct"))
        before_f1 = float(before.get("f1_score", 0.0) or 0.0)
        after_f1 = float(after.get("f1_score", 0.0) or 0.0)
        if before_pass != after_pass or abs(after_f1 - before_f1) > 1e-9:
            changed.append((after_f1 - before_f1, qid, before, after))

    changed.sort(key=lambda item: (item[2].get("correct") == item[3].get("correct"), -abs(item[0])))

    print()
    print(f"Changed rows: {len(changed)}")
    for delta, qid, before, after in changed[: args.top]:
        marker = "IMPROVED" if after.get("correct") and not before.get("correct") else (
            "REGRESSED" if before.get("correct") and not after.get("correct") else "F1_CHANGED"
        )
        print()
        print(f"[{marker}] {qid} ({after.get('question_type')}) delta_f1={delta:+.3f}")
        print(f"  Q: {_short(after.get('question', ''))}")
        print(f"  Gold: {_short(after.get('gold_answer', ''))}")
        print(f"  {args.before_label}: pass={before.get('correct')} f1={before.get('f1_score'):.3f} pred={_short(before.get('predicted_answer', ''))}")
        print(f"    {_diag(before)}")
        print(f"    first chunk: {_first_selected_preview(before)}")
        print(f"  {args.after_label}: pass={after.get('correct')} f1={after.get('f1_score'):.3f} pred={_short(after.get('predicted_answer', ''))}")
        print(f"    {_diag(after)}")
        print(f"    first chunk: {_first_selected_preview(after)}")


if __name__ == "__main__":
    main()
