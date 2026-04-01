from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--capture-layers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kv_memory import MemoryConfig, QwenKVMemoryEngine

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    config = MemoryConfig(
        top_k=args.top_k,
        prefer_hf_embeddings=False,
    )
    capture_layers = set(range(max(0, args.capture_layers)))
    engine = QwenKVMemoryEngine(
        model=model,
        tokenizer=tokenizer,
        config=config,
        capture_layers=capture_layers,
    )

    transcript = [
        ("user", "Preference: Keep every UI in dark mode."),
        ("assistant", "Stored: dark mode UI preference."),
        ("user", "Deployment: We run on a single A100 and prioritize answer quality."),
        ("assistant", "Stored: single A100 and quality-first objective."),
        ("user", "Use turn-window memory for retrieval granularity."),
        ("assistant", "Stored: turn-window retrieval units."),
    ]

    for role, content in transcript:
        engine.add_turn(role, content)

    engine.rebuild_indices(sync_kv=True)

    query = "What did I say about UI preference and deployment setup?"
    answer = engine.generate(query, max_new_tokens=args.max_new_tokens)
    decision = engine.decide(query)

    print("Query:", query)
    print("Selected memory unit IDs:", decision.selected_unit_ids)
    print("Selected memory scores:", [round(score, 4) for score in decision.selected_scores])
    print("Answer:", answer)


if __name__ == "__main__":
    main()
