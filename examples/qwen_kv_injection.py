from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kv_memory import HFMemoryRunner, MemoryKVBank

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    runner = HFMemoryRunner(model, tokenizer)
    bank = MemoryKVBank()

    memory_1 = "User preference: dark mode for all dashboards and coding UI."
    memory_2 = "User deployment: single A100 GPU, optimize quality over latency."

    runner.capture_memory_entry(entry_id="mem-1", text=memory_1, bank=bank)
    runner.capture_memory_entry(entry_id="mem-2", text=memory_2, bank=bank)

    query = "What UI preference and deployment setting did I tell you?"
    answer = runner.generate_with_memory(
        query_text=query,
        bank=bank,
        entry_ids=["mem-1", "mem-2"],
        max_new_tokens=args.max_new_tokens,
    )

    print("Query:", query)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
