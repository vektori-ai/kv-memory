from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt", default="Summarize why memory units matter for long conversations.")
    parser.add_argument("--max-layers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kv_memory import KVCaptureSession, attach_qwen_kv_projection_hooks

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    capture_layers = set(range(args.max_layers)) if args.max_layers > 0 else None
    session = KVCaptureSession()
    hooks = attach_qwen_kv_projection_hooks(model, session, capture_layers=capture_layers)

    encoded = tokenizer(args.prompt, return_tensors="pt")
    with torch.no_grad():
        model(**encoded)

    hooks.remove()

    records = session.records()
    if not records:
        print("No KV projections captured.")
        return

    for record in records:
        key_shape = tuple(record.key_states[-1].shape) if record.key_states else None
        value_shape = tuple(record.value_states[-1].shape) if record.value_states else None
        print(f"layer={record.layer_idx} key_shape={key_shape} value_shape={value_shape}")


if __name__ == "__main__":
    main()
