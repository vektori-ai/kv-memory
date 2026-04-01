# kv-memory

Qwen-first, quality-oriented conversational memory scaffolding.

This repo starts with a practical MVP for long conversation memory:
- turn-window memory units (default `window_size=8`, `stride=4`)
- pluggable embedding backends (quality-first HF/Qwen + hash fallback)
- in-memory store with recency + importance scoring
- retrieval API ready to plug into generation pipelines
- Qwen attention hook scaffold for KV projection capture
- past-key-value injection scaffold using captured memory entries
- end-to-end Qwen KV memory engine (retrieve units -> sync KV -> inject -> generate)

## Why "memory unit" matters

A memory unit is the chunk size retrieved as one item.

For conversation:
- `single message` = too brittle
- `turn window` = best v1 default
- `topic segment` = better semantics, more complexity

This MVP uses turn windows first.

## Install

```bash
pip install -e .
```

Optional HF backend:

```bash
pip install -e .[hf]
```

By default, HF embedding auto-download is disabled (`embedding_allow_download=False`) to keep local smoke runs fast. Enable it explicitly in `MemoryConfig` when you want quality-first HF embeddings online.

## Quick demo

```bash
python examples/basic_chat_memory.py
```

KV capture scaffold demo (requires HF deps + model download):

```bash
python3 examples/qwen_kv_capture.py --max-layers 2
```

KV injection scaffold demo (capture memory entries then generate with injected memory):

```bash
python3 examples/qwen_kv_injection.py
```

End-to-end memory chat demo:

```bash
python3 examples/qwen_e2e_memory_chat.py --capture-layers 2 --top-k 2
```

## Evaluation

Run retrieval benchmark (synthetic):

```bash
python3 examples/eval_retrieval_benchmark.py --dataset synthetic --num-samples 100
```

Run LoCoMo evidence-aware retrieval benchmark (small slice):

```bash
python3 examples/eval_retrieval_benchmark.py --dataset locomo --locomo-id Rabbidon/locomo10-flat --split 'train[:6]' --max-questions 240 --top-k 5
```

Latest benchmark outputs are tracked in [docs/EVAL_RESULTS.md](docs/EVAL_RESULTS.md).

Retrieval benchmark (quick local synthetic evaluation):

```bash
python3 examples/eval_retrieval_benchmark.py --dataset synthetic --num-samples 30 --top-k 3
```

Optional LoCoMo loader check (when network and dataset ID are available):

```bash
python3 examples/eval_retrieval_benchmark.py --dataset locomo --locomo-id <hf-dataset-id> --split test[:20]
```

## Current scope

Track implementation progress in [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md).

Implemented now:
- conversation windowing
- memory indexing and retrieval
- recency/importance weighted scoring
- Qwen/HF-compatible embedding interface
- Qwen KV projection hook capture (`k_proj`/`v_proj`)
- memory KV bank + merge to `past_key_values`
- HF memory runner (`capture_memory_entry`, `forward_with_memory`, `generate_with_memory`)
- `QwenKVMemoryEngine` orchestration (`rebuild_indices`, `decide`, `generate`)

## End-to-end flow (current implementation)

1. Add conversation turns.
2. Build turn-window memory units and embed them.
3. Retrieve top-k relevant units for a query.
4. Ensure selected units have captured KV entries in `MemoryKVBank`.
5. Merge selected KV entries into full-layer `past_key_values`.
6. Generate with memory-injected `past_key_values`.

Next:
- KV capture/injection hooks for HF attention blocks
- compression pipeline (layer selection, pooling, quantization)
- LMCache/vLLM backend adapter
