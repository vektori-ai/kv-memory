# Evaluation Results

Last updated: 2026-04-01

## Dataset
- Source: `Rabbidon/locomo10-flat` (Hugging Face)
- Task: evidence-aware retrieval on LoCoMo QA
- Metric definition:
  - `Hit@k`: at least one retrieved memory window overlaps any ground-truth `evidence` dialog id
  - `MRR@k`: reciprocal rank of first evidence-overlapping window
  - Baselines: random-window and recent-window-only retrieval

## Run command examples
- `python3 examples/eval_retrieval_benchmark.py --dataset locomo --locomo-id Rabbidon/locomo10-flat --split 'train[:6]' --max-questions 240 --top-k 3 --window-size 8 --stride 4`
- `python3 examples/eval_retrieval_benchmark.py --dataset locomo --locomo-id Rabbidon/locomo10-flat --split 'train[:6]' --max-questions 240 --top-k 5 --window-size 8 --stride 4`
- `python3 examples/eval_retrieval_benchmark.py --dataset locomo --locomo-id Rabbidon/locomo10-flat --split 'train[:6]' --max-questions 240 --top-k 5 --window-size 10 --stride 5`

## Results
| Config | Questions | Hit@k | MRR@k | Random Hit@k | Recent Hit@k |
|---|---:|---:|---:|---:|---:|
| `window=8, stride=4, top_k=3` | 240 | 0.4417 | 0.3569 | 0.0750 | 0.0250 |
| `window=8, stride=4, top_k=5` | 240 | 0.5542 | 0.3828 | 0.1375 | 0.0458 |
| `window=10, stride=5, top_k=5` | 240 | 0.5625 | 0.3800 | 0.1333 | 0.0500 |

## Full-run (all available rows)
- Source: `Rabbidon/locomo10-flat`, `split=train`
- Coverage: `Rows evaluated=10`, `Questions evaluated=1982`

| Config | Questions | Hit@k | MRR@k | Random Hit@k | Recent Hit@k | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| `window=8, stride=4, top_k=3` | 1982 | 0.4450 | 0.3579 | 0.0459 | 0.0207 | 11.62s |
| `window=8, stride=4, top_k=5` | 1982 | 0.5363 | 0.3785 | 0.1019 | 0.0328 | 11.84s |
| `window=10, stride=5, top_k=5` | 1982 | 0.5489 | 0.4002 | 0.1140 | 0.0499 | 16.32s |

## Notes
- These runs used `prefer_hf_embeddings=False` (hash fallback) for fast local benchmarking.
- Even with hash fallback, retrieval beats both random and recency baselines by a large margin.
- Next quality step: rerun with a stronger embedding backend and compare deltas on the same slice.
