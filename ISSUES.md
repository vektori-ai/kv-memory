# KV Memory — Current Issues

## 1. Retrieval Collapse (Critical)

**What:** Cosine similarities for ALL stored blocks are 0.9994–0.9996 regardless of content — only ~0.0002 margin between top and bottom candidate. Vector retrieval is essentially random.

**Why:** Short synthetic facts (1–2 sentences) produce nearly identical hidden state distributions. Most of the signal is in structural/positional tokens, not content. Entropy weighting helps but not enough at this scale.

**Workaround in place:** BM25 keyword reranker on `chunk_text` payload. Works for synthetic QA where question words appear in the context. Will fail on paraphrased or multi-hop queries.

**Real fix needed:** Either (a) use a dedicated embedding model (e.g. `bge-m3`, `e5-mistral`) for retrieval vectors instead of the generative model's hidden states, or (b) test on longer documents where hidden states are more separable.

---

## 2. Wrong Answer From Correct Block (~30% of cases)

**What:** Retrieval finds the right block, injection works, but model picks the wrong fact from within the context.

**Example:** Context = "Alice works as a software engineer… leads the backend team." Gold = "software engineer". Pred = "manager of the backend team".

**Why:** The 7B-Instruct model reads the injected KV correctly but generates a plausible-sounding answer from the same context rather than the exact gold phrase. Also: the model says "manager" when the context says "leads" — possible attention confusion between injected KV positions and prefill positions.

**Fix direction:** More constrained generation prompt (e.g. "Answer in exactly one word/phrase:"), or few-shot examples in the system message.

---

## 3. `!` Token Artifacts in 7B-Instruct Generation

**What:** Model generates `![](!answer!tokens)` markdown image syntax and stray `!` tokens between content words.

**Examples:**
- `"![](!conference!room!B)"` → cleaned to `"conference room B"` ✓
- `"!April !1!5!th"` → cleaned to `"April 5th"` ✗ (loses the `1`)
- `"$1!5!"` → cleaned to `"$15"` ✓

**Why:** Injected KV covers positions 0..N with chat-format tokens. When the 7B-Instruct model generates after `<|im_start|>assistant\n`, attention patterns are slightly off from training distribution — the model defaults to generating `!` (likely token 0 in tiktoken BPE, the lowest-surprise fallback token) between content tokens.

**Workaround in place:** Regex strips `![]()` markdown syntax and bare `!` characters. Breaks on cases like `!1!5` → `15` works, but `!1!5!th` → `5th` loses a digit.

**Fix direction:** Use `model.generate()` with native `past_key_values` support instead of the manual decode loop in `_manual_generate`. The manual loop may have a subtle causal mask issue causing the attention degeneration.

---

## 4. INT8 Quantization Precision Loss on 7B

**What:** 7B model KV tensors have larger dynamic range than 0.5B. Per-tensor INT8 quantization loses more precision, degrading injection quality.

**Not yet tested:** Whether switching to per-channel or FP16 storage for the 7B fixes the `!` artifacts (the oracle test uses direct FP16 and still gets artifacts, so this may not be the root cause).

**Fix direction:** Per-channel INT8 (quantize each KV head separately) or just store 7B KV in FP16 with ~2× storage cost.

---

## 5. Retrieval Infrastructure Mismatch

**What:** `stage1_coarse` retrieves candidates via vector ANN, then text reranking re-orders them. But if the ANN result set doesn't include the correct block at all (due to collapse), text reranking can't recover it.

**Current state:** Because all cosine similarities are within 0.0002, Qdrant's ANN essentially returns all stored blocks as candidates (coarse_top_k=200 with only 8 unique contexts → all 8 returned). So retrieval recall is 100% at stage 1, and text reranking handles stage 2. This only works at small scale.

**At real scale (thousands of blocks):** ANN will only return 200 candidates. If the correct block isn't in those 200 (which it won't be reliably given collapsed similarities), text reranking can't help.

---

## Current Numbers (7B-Instruct, n=20 synthetic)

| Metric | KV Memory | RAG | Delta |
|--------|-----------|-----|-------|
| Overall F1 | 60.5% | 25.8% | +34.7 |
| Exact Match | 60.0% | 40.0% | +20.0 |
| Prefill tokens | 19 | 71 | −73.7% |
| Latency | 408ms | 1611ms | −4× |

Token savings and latency are working. Accuracy needs retrieval fixed.
