# Routerless MSA Future Plan

This document captures the current findings and forward plan for the KV-memory
project after the local BEAM experiments and the routerless-MSA design
discussion. The goal is to preserve the insights before implementation starts.

## 1. Current Goal

The working hypothesis is not "copy MSA exactly." The working hypothesis is:

> Can we get MSA-like memory without training new Router-Q and Router-K
> projections, by using the base model's native attention Q/K projections as the
> routing mechanism?

In MSA, routing is trained:

```text
QR = H_query  * W_QR
KR = H_memory * W_KR
score = QR dot KR
```

Our routerless hypothesis is:

```text
Q = H_query  * W_Q
K = H_memory * W_K
score = Q dot K
```

The full K/V cache remains the payload injected into the model after retrieval.
The key change is that routing should be attention-style Q-to-K matching, not
K-to-K cosine matching.

## 2. What The Local BEAM Run Proved

Command used:

```powershell
python -m tests.beam_eval --hf --scale 100K --n 5 --model .\models\Qwen2.5-7B-Instruct --dtype float16 --retrieval-layers 8 16 24 --capture-batch-size 1 --max-new-tokens 50 --output results_beam_local_100K_n5.json
```

Run timing:

```text
Started:  2026-04-13 16:52:50
Finished: 2026-04-13 16:56:22
Wall time: about 3 min 31 sec
```

Pipeline facts:

```text
BEAM loaded from local HF cache.
Local Qwen2.5-7B-Instruct loaded.
Qdrant at localhost:6333 worked.
KV write pipeline stored 1 unique BEAM context.
Chunks created: 941
Chunks written: 930
Hash duplicates skipped: 11
Capture failures: 0
KV Memory eval completed.
Lexical RAG baseline completed.
Sliding-window baseline completed.
```

Metrics:

```text
KV Memory:
  Correct: 0/5
  Overall F1: 15.3%
  Avg prefill tokens: 44
  Avg latency: 2643 ms

Lexical RAG:
  Correct: 1/5
  Overall F1: 35.7%
  Avg prefill tokens: 3894
  Avg latency: 4790 ms

Sliding Window:
  Correct: 0/5
  Overall F1: 18.4%
  Avg prefill tokens: 1988
  Avg latency: 3322 ms
```

Good result:

```text
KV achieved about 98.9% prefill reduction vs RAG.
KV was faster than RAG.
KV storage, Qdrant indexing, fetch, dequantization, injection, and generation
ran end-to-end.
```

Bad result:

```text
KV retrieved mostly wrong chunks and answered generically.
For "When does my first sprint end?", it retrieved chunks like "Create a new
session if the passwords match" instead of the March 29 sprint evidence.
```

Conclusion:

```text
The current failure is primarily retrieval quality, not infrastructure.
Injection may still need validation, but the observed failure happened before
generation because the selected blocks were mostly wrong.
```

## 3. Current Retrieval Approach

The current repo stores two kinds of data for each memory chunk:

```text
1. Full K/V tensors in kv_store for injection.
2. Lightweight retrieval vectors in Qdrant for search/ranking.
```

The current routing/retrieval key is approximately:

```text
store: query-independent memory chunk -> mean-pooled K vector
read:  user question -> mean-pooled K vector
score: cosine(mean(K_question), mean(K_chunk))
```

Stage 1:

```text
Qdrant ANN search over named vectors layer_8, layer_16, layer_24.
ANN is only the search method.
Cosine similarity is still the scoring metric.
```

Stage 2:

```text
MMR reranking over Stage 1 candidate IDs.
MMR score = 0.7 * relevance - 0.3 * redundancy.
final_top_k defaults to 10.
token_budget defaults to 2000.
```

Why this is weak:

```text
Attention lookup is Q_query dot K_memory, not K_query dot K_memory.
Mean pooling loses token-level evidence.
Short junk chunks can get high cosine scores.
MMR cannot fix bad relevance scores; it can only produce "diverse wrong chunks."
final_top_k=10 can fill with tiny fragments while using very little budget.
Routing vectors may be RoPE/position-entangled.
No neighbor expansion is currently used.
No temporal or conflict-aware metadata is used.
```

## 4. MSA Paper vs This Repo

The MSA paper in `docs/research1.md` describes a trained architecture:

```text
Router-K projector creates routing keys KR.
Router-Q projector creates routing queries QR.
Auxiliary loss supervises routing decisions.
Documents/chunks have compressed K/V/KR representations.
Document-wise RoPE is part of the architecture.
Memory Interleave handles multi-hop retrieval iteratively.
```

This repo is not that.

This repo is currently:

```text
External KV cache store.
Qdrant retrieval over untrained pooled vectors.
Manual HuggingFace KV injection.
No trained router projections.
No auxiliary routing loss.
No Memory Interleave.
No structured temporal/conflict memory layer.
```

That is fine because the research question is different:

```text
Can native attention Q/K be enough to act as the router, avoiding a trained MSA
router module?
```

## 5. Ramp Labs Note vs This Repo

The Ramp Labs note in `docs/ramp_labs_research` is also not exactly this repo.
It proposes task-guided KV cache compaction:

```text
Use task prompt queries Q.
Compare those queries against trajectory K.
Use attention scores to identify relevant positions.
Aggregate across layers/heads.
Use MAD thresholding instead of fixed top-k.
Compact KV cache so worker model sees relevant latent context.
```

Relevant ideas to borrow:

```text
Use Q-to-K attention scores, not K-to-K cosine.
Aggregate across layers and heads.
Use thresholding/budgeting instead of always taking fixed top-k.
Avoid over-compaction and under-compaction; the best threshold can vary by task.
```

## 6. Oracle Injection

Oracle injection is a diagnostic, not a benchmark.

Normal path:

```text
question -> Stage 1 ANN -> Stage 2 MMR -> selected blocks -> inject KV -> answer
```

Oracle path:

```text
question + gold answer -> lexically select likely answer chunks -> inject KV -> answer
```

Purpose:

```text
If oracle injection succeeds, retrieval is the bottleneck.
If oracle injection fails, injection/RoPE/manual generation is the bottleneck.
```

The existing script is:

```powershell
python scripts\oracle_inject_beam.py --run-id latest --model .\models\Qwen2.5-7B-Instruct --dtype float16 --n 5 --max-blocks 8 --output results_oracle_local_100K_n5.json
```

## 7. RoPE Problem

There are two separate RoPE concerns:

```text
1. Routing RoPE: how we compare query and memory vectors.
2. Injection RoPE: how injected cache positions are presented to the model.
```

For routing:

```text
Best: compare pre-RoPE Q to pre-RoPE K.
Second best: de-rotate post-RoPE Q/K into a neutral position frame.
Risky: compare post-RoPE vectors directly, because position can pollute
content similarity.
```

For injection, current code uses one offset for both:

```text
causal cache/mask length
query RoPE position offset
```

Current approximation:

```text
Each memory chunk is captured from local positions 0..chunk_len.
Selected chunks are concatenated into a single past cache.
Query position_ids start at total_past_tokens.
DynamicCache.get_seq_length is overridden/synchronized so causal masking works.
```

This is a practical HuggingFace cache fix, but not a full MSA document-wise
RoPE implementation.

Longer-term injection fix:

```text
Split causal_past_len from query_rope_offset.
causal_past_len controls attention mask/cache length.
query_rope_offset controls RoPE position_ids for active query tokens.
```

The first Q-K reranking experiment can proceed before this full RoPE rewrite,
but final routerless-MSA quality depends on making routing RoPE-neutral.

## 8. New Routerless Q-K Retrieval Approach

New retrieval should use:

```text
score(chunk) = aggregate(Q_query dot K_chunk / sqrt(head_dim))
```

Do not compare values for retrieval:

```text
K = lookup address.
V = content payload.
Retrieval should use Q-to-K.
Generation uses selected K and V.
```

Candidate scoring shape:

```text
for each retrieval layer:
  for each KV head:
    for each query token:
      score query Q against memory K tokens or pooled K segments
aggregate across memory segments, query tokens, heads, and layers
```

A good initial aggregation rule:

```text
chunk_score =
  weighted_mean_over_layers(
    max_over_query_tokens(
      mean_over_heads(
        max_over_memory_segments(Q dot K / sqrt(d))
      )
    )
  )
```

Why max operations:

```text
A chunk can be relevant because of one phrase such as March 29 or 165 commits.
Mean pooling over the whole chunk can dilute that evidence.
```

## 9. Chunk Selection Under Token Budget

The selection policy should not be "always take top 10."

Initial simple policy:

```text
1. Score Stage 1 candidates by Q-K score.
2. Drop chunks below min_tokens, e.g. 20 or 32 tokens.
3. Sort by Q-K score descending.
4. Select chunks until token_budget is reached.
5. Stop at max_chunks only as a safety cap, e.g. 16.
```

Pseudo-code:

```python
selected = []
tokens_used = 0

for chunk in sorted(candidates, key=lambda c: c.qk_score, reverse=True):
    if chunk.token_count < min_tokens:
        continue
    if chunk.qk_score < min_score:
        continue
    if tokens_used + chunk.token_count > token_budget:
        continue
    selected.append(chunk)
    tokens_used += chunk.token_count
    if len(selected) >= max_chunks:
        break
```

Preferred later policy:

```text
Use MAD or percentile thresholding over Q-K scores.
Keep chunks statistically above the pack.
Then budget-pack those chunks.
Use a small redundancy penalty only after relevance is working.
```

Possible final score:

```text
final_score = qk_relevance - 0.15 * redundancy
```

MMR should become a secondary anti-duplicate penalty, not the primary relevance
source.

## 10. Dynamic Memory Model

Memory is not a fixed corpus. It changes over time.

Therefore, the KV memory layer should be append-only:

```text
new conversation/event
  -> chunk it
  -> capture K/V cache
  -> store KV blob
  -> store routing metadata
  -> store structured claims if available
```

Do not overwrite old KV blocks when new information conflicts. Keep both.

Example:

```text
event 1: "first sprint ends March 29"
event 2: "first sprint now targets March 31"
```

Both facts are useful:

```text
"What is the current deadline?" -> March 31
"What was the original deadline?" -> March 29
```

## 11. Conflict Resolution

Conflict detection cannot scan all memory on every query. It must use an index.

Write-time claim extraction:

```text
new chunk -> extract claims -> canonicalize entity/relation -> look up same key
```

Fact key:

```text
entity_id + attribute/relation
```

Example:

```text
entity: sprint_1
attribute: deadline
value: March 31
```

Conflict rule:

```text
same entity + same attribute + different value = possible conflict/update
```

The lookup is only over the relevant key, not the entire memory bank:

```text
facts_by_key["sprint_1.deadline"] -> existing facts
```

If entity resolution is uncertain, retrieve a small candidate set from the
entity/fact index and compare only those candidates.

## 12. Entity And Attribute Extraction

Entity extraction can be staged:

```text
Level 1: heuristics/regex for obvious patterns.
Level 2: local LLM structured JSON extraction.
Level 3: entity resolution against existing aliases and nearby session context.
```

Example extraction:

```json
{
  "claims": [
    {
      "subject": "first sprint",
      "relation": "deadline",
      "object": "2024-03-31",
      "modality": "asserted",
      "change_type": "update",
      "confidence": 0.91,
      "evidence_span": "first sprint now targets completion by March 31"
    }
  ]
}
```

Canonicalization examples:

```text
"first sprint" -> sprint_1
"sprint 1" -> sprint_1
"initial two-week sprint" -> sprint_1

"ends on" -> deadline
"targets completion by" -> deadline
"current subscription price" -> subscription_price
"response time" -> dashboard_api_response_time
```

Change-type cues:

```text
Correction:
  "actually", "correction", "I meant", "no, it is", "rather", "instead"

Update:
  "changed", "moved", "now", "latest", "updated", "new deadline"

Historical addition:
  "originally", "previously", "used to", "before", "at first"

Uncertain:
  "maybe", "I think", "probably", "not sure", "might"
```

Resolution policy for current/latest questions:

```text
explicit correction > newer update > higher-confidence source > older memory
```

But this policy is query-intent dependent. Historical/original questions should
prefer older or superseded facts instead.

## 13. Temporal Memory

Temporal memory needs multiple time fields:

```text
observed_at = when the system heard/stored the text
event_time  = when the described event happened
value_time  = if the fact value is itself a date
```

Example:

```text
"On March 20, I moved the deadline to March 31."

observed_at: when stored by the system
event_time:  March 20
value_time:  March 31
value:       March 31
```

BEAM contexts often contain time anchors:

```text
Time Anchor: March 15, 2024
"two weeks from now" -> March 29, 2024
```

Date extraction should use:

```text
1. explicit date parser for March 29, 2024-03-29, Q1 2024
2. relative date resolver using time anchors
3. LLM fallback for messy temporal statements
4. uncertainty field for partial/unresolved dates
```

Query intent categories:

```text
current/latest:
  "current", "now", "latest", "currently"
  prefer current facts and recency

as-of:
  "as of March 20", "at that time"
  filter facts valid at that date

historical/original:
  "original", "first plan", "before it changed"
  prefer older/superseded facts

timeline:
  "how did it change"
  retrieve all related facts in time order

aggregate:
  "how many columns did I add across requests"
  retrieve all matching events, not just latest
```

## 14. Proposed Data Structures For Later

Raw event/KV block:

```text
block_id
session_id
doc_id
chunk_index
chunk_text
token_count
observed_at
event_time if extractable
kv_by_layer
routing_k_summary
```

Structured fact record:

```text
fact_id
subject_id
subject_text
relation
value
value_type
confidence
change_type
observed_at
event_time
valid_from
valid_to
status: current | superseded | historical | uncertain
supersedes: [fact_id]
source_block_ids: [block_id]
```

## 15. Implementation Roadmap

Phase 0: preserve this plan.

```text
Create this document on a new branch and commit it.
```

Phase 1: diagnostics.

```text
Add logs to distinguish:
  Stage 1 missed the gold/evidence chunk.
  Stage 1 found it but Stage 2 dropped it.
  Stage 2 selected it but generation ignored it.
```

Log per question:

```text
candidate_count
selected IDs
selected chunk token counts
selected chunk text previews
scores
likely gold/evidence term presence
budget usage
```

Phase 2: Q-K reranking experiment.

```text
Add adapter method to compute query-side native Q for selected layers.
Add Q-K reranker behind a --rerank qk flag.
Keep current MMR as --rerank mmr default for comparison.
```

Phase 3: chunk selection quality.

```text
Add min token guard.
Switch from fixed final_top_k-only behavior to budget-aware selection.
Add optional neighbor expansion by chunk_index/doc_id.
Sort final injected blocks in chronological/source order when appropriate.
```

Phase 4: RoPE-neutral routing.

```text
Store pre-RoPE or de-rotated K routing summaries.
Compute pre-RoPE or de-rotated Q query summaries.
Compare in neutral frame.
Keep full K/V payload for injection.
```

Phase 5: dynamic memory semantics.

```text
Add claim extraction and fact index.
Add entity/attribute canonicalization.
Add conflict/update classification.
Add observed_at/event_time/value_time fields.
Add query intent classification for current/historical/as-of/timeline/aggregate.
```

## 16. First Implementation Target After This Document

Do not jump straight to full dynamic memory.

First code target:

```text
Q-K reranking for BEAM n=5.
```

Minimum implementation:

```text
1. Add native query-Q capture.
2. Add stage2_rerank_qk.
3. Add --stage2-reranker mmr|qk to tests/beam_eval.py.
4. Add selected chunk diagnostics.
5. Add min token guard.
```

Acceptance:

```text
Q-K rerank should improve evidence selection over MMR on the same BEAM n=5 run,
especially for March 29, 250ms, 165 commits, category, and notes.
```

If Q-K rerank improves evidence selection:

```text
Proceed to RoPE-neutral routing cache and better chunk selection.
```

If Q-K rerank does not improve evidence selection:

```text
Reassess whether trained router projections or hybrid lexical/embedding
retrieval are required.
```

Implementation checkpoint:

```text
KVMemoryConfig.retrieval_query_source = "k_vectors" | "q_vectors"
KVMemoryConfig.stage2_reranker        = "mmr" | "qk"
KVMemoryConfig.retrieval_rope_mode    = "native" | "neutral"
HFAdapter.capture_query_vecs()
HFAdapter.capture_key_vecs()
stage2_rerank_qk()
BEAM CLI flags:
  --retrieval-query-source q_vectors
  --stage2-reranker qk
  --retrieval-rope-mode neutral
```

Local BEAM command for the routerless Q-K probe:

```powershell
python -m tests.beam_eval --hf --scale 100K --n 5 --model .\models\Qwen2.5-7B-Instruct --dtype float16 --retrieval-layers 8 16 24 --capture-batch-size 1 --max-new-tokens 50 --retrieval-query-source q_vectors --stage2-reranker qk --retrieval-rope-mode neutral --output results_beam_local_100K_n5_qk_neutral.json
```

## 17. Open Risks

```text
Native Q/K may still not be a good zero-shot router without training.
Post-RoPE Q/K comparison may mislead retrieval.
Oracle injection may show injection issues even with correct chunks.
INT8 KV quantization may degrade 7B injection quality.
Manual generation path may differ from model.generate behavior.
Dynamic conflict resolution depends on imperfect claim extraction.
Temporal parsing will need confidence/uncertainty handling.
```

## 18. Key Principle

Do not treat the KV block as the only memory object.

Use:

```text
KV blocks as raw latent evidence.
Routing summaries as lookup keys.
Fact records as dynamic semantic indexes.
Query intent as the resolver that decides current vs historical vs aggregate.
```

This keeps memory append-only while still allowing current facts, old facts,
updates, and timelines to coexist.
