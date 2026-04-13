# KV Memory Benchmark Issues

## Critical

1. **Issue 1: `store_layers` defaults to `retrieval_layers`, so only 3/28 layers get KV tensors stored for injection**

   Locations: `kvmemory/config.py:21`, `tests/beam_eval.py:1256-1266`

   `KVMemoryConfig.__post_init__` defaults `store_layers` to `retrieval_layers`. The BEAM eval does not set `store_layers` explicitly. For Qwen2.5-7B with `--retrieval-layers 7 14 21`, only those three layers are captured and stored. During injection, `_build_cache` iterates all 28 layers; the missing 25 layers receive zero-length placeholder tensors. The model runs without crashing, but most attention layers see no injected context.

2. **Issue 21: `hidden_by_layer` is not filtered before Qdrant upsert, so `store_layers != retrieval_layers` breaks the write path**

   Locations: `kvmemory/core/write_pipeline.py:275`, `kvmemory/core/write_pipeline.py:342-344`

   `capture_batch(config.store_layers)` returns `hidden_by_layer` keyed by every captured store layer. The write pipeline computes retrieval vectors for every key in `hidden_by_layer` and sends all of them to Qdrant. If `store_layers` is all model layers but the Qdrant collection schema only contains named vectors for retrieval layers, upsert attempts to write vectors for non-schema layer names.

3. **Issue 16: Correct evidence is stored in Qdrant but not being retrieved; diagnostics do not explain why**

   Locations: `tests/beam_eval.py:529-556`

   The smoke run stored evidence chunks containing the correct March 29 answer, but Stage 2 selected unrelated micro-chunks. Current observability records candidate count and selected count, but not the rank or cosine score of gold/evidence blocks compared with top-ranked junk blocks. This hides whether the failure is candidate absence, layer/schema mismatch, vector scoring quality, or reranking.

## High Priority

4. **Issue 2: Chunker allows micro-fragments that produce noisy retrieval vectors**

   Locations: `kvmemory/core/chunker.py:71-77`, `kvmemory/core/write_pipeline.py:159`

   `chunk_turn` uses `target_tokens=100` and `min_tokens=20`, but mid-document short sentences can still become standalone chunks. The smoke run selected fragments as small as 5-15 tokens. These tiny chunks produce unstable retrieval vectors and can outscore semantically useful chunks.

5. **Issue 3: Stage 2 MMR has no minimum-token guard**

   Location: `kvmemory/core/retrieval.py:211`

   Stage 2 applies an upper token budget but no lower bound. A 5-token block competes on equal footing with a 150-token block, allowing micro-fragments to fill the selected set.

6. **Issue 4: No Qdrant layer-schema consistency check between runs**

   Locations: `tests/beam_eval.py:1252-1266`, `kvmemory/storage/vector_db.py:63-76`

   `ensure_collection` returns early when a collection already exists and does not verify that the existing named-vector schema matches the current `retrieval_layers`. A collection created with one layer set can be reused with another layer set, causing query or upsert behavior to degrade or fail depending on Qdrant/client behavior.

7. **Issue 17a: `final_top_k=10` hard-stops MMR regardless of budget utilization**

   Locations: `kvmemory/core/retrieval.py:185`, `kvmemory/config.py:26`

   The MMR loop stops when `len(selected) < config.final_top_k` is no longer true. In the smoke run, 10 small blocks filled the selected slots while using only 146-266 tokens out of a 2000-token budget.

8. **Issue 17b: Stage 2 has no budget utilization floor**

   Location: `kvmemory/core/retrieval.py:185`

   Stage 2 has a hard maximum token budget but no minimum useful packing target. It can exit with very low token utilization as soon as the selected block count cap is reached.

9. **Issue 18: Short-query vs long-chunk mean-pool asymmetry degrades cosine scores**

   Locations: `kvmemory/core/retrieval.py:59-68`, `kvmemory/core/retrieval.py:82-90`

   Query vectors are mean-pooled from short bare questions, often only 6-8 tokens. Stored chunk vectors are mean-pooled from much longer conversational chunks. This asymmetry can dilute relevant evidence in chunk vectors and compress cosine scores, allowing short junk chunks to outrank semantically correct longer chunks.

10. **Issue 7: `prefill_reduction_pct` is computed for display only and serializes as `0.0`**

    Locations: `tests/beam_eval.py:1115-1119`, `tests/beam_eval.py:1299-1316`

    `prefill_reduction_pct` is computed inside `print_comparison` for console output, but the metric object is serialized before that value is stored back into the JSON summary/output path.

## Medium Priority

11. **Issue 6: Blocks are injected in MMR relevance order, not chronological order**

    Locations: `kvmemory/core/retrieval.py:223`, `kvmemory/adapters/hf_adapter.py:350-377`

    `stage2_rerank_mmr` returns block IDs in selection order. `_build_cache` concatenates blocks in the fetched/selected order. For temporal or multi-session reasoning, this can present context fragments out of their original chronology.

12. **Issue 12: KV prompt has no conciseness instruction while RAG and sliding window do**

    Locations: `tests/beam_eval.py:514-518`, `tests/beam_eval.py:761`

    RAG and sliding-window prompts include `Answer concisely.`. The KV path uses only the raw question inside the chat template. This makes the comparison uneven and contributes to verbose answers that hit the `max_new_tokens` cap.

13. **Issue 8: `em_score` uses substring containment, not strict exact match**

    Location: `tests/beam_eval.py:315`

    The scoring code marks exact match as true when the normalized gold answer is contained anywhere inside the normalized prediction. This is a containment score, not strict exact match.

14. **Issue 19: No chunk adjacency expansion**

    Location: `kvmemory/core/retrieval.py:180-239`

    MMR retrieves scattered best-matching chunks. When a selected chunk belongs to a larger multi-chunk plan, nearby chunks containing required dates or task details are not fetched automatically.

15. **Issue 20: `reset_baseline` before every store call resets EMA to cold-start**

    Locations: `tests/beam_eval.py:444-446`, `kvmemory/core/importance.py:98`

    `BaselineLossTracker` starts at `2.0` with `alpha=0.1`. BEAM resets the tracker before every unique context store, so each document begins with a cold baseline. This is masked in BEAM by `explicit_signal=1.0`, but it affects importance scoring behavior when explicit signal is not used.

## Low Priority

16. **Issue 9: `overall_accuracy` is a serialized alias for `overall_em` with a misleading name**

    Locations: `tests/beam_eval.py:99`, `tests/beam_eval.py:1057`

    `overall_accuracy` and `overall_em` serialize to the same value. The name `accuracy` is ambiguous because `overall_correct_rate` is the metric that reflects `EM or F1 > 0.5`.

17. **Issue 10: `accuracy_by_type` and `em_by_type` are the same dict object**

    Location: `tests/beam_eval.py:1034`

    `em_by_type = accuracy_by_type` assigns the same dictionary object to both fields. Both serialize to identical values.

18. **Issue 5: `prefill_tokens` uses chat-template token length, not raw question length**

    Location: `tests/beam_eval.py:622`

    `prefill_tokens=len(generation_tokens)` counts the actual chat-formatted generation prompt. For Qwen2.5, this includes system/role/template tokens and is longer than the raw retrieval question tokens.

19. **Issue 14: `_manual_generate` silently ignores generation kwargs beyond the supported subset**

    Location: `kvmemory/adapters/hf_adapter.py:417-424`

    The KV injection path reads only `max_new_tokens`, `do_sample`, `temperature`, `top_p`, and `eos_token_id`. Other generation kwargs are ignored in the manual path but handled by `model.generate()` in the no-blocks path.

20. **Issue 15: Access count is tracked redundantly in both KVStore and VectorDB**

    Locations: `kvmemory/storage/kv_store.py:132-150`, `kvmemory/storage/vector_db.py:354-374`

    Fetching blocks can trigger metadata updates in both local blob storage and Qdrant payloads. This duplicates background I/O and keeps access metadata in two places.
