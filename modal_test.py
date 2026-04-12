"""
modal_test.py — End-to-end KV Memory RAG test on Modal T4 GPU.

Runs a staged diagnostic to isolate exactly where basic RAG breaks.
Uses Qwen2.5-0.5B (fits on T4, has GQA like production models).
Uses in-memory Qdrant — no external server needed.

Usage:
    modal run modal_test.py

Each stage prints PASS/FAIL with full debug info so you can see exactly
which step (write, retrieval, injection) is the failure point.
"""

import modal

# ---------------------------------------------------------------------------
# Image: install all deps + the local kvmemory package
# ---------------------------------------------------------------------------
def _exclude(p) -> bool:
    ps = str(p)
    # Use os.sep-aware patterns to avoid matching "kv_store.py" when excluding "kv_store/" dir
    return (
        ".git" in ps
        or "__pycache__" in ps
        or ".pkl" in ps
        or ".egg-info" in ps
        or ps.endswith("/kv_store")
        or "/kv_store/" in ps
        or ps == "kv_store"
    )

image = (
    # pytorch/pytorch already has CUDA + torch baked in — skips 780MB download
    modal.Image.from_registry(
        "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime",
        add_python="3.11",
    )
    .pip_install(
        "transformers==4.45.2",
        "numpy>=1.24",
        "qdrant-client>=1.8",
        "accelerate>=0.26",
        "sentencepiece",
        "huggingface_hub",
    )
    # copy=True required when run_commands follows add_local_dir (Modal 1.x)
    .add_local_dir(".", remote_path="/root/kv-memory", ignore=_exclude, copy=True)
    .run_commands(
        "pip install -q --upgrade setuptools pip",
        "pip install -e /root/kv-memory --no-deps -q",
    )
)

app = modal.App("kv-memory-rag-test", image=image)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
# Qwen2.5-0.5B: 24 layers (indices 0–23).
# Default [8, 16, 24] puts layer 24 out of range → silently drops vectors.
# Use [6, 12, 18] instead: 25%, 50%, 75% of depth.
RETRIEVAL_LAYERS = [6, 12, 18]

# ---------------------------------------------------------------------------
# Helper: colour output for terminal readability
# ---------------------------------------------------------------------------

def ok(msg: str) -> str:
    return f"  \033[32m[PASS]\033[0m {msg}"

def fail(msg: str) -> str:
    return f"  \033[31m[FAIL]\033[0m {msg}"

def info(msg: str) -> str:
    return f"  \033[34m[INFO]\033[0m {msg}"

def header(msg: str) -> None:
    print(f"\n\033[1m{'─'*60}\033[0m")
    print(f"\033[1m{msg}\033[0m")
    print(f"\033[1m{'─'*60}\033[0m")

# ---------------------------------------------------------------------------
# The actual test — runs inside the Modal container on a T4
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    timeout=600,
)
def run_rag_test():
    import sys, os, asyncio, logging, tempfile, traceback
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    results = {}

    # -----------------------------------------------------------------------
    # STAGE 0: GPU / environment
    # -----------------------------------------------------------------------
    header("STAGE 0: Environment")
    print(info(f"Python {sys.version}"))
    print(info(f"PyTorch {torch.__version__}"))
    print(info(f"CUDA available: {torch.cuda.is_available()}"))
    if torch.cuda.is_available():
        dev = torch.cuda.get_device_properties(0)
        print(info(f"GPU: {dev.name}, {dev.total_memory // 1024**3} GB"))
    import transformers
    print(info(f"transformers {transformers.__version__}"))

    from kvmemory.config import KVMemoryConfig
    from kvmemory.adapters.hf_adapter import HFAdapter
    from kvmemory.memory import KVMemory
    print(ok("kvmemory package importable"))
    results["env"] = "PASS"

    # -----------------------------------------------------------------------
    # STAGE 1: Load model
    # -----------------------------------------------------------------------
    header(f"STAGE 1: Load {MODEL_ID}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
        )
        model.eval()

        n_layers = model.config.num_hidden_layers
        d_model  = model.config.hidden_size
        n_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
        print(ok(f"Model loaded: layers={n_layers}, d_model={d_model}, kv_heads={n_kv_heads}"))

        # Validate retrieval layers
        bad = [l for l in RETRIEVAL_LAYERS if l >= n_layers]
        if bad:
            print(fail(f"retrieval_layers {bad} are >= num_hidden_layers ({n_layers})! "
                       f"These will silently produce no retrieval vectors."))
        else:
            print(ok(f"retrieval_layers={RETRIEVAL_LAYERS} all valid (model has {n_layers} layers)"))

        results["model_load"] = "PASS"
    except Exception as e:
        print(fail(f"Model load failed: {e}"))
        traceback.print_exc()
        results["model_load"] = f"FAIL: {e}"
        return results

    adapter = HFAdapter(model, tokenizer)

    # -----------------------------------------------------------------------
    # STAGE 2: Single capture
    # -----------------------------------------------------------------------
    header("STAGE 2: adapter.capture() — extract KV tensors + hidden states")
    CONTEXT_TEXT = (
        "The Eiffel Tower was built between 1887 and 1889 as the entrance arch "
        "to the 1889 World's Fair in Paris. It was designed by Gustave Eiffel. "
        "The tower stands 330 meters tall and was the tallest man-made structure "
        "in the world for 41 years until the Chrysler Building was erected in 1930."
    )
    try:
        ctx_tokens = tokenizer.encode(CONTEXT_TEXT)
        print(info(f"Context: {len(ctx_tokens)} tokens"))

        kv_by_layer, hidden_by_layer = adapter.capture(
            tokens=ctx_tokens,
            text=CONTEXT_TEXT,
            layers=RETRIEVAL_LAYERS,
        )
        print(info(f"KV layers captured: {sorted(kv_by_layer.keys())}"))
        print(info(f"Hidden layers captured: {sorted(hidden_by_layer.keys())}"))

        missing_kv = [l for l in RETRIEVAL_LAYERS if l not in kv_by_layer]
        missing_hs = [l for l in RETRIEVAL_LAYERS if l not in hidden_by_layer]
        if missing_kv:
            print(fail(f"KV missing for layers {missing_kv}"))
        if missing_hs:
            print(fail(f"Hidden states missing for layers {missing_hs}"))

        for layer_idx in kv_by_layer:
            K, V = kv_by_layer[layer_idx]
            h = hidden_by_layer.get(layer_idx)
            print(info(f"  Layer {layer_idx}: K={tuple(K.shape)}, V={tuple(V.shape)}, "
                       f"hidden={tuple(h.shape) if h is not None else 'MISSING'}"))

        if not missing_kv and not missing_hs:
            print(ok("capture() returned all expected layers"))
            results["capture"] = "PASS"
        else:
            results["capture"] = f"FAIL: missing KV={missing_kv}, HS={missing_hs}"
    except Exception as e:
        print(fail(f"capture() raised: {e}"))
        traceback.print_exc()
        results["capture"] = f"FAIL: {e}"

    # -----------------------------------------------------------------------
    # STAGE 3: Write pipeline (store into in-memory Qdrant)
    # -----------------------------------------------------------------------
    header("STAGE 3: write pipeline — store context into KV memory")
    tmp_blob = tempfile.mkdtemp()
    config = KVMemoryConfig(
        model_id="qwen25-05b-test",
        retrieval_layers=RETRIEVAL_LAYERS,
        qdrant_url=":memory:",   # in-memory Qdrant, no server needed
        blob_store_path=tmp_blob,
        importance_threshold=0.0,  # disable importance filter so nothing gets dropped
        dedup_threshold=1.01,       # disable dedup (threshold > 1.0 → never a duplicate)
    )
    print(info(f"Config: importance_threshold={config.importance_threshold} (disabled), "
               f"dedup_threshold={config.dedup_threshold} (disabled)"))

    try:
        memory = KVMemory(adapter=adapter, config=config)
        print(ok("KVMemory initialized, Qdrant collection created"))
    except Exception as e:
        print(fail(f"KVMemory init failed: {e}"))
        traceback.print_exc()
        results["memory_init"] = f"FAIL: {e}"
        return results

    results["memory_init"] = "PASS"

    # Store a few facts
    FACTS = [
        "The Eiffel Tower was built between 1887 and 1889 in Paris, France. "
        "It was designed by Gustave Eiffel and stands 330 meters tall.",

        "The Chrysler Building in New York City was completed in 1930. "
        "It surpassed the Eiffel Tower as the world's tallest structure when it was built.",

        "The Great Wall of China stretches over 21,196 kilometers. "
        "It was built over many centuries to protect Chinese states from northern invasions.",
    ]

    try:
        async def _store_facts():
            for i, fact in enumerate(FACTS):
                print(info(f"  Storing fact {i+1}: '{fact[:60]}...'"))
                await memory.store(fact, session_id="test-session", explicit_signal=1.0)
            await memory.drain_writes(timeout=60)

        asyncio.run(_store_facts())

        # Check blob store
        pkl_files = [f for f in os.listdir(os.path.join(tmp_blob, "qwen25-05b-test"))
                     if f.endswith(".pkl")]
        n_blocks = len(pkl_files)
        print(info(f"Blob store: {n_blocks} .pkl files written"))

        if n_blocks == 0:
            print(fail("No blocks written to disk! Write pipeline silently dropped everything."))
            print(info("Check: importance_threshold too high? chunks too short? capture failed?"))
            results["write"] = "FAIL: 0 blocks written"
        else:
            print(ok(f"{n_blocks} blocks written to disk"))
            results["write"] = f"PASS: {n_blocks} blocks"

    except Exception as e:
        print(fail(f"store() / drain_writes() raised: {e}"))
        traceback.print_exc()
        results["write"] = f"FAIL: {e}"

    # -----------------------------------------------------------------------
    # STAGE 4: Retrieval — does stage1 find candidates?
    # -----------------------------------------------------------------------
    header("STAGE 4: retrieval — stage1 coarse ANN search")
    try:
        from kvmemory.core.retrieval import stage1_coarse

        async def _stage1():
            query = "Who designed the Eiffel Tower and when was it built?"
            q_tokens = tokenizer.encode(query)
            print(info(f"Query: '{query}'"))
            print(info(f"Query tokens: {len(q_tokens)}"))

            session_filter = {
                "model_id": config.model_id,
                "session_id": "test-session",
                "agent_id": None,
                "retrieve_shared": False,
            }
            candidate_ids, query_vecs = await stage1_coarse(
                query_tokens=q_tokens,
                adapter=adapter,
                config=config,
                vector_db=memory.vector_db,
                session_filter=session_filter,
            )
            print(info(f"Stage 1 candidates: {len(candidate_ids)}"))
            print(info(f"Candidate IDs: {candidate_ids[:5]}{'...' if len(candidate_ids) > 5 else ''}"))
            print(info(f"Query vecs layers present: {sorted(query_vecs.keys())}"))
            return candidate_ids, query_vecs

        candidate_ids, query_vecs = asyncio.run(_stage1())

        if not candidate_ids:
            print(fail("Stage 1 returned 0 candidates — retrieval is broken!"))
            print(info("Possible causes:"))
            print(info("  1. No blocks written (check stage 3)"))
            print(info("  2. session_id filter mismatch"))
            print(info("  3. retrieval_layers mismatch between write and query"))
            print(info("  4. Qdrant collection schema wrong"))
            results["stage1"] = "FAIL: 0 candidates"
        else:
            print(ok(f"Stage 1 found {len(candidate_ids)} candidates"))
            results["stage1"] = f"PASS: {len(candidate_ids)} candidates"

    except Exception as e:
        print(fail(f"stage1_coarse raised: {e}"))
        traceback.print_exc()
        results["stage1"] = f"FAIL: {e}"

    # -----------------------------------------------------------------------
    # STAGE 5: MMR rerank
    # -----------------------------------------------------------------------
    header("STAGE 5: retrieval — stage2 MMR rerank")
    try:
        from kvmemory.core.retrieval import stage2_rerank_mmr

        if "candidate_ids" not in dir() or not candidate_ids:
            print(fail("Skipped — no candidates from stage 1"))
            results["stage2"] = "SKIP"
        else:
            final_ids = stage2_rerank_mmr(
                candidate_ids=candidate_ids,
                query_vecs=query_vecs,
                config=config,
                vector_db=memory.vector_db,
                token_budget=config.token_budget,
            )
            print(info(f"Stage 2 selected: {len(final_ids)} blocks"))
            print(info(f"Selected IDs: {final_ids}"))

            if not final_ids:
                print(fail("Stage 2 returned 0 blocks!"))
                results["stage2"] = "FAIL: 0 selected"
            else:
                print(ok(f"Stage 2 selected {len(final_ids)} blocks"))
                results["stage2"] = f"PASS: {len(final_ids)} selected"

    except Exception as e:
        print(fail(f"stage2_rerank_mmr raised: {e}"))
        traceback.print_exc()
        results["stage2"] = f"FAIL: {e}"

    # -----------------------------------------------------------------------
    # STAGE 6: KV fetch + inject + generate
    # -----------------------------------------------------------------------
    header("STAGE 6: KV fetch → inject → generate")
    try:
        if "final_ids" not in dir() or not final_ids:
            print(fail("Skipped — no blocks from stage 2"))
            results["generate"] = "SKIP"
        else:
            blocks = memory.kv_store.fetch(final_ids, model_id=config.model_id)
            print(info(f"Fetched {len(blocks)} KVBlocks from disk"))
            for b in blocks:
                print(info(f"  Block {b.block_id[:12]}...: {b.token_count} tokens, "
                           f"layers={sorted(b.kv_by_layer.keys())}"))

            query = "Who designed the Eiffel Tower?"
            q_tokens = tokenizer.encode(query)

            output = adapter.inject_and_generate(
                blocks=blocks,
                current_tokens=q_tokens,
                generation_kwargs={"max_new_tokens": 50},
            )
            print(info(f"Generated text: {repr(output.text[:200])}"))

            # Does the answer mention Gustave Eiffel?
            if "eiffel" in output.text.lower() or "gustave" in output.text.lower():
                print(ok("Answer references Eiffel/Gustave — RAG is working!"))
                results["generate"] = "PASS"
            else:
                print(fail("Answer doesn't mention Gustave Eiffel despite injected context."))
                print(info("May indicate injection is happening but model isn't using the context."))
                results["generate"] = "PARTIAL: generated but answer seems context-unaware"

    except Exception as e:
        print(fail(f"inject_and_generate raised: {e}"))
        traceback.print_exc()
        results["generate"] = f"FAIL: {e}"

    # -----------------------------------------------------------------------
    # STAGE 7: Full memory.generate() round-trip
    # -----------------------------------------------------------------------
    header("STAGE 7: full memory.generate() round-trip")
    try:
        async def _full_gen():
            return await memory.generate(
                prompt="Who designed the Eiffel Tower and when was it completed?",
                session_id="test-session",
                generation_kwargs={"max_new_tokens": 60},
            )

        out = asyncio.run(_full_gen())
        print(info(f"Full generate output: {repr(out.text[:300])}"))

        if out.text and len(out.text) > 10:
            print(ok("memory.generate() returned a non-empty response"))
            results["full_generate"] = "PASS"
        else:
            print(fail(f"Empty or very short response: {repr(out.text)}"))
            results["full_generate"] = "FAIL: empty response"

    except Exception as e:
        print(fail(f"memory.generate() raised: {e}"))
        traceback.print_exc()
        results["full_generate"] = f"FAIL: {e}"

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    header("SUMMARY")
    all_pass = True
    for stage, result in results.items():
        passed = result.startswith("PASS")
        icon = "\033[32m✓\033[0m" if passed else "\033[31m✗\033[0m"
        print(f"  {icon}  {stage:<20} {result}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("\033[32m\033[1mAll stages passed — basic RAG is working.\033[0m")
    else:
        print("\033[31m\033[1mSome stages failed — see above for details.\033[0m")

    return results


# ---------------------------------------------------------------------------
# Local entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print("\nRunning KV Memory RAG diagnostic on Modal T4...\n")
    results = run_rag_test.remote()
    print("\n\nFinal results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
