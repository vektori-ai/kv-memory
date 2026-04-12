"""
modal_beam.py — Run BEAM benchmark (synthetic) on Modal GPU.

Runs beam_eval.py in synthetic mode (no dataset download needed):
  - Generates N synthetic QA pairs with multi-hop context
  - Evaluates KV Memory vs RAG baseline
  - Reports accuracy + token savings

RAG baseline: chunks the context, retrieves top chunks by bag-of-words overlap,
prepends them as text to the prompt. At scale (100K+ token conversations) this
retrieves ~256 tokens; on tiny synthetic contexts it retrieves the full context.
KV Memory:   retrieves the KV tensors of the relevant block, injects into
             attention — only the query hits prefill.

Usage:
    modal run modal_beam.py                        # 0.5B on T4
    modal run modal_beam.py --model 7b             # 7B-Instruct on A10G
    modal run modal_beam.py --n 50                 # more questions
    modal run modal_beam.py --diag                 # retrieval + injection diagnostic
    modal run modal_beam.py --model 7b --diag      # diagnostic with 7B
"""

import modal

def _exclude(p) -> bool:
    ps = str(p)
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
        "fastapi>=0.115",
        "uvicorn>=0.30",
    )
    .add_local_dir(".", remote_path="/root/kv-memory", ignore=_exclude, copy=True)
    .run_commands(
        "pip install -q --upgrade setuptools pip",
        "pip install -e /root/kv-memory --no-deps -q",
    )
)

app = modal.App("kv-memory-beam", image=image)

# Model configs — selected by --model flag
MODELS = {
    "0.5b": {
        "id": "Qwen/Qwen2.5-0.5B",
        "gpu": "T4",
        "instruct": False,
    },
    "7b": {
        "id": "Qwen/Qwen2.5-7B-Instruct",
        "gpu": "A10G",
        "instruct": True,
    },
}


def _load_model_and_adapter(model_key: str = "0.5b"):
    """Load model + adapter, shared across diagnostic and BEAM eval."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvmemory.adapters.hf_adapter import HFAdapter

    cfg = MODELS[model_key]
    model_id = cfg["id"]

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {model_id}  (instruct={cfg['instruct']})")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",   # force eager so output_attentions works cleanly
        trust_remote_code=True,
    )
    model.eval()
    return HFAdapter(model, tokenizer), cfg


def _make_memory(adapter, model_key: str = "0.5b", blob_dir=None):
    """Build a fresh in-memory KVMemory instance."""
    import tempfile
    from kvmemory.config import KVMemoryConfig
    from kvmemory.memory import KVMemory

    # 7B has discriminative representations — vector retrieval works fine.
    # 0.5B is near-degenerate — we fall back to text reranking in the eval loop.
    is_7b = model_key == "7b"
    config = KVMemoryConfig(
        model_id=f"beam-qwen-{model_key}",
        qdrant_url=":memory:",
        blob_store_path=blob_dir or tempfile.mkdtemp(),
        importance_threshold=0.0,
        dedup_threshold=1.01,
        final_top_k=3 if is_7b else 1,
        token_budget=1024 if is_7b else 512,
    )
    memory = KVMemory(adapter=adapter, config=config)
    print(f"retrieval_layers: {config.retrieval_layers}  store_layers: {len(config.store_layers)} layers")
    return memory


# -----------------------------------------------------------------------
# Diagnostic: validate retrieval discrimination + injection correctness
# -----------------------------------------------------------------------

@app.function(gpu="A10G", timeout=900)
def run_diagnostic(model_key: str = "0.5b"):
    """
    Two-part diagnostic:

    Part A — Retrieval discrimination:
      Store 5 distinct facts. For each, query with the original question
      and check if the correct block is retrieved (top-1). Print similarity
      scores for all 5 candidates so we can see how discriminative the
      retrieval vectors are.

    Part B — Oracle injection:
      Manually inject the CORRECT block for each question (bypassing retrieval)
      and see if the model can answer correctly from the injected KV context.
      If this passes but Part A fails → pure retrieval problem.
      If this also fails → injection mechanism is broken.
    """
    import sys, asyncio, numpy as np
    sys.path.insert(0, "/root/kv-memory")

    import torch
    adapter, model_cfg = _load_model_and_adapter(model_key)
    memory = _make_memory(adapter, model_key)
    is_instruct = model_cfg["instruct"]

    # ---- Known fact pairs ----
    facts = [
        ("Alice works as a software engineer at TechCorp. She joined in 2019 and leads the backend team.",
         "What is Alice's role at TechCorp?",
         "software engineer"),
        ("The Eiffel Tower is located in Paris, France. It was built in 1889.",
         "Where is the Eiffel Tower?",
         "Paris"),
        ("Python was created by Guido van Rossum and first released in 1991.",
         "Who created Python?",
         "Guido van Rossum"),
        ("Mount Everest is the tallest mountain on Earth at 8,849 metres.",
         "How tall is Mount Everest?",
         "8849"),
        ("Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1911.",
         "How many Nobel Prizes did Marie Curie win?",
         "two"),
    ]

    session_id = "diag-session"

    # ------------------------------------------------------------------
    # Part A: store all facts, then check retrieval for each question
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PART A: Retrieval discrimination")
    print("="*60)

    async def _store_all():
        for context, question, answer in facts:
            memory.reset_baseline(session_id)
            await memory.store(text=context, session_id=session_id, explicit_signal=1.0)
        await memory.drain_writes()

    asyncio.run(_store_all())
    print(f"Stored {len(facts)} fact blocks in Qdrant (in-memory).\n")

    from kvmemory.core.retrieval import compute_query_vecs, stage1_coarse, stage2_rerank_mmr
    from kvmemory.memory import _build_session_filter

    hits = 0
    for i, (context, question, gold) in enumerate(facts):
        q_tokens = adapter.tokenizer.encode(question)

        async def _retrieve(q_tokens=q_tokens):
            return await stage1_coarse(
                query_tokens=q_tokens,
                adapter=memory.adapter,
                config=memory.config,
                vector_db=memory.vector_db,
                session_filter=_build_session_filter(
                    memory.config.model_id, session_id, None, False
                ),
            )

        candidate_ids, query_vecs = asyncio.run(_retrieve())

        # Fetch all candidates with their vectors to show similarity scores
        cands = memory.vector_db.client.scroll(
            collection_name=memory.config.model_id,
            with_vectors=True,
            with_payload=True,
            limit=100,
        )[0]

        # Compute cosine sim between query vec (first retrieval layer) and each block
        rl = memory.config.retrieval_layers[0]
        layer_key = f"layer_{rl}"
        q_vec = query_vecs.get(rl)
        if q_vec is None and query_vecs:
            q_vec = next(iter(query_vecs.values()))

        print(f"Q{i}: '{question}'  (gold: '{gold}')")
        sims = []
        for pt in cands:
            vec = pt.vector.get(layer_key) if isinstance(pt.vector, dict) else None
            if vec is None:
                # named vectors format
                if hasattr(pt, 'vector') and isinstance(pt.vector, dict):
                    vec = list(pt.vector.values())[0]
            if vec is not None and q_vec is not None:
                sim = float(np.dot(q_vec, vec))
                chunk_text = pt.payload.get("chunk_text", "")[:60]
                sims.append((sim, chunk_text, pt.id))

        sims.sort(reverse=True)
        for rank, (sim, chunk_text, bid) in enumerate(sims[:5]):
            marker = "<-- RETRIEVED" if rank == 0 else ""
            expected = "<-- EXPECTED" if context[:40] in chunk_text or chunk_text in context[:60] else ""
            print(f"  rank{rank+1}: sim={sim:.4f}  '{chunk_text}' {marker}{expected}")

        # Check if top-1 retrieved block corresponds to the correct context
        if sims:
            top_chunk = sims[0][1]
            correct_keywords = [w for w in context.split()[:5] if len(w) > 3]
            retrieved_correct = any(kw.lower() in top_chunk.lower() for kw in correct_keywords[:3])
            if retrieved_correct:
                hits += 1
                print(f"  -> HIT")
            else:
                print(f"  -> MISS (expected context about: '{context[:50]}')")
        print()

    print(f"Retrieval accuracy: {hits}/{len(facts)} = {hits/len(facts)*100:.0f}%")

    # ------------------------------------------------------------------
    # Part B: oracle injection — bypass retrieval, inject correct block
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("PART B: Oracle injection (correct block injected directly)")
    print("="*60)

    from kvmemory.core.injector import inject_and_generate
    from kvmemory.storage.schema import KVBlock
    from kvmemory.core.write_pipeline import quantize_int8
    from kvmemory.core.retrieval import compute_retrieval_vec

    from tests.beam_eval import _apply_chat_template

    oracle_hits = 0
    for context, question, gold in facts:
        # For instruct models: store context as the opening of a user message
        # (no close tag) so the captured KV occupies positions 0..N in a
        # structurally valid chat prefix that matches Qwen2.5-Instruct's training format.
        #
        # Full sequence (injected KV + query tokens):
        #   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        #   <|im_start|>user\nContext: {context}\n\nAnswer in one phrase.\n\n{question}<|im_end|>\n
        #   <|im_start|>assistant\n
        #
        # ctx_tokens (captured & injected as KV):
        #   system message + start of user message (no close tag yet)
        # q_tokens (hit prefill):
        #   rest of user message + assistant open tag
        if is_instruct:
            ctx_tokens = adapter.tokenizer.encode(
                f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\nContext: {context}\n\n",
                add_special_tokens=False,
            )
        else:
            ctx_tokens = adapter.tokenizer.encode(context)

        store_layers = memory.config.store_layers
        kv_by_layer, hidden_by_layer = adapter.capture(
            tokens=ctx_tokens, text=context, layers=store_layers
        )

        # Build retrieval vecs for only retrieval_layers
        hidden_vecs = {}
        for l in memory.config.retrieval_layers:
            if l in hidden_by_layer:
                hidden_vecs[l] = compute_retrieval_vec(hidden_by_layer[l], len(ctx_tokens))

        n_ctx = len(ctx_tokens)

        # For the oracle test, bypass INT8 quantization entirely and build the
        # past_key_values cache directly from the raw FP16 tensors.
        # This isolates injection mechanism correctness from quantization precision.
        # (INT8 with per-tensor scale works fine for 0.5B; 7B has higher dynamic range.)
        if is_instruct:
            # ctx_tokens opened a user turn: <|im_start|>user\nContext: {ctx}\n\n
            # q_tokens must CONTINUE that same user turn (no new <|im_start|>user):
            #   Answer in one phrase.\n\n{question}<|im_end|>\n<|im_start|>assistant\n
            q_tokens = adapter.tokenizer.encode(
                f"Answer in one phrase.\n\n{question}<|im_end|>\n<|im_start|>assistant\n",
                add_special_tokens=False,
            )
        else:
            q_tokens = adapter.tokenizer.encode(f"Q: {question}\nA:")

        # Build cache directly (no quantization round-trip)
        from kvmemory.adapters.hf_adapter import HFAdapter
        import torch

        head_dim = adapter._d_model // adapter._num_heads
        kv_dtype = adapter.model.dtype
        device = adapter.model.device

        try:
            from transformers import DynamicCache

            class _DirectCache(DynamicCache):
                _total_past: int
                def get_seq_length(self, layer_idx: int = 0) -> int:
                    return self._total_past

            cache = _DirectCache()
            cache._total_past = n_ctx
            for li in range(adapter._num_layers):
                if li in kv_by_layer:
                    K, V = kv_by_layer[li]
                    K_in = K.to(dtype=kv_dtype, device=device).unsqueeze(0)
                    V_in = V.to(dtype=kv_dtype, device=device).unsqueeze(0)
                else:
                    K_in = torch.zeros(1, adapter._num_kv_heads, 0, head_dim, dtype=kv_dtype, device=device)
                    V_in = torch.zeros_like(K_in)
                cache.update(K_in, V_in, li)

            input_ids = torch.tensor([q_tokens], dtype=torch.long, device=device)
            out = adapter._manual_generate(
                input_ids=input_ids,
                past_kv=cache,
                position_offset=n_ctx,
                generation_kwargs={"max_new_tokens": 20, "do_sample": False},
            )
        except Exception as e:
            print(f"  Direct cache failed ({e}), falling back to quantized path")
            kv_q, scales = {}, {}
            for l, (K, V) in kv_by_layer.items():
                Kq, ks = quantize_int8(K.float())
                Vq, vs = quantize_int8(V.float())
                kv_q[l] = (Kq, Vq)
                scales[l] = (ks, vs)
            block = KVBlock.new(
                model_id=memory.config.model_id,
                session_id="oracle",
                chunk_text=context,
                token_count=n_ctx,
                hidden_vecs=hidden_vecs,
                kv_by_layer=kv_q,
                quant_scales=scales,
                original_positions=list(range(n_ctx)),
            )
            out = inject_and_generate(
                adapter=adapter, blocks=[block],
                current_tokens=q_tokens,
                generation_kwargs={"max_new_tokens": 20, "do_sample": False},
            )

        # Strip query from output
        answer_text = adapter.tokenizer.decode(
            out.sequences[0][len(q_tokens):], skip_special_tokens=True
        ).strip()

        gold_lower = gold.lower()
        ans_lower = answer_text.lower()
        correct = gold_lower in ans_lower or any(w in ans_lower for w in gold_lower.split() if len(w) > 2)

        if correct:
            oracle_hits += 1
        status = "PASS" if correct else "FAIL"
        print(f"[{status}] Q: '{question}'")
        print(f"       gold='{gold}'  pred='{answer_text[:80]}'")

    print(f"\nOracle injection accuracy: {oracle_hits}/{len(facts)} = {oracle_hits/len(facts)*100:.0f}%")
    print("\nDiagnosis:")
    if oracle_hits >= 4 and hits < 4:
        print("-> Injection works. Retrieval is broken (similarity collapse).")
    elif oracle_hits < 2 and hits < 2:
        print("-> BOTH retrieval and injection are broken. Check _build_cache / _manual_generate.")
    elif oracle_hits >= 4 and hits >= 4:
        print("-> Both retrieval and injection work! BEAM failure must be elsewhere.")
    else:
        print(f"-> Partial: retrieval hits={hits}/5, oracle hits={oracle_hits}/5. Investigate further.")

    return {
        "retrieval_hits": hits,
        "oracle_hits": oracle_hits,
        "n_facts": len(facts),
    }


# -----------------------------------------------------------------------
# BEAM benchmark
# -----------------------------------------------------------------------

@app.function(gpu="A10G", timeout=1800)
def run_beam(n: int = 20, model_key: str = "0.5b", text_rerank: bool = False):
    import sys, asyncio
    sys.path.insert(0, "/root/kv-memory")

    import torch
    adapter, model_cfg = _load_model_and_adapter(model_key)
    is_instruct = model_cfg["instruct"]
    print(f"Entropy-weighted pooling: ENABLED (eager attention)\n")

    import tempfile
    memory = _make_memory(adapter, model_key, tempfile.mkdtemp())
    print(f"Auto retrieval_layers: {memory.config.retrieval_layers}  (25/50/75% of {adapter.num_layers} layers)\n")

    from tests.beam_eval import (
        create_synthetic_dataset, run_rag_baseline, compute_metrics,
        print_comparison, score_answer, BEAMResult,
    )
    from kvmemory.core.retrieval import stage1_coarse, stage2_rerank_mmr
    from kvmemory.core.injector import inject_and_generate
    from kvmemory.memory import _build_session_filter
    import json as _json

    dataset = create_synthetic_dataset(n)
    print(f"Synthetic dataset: {len(dataset)} questions\n")

    # ---------- Phase 1: store all unique contexts ----------
    async def _store():
        seen = set()
        for q in dataset:
            h = hash(q.context)
            if h in seen:
                continue
            seen.add(h)
            memory.reset_baseline("beam")
            # Instruct models: store the system message + start of user message (no
            # close tag yet). The query tokens will complete this user turn, producing:
            #   <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
            #   <|im_start|>user\nContext: {ctx}\n\nAnswer in one phrase.\n\n{q}<|im_end|>\n
            #   <|im_start|>assistant\n
            # This is Qwen2.5-Instruct's standard format — the system message anchor
            # prevents markdown `![]()` generation artifacts from the model.
            if is_instruct:
                text_to_store = (
                    f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\nContext: {q.context}\n\n"
                )
            else:
                text_to_store = q.context
            await memory.store(text=text_to_store, session_id="beam",
                               explicit_signal=1.0, dedup_mode="hash")
        from kvmemory.core.write_pipeline import reset_hash_dedup
        await memory.drain_writes()
        reset_hash_dedup(memory.config.model_id, "beam")

    asyncio.run(_store())
    fmt = "user-message-prefix (instruct)" if is_instruct else "Q/A"
    print(f"Stored contexts. Running KV Memory eval ({fmt} format)...\n")

    # ---------- Phase 2: evaluate with Q: ... A: format ----------
    # This bypasses apply_chat_template so the base model sees the Q/A pattern
    # and naturally completes the answer from the injected KV context.
    import time
    kv_results = []
    session_filter = _build_session_filter(memory.config.model_id, "beam", None, False)

    def _text_rerank(candidate_ids: list[str], question: str) -> list[str]:
        """
        Re-rank candidate block IDs by BM25-style keyword overlap between
        the question and each block's stored chunk_text payload.

        On a 0.5B model the vector similarity margin is ~0.0003, which is
        too small to discriminate reliably. Keyword overlap over chunk_text
        (already in Qdrant payload) gives much stronger signal for factual QA.

        Returns: re-ranked list (best match first), same IDs as input.
        """
        import math, re

        if not candidate_ids:
            return candidate_ids

        # Fetch chunk_texts from Qdrant for all candidates in one scroll
        id_to_text: dict[str, str] = {}
        try:
            pts = memory.vector_db.client.retrieve(
                collection_name=memory.config.model_id,
                ids=candidate_ids,
                with_payload=True,
                with_vectors=False,
            )
            for pt in pts:
                id_to_text[str(pt.id)] = pt.payload.get("chunk_text", "")
        except Exception:
            return candidate_ids  # fallback: leave order unchanged

        # Simple BM25-like score: term frequency * IDF-proxy (log(1 + N/df))
        def tokenize(text: str) -> list[str]:
            return re.sub(r"[^\w]", " ", text.lower()).split()

        q_terms = set(tokenize(question))
        # IDF proxy: uniform (no corpus stats needed for small synthetic set)
        scores: list[tuple[float, str]] = []
        for bid in candidate_ids:
            text = id_to_text.get(bid, "")
            doc_terms = tokenize(text)
            if not doc_terms:
                scores.append((0.0, bid))
                continue
            tf_score = sum(doc_terms.count(t) for t in q_terms) / len(doc_terms)
            overlap = len(q_terms & set(doc_terms))
            scores.append((overlap * (1 + tf_score), bid))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [bid for _, bid in scores]

    from tests.beam_eval import _apply_chat_template

    async def _eval_one(q):
        retrieval_tokens = memory.adapter.tokenizer.encode(q.question)
        candidate_ids, query_vecs = await stage1_coarse(
            query_tokens=retrieval_tokens,
            adapter=memory.adapter,
            config=memory.config,
            vector_db=memory.vector_db,
            session_filter=session_filter,
        )

        if text_rerank:
            # Legacy BM25 text reranking (workaround for hidden-state retrieval collapse)
            reranked_ids = _text_rerank(candidate_ids, q.question)
            final_ids = reranked_ids[:memory.config.final_top_k] if reranked_ids else []
        else:
            # K-vector retrieval: use MMR reranking on cosine similarity scores
            final_ids = stage2_rerank_mmr(
                candidate_ids=candidate_ids,
                query_vecs=query_vecs,
                config=memory.config,
                vector_db=memory.vector_db,
                token_budget=memory.config.token_budget,
            )

        blocks = memory.kv_store.fetch(final_ids, model_id=memory.config.model_id)

        # Query tokens continue from the stored KV format.
        # Instruct: context stored as <|im_start|>user\nContext: {ctx}\n\n (no close tag)
        #           → query continues same user turn: Answer in one phrase.\n\n{q}<|im_end|>\n<|im_start|>assistant\n
        # Base: context stored as plain text → query as Q:/A: completion.
        if is_instruct:
            # ctx KV opened: <|im_start|>user\nContext: {ctx}\n\n  (no close tag)
            # gen_tokens continues that user turn without reopening it:
            gen_tokens = memory.adapter.tokenizer.encode(
                f"Answer in one phrase.\n\n{q.question}<|im_end|>\n<|im_start|>assistant\n",
                add_special_tokens=False,
            )
        else:
            gen_tokens = memory.adapter.tokenizer.encode(f"Q: {q.question}\nA:")

        t0 = time.perf_counter()
        output = inject_and_generate(
            adapter=memory.adapter,
            blocks=blocks,
            current_tokens=gen_tokens,
            generation_kwargs={"max_new_tokens": 30, "do_sample": False},
        )
        gen_ms = (time.perf_counter() - t0) * 1000

        answer_text = memory.adapter.tokenizer.decode(
            output.sequences[0][len(gen_tokens):], skip_special_tokens=True
        ).strip()
        answer_text = answer_text.split("\n")[0].strip()

        # Instruct models injected via KV sometimes generate markdown image
        # artifacts: "![](!answer!tokens)".  Remove the pattern, then strip any
        # remaining stray '!' tokens (which are injection noise, never part of
        # factual answers in this benchmark).
        if is_instruct:
            import re as _re
            # ![alt](url) → prefer non-empty group
            answer_text = _re.sub(
                r'!\[([^\]]*)\]\(([^)]*)\)',
                lambda m: (m.group(1) or m.group(2)).replace('!', ' '),
                answer_text,
            )
            answer_text = answer_text.replace('!', '')
            answer_text = ' '.join(answer_text.split())

        em, f1 = score_answer(answer_text, q.gold_answer)
        print(f"  [{q.question_id}] gold='{q.gold_answer}' pred='{answer_text[:60]}' "
              f"blocks={len(blocks)} EM={em:.0f} F1={f1:.2f}")

        return BEAMResult(
            question_id=q.question_id,
            question_type=q.question_type,
            predicted_answer=answer_text,
            gold_answer=q.gold_answer,
            correct=em > 0 or f1 > 0.5,
            em_score=em,
            f1_score=f1,
            prefill_tokens=len(gen_tokens),
            latency_ms=gen_ms,
            question=q.question,
            context=q.context,
            retrieved_chunks=[b.chunk_text for b in blocks],
        )

    async def _run():
        results = []
        for q in dataset:
            results.append(await _eval_one(q))
        rag = await run_rag_baseline(
            questions=dataset,
            adapter=adapter,
            generation_kwargs={"max_new_tokens": 60, "do_sample": False},
        )
        return results, rag

    kv_results, rag_results = asyncio.run(_run())
    kv_metrics  = compute_metrics(kv_results)
    rag_metrics = compute_metrics(rag_results)

    print_comparison(kv_metrics, rag_metrics, None)

    # Build HITL data for local review
    rag_by_id = {r.question_id: r for r in rag_results}
    hitl = []
    for r in kv_results:
        rag = rag_by_id.get(r.question_id)
        hitl.append({
            "id": r.question_id,
            "type": r.question_type,
            "question": r.question,
            "context": r.context,
            "gold": r.gold_answer,
            "kv_pred": r.predicted_answer,
            "kv_pass": r.correct,
            "rag_pred": rag.predicted_answer if rag else None,
            "rag_pass": rag.correct if rag else None,
            "retrieved_chunks": r.retrieved_chunks,
        })

    return {
        "kv_f1":  kv_metrics.overall_f1,
        "rag_f1": rag_metrics.overall_f1,
        "kv_tokens":  kv_metrics.avg_prefill_tokens,
        "rag_tokens": rag_metrics.avg_prefill_tokens,
        "token_savings_pct": round(
            100 * (1 - kv_metrics.avg_prefill_tokens / max(rag_metrics.avg_prefill_tokens, 1)), 1
        ),
        "hitl": hitl,
    }


@app.local_entrypoint()
def main(n: int = 20, diag: bool = False, model: str = "0.5b",
         text_rerank: bool = False, out: str = ""):
    """
    --model 0.5b        Qwen2.5-0.5B base on A10G  (default)
    --model 7b          Qwen2.5-7B-Instruct on A10G
    --diag              Retrieval + injection diagnostic (shows similarity scores)
    --n N               Number of synthetic questions (default 20)
    --text-rerank       Use BM25 text reranking instead of K-vector MMR
    --out results.json  Save HITL JSON to this file
    """
    import json

    if model not in MODELS:
        print(f"Unknown model '{model}'. Choose from: {list(MODELS.keys())}")
        return

    gpu = MODELS[model]["gpu"]

    if diag:
        print(f"\nRunning diagnostic with {MODELS[model]['id']} on {gpu}...\n")
        result = run_diagnostic.remote(model_key=model)
        print(f"\nRetrieval hits: {result['retrieval_hits']}/{result['n_facts']}")
        print(f"Oracle hits:    {result['oracle_hits']}/{result['n_facts']}")
        return

    retrieval_mode = "text-rerank (BM25)" if text_rerank else "K-vector MMR"
    print(f"\nRunning BEAM synthetic (n={n}) with {MODELS[model]['id']} on {gpu}")
    print(f"Retrieval: {retrieval_mode}\n")
    result = run_beam.remote(n=n, model_key=model, text_rerank=text_rerank)

    print(f"\n{'='*50}")
    print(f"KV Memory F1:  {result['kv_f1']:.3f}")
    print(f"RAG F1:        {result['rag_f1']:.3f}")
    delta = result['kv_f1'] - result['rag_f1']
    print(f"F1 delta:      {'+' if delta >= 0 else ''}{delta:.3f}")
    print(f"KV tokens/q:   {result['kv_tokens']:.0f}")
    print(f"RAG tokens/q:  {result['rag_tokens']:.0f}")
    print(f"Token savings: {result['token_savings_pct']}%")

    hitl_path = out or f"hitl_{model}.json"
    with open(hitl_path, "w") as f:
        json.dump(result["hitl"], f, indent=2)
    print(f"\nHITL results saved to {hitl_path}")
