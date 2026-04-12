"""
modal_bench.py — Does KV memory actually work?

Stores N facts, asks questions, compares:
  - Baseline: raw model.generate(), no memory
  - KV Memory: retrieve + inject (sweep of final_top_k + min_relevance configs)

Shows per-question which chunks were retrieved so you can see
exactly where retrieval is going wrong.

Usage:
    modal run modal_bench.py
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
    )
    .add_local_dir(".", remote_path="/root/kv-memory", ignore=_exclude, copy=True)
    .run_commands(
        "pip install -q --upgrade setuptools pip",
        "pip install -e /root/kv-memory --no-deps -q",
    )
)

app = modal.App("kv-memory-bench", image=image)

# (fact, question, answer_keywords)
BENCH = [
    (
        "Marie Curie was born on November 7, 1867, in Warsaw, Poland. "
        "She was the first woman to win a Nobel Prize.",
        "In what city was Marie Curie born?",
        ["warsaw"],
    ),
    (
        "The Amazon River is approximately 6,400 kilometers long, "
        "making it the longest river in South America.",
        "How long is the Amazon River?",
        ["6400", "6,400"],
    ),
    (
        "Python programming language was created by Guido van Rossum "
        "and first released in 1991.",
        "Who created the Python programming language?",
        ["guido", "rossum"],
    ),
    (
        "The human body has 206 bones in total. "
        "The femur, located in the thigh, is the longest bone.",
        "How many bones does the human body have?",
        ["206"],
    ),
    (
        "Mount Everest, located in the Himalayas on the border of Nepal and Tibet, "
        "stands at 8,849 meters above sea level.",
        "How tall is Mount Everest in meters?",
        ["8849", "8,849"],
    ),
    (
        "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
        "What is the speed of light in km/s?",
        ["299", "299,792", "299792"],
    ),
    (
        "William Shakespeare was born in Stratford-upon-Avon, England, in April 1564. "
        "He wrote 37 plays and 154 sonnets.",
        "In which town was William Shakespeare born?",
        ["stratford", "avon"],
    ),
    (
        "The Great Barrier Reef, located off the coast of Queensland, Australia, "
        "is the world's largest coral reef system, stretching over 2,300 kilometers.",
        "In which country is the Great Barrier Reef located?",
        ["australia"],
    ),
    (
        "Albert Einstein published his special theory of relativity in 1905. "
        "His famous equation E=mc² describes the equivalence of mass and energy.",
        "What year did Einstein publish special relativity?",
        ["1905"],
    ),
    (
        "The Sahara Desert covers approximately 9.2 million square kilometers, "
        "making it the largest hot desert on Earth.",
        "How many square kilometers does the Sahara Desert cover?",
        ["9.2", "9,200,000"],
    ),
]

MODEL_ID = "Qwen/Qwen2.5-0.5B"
RETRIEVAL_LAYERS = [6, 12, 18]


@app.function(gpu="T4", timeout=1800)
def run_bench():
    import asyncio, tempfile, sys, importlib
    sys.path.insert(0, "/root/kv-memory")

    import torch, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvmemory.config import KVMemoryConfig
    from kvmemory.adapters.hf_adapter import HFAdapter
    from kvmemory.memory import KVMemory
    from kvmemory.core.retrieval import (
        compute_query_vecs, stage1_coarse, stage2_rerank_mmr, _default_layer_weights
    )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model: {MODEL_ID}  |  Questions: {len(BENCH)}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True
    )
    model.eval()
    adapter = HFAdapter(model, tokenizer)

    GEN_KWARGS = {"max_new_tokens": 40}
    n = len(BENCH)

    def score(answer: str, keywords: list) -> bool:
        a = answer.lower()
        return any(k.lower() in a for k in keywords)

    def clean(text: str, prompt: str) -> str:
        if text.startswith(prompt):
            text = text[len(prompt):]
        return text.strip()

    # -----------------------------------------------------------------------
    # BASELINE
    # -----------------------------------------------------------------------
    print("=" * 64)
    print("BASELINE — raw generate, no memory")
    print("=" * 64)
    baseline_scores = []
    for fact, question, keywords in BENCH:
        tokens = tokenizer.encode(question)
        out = adapter.generate(tokens, GEN_KWARGS)
        answer = clean(out.text, question)
        ok = score(answer, keywords)
        baseline_scores.append(ok)
        print(f"  {'✓' if ok else '✗'}  {question}")
        if not ok:
            print(f"       → {answer[:90]}")

    baseline_correct = sum(baseline_scores)
    print(f"\n  Score: {baseline_correct}/{n} ({100*baseline_correct//n}%)\n")

    # -----------------------------------------------------------------------
    # Helper: build a fresh KVMemory with given config params, store all facts
    # -----------------------------------------------------------------------
    def build_memory(final_top_k: int, min_relevance: float) -> KVMemory:
        tmp = tempfile.mkdtemp()
        cfg = KVMemoryConfig(
            model_id="bench",
            retrieval_layers=RETRIEVAL_LAYERS,
            qdrant_url=":memory:",
            blob_store_path=tmp,
            importance_threshold=0.0,
            dedup_threshold=1.01,
            final_top_k=final_top_k,
            min_relevance=min_relevance,
        )
        mem = KVMemory(adapter=adapter, config=cfg)

        async def _store():
            for fact, _, _ in BENCH:
                await mem.store(fact, session_id="bench", explicit_signal=1.0)
            await mem.drain_writes(timeout=120)

        asyncio.run(_store())
        return mem

    # -----------------------------------------------------------------------
    # RETRIEVAL DIAGNOSTICS — show what's being fetched per question
    # -----------------------------------------------------------------------
    print("=" * 64)
    print("RETRIEVAL DIAGNOSTICS  (top_k=10, no min_relevance)")
    print("Shows which stored fact is retrieved for each question")
    print("=" * 64)
    diag_mem = build_memory(final_top_k=10, min_relevance=0.0)
    layer_weights = _default_layer_weights(RETRIEVAL_LAYERS)

    async def _diagnose():
        for fact, question, keywords in BENCH:
            q_tokens = tokenizer.encode(question)
            sf = {"model_id": "bench", "session_id": "bench",
                  "agent_id": None, "retrieve_shared": False}
            cand_ids, qvecs = await stage1_coarse(
                query_tokens=q_tokens, adapter=adapter,
                config=diag_mem.config, vector_db=diag_mem.vector_db,
                session_filter=sf,
            )
            # Fetch candidates with their vectors to compute per-block similarity
            candidates = diag_mem.vector_db.fetch_with_vectors("bench", cand_ids)
            scored = []
            for c in candidates:
                sim = sum(
                    w * float(np.dot(qvecs[l], np.array(c["vector"][f"layer_{l}"], dtype=np.float32)))
                    for l, w in layer_weights.items()
                    if f"layer_{l}" in (c["vector"] or {})
                )
                text = (c["payload"] or {}).get("chunk_text", "")[:60]
                scored.append((sim, text))
            scored.sort(reverse=True)

            print(f"\n  Q: {question}")
            for rank, (sim, txt) in enumerate(scored[:3], 1):
                print(f"    #{rank} sim={sim:.3f}  '{txt}...'")

    asyncio.run(_diagnose())

    # -----------------------------------------------------------------------
    # SWEEP: test configs — top_k and min_relevance
    # -----------------------------------------------------------------------
    configs = [
        ("top_k=10, no cutoff",  10, 0.0),
        ("top_k=3,  no cutoff",   3, 0.0),
        ("top_k=1,  no cutoff",   1, 0.0),
        ("top_k=3,  min_rel=0.1", 3, 0.1),
        ("top_k=3,  min_rel=0.2", 3, 0.2),
    ]

    all_results = {}
    for label, top_k, min_rel in configs:
        mem = build_memory(final_top_k=top_k, min_relevance=min_rel)

        async def _eval(mem=mem):
            results = []
            for fact, question, keywords in BENCH:
                out = await mem.generate(
                    prompt=question,
                    session_id="bench",
                    generation_kwargs=GEN_KWARGS,
                )
                answer = clean(out.text, question)
                results.append(score(answer, keywords))
            return results

        scores = asyncio.run(_eval())
        correct = sum(scores)
        all_results[label] = (correct, scores)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 64)
    print("RESULTS SUMMARY")
    print("=" * 64)
    print(f"  {'Config':<28}  {'Score':>6}  {'vs Baseline':>12}")
    print(f"  {'-'*28}  {'-'*6}  {'-'*12}")
    print(f"  {'Baseline (no memory)':<28}  {baseline_correct}/{n} ({100*baseline_correct//n:2d}%)")
    best_correct = -1
    best_label = ""
    for label, (correct, _) in all_results.items():
        delta = correct - baseline_correct
        sign = "+" if delta >= 0 else ""
        verdict = f"{sign}{delta}"
        print(f"  {label:<28}  {correct}/{n} ({100*correct//n:2d}%)  {verdict:>12}")
        if correct > best_correct:
            best_correct = correct
            best_label = label

    # Per-question breakdown for best config
    print(f"\n  Best config: {best_label}  ({best_correct}/{n})")
    best_scores = all_results[best_label][1]
    baseline_scores_arr = baseline_scores

    helped = [BENCH[i][1] for i in range(n) if best_scores[i] and not baseline_scores_arr[i]]
    hurt   = [BENCH[i][1] for i in range(n) if not best_scores[i] and baseline_scores_arr[i]]
    both_wrong = [BENCH[i][1] for i in range(n) if not best_scores[i] and not baseline_scores_arr[i]]

    if helped:
        print(f"\n  Memory HELPED ({len(helped)}):")
        for q in helped: print(f"    + {q}")
    if hurt:
        print(f"\n  Memory HURT ({len(hurt)}):")
        for q in hurt: print(f"    - {q}")
    if both_wrong:
        print(f"\n  Both wrong ({len(both_wrong)}):")
        for q in both_wrong: print(f"    ? {q}")

    overall_verdict = "WORKS" if best_correct > baseline_correct else (
        "NEUTRAL" if best_correct == baseline_correct else "BROKEN"
    )
    print(f"\n  Overall verdict: KV Memory is {overall_verdict}\n")

    return {
        "baseline": baseline_correct,
        "results": {k: v[0] for k, v in all_results.items()},
        "verdict": overall_verdict,
        "n": n,
    }


@app.local_entrypoint()
def main():
    r = run_bench.remote()
    print(f"\nBaseline {r['baseline']}/{r['n']} | "
          + " | ".join(f"{k}: {v}/{r['n']}" for k, v in r["results"].items())
          + f" | Verdict: {r['verdict']}")
