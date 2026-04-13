"""
beam_eval.py — BEAM benchmark runner. Phase 3.

Measures:
  - Answer accuracy per question type
  - Prefill token count per turn vs RAG baseline (cost reduction metric)
  - Per-stage latency breakdown (stage1 ANN, stage2 MMR, fetch, inject+generate)

Baselines:
  - RAG (re-prefill retrieved text every turn)
  - Sliding window (fixed-size rolling context window)

Dataset:
  HuggingFace: Mohammadta/BEAM  (pip install datasets)
  Paper: "Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs"
         https://arxiv.org/abs/2510.27246

Usage:
    # From HuggingFace (recommended)
    python -m tests.beam_eval --model meta-llama/Llama-3-8B --hf --scale 128k

    # From local JSONL file
    python -m tests.beam_eval --model meta-llama/Llama-3-8B --dataset beam.jsonl

    # Quick smoke test with synthetic data (no dataset needed)
    python -m tests.beam_eval --model sshleifer/tiny-gpt2 --synthetic --n 10
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import dataclasses
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Dataset-native BEAM question types used by this benchmark.
QUESTION_TYPES = [
    "information_extraction",
    "knowledge_update",
    "multi_session_reasoning",
    "temporal_reasoning",
]


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class BEAMQuestion:
    question_id: str
    question_type: str
    context: str       # full conversation / document context
    question: str
    gold_answer: str


@dataclass
class BEAMResult:
    question_id: str
    question_type: str
    predicted_answer: str
    gold_answer: str
    correct: bool
    em_score: float          # exact match (0 or 1)
    f1_score: float          # token-level F1
    prefill_tokens: int      # tokens that hit prefill (query only for KV, full context for RAG)
    latency_ms: float        # total time from query to output
    stage1_ms: float = 0.0   # coarse ANN search
    stage2_ms: float = 0.0   # MMR rerank
    fetch_ms: float = 0.0    # blob store fetch
    generate_ms: float = 0.0 # inject + generate
    # HITL fields — populated by KV Memory eval, empty for baselines
    question: str = ""
    context: str = ""
    retrieved_chunks: list = field(default_factory=list)  # chunk texts that were injected
    candidate_count: int = 0
    selected_block_ids: list = field(default_factory=list)
    retrieval_diagnostics: dict = field(default_factory=dict)


@dataclass
class BEAMMetrics:
    accuracy_by_type: dict[str, float] = field(default_factory=dict)
    f1_by_type: dict[str, float] = field(default_factory=dict)
    overall_em: float = 0.0
    em_by_type: dict[str, float] = field(default_factory=dict)
    correct_rate_by_type: dict[str, float] = field(default_factory=dict)
    overall_correct_rate: float = 0.0
    overall_f1: float = 0.0
    avg_prefill_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    avg_stage1_ms: float = 0.0
    avg_stage2_ms: float = 0.0
    avg_fetch_ms: float = 0.0
    avg_generate_ms: float = 0.0
    prefill_reduction_pct: float = 0.0   # vs RAG baseline, filled in by print_comparison
    overall_accuracy: float = 0.0        # alias for overall_em, serialized by dataclasses.asdict()


# ------------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------------

def load_beam_jsonl(path: str) -> list[BEAMQuestion]:
    """
    Load BEAM questions from a local JSONL file.

    Expected format per line:
        {"id": "...", "type": "...", "context": "...", "question": "...", "answer": "..."}
    """
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            questions.append(BEAMQuestion(
                question_id=d["id"],
                question_type=d.get("type", "unknown"),
                context=d["context"],
                question=d["question"],
                gold_answer=d["answer"],
            ))
    logger.info("Loaded %d questions from %s", len(questions), path)
    return questions


def load_beam_hf(
    scale: str = "100K",
    question_types: Optional[list[str]] = None,
    max_questions: Optional[int] = None,
) -> list[BEAMQuestion]:
    """
    Load BEAM dataset from HuggingFace (Mohammadta/BEAM).

    Schema (one row = one full conversation):
      - chat:              list of sessions, each session is a list of
                           {role, content, time_anchor, ...} dicts
      - probing_questions: Python-string-serialized dict keyed by question type,
                           each value is a list of question dicts
      - conversation_id:   string

    Args:
        scale:          "100K", "500K", "1M"  (note: capitals, matches HF splits)
        question_types: filter to specific types (default: all 4 plan targets)
        max_questions:  cap total questions across all conversations

    Requires: pip install datasets
    """
    import ast

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library required. Install with: pip install datasets"
        )

    if question_types is None:
        question_types = QUESTION_TYPES

    logger.info("Loading BEAM dataset (scale=%s) from HuggingFace...", scale)
    ds = load_dataset("Mohammadta/BEAM", split=scale)

    questions: list[BEAMQuestion] = []
    selected_types = {qt.lower().strip() for qt in question_types}

    for row in ds:
        conv_id = str(row.get("conversation_id", f"conv_{len(questions)}"))

        # Build context from nested chat sessions.
        chat = row.get("chat", [])
        context_parts = []
        for session in chat:
            if not isinstance(session, list):
                continue
            for msg in session:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).strip()
                content = str(msg.get("content", "")).strip()
                if content:
                    label = role.capitalize() if role else "Unknown"
                    context_parts.append(f"{label}: {content}")
        context = "\n".join(context_parts)

        if not context:
            continue

        # Parse probing_questions (stored as Python string literal)
        raw_pq = row.get("probing_questions", "{}")
        try:
            probing_qs = ast.literal_eval(raw_pq) if isinstance(raw_pq, str) else raw_pq
        except (ValueError, SyntaxError):
            logger.warning("Failed to parse probing_questions for conv %s", conv_id)
            continue

        if not isinstance(probing_qs, dict):
            logger.warning("Unexpected probing_questions shape for conv %s", conv_id)
            continue

        # probing_qs is a dict: {question_type_label: [question_dict, ...]}
        for raw_type, qa_list in probing_qs.items():
            dataset_type = str(raw_type).lower().strip()
            if dataset_type not in selected_types:
                continue
            if not isinstance(qa_list, list):
                continue

            for i, qa in enumerate(qa_list):
                if not isinstance(qa, dict):
                    continue
                q_text = str(qa.get("question", "")).strip()
                a_text = str(qa.get("answer") or qa.get("ideal_response") or "").strip()
                if not q_text or not a_text:
                    continue

                questions.append(BEAMQuestion(
                    question_id=f"{conv_id}_{dataset_type}_{i}",
                    question_type=dataset_type,
                    context=context,
                    question=q_text,
                    gold_answer=a_text,
                ))

                if max_questions and len(questions) >= max_questions:
                    logger.info("Reached max_questions=%d, stopping", max_questions)
                    return questions

    logger.info("Loaded %d questions from %d conversations (types: %s)",
                len(questions), len(ds), question_types)
    return questions


def create_synthetic_dataset(n: int = 20) -> list[BEAMQuestion]:
    """
    Generate a small synthetic dataset for smoke testing without downloading BEAM.

    Covers all 4 question types with simple factual QA pairs.
    """
    templates = [
        # (type, context, question, answer)
        ("information_extraction",
         "Alice works as a software engineer at TechCorp. She joined in 2019 and leads the backend team.",
         "What is Alice's role at TechCorp?",
         "software engineer"),
        ("information_extraction",
         "The meeting was scheduled for Tuesday at 3pm in conference room B. Bob and Carol attended.",
         "Where was the meeting held?",
         "conference room B"),
        ("multi_session_reasoning",
         "Paris is the capital of France. France is in Western Europe. The Eiffel Tower is in Paris.",
         "In which continent is the Eiffel Tower located?",
         "Western Europe"),
        ("multi_session_reasoning",
         "David is the manager of Emma. Emma is the team lead of Frank. Frank works on the payments service.",
         "Who is the manager of the person leading Frank's team?",
         "David"),
        ("knowledge_update",
         "The project deadline was originally March 1st. After a client request, it was moved to April 15th.",
         "What is the current project deadline?",
         "April 15th"),
        ("knowledge_update",
         "The price of the subscription was $10/month. In the latest update, it was changed to $15/month.",
         "What is the current subscription price?",
         "$15"),
        ("temporal_reasoning",
         "The product launched in Q1 2023. The first major update came in Q3 2023. Version 2.0 shipped in Q1 2024.",
         "When did version 2.0 ship?",
         "Q1 2024"),
        ("temporal_reasoning",
         "Jane started the project in January. She completed the design phase in March. Development finished in June.",
         "What did Jane complete in March?",
         "design phase"),
    ]

    questions = []
    for i in range(n):
        t = templates[i % len(templates)]
        questions.append(BEAMQuestion(
            question_id=f"synthetic-{i}",
            question_type=t[0],
            context=t[1],
            question=t[2],
            gold_answer=t[3],
        ))
    return questions


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def _tokenize_answer(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for F1 scoring."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def score_answer(predicted: str, gold: str) -> tuple[float, float]:
    """
    Returns (exact_match, f1).

    exact_match: 1.0 if normalized strings match exactly.
    f1:          token-level F1 between predicted and gold.
    """
    pred_norm = " ".join(_tokenize_answer(predicted))
    gold_norm = " ".join(_tokenize_answer(gold))

    em = 1.0 if pred_norm == gold_norm or gold_norm in pred_norm else 0.0

    pred_tokens = _tokenize_answer(predicted)
    gold_tokens = _tokenize_answer(gold)

    if not pred_tokens or not gold_tokens:
        return em, 0.0

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)
    common_count = sum((pred_counter & gold_counter).values())
    precision = common_count / sum(pred_counter.values()) if pred_counter else 0.0
    recall    = common_count / sum(gold_counter.values()) if gold_counter else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return em, f1


# ------------------------------------------------------------------
# Tokenizer helpers
# ------------------------------------------------------------------

def _apply_chat_template(tokenizer, messages: list[dict]) -> list[int]:
    """Apply chat template and guarantee a list[int] regardless of tokenizer version.

    Handles: plain list[int], str, BatchEncoding/Mapping, tensor, nested list.
    """
    from collections.abc import Mapping

    result = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )

    if isinstance(result, str):
        try:
            return tokenizer.encode(result, add_special_tokens=False)
        except TypeError:
            return tokenizer.encode(result)

    if isinstance(result, Mapping):
        result = result["input_ids"]

    if hasattr(result, "tolist"):
        result = result.tolist()

    if result and isinstance(result[0], list):
        result = result[0]

    return [int(x) for x in result]


# ------------------------------------------------------------------
# KV Memory evaluation — instrumented, race-condition-free
# ------------------------------------------------------------------

async def run_kv_memory_eval(
    questions: list[BEAMQuestion],
    memory,
    session_id: Optional[str] = None,
    generation_kwargs: Optional[dict] = None,
    observer=None,
) -> list[BEAMResult]:
    """
    Evaluate KVMemory system on BEAM questions.

    Two-phase design to avoid the store→generate race condition:
      Phase 1: store all contexts, then drain the write queue fully.
      Phase 2: ask all questions with per-stage latency instrumentation.

    Per-stage timing is collected by calling the retrieval pipeline directly
    rather than going through memory.generate(), which gives us fine-grained
    breakdown without modifying the public API.
    """
    from kvmemory.core.retrieval import (
        build_candidate_diagnostics,
        stage1_coarse,
        stage2_rerank,
    )
    from kvmemory.core.injector import inject_and_generate
    from kvmemory.memory import _build_session_filter

    if generation_kwargs is None:
        generation_kwargs = {"max_new_tokens": 50, "do_sample": False}
    session_id = session_id or f"beam_session_{int(time.time())}"
    eval_observer = observer.child(phase="kv_memory_eval", benchmark="kv_memory") if observer else None

    # ----------------------------------------------------------------
    # Phase 1: store all contexts and drain the write queue
    # ----------------------------------------------------------------
    # Qwen2.5-7B (and most HF models) have a hard max_position_embeddings.
    # Contexts longer than that produce indexing errors in the attention layer.
    # Truncate to the model's limit before storing — the write pipeline will
    # chunk it anyway, so we just drop tokens that would be unreachable.
    tokenizer = memory.adapter.tokenizer
    _model_max = getattr(tokenizer, "model_max_length", None) or 131072
    # model_max_length is sometimes set to a huge sentinel (e.g. 1e30); cap sanely.
    if _model_max > 200_000:
        _model_max = 131072

    def _truncate_context(text: str) -> str:
        toks = tokenizer.encode(text)
        if len(toks) <= _model_max:
            return text
        logger.warning(
            "Context truncated from %d to %d tokens (model max=%d)",
            len(toks), _model_max, _model_max,
        )
        return tokenizer.decode(toks[:_model_max], skip_special_tokens=True)

    # Deduplicate contexts — multiple questions share the same conversation context.
    # Store each unique context only once to avoid redundant GPU passes + dedup churn.
    seen_contexts: set[int] = set()
    unique_questions = []
    for q in questions:
        h = hash(q.context)
        if h not in seen_contexts:
            seen_contexts.add(h)
            unique_questions.append(q)

    logger.info(
        "Phase 1: storing %d unique contexts (from %d questions)...",
        len(unique_questions), len(questions),
    )
    if eval_observer:
        eval_observer.emit(
            "phase_started",
            phase="store_contexts",
            benchmark="kv_memory",
            total_questions=len(questions),
            unique_contexts=len(unique_questions),
            session_id=session_id,
        )
    for q in unique_questions:
        # Reset the baseline loss tracker per document so each context gets a
        # fresh importance baseline rather than inheriting from previous documents.
        memory.reset_baseline(session_id)
        await memory.store(
            text=_truncate_context(q.context),
            session_id=session_id,
            trace_context={
                "run_id": eval_observer.run_id if eval_observer else None,
                "phase": "store_contexts",
                "question_id": q.question_id,
                "question_type": q.question_type,
                "benchmark": "kv_memory",
            },
            explicit_signal=1.0,   # bypass importance filter — context is always relevant
            dedup_mode="hash",     # hash dedup: each source chunk appears exactly once
        )

    # Drain: wait until every write has landed in Qdrant before asking questions.
    # Without this, generate() would retrieve from an empty collection.
    logger.info("Draining write queue (waiting for Qdrant writes to complete)...")
    await memory.drain_writes()
    # Reset hash dedup state AFTER drain — not before, to avoid clearing state
    # for in-flight queued writes.
    from kvmemory.core.write_pipeline import reset_hash_dedup
    reset_hash_dedup(memory.config.model_id, session_id)
    if eval_observer:
        eval_observer.emit(
            "phase_finished",
            phase="store_contexts",
            benchmark="kv_memory",
            total_questions=len(questions),
            session_id=session_id,
        )

    # ----------------------------------------------------------------
    # Phase 2: answer questions with per-stage timing
    # ----------------------------------------------------------------
    logger.info("Phase 2: answering %d questions...", len(questions))
    results = []
    if eval_observer:
        eval_observer.emit(
            "phase_started",
            phase="answer_questions",
            benchmark="kv_memory",
            total_questions=len(questions),
            session_id=session_id,
        )

    for q in questions:
        question_observer = (
            eval_observer.child(
                phase="answer_questions",
                question_id=q.question_id,
                question_type=q.question_type,
                session_id=session_id,
            )
            if eval_observer
            else None
        )
        if question_observer:
            question_observer.emit(
                "question_started",
                benchmark="kv_memory",
                question=q.question,
            )
        # For ANN retrieval — bare question tokens, no chat markers
        retrieval_tokens = memory.adapter.tokenizer.encode(q.question)

        # For generation — full chat-formatted prompt
        if hasattr(memory.adapter.tokenizer, "apply_chat_template"):
            generation_tokens = _apply_chat_template(
                memory.adapter.tokenizer,
                [{"role": "user", "content": q.question}],
            )
        else:
            generation_tokens = memory.adapter.tokenizer.encode(q.question)  # fallback

        session_filter = _build_session_filter(
            model_id=memory.config.model_id,
            session_id=session_id,
            agent_id=None,
            retrieve_shared=False,
        )

        t_start = time.perf_counter()

        # Stage 1: coarse ANN
        t0 = time.perf_counter()
        candidate_ids, query_vecs = await stage1_coarse(
            query_tokens=retrieval_tokens,
            adapter=memory.adapter,
            config=memory.config,
            vector_db=memory.vector_db,
            session_filter=session_filter,
        )
        t1 = time.perf_counter()
        if question_observer:
            question_observer.emit(
                "retrieval_stage1_done",
                benchmark="kv_memory",
                duration_ms=(t1 - t0) * 1000,
                candidate_count=len(candidate_ids),
                query_token_count=len(retrieval_tokens),
                retrieval_query_token_count=len(retrieval_tokens),
            )

        # Stage 2: configured rerank
        final_ids = stage2_rerank(
            candidate_ids=candidate_ids,
            query_vecs=query_vecs,
            config=memory.config,
            vector_db=memory.vector_db,
            token_budget=memory.config.token_budget,
        )
        t2 = time.perf_counter()
        if question_observer:
            question_observer.emit(
                "retrieval_stage2_done",
                benchmark="kv_memory",
                duration_ms=(t2 - t1) * 1000,
                selected_count=len(final_ids),
                selected_ids=final_ids,
            )

        # Fetch KV tensors from blob store
        blocks = memory.kv_store.fetch(final_ids, model_id=memory.config.model_id)
        t3 = time.perf_counter()
        if question_observer:
            question_observer.emit(
                "kv_fetch_done",
                benchmark="kv_memory",
                duration_ms=(t3 - t2) * 1000,
                block_count=len(blocks),
                block_ids=final_ids,
                token_count=sum(block.token_count for block in blocks),
            )

        # Inject + generate
        output = inject_and_generate(
            adapter=memory.adapter,
            blocks=blocks,
            current_tokens=generation_tokens,
            generation_kwargs=generation_kwargs,
        )
        t4 = time.perf_counter()
        if question_observer:
            question_observer.emit(
                "generation_done",
                benchmark="kv_memory",
                duration_ms=(t4 - t3) * 1000,
                output_chars=len(output.text),
            )

        # Strip the generation prompt from the output before scoring
        answer_text = memory.adapter.tokenizer.decode(
            output.sequences[0][len(generation_tokens):],
            skip_special_tokens=True,
        )

        em, f1 = score_answer(answer_text, q.gold_answer)
        try:
            retrieval_diagnostics = build_candidate_diagnostics(
                candidate_ids=candidate_ids,
                selected_ids=final_ids,
                query_vecs=query_vecs,
                config=memory.config,
                vector_db=memory.vector_db,
                question=q.question,
                gold_answer=q.gold_answer,
            )
        except Exception as exc:
            logger.warning("Failed to build retrieval diagnostics for %s: %s", q.question_id, exc)
            retrieval_diagnostics = {"error": str(exc), "error_type": type(exc).__name__}

        if question_observer:
            question_observer.emit(
                "score_done",
                benchmark="kv_memory",
                em_score=em,
                f1_score=f1,
                correct=em > 0 or f1 > 0.5,
                predicted_answer=answer_text,
                gold_answer=q.gold_answer,
            )
            question_observer.emit(
                "retrieval_diagnostics_done",
                benchmark="kv_memory",
                candidate_count=len(candidate_ids),
                selected_count=len(final_ids),
                gold_in_stage1=retrieval_diagnostics.get("gold_in_stage1"),
                gold_in_selected=retrieval_diagnostics.get("gold_in_selected"),
                best_gold_rerank_rank=retrieval_diagnostics.get("best_gold_rerank_rank"),
                best_gold_stage1_rank=retrieval_diagnostics.get("best_gold_stage1_rank"),
            )

        results.append(BEAMResult(
            question_id=q.question_id,
            question_type=q.question_type,
            predicted_answer=answer_text,
            gold_answer=q.gold_answer,
            correct=em > 0 or f1 > 0.5,
            em_score=em,
            f1_score=f1,
            prefill_tokens=len(generation_tokens),   # actual prefill cost
            latency_ms=(t4 - t_start) * 1000,
            stage1_ms=(t1 - t0) * 1000,
            stage2_ms=(t2 - t1) * 1000,
            fetch_ms=(t3 - t2) * 1000,
            generate_ms=(t4 - t3) * 1000,
            question=q.question,
            context=q.context,
            retrieved_chunks=[b.chunk_text for b in blocks],
            candidate_count=len(candidate_ids),
            selected_block_ids=final_ids,
            retrieval_diagnostics=retrieval_diagnostics,
        ))

        logger.debug(
            "Q %s [%s]: EM=%.0f F1=%.2f stage1=%.0fms stage2=%.0fms fetch=%.0fms gen=%.0fms",
            q.question_id, q.question_type, em, f1,
            results[-1].stage1_ms, results[-1].stage2_ms,
            results[-1].fetch_ms, results[-1].generate_ms,
        )
        if question_observer:
            question_observer.emit(
                "question_finished",
                benchmark="kv_memory",
                latency_ms=results[-1].latency_ms,
                prefill_tokens=results[-1].prefill_tokens,
            )

    if eval_observer:
        eval_observer.emit(
            "phase_finished",
            phase="answer_questions",
            benchmark="kv_memory",
            total_questions=len(results),
            session_id=session_id,
        )
    return results


# ------------------------------------------------------------------
# RAG baseline
# ------------------------------------------------------------------

def _rag_chunk_and_retrieve(
    context_tokens: list[int],
    query_tokens: list[int],
    context_budget: int,
    chunk_size: int = 256,
) -> list[int]:
    """
    Chunk context_tokens into fixed-size chunks, rank by cosine similarity
    to the query (mean token-ID vectors, no external model), and return the
    top chunks that fit within context_budget tokens (highest-scored first).
    """
    import math

    # Build chunks
    chunks = [
        context_tokens[i : i + chunk_size]
        for i in range(0, len(context_tokens), chunk_size)
        if context_tokens[i : i + chunk_size]
    ]
    if not chunks:
        return []

    # Vocab size is implicit — we just use raw token IDs as a sparse vector.
    # mean token ID as a 1-D "embedding" is too weak; instead build a
    # term-frequency bag-of-words vector over a fixed vocab bucket size.
    BUCKETS = 8192

    def bow(tokens: list[int]) -> list[float]:
        vec = [0.0] * BUCKETS
        for t in tokens:
            vec[t % BUCKETS] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    q_vec = bow(query_tokens)

    def cosine(chunk_vec: list[float]) -> float:
        return sum(a * b for a, b in zip(q_vec, chunk_vec))

    scored = sorted(
        ((cosine(bow(c)), c) for c in chunks),
        key=lambda x: x[0],
        reverse=True,
    )

    selected: list[int] = []
    used = 0
    for _, chunk in scored:
        if used + len(chunk) > context_budget:
            break
        selected.extend(chunk)
        used += len(chunk)

    return selected


async def run_rag_baseline(
    questions: list[BEAMQuestion],
    adapter,
    generation_kwargs: Optional[dict] = None,
    context_tokens: int = 4096,
    observer=None,
) -> list[BEAMResult]:
    """
    RAG baseline: chunk the context, retrieve top chunks by cosine similarity
    (bag-of-words over token IDs), and prepend them to the query.

    Context is capped at `context_tokens` so the prefill fits in GPU memory.
    """
    if generation_kwargs is None:
        generation_kwargs = {"max_new_tokens": 50, "do_sample": False}

    baseline_observer = observer.child(phase="lexical_rag_baseline", benchmark="lexical_rag_baseline") if observer else None
    if baseline_observer:
        baseline_observer.emit(
            "phase_started",
            phase="lexical_rag_baseline",
            benchmark="lexical_rag_baseline",
            total_questions=len(questions),
        )

    results = []
    for q in questions:
        question_observer = (
            baseline_observer.child(question_id=q.question_id, question_type=q.question_type)
            if baseline_observer
            else None
        )
        if question_observer:
            question_observer.emit("question_started", benchmark="lexical_rag_baseline", question=q.question)

        # Lexical retrieval — needs bare question token IDs
        query_retrieval_tokens = adapter.tokenizer.encode(q.question)

        ctx_tokens = adapter.tokenizer.encode(q.context)
        # Reserve prompt overhead from budget (rough estimate: 50 tokens for template markers)
        PROMPT_OVERHEAD = 50
        chunk_budget = max(0, context_tokens - len(query_retrieval_tokens) - PROMPT_OVERHEAD)
        retrieved_token_ids = _rag_chunk_and_retrieve(ctx_tokens, query_retrieval_tokens, chunk_budget)

        # Build one coherent chat message: context + question
        retrieved_text = adapter.tokenizer.decode(retrieved_token_ids, skip_special_tokens=True)
        user_msg = f"Context:\n{retrieved_text}\n\nQuestion:\n{q.question}\nAnswer concisely."

        if hasattr(adapter.tokenizer, "apply_chat_template"):
            combined_tokens = _apply_chat_template(
                adapter.tokenizer,
                [{"role": "user", "content": user_msg}],
            )
        else:
            combined_tokens = adapter.tokenizer.encode(user_msg)

        # Final budget guard: if combined_tokens still exceeds limit, trim context and rebuild
        if len(combined_tokens) > context_tokens:
            lo, hi = 0, len(retrieved_token_ids)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                candidate_text = adapter.tokenizer.decode(retrieved_token_ids[:mid], skip_special_tokens=True)
                candidate_msg = f"Context:\n{candidate_text}\n\nQuestion:\n{q.question}\nAnswer concisely."
                if hasattr(adapter.tokenizer, "apply_chat_template"):
                    candidate_toks = _apply_chat_template(
                        adapter.tokenizer,
                        [{"role": "user", "content": candidate_msg}],
                    )
                else:
                    candidate_toks = adapter.tokenizer.encode(candidate_msg)
                if len(candidate_toks) <= context_tokens:
                    lo = mid
                else:
                    hi = mid - 1
            if lo == 0:
                logger.debug("budget_guard: no context fits within budget, using question-only prompt")
                retrieved_text = ""
            else:
                retrieved_text = adapter.tokenizer.decode(retrieved_token_ids[:lo], skip_special_tokens=True)
            user_msg = f"Context:\n{retrieved_text}\n\nQuestion:\n{q.question}\nAnswer concisely."
            if hasattr(adapter.tokenizer, "apply_chat_template"):
                combined_tokens = _apply_chat_template(
                    adapter.tokenizer,
                    [{"role": "user", "content": user_msg}],
                )
            else:
                combined_tokens = adapter.tokenizer.encode(user_msg)

        t0 = time.perf_counter()
        output = adapter.inject_and_generate([], combined_tokens, generation_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        if question_observer:
            question_observer.emit(
                "generation_done",
                benchmark="lexical_rag_baseline",
                duration_ms=latency_ms,
                output_chars=len(output.text),
            )

        # Strip prompt from output before scoring
        rag_answer_text = adapter.tokenizer.decode(
            output.sequences[0][len(combined_tokens):],
            skip_special_tokens=True,
        )

        em, f1 = score_answer(rag_answer_text, q.gold_answer)
        if question_observer:
            question_observer.emit(
                "score_done",
                benchmark="lexical_rag_baseline",
                em_score=em,
                f1_score=f1,
                correct=em > 0 or f1 > 0.5,
                predicted_answer=rag_answer_text,
                gold_answer=q.gold_answer,
            )

        results.append(BEAMResult(
            question_id=q.question_id,
            question_type=q.question_type,
            predicted_answer=rag_answer_text,
            gold_answer=q.gold_answer,
            correct=em > 0 or f1 > 0.5,
            em_score=em,
            f1_score=f1,
            prefill_tokens=len(combined_tokens),
            latency_ms=latency_ms,
        ))
        if question_observer:
            question_observer.emit(
                "question_finished",
                benchmark="lexical_rag_baseline",
                latency_ms=latency_ms,
                prefill_tokens=len(combined_tokens),
            )

    if baseline_observer:
        baseline_observer.emit(
            "phase_finished",
            phase="lexical_rag_baseline",
            benchmark="lexical_rag_baseline",
            total_questions=len(results),
        )
    return results


# ------------------------------------------------------------------
# Sliding window baseline
# ------------------------------------------------------------------

async def run_sliding_window_baseline(
    questions: list[BEAMQuestion],
    adapter,
    window_tokens: int = 2000,
    generation_kwargs: Optional[dict] = None,
    observer=None,
) -> list[BEAMResult]:
    """
    Sliding window baseline: keep the last N tokens of context.

    Simulates a simple "always keep recent context" approach.
    """
    if generation_kwargs is None:
        generation_kwargs = {"max_new_tokens": 50, "do_sample": False}

    baseline_observer = observer.child(
        phase="sliding_window_baseline",
        benchmark="sliding_window_baseline",
    ) if observer else None
    if baseline_observer:
        baseline_observer.emit(
            "phase_started",
            phase="sliding_window_baseline",
            benchmark="sliding_window_baseline",
            total_questions=len(questions),
            window_tokens=window_tokens,
        )

    results = []
    for q in questions:
        question_observer = (
            baseline_observer.child(question_id=q.question_id, question_type=q.question_type)
            if baseline_observer
            else None
        )
        if question_observer:
            question_observer.emit(
                "question_started",
                benchmark="sliding_window_baseline",
                question=q.question,
            )
        query_retrieval_tokens = adapter.tokenizer.encode(q.question)
        context_token_ids = adapter.tokenizer.encode(q.context)

        PROMPT_OVERHEAD = 50
        available = window_tokens - len(query_retrieval_tokens) - PROMPT_OVERHEAD
        window_ids = context_token_ids[-available:] if available > 0 else []
        window_text = adapter.tokenizer.decode(window_ids, skip_special_tokens=True)

        sw_user_msg = f"Context:\n{window_text}\n\nQuestion:\n{q.question}\nAnswer concisely."
        if hasattr(adapter.tokenizer, "apply_chat_template"):
            combined_tokens = _apply_chat_template(
                adapter.tokenizer,
                [{"role": "user", "content": sw_user_msg}],
            )
        else:
            combined_tokens = adapter.tokenizer.encode(sw_user_msg)

        # Final budget guard: binary-search the maximum tail of context_token_ids that fits.
        if len(combined_tokens) > window_tokens:
            lo, hi = 0, len(context_token_ids)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                candidate_text = adapter.tokenizer.decode(context_token_ids[-mid:], skip_special_tokens=True)
                candidate_msg = f"Context:\n{candidate_text}\n\nQuestion:\n{q.question}\nAnswer concisely."
                if hasattr(adapter.tokenizer, "apply_chat_template"):
                    candidate_toks = _apply_chat_template(
                        adapter.tokenizer,
                        [{"role": "user", "content": candidate_msg}],
                    )
                else:
                    candidate_toks = adapter.tokenizer.encode(candidate_msg)
                if len(candidate_toks) <= window_tokens:
                    lo = mid
                else:
                    hi = mid - 1
            if lo == 0:
                logger.debug("budget_guard: no context fits within budget, using question-only prompt")
                window_text = ""
            else:
                window_text = adapter.tokenizer.decode(context_token_ids[-lo:], skip_special_tokens=True)
            sw_user_msg = f"Context:\n{window_text}\n\nQuestion:\n{q.question}\nAnswer concisely."
            if hasattr(adapter.tokenizer, "apply_chat_template"):
                combined_tokens = _apply_chat_template(
                    adapter.tokenizer,
                    [{"role": "user", "content": sw_user_msg}],
                )
            else:
                combined_tokens = adapter.tokenizer.encode(sw_user_msg)

        t0 = time.perf_counter()
        output = adapter.inject_and_generate([], combined_tokens, generation_kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000
        if question_observer:
            question_observer.emit(
                "generation_done",
                benchmark="sliding_window_baseline",
                duration_ms=latency_ms,
                output_chars=len(output.text),
            )

        # Strip prompt from output before scoring
        sw_answer_text = adapter.tokenizer.decode(
            output.sequences[0][len(combined_tokens):],
            skip_special_tokens=True,
        )

        em, f1 = score_answer(sw_answer_text, q.gold_answer)
        if question_observer:
            question_observer.emit(
                "score_done",
                benchmark="sliding_window_baseline",
                em_score=em,
                f1_score=f1,
                correct=em > 0 or f1 > 0.5,
                predicted_answer=sw_answer_text,
                gold_answer=q.gold_answer,
            )

        results.append(BEAMResult(
            question_id=q.question_id,
            question_type=q.question_type,
            predicted_answer=sw_answer_text,
            gold_answer=q.gold_answer,
            correct=em > 0 or f1 > 0.5,
            em_score=em,
            f1_score=f1,
            prefill_tokens=len(combined_tokens),
            latency_ms=latency_ms,
        ))
        if question_observer:
            question_observer.emit(
                "question_finished",
                benchmark="sliding_window_baseline",
                latency_ms=latency_ms,
                prefill_tokens=len(combined_tokens),
            )

    if baseline_observer:
        baseline_observer.emit(
            "phase_finished",
            phase="sliding_window_baseline",
            benchmark="sliding_window_baseline",
            total_questions=len(results),
            window_tokens=window_tokens,
        )
    return results


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_metrics(results: list[BEAMResult]) -> BEAMMetrics:
    if not results:
        return BEAMMetrics()

    by_type: dict[str, list[BEAMResult]] = {}
    for r in results:
        by_type.setdefault(r.question_type, []).append(r)

    accuracy_by_type = {
        qt: sum(r.em_score for r in rs) / len(rs)
        for qt, rs in by_type.items()
    }
    f1_by_type = {
        qt: sum(r.f1_score for r in rs) / len(rs)
        for qt, rs in by_type.items()
    }
    em_by_type = accuracy_by_type  # EM == accuracy
    correct_rate_by_type = {
        qt: sum(1 for r in rs if r.em_score > 0 or r.f1_score > 0.5) / len(rs)
        for qt, rs in by_type.items()
    }

    n = len(results)
    overall_em = sum(r.em_score for r in results) / n
    overall_correct_rate = sum(1 for r in results if r.em_score > 0 or r.f1_score > 0.5) / n
    return BEAMMetrics(
        accuracy_by_type=accuracy_by_type,
        f1_by_type=f1_by_type,
        overall_em=overall_em,
        em_by_type=em_by_type,
        correct_rate_by_type=correct_rate_by_type,
        overall_correct_rate=overall_correct_rate,
        overall_f1=sum(r.f1_score for r in results) / n,
        avg_prefill_tokens=sum(r.prefill_tokens for r in results) / n,
        avg_latency_ms=sum(r.latency_ms for r in results) / n,
        avg_stage1_ms=sum(r.stage1_ms for r in results) / n,
        avg_stage2_ms=sum(r.stage2_ms for r in results) / n,
        avg_fetch_ms=sum(r.fetch_ms for r in results) / n,
        avg_generate_ms=sum(r.generate_ms for r in results) / n,
        overall_accuracy=overall_em,
    )


def print_comparison(
    kv_metrics: BEAMMetrics,
    rag_metrics: BEAMMetrics,
    sw_metrics: Optional[BEAMMetrics] = None,
) -> None:
    W = 72
    cols = ["KV Memory", "Lexical RAG"]
    if sw_metrics:
        cols.append("Sliding Win")

    def row(label, *vals):
        fmt = f"  {{:<33}}"
        for v in vals:
            fmt += " {:>14}"
        print(fmt.format(label, *vals))

    print(f"\n{'=' * W}")
    print("BEAM Benchmark Results")
    print(f"{'=' * W}")
    print(f"  {'Metric':<33} {'KV Memory':>14} {'Lexical RAG':>14}" +
          (f" {'Sliding Win':>14}" if sw_metrics else ""))
    print(f"  {'-' * (W - 2)}")

    row("Exact Match (EM)",
        f"{kv_metrics.overall_em:.1%}",
        f"{rag_metrics.overall_em:.1%}",
        *([f"{sw_metrics.overall_em:.1%}"] if sw_metrics else []))
    row("Correct Rate (EM or F1>0.5)",
        f"{kv_metrics.overall_correct_rate:.1%}",
        f"{rag_metrics.overall_correct_rate:.1%}",
        *([f"{sw_metrics.overall_correct_rate:.1%}"] if sw_metrics else []))
    row("Overall F1",
        f"{kv_metrics.overall_f1:.1%}",
        f"{rag_metrics.overall_f1:.1%}",
        *([f"{sw_metrics.overall_f1:.1%}"] if sw_metrics else []))

    print()
    for qt in QUESTION_TYPES:
        kv_f1 = kv_metrics.f1_by_type.get(qt)
        rag_f1 = rag_metrics.f1_by_type.get(qt)
        if kv_f1 is None and rag_f1 is None:
            continue
        label = "  " + qt.replace("_", " ").title()
        row(label,
            f"{kv_f1:.1%}" if kv_f1 is not None else "  —",
            f"{rag_f1:.1%}" if rag_f1 is not None else "  —",
            *([f"{sw_metrics.f1_by_type.get(qt, 0):.1%}"] if sw_metrics else []))

    print()
    row("Avg Prefill Tokens",
        f"{kv_metrics.avg_prefill_tokens:.0f}",
        f"{rag_metrics.avg_prefill_tokens:.0f}",
        *([f"{sw_metrics.avg_prefill_tokens:.0f}"] if sw_metrics else []))

    if rag_metrics.avg_prefill_tokens > 0:
        reduction = 1 - kv_metrics.avg_prefill_tokens / rag_metrics.avg_prefill_tokens
        row("Prefill Reduction vs RAG", f"{reduction:.1%}", "—",
            *([f"{1 - sw_metrics.avg_prefill_tokens / rag_metrics.avg_prefill_tokens:.1%}"]
              if sw_metrics else []))

    print()
    row("Avg Total Latency (ms)",
        f"{kv_metrics.avg_latency_ms:.0f}",
        f"{rag_metrics.avg_latency_ms:.0f}",
        *([f"{sw_metrics.avg_latency_ms:.0f}"] if sw_metrics else []))

    if kv_metrics.avg_stage1_ms > 0:
        print()
        print("  KV Memory latency breakdown:")
        row("    Stage 1 ANN (ms)", f"{kv_metrics.avg_stage1_ms:.0f}", "", "")
        row("    Stage 2 Rerank (ms)", f"{kv_metrics.avg_stage2_ms:.0f}", "", "")
        row("    Fetch KV (ms)",    f"{kv_metrics.avg_fetch_ms:.0f}", "", "")
        row("    Inject+Gen (ms)",  f"{kv_metrics.avg_generate_ms:.0f}", "", "")

    print(f"{'=' * W}\n")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

async def main(args) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # httpcore/httpx are extremely chatty at DEBUG; cap them at WARNING
    for _noisy in ("httpcore", "httpx", "hpack"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    from kvmemory import KVMemory, KVMemoryConfig, ObservabilityStore
    from kvmemory.adapters.hf_adapter import HFAdapter
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    dataset_source = "synthetic" if args.synthetic else ("huggingface" if args.hf else "jsonl")
    obs_store = ObservabilityStore(args.obs_dir)
    run_observer = obs_store.create_run(
        config={
            "model": args.model,
            "dtype": args.dtype,
            "dataset": args.dataset,
            "hf": args.hf,
            "synthetic": args.synthetic,
            "scale": args.scale,
            "n": args.n,
            "retrieval_layers": args.retrieval_layers,
            "retrieval_query_source": args.retrieval_query_source,
            "stage2_reranker": args.stage2_reranker,
            "token_budget": args.token_budget,
            "max_new_tokens": args.max_new_tokens,
        },
        metadata={"dataset_source": dataset_source, "run_kind": "beam_benchmark"},
    )

    # Safe atexit handler: marks the run as failed if process exits unexpectedly.
    # Note: atexit won't catch SIGKILL or CUDA process death — stale-run detection
    # in list_runs() handles those cases.
    _run_finished = False

    def _atexit_handler():
        if not _run_finished:
            try:
                run_observer.fail(error="process terminated unexpectedly")
            except Exception:
                pass  # atexit cannot propagate exceptions

    atexit.register(_atexit_handler)

    memory = None
    try:
        run_observer.emit("run_started", dataset_source=dataset_source)
        run_observer.emit("phase_started", phase="dataset_load", dataset_source=dataset_source)

        if args.synthetic:
            questions = create_synthetic_dataset(n=args.n or 20)
            logger.info("Using %d synthetic questions", len(questions))
        elif args.hf:
            questions = load_beam_hf(
                scale=args.scale,
                question_types=QUESTION_TYPES,
                max_questions=args.n,
            )
        else:
            questions = load_beam_jsonl(args.dataset)
            if args.n:
                questions = questions[:args.n]

        if not questions:
            logger.error("No questions loaded. Exiting.")
            run_observer.fail(error="No questions loaded")
            return

        run_observer.update_metadata(
            question_count=len(questions),
            question_types=sorted({question.question_type for question in questions}),
        )
        run_observer.emit(
            "phase_finished",
            phase="dataset_load",
            dataset_source=dataset_source,
            question_count=len(questions),
        )

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtype_map.get(args.dtype, torch.float16)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        run_observer.emit(
            "phase_started",
            phase="model_load",
            model=args.model,
            dtype=args.dtype,
            device=device,
        )
        model_kwargs = {"torch_dtype": torch_dtype}
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        if device != "cuda":
            model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        adapter = HFAdapter(model, tokenizer)
        run_observer.emit(
            "phase_finished",
            phase="model_load",
            model=args.model,
            dtype=args.dtype,
            device=device,
        )

        from kvmemory.utils.model_id import sanitize_model_id
        # After Fixes 1, 8, 9, 18, wipe stale data before a real benchmark run:
        #   Qdrant:    vector_db.delete_collection(sanitize_model_id(args.model))
        #   Blob store: rename/move ./kv_store/<sanitize_model_id(args.model)>/ to a backup dir
        # Do NOT auto-delete. Wipe manually once, then re-run.
        config_kwargs = dict(
            model_id=sanitize_model_id(args.model),
            retrieval_layers=args.retrieval_layers,
            retrieval_query_source=args.retrieval_query_source,
            stage2_reranker=args.stage2_reranker,
            token_budget=args.token_budget,
            importance_threshold=0.0 if args.synthetic else 0.3,
        )
        if args.dedup_threshold is not None:
            config_kwargs["dedup_threshold"] = args.dedup_threshold
        if args.capture_batch_size is not None:
            config_kwargs["capture_batch_size"] = args.capture_batch_size
        config = KVMemoryConfig(**config_kwargs)
        memory = KVMemory(adapter=adapter, config=config, observer=run_observer)
        gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
        benchmark_session_id = f"beam_{run_observer.run_id}"
        run_observer.update_metadata(session_id=benchmark_session_id)

        logger.info("Running KV Memory evaluation (%d questions)...", len(questions))
        kv_results = await run_kv_memory_eval(
            questions,
            memory,
            session_id=benchmark_session_id,
            generation_kwargs=gen_kwargs,
            observer=run_observer,
        )

        logger.info("Running Lexical RAG baseline...")
        rag_results = await run_rag_baseline(
            questions,
            adapter,
            generation_kwargs=gen_kwargs,
            context_tokens=args.rag_context_tokens,
            observer=run_observer,
        )

        logger.info("Running sliding window baseline (window=%d tokens)...", args.token_budget)
        sw_results = await run_sliding_window_baseline(
            questions,
            adapter,
            window_tokens=args.token_budget,
            generation_kwargs=gen_kwargs,
            observer=run_observer,
        )

        kv_metrics = compute_metrics(kv_results)
        rag_metrics = compute_metrics(rag_results)
        sw_metrics = compute_metrics(sw_results)
        print_comparison(kv_metrics, rag_metrics, sw_metrics)

        summary = {
            "question_count": len(questions),
            "kv_metrics": dataclasses.asdict(kv_metrics),
            "lexical_rag_metrics": dataclasses.asdict(rag_metrics),
            "sliding_window_metrics": dataclasses.asdict(sw_metrics),
            "kv_results": [dataclasses.asdict(r) for r in kv_results],
            "lexical_rag_results": [dataclasses.asdict(r) for r in rag_results],
            "sliding_window_results": [dataclasses.asdict(r) for r in sw_results],
        }
        run_observer.update_summary(**summary)

        if args.output:
            _save_results(args.output, kv_results, rag_results, sw_results, kv_metrics, rag_metrics, sw_metrics)
            logger.info("Results saved to %s", args.output)
            run_observer.emit("results_saved", output_path=args.output)

        _run_finished = True
        run_observer.finish(summary=summary)
    except Exception as exc:
        run_observer.emit(
            "error",
            level="error",
            phase="benchmark_run",
            message="Benchmark run failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        _run_finished = True
        run_observer.fail(error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        if memory is not None:
            await memory.close()


def _save_results(path, kv_results, rag_results, sw_results,
                  kv_metrics, rag_metrics, sw_metrics) -> None:
    with open(path, "w") as f:
        json.dump({
            "kv_memory": {
                "metrics": dataclasses.asdict(kv_metrics),
                "results": [dataclasses.asdict(r) for r in kv_results],
            },
            "lexical_rag_baseline": {
                "metrics": dataclasses.asdict(rag_metrics),
                "results": [dataclasses.asdict(r) for r in rag_results],
            },
            "sliding_window": {
                "metrics": dataclasses.asdict(sw_metrics),
                "results": [dataclasses.asdict(r) for r in sw_results],
            },
        }, f, indent=2)

    # Also write a clean HITL review file: one entry per question, easy to read
    hitl_path = path.replace(".json", "_hitl.json")
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
            "candidate_count": r.candidate_count,
            "selected_block_ids": r.selected_block_ids,
            "retrieval_diagnostics": r.retrieval_diagnostics,
        })
    with open(hitl_path, "w") as f:
        json.dump(hitl, f, indent=2)
    logger.info("HITL review file saved to %s", hitl_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEAM benchmark for KV Memory System")

    # Dataset source (mutually exclusive)
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--dataset", help="Path to local JSONL file")
    src.add_argument("--hf", action="store_true", help="Load from HuggingFace (Mohammadta/BEAM)")
    src.add_argument("--synthetic", action="store_true",
                     help="Use synthetic data for smoke testing (no download needed)")

    # Dataset options
    parser.add_argument("--scale", default="100K",
                        choices=["100K", "500K", "1M"],
                        help="BEAM dataset scale (only with --hf)")
    parser.add_argument("--n", type=int, default=None,
                        help="Max number of questions to evaluate")

    # Model
    parser.add_argument("--model", required=True,
                        help="HuggingFace model name or path")
    parser.add_argument("--retrieval-layers", type=int, nargs="+", default=[8, 16, 24],
                        help="Layers to use for retrieval (default: 8 16 24)")
    parser.add_argument("--retrieval-query-source", default="k_vectors",
                        choices=["k_vectors", "q_vectors"],
                        help="Query vector source for retrieval (default: k_vectors). "
                             "Use q_vectors for routerless Q-to-K retrieval.")
    parser.add_argument("--stage2-reranker", default="mmr",
                        choices=["mmr", "qk"],
                        help="Stage 2 selector (default: mmr). qk ranks by pure Q.K relevance.")
    parser.add_argument("--token-budget", type=int, default=2000,
                        help="Hard cap on injected tokens (default: 2000)")
    parser.add_argument("--rag-context-tokens", type=int, default=4096,
                        help="Max context tokens for RAG baseline (default: 4096)")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--capture-batch-size", type=int, default=None,
                        help="Chunks captured per GPU forward pass (default: 8). "
                             "Lower = less GPU memory pressure.")
    parser.add_argument("--dedup-threshold", type=float, default=None,
                        help="Semantic dedup threshold (minimum cosine score to flag a chunk as "
                             "duplicate; higher = fewer duplicates removed). "
                             "Default: use KVMemoryConfig default (0.95). "
                             "Note: this is a MINIMUM score threshold — lower values cause MORE "
                             "chunks to be flagged as duplicates.")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model weight dtype (default: float16)")

    # Output
    parser.add_argument("--output", help="Path to save JSON results")
    parser.add_argument("--obs-dir", default=".kvmem_obs",
                        help="Directory for durable run/event tracking")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if not args.dataset and not args.hf and not args.synthetic:
        parser.error("Specify one of --dataset, --hf, or --synthetic")

    asyncio.run(main(args))
