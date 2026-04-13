"""
retrieval.py — Two-stage retrieval: coarse ANN + MMR rerank.

Stage 1 (coarse): Multi-layer ANN search in Qdrant. Returns top-200 candidate block_ids.
Stage 2 (MMR):    Maximal Marginal Relevance rerank over candidates. Returns top-10 winners.

Both stages enforce the token budget hard cap.
"""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn.functional as F  # used by stage2_rerank_mmr (normalize)

if TYPE_CHECKING:
    from ..adapters.base import BaseAdapter
    from ..config import KVMemoryConfig
    from ..storage.vector_db import VectorDB

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Hidden vector computation (shared between write and read paths)
# ------------------------------------------------------------------

def compute_retrieval_vec(
    hidden: torch.Tensor,
    token_count: int,
) -> np.ndarray:
    """
    Compute a single normalized retrieval vector from a hidden state tensor.

    Steps:
      1. Unweighted mean pool over sequence dimension — simple and proven.
         (Previous entropy-softmax weighting was broken: softmax over d_model
         dims ~3584 produces a near-flat distribution for every token, making
         all entropies near-identical and all salience weights near-uniform,
         which destroyed inter-chunk discrimination.)
      2. sqrt(n) length normalization (harmless after L2 norm; does not change
         final cosine direction, kept for compatibility)
      3. L2 normalization (stable cosine comparison in Qdrant)

    Args:
        hidden:      [seq, d_model] float32 tensor
        token_count: number of tokens in the chunk

    Returns:
        [d_model] float32 numpy array, L2-normalized
    """
    if hidden.dim() != 2:
        raise ValueError(f"Expected [seq, d_model], got {hidden.shape}")

    # Step 1: Simple unweighted mean pool
    pooled = hidden.mean(dim=0)   # [d_model]

    # Step 2: sqrt(n) length normalization
    pooled = pooled / math.sqrt(max(token_count, 1))

    # Step 3: L2 normalization
    pooled = F.normalize(pooled, dim=0)

    return pooled.cpu().numpy().astype(np.float32)


def compute_k_vec(K: torch.Tensor) -> np.ndarray:
    """
    Compute a retrieval vector from attention K tensors (Option B).

    Uses the model's attention keys instead of hidden states — W_K projects
    into the attention subspace, which is more content-discriminative than
    raw hidden states for short chunks.

    Args:
        K: [kv_heads, seq, head_dim] float tensor

    Returns:
        [kv_heads * head_dim] float32 numpy array, L2-normalized
    """
    if K.dim() != 3:
        raise ValueError(f"Expected [kv_heads, seq, head_dim], got {K.shape}")

    pooled = K.float().mean(dim=1)      # [kv_heads, head_dim] — mean over seq
    flat = pooled.flatten()              # [kv_heads * head_dim]
    normed = F.normalize(flat, dim=0)
    return normed.cpu().numpy().astype(np.float32)


def compute_q_vec(Q: torch.Tensor, num_kv_heads: Optional[int] = None) -> np.ndarray:
    """
    Compute a retrieval query vector from attention Q tensors.

    Q has query-head shape [num_heads, seq, head_dim]. Stored memory K vectors
    use KV-head shape [num_kv_heads, seq, head_dim]. For GQA models, multiple
    query heads share one KV head in attention, so each query-head group is
    averaged down to the matching KV-head shape before pooling.
    """
    if Q.dim() != 3:
        raise ValueError(f"Expected [num_heads, seq, head_dim], got {Q.shape}")

    q = Q.float()
    q_heads, seq_len, head_dim = q.shape
    if num_kv_heads is not None and num_kv_heads > 0 and q_heads != num_kv_heads:
        if q_heads % num_kv_heads != 0:
            raise ValueError(
                f"Cannot group {q_heads} query heads into {num_kv_heads} KV heads"
            )
        group_size = q_heads // num_kv_heads
        q = q.reshape(num_kv_heads, group_size, seq_len, head_dim).mean(dim=1)

    pooled = q.mean(dim=1)       # [heads_or_kv_heads, head_dim]
    flat = pooled.flatten()
    normed = F.normalize(flat, dim=0)
    return normed.cpu().numpy().astype(np.float32)


def compute_query_vecs(
    tokens: list[int],
    adapter: "BaseAdapter",
    config: "KVMemoryConfig",
) -> dict[int, np.ndarray]:
    """
    Compute retrieval vectors for the current query.

    Runs a partial forward pass to max(retrieval_layers).
    Returns one normalized vector per retrieval layer.

    Uses K vectors (attention keys) when retrieval_vec_source == "k_vectors"
    (default), otherwise falls back to hidden states. K vectors are projected
    into the attention subspace via W_K, giving more discriminative signal
    for short texts than raw hidden states.
    """
    query_source = getattr(
        config,
        "retrieval_query_source",
        getattr(config, "retrieval_vec_source", "k_vectors"),
    )
    store_source = getattr(config, "retrieval_vec_source", "k_vectors")
    if query_source == "q_vectors" and store_source != "k_vectors":
        logger.warning(
            "retrieval_query_source='q_vectors' requires retrieval_vec_source='k_vectors'; "
            "falling back to hidden-state query capture"
        )
    elif query_source == "q_vectors":
        capture_query_vecs = getattr(adapter, "capture_query_vecs", None)
        if capture_query_vecs is not None:
            try:
                return capture_query_vecs(
                    tokens=tokens,
                    layers=config.retrieval_layers,
                    rope_mode=getattr(config, "retrieval_rope_mode", "native"),
                )
            except Exception as exc:
                logger.warning(
                    "Q-vector query capture failed (%s); falling back to K-vector query capture",
                    exc,
                )
        else:
            logger.warning(
                "Adapter %s does not expose capture_query_vecs(); falling back to K-vector query capture",
                type(adapter).__name__,
            )

    kv_by_layer, hidden_by_layer = adapter.capture(
        tokens=tokens,
        text="",
        layers=config.retrieval_layers,
    )

    if getattr(config, "retrieval_vec_source", "k_vectors") == "k_vectors":
        return {
            layer: compute_k_vec(K)
            for layer, (K, _) in kv_by_layer.items()
            if layer in config.retrieval_layers
        }
    else:
        return {
            layer: compute_retrieval_vec(hidden, len(tokens))
            for layer, hidden in hidden_by_layer.items()
        }


# ------------------------------------------------------------------
# Stage 1: Coarse ANN filter
# ------------------------------------------------------------------

async def stage1_coarse(
    query_tokens: list[int],
    adapter: "BaseAdapter",
    config: "KVMemoryConfig",
    vector_db: "VectorDB",
    session_filter: Optional[dict] = None,
) -> tuple[list[str], dict[int, np.ndarray]]:
    """
    Stage 1: Multi-layer ANN search to produce a candidate pool.

    Returns:
        (candidate_block_ids, query_vecs)
        query_vecs is passed to Stage 2 to avoid recomputing.
    """
    query_vecs = compute_query_vecs(query_tokens, adapter, config)

    layer_weights = _default_layer_weights(config.retrieval_layers)
    candidate_ids = vector_db.search_coarse(
        model_id=config.model_id,
        query_vecs=query_vecs,
        retrieval_layers=config.retrieval_layers,
        top_k=config.coarse_top_k,
        session_filter=session_filter,
        layer_weights=layer_weights,
    )

    logger.debug("Stage 1 found %d candidates", len(candidate_ids))
    return candidate_ids, query_vecs


# ------------------------------------------------------------------
# Stage 2: MMR rerank
# ------------------------------------------------------------------

def stage2_rerank_mmr(
    candidate_ids: list[str],
    query_vecs: dict[int, np.ndarray],
    config: "KVMemoryConfig",
    vector_db: "VectorDB",
    token_budget: int,
    mmr_lambda: float = 0.7,
    min_relevance: float = 0.0,
) -> list[str]:
    """
    Stage 2: MMR rerank with token budget enforcement.

    MMR formula:
        score = lambda * relevance(c) - (1 - lambda) * max_redundancy(c, selected)

    lambda=0.7: biased toward relevance, but actively penalizes duplicates.

    Args:
        candidate_ids: block_ids from Stage 1
        query_vecs:    normalized query vectors per layer
        config:        KVMemoryConfig
        vector_db:     VectorDB for fetching candidate vectors + payloads
        token_budget:  hard cap on total injected tokens
        mmr_lambda:    relevance vs diversity trade-off [0,1]

    Returns:
        Ordered list of selected block_ids (at most config.final_top_k)
    """
    if not candidate_ids:
        return []

    # Fetch lightweight payloads + vectors (no KV tensors yet)
    candidates = vector_db.fetch_with_vectors(config.model_id, candidate_ids)
    if not candidates:
        return []

    layer_weights = _default_layer_weights(config.retrieval_layers)
    middle_layer = config.retrieval_layers[len(config.retrieval_layers) // 2]

    def relevance(cand: dict) -> float:
        return _candidate_relevance(cand, query_vecs, layer_weights)

    selected: list[str] = []
    selected_vecs: list[np.ndarray] = []
    tokens_used = 0
    remaining = list(candidates)

    while remaining and len(selected) < config.final_top_k:
        best = None
        best_score = float("-inf")

        for cand in remaining:
            rel = relevance(cand)
            # Redundancy: max cosine similarity to already-selected blocks
            if selected_vecs:
                mid_key = f"layer_{middle_layer}"
                cand_vec = np.array(
                    (cand["vector"] or {}).get(mid_key, np.zeros(1)),
                    dtype=np.float32,
                )
                red = max(float(np.dot(cand_vec, sv)) for sv in selected_vecs)
            else:
                red = 0.0

            score = mmr_lambda * rel - (1 - mmr_lambda) * red

            if score > best_score:
                best_score = score
                best = cand

        if best is None:
            break

        # Drop blocks below minimum relevance — prevents injecting irrelevant context
        if min_relevance > 0.0 and relevance(best) < min_relevance:
            logger.debug(
                "MMR: stopping — best remaining block relevance %.3f < min_relevance %.3f",
                relevance(best), min_relevance,
            )
            break

        n_tokens = (best["payload"] or {}).get("token_count", 0)

        # Hard token budget enforcement (non-negotiable per spec)
        if tokens_used + n_tokens > token_budget:
            # Skip this block but keep looking — a smaller block might fit
            logger.debug(
                "MMR: block %s skipped — %d tokens > %d remaining budget",
                best["id"], n_tokens, token_budget - tokens_used,
            )
            remaining.remove(best)
            continue

        selected.append(best["id"])
        mid_key = f"layer_{middle_layer}"
        best_vec = np.array(
            (best["vector"] or {}).get(mid_key, np.zeros(1)),
            dtype=np.float32,
        )
        selected_vecs.append(best_vec)
        tokens_used += n_tokens
        remaining.remove(best)

    logger.debug(
        "Stage 2 selected %d blocks, %d tokens used (budget: %d)",
        len(selected),
        tokens_used,
        token_budget,
    )
    return selected


def stage2_rerank_qk(
    candidate_ids: list[str],
    query_vecs: dict[int, np.ndarray],
    config: "KVMemoryConfig",
    vector_db: "VectorDB",
    token_budget: int,
    min_relevance: float = 0.0,
) -> list[str]:
    """
    Stage 2: pure query-to-key relevance rank with hard token-budget enforcement.

    If retrieval_query_source == "q_vectors", relevance is native query-Q versus
    memory-K. Otherwise this degenerates to the older query-K versus memory-K
    score while still skipping the MMR diversity penalty.
    """
    if not candidate_ids:
        return []

    candidates = vector_db.fetch_with_vectors(config.model_id, candidate_ids)
    if not candidates:
        return []

    layer_weights = _default_layer_weights(config.retrieval_layers)
    scored: list[tuple[float, dict]] = []
    for cand in candidates:
        rel = _candidate_relevance(cand, query_vecs, layer_weights)
        if min_relevance > 0.0 and rel < min_relevance:
            continue
        scored.append((rel, cand))

    scored.sort(key=lambda item: item[0], reverse=True)

    selected: list[str] = []
    tokens_used = 0
    for _rel, cand in scored:
        if len(selected) >= config.final_top_k:
            break

        n_tokens = (cand["payload"] or {}).get("token_count", 0)
        if tokens_used + n_tokens > token_budget:
            logger.debug(
                "QK: block %s skipped -- %d tokens > %d remaining budget",
                cand["id"], n_tokens, token_budget - tokens_used,
            )
            continue

        selected.append(cand["id"])
        tokens_used += n_tokens

    logger.debug(
        "Stage 2 QK selected %d blocks, %d tokens used (budget: %d)",
        len(selected),
        tokens_used,
        token_budget,
    )
    return selected


def stage2_rerank(
    candidate_ids: list[str],
    query_vecs: dict[int, np.ndarray],
    config: "KVMemoryConfig",
    vector_db: "VectorDB",
    token_budget: int,
    mmr_lambda: float = 0.7,
    min_relevance: float = 0.0,
) -> list[str]:
    """Dispatch Stage 2 selection according to config.stage2_reranker."""
    reranker = getattr(config, "stage2_reranker", "mmr")
    if reranker == "qk":
        return stage2_rerank_qk(
            candidate_ids=candidate_ids,
            query_vecs=query_vecs,
            config=config,
            vector_db=vector_db,
            token_budget=token_budget,
            min_relevance=min_relevance,
        )
    if reranker != "mmr":
        raise ValueError(f"Unknown stage2_reranker={reranker!r}; expected 'mmr' or 'qk'")
    return stage2_rerank_mmr(
        candidate_ids=candidate_ids,
        query_vecs=query_vecs,
        config=config,
        vector_db=vector_db,
        token_budget=token_budget,
        mmr_lambda=mmr_lambda,
        min_relevance=min_relevance,
    )


def build_candidate_diagnostics(
    candidate_ids: list[str],
    selected_ids: list[str],
    query_vecs: dict[int, np.ndarray],
    config: "KVMemoryConfig",
    vector_db: "VectorDB",
    question: str = "",
    gold_answer: str = "",
    top_n: int = 20,
) -> dict:
    """
    Build JSON-serializable retrieval diagnostics without fetching KV blobs.

    The output explains whether the answer-looking chunk was absent from Stage 1,
    present but ranked below selected chunks, or selected but still failed during
    injection/generation.
    """
    if not candidate_ids:
        return {
            "candidate_count": 0,
            "selected_count": len(selected_ids),
            "query_source": getattr(config, "retrieval_query_source", "k_vectors"),
            "stage2_reranker": getattr(config, "stage2_reranker", "mmr"),
            "top_candidates": [],
            "selected_ids": selected_ids,
            "gold_in_stage1": False,
            "gold_in_selected": False,
            "best_gold_rerank_rank": None,
        }

    candidates = vector_db.fetch_with_vectors(config.model_id, candidate_ids)
    layer_weights = _default_layer_weights(config.retrieval_layers)
    stage1_rank = {block_id: idx + 1 for idx, block_id in enumerate(candidate_ids)}
    selected_order = {block_id: idx + 1 for idx, block_id in enumerate(selected_ids)}

    question_terms = set(_diagnostic_terms(question))
    gold_terms = set(_diagnostic_terms(gold_answer))
    gold_phrase = " ".join(_diagnostic_terms(gold_answer))

    rows: list[dict] = []
    for cand in candidates:
        block_id = cand["id"]
        payload = cand["payload"] or {}
        chunk_text = str(payload.get("chunk_text", ""))
        chunk_terms = set(_diagnostic_terms(chunk_text))
        rel = _candidate_relevance(cand, query_vecs, layer_weights)
        gold_overlap_terms = sorted(gold_terms & chunk_terms)
        question_overlap_terms = sorted(question_terms & chunk_terms)
        chunk_phrase = " ".join(_diagnostic_terms(chunk_text))
        gold_substring = bool(gold_phrase and gold_phrase in chunk_phrase)

        rows.append({
            "block_id": block_id,
            "stage1_rank": stage1_rank.get(block_id),
            "selected": block_id in selected_order,
            "selected_order": selected_order.get(block_id),
            "relevance": round(rel, 6),
            "token_count": int(payload.get("token_count", 0) or 0),
            "importance_score": float(payload.get("importance_score", 0.0) or 0.0),
            "question_overlap": round(
                len(question_overlap_terms) / max(len(question_terms), 1),
                6,
            ),
            "gold_overlap": round(len(gold_overlap_terms) / max(len(gold_terms), 1), 6),
            "gold_substring": gold_substring,
            "question_overlap_terms": question_overlap_terms[:20],
            "gold_overlap_terms": gold_overlap_terms[:20],
            "chunk_preview": chunk_text[:500],
        })

    rows.sort(key=lambda row: row["relevance"], reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rerank_rank"] = idx

    diagnostic_rows = rows[:top_n]
    keep_ids = {row["block_id"] for row in diagnostic_rows}
    for row in rows[top_n:]:
        should_keep = row["selected"] or row["gold_overlap"] > 0 or row["gold_substring"]
        if should_keep and row["block_id"] not in keep_ids:
            diagnostic_rows.append(row)
            keep_ids.add(row["block_id"])

    gold_rows = [
        row for row in rows
        if row["gold_overlap"] > 0 or row["gold_substring"]
    ]
    selected_gold_rows = [row for row in gold_rows if row["selected"]]

    return {
        "candidate_count": len(candidate_ids),
        "selected_count": len(selected_ids),
        "query_source": getattr(config, "retrieval_query_source", "k_vectors"),
        "retrieval_vec_source": getattr(config, "retrieval_vec_source", "k_vectors"),
        "stage2_reranker": getattr(config, "stage2_reranker", "mmr"),
        "retrieval_rope_mode": getattr(config, "retrieval_rope_mode", "native"),
        "top_candidates": diagnostic_rows,
        "selected_ids": selected_ids,
        "gold_in_stage1": bool(gold_rows),
        "gold_in_selected": bool(selected_gold_rows),
        "best_gold_rerank_rank": min(
            (row["rerank_rank"] for row in gold_rows),
            default=None,
        ),
        "best_gold_stage1_rank": min(
            (row["stage1_rank"] for row in gold_rows if row["stage1_rank"] is not None),
            default=None,
        ),
    }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _candidate_relevance(
    cand: dict,
    query_vecs: dict[int, np.ndarray],
    layer_weights: dict[int, float],
) -> float:
    score = 0.0
    vectors = cand["vector"] or {}
    for layer, w in layer_weights.items():
        key = f"layer_{layer}"
        if key not in vectors or layer not in query_vecs:
            continue
        q = query_vecs[layer]
        k = np.array(vectors[key], dtype=np.float32)
        score += w * float(np.dot(q, k))
    return score


def _default_layer_weights(retrieval_layers: list[int]) -> dict[int, float]:
    """25% shallow, 50% middle, 25% deep for 3-layer setup. Equal split otherwise."""
    n = len(retrieval_layers)
    if n == 3:
        return dict(zip(retrieval_layers, [0.25, 0.50, 0.25]))
    return {layer: 1.0 / n for layer in retrieval_layers}


_DIAGNOSTIC_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do",
    "for", "from", "how", "i", "in", "is", "it", "me", "my", "of", "on",
    "or", "over", "such", "that", "the", "this", "to", "was", "were",
    "what", "when", "where", "which", "who", "with", "you", "your",
}


def _diagnostic_terms(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _DIAGNOSTIC_STOPWORDS and len(token) > 1
    ]
