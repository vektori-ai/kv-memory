"""
retrieval.py — Two-stage retrieval: coarse ANN + MMR rerank.

Stage 1 (coarse): Multi-layer ANN search in Qdrant. Returns top-200 candidate block_ids.
Stage 2 (MMR):    Maximal Marginal Relevance rerank over candidates. Returns top-10 winners.

Both stages enforce the token budget hard cap.
"""

from __future__ import annotations

import logging
import math
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


def compute_query_vecs(
    tokens: list[int],
    adapter: "BaseAdapter",
    config: "KVMemoryConfig",
) -> dict[int, np.ndarray]:
    """
    Compute retrieval vectors for the current query.

    Runs a partial forward pass to max(retrieval_layers).
    Returns one normalized vector per retrieval layer.
    """
    _, hidden_by_layer = adapter.capture(
        tokens=tokens,
        text="",
        layers=config.retrieval_layers,
    )
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
        score = 0.0
        for layer, w in layer_weights.items():
            key = f"layer_{layer}"
            if key not in (cand["vector"] or {}):
                continue
            q = query_vecs[layer]
            k = np.array(cand["vector"][key], dtype=np.float32)
            score += w * float(np.dot(q, k))
        return score

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


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _default_layer_weights(retrieval_layers: list[int]) -> dict[int, float]:
    """25% shallow, 50% middle, 25% deep for 3-layer setup. Equal split otherwise."""
    n = len(retrieval_layers)
    if n == 3:
        return dict(zip(retrieval_layers, [0.25, 0.50, 0.25]))
    return {layer: 1.0 / n for layer in retrieval_layers}
