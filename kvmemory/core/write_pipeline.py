"""
write_pipeline.py — Full write path executed by the async queue.

Pipeline per turn:
  1. chunk_turn()          — split text into semantic chunks
  2. compute_chunk_loss()  — perplexity proxy via model forward pass
  3. score_importance()    — importance gate (drop below threshold)
  4. adapter.capture()     — extract KV tensors + hidden states
  5. compute_retrieval_vec() — normalize hidden states per layer
  6. is_duplicate()        — dedup check against vector DB
  7. quantize_int8()       — INT8 compress KV tensors
  8. kv_store.write()      — persist KVBlock to disk
  9. vector_db.upsert()    — persist retrieval vectors to Qdrant

The entire pipeline is async and fire-and-forget from the user's POV.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..adapters.base import BaseAdapter
from ..config import KVMemoryConfig
from ..core.chunker import chunk_turn
from ..core.importance import (
    BaselineLossTracker,
    compute_chunk_loss,
    score_importance,
)
from ..core.retrieval import compute_retrieval_vec
from ..observability import RunObserver
from ..storage.kv_store import KVStore
from ..storage.schema import KVBlock
from ..storage.vector_db import VectorDB

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# INT8 quantization
# ------------------------------------------------------------------

def quantize_int8(tensor: torch.Tensor) -> tuple[np.ndarray, float]:
    """
    Quantize a float16/float32 tensor to INT8.

    Scale: abs_max / 127.0
    2x size reduction vs float16. Minimal quality loss at this scale.

    Returns:
        (quantized_array: int8 ndarray, scale: float)
    """
    abs_max = tensor.abs().max().item()
    if abs_max == 0:
        return tensor.cpu().numpy().astype(np.int8), 1.0

    scale = abs_max / 127.0
    quantized = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized.cpu().numpy(), scale


def dequantize_int8(quantized: np.ndarray, scale: float) -> torch.Tensor:
    """Reverse INT8 quantization. Returns float32 tensor."""
    return torch.from_numpy(quantized.astype(np.float32)) * scale


# ------------------------------------------------------------------
# Per-session baseline loss trackers
# ------------------------------------------------------------------

_baseline_trackers: dict[str, BaselineLossTracker] = {}


def _get_baseline_tracker(session_id: str) -> BaselineLossTracker:
    if session_id not in _baseline_trackers:
        _baseline_trackers[session_id] = BaselineLossTracker()
    return _baseline_trackers[session_id]


# ------------------------------------------------------------------
# Main write pipeline
# ------------------------------------------------------------------

async def run_write_pipeline(
    session_id: str,
    tokens: list[int],
    text: str,
    adapter: BaseAdapter,
    config: KVMemoryConfig,
    kv_store: KVStore,
    vector_db: VectorDB,
    agent_id: Optional[str] = None,
    shared: bool = False,
    explicit_signal: float = 0.0,
    observer: Optional[RunObserver] = None,
    trace_context: Optional[dict[str, Any]] = None,
) -> list[str]:
    """
    Full async write pipeline for one conversation turn.

    Args:
        session_id:      current session identifier
        tokens:          token IDs for full turn (prompt + response)
        text:            full turn text
        adapter:         framework adapter
        config:          KVMemoryConfig
        kv_store:        KVStore instance
        vector_db:       VectorDB instance
        agent_id:        optional agent identifier
        shared:          allow cross-agent retrieval
        explicit_signal: 1.0 if user flagged 'remember this'

    Returns:
        list of block_ids that were successfully written
    """
    written_ids: list[str] = []
    event_observer = None
    if observer:
        event_observer = observer.child(
            session_id=session_id,
            agent_id=agent_id,
            shared=shared,
            **(trace_context or {}),
        )

    # Step 1: Chunk the turn text
    chunks = chunk_turn(text, adapter.tokenizer, target_tokens=100)
    if not chunks:
        logger.debug("write_pipeline: no chunks from session=%s", session_id)
        if event_observer:
            event_observer.emit(
                "write_chunk_skipped",
                phase=(trace_context or {}).get("phase", "write"),
                reason="no_chunks",
            )
        return written_ids

    baseline_tracker = _get_baseline_tracker(session_id)

    for chunk_index, chunk_text in enumerate(chunks, start=1):
        chunk_tokens = adapter.tokenizer.encode(chunk_text)
        chunk_observer = event_observer.child(chunk_index=chunk_index, total_chunks=len(chunks)) if event_observer else None
        if chunk_observer:
            chunk_observer.emit(
                "write_chunk_created",
                phase=(trace_context or {}).get("phase", "write"),
                chunk_preview=chunk_text[:160],
                token_count=len(chunk_tokens),
            )
        if len(chunk_tokens) < 2:
            if chunk_observer:
                chunk_observer.emit(
                    "write_chunk_skipped",
                    phase=(trace_context or {}).get("phase", "write"),
                    reason="too_short",
                    token_count=len(chunk_tokens),
                )
            continue

        # Step 2: Compute chunk loss (perplexity proxy)
        try:
            model_loss = compute_chunk_loss(chunk_tokens, adapter.model)
        except Exception as e:
            logger.warning("Failed to compute chunk loss: %s", e)
            model_loss = baseline_tracker.value  # neutral
            if chunk_observer:
                chunk_observer.emit(
                    "error",
                    level="error",
                    phase=(trace_context or {}).get("phase", "write"),
                    message="Failed to compute chunk loss",
                    error=str(e),
                    error_type=type(e).__name__,
                )

        baseline_loss = baseline_tracker.value
        baseline_tracker.update(model_loss)

        # Step 3: Importance filter (Gate 1)
        importance = score_importance(
            chunk_text=chunk_text,
            model_loss=model_loss,
            baseline_loss=baseline_loss,
            explicit_signal=explicit_signal,
        )
        if importance < config.importance_threshold:
            logger.debug(
                "Skipping chunk (importance=%.3f < threshold=%.3f): '%s...'",
                importance,
                config.importance_threshold,
                chunk_text[:60],
            )
            if chunk_observer:
                chunk_observer.emit(
                    "write_chunk_skipped",
                    phase=(trace_context or {}).get("phase", "write"),
                    reason="importance_threshold",
                    importance_score=importance,
                    importance_threshold=config.importance_threshold,
                )
            continue

        # Step 4: Capture KV tensors and hidden states
        try:
            kv_by_layer_float, hidden_by_layer = adapter.capture(
                tokens=chunk_tokens,
                text=chunk_text,
                layers=config.store_layers,
            )
        except Exception as e:
            logger.error("adapter.capture failed: %s", e)
            if chunk_observer:
                chunk_observer.emit(
                    "error",
                    level="error",
                    phase=(trace_context or {}).get("phase", "write"),
                    message="adapter.capture failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
            continue

        # Step 5: Compute normalized retrieval vectors
        hidden_vecs: dict[int, np.ndarray] = {}
        for layer, hidden in hidden_by_layer.items():
            hidden_vecs[layer] = compute_retrieval_vec(hidden, len(chunk_tokens))

        # Step 6: Dedup check
        # Use middle retrieval layer as representative vector
        middle_layer = config.retrieval_layers[len(config.retrieval_layers) // 2]
        if middle_layer in hidden_vecs:
            existing_id = vector_db.find_duplicate(
                model_id=config.model_id,
                hidden_vec=hidden_vecs[middle_layer],
                layer=middle_layer,
                threshold=config.dedup_threshold,
                session_id=session_id,
            )
            if existing_id:
                logger.debug(
                    "Dedup: skipping chunk, similar block %s already exists",
                    existing_id,
                )
                if chunk_observer:
                    chunk_observer.emit(
                        "write_chunk_skipped",
                        phase=(trace_context or {}).get("phase", "write"),
                        reason="duplicate",
                        duplicate_block_id=existing_id,
                        importance_score=importance,
                    )
                continue

        # Step 7: INT8 quantization of KV tensors
        kv_by_layer_quant: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        quant_scales: dict[int, tuple[float, float]] = {}

        for layer, (K, V) in kv_by_layer_float.items():
            K_q, k_scale = quantize_int8(K.float())
            V_q, v_scale = quantize_int8(V.float())
            kv_by_layer_quant[layer] = (K_q, V_q)
            quant_scales[layer] = (k_scale, v_scale)

        # Original positions: sequential [0, 1, ..., n-1]
        original_positions = list(range(len(chunk_tokens)))

        # Build KVBlock
        block = KVBlock.new(
            model_id=config.model_id,
            session_id=session_id,
            chunk_text=chunk_text,
            token_count=len(chunk_tokens),
            hidden_vecs=hidden_vecs,
            kv_by_layer=kv_by_layer_quant,
            quant_scales=quant_scales,
            original_positions=original_positions,
            agent_id=agent_id,
            shared=shared,
            importance_score=importance,
        )

        # Step 8: Write KV tensors to blob store
        try:
            kv_store.write(block)
        except Exception as e:
            logger.error("kv_store.write failed for %s: %s", block.block_id, e)
            if chunk_observer:
                chunk_observer.emit(
                    "error",
                    level="error",
                    phase=(trace_context or {}).get("phase", "write"),
                    message="kv_store.write failed",
                    block_id=block.block_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
            continue

        # Step 9: Write retrieval vectors to Qdrant
        try:
            trace_payload = {}
            if event_observer:
                trace_payload.update(event_observer.context_fields)
            trace_payload.update(trace_context or {})
            vector_db.upsert(
                model_id=config.model_id,
                block_id=block.block_id,
                hidden_vecs=hidden_vecs,
                session_id=session_id,
                agent_id=agent_id,
                shared=shared,
                token_count=len(chunk_tokens),
                chunk_text=chunk_text,
                importance_score=importance,
                trace_payload=trace_payload,
            )
        except Exception as e:
            logger.error("vector_db.upsert failed for %s: %s", block.block_id, e)
            # Blob store write succeeded but vector DB failed.
            # Block exists in blob but won't be retrievable. Log and continue.
            # TODO(Phase 4): implement reconciliation pass.
            if chunk_observer:
                chunk_observer.emit(
                    "error",
                    level="error",
                    phase=(trace_context or {}).get("phase", "write"),
                    message="vector_db.upsert failed",
                    block_id=block.block_id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
            continue

        written_ids.append(block.block_id)
        logger.debug(
            "Wrote block %s (importance=%.3f, tokens=%d)",
            block.block_id,
            importance,
            len(chunk_tokens),
        )
        if chunk_observer:
            chunk_observer.emit(
                "write_chunk_stored",
                phase=(trace_context or {}).get("phase", "write"),
                block_id=block.block_id,
                importance_score=importance,
                token_count=len(chunk_tokens),
                retrieval_layers=sorted(hidden_vecs),
            )

    logger.info(
        "write_pipeline: session=%s, chunks=%d, written=%d",
        session_id,
        len(chunks),
        len(written_ids),
    )
    if event_observer:
        event_observer.emit(
            "phase_finished",
            phase=(trace_context or {}).get("phase", "write"),
            chunk_count=len(chunks),
            written_count=len(written_ids),
        )
    return written_ids
