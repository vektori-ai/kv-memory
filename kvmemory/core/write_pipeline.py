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

import hashlib
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
from ..core.retrieval import compute_retrieval_vec, compute_k_vec
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
# Per-session hash dedup state (keyed by (model_id, session_id))
# ------------------------------------------------------------------

_seen_hashes: dict[tuple[str, str], set[str]] = {}


def reset_hash_dedup(model_id: str, session_id: str) -> None:
    """Remove in-memory hash dedup state for a given (model_id, session_id) pair.

    Call this AFTER drain_writes() completes to free memory without
    disrupting in-flight queued writes.
    """
    _seen_hashes.pop((model_id, session_id), None)


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
    dedup_mode: str = "semantic",
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
    # Write summary counters (Fix 11)
    n_chunks_total = 0
    n_candidates = 0
    n_importance_skipped = 0
    n_hash_dup_skipped = 0
    n_semantic_dup_skipped = 0
    n_capture_failed = 0
    n_written = 0

    event_observer = None
    if observer:
        event_observer = observer.child(
            session_id=session_id,
            agent_id=agent_id,
            shared=shared,
            **(trace_context or {}),
        )

    # Step 1: Chunk the turn text
    chunks = chunk_turn(text, adapter.tokenizer, target_tokens=100, hard_max=400)
    n_chunks_total = len(chunks)
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
    phase = (trace_context or {}).get("phase", "write")

    # Steps 2–3: importance scoring (sequential — each chunk updates the baseline tracker)
    # Collect chunks that pass the importance gate, tagged with their index for logging.
    candidate_chunks: list[tuple[int, str, list[int], float]] = []  # (orig_index, text, tokens, importance)
    for chunk_index, chunk_text in enumerate(chunks, start=1):
        chunk_tokens = adapter.tokenizer.encode(chunk_text)
        chunk_observer = event_observer.child(chunk_index=chunk_index, total_chunks=len(chunks)) if event_observer else None
        if chunk_observer:
            chunk_observer.emit(
                "write_chunk_created",
                phase=phase,
                chunk_preview=chunk_text[:160],
                token_count=len(chunk_tokens),
            )
        if len(chunk_tokens) < 2:
            if chunk_observer:
                chunk_observer.emit(
                    "write_chunk_skipped",
                    phase=phase,
                    reason="too_short",
                    token_count=len(chunk_tokens),
                )
            continue

        # Step 2: Compute chunk loss (perplexity proxy)
        try:
            model_loss = compute_chunk_loss(chunk_tokens, adapter.model)
        except Exception as e:
            logger.warning("Failed to compute chunk loss: %s", e)
            model_loss = baseline_tracker.value
            if chunk_observer:
                chunk_observer.emit(
                    "error",
                    level="error",
                    phase=phase,
                    message="Failed to compute chunk loss",
                    error=str(e),
                    error_type=type(e).__name__,
                )

        baseline_loss = baseline_tracker.value
        baseline_tracker.update(model_loss)

        # Step 3: Importance filter
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
            n_importance_skipped += 1
            if chunk_observer:
                chunk_observer.emit(
                    "write_chunk_skipped",
                    phase=phase,
                    reason="importance_threshold",
                    importance_score=importance,
                    importance_threshold=config.importance_threshold,
                )
            continue

        n_candidates += 1
        candidate_chunks.append((chunk_index, chunk_text, chunk_tokens, importance))

    # Steps 4–9 (streaming): each batch completes the full pipeline before the next
    # batch is captured. This caps GPU memory usage to one batch at a time.
    CAPTURE_BATCH_SIZE = config.capture_batch_size

    from kvmemory.core.retrieval import _default_layer_weights as _layer_wts
    _semantic_layer_weights = _layer_wts(config.retrieval_layers)

    for batch_start in range(0, len(candidate_chunks), CAPTURE_BATCH_SIZE):
        batch = candidate_chunks[batch_start: batch_start + CAPTURE_BATCH_SIZE]
        batch_token_lists = [tok for _, _, tok, _ in batch]
        max_token_len = max(len(t) for t in batch_token_lists) if batch_token_lists else 0

        # --- Step 4: Capture batch ---
        batch_obs = event_observer.child(
            batch_start=batch_start,
            batch_size=len(batch),
        ) if event_observer else None
        if batch_obs:
            batch_obs.emit(
                "capture_batch_started",
                phase=phase,
                batch_size=len(batch),
                batch_start=batch_start,
                max_token_len=max_token_len,
                layer_count=len(config.store_layers),
            )

        capture_results: dict[int, tuple[dict, dict]] = {}
        batch_t0 = __import__("time").perf_counter()
        if hasattr(adapter, "capture_batch"):
            try:
                batch_results = adapter.capture_batch(batch_token_lists, config.store_layers)
                for (orig_index, _, _, _), result in zip(batch, batch_results):
                    capture_results[orig_index] = result
            except Exception as e:
                logger.error("adapter.capture_batch failed for batch starting at %d: %s", batch_start, e)
                if batch_obs:
                    batch_obs.emit(
                        "capture_batch_failed",
                        phase=phase,
                        error=str(e),
                        error_type=type(e).__name__,
                        batch_start=batch_start,
                        batch_size=len(batch),
                    )
                # Fall back to individual capture() for this batch
                for orig_index, chunk_text_fb, chunk_tokens_fb, _ in batch:
                    try:
                        result_fb = adapter.capture(chunk_tokens_fb, chunk_text_fb, config.store_layers)
                        capture_results[orig_index] = result_fb
                    except Exception as e2:
                        logger.error("adapter.capture fallback also failed for chunk %d: %s", orig_index, e2)
                        if batch_obs:
                            batch_obs.emit(
                                "capture_fallback_failed",
                                phase=phase,
                                chunk_index=orig_index,
                                error=str(e2),
                                error_type=type(e2).__name__,
                            )
        else:
            # Adapter doesn't support batching — fall back to one-at-a-time
            for orig_index, chunk_text_fb, chunk_tokens_fb, _ in batch:
                try:
                    result_fb = adapter.capture(chunk_tokens_fb, chunk_text_fb, config.store_layers)
                    capture_results[orig_index] = result_fb
                except Exception as e:
                    logger.error("adapter.capture failed for chunk %d: %s", orig_index, e)

        if batch_obs:
            batch_dur_ms = (__import__("time").perf_counter() - batch_t0) * 1000
            batch_obs.emit(
                "capture_batch_done",
                phase=phase,
                batch_size=len(batch),
                duration_ms=batch_dur_ms,
            )

        # --- Steps 5–9: dedup, quantize, write (per chunk in this batch) ---
        for orig_index, chunk_text, chunk_tokens, importance in batch:
            chunk_observer = event_observer.child(chunk_index=orig_index, total_chunks=len(chunks)) if event_observer else None

            if orig_index not in capture_results:
                n_capture_failed += 1
                if chunk_observer:
                    chunk_observer.emit(
                        "write_chunk_skipped",
                        phase=phase,
                        reason="capture_failed",
                        chunk_index=orig_index,
                    )
                continue

            kv_by_layer_float, hidden_by_layer = capture_results[orig_index]
            # Free the capture result immediately to release GPU memory
            del capture_results[orig_index]

            # Step 5: Compute normalized retrieval vectors
            # Option B: use attention K vectors (more discriminative than hidden states
            # for short chunks — W_K projects into content-specific attention subspace).
            # Falls back to hidden states if retrieval_vec_source != "k_vectors".
            hidden_vecs: dict[int, np.ndarray] = {}
            use_k_vecs = getattr(config, "retrieval_vec_source", "k_vectors") == "k_vectors"
            for layer in config.retrieval_layers:
                if use_k_vecs and layer in kv_by_layer_float:
                    K, _ = kv_by_layer_float[layer]
                    hidden_vecs[layer] = compute_k_vec(K)
                elif layer in hidden_by_layer:
                    hidden_vecs[layer] = compute_retrieval_vec(
                        hidden_by_layer[layer],
                        len(chunk_tokens),
                    )

            # Step 6: Dedup check
            skip_as_dup = False
            dup_reason = "duplicate"
            dup_block_id: Optional[str] = None

            if dedup_mode == "hash":
                h = hashlib.sha256(chunk_text.encode()).hexdigest()
                key = (config.model_id, session_id)
                if key not in _seen_hashes:
                    _seen_hashes[key] = set()
                if h in _seen_hashes[key]:
                    skip_as_dup = True
                    dup_reason = "hash_duplicate"
                else:
                    _seen_hashes[key].add(h)
            else:
                # "semantic" mode — multi-layer candidate-union dedup
                existing_id = vector_db.find_duplicate_multilayer(
                    model_id=config.model_id,
                    hidden_vecs=hidden_vecs,
                    layer_weights=_semantic_layer_weights,
                    threshold=config.dedup_threshold,
                    session_id=session_id,
                )
                if existing_id:
                    skip_as_dup = True
                    dup_block_id = existing_id

            if skip_as_dup:
                logger.debug(
                    "Dedup (%s): skipping chunk (dup_block=%s)",
                    dup_reason,
                    dup_block_id,
                )
                if dup_reason == "hash_duplicate":
                    n_hash_dup_skipped += 1
                else:
                    n_semantic_dup_skipped += 1
                if chunk_observer:
                    chunk_observer.emit(
                        "write_chunk_skipped",
                        phase=phase,
                        reason=dup_reason,
                        duplicate_block_id=dup_block_id,
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
                        phase=phase,
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
                if chunk_observer:
                    chunk_observer.emit(
                        "error",
                        level="error",
                        phase=phase,
                        message="vector_db.upsert failed",
                        block_id=block.block_id,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                continue

            written_ids.append(block.block_id)
            n_written += 1
            logger.debug(
                "Wrote block %s (importance=%.3f, tokens=%d)",
                block.block_id,
                importance,
                len(chunk_tokens),
            )
            if chunk_observer:
                chunk_observer.emit(
                    "write_chunk_stored",
                    phase=phase,
                    block_id=block.block_id,
                    importance_score=importance,
                    token_count=len(chunk_tokens),
                    retrieval_layers=sorted(hidden_vecs),
                )

    logger.info(
        "write_pipeline: session=%s, chunks=%d, candidates=%d, written=%d "
        "(importance_skipped=%d, hash_dup=%d, semantic_dup=%d, capture_failed=%d)",
        session_id, n_chunks_total, n_candidates, n_written,
        n_importance_skipped, n_hash_dup_skipped, n_semantic_dup_skipped, n_capture_failed,
    )
    if event_observer:
        event_observer.emit(
            "phase_finished",
            phase=(trace_context or {}).get("phase", "write"),
            chunk_count=n_chunks_total,
            written_count=n_written,
            n_chunks_total=n_chunks_total,
            n_candidates=n_candidates,
            n_importance_skipped=n_importance_skipped,
            n_hash_dup_skipped=n_hash_dup_skipped,
            n_semantic_dup_skipped=n_semantic_dup_skipped,
            n_capture_failed=n_capture_failed,
            n_written=n_written,
        )
    return written_ids
