"""
importance.py — Perplexity-based importance scorer.

Gate 1: chunks below the importance threshold are never stored.
Reduces write volume by ~60-70%.

Core insight: high perplexity (loss) = model found this surprising = worth storing.
Low perplexity = model already knew it = low marginal value to store.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def score_importance(
    chunk_text: str,
    model_loss: float,
    baseline_loss: float,
    explicit_signal: float = 0.0,
) -> float:
    """
    Compute importance score for a chunk.

    Args:
        chunk_text:      the chunk text (unused currently, reserved for future signals)
        model_loss:      cross-entropy loss on this chunk's tokens
        baseline_loss:   running average loss across the session
        explicit_signal: 1.0 if user explicitly flagged "remember this", else 0.0

    Returns:
        importance score in [0.0, 1.0]
        Higher = more worth storing.

    Formula:
        perplexity_score = max(0, loss - baseline) / baseline
        importance = 0.6 * perplexity_score + 0.4 * explicit_signal
    """
    if baseline_loss <= 0:
        # Avoid division by zero; treat as neutral
        perplexity_score = 0.0
    else:
        perplexity_score = max(0.0, model_loss - baseline_loss) / baseline_loss

    # Clamp perplexity_score to [0, 1] — extreme loss spikes shouldn't dominate
    perplexity_score = min(1.0, perplexity_score)

    score = 0.6 * perplexity_score + 0.4 * explicit_signal
    return float(min(1.0, max(0.0, score)))


def compute_chunk_loss(
    chunk_tokens: list[int],
    model,
    device: Optional[str] = None,
) -> float:
    """
    Compute cross-entropy loss for a chunk using the language model.

    This is the perplexity proxy: high loss = model found this surprising.

    Args:
        chunk_tokens: token IDs for the chunk
        model:        HuggingFace causal LM (or any model with loss output)
        device:       device string; defaults to model.device

    Returns:
        mean cross-entropy loss (float)
    """
    if len(chunk_tokens) < 2:
        # Need at least 2 tokens for a next-token prediction
        return 0.0

    dev = device or str(next(model.parameters()).device)
    input_ids = torch.tensor([chunk_tokens], dtype=torch.long).to(dev)

    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)

    return float(out.loss.item())


class BaselineLossTracker:
    """
    Running exponential moving average of model loss across a session.

    Used as the denominator in the perplexity_score formula.
    Initialized with a reasonable default (2.0) so the first few chunks
    don't get artificially inflated scores.
    """

    def __init__(self, initial: float = 2.0, alpha: float = 0.1) -> None:
        self._ema = initial
        self._alpha = alpha  # EMA smoothing factor

    @property
    def value(self) -> float:
        return self._ema

    def update(self, loss: float) -> None:
        """Update EMA with a new loss observation."""
        self._ema = (1 - self._alpha) * self._ema + self._alpha * loss

    def reset(self) -> None:
        """Reset to default (useful when switching sessions)."""
        self._ema = 2.0
