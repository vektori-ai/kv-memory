"""
schema.py — Core data structures for the KV Memory System.

All other modules depend on these. Defined first per build order.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class KVBlock:
    """
    A single stored memory unit: KV tensors from one semantic chunk.

    Retrieval keys (hidden_vecs) live in the vector DB.
    KV tensors (kv_by_layer) live in the blob store.
    Both are indexed by block_id.
    """

    # Identity
    block_id: str                                           # uuid4
    model_id: str                                           # e.g. 'llama-3-8b'
    session_id: str
    agent_id: Optional[str]
    shared: bool                                            # cross-agent retrieval allowed

    # Retrieval keys — stored in vector DB
    # layer -> [d_model] float32 (weighted mean pool, sqrt(n) norm, L2 norm)
    hidden_vecs: dict[int, np.ndarray]
    token_count: int
    chunk_text: str                                         # original text, debug + recompute

    # KV tensors — stored in blob store
    # layer -> (K, V), each shape [heads, seq, head_dim] int8
    kv_by_layer: dict[int, tuple[np.ndarray, np.ndarray]]
    # layer -> (K_scale, V_scale) for dequantization
    quant_scales: dict[int, tuple[float, float]]
    original_positions: list[int]                           # for doc-wise RoPE

    # Lifecycle
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    importance_score: float = 0.0                           # 0.0 to 1.0

    @classmethod
    def new(
        cls,
        model_id: str,
        session_id: str,
        chunk_text: str,
        token_count: int,
        hidden_vecs: dict[int, np.ndarray],
        kv_by_layer: dict[int, tuple[np.ndarray, np.ndarray]],
        quant_scales: dict[int, tuple[float, float]],
        original_positions: list[int],
        agent_id: Optional[str] = None,
        shared: bool = False,
        importance_score: float = 0.0,
    ) -> "KVBlock":
        return cls(
            block_id=str(uuid.uuid4()),
            model_id=model_id,
            session_id=session_id,
            agent_id=agent_id,
            shared=shared,
            hidden_vecs=hidden_vecs,
            token_count=token_count,
            chunk_text=chunk_text,
            kv_by_layer=kv_by_layer,
            quant_scales=quant_scales,
            original_positions=original_positions,
            importance_score=importance_score,
        )

    def touch(self) -> None:
        """Update access metadata on retrieval."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class GenerationOutput:
    """
    Unified output wrapper across adapters.
    sequences: list of token ID lists (batch).
    text: decoded string of sequences[0].
    """
    sequences: list[list[int]]
    text: str
    metadata: dict = field(default_factory=dict)
