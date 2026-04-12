"""
config.py — KVMemoryConfig dataclass.

All runtime knobs live here. Passed to every subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KVMemoryConfig:
    # Required
    model_id: str

    # Which layers to extract hidden states + KV tensors from.
    # Empty list = auto-compute as [25%, 50%, 75%] of model depth at init time.
    # Override only if you want specific layers: e.g. [8, 16, 24] for a 32-layer model.
    retrieval_layers: list[int] = field(default_factory=list)
    store_layers: list[int] = field(default_factory=list)   # empty means auto: all model layers

    # Retrieval budget
    token_budget: int = 2000        # HARD CAP on injected tokens across all retrieved blocks
    coarse_top_k: int = 200         # stage 1: ANN candidate pool
    final_top_k: int = 10           # stage 2: MMR winners

    # Quality gates
    importance_threshold: float = 0.3   # chunks below this score are never stored
    dedup_threshold: float = 0.95       # skip write if similarity > this to existing block
    min_relevance: float = 0.0          # stage2: don't inject blocks below this cosine sim

    # Capture batch size — lower = less GPU memory pressure per write call
    capture_batch_size: int = 8

    # Runtime
    async_write: bool = True

    # Infrastructure
    qdrant_url: str = "localhost"
    qdrant_port: int = 6333
    blob_store_path: str = "./kv_store"
