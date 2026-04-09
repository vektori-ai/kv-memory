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
    # Target 25%, 50%, 75% of model depth.
    # e.g. [8, 16, 24] for a 32-layer model.
    retrieval_layers: list[int] = field(default_factory=lambda: [8, 16, 24])
    store_layers: list[int] = field(default_factory=list)   # defaults to retrieval_layers

    # Retrieval budget
    token_budget: int = 2000        # HARD CAP on injected tokens across all retrieved blocks
    coarse_top_k: int = 200         # stage 1: ANN candidate pool
    final_top_k: int = 10           # stage 2: MMR winners

    # Quality gates
    importance_threshold: float = 0.3   # chunks below this score are never stored
    dedup_threshold: float = 0.95       # skip write if similarity > this to existing block

    # Runtime
    async_write: bool = True

    # Infrastructure
    qdrant_url: str = "localhost"
    qdrant_port: int = 6333
    blob_store_path: str = "./kv_store"

    def __post_init__(self) -> None:
        # store_layers defaults to retrieval_layers if not explicitly set
        if not self.store_layers:
            self.store_layers = list(self.retrieval_layers)
