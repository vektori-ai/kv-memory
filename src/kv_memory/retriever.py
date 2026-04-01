from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .store import MemoryStore
from .types import MemoryUnit


@dataclass(slots=True)
class ScoredMemoryUnit:
    unit: MemoryUnit
    score: float


class Retriever:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def top_k(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        k: int,
        *,
        min_score: float = -1e9,
    ) -> list[MemoryUnit]:
        scored = self.store.score(query_text, query_embedding)
        filtered = [(unit, score) for unit, score in scored if score >= min_score]
        return [unit for unit, _ in filtered[:k]]

    def top_k_with_scores(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        k: int,
        *,
        min_score: float = -1e9,
    ) -> list[ScoredMemoryUnit]:
        scored = self.store.score(query_text, query_embedding)
        filtered = [(unit, score) for unit, score in scored if score >= min_score]
        return [ScoredMemoryUnit(unit=unit, score=score) for unit, score in filtered[:k]]
