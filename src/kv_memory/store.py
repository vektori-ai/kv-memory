from __future__ import annotations

import math
import time

import numpy as np

from .config import MemoryConfig
from .types import MemoryUnit


class MemoryStore:
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.units: list[MemoryUnit] = []

    def clear(self) -> None:
        self.units.clear()

    def add_units(self, units: list[MemoryUnit]) -> None:
        self.units.extend(units)

    def score(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        now_ts: float | None = None,
    ) -> list[tuple[MemoryUnit, float]]:
        now = now_ts if now_ts is not None else time.time()
        scored: list[tuple[MemoryUnit, float]] = []
        query_tokens = set(query_text.lower().split())

        for unit in self.units:
            if unit.embedding is None:
                continue

            semantic = float(np.dot(query_embedding, unit.embedding))
            unit_tokens = set(unit.text.lower().split())
            lexical = 0.0
            if query_tokens and unit_tokens:
                lexical = len(query_tokens & unit_tokens) / len(query_tokens)
            age = max(0.0, now - unit.timestamp)
            recency = math.exp(-self.config.recency_lambda * age)
            importance = unit.importance

            final_score = (
                self.config.semantic_weight * semantic
                + self.config.lexical_weight * lexical
                + self.config.recency_weight * recency
                + self.config.importance_weight * importance
            )
            scored.append((unit, final_score))

        scored.sort(key=lambda value: value[1], reverse=True)
        return scored
