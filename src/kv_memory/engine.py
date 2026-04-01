from __future__ import annotations

import time

from .config import MemoryConfig
from .embeddings import Embedder, create_default_embedder
from .retriever import Retriever, ScoredMemoryUnit
from .store import MemoryStore
from .types import ConversationTurn, MemoryUnit
from .windowing import build_turn_windows


class ConversationMemoryEngine:
    def __init__(
        self,
        config: MemoryConfig | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.embedder = embedder or create_default_embedder(
            prefer_hf=self.config.prefer_hf_embeddings,
            embedding_dim=self.config.embedding_dim,
            model_name=self.config.embedding_model_name,
            device=self.config.embedding_device,
            allow_download=self.config.embedding_allow_download,
        )

        self.turns: list[ConversationTurn] = []
        self.store = MemoryStore(self.config)
        self.retriever = Retriever(self.store)

    def add_turn(
        self,
        role: str,
        content: str,
        *,
        importance: float = 1.0,
        timestamp: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=timestamp if timestamp is not None else time.time(),
            importance=importance,
            metadata=metadata or {},
        )
        self.turns.append(turn)

    def rebuild_memory(self) -> list[MemoryUnit]:
        units = build_turn_windows(
            self.turns,
            window_size=self.config.window_size,
            stride=self.config.stride,
        )

        for unit in units:
            unit.embedding = self.embedder.embed(unit.text)

        self.store.clear()
        self.store.add_units(units)
        return units

    def retrieve(self, query: str, top_k: int | None = None) -> list[MemoryUnit]:
        if not self.store.units:
            self.rebuild_memory()

        query_embedding = self.embedder.embed(query)
        k = top_k if top_k is not None else self.config.top_k
        return self.retriever.top_k(
            query,
            query_embedding,
            k,
            min_score=self.config.min_retrieval_score,
        )

    def retrieve_with_scores(self, query: str, top_k: int | None = None) -> list[ScoredMemoryUnit]:
        if not self.store.units:
            self.rebuild_memory()

        query_embedding = self.embedder.embed(query)
        k = top_k if top_k is not None else self.config.top_k
        return self.retriever.top_k_with_scores(
            query,
            query_embedding,
            k,
            min_score=self.config.min_retrieval_score,
        )
