from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .config import MemoryConfig
from .engine import ConversationMemoryEngine
from .hf_memory_runner import HFMemoryRunner
from .kv_bank import MemoryKVBank
from .retriever import ScoredMemoryUnit
from .types import MemoryUnit


def _digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class RetrievalDecision:
    query: str
    selected_unit_ids: list[str]
    selected_texts: list[str]
    selected_scores: list[float]


class QwenKVMemoryEngine:
    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        config: MemoryConfig | None = None,
        capture_layers: set[int] | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self.text_engine = ConversationMemoryEngine(config=self.config)
        self.runner = HFMemoryRunner(model=model, tokenizer=tokenizer)
        self.kv_bank = MemoryKVBank(max_entries=self.config.max_kv_entries)

        self.capture_layers = capture_layers
        self._unit_digest_index: dict[str, str] = {}

    def _cleanup_digest_index(self) -> None:
        valid_unit_ids = {unit.unit_id for unit in self.text_engine.store.units}
        bank_ids = set(self.kv_bank.list_entry_ids())
        for unit_id in list(self._unit_digest_index.keys()):
            if unit_id not in valid_unit_ids or unit_id not in bank_ids:
                self._unit_digest_index.pop(unit_id, None)

    def add_turn(
        self,
        role: str,
        content: str,
        *,
        importance: float = 1.0,
        timestamp: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.text_engine.add_turn(
            role=role,
            content=content,
            importance=importance,
            timestamp=timestamp,
            metadata=metadata,
        )

    def rebuild_indices(self, *, sync_kv: bool = True, force_resync: bool = False) -> list[MemoryUnit]:
        units = self.text_engine.rebuild_memory()
        valid_ids = {unit.unit_id for unit in units}
        removed_ids = self.kv_bank.retain_only(valid_ids)
        for unit_id in removed_ids:
            self._unit_digest_index.pop(unit_id, None)
        for unit_id in list(self._unit_digest_index.keys()):
            if unit_id not in valid_ids:
                self._unit_digest_index.pop(unit_id, None)

        if sync_kv:
            self.sync_kv_for_units(units, force=force_resync)
        return units

    def sync_kv_for_units(self, units: list[MemoryUnit], *, force: bool = False) -> None:
        for unit in units:
            digest = _digest_text(unit.text)
            previous_digest = self._unit_digest_index.get(unit.unit_id)
            is_current = previous_digest == digest and self.kv_bank.has_entry(unit.unit_id)

            if not force and is_current:
                continue

            metadata = {
                "turn_start": unit.turn_start,
                "turn_end": unit.turn_end,
                "unit_digest": digest,
            }
            self.runner.capture_memory_entry(
                entry_id=unit.unit_id,
                text=unit.text,
                bank=self.kv_bank,
                capture_layers=self.capture_layers,
                metadata=metadata,
            )
            self._unit_digest_index[unit.unit_id] = digest

        self._cleanup_digest_index()

    def retrieve_units(self, query: str, *, top_k: int | None = None) -> list[MemoryUnit]:
        if not self.text_engine.store.units:
            self.rebuild_indices(sync_kv=False)
        k = top_k if top_k is not None else self.config.top_k
        return self.text_engine.retrieve(query, top_k=k)

    def retrieve_scored_units(self, query: str, *, top_k: int | None = None) -> list[ScoredMemoryUnit]:
        if not self.text_engine.store.units:
            self.rebuild_indices(sync_kv=False)
        k = top_k if top_k is not None else self.config.top_k
        return self.text_engine.retrieve_with_scores(query, top_k=k)

    def decide(self, query: str, *, top_k: int | None = None) -> RetrievalDecision:
        scored_units = self.retrieve_scored_units(query, top_k=top_k)
        return RetrievalDecision(
            query=query,
            selected_unit_ids=[item.unit.unit_id for item in scored_units],
            selected_texts=[item.unit.text for item in scored_units],
            selected_scores=[item.score for item in scored_units],
        )

    def generate(self, query: str, *, top_k: int | None = None, max_new_tokens: int = 128) -> str:
        decision = self.decide(query, top_k=top_k)

        if decision.selected_unit_ids:
            selected_units = [
                unit
                for unit in self.text_engine.store.units
                if unit.unit_id in set(decision.selected_unit_ids)
            ]
            self.sync_kv_for_units(selected_units)

        return self.runner.generate_with_memory(
            query_text=query,
            bank=self.kv_bank,
            entry_ids=decision.selected_unit_ids,
            max_new_tokens=max_new_tokens,
        )
