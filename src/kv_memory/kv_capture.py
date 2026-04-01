from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LayerKVRecord:
    layer_idx: int
    key_states: list[Any] = field(default_factory=list)
    value_states: list[Any] = field(default_factory=list)


class KVCaptureSession:
    def __init__(self) -> None:
        self.enabled = True
        self._records: dict[int, LayerKVRecord] = {}

    def clear(self) -> None:
        self._records.clear()

    def add_key(self, layer_idx: int, tensor: Any) -> None:
        if not self.enabled:
            return
        record = self._records.setdefault(layer_idx, LayerKVRecord(layer_idx=layer_idx))
        record.key_states.append(tensor.detach().cpu())

    def add_value(self, layer_idx: int, tensor: Any) -> None:
        if not self.enabled:
            return
        record = self._records.setdefault(layer_idx, LayerKVRecord(layer_idx=layer_idx))
        record.value_states.append(tensor.detach().cpu())

    def records(self) -> list[LayerKVRecord]:
        return [self._records[index] for index in sorted(self._records.keys())]
