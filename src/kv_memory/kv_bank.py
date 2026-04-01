from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LayerKVTensor:
    layer_idx: int
    key: Any
    value: Any


@dataclass(slots=True)
class MemoryKVEntry:
    entry_id: str
    text: str
    timestamp: float
    layer_kv: list[LayerKVTensor] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryKVBank:
    def __init__(self, max_entries: int | None = None) -> None:
        self._entries: dict[str, MemoryKVEntry] = {}
        self.max_entries = max_entries

    def add_entry(self, entry: MemoryKVEntry) -> None:
        if self.max_entries is not None and self.max_entries > 0:
            while len(self._entries) >= self.max_entries and entry.entry_id not in self._entries:
                oldest_id, _ = min(
                    self._entries.items(),
                    key=lambda item: item[1].timestamp,
                )
                self._entries.pop(oldest_id, None)
        self._entries[entry.entry_id] = entry

    def has_entry(self, entry_id: str) -> bool:
        return entry_id in self._entries

    def remove_entry(self, entry_id: str) -> None:
        self._entries.pop(entry_id, None)

    def retain_only(self, entry_ids: set[str]) -> list[str]:
        removed_ids = [entry_id for entry_id in self._entries.keys() if entry_id not in entry_ids]
        for entry_id in removed_ids:
            self._entries.pop(entry_id, None)
        return removed_ids

    def get_entry(self, entry_id: str) -> MemoryKVEntry:
        return self._entries[entry_id]

    def get_entries(self, entry_ids: list[str]) -> list[MemoryKVEntry]:
        return [self._entries[entry_id] for entry_id in entry_ids if entry_id in self._entries]

    def list_entry_ids(self) -> list[str]:
        return sorted(self._entries.keys())

    def size(self) -> int:
        return len(self._entries)


def _reshape_proj_to_cache(proj_tensor: Any, num_kv_heads: int, head_dim: int) -> Any:
    tensor = proj_tensor
    if tensor.dim() != 3:
        raise ValueError(f"Expected projection tensor with shape [batch, seq, hidden], got {tuple(tensor.shape)}")

    batch, seq_len, hidden = tensor.shape
    expected_hidden = num_kv_heads * head_dim
    if hidden != expected_hidden:
        raise ValueError(
            f"Projection hidden size mismatch: hidden={hidden}, expected={expected_hidden} "
            f"(num_kv_heads={num_kv_heads}, head_dim={head_dim})"
        )

    reshaped = tensor.view(batch, seq_len, num_kv_heads, head_dim)
    return reshaped.permute(0, 2, 1, 3).contiguous()


def build_memory_entry_from_capture(
    *,
    entry_id: str,
    text: str,
    timestamp: float,
    records: list[Any],
    num_kv_heads: int,
    head_dim: int,
    metadata: dict[str, Any] | None = None,
) -> MemoryKVEntry:
    layer_kv: list[LayerKVTensor] = []
    for record in records:
        if not record.key_states or not record.value_states:
            continue

        key_proj = record.key_states[-1]
        value_proj = record.value_states[-1]
        key_cache = _reshape_proj_to_cache(key_proj, num_kv_heads, head_dim)
        value_cache = _reshape_proj_to_cache(value_proj, num_kv_heads, head_dim)
        layer_kv.append(LayerKVTensor(layer_idx=record.layer_idx, key=key_cache, value=value_cache))

    layer_kv.sort(key=lambda item: item.layer_idx)
    return MemoryKVEntry(
        entry_id=entry_id,
        text=text,
        timestamp=timestamp,
        layer_kv=layer_kv,
        metadata=metadata or {},
    )


def merge_entries_to_past_key_values(
    entries: list[MemoryKVEntry],
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: Any,
    device: Any,
) -> tuple[tuple[Any, Any], ...]:
    import torch

    if num_layers <= 0:
        return tuple()

    if not entries:
        empty = torch.empty((1, num_kv_heads, 0, head_dim), dtype=dtype, device=device)
        return tuple((empty, empty.clone()) for _ in range(num_layers))

    layer_map: dict[int, list[LayerKVTensor]] = {}
    for entry in entries:
        for layer in entry.layer_kv:
            layer_map.setdefault(layer.layer_idx, []).append(layer)

    merged: list[tuple[Any, Any]] = []
    for layer_idx in range(num_layers):
        if layer_idx in layer_map:
            layer_list = layer_map[layer_idx]
            key = torch.cat([layer.key.to(device=device, dtype=dtype) for layer in layer_list], dim=2)
            value = torch.cat([layer.value.to(device=device, dtype=dtype) for layer in layer_list], dim=2)
            merged.append((key, value))
            continue

        empty = torch.empty((1, num_kv_heads, 0, head_dim), dtype=dtype, device=device)
        merged.append((empty, empty.clone()))

    return tuple(merged)
