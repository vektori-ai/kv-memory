from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .kv_capture import KVCaptureSession


@dataclass(slots=True)
class HookHandleGroup:
    handles: list[Any]

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def attach_qwen_kv_projection_hooks(
    model: Any,
    session: KVCaptureSession,
    capture_layers: set[int] | None = None,
) -> HookHandleGroup:
    handles: list[Any] = []

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Unsupported model structure: expected model.model.layers")

    for layer_idx, layer in enumerate(model.model.layers):
        if capture_layers is not None and layer_idx not in capture_layers:
            continue

        attention = layer.self_attn
        if not hasattr(attention, "k_proj") or not hasattr(attention, "v_proj"):
            continue

        def key_hook(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_idx) -> None:
            session.add_key(idx, output)

        def value_hook(_module: Any, _inputs: tuple[Any, ...], output: Any, idx: int = layer_idx) -> None:
            session.add_value(idx, output)

        handles.append(attention.k_proj.register_forward_hook(key_hook))
        handles.append(attention.v_proj.register_forward_hook(value_hook))

    return HookHandleGroup(handles=handles)
