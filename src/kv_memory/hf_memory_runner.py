from __future__ import annotations

import time
from typing import Any

from .hf_hooks import attach_qwen_kv_projection_hooks
from .kv_bank import MemoryKVBank, build_memory_entry_from_capture, merge_entries_to_past_key_values
from .kv_capture import KVCaptureSession


class HFMemoryRunner:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def _infer_kv_layout(self) -> tuple[int, int]:
        config = self.model.config
        num_kv_heads = getattr(config, "num_key_value_heads", None)
        num_attention_heads = getattr(config, "num_attention_heads", None)
        hidden_size = getattr(config, "hidden_size", None)

        if num_kv_heads is None:
            if num_attention_heads is None:
                raise ValueError("Unable to infer num_key_value_heads from model config")
            num_kv_heads = num_attention_heads

        if hidden_size is None or num_attention_heads is None:
            raise ValueError("Unable to infer head dimension from model config")

        head_dim = hidden_size // num_attention_heads
        return num_kv_heads, head_dim

    def _infer_num_layers(self) -> int:
        num_layers = getattr(self.model.config, "num_hidden_layers", None)
        if num_layers is None:
            raise ValueError("Unable to infer number of hidden layers from model config")
        return num_layers

    def _model_device(self) -> Any:
        return next(self.model.parameters()).device

    def _model_dtype(self) -> Any:
        return next(self.model.parameters()).dtype

    def _to_model_cache(self, past_key_values: Any) -> Any:
        if past_key_values is None:
            return None

        try:
            from transformers.cache_utils import Cache, DynamicCache
        except Exception:
            return past_key_values

        if isinstance(past_key_values, Cache):
            return past_key_values

        try:
            return DynamicCache.from_legacy_cache(past_key_values)
        except Exception:
            return past_key_values

    def capture_memory_entry(
        self,
        *,
        entry_id: str,
        text: str,
        bank: MemoryKVBank,
        capture_layers: set[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        session = KVCaptureSession()
        hooks = attach_qwen_kv_projection_hooks(self.model, session, capture_layers=capture_layers)

        try:
            encoded = self.tokenizer(text, return_tensors="pt")
            model_device = self._model_device()
            encoded = {name: value.to(model_device) for name, value in encoded.items()}

            import torch

            with torch.no_grad():
                self.model(**encoded, use_cache=True)
        finally:
            hooks.remove()

        num_kv_heads, head_dim = self._infer_kv_layout()
        entry = build_memory_entry_from_capture(
            entry_id=entry_id,
            text=text,
            timestamp=time.time(),
            records=session.records(),
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            metadata=metadata,
        )
        bank.add_entry(entry)

    def build_past_key_values(
        self,
        *,
        bank: MemoryKVBank,
        entry_ids: list[str],
    ) -> tuple[tuple[Any, Any], ...] | None:
        entries = bank.get_entries(entry_ids)
        usable_entries = [entry for entry in entries if entry.layer_kv]
        if not usable_entries:
            return None

        num_kv_heads, head_dim = self._infer_kv_layout()
        num_layers = self._infer_num_layers()
        past_key_values = merge_entries_to_past_key_values(
            usable_entries,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=self._model_dtype(),
            device=self._model_device(),
        )

        has_any_tokens = any(layer_key.shape[2] > 0 for layer_key, _ in past_key_values)
        if not has_any_tokens:
            return None

        return past_key_values

    def forward_with_memory(
        self,
        *,
        query_text: str,
        bank: MemoryKVBank,
        entry_ids: list[str],
    ) -> Any:
        encoded = self.tokenizer(query_text, return_tensors="pt")
        encoded = {name: value.to(self._model_device()) for name, value in encoded.items()}
        past_key_values = self.build_past_key_values(bank=bank, entry_ids=entry_ids)

        import torch

        with torch.no_grad():
            if not entry_ids or past_key_values is None:
                return self.model(**encoded, use_cache=True)
            return self.model(
                **encoded,
                past_key_values=self._to_model_cache(past_key_values),
                use_cache=True,
            )

    def generate_with_memory(
        self,
        *,
        query_text: str,
        bank: MemoryKVBank,
        entry_ids: list[str],
        max_new_tokens: int = 64,
    ) -> str:
        import torch

        encoded = self.tokenizer(query_text, return_tensors="pt")
        encoded = {name: value.to(self._model_device()) for name, value in encoded.items()}

        if entry_ids:
            past_key_values = self.build_past_key_values(bank=bank, entry_ids=entry_ids)
            if past_key_values is not None:
                input_ids = encoded["input_ids"]
                base_attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
                memory_tokens = int(past_key_values[0][0].shape[2]) if past_key_values else 0
                if memory_tokens > 0:
                    memory_mask = torch.ones(
                        (base_attention_mask.shape[0], memory_tokens),
                        dtype=base_attention_mask.dtype,
                        device=base_attention_mask.device,
                    )
                    attention_mask = torch.cat([memory_mask, base_attention_mask], dim=1)
                else:
                    attention_mask = base_attention_mask

                generated_ids = input_ids
                current_ids = input_ids
                current_mask = attention_mask
                current_past = self._to_model_cache(past_key_values)

                with torch.no_grad():
                    for _ in range(max_new_tokens):
                        outputs = self.model(
                            input_ids=current_ids,
                            attention_mask=current_mask,
                            past_key_values=current_past,
                            use_cache=True,
                        )
                        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                        generated_ids = torch.cat([generated_ids, next_token], dim=1)
                        current_ids = next_token
                        current_past = outputs.past_key_values
                        one_mask = torch.ones(
                            (current_mask.shape[0], 1),
                            dtype=current_mask.dtype,
                            device=current_mask.device,
                        )
                        current_mask = torch.cat([current_mask, one_mask], dim=1)

                output = generated_ids
            else:
                output = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                )
        else:
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
