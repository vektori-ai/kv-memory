"""
hf_adapter.py — HuggingFace transformers adapter.

Phase 1 adapter. Uses:
  - output_hidden_states=True for capture()
  - past_key_values for inject_and_generate()

KV layout from HF:
  past_key_values[layer] = (K, V)
  K / V shape: [batch, heads, seq, head_dim]

We store with batch dim squeezed:
  K / V shape: [heads, seq, head_dim]
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from .base import BaseAdapter
from ..storage.schema import KVBlock, GenerationOutput

logger = logging.getLogger(__name__)


class HFAdapter(BaseAdapter):
    """
    HuggingFace adapter for capture and KV-injected generation.

    Usage:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(...)
        tokenizer = AutoTokenizer.from_pretrained(...)
        adapter = HFAdapter(model, tokenizer)
    """

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self._tokenizer = tokenizer
        self._num_layers: int = model.config.num_hidden_layers
        self._d_model: int = model.config.hidden_size
        self._num_heads: int = model.config.num_attention_heads

    # ------------------------------------------------------------------
    # BaseAdapter properties
    # ------------------------------------------------------------------

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def tokenizer(self):
        return self._tokenizer

    def supports_kv_inject(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # capture()
    # ------------------------------------------------------------------

    def capture(
        self,
        tokens: list[int],
        text: str,
        layers: list[int],
    ) -> tuple[
        dict[int, tuple[torch.Tensor, torch.Tensor]],
        dict[int, torch.Tensor],
    ]:
        """
        Single forward pass to extract KV tensors and hidden states.

        Returns tensors with batch dim removed (squeezed).
        K/V: [heads, seq, head_dim] float16
        hidden: [seq, d_model] float32
        """
        if not tokens:
            raise ValueError("capture() called with empty token list")

        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.model.device)

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=True,
            )

        # past_key_values: tuple of (K, V) per layer
        # K/V shape from HF: [batch=1, heads, seq, head_dim]
        kv_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_idx in layers:
            if layer_idx >= len(out.past_key_values):
                logger.warning(
                    "Layer %d out of range (model has %d layers), skipping",
                    layer_idx,
                    len(out.past_key_values),
                )
                continue
            K = out.past_key_values[layer_idx][0].squeeze(0)  # [heads, seq, head_dim]
            V = out.past_key_values[layer_idx][1].squeeze(0)
            kv_by_layer[layer_idx] = (K, V)

        # hidden_states: tuple of tensors [batch, seq, d_model], one per layer + embedding
        # Index offset: hidden_states[0] is embedding, hidden_states[l+1] is layer l output
        hidden_by_layer: dict[int, torch.Tensor] = {}
        for layer_idx in layers:
            hs_idx = layer_idx + 1  # +1 because index 0 is the embedding layer
            if hs_idx >= len(out.hidden_states):
                logger.warning(
                    "Hidden state index %d out of range, skipping layer %d",
                    hs_idx,
                    layer_idx,
                )
                continue
            hidden = out.hidden_states[hs_idx].squeeze(0).float()  # [seq, d_model]
            hidden_by_layer[layer_idx] = hidden

        return kv_by_layer, hidden_by_layer

    # ------------------------------------------------------------------
    # inject_and_generate()
    # ------------------------------------------------------------------

    def inject_and_generate(
        self,
        blocks: list[KVBlock],
        current_tokens: list[int],
        generation_kwargs: dict,
    ) -> GenerationOutput:
        """
        Dequantize stored KV blocks and prepend as past_key_values.

        Doc-wise RoPE: each block's KV was captured with positions [0..n],
        so no re-rotation is needed. The model sees them as independent
        documents. Query tokens get positions starting at len(blocks).

        If no blocks are provided, falls back to plain generation.
        """
        if not current_tokens:
            raise ValueError("inject_and_generate() called with empty current_tokens")

        combined_kv = self._build_past_key_values(blocks)
        input_ids = torch.tensor([current_tokens], dtype=torch.long).to(self.model.device)

        gen_kwargs = dict(generation_kwargs)  # copy to avoid mutation
        if combined_kv is not None:
            gen_kwargs["past_key_values"] = combined_kv

        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, **gen_kwargs)

        sequences = out.tolist()  # [batch, seq]
        text = self._tokenizer.decode(sequences[0], skip_special_tokens=True)
        return GenerationOutput(sequences=sequences, text=text)

    def generate(
        self,
        tokens: list[int],
        generation_kwargs: dict,
    ) -> GenerationOutput:
        """
        Plain generation without KV injection. Used for baseline comparison.
        """
        return self.inject_and_generate([], tokens, generation_kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_past_key_values(
        self,
        blocks: list[KVBlock],
    ) -> Optional[list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Assemble past_key_values from a list of KVBlocks.

        For each layer, concatenate K and V from all blocks that have
        that layer stored. Dequantizes int8 -> float16 on the fly.

        Returns None if no blocks have any KV data.
        """
        if not blocks:
            return None

        combined: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = [
            None
        ] * self._num_layers

        for layer_idx in range(self._num_layers):
            K_parts: list[torch.Tensor] = []
            V_parts: list[torch.Tensor] = []

            for block in blocks:
                if layer_idx not in block.kv_by_layer:
                    continue
                K_q, V_q = block.kv_by_layer[layer_idx]
                k_scale, v_scale = block.quant_scales[layer_idx]

                # Dequantize: int8 -> float32 -> scale -> float16
                K = torch.from_numpy(K_q.astype(np.float32)) * k_scale
                V = torch.from_numpy(V_q.astype(np.float32)) * v_scale
                K = K.to(dtype=torch.float16, device=self.model.device)
                V = V.to(dtype=torch.float16, device=self.model.device)
                K_parts.append(K)   # [heads, seq_i, head_dim]
                V_parts.append(V)

            if K_parts:
                # Concatenate along seq dim, then add batch dim
                K_cat = torch.cat(K_parts, dim=1).unsqueeze(0)  # [1, heads, total_seq, head_dim]
                V_cat = torch.cat(V_parts, dim=1).unsqueeze(0)
                combined[layer_idx] = (K_cat, V_cat)

        # Only return if at least one layer has data
        if any(x is not None for x in combined):
            # Fill None slots with empty tensors at correct shapes so HF doesn't break
            # We only need layers that exist in the model
            return combined
        return None
