"""
hf_adapter.py — HuggingFace transformers adapter.

Phase 1 adapter. Uses:
  - output_hidden_states=True for capture()
  - past_key_values for inject_and_generate()

KV layout from HF (transformers 5.x):
  past_key_values.layers[i].keys  → K: [batch, kv_heads, seq, head_dim]
  past_key_values.layers[i].values → V: [batch, kv_heads, seq, head_dim]

  For GQA models (e.g. Qwen2.5-7B): kv_heads < num_attention_heads.

We store with batch dim squeezed:
  K / V shape: [kv_heads, seq, head_dim]
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
        # GQA: KV heads may differ from Q heads (e.g. Qwen2.5-7B has 28 Q heads, 4 KV heads)
        self._num_kv_heads: int = getattr(
            model.config, "num_key_value_heads", model.config.num_attention_heads
        )

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

        # Normalize past_key_values to a list of (K, V) tuples — one per layer.
        # K/V shape from HF: [batch=1, kv_heads, seq, head_dim]
        #
        # transformers 5.x:  DynamicCache with .layers list; each DynamicLayer has .keys/.values
        # transformers 4.38+: DynamicCache with .to_legacy_cache() or .key_cache/.value_cache
        # transformers <4.38: legacy tuple of (K, V) per layer (directly subscriptable)
        pkv = out.past_key_values
        if hasattr(pkv, "layers") and pkv.layers:
            # transformers 5.x
            pkv_list: list[tuple[torch.Tensor, torch.Tensor]] = [
                (layer.keys, layer.values)
                for layer in pkv.layers
                if getattr(layer, "is_initialized", True) and hasattr(layer, "keys")
            ]
            logger.debug("capture: transformers 5.x DynamicCache, %d layers", len(pkv_list))
        elif hasattr(pkv, "to_legacy_cache"):
            # transformers 4.38–4.x
            pkv_list = list(pkv.to_legacy_cache())
            logger.debug("capture: transformers 4.x DynamicCache (to_legacy_cache), %d layers", len(pkv_list))
        elif hasattr(pkv, "key_cache"):
            pkv_list = list(zip(pkv.key_cache, pkv.value_cache))
            logger.debug("capture: transformers 4.x DynamicCache (key_cache), %d layers", len(pkv_list))
        else:
            pkv_list = list(pkv)
            logger.debug("capture: legacy tuple cache, %d layers", len(pkv_list))

        kv_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_idx in layers:
            if layer_idx >= len(pkv_list):
                logger.warning(
                    "Layer %d out of range (cache has %d layers), skipping",
                    layer_idx, len(pkv_list),
                )
                continue
            K = pkv_list[layer_idx][0].squeeze(0)  # [kv_heads, seq, head_dim]
            V = pkv_list[layer_idx][1].squeeze(0)
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
        Dequantize stored KV blocks and prepend as past_key_values, then generate.

        When blocks are provided: uses a manual prefill + decode loop with explicit
        position_ids. This bypasses transformers.generate() which has breaking
        changes in 4.40+ for legacy past_key_values formats.

        When no blocks: delegates to model.generate() normally.

        Doc-wise RoPE: query tokens start at position = total_past_tokens,
        placing them correctly after all injected context regardless of which
        layers have stored KV.
        """
        if not current_tokens:
            raise ValueError("inject_and_generate() called with empty current_tokens")

        input_ids = torch.tensor([current_tokens], dtype=torch.long).to(self.model.device)

        if not blocks:
            # No injection — use generate() directly, no API issues
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **dict(generation_kwargs),
                )
            sequences = out.tolist()
            text = self._tokenizer.decode(sequences[0], skip_special_tokens=True)
            return GenerationOutput(sequences=sequences, text=text)

        # KV injection path: manual decode loop to avoid transformers version issues
        past_kv, total_past_tokens = self._build_cache(blocks)
        return self._manual_generate(
            input_ids=input_ids,
            past_kv=past_kv,
            position_offset=total_past_tokens,
            generation_kwargs=generation_kwargs,
        )

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

    def _build_cache(
        self,
        blocks: list[KVBlock],
    ) -> tuple[object, int]:
        """
        Assemble a KV cache from a list of KVBlocks.

        Builds a DynamicCache (transformers >= 4.38) when available.
        Falls back to legacy tuple-of-tuples for older versions.

        Every layer gets an entry: real KV for stored layers, 0-length placeholder
        tensors for unstored layers. All tensors are cast to the model's native dtype
        so SDPA never sees a dtype mismatch between query and injected KV.

        Returns:
            (past_key_values, total_past_tokens)
            total_past_tokens: sum of token counts across all injected blocks,
                               used to set position_ids and the attention mask offset.
        """
        head_dim = self._d_model // self._num_heads
        kv_dtype = self.model.dtype  # match model's compute dtype (fp16, bf16, fp32)
        total_past_tokens = sum(b.token_count for b in blocks)

        # Build per-layer (K_cat, V_cat) — 0-length placeholders for unstored layers.
        # Use _num_kv_heads (not _num_heads) — GQA models have fewer KV heads than Q heads.
        layer_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(self._num_layers):
            K_parts: list[torch.Tensor] = []
            V_parts: list[torch.Tensor] = []

            for block in blocks:
                if layer_idx not in block.kv_by_layer:
                    continue
                K_q, V_q = block.kv_by_layer[layer_idx]
                k_scale, v_scale = block.quant_scales[layer_idx]
                K = torch.from_numpy(K_q.astype(np.float32)) * k_scale
                V = torch.from_numpy(V_q.astype(np.float32)) * v_scale
                K = K.to(dtype=kv_dtype, device=self.model.device)
                V = V.to(dtype=kv_dtype, device=self.model.device)
                K_parts.append(K)  # [kv_heads, seq_i, head_dim]
                V_parts.append(V)

            if K_parts:
                K_cat = torch.cat(K_parts, dim=1).unsqueeze(0)  # [1, kv_heads, total_seq, head_dim]
                V_cat = torch.cat(V_parts, dim=1).unsqueeze(0)
            else:
                # 0-length placeholder: this layer has no prior context.
                K_cat = torch.zeros(1, self._num_kv_heads, 0, head_dim,
                                    dtype=kv_dtype, device=self.model.device)
                V_cat = torch.zeros(1, self._num_kv_heads, 0, head_dim,
                                    dtype=kv_dtype, device=self.model.device)

            layer_kvs.append((K_cat, V_cat))

        # Build a DynamicCache — newer transformers (4.38+) requires an object with
        # get_seq_length(). We subclass it so that get_seq_length() always returns
        # total_past_tokens: masking_utils uses this as q_offset for the causal mask,
        # and it must match the position_ids we pass (which also start at total_past_tokens).
        # Without this, layers with 0-length placeholders would return 0 from get_seq_length(),
        # causing the causal mask to be built with the wrong offset.
        try:
            from transformers import DynamicCache

            class _InjectedCache(DynamicCache):
                """DynamicCache with a fixed get_seq_length() for correct causal masking."""
                _total_past: int

                def get_seq_length(self, layer_idx: int = 0) -> int:  # type: ignore[override]
                    return self._total_past

            cache = _InjectedCache()
            cache._total_past = total_past_tokens
            for layer_idx, (K_cat, V_cat) in enumerate(layer_kvs):
                cache.update(K_cat, V_cat, layer_idx)
            return cache, total_past_tokens
        except ImportError:
            # Fallback for very old transformers
            return tuple(layer_kvs), total_past_tokens

    def _manual_generate(
        self,
        input_ids: torch.Tensor,
        past_kv,
        position_offset: int,
        generation_kwargs: dict,
    ) -> "GenerationOutput":
        """
        Manual prefill + greedy/sampling decode loop with explicit position_ids.

        Handles: max_new_tokens, do_sample, temperature, top_p, eos_token_id.
        Called only when KV blocks are injected.
        """
        max_new_tokens: int = generation_kwargs.get("max_new_tokens", 20)
        do_sample: bool = generation_kwargs.get("do_sample", False)
        temperature: float = float(generation_kwargs.get("temperature", 1.0))
        top_p: float = float(generation_kwargs.get("top_p", 1.0))
        eos_id = generation_kwargs.get(
            "eos_token_id",
            getattr(self._tokenizer, "eos_token_id", None),
        )

        seq_len = input_ids.shape[1]
        generated: list[int] = input_ids[0].tolist()

        # Prefill: process all query tokens with injected KV context.
        # attention_mask must cover the full sequence (injected past + current query)
        # so the model knows which positions are valid. Without it, newer transformers
        # infers a mask sized only to input_ids, ignoring the injected past tokens.
        pos_ids = torch.arange(
            position_offset, position_offset + seq_len, device=self.model.device
        ).unsqueeze(0)  # [1, seq]
        attn_mask = torch.ones(
            1, position_offset + seq_len, device=self.model.device, dtype=torch.long
        )

        with torch.no_grad():
            fwd = self.model(
                input_ids=input_ids,
                past_key_values=past_kv,
                position_ids=pos_ids,
                attention_mask=attn_mask,
                use_cache=True,
            )

        current_pkv = fwd.past_key_values
        next_pos = position_offset + seq_len

        # Autoregressive decode
        for _ in range(max_new_tokens):
            logits = fwd.logits[0, -1, :]  # [vocab]

            if do_sample:
                logits = logits / max(temperature, 1e-8)
                if top_p < 1.0:
                    logits = self._top_p_filter(logits, top_p)
                probs = torch.softmax(logits, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_token = int(logits.argmax().item())

            generated.append(next_token)
            if eos_id is not None and next_token == eos_id:
                break

            step_ids = torch.tensor([[next_token]], device=self.model.device)
            step_pos = torch.tensor([[next_pos]], device=self.model.device)
            # Grow the mask by 1 each step to include the new token
            step_mask = torch.ones(1, next_pos + 1, device=self.model.device, dtype=torch.long)
            with torch.no_grad():
                fwd = self.model(
                    input_ids=step_ids,
                    past_key_values=current_pkv,
                    position_ids=step_pos,
                    attention_mask=step_mask,
                    use_cache=True,
                )
            current_pkv = fwd.past_key_values
            next_pos += 1

        text = self._tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationOutput(sequences=[generated], text=text)

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Zero out logits below the nucleus (top-p) threshold."""
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens once cumulative prob exceeds top_p
        remove_mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[remove_mask] = float("-inf")
        # Scatter back to original order
        return logits.scatter(0, sorted_idx, sorted_logits)
