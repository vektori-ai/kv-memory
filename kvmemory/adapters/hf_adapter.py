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


def _apply_entropy_weights(
    hidden: torch.Tensor,
    attentions,
    layer_idx: int,
    seq_len: int,
    batch_idx: int = 0,
) -> torch.Tensor:
    """
    Pre-scale hidden states by attention-entropy salience weights so that
    compute_retrieval_vec's mean pooling produces an entropy-weighted mean.

    Salience = softmax(-entropy) over token positions.
    Low-entropy tokens have focused attention → they are content words → higher weight.

    Pre-scaling trick:
        hidden_scaled[i] = hidden[i] * weight[i] * seq_len
        mean(hidden_scaled) = sum(weight[i] * hidden[i])   ← weighted mean ✓

    Falls back to unweighted hidden states if attentions are unavailable
    (some models set output_attentions=True but return None for certain layers).
    """
    if attentions is None or layer_idx >= len(attentions) or attentions[layer_idx] is None:
        return hidden  # graceful fallback: unweighted

    try:
        attn = attentions[layer_idx]          # [batch, heads, seq, seq]
        attn_item = attn[batch_idx, :, :seq_len, :seq_len].float()  # [heads, seq, seq]

        # Mean attention distribution per query token across heads
        mean_attn = attn_item.mean(dim=0)     # [seq, seq]

        # Renormalize rows (numerical safety — softmax rows should already sum to 1)
        row_sums = mean_attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        mean_attn = mean_attn / row_sums

        # Row entropy: how focused is each token's attention?  [seq]
        entropy = -(mean_attn * torch.log(mean_attn.clamp(min=1e-9))).sum(dim=-1)

        # Salience: low entropy = focused = important token
        salience = torch.softmax(-entropy, dim=0)  # [seq], sums to 1

        # Pre-scale so downstream mean pooling yields the weighted mean
        return hidden * (salience.unsqueeze(-1) * seq_len)

    except Exception as e:
        logger.debug("Entropy weighting failed for layer %d, falling back: %s", layer_idx, e)
        return hidden


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
    def num_kv_heads(self) -> int:
        return self._num_kv_heads

    @property
    def head_dim(self) -> int:
        return self._d_model // self._num_heads

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
        Independent single-item forward pass to extract KV tensors and hidden states.

        Uses its own direct forward pass (not delegating to capture_batch) so that
        the fallback path in write_pipeline does not share the same code path as
        the primary batch path (if capture_batch fails, capture() may still succeed).

        Returns tensors with batch dim removed (squeezed).
        K/V: [kv_heads, seq, head_dim]
        hidden: [seq, d_model] float32
        """
        if not tokens:
            raise ValueError("capture() called with empty token list")

        device = self.model.device
        seq_len = len(tokens)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)  # [1, seq_len]
        attn_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                output_attentions=True,   # needed for entropy-weighted pooling
                use_cache=True,
            )

        # Normalise past_key_values (same logic as capture_batch)
        pkv = out.past_key_values
        if hasattr(pkv, "layers") and pkv.layers:
            pkv_list: list[tuple[torch.Tensor, torch.Tensor]] = [
                (layer.keys, layer.values)
                for layer in pkv.layers
                if getattr(layer, "is_initialized", True) and hasattr(layer, "keys")
            ]
        elif hasattr(pkv, "to_legacy_cache"):
            pkv_list = list(pkv.to_legacy_cache())
        elif hasattr(pkv, "key_cache"):
            pkv_list = list(zip(pkv.key_cache, pkv.value_cache))
        else:
            pkv_list = list(pkv)

        kv_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_idx in layers:
            if layer_idx >= len(pkv_list):
                logger.warning(
                    "Layer %d out of range (cache has %d layers), skipping",
                    layer_idx, len(pkv_list),
                )
                continue
            K = pkv_list[layer_idx][0][0, :, :seq_len, :]  # [kv_heads, seq_len, head_dim]
            V = pkv_list[layer_idx][1][0, :, :seq_len, :]
            kv_by_layer[layer_idx] = (K, V)

        hidden_by_layer: dict[int, torch.Tensor] = {}
        for layer_idx in layers:
            hs_idx = layer_idx + 1  # hidden_states[0] is embedding layer
            if hs_idx >= len(out.hidden_states):
                logger.warning(
                    "Hidden state index %d out of range, skipping layer %d",
                    hs_idx, layer_idx,
                )
                continue
            hidden = out.hidden_states[hs_idx][0, :seq_len, :].float()  # [seq_len, d_model]
            hidden_by_layer[layer_idx] = _apply_entropy_weights(
                hidden, out.attentions, layer_idx, seq_len
            )

        return kv_by_layer, hidden_by_layer

    def capture_batch(
        self,
        batch_tokens: list[list[int]],
        layers: list[int],
    ) -> list[tuple[
        dict[int, tuple[torch.Tensor, torch.Tensor]],
        dict[int, torch.Tensor],
    ]]:
        """
        Batched forward pass: process multiple chunks in one model() call.

        Chunks are right-padded to the longest sequence in the batch.
        Padding positions are masked out and never appear in the returned
        KV/hidden tensors — each result is sliced to its original length.

        Args:
            batch_tokens: list of token-ID lists, one per chunk
            layers:       which layers to extract KV + hidden states from

        Returns:
            List of (kv_by_layer, hidden_by_layer) — one entry per chunk,
            in the same order as batch_tokens. Shapes match single capture():
              K/V:    [kv_heads, seq_i, head_dim]
              hidden: [seq_i, d_model] float32
        """
        if not batch_tokens:
            return []

        device = self.model.device
        lengths = [len(t) for t in batch_tokens]
        max_len = max(lengths)
        pad_id = getattr(self._tokenizer, "pad_token_id", None) or 0

        # Right-pad all sequences to max_len
        padded = [t + [pad_id] * (max_len - len(t)) for t in batch_tokens]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)  # [B, max_len]

        # Attention mask: 1 for real tokens, 0 for padding
        attn_mask = torch.zeros_like(input_ids)
        for i, length in enumerate(lengths):
            attn_mask[i, :length] = 1

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                output_attentions=True,   # needed for entropy-weighted pooling
                use_cache=True,
            )

        # Normalise past_key_values to a flat list of (K, V) per layer.
        # K/V shape from HF: [batch, kv_heads, seq, head_dim]
        pkv = out.past_key_values
        if hasattr(pkv, "layers") and pkv.layers:
            pkv_list: list[tuple[torch.Tensor, torch.Tensor]] = [
                (layer.keys, layer.values)
                for layer in pkv.layers
                if getattr(layer, "is_initialized", True) and hasattr(layer, "keys")
            ]
            logger.debug("capture_batch: transformers 5.x DynamicCache, %d layers", len(pkv_list))
        elif hasattr(pkv, "to_legacy_cache"):
            pkv_list = list(pkv.to_legacy_cache())
            logger.debug("capture_batch: transformers 4.x (to_legacy_cache), %d layers", len(pkv_list))
        elif hasattr(pkv, "key_cache"):
            pkv_list = list(zip(pkv.key_cache, pkv.value_cache))
            logger.debug("capture_batch: transformers 4.x (key_cache), %d layers", len(pkv_list))
        else:
            pkv_list = list(pkv)
            logger.debug("capture_batch: legacy tuple cache, %d layers", len(pkv_list))

        results = []
        for batch_idx, seq_len in enumerate(lengths):
            kv_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
            for layer_idx in layers:
                if layer_idx >= len(pkv_list):
                    logger.warning(
                        "Layer %d out of range (cache has %d layers), skipping",
                        layer_idx, len(pkv_list),
                    )
                    continue
                # Slice batch item and strip padding: [:, :seq_len, :]
                K = pkv_list[layer_idx][0][batch_idx, :, :seq_len, :]  # [kv_heads, seq_len, head_dim]
                V = pkv_list[layer_idx][1][batch_idx, :, :seq_len, :]
                kv_by_layer[layer_idx] = (K, V)

            hidden_by_layer: dict[int, torch.Tensor] = {}
            for layer_idx in layers:
                hs_idx = layer_idx + 1  # hidden_states[0] is embedding layer
                if hs_idx >= len(out.hidden_states):
                    logger.warning(
                        "Hidden state index %d out of range, skipping layer %d",
                        hs_idx, layer_idx,
                    )
                    continue
                hidden = out.hidden_states[hs_idx][batch_idx, :seq_len, :].float()  # [seq_len, d_model]
                hidden_by_layer[layer_idx] = _apply_entropy_weights(
                    hidden, out.attentions, layer_idx, seq_len, batch_idx=batch_idx
                )

            results.append((kv_by_layer, hidden_by_layer))

        return results

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

        # After the prefill, the KV cache has grown by seq_len.
        # _InjectedCache.get_seq_length() was set to position_offset (the initial
        # injected past), but the model uses get_seq_length() to compute the causal
        # mask offset for each decode step. If it stays at position_offset, the
        # causal mask only allows attending to positions 0..position_offset, masking
        # out the query tokens that were just processed in the prefill — so decode
        # tokens can't attend to the full history and generate garbage.
        # Fix: keep _total_past in sync with the actual cache length after each step.
        def _sync_cache_length(pkv, length: int) -> None:
            if hasattr(pkv, "_total_past"):
                pkv._total_past = length

        _sync_cache_length(current_pkv, next_pos)

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
            _sync_cache_length(current_pkv, next_pos)

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
