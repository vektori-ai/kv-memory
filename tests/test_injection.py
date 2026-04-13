"""
test_injection.py — Injection pipeline tests.

Tests the injector module and the full write -> retrieve -> inject cycle
using mock components (no real model, no real Qdrant).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from kvmemory.core.injector import inject_and_generate
from kvmemory.core.write_pipeline import quantize_int8, dequantize_int8
from kvmemory.storage.kv_store import KVStore
from kvmemory.storage.schema import KVBlock, GenerationOutput


class TestInjector:
    def test_no_blocks_calls_adapter(self, mock_adapter):
        tokens = mock_adapter.tokenizer.encode("test query")
        out = inject_and_generate(mock_adapter, [], tokens, {})
        assert isinstance(out, GenerationOutput)

    def test_with_blocks_calls_inject(self, mock_adapter, sample_kv_block):
        tokens = mock_adapter.tokenizer.encode("test query")
        out = inject_and_generate(mock_adapter, [sample_kv_block], tokens, {})
        assert isinstance(out, GenerationOutput)
        assert out.sequences is not None

    def test_degraded_mode_uses_text_prefix(self, mock_adapter, sample_kv_block):
        """When adapter doesn't support KV inject, text prefix is used."""
        mock_adapter_copy = MagicMock()
        mock_adapter_copy.supports_kv_inject.return_value = False
        mock_adapter_copy.tokenizer = mock_adapter.tokenizer

        captured_tokens = []

        def fake_inject(blocks, tokens, kwargs):
            captured_tokens.extend(tokens)
            return GenerationOutput(sequences=[tokens], text="output")

        mock_adapter_copy.inject_and_generate.side_effect = fake_inject

        query_tokens = mock_adapter.tokenizer.encode("query here")
        inject_and_generate(mock_adapter_copy, [sample_kv_block], query_tokens, {})

        # In degraded mode, the text prefix tokens are prepended to query tokens
        # So captured_tokens should be longer than just query_tokens
        prefix_tokens = mock_adapter.tokenizer.encode(sample_kv_block.chunk_text)
        expected_len = len(prefix_tokens) + len(query_tokens)
        assert len(captured_tokens) == expected_len

    def test_consistent_block_order(self, mock_adapter, sample_kv_block):
        """Same blocks in same order should produce same output."""
        tokens = mock_adapter.tokenizer.encode("consistent test")
        out1 = inject_and_generate(mock_adapter, [sample_kv_block], tokens, {})
        out2 = inject_and_generate(mock_adapter, [sample_kv_block], tokens, {})
        assert out1.sequences[0] == out2.sequences[0]


class TestWriteReadInjectCycle:
    """
    End-to-end write -> retrieve (mocked) -> inject cycle.
    """

    def test_write_then_fetch_then_inject(self, tmp_path, mock_adapter, base_config, sample_kv_block):
        """Full cycle: write to blob store, fetch, inject."""
        store = KVStore(blob_store_path=str(tmp_path))
        store.write(sample_kv_block)

        # Fetch
        blocks = store.fetch([sample_kv_block.block_id], model_id=sample_kv_block.model_id)
        assert len(blocks) == 1

        # Inject
        tokens = mock_adapter.tokenizer.encode("What is the capital of France?")
        out = inject_and_generate(mock_adapter, blocks, tokens, {})
        assert isinstance(out, GenerationOutput)
        assert out.text is not None

    def test_multiple_blocks_inject(self, tmp_path, mock_adapter, base_config, sample_kv_block):
        """Multiple blocks should all be injected."""
        # Create second block with different text
        text2 = "Berlin is the capital of Germany and it is large."
        tokens2 = mock_adapter.tokenizer.encode(text2)
        kv2, h2 = mock_adapter.capture(tokens=tokens2, text=text2, layers=base_config.store_layers)

        from kvmemory.core.retrieval import compute_retrieval_vec

        hidden_vecs2 = {l: compute_retrieval_vec(h, len(tokens2)) for l, h in h2.items()}
        kv_q2, scales2 = {}, {}
        for l, (K, V) in kv2.items():
            Kq, ks = quantize_int8(K.float())
            Vq, vs = quantize_int8(V.float())
            kv_q2[l] = (Kq, Vq)
            scales2[l] = (ks, vs)

        block2 = KVBlock.new(
            model_id=base_config.model_id,
            session_id="test-session",
            chunk_text=text2,
            token_count=len(tokens2),
            hidden_vecs=hidden_vecs2,
            kv_by_layer=kv_q2,
            quant_scales=scales2,
            original_positions=list(range(len(tokens2))),
        )

        store = KVStore(blob_store_path=str(tmp_path))
        store.write(sample_kv_block)
        store.write(block2)

        blocks = store.fetch(
            [sample_kv_block.block_id, block2.block_id],
            model_id=base_config.model_id,
        )
        assert len(blocks) == 2

        tokens = mock_adapter.tokenizer.encode("capitals quiz")
        out = inject_and_generate(mock_adapter, blocks, tokens, {})
        assert out is not None


class TestImportanceFilter:
    def test_score_importance_perplexity_only(self):
        from kvmemory.core.importance import score_importance

        # High loss relative to baseline -> high score
        high = score_importance("text", model_loss=5.0, baseline_loss=2.0)
        low = score_importance("text", model_loss=2.0, baseline_loss=2.0)
        assert high > low

    def test_explicit_signal_boosts_score(self):
        from kvmemory.core.importance import score_importance

        without_signal = score_importance("text", model_loss=2.0, baseline_loss=2.0)
        with_signal = score_importance("text", model_loss=2.0, baseline_loss=2.0, explicit_signal=1.0)
        assert with_signal > without_signal

    def test_score_bounded_0_1(self):
        from kvmemory.core.importance import score_importance

        for loss, baseline in [(0.1, 2.0), (100.0, 2.0), (2.0, 2.0)]:
            s = score_importance("text", model_loss=loss, baseline_loss=baseline, explicit_signal=0.5)
            assert 0.0 <= s <= 1.0

    def test_zero_baseline_handled(self):
        from kvmemory.core.importance import score_importance

        s = score_importance("text", model_loss=3.0, baseline_loss=0.0)
        assert 0.0 <= s <= 1.0  # No crash, returns bounded value

    def test_baseline_tracker_updates(self):
        from kvmemory.core.importance import BaselineLossTracker

        tracker = BaselineLossTracker(initial=2.0, alpha=0.5)
        tracker.update(4.0)
        assert tracker.value == pytest.approx(2.0 * 0.5 + 4.0 * 0.5)

    def test_baseline_tracker_reset(self):
        from kvmemory.core.importance import BaselineLossTracker

        tracker = BaselineLossTracker(initial=2.0)
        tracker.update(10.0)
        tracker.reset()
        assert tracker.value == pytest.approx(2.0)


class TestPrefillReduction:
    """
    Test 2 from plan.md: verify injected KV tokens do NOT hit prefill.

    When KV blocks are injected via past_key_values, the model's forward
    call should receive only the query tokens as input_ids — not the stored
    context tokens. The injected tokens are already materialised as KV and
    bypass the prefill computation entirely.
    """

    def test_prefill_only_receives_query_tokens(self, mock_adapter, sample_kv_block):
        """
        Injected block tokens must NOT appear in the input_ids sent to the model.
        Only current_tokens hit prefill.
        """
        received_input_ids = []

        original_inject = mock_adapter.inject_and_generate

        def capturing_inject(blocks, tokens, kwargs):
            # Record what tokens were passed as the "current" query
            received_input_ids.extend(tokens)
            return original_inject(blocks, tokens, kwargs)

        mock_adapter.inject_and_generate = capturing_inject

        query = "What is the capital of France?"
        query_tokens = mock_adapter.tokenizer.encode(query)
        stored_token_count = sample_kv_block.token_count

        inject_and_generate(mock_adapter, [sample_kv_block], query_tokens, {})

        # The adapter must receive exactly the query tokens, not query + stored tokens
        assert len(received_input_ids) == len(query_tokens), (
            f"Prefill received {len(received_input_ids)} tokens but query is only "
            f"{len(query_tokens)} tokens. Stored block has {stored_token_count} tokens "
            "that should be injected via past_key_values, not re-prefilled."
        )
        assert received_input_ids == query_tokens, (
            "input_ids passed to adapter do not match query tokens exactly."
        )

    def test_prefill_count_less_than_total_context(self, mock_adapter, sample_kv_block):
        """
        Total context = query + stored. Prefill must be strictly less than total.
        This is the core cost saving claim of the system.
        """
        received_input_ids = []
        original_inject = mock_adapter.inject_and_generate

        def capturing_inject(blocks, tokens, kwargs):
            received_input_ids.extend(tokens)
            return original_inject(blocks, tokens, kwargs)

        mock_adapter.inject_and_generate = capturing_inject

        query_tokens = mock_adapter.tokenizer.encode("test query for prefill check")
        inject_and_generate(mock_adapter, [sample_kv_block], query_tokens, {})

        prefill_count = len(received_input_ids)
        total_context = len(query_tokens) + sample_kv_block.token_count

        assert prefill_count < total_context, (
            f"Prefill count ({prefill_count}) should be less than total context "
            f"({total_context} = {len(query_tokens)} query + {sample_kv_block.token_count} stored). "
            "KV injection is supposed to skip prefill for stored tokens."
        )
