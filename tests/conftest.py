"""
conftest.py — Shared pytest fixtures.

Provides a lightweight mock adapter and a tiny in-memory tokenizer
so tests can run without a GPU or a real HuggingFace model.
"""

from __future__ import annotations

import math
import uuid
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from kvmemory.adapters.base import BaseAdapter
from kvmemory.config import KVMemoryConfig
from kvmemory.core.write_pipeline import quantize_int8
from kvmemory.storage.schema import GenerationOutput, KVBlock


# ------------------------------------------------------------------
# Minimal tokenizer stub
# ------------------------------------------------------------------

class FakeTokenizer:
    """
    Minimal tokenizer for testing.
    Encodes by splitting on whitespace + mapping to IDs.
    """

    def __init__(self, vocab_size: int = 1000) -> None:
        self._vocab: dict[str, int] = {}
        self._vocab_size = vocab_size
        self._next_id = 1

    def encode(self, text: str) -> list[int]:
        tokens = []
        for word in text.split():
            if word not in self._vocab:
                if self._next_id < self._vocab_size:
                    self._vocab[word] = self._next_id
                    self._next_id += 1
                else:
                    # Unknown token maps to 0
                    self._vocab[word] = 0
            tokens.append(self._vocab[word])
        return tokens if tokens else [0]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return " ".join(inv.get(i, "<unk>") for i in ids)


# ------------------------------------------------------------------
# Mock adapter
# ------------------------------------------------------------------

class MockAdapter(BaseAdapter):
    """
    Deterministic mock adapter for unit tests.

    Produces predictable KV tensors and hidden states based on input tokens.
    Does NOT require a GPU or real model weights.
    """

    D_MODEL = 64
    NUM_LAYERS = 4
    NUM_HEADS = 4
    HEAD_DIM = D_MODEL // NUM_HEADS  # 16

    def __init__(self) -> None:
        self._tokenizer = FakeTokenizer()

    @property
    def d_model(self) -> int:
        return self.D_MODEL

    @property
    def num_layers(self) -> int:
        return self.NUM_LAYERS

    @property
    def num_heads(self) -> int:
        return self.NUM_HEADS

    @property
    def tokenizer(self):
        return self._tokenizer

    def supports_kv_inject(self) -> bool:
        return True

    def capture(
        self,
        tokens: list[int],
        text: str,
        layers: list[int],
    ):
        seq_len = len(tokens)
        # Deterministic tensors based on sum of tokens (reproducible)
        seed = sum(tokens) % 1000
        torch.manual_seed(seed)

        kv_by_layer = {}
        hidden_by_layer = {}
        for layer in layers:
            K = torch.randn(self.NUM_HEADS, seq_len, self.HEAD_DIM, dtype=torch.float16)
            V = torch.randn(self.NUM_HEADS, seq_len, self.HEAD_DIM, dtype=torch.float16)
            hidden = torch.randn(seq_len, self.D_MODEL, dtype=torch.float32)
            kv_by_layer[layer] = (K, V)
            hidden_by_layer[layer] = hidden

        return kv_by_layer, hidden_by_layer

    def inject_and_generate(
        self,
        blocks: list[KVBlock],
        current_tokens: list[int],
        generation_kwargs: dict,
    ) -> GenerationOutput:
        # Simulate that injection changes output — return different tokens when blocks present
        base_tokens = list(current_tokens)
        if blocks:
            # XOR with block count to simulate different output
            output_tokens = [t ^ len(blocks) for t in base_tokens]
        else:
            output_tokens = base_tokens

        text = self._tokenizer.decode(output_tokens)
        return GenerationOutput(sequences=[output_tokens], text=text)

    def generate(
        self,
        tokens: list[int],
        generation_kwargs: dict,
    ) -> GenerationOutput:
        return self.inject_and_generate([], tokens, generation_kwargs)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def mock_adapter() -> MockAdapter:
    return MockAdapter()


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def base_config() -> KVMemoryConfig:
    return KVMemoryConfig(
        model_id="test-model",
        retrieval_layers=[1, 2, 3],
        store_layers=[1, 2, 3],
        token_budget=500,
        coarse_top_k=20,
        final_top_k=5,
        importance_threshold=0.0,  # accept everything in tests
        dedup_threshold=0.95,
        blob_store_path="./test_kv_store",
    )


@pytest.fixture
def sample_kv_block(mock_adapter, base_config) -> KVBlock:
    """A real KVBlock built from mock adapter capture output."""
    text = "The capital of France is Paris and it is a beautiful city."
    tokens = mock_adapter.tokenizer.encode(text)

    kv_by_layer_float, hidden_by_layer = mock_adapter.capture(
        tokens=tokens,
        text=text,
        layers=base_config.store_layers,
    )

    from kvmemory.core.retrieval import compute_retrieval_vec
    from kvmemory.core.write_pipeline import quantize_int8

    hidden_vecs = {
        layer: compute_retrieval_vec(hidden, len(tokens))
        for layer, hidden in hidden_by_layer.items()
    }

    kv_by_layer_quant = {}
    quant_scales = {}
    for layer, (K, V) in kv_by_layer_float.items():
        K_q, k_scale = quantize_int8(K.float())
        V_q, v_scale = quantize_int8(V.float())
        kv_by_layer_quant[layer] = (K_q, V_q)
        quant_scales[layer] = (k_scale, v_scale)

    return KVBlock.new(
        model_id=base_config.model_id,
        session_id="test-session",
        chunk_text=text,
        token_count=len(tokens),
        hidden_vecs=hidden_vecs,
        kv_by_layer=kv_by_layer_quant,
        quant_scales=quant_scales,
        original_positions=list(range(len(tokens))),
        importance_score=0.8,
    )
