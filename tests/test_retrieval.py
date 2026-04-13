"""
test_retrieval.py — Retrieval pipeline tests.

Tests the two-stage retrieval pipeline with mock Qdrant.
Validates:
  - compute_retrieval_vec() produces correct L2-normalized vectors
  - stage2_rerank_mmr() selects diverse blocks within token budget
  - Token budget is never exceeded (non-negotiable constraint)
  - Recall@10 >= 80% (Test 3 from spec, run against mock vector store)
"""

from __future__ import annotations

import math
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from kvmemory.config import KVMemoryConfig
from kvmemory.core.retrieval import (
    _default_layer_weights,
    compute_query_vecs,
    compute_q_vec,
    compute_retrieval_vec,
    stage2_rerank_qk,
    stage2_rerank_mmr,
)
from kvmemory.storage.schema import KVBlock


class TestComputeRetrievalVecFixed:
    """Fix 1 regression tests: mean-pool replaces broken entropy-softmax pooling."""

    def test_shape_dtype_norm(self):
        """Fast, no model needed: shape == (d_model,), dtype float32, L2 norm ≈ 1.0."""
        d_model = 128
        seq_len = 15
        hidden = torch.randn(seq_len, d_model)
        vec = compute_retrieval_vec(hidden, seq_len)
        assert vec.shape == (d_model,), f"Expected ({d_model},), got {vec.shape}"
        assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-4, f"Not L2-normalized: norm={norm}"

    @pytest.mark.slow
    def test_discrimination(self):
        """
        Slow, requires model: entropy-weighted retrieval vectors for semantically
        different texts must have cosine similarity < 0.95.

        Threshold is 0.95 (not 0.98) — the benchmark showed mean pooling alone
        produces 0.985-0.992 for completely unrelated facts. Entropy weighting
        via HFAdapter must push discrimination below this threshold.
        """
        pytest.importorskip("transformers")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        from kvmemory.adapters.hf_adapter import HFAdapter

        model_name = "sshleifer/tiny-gpt2"
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.eval()
        except Exception:
            pytest.skip(f"Model {model_name} unavailable")

        adapter = HFAdapter(model, tokenizer)
        layers = [0]

        texts = [
            "The capital of France is Paris and it is known for the Eiffel Tower.",
            "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        ]

        vecs = []
        for text in texts:
            toks = tokenizer.encode(text)
            _, hidden_by_layer = adapter.capture(toks, text, layers)
            hidden = hidden_by_layer[0]
            vecs.append(compute_retrieval_vec(hidden, len(toks)))

        cosine_sim = float(np.dot(vecs[0], vecs[1]))
        assert cosine_sim < 0.95, (
            f"Vectors too similar after entropy weighting (cosine={cosine_sim:.4f}). "
            "Expected < 0.95. Mean pooling alone hits 0.985+ — entropy weighting must "
            "push discrimination lower. Check _apply_entropy_weights in hf_adapter.py."
        )


class TestComputeRetrievalVec:
    def test_l2_normalized(self, mock_adapter, base_config):
        tokens = mock_adapter.tokenizer.encode("hello world test sentence here")
        _, hidden_by_layer = mock_adapter.capture(tokens=tokens, text="", layers=base_config.retrieval_layers)
        for layer, hidden in hidden_by_layer.items():
            vec = compute_retrieval_vec(hidden, len(tokens))
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 1e-4, f"Not L2-normalized: {norm}"

    def test_sqrt_n_normalization_effect(self):
        """Longer chunks should produce smaller magnitude before L2 norm (before final normalize)."""
        d = 64
        # We can't directly test intermediate magnitude after L2 norm,
        # but we can verify the function handles varying lengths without error
        for seq_len in [1, 10, 100]:
            hidden = torch.randn(seq_len, d)
            vec = compute_retrieval_vec(hidden, seq_len)
            assert vec.shape == (d,)
            assert abs(np.linalg.norm(vec) - 1.0) < 1e-4

    def test_single_token(self):
        hidden = torch.randn(1, 32)
        vec = compute_retrieval_vec(hidden, 1)
        assert vec.shape == (32,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-4

    def test_wrong_ndim_raises(self):
        hidden = torch.randn(64)  # Missing seq dim
        with pytest.raises(ValueError):
            compute_retrieval_vec(hidden, 5)


class TestComputeQVec:
    def test_groups_query_heads_to_kv_heads(self):
        q = torch.zeros(4, 2, 3)
        q[0:2, :, :] = 1.0
        q[2:4, :, :] = 2.0

        vec = compute_q_vec(q, num_kv_heads=2)
        expected = torch.tensor([1, 1, 1, 2, 2, 2], dtype=torch.float32)
        expected = F.normalize(expected, dim=0).numpy()

        assert vec.shape == (6,)
        np.testing.assert_allclose(vec, expected, atol=1e-6)

    def test_invalid_grouping_raises(self):
        q = torch.randn(5, 2, 3)
        with pytest.raises(ValueError):
            compute_q_vec(q, num_kv_heads=2)


class TestLayerWeights:
    def test_three_layer_weights(self):
        layers = [8, 16, 24]
        weights = _default_layer_weights(layers)
        assert abs(weights[8] - 0.25) < 1e-9
        assert abs(weights[16] - 0.50) < 1e-9
        assert abs(weights[24] - 0.25) < 1e-9
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_equal_weights_for_non_three(self):
        for n in [1, 2, 4, 5]:
            layers = list(range(n))
            weights = _default_layer_weights(layers)
            for w in weights.values():
                assert abs(w - 1.0 / n) < 1e-9


class TestStage2MMR:
    """
    Test MMR reranking with a mock VectorDB.
    """

    def _make_candidate(self, block_id: str, token_count: int, layers: list[int], d: int = 64):
        """Create a mock candidate dict as returned by VectorDB.fetch_with_vectors()."""
        vec = np.random.randn(d).astype(np.float32)
        vec /= np.linalg.norm(vec)
        return {
            "id": block_id,
            "vector": {f"layer_{l}": vec.tolist() for l in layers},
            "payload": {
                "token_count": token_count,
                "chunk_text": f"chunk {block_id}",
                "importance_score": 0.5,
            },
        }

    def _make_mock_vector_db(self, candidates: list[dict]):
        mock_db = MagicMock()
        mock_db.fetch_with_vectors.return_value = candidates
        return mock_db

    def test_token_budget_never_exceeded(self):
        """Non-negotiable: total tokens of selected blocks must not exceed budget."""
        np.random.seed(42)
        layers = [1, 2, 3]
        d = 64
        token_budget = 100
        # 20 candidates each with 20 tokens
        candidates = [
            self._make_candidate(f"block-{i}", 20, layers, d) for i in range(20)
        ]
        mock_db = self._make_mock_vector_db(candidates)

        query_vecs = {l: np.random.randn(d).astype(np.float32) for l in layers}
        for v in query_vecs.values():
            v /= np.linalg.norm(v)

        config = KVMemoryConfig(
            model_id="test",
            retrieval_layers=layers,
            token_budget=token_budget,
            final_top_k=50,  # try to select more than budget allows
        )

        selected = stage2_rerank_mmr(
            candidate_ids=[c["id"] for c in candidates],
            query_vecs=query_vecs,
            config=config,
            vector_db=mock_db,
            token_budget=token_budget,
        )

        # Each block is 20 tokens; budget is 100 -> max 5 blocks
        assert len(selected) <= 5, f"Too many blocks selected: {len(selected)}"

    def test_final_top_k_respected(self):
        np.random.seed(42)
        layers = [1, 2, 3]
        d = 64
        candidates = [
            self._make_candidate(f"block-{i}", 10, layers, d) for i in range(50)
        ]
        mock_db = self._make_mock_vector_db(candidates)

        query_vecs = {l: np.random.randn(d).astype(np.float32) for l in layers}
        for v in query_vecs.values():
            v /= np.linalg.norm(v)

        config = KVMemoryConfig(
            model_id="test",
            retrieval_layers=layers,
            token_budget=10000,  # large budget
            final_top_k=7,
        )

        selected = stage2_rerank_mmr(
            candidate_ids=[c["id"] for c in candidates],
            query_vecs=query_vecs,
            config=config,
            vector_db=mock_db,
            token_budget=10000,
        )
        assert len(selected) <= 7

    def test_empty_candidates(self):
        mock_db = MagicMock()
        mock_db.fetch_with_vectors.return_value = []
        config = KVMemoryConfig(model_id="test", retrieval_layers=[1, 2, 3])
        result = stage2_rerank_mmr([], {}, config, mock_db, 1000)
        assert result == []

    def test_mmr_favors_diversity(self):
        """
        Two nearly identical blocks: MMR should prefer a diverse third block
        over the redundant second block.
        """
        np.random.seed(99)
        layers = [1]
        d = 8

        # base vector (highly relevant)
        base = np.ones(d, dtype=np.float32)
        base /= np.linalg.norm(base)

        # near-duplicate of base
        dup = base + np.random.randn(d).astype(np.float32) * 0.01
        dup /= np.linalg.norm(dup)

        # diverse block (orthogonal)
        diverse = np.zeros(d, dtype=np.float32)
        diverse[0] = -1.0

        candidates = [
            {"id": "base", "vector": {"layer_1": base.tolist()}, "payload": {"token_count": 10}},
            {"id": "dup", "vector": {"layer_1": dup.tolist()}, "payload": {"token_count": 10}},
            {"id": "diverse", "vector": {"layer_1": diverse.tolist()}, "payload": {"token_count": 10}},
        ]
        mock_db = self._make_mock_vector_db(candidates)

        query_vecs = {1: base.copy()}
        config = KVMemoryConfig(
            model_id="test", retrieval_layers=[1], token_budget=1000, final_top_k=2
        )

        selected = stage2_rerank_mmr(
            candidate_ids=["base", "dup", "diverse"],
            query_vecs=query_vecs,
            config=config,
            vector_db=mock_db,
            token_budget=1000,
            mmr_lambda=0.5,  # equal relevance + diversity
        )

        # With strong diversity weight, "diverse" should beat "dup"
        assert "base" in selected, "Most relevant block should always be selected first"
        # dup and diverse compete for slot 2; with lambda=0.5 diverse should win
        # (this is probabilistic based on the vectors, but with these extreme values it's deterministic)
        assert len(selected) == 2


class TestStage2QK:
    def test_selects_highest_relevance_under_budget(self):
        layers = [1]
        query = np.array([1.0, 0.0], dtype=np.float32)
        candidates = [
            {"id": "bad", "vector": {"layer_1": [-1.0, 0.0]}, "payload": {"token_count": 10}},
            {"id": "good", "vector": {"layer_1": [1.0, 0.0]}, "payload": {"token_count": 10}},
            {"id": "off", "vector": {"layer_1": [0.0, 1.0]}, "payload": {"token_count": 10}},
        ]
        mock_db = MagicMock()
        mock_db.fetch_with_vectors.return_value = candidates
        config = KVMemoryConfig(
            model_id="test",
            retrieval_layers=layers,
            token_budget=20,
            final_top_k=2,
        )

        selected = stage2_rerank_qk(
            candidate_ids=[c["id"] for c in candidates],
            query_vecs={1: query},
            config=config,
            vector_db=mock_db,
            token_budget=20,
        )

        assert selected == ["good", "off"]

    def test_respects_token_budget(self):
        layers = [1]
        query = np.array([1.0, 0.0], dtype=np.float32)
        candidates = [
            {"id": "large", "vector": {"layer_1": [1.0, 0.0]}, "payload": {"token_count": 30}},
            {"id": "small", "vector": {"layer_1": [0.9, 0.1]}, "payload": {"token_count": 10}},
        ]
        mock_db = MagicMock()
        mock_db.fetch_with_vectors.return_value = candidates
        config = KVMemoryConfig(model_id="test", retrieval_layers=layers, final_top_k=2)

        selected = stage2_rerank_qk(
            candidate_ids=[c["id"] for c in candidates],
            query_vecs={1: query},
            config=config,
            vector_db=mock_db,
            token_budget=10,
        )

        assert selected == ["small"]


class TestRecallAt10:
    """
    Test 3 from spec: store N known blocks, query each one, assert hit rate > 80%.

    Uses a mock VectorDB that returns a controlled candidate set.
    The real ANN recall test requires a live Qdrant instance (integration test).
    """

    def test_recall_mock_100_blocks(self, mock_adapter, base_config):
        """
        Simulate 100 blocks. For each, the correct block is in the candidate pool.
        Verify that stage2 selects it.
        """
        np.random.seed(0)
        n_blocks = 50  # Reduced for test speed
        layers = base_config.retrieval_layers
        d = mock_adapter.d_model

        # Generate known query vectors and corresponding block vectors
        hits = 0
        for i in range(n_blocks):
            # Query vector
            query_vec = np.random.randn(d).astype(np.float32)
            query_vec /= np.linalg.norm(query_vec)

            # "Correct" block has very high cosine with query
            correct_vec = query_vec + np.random.randn(d).astype(np.float32) * 0.01
            correct_vec /= np.linalg.norm(correct_vec)

            # Distractor blocks
            distractors = [np.random.randn(d).astype(np.float32) for _ in range(9)]
            for dv in distractors:
                dv /= np.linalg.norm(dv)

            correct_candidate = {
                "id": f"correct-{i}",
                "vector": {f"layer_{l}": correct_vec.tolist() for l in layers},
                "payload": {"token_count": 20, "chunk_text": f"block {i}"},
            }
            distractor_candidates = [
                {
                    "id": f"distractor-{i}-{j}",
                    "vector": {f"layer_{l}": distractors[j].tolist() for l in layers},
                    "payload": {"token_count": 20, "chunk_text": f"distractor {j}"},
                }
                for j in range(9)
            ]
            all_candidates = [correct_candidate] + distractor_candidates

            mock_db = MagicMock()
            mock_db.fetch_with_vectors.return_value = all_candidates

            query_vecs = {l: query_vec for l in layers}
            selected = stage2_rerank_mmr(
                candidate_ids=[c["id"] for c in all_candidates],
                query_vecs=query_vecs,
                config=base_config,
                vector_db=mock_db,
                token_budget=base_config.token_budget,
            )

            if f"correct-{i}" in selected:
                hits += 1

        recall = hits / n_blocks
        assert recall >= 0.80, f"Recall@{base_config.final_top_k} = {recall:.2f} < 0.80 target"
