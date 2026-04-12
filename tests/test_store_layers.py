"""
Regression tests for store_layers defaults and retrieval-vector filtering.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from kvmemory.config import KVMemoryConfig
from kvmemory.core.write_pipeline import run_write_pipeline


class TestStoreLayersDefault:
    def test_empty_store_layers_resolves_to_all_adapter_layers(self, mock_adapter, monkeypatch):
        from kvmemory import memory as memory_module

        vector_db_cls = MagicMock()
        kv_store_cls = MagicMock()
        monkeypatch.setattr(memory_module, "VectorDB", vector_db_cls)
        monkeypatch.setattr(memory_module, "KVStore", kv_store_cls)

        config = KVMemoryConfig(
            model_id="test-model",
            retrieval_layers=[1, 3],
            store_layers=[],
        )

        memory = memory_module.KVMemory(adapter=mock_adapter, config=config)

        assert memory.config.store_layers == list(range(mock_adapter.num_layers))
        assert memory.config.retrieval_layers == [1, 3]
        expected_vec_dim = mock_adapter.num_kv_heads * mock_adapter.head_dim
        vector_db_cls.return_value.ensure_collection.assert_called_once_with(
            model_id="test-model",
            retrieval_layers=[1, 3],
            vec_dim=expected_vec_dim,
        )

    def test_explicit_store_layers_are_preserved(self, mock_adapter, monkeypatch):
        from kvmemory import memory as memory_module

        vector_db_cls = MagicMock()
        kv_store_cls = MagicMock()
        monkeypatch.setattr(memory_module, "VectorDB", vector_db_cls)
        monkeypatch.setattr(memory_module, "KVStore", kv_store_cls)

        config = KVMemoryConfig(
            model_id="test-model",
            retrieval_layers=[1, 3],
            store_layers=[1, 3],
        )

        memory = memory_module.KVMemory(adapter=mock_adapter, config=config)

        assert memory.config.store_layers == [1, 3]
        assert memory.config.retrieval_layers == [1, 3]


@pytest.mark.asyncio
async def test_write_pipeline_upserts_only_retrieval_layer_vectors(mock_adapter):
    config = KVMemoryConfig(
        model_id="test-model",
        retrieval_layers=[1, 3],
        store_layers=[0, 1, 2, 3],
        importance_threshold=0.0,
    )
    kv_store = MagicMock()
    vector_db = MagicMock()

    written = await run_write_pipeline(
        session_id="session-1",
        tokens=mock_adapter.tokenizer.encode("alpha beta gamma delta"),
        text="alpha beta gamma delta",
        adapter=mock_adapter,
        config=config,
        kv_store=kv_store,
        vector_db=vector_db,
        explicit_signal=1.0,
        dedup_mode="hash",
    )

    assert len(written) == 1
    block = kv_store.write.call_args.args[0]
    assert set(block.kv_by_layer) == {0, 1, 2, 3}
    assert set(block.hidden_vecs) == {1, 3}

    vector_db.upsert.assert_called_once()
    assert set(vector_db.upsert.call_args.kwargs["hidden_vecs"]) == {1, 3}
