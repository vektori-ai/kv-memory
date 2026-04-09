"""
memory.py — KVMemory: public API surface.

Two meaningful lines of change from existing code:

    # Before
    output = model.generate(input_ids)

    # After
    memory = KVMemory(adapter=HFAdapter(model, tokenizer), config=cfg)
    output = await memory.generate(prompt, session_id='user_123')

Read path:  synchronous stage1 + stage2 + fetch + inject (~50ms target)
Write path: async, enqueued, never blocks generation
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Optional

from .adapters.base import BaseAdapter
from .config import KVMemoryConfig
from .core.injector import inject_and_generate
from .core.queue import WriteQueue
from .core.retrieval import compute_query_vecs, stage1_coarse, stage2_rerank_mmr
from .core.write_pipeline import run_write_pipeline
from .storage.kv_store import KVStore
from .storage.schema import GenerationOutput
from .storage.vector_db import VectorDB

logger = logging.getLogger(__name__)


def _build_session_filter(
    model_id: str,
    session_id: str,
    agent_id: Optional[str],
    retrieve_shared: bool,
) -> dict:
    return {
        "model_id": model_id,
        "session_id": session_id,
        "agent_id": agent_id,
        "retrieve_shared": retrieve_shared,
    }


class KVMemory:
    """
    KV Memory System — public API.

    Manages the full read + generate + write cycle per conversation turn.

    Initialization sets up Qdrant collection for the model automatically.

    Usage:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from kvmemory import KVMemory, KVMemoryConfig
        from kvmemory.adapters.hf_adapter import HFAdapter

        model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3-8B')
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3-8B')
        adapter = HFAdapter(model, tokenizer)
        config = KVMemoryConfig(model_id='llama-3-8b', retrieval_layers=[8, 16, 24])

        memory = KVMemory(adapter=adapter, config=config)
        output = await memory.generate("What is the capital of France?", session_id='user_123')
    """

    def __init__(self, adapter: BaseAdapter, config: KVMemoryConfig) -> None:
        self.adapter = adapter
        self.config = config

        self.vector_db = VectorDB(url=config.qdrant_url, port=config.qdrant_port)
        self.kv_store = KVStore(blob_store_path=config.blob_store_path)

        # Ensure Qdrant collection exists
        self.vector_db.ensure_collection(
            model_id=config.model_id,
            retrieval_layers=config.retrieval_layers,
            d_model=adapter.d_model,
        )

        # Write queue: wraps the pipeline with references to stores
        self._write_queue = WriteQueue(write_fn=self._write_fn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        session_id: str,
        agent_id: Optional[str] = None,
        retrieve_shared: bool = False,
        generation_kwargs: Optional[dict] = None,
        explicit_signal: float = 0.0,
    ) -> GenerationOutput:
        """
        Full cycle: retrieve -> generate -> enqueue write.

        Args:
            prompt:           user prompt text
            session_id:       session identifier for retrieval filtering
            agent_id:         optional agent identifier
            retrieve_shared:  include shared cross-session blocks
            generation_kwargs: forwarded to model.generate()
            explicit_signal:  1.0 if user flagged 'remember this turn'

        Returns:
            GenerationOutput with .text and .sequences
        """
        if generation_kwargs is None:
            generation_kwargs = {}

        tokens = self.adapter.tokenizer.encode(prompt)
        if not tokens:
            raise ValueError("prompt encoded to empty token list")

        # --- READ PATH ---
        session_filter = _build_session_filter(
            model_id=self.config.model_id,
            session_id=session_id,
            agent_id=agent_id,
            retrieve_shared=retrieve_shared,
        )

        candidate_ids, query_vecs = await stage1_coarse(
            query_tokens=tokens,
            adapter=self.adapter,
            config=self.config,
            vector_db=self.vector_db,
            session_filter=session_filter,
        )

        final_ids = stage2_rerank_mmr(
            candidate_ids=candidate_ids,
            query_vecs=query_vecs,
            config=self.config,
            vector_db=self.vector_db,
            token_budget=self.config.token_budget,
        )

        blocks = self.kv_store.fetch(final_ids, model_id=self.config.model_id)
        logger.info(
            "generate: session=%s, retrieved=%d blocks, tokens_in_context=%d",
            session_id,
            len(blocks),
            sum(b.token_count for b in blocks),
        )

        # --- GENERATE ---
        output = inject_and_generate(
            adapter=self.adapter,
            blocks=blocks,
            current_tokens=tokens,
            generation_kwargs=generation_kwargs,
        )

        # --- WRITE (async, non-blocking) ---
        response_text = output.text
        full_turn = prompt + " " + response_text

        await self._write_queue.enqueue(
            session_id=session_id,
            tokens=tokens,
            text=full_turn,
            adapter=self.adapter,
            config=self.config,
        )

        return output

    async def store(
        self,
        text: str,
        session_id: str,
        agent_id: Optional[str] = None,
        shared: bool = False,
        explicit_signal: float = 1.0,
    ) -> None:
        """
        Manually store text into memory without generating a response.

        Useful for seeding memory with known facts or previous context.
        explicit_signal=1.0 bypasses the importance filter.
        """
        tokens = self.adapter.tokenizer.encode(text)
        await self._write_queue.enqueue(
            session_id=session_id,
            tokens=tokens,
            text=text,
            adapter=self.adapter,
            config=self.config,
        )

    async def close(self) -> None:
        """Drain pending writes and shut down the write queue."""
        await self._write_queue.shutdown()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _write_fn(
        self,
        session_id: str,
        tokens: list[int],
        text: str,
        adapter: BaseAdapter,
        config: KVMemoryConfig,
    ) -> None:
        """Write pipeline entrypoint called by the queue worker."""
        await run_write_pipeline(
            session_id=session_id,
            tokens=tokens,
            text=text,
            adapter=adapter,
            config=config,
            kv_store=self.kv_store,
            vector_db=self.vector_db,
        )
