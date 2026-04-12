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
import logging
import time
from typing import Optional

from .adapters.base import BaseAdapter
from .config import KVMemoryConfig
from .core.injector import inject_and_generate
from .core.queue import WriteQueue
from .core.retrieval import compute_query_vecs, stage1_coarse, stage2_rerank_mmr
from .core.write_pipeline import run_write_pipeline
from .observability import RunObserver
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

    def __init__(
        self,
        adapter: BaseAdapter,
        config: KVMemoryConfig,
        observer: Optional[RunObserver] = None,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.observer = observer
        # Auto-compute retrieval_layers as 25/50/75% of model depth if not set
        if not self.config.retrieval_layers:
            n = adapter.num_layers
            self.config.retrieval_layers = [n // 4, n // 2, (3 * n) // 4]

        if not self.config.store_layers:
            self.config.store_layers = list(range(adapter.num_layers))

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
        trace_context: Optional[dict] = None,
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
        observer = None
        if self.observer:
            observer = self.observer.child(
                session_id=session_id,
                agent_id=agent_id,
                retrieve_shared=retrieve_shared,
                **(trace_context or {}),
            )

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

        stage1_start = time.perf_counter()
        candidate_ids, query_vecs = await stage1_coarse(
            query_tokens=tokens,
            adapter=self.adapter,
            config=self.config,
            vector_db=self.vector_db,
            session_filter=session_filter,
        )
        stage1_ms = (time.perf_counter() - stage1_start) * 1000
        if observer:
            observer.emit(
                "retrieval_stage1_done",
                phase=(trace_context or {}).get("phase", "generate"),
                duration_ms=stage1_ms,
                candidate_count=len(candidate_ids),
                query_token_count=len(tokens),
            )

        stage2_start = time.perf_counter()
        final_ids = stage2_rerank_mmr(
            candidate_ids=candidate_ids,
            query_vecs=query_vecs,
            config=self.config,
            vector_db=self.vector_db,
            token_budget=self.config.token_budget,
            min_relevance=self.config.min_relevance,
        )
        stage2_ms = (time.perf_counter() - stage2_start) * 1000
        if observer:
            observer.emit(
                "retrieval_stage2_done",
                phase=(trace_context or {}).get("phase", "generate"),
                duration_ms=stage2_ms,
                selected_count=len(final_ids),
                selected_ids=final_ids,
            )

        fetch_start = time.perf_counter()
        blocks = self.kv_store.fetch(final_ids, model_id=self.config.model_id)
        fetch_ms = (time.perf_counter() - fetch_start) * 1000
        logger.info(
            "generate: session=%s, retrieved=%d blocks, tokens_in_context=%d",
            session_id,
            len(blocks),
            sum(b.token_count for b in blocks),
        )
        if observer:
            observer.emit(
                "kv_fetch_done",
                phase=(trace_context or {}).get("phase", "generate"),
                duration_ms=fetch_ms,
                block_count=len(blocks),
                token_count=sum(b.token_count for b in blocks),
                block_ids=final_ids,
            )
        if final_ids:
            asyncio.ensure_future(self._increment_access_counts(final_ids))

        # --- GENERATE ---
        generation_start = time.perf_counter()
        output = inject_and_generate(
            adapter=self.adapter,
            blocks=blocks,
            current_tokens=tokens,
            generation_kwargs=generation_kwargs,
        )
        generation_ms = (time.perf_counter() - generation_start) * 1000
        if observer:
            observer.emit(
                "generation_done",
                phase=(trace_context or {}).get("phase", "generate"),
                duration_ms=generation_ms,
                output_chars=len(output.text),
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
            agent_id=agent_id,
            shared=False,
            explicit_signal=explicit_signal,
            observer=observer,
            trace_context=trace_context,
        )
        if observer:
            observer.emit(
                "write_enqueued",
                phase=(trace_context or {}).get("phase", "generate"),
                token_count=len(tokens),
                queue_depth=self._write_queue.pending,
            )

        return output

    async def store(
        self,
        text: str,
        session_id: str,
        agent_id: Optional[str] = None,
        shared: bool = False,
        explicit_signal: float = 1.0,
        trace_context: Optional[dict] = None,
        dedup_mode: str = "semantic",
    ) -> None:
        """
        Manually store text into memory without generating a response.

        Useful for seeding memory with known facts or previous context.
        explicit_signal=1.0 bypasses the importance filter.

        Args:
            dedup_mode: "semantic" (default) or "hash". Pass "hash" for bulk
                        ingestion of unique-per-source text (e.g. BEAM contexts)
                        to avoid semantic collapse. Reset with reset_hash_dedup()
                        after drain_writes() completes.
        """
        tokens = self.adapter.tokenizer.encode(text)
        observer = None
        if self.observer:
            observer = self.observer.child(
                session_id=session_id,
                agent_id=agent_id,
                shared=shared,
                **(trace_context or {}),
            )
        await self._write_queue.enqueue(
            session_id=session_id,
            tokens=tokens,
            text=text,
            adapter=self.adapter,
            config=self.config,
            agent_id=agent_id,
            shared=shared,
            explicit_signal=explicit_signal,
            observer=observer,
            trace_context=trace_context,
            dedup_mode=dedup_mode,
        )
        if observer:
            observer.emit(
                "write_enqueued",
                phase=(trace_context or {}).get("phase", "store"),
                token_count=len(tokens),
                queue_depth=self._write_queue.pending,
            )

    def reset_baseline(self, session_id: str) -> None:
        """Remove the baseline loss tracker for a session.

        Call this before storing each new document to prevent the importance
        scorer from treating the first document's loss distribution as the
        baseline for all subsequent documents.
        """
        from .core.write_pipeline import _baseline_trackers
        _baseline_trackers.pop(session_id, None)

    async def drain_writes(self, timeout: float | None = None) -> None:
        """
        Wait until all queued writes have landed in Qdrant/blob store.

        Does NOT stop the write queue — the memory object remains fully usable
        after this call. Use close() to fully shut down.

        Args:
            timeout: max seconds to wait (None = wait forever)
        """
        await self._write_queue.drain(timeout=timeout)

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
        agent_id: Optional[str] = None,
        shared: bool = False,
        explicit_signal: float = 0.0,
        observer: Optional[RunObserver] = None,
        trace_context: Optional[dict] = None,
        dedup_mode: str = "semantic",
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
            agent_id=agent_id,
            shared=shared,
            explicit_signal=explicit_signal,
            observer=observer,
            trace_context=trace_context,
            dedup_mode=dedup_mode,
        )

    async def _increment_access_counts(self, block_ids: list[str]) -> None:
        """Fire-and-forget: increment access counts in vector DB after retrieval."""
        try:
            self.vector_db.increment_access_count(self.config.model_id, block_ids)
        except Exception as e:
            logger.debug("Failed to increment access counts in vector DB: %s", e)
