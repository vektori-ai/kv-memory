"""
queue.py — Async write queue.

The write path runs asynchronously after every response.
Users see zero added latency from storage operations.

If the process crashes before a write completes, that turn's memory
is lost. This is an acceptable tradeoff per the spec.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from ..adapters.base import BaseAdapter
    from ..config import KVMemoryConfig
    from ..observability import RunObserver

logger = logging.getLogger(__name__)


class WriteQueue:
    """
    Single-worker async write queue.

    Enqueue write tasks immediately after a response is generated.
    The background worker processes them sequentially, never blocking
    the user-facing request path.

    Usage:
        queue = WriteQueue(write_fn)
        queue.start()
        await queue.enqueue(session_id, tokens, text, adapter, config)
        # ... later ...
        await queue.shutdown()
    """

    def __init__(self, write_fn: Callable) -> None:
        """
        Args:
            write_fn: async callable with signature
                      async def write_fn(session_id, tokens, text, adapter, config,
                                         agent_id, shared, explicit_signal) -> None
        """
        self._write_fn = write_fn
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    def start(self) -> None:
        """Start the background worker. Must be called inside an async context."""
        if self._running:
            return
        self._running = True
        self._worker_task = asyncio.ensure_future(self._process())
        logger.debug("WriteQueue started")

    async def enqueue(
        self,
        session_id: str,
        tokens: list[int],
        text: str,
        adapter: "BaseAdapter",
        config: "KVMemoryConfig",
        agent_id: Optional[str] = None,
        shared: bool = False,
        explicit_signal: float = 0.0,
        observer: Optional["RunObserver"] = None,
        trace_context: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Enqueue a write task. Returns immediately — never blocks.

        Args:
            session_id:      current session
            tokens:          full turn token IDs
            text:            full turn text (prompt + response)
            adapter:         framework adapter for capture()
            config:          KVMemoryConfig
            agent_id:        optional agent identifier
            shared:          whether to allow cross-agent retrieval of this turn
            explicit_signal: 1.0 if user flagged 'remember this'
        """
        if not self._running:
            self.start()
        await self._queue.put(
            (
                session_id,
                tokens,
                text,
                adapter,
                config,
                agent_id,
                shared,
                explicit_signal,
                observer,
                trace_context,
            )
        )
        logger.debug("WriteQueue.enqueue: session=%s, tokens=%d", session_id, len(tokens))

    async def shutdown(self, timeout: float = 10.0) -> None:
        """
        Drain the queue and stop the worker.

        Waits up to `timeout` seconds for pending writes to complete.
        """
        if not self._running:
            return
        self._running = False
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("WriteQueue.shutdown: timed out with %d items remaining", self._queue.qsize())
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.debug("WriteQueue stopped")

    @property
    def pending(self) -> int:
        """Number of items waiting in the queue."""
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _process(self) -> None:
        """
        Background worker loop.

        Processes items from the queue indefinitely until shutdown.
        Errors in individual writes are logged but never crash the process.
        """
        while True:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                break

            (
                session_id,
                tokens,
                text,
                adapter,
                config,
                agent_id,
                shared,
                explicit_signal,
                observer,
                trace_context,
            ) = item
            try:
                await self._write_fn(
                    session_id,
                    tokens,
                    text,
                    adapter,
                    config,
                    agent_id,
                    shared,
                    explicit_signal,
                    observer,
                    trace_context,
                )
            except Exception as e:
                logger.error(
                    "WriteQueue._process: write failed for session=%s: %s",
                    session_id,
                    e,
                    exc_info=True,
                )
            finally:
                self._queue.task_done()
