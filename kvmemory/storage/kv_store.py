"""
kv_store.py — Local blob store for KV tensors.

Stores KVBlock objects as pickle files on disk, organized by model_id.
Path format: {blob_store_path}/{model_id}/{block_id}.pkl

Phase 1: simple pickle. Phase 4 target: S3/Redis tiering.

Access metadata is updated asynchronously to avoid blocking the read path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pickle
import time
from typing import Optional

from ..storage.schema import KVBlock
from ..utils.model_id import sanitize_model_id

logger = logging.getLogger(__name__)


class KVStore:
    """
    Disk-backed blob store for KVBlock objects.

    Thread/async safety: reads are concurrent-safe (file read).
    Writes serialize per block_id via individual file creation.
    """

    def __init__(self, blob_store_path: str) -> None:
        self.blob_store_path = blob_store_path

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, block: KVBlock) -> None:
        """
        Persist a KVBlock to disk.

        Creates model_id subdirectory if needed.
        Raises on I/O error — caller should handle.
        """
        dir_path = self._block_dir(block.model_id)
        os.makedirs(dir_path, exist_ok=True)
        file_path = self._block_path(block.model_id, block.block_id)
        with open(file_path, "wb") as f:
            pickle.dump(block, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug("KVStore.write: saved %s (%d tokens)", block.block_id, block.token_count)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def fetch(self, block_ids: list[str], model_id: str) -> list[KVBlock]:
        """
        Load KVBlocks from disk. Skips missing blocks with a warning.

        Schedules async metadata update (fire-and-forget).
        """
        blocks: list[KVBlock] = []
        found_ids: list[str] = []

        for bid in block_ids:
            path = self._block_path(model_id, bid)
            if not os.path.exists(path):
                logger.warning("KVStore.fetch: block %s not found at %s", bid, path)
                continue
            with open(path, "rb") as f:
                block: KVBlock = pickle.load(f)
            blocks.append(block)
            found_ids.append(bid)

        # Fire-and-forget async metadata update
        if found_ids:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(self._update_access(found_ids, model_id))
                else:
                    # Synchronous fallback when not inside an async context
                    self._update_access_sync(found_ids, model_id)
            except RuntimeError:
                self._update_access_sync(found_ids, model_id)

        return blocks

    def fetch_one(self, block_id: str, model_id: str) -> Optional[KVBlock]:
        """Fetch a single block. Returns None if not found."""
        blocks = self.fetch([block_id], model_id)
        return blocks[0] if blocks else None

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(self, block_id: str, model_id: str) -> bool:
        """Remove a block from disk. Returns True if deleted."""
        path = self._block_path(model_id, block_id)
        if os.path.exists(path):
            os.remove(path)
            logger.debug("KVStore.delete: removed %s", block_id)
            return True
        return False

    def list_block_ids(self, model_id: str) -> list[str]:
        """Return all block_ids for a given model_id."""
        dir_path = self._block_dir(model_id)
        if not os.path.exists(dir_path):
            return []
        return [
            fname[:-4]  # strip .pkl
            for fname in os.listdir(dir_path)
            if fname.endswith(".pkl")
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _block_dir(self, model_id: str) -> str:
        return os.path.join(self.blob_store_path, sanitize_model_id(model_id))

    def _block_path(self, model_id: str, block_id: str) -> str:
        return os.path.join(self._block_dir(model_id), f"{block_id}.pkl")

    async def _update_access(self, block_ids: list[str], model_id: str) -> None:
        """
        Async: update last_accessed and access_count for fetched blocks.
        Fire-and-forget; errors are logged but never propagated.
        """
        now = time.time()
        for bid in block_ids:
            try:
                path = self._block_path(model_id, bid)
                if not os.path.exists(path):
                    continue
                with open(path, "rb") as f:
                    block: KVBlock = pickle.load(f)
                block.last_accessed = now
                block.access_count += 1
                with open(path, "wb") as f:
                    pickle.dump(block, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                logger.error("KVStore._update_access failed for %s: %s", bid, e)

    def _update_access_sync(self, block_ids: list[str], model_id: str) -> None:
        """Synchronous version of access metadata update."""
        now = time.time()
        for bid in block_ids:
            try:
                path = self._block_path(model_id, bid)
                if not os.path.exists(path):
                    continue
                with open(path, "rb") as f:
                    block: KVBlock = pickle.load(f)
                block.last_accessed = now
                block.access_count += 1
                with open(path, "wb") as f:
                    pickle.dump(block, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                logger.error("KVStore._update_access_sync failed for %s: %s", bid, e)
