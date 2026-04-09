"""
injector.py — KV injection with doc-wise RoPE.

Core mechanism:
  Each retrieved block is an independent document with positions [0..n].
  The current query gets positions offset by the number of blocks.
  No re-rotation needed — positions are always valid at injection time.

For adapters that don't support KV injection (vLLM):
  Fall back to text prefix injection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..adapters.base import BaseAdapter
    from ..storage.schema import GenerationOutput, KVBlock

logger = logging.getLogger(__name__)


def inject_and_generate(
    adapter: "BaseAdapter",
    blocks: list["KVBlock"],
    current_tokens: list[int],
    generation_kwargs: dict,
) -> "GenerationOutput":
    """
    Inject retrieved KV blocks and generate a response.

    If the adapter supports KV injection: prepend KV tensors directly
    into attention via adapter.inject_and_generate().

    If not (e.g. vLLM degraded mode): concatenate chunk_text as a
    text prefix and run a normal forward pass.

    Doc-wise RoPE is handled inside the adapter implementation.
    Blocks must be passed in a consistent order (caller's responsibility).

    Args:
        adapter:           framework adapter
        blocks:            retrieved KVBlocks, ordered for injection
        current_tokens:    query token IDs
        generation_kwargs: forwarded to underlying generate()

    Returns:
        GenerationOutput
    """
    if not blocks:
        # No retrieved context — plain generation
        return adapter.inject_and_generate([], current_tokens, generation_kwargs)

    if adapter.supports_kv_inject():
        logger.debug(
            "KV injection: %d blocks, %d tokens",
            len(blocks),
            sum(b.token_count for b in blocks),
        )
        return adapter.inject_and_generate(blocks, current_tokens, generation_kwargs)

    # Degraded mode: inject as text prefix
    logger.debug("Degraded injection: %d blocks as text prefix", len(blocks))
    prefix = " ".join(b.chunk_text for b in blocks)
    prefix_tokens = adapter.tokenizer.encode(prefix)
    combined_tokens = prefix_tokens + current_tokens
    return adapter.inject_and_generate([], combined_tokens, generation_kwargs)
