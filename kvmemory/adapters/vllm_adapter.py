"""
vllm_adapter.py — vLLM adapter (degraded mode). Phase 2.

vLLM's block manager doesn't expose KV injection cleanly.
Degraded mode: hidden states captured for retrieval vectors,
but injection falls back to text prefix.

supports_kv_inject() == False
inject_and_generate() from injector.py handles the text prefix fallback.

Status: stub — interface defined, implementation pending Phase 2.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import BaseAdapter
from ..storage.schema import GenerationOutput

if TYPE_CHECKING:
    import torch
    from ..storage.schema import KVBlock

logger = logging.getLogger(__name__)


class VLLMAdapter(BaseAdapter):
    """
    vLLM adapter — degraded mode.

    supports_kv_inject() == False.
    Capture() extracts hidden states for retrieval vectors.
    inject_and_generate() is a text-prefix fallback (handled by injector.py).

    Phase 2 implementation target.
    """

    def __init__(self, llm, tokenizer) -> None:
        """
        Args:
            llm:       vllm.LLM instance
            tokenizer: HuggingFace tokenizer (for encode/decode)
        """
        self._llm = llm
        self._tokenizer = tokenizer

    def capture(self, tokens, text, layers):
        raise NotImplementedError(
            "VLLMAdapter.capture() is a Phase 2 target. "
            "Use HFAdapter for Phase 1."
        )

    def inject_and_generate(self, blocks, current_tokens, generation_kwargs):
        raise NotImplementedError(
            "VLLMAdapter.inject_and_generate() is a Phase 2 target. "
            "Text-prefix fallback handled by injector.py."
        )

    def supports_kv_inject(self) -> bool:
        return False  # vLLM degraded mode

    @property
    def d_model(self) -> int:
        return self._llm.llm_engine.model_config.hidden_size

    @property
    def num_layers(self) -> int:
        return self._llm.llm_engine.model_config.num_hidden_layers

    @property
    def num_heads(self) -> int:
        return self._llm.llm_engine.model_config.num_attention_heads

    @property
    def tokenizer(self):
        return self._tokenizer
