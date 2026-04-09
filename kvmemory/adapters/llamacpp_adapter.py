"""
llamacpp_adapter.py — llama.cpp C API adapter. Phase 2.

Implements KV capture and injection via the llama.cpp C API.
Requires the llama-cpp-python package: pip install llama-cpp-python

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


class LlamaCppAdapter(BaseAdapter):
    """
    llama.cpp adapter for KV capture and injection.

    Uses the llama.cpp C API via llama-cpp-python.
    Supports full KV injection (supports_kv_inject() == True).

    Phase 2 implementation target.
    """

    def __init__(self, model_path: str, **kwargs) -> None:
        """
        Args:
            model_path: path to GGUF model file
            **kwargs:   forwarded to llama_cpp.Llama()
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for LlamaCppAdapter. "
                "Install with: pip install llama-cpp-python"
            )

        self._llama = Llama(model_path=model_path, **kwargs)
        self._model_path = model_path

    def capture(self, tokens, text, layers):
        raise NotImplementedError(
            "LlamaCppAdapter.capture() is a Phase 2 target. "
            "Use HFAdapter for Phase 1."
        )

    def inject_and_generate(self, blocks, current_tokens, generation_kwargs):
        raise NotImplementedError(
            "LlamaCppAdapter.inject_and_generate() is a Phase 2 target."
        )

    def supports_kv_inject(self) -> bool:
        return True

    @property
    def d_model(self) -> int:
        return self._llama.model.n_embd()

    @property
    def num_layers(self) -> int:
        return self._llama.model.n_layer()

    @property
    def num_heads(self) -> int:
        return self._llama.model.n_head()

    @property
    def tokenizer(self):
        return self._llama
