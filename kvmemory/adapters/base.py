"""
base.py — BaseAdapter abstract interface.

Every framework adapter (HF, llama.cpp, vLLM) implements this contract.
Zero framework-specific code exists anywhere outside adapters/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from ..storage.schema import KVBlock, GenerationOutput


class BaseAdapter(ABC):
    """
    Framework-agnostic adapter interface.

    Implement capture() and inject_and_generate() for a new backend.
    The rest of the library calls only these methods.
    """

    @abstractmethod
    def capture(
        self,
        tokens: list[int],
        text: str,
        layers: list[int],
    ) -> tuple[
        dict[int, tuple["torch.Tensor", "torch.Tensor"]],
        dict[int, "torch.Tensor"],
    ]:
        """
        Run a partial forward pass through the model.

        Extracts KV tensors and hidden states at each target layer in
        one forward pass (up to max(layers)).

        Args:
            tokens:  token IDs to process
            text:    original text string (for logging / debug)
            layers:  which layers to extract from

        Returns:
            kv_by_layer:     layer -> (K, V) tensors
                             K/V shape: [heads, seq, head_dim] float16
            hidden_by_layer: layer -> hidden states [seq, d_model] float32
        """
        ...

    @abstractmethod
    def inject_and_generate(
        self,
        blocks: list["KVBlock"],
        current_tokens: list[int],
        generation_kwargs: dict,
    ) -> "GenerationOutput":
        """
        Prepend retrieved KV blocks into attention, then generate.

        Doc-wise RoPE contract:
            block_i gets positions [0 .. block_i.token_count]
            query   gets positions [num_blocks .. num_blocks + len(current_tokens)]

        Only current_tokens hit prefill. Retrieved blocks are free.

        Args:
            blocks:            retrieved KVBlock list, ordered consistently
            current_tokens:    query token IDs
            generation_kwargs: forwarded to underlying generate()

        Returns:
            GenerationOutput with sequences and decoded text
        """
        ...

    @abstractmethod
    def supports_kv_inject(self) -> bool:
        """
        Whether this adapter supports direct KV injection.

        HF + llama.cpp: True   — inject via past_key_values / C API
        vLLM:           False  — degraded mode, inject as text prefix
        """
        ...

    @property
    @abstractmethod
    def d_model(self) -> int:
        """Hidden state dimension."""
        ...

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Total transformer layers in the model."""
        ...

    @property
    @abstractmethod
    def num_heads(self) -> int:
        """Number of attention heads."""
        ...

    @property
    @abstractmethod
    def tokenizer(self):
        """Tokenizer instance (must have encode / decode methods)."""
        ...
