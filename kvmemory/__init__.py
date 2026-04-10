"""
kvmemory — KV Memory System for self-hosted LLMs.

Stores KV tensors from past interactions, retrieves via two-stage ranking,
and injects directly into attention layers. Prefill cost paid once at write.

Quick start:
    from kvmemory import KVMemory, KVMemoryConfig
    from kvmemory.adapters.hf_adapter import HFAdapter

    adapter = HFAdapter(model, tokenizer)
    config = KVMemoryConfig(model_id='llama-3-8b', retrieval_layers=[8, 16, 24])
    memory = KVMemory(adapter=adapter, config=config)
    output = await memory.generate("Your prompt here", session_id="user_123")
"""

from .config import KVMemoryConfig
from .memory import KVMemory
from .observability import ObservabilityStore, RunObserver
from .storage.schema import GenerationOutput, KVBlock

__all__ = [
    "KVMemory",
    "KVMemoryConfig",
    "ObservabilityStore",
    "RunObserver",
    "KVBlock",
    "GenerationOutput",
]
