"""
model_id.py — Canonical model-id sanitizer.

Used by VectorDB, KVStore, and beam_eval to produce a consistent,
filesystem- and Qdrant-safe identifier from any HuggingFace model path.
"""


def sanitize_model_id(model_id: str) -> str:
    """Replace path separators and colons with underscores.

    Examples:
        "meta-llama/Llama-3-8B"  -> "meta-llama_Llama-3-8B"
        "models\\Qwen2.5-7B"     -> "models_Qwen2.5-7B"
        "registry:model:v1"      -> "registry_model_v1"
    """
    return model_id.replace("/", "_").replace("\\", "_").replace(":", "_")
