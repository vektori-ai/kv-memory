from dataclasses import dataclass


@dataclass(slots=True)
class MemoryConfig:
    window_size: int = 8
    stride: int = 4
    top_k: int = 5
    min_retrieval_score: float = -1e9

    recency_lambda: float = 0.005
    semantic_weight: float = 1.0
    lexical_weight: float = 0.35
    recency_weight: float = 0.15
    importance_weight: float = 0.2

    embedding_dim: int = 768

    prefer_hf_embeddings: bool = True
    embedding_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    embedding_device: str = "cpu"
    embedding_allow_download: bool = False

    max_kv_entries: int | None = 512
