from __future__ import annotations

import hashlib
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray: ...


class HashEmbedder:
    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], byteorder="big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector


class HFMeanPoolEmbedder:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        *,
        device: str = "cpu",
        max_length: int = 2048,
        allow_download: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HFMeanPoolEmbedder requires optional dependencies: pip install -e .[hf]"
            ) from exc

        self._torch = torch
        self._max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=not allow_download,
        )
        self._model = AutoModel.from_pretrained(
            model_name,
            local_files_only=not allow_download,
        )
        self._model.to(device)
        self._device = device
        self._model.eval()

    def embed(self, text: str) -> np.ndarray:
        with self._torch.no_grad():
            encoded = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self._max_length,
            ).to(self._device)
            outputs = self._model(**encoded)
            hidden = outputs.last_hidden_state.squeeze(0)
            pooled = hidden.mean(dim=0)
            vector = pooled.detach().cpu().numpy().astype(np.float32)

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector


def create_default_embedder(
    *,
    prefer_hf: bool,
    embedding_dim: int,
    model_name: str,
    device: str,
    allow_download: bool,
) -> Embedder:
    if prefer_hf:
        try:
            return HFMeanPoolEmbedder(
                model_name=model_name,
                device=device,
                allow_download=allow_download,
            )
        except Exception:
            pass
    return HashEmbedder(dim=embedding_dim)
