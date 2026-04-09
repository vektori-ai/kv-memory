"""
vector_db.py — Qdrant wrapper for retrieval vector storage.

Stores one named vector per retrieval layer per KVBlock.
Payload stores lightweight metadata for filtering and MMR scoring.

Named vector schema per collection:
    'layer_{l}' -> float32[d_model], Distance.COSINE

Qdrant must be running: docker run -p 6333:6333 qdrant/qdrant
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

logger = logging.getLogger(__name__)


class VectorDB:
    """
    Qdrant wrapper. One collection per model_id.
    Each point represents one KVBlock, with named vectors per layer.
    """

    def __init__(self, url: str = "localhost", port: int = 6333) -> None:
        self.client = QdrantClient(host=url, port=port)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(
        self,
        model_id: str,
        retrieval_layers: list[int],
        d_model: int,
    ) -> None:
        """
        Create Qdrant collection if it doesn't exist.

        Uses Distance.COSINE because vectors are L2-normalized at write time.
        Qdrant uses dot product internally for unit vectors — fast and correct.
        """
        collection_name = self._collection_name(model_id)
        existing = [c.name for c in self.client.get_collections().collections]

        if collection_name in existing:
            logger.debug("Collection %s already exists", collection_name)
            return

        vectors_config = {
            f"layer_{layer}": VectorParams(size=d_model, distance=Distance.COSINE)
            for layer in retrieval_layers
        }
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
        )
        logger.info(
            "Created Qdrant collection %s with layers %s, d_model=%d",
            collection_name,
            retrieval_layers,
            d_model,
        )

    def delete_collection(self, model_id: str) -> None:
        """Drop a collection. Used in tests."""
        collection_name = self._collection_name(model_id)
        try:
            self.client.delete_collection(collection_name)
        except Exception as e:
            logger.warning("delete_collection failed: %s", e)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert(
        self,
        model_id: str,
        block_id: str,
        hidden_vecs: dict[int, np.ndarray],
        session_id: str,
        agent_id: Optional[str],
        shared: bool,
        token_count: int,
        chunk_text: str,
        importance_score: float,
    ) -> None:
        """
        Insert or update a KVBlock's retrieval entry.

        hidden_vecs: layer -> [d_model] float32, already normalized.
        Payload stores lightweight metadata used for filtering and MMR.
        """
        collection_name = self._collection_name(model_id)

        vectors = {
            f"layer_{layer}": vec.tolist()
            for layer, vec in hidden_vecs.items()
        }
        payload = {
            "model_id": model_id,
            "session_id": session_id,
            "agent_id": agent_id,
            "shared": shared,
            "token_count": token_count,
            "chunk_text": chunk_text,
            "created_at": time.time(),
            "importance_score": importance_score,
            "access_count": 0,
        }

        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=block_id,
                    vector=vectors,
                    payload=payload,
                )
            ],
        )
        logger.debug("VectorDB.upsert: %s (%d tokens)", block_id, token_count)

    # ------------------------------------------------------------------
    # Read — Stage 1 coarse ANN
    # ------------------------------------------------------------------

    def search_coarse(
        self,
        model_id: str,
        query_vecs: dict[int, np.ndarray],
        retrieval_layers: list[int],
        top_k: int,
        session_filter: Optional[dict] = None,
        layer_weights: Optional[dict[int, float]] = None,
    ) -> list[str]:
        """
        Multi-layer ANN search. Returns top_k block_ids by combined score.

        Performs one Qdrant search per layer, then merges scores with
        weighted sum. Returns sorted block_ids.
        """
        if layer_weights is None:
            # Default: 25% shallow, 50% middle, 25% deep
            n = len(retrieval_layers)
            if n == 3:
                weights_list = [0.25, 0.50, 0.25]
            else:
                weights_list = [1.0 / n] * n
            layer_weights = dict(zip(retrieval_layers, weights_list))

        qdrant_filter = self._build_filter(model_id, session_filter)
        candidate_scores: dict[str, float] = {}

        for layer in retrieval_layers:
            if layer not in query_vecs:
                continue
            q_vec = query_vecs[layer]
            weight = layer_weights.get(layer, 0.0)
            if weight == 0.0:
                continue

            results = self.client.query_points(
                collection_name=self._collection_name(model_id),
                query=q_vec.tolist(),
                using=f"layer_{layer}",
                limit=top_k,
                query_filter=qdrant_filter,
                with_vectors=False,
            ).points
            for r in results:
                bid = str(r.id)
                candidate_scores[bid] = candidate_scores.get(bid, 0.0) + weight * r.score

        sorted_ids = sorted(candidate_scores, key=candidate_scores.__getitem__, reverse=True)
        return sorted_ids[:top_k]

    # ------------------------------------------------------------------
    # Read — fetch payloads + vectors for MMR rerank
    # ------------------------------------------------------------------

    def fetch_with_vectors(
        self,
        model_id: str,
        block_ids: list[str],
    ) -> list[dict]:
        """
        Retrieve points with their named vectors and payload.
        Returns list of dicts: {id, vector, payload}.
        """
        if not block_ids:
            return []

        collection_name = self._collection_name(model_id)
        results = self.client.retrieve(
            collection_name=collection_name,
            ids=block_ids,
            with_vectors=True,
            with_payload=True,
        )

        return [
            {
                "id": str(p.id),
                "vector": p.vector,   # dict: layer_name -> list[float]
                "payload": p.payload,
            }
            for p in results
        ]

    # ------------------------------------------------------------------
    # Dedup check
    # ------------------------------------------------------------------

    def find_duplicate(
        self,
        model_id: str,
        hidden_vec: np.ndarray,
        layer: int,
        threshold: float = 0.95,
    ) -> Optional[str]:
        """
        Check if a near-identical block already exists.

        Returns block_id of the duplicate if found, else None.
        Uses the middle retrieval layer as the representative vector.
        """
        results = self.client.query_points(
            collection_name=self._collection_name(model_id),
            query=hidden_vec.tolist(),
            using=f"layer_{layer}",
            limit=1,
            score_threshold=threshold,
        ).points
        return str(results[0].id) if results else None

    # ------------------------------------------------------------------
    # Access count update
    # ------------------------------------------------------------------

    def increment_access_count(self, model_id: str, block_ids: list[str]) -> None:
        """Increment access_count payload field for retrieved blocks."""
        collection_name = self._collection_name(model_id)
        for bid in block_ids:
            try:
                results = self.client.retrieve(
                    collection_name=collection_name,
                    ids=[bid],
                    with_payload=True,
                    with_vectors=False,
                )
                if not results:
                    continue
                current = (results[0].payload or {}).get("access_count", 0) or 0
                self.client.set_payload(
                    collection_name=collection_name,
                    payload={"access_count": current + 1},
                    points=[bid],
                )
            except Exception as e:
                logger.debug("increment_access_count failed for %s: %s", bid, e)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _collection_name(model_id: str) -> str:
        """Sanitize model_id to a valid Qdrant collection name."""
        return model_id.replace("/", "_").replace("\\", "_").replace(":", "_")

    @staticmethod
    def _build_filter(
        model_id: str,
        session_filter: Optional[dict],
    ) -> Optional[Filter]:
        """
        Build a Qdrant filter that enforces model_id and optional session constraints.

        model_id enforcement is non-negotiable per the spec.

        Logic:
          - Always: model_id == this
          - retrieve_shared=False: AND session_id == this
          - retrieve_shared=True:  AND (session_id == this OR shared == True)
        """
        must: list[FieldCondition] = [
            FieldCondition(key="model_id", match=MatchValue(value=model_id))
        ]

        if session_filter:
            session_id = session_filter.get("session_id")
            retrieve_shared = session_filter.get("retrieve_shared", False)

            if session_id:
                if not retrieve_shared:
                    # Strict: only this session
                    must.append(
                        FieldCondition(key="session_id", match=MatchValue(value=session_id))
                    )
                else:
                    # OR: this session's blocks + any shared block
                    # Qdrant: must (model_id) + should (session_id OR shared=True)
                    # When both must and should are present, should acts as "at least one"
                    return Filter(
                        must=must,
                        should=[
                            FieldCondition(key="session_id", match=MatchValue(value=session_id)),
                            FieldCondition(key="shared", match=MatchValue(value=True)),
                        ],
                    )

        return Filter(must=must)
