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

from ..utils.model_id import sanitize_model_id
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    MatchText,
    PointStruct,
    Range,
    VectorParams,
)

logger = logging.getLogger(__name__)


class VectorDB:
    """
    Qdrant wrapper. One collection per model_id.
    Each point represents one KVBlock, with named vectors per layer.
    """

    def __init__(self, url: str = "localhost", port: int = 6333) -> None:
        if url == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(host=url, port=port)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(
        self,
        model_id: str,
        retrieval_layers: list[int],
        vec_dim: int,
        d_model: int = 0,  # deprecated, ignored — kept for backwards compat
    ) -> None:
        """
        Create Qdrant collection if it doesn't exist.

        Uses Distance.COSINE because vectors are L2-normalized at write time.
        Qdrant uses dot product internally for unit vectors — fast and correct.

        vec_dim: dimension of stored retrieval vectors.
            - hidden-state mode: d_model (e.g. 5120 for Qwen3-14B)
            - k-vector mode:     num_kv_heads * head_dim (e.g. 8*128=1024)
        """
        collection_name = self._collection_name(model_id)
        existing = [c.name for c in self.client.get_collections().collections]

        if collection_name in existing:
            logger.debug("Collection %s already exists", collection_name)
            return

        vectors_config = {
            f"layer_{layer}": VectorParams(size=vec_dim, distance=Distance.COSINE)
            for layer in retrieval_layers
        }
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
        )
        logger.info(
            "Created Qdrant collection %s with layers %s, vec_dim=%d",
            collection_name,
            retrieval_layers,
            vec_dim,
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
        trace_payload: Optional[dict] = None,
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
            "retrieval_layers": sorted(hidden_vecs),
        }
        payload.update({
            key: value
            for key, value in (trace_payload or {}).items()
            if value is not None and key not in {"chunk_index", "total_chunks"}
        })

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
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Check if a near-identical block already exists within the same session.

        Returns block_id of the duplicate if found, else None.
        session_id scoping is required: without it, blocks from previous eval runs
        or other sessions would be treated as duplicates of current session content.
        """
        qdrant_filter = self._build_filter(
            model_id,
            {"session_id": session_id} if session_id else None,
        )
        results = self.client.query_points(
            collection_name=self._collection_name(model_id),
            query=hidden_vec.tolist(),
            using=f"layer_{layer}",
            limit=1,
            score_threshold=threshold,
            query_filter=qdrant_filter,
        ).points
        return str(results[0].id) if results else None

    def find_duplicate_multilayer(
        self,
        model_id: str,
        hidden_vecs: dict[int, np.ndarray],
        layer_weights: dict[int, float],
        threshold: float = 0.95,
        session_id: Optional[str] = None,
        top_k_per_layer: int = 20,
    ) -> Optional[str]:
        """
        Multi-layer semantic dedup using candidate union + Python-side scoring.

        For each layer: query top-K candidates with NO score_threshold (so Qdrant
        returns the K nearest regardless of score, avoiding version-dependent
        threshold semantics). Union all candidate block IDs, then compute a
        weighted combined cosine score in Python using the per-layer weights.
        Returns the block_id of the best candidate if its combined score >= threshold.

        Args:
            hidden_vecs:       layer -> normalized [d_model] float32 vector
            layer_weights:     layer -> weight (should sum to 1.0)
            threshold:         minimum combined cosine score to flag as duplicate
            session_id:        scope dedup to this session only
            top_k_per_layer:   candidates fetched per layer
        """
        qdrant_filter = self._build_filter(
            model_id,
            {"session_id": session_id} if session_id else None,
        )
        collection_name = self._collection_name(model_id)

        # Step 1: collect candidate IDs from all layers (no score threshold)
        candidate_ids: set[str] = set()
        for layer, vec in hidden_vecs.items():
            if layer not in layer_weights or layer_weights[layer] == 0.0:
                continue
            results = self.client.query_points(
                collection_name=collection_name,
                query=vec.tolist(),
                using=f"layer_{layer}",
                limit=top_k_per_layer,
                query_filter=qdrant_filter,
                with_vectors=False,
            ).points
            for r in results:
                candidate_ids.add(str(r.id))

        if not candidate_ids:
            return None

        # Step 2: fetch stored vectors for all candidates
        candidate_points = self.client.retrieve(
            collection_name=collection_name,
            ids=list(candidate_ids),
            with_vectors=True,
            with_payload=False,
        )

        # Step 3: compute weighted combined cosine score in Python
        best_id: Optional[str] = None
        best_score = -1.0
        for point in candidate_points:
            vectors = point.vector or {}
            score = 0.0
            for layer, w in layer_weights.items():
                key = f"layer_{layer}"
                if key not in vectors or layer not in hidden_vecs:
                    continue
                q = hidden_vecs[layer]
                k = np.array(vectors[key], dtype=np.float32)
                score += w * float(np.dot(q, k))
            if score > best_score:
                best_score = score
                best_id = str(point.id)

        if best_score >= threshold:
            return best_id
        return None

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
                    payload={"access_count": current + 1, "last_accessed_at": time.time()},
                    points=[bid],
                )
            except Exception as e:
                logger.debug("increment_access_count failed for %s: %s", bid, e)

    def find_fact_claims(
        self,
        model_id: str,
        claim_key: str,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Fetch stored fact claims for one entity.attribute key."""
        must: list[FieldCondition] = [
            FieldCondition(key="model_id", match=MatchValue(value=model_id)),
            FieldCondition(key="fact_keys", match=MatchValue(value=claim_key)),
        ]
        if session_id:
            must.append(FieldCondition(key="session_id", match=MatchValue(value=session_id)))

        try:
            records, _ = self.client.scroll(
                collection_name=self._collection_name(model_id),
                scroll_filter=Filter(must=must),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.debug("find_fact_claims failed for %s: %s", claim_key, exc)
            return []

        claims: list[dict] = []
        for record in records:
            payload = record.payload or {}
            for claim in payload.get("fact_claims", []) or []:
                if claim.get("claim_key") == claim_key:
                    claims.append(claim)
        return claims

    def list_collections(self) -> list[dict]:
        """Return lightweight collection metadata for dashboard browsing."""
        collections = []
        for collection in self.client.get_collections().collections:
            info = self.client.get_collection(collection.name)
            collections.append(
                {
                    "name": collection.name,
                    "status": str(getattr(info, "status", "")),
                    "points_count": getattr(info, "points_count", None),
                    "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
                    "vectors_count": getattr(info, "vectors_count", None),
                }
            )
        return collections

    def scroll_points(
        self,
        *,
        collection_name: str,
        limit: int = 100,
        offset: Optional[str] = None,
        with_vectors: bool = False,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        question_id: Optional[str] = None,
        phase: Optional[str] = None,
        agent_id: Optional[str] = None,
        shared: Optional[bool] = None,
        importance_min: Optional[float] = None,
        importance_max: Optional[float] = None,
        token_count_min: Optional[int] = None,
        token_count_max: Optional[int] = None,
        created_after: Optional[float] = None,
        created_before: Optional[float] = None,
        text_contains: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> dict:
        qdrant_filter = self._build_dashboard_filter(
            run_id=run_id,
            session_id=session_id,
            question_id=question_id,
            phase=phase,
            agent_id=agent_id,
            shared=shared,
            importance_min=importance_min,
            importance_max=importance_max,
            token_count_min=token_count_min,
            token_count_max=token_count_max,
            created_after=created_after,
            created_before=created_before,
            text_contains=text_contains,
        )
        records, next_offset = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=qdrant_filter,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=with_vectors,
        )

        points = []
        for record in records:
            payload = record.payload or {}
            if layer is not None and layer not in payload.get("retrieval_layers", []):
                continue
            points.append(
                {
                    "id": str(record.id),
                    "payload": payload,
                    "vector": record.vector if with_vectors else None,
                }
            )

        return {
            "collection": collection_name,
            "points": points,
            "next_offset": str(next_offset) if next_offset is not None else None,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _collection_name(model_id: str) -> str:
        """Sanitize model_id to a valid Qdrant collection name."""
        return sanitize_model_id(model_id)

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

    @staticmethod
    def _build_dashboard_filter(
        *,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        question_id: Optional[str] = None,
        phase: Optional[str] = None,
        agent_id: Optional[str] = None,
        shared: Optional[bool] = None,
        importance_min: Optional[float] = None,
        importance_max: Optional[float] = None,
        token_count_min: Optional[int] = None,
        token_count_max: Optional[int] = None,
        created_after: Optional[float] = None,
        created_before: Optional[float] = None,
        text_contains: Optional[str] = None,
    ) -> Optional[Filter]:
        must: list[FieldCondition] = []

        if run_id:
            must.append(FieldCondition(key="run_id", match=MatchValue(value=run_id)))
        if session_id:
            must.append(FieldCondition(key="session_id", match=MatchValue(value=session_id)))
        if question_id:
            must.append(FieldCondition(key="question_id", match=MatchValue(value=question_id)))
        if phase:
            must.append(FieldCondition(key="phase", match=MatchValue(value=phase)))
        if agent_id:
            must.append(FieldCondition(key="agent_id", match=MatchValue(value=agent_id)))
        if shared is not None:
            must.append(FieldCondition(key="shared", match=MatchValue(value=shared)))
        if importance_min is not None or importance_max is not None:
            must.append(
                FieldCondition(
                    key="importance_score",
                    range=Range(gte=importance_min, lte=importance_max),
                )
            )
        if token_count_min is not None or token_count_max is not None:
            must.append(
                FieldCondition(
                    key="token_count",
                    range=Range(gte=token_count_min, lte=token_count_max),
                )
            )
        if created_after is not None or created_before is not None:
            must.append(
                FieldCondition(
                    key="created_at",
                    range=Range(gte=created_after, lte=created_before),
                )
            )
        if text_contains:
            must.append(FieldCondition(key="chunk_text", match=MatchText(text=text_contains)))

        if not must:
            return None
        return Filter(must=must)
