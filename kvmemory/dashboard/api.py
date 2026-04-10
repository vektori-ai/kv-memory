"""
api.py - FastAPI backend for local benchmark observability.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..observability import ObservabilityStore
from ..storage.vector_db import VectorDB


def _parse_time(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return datetime.fromisoformat(value).timestamp()


def create_app(
    *,
    obs_dir: str = ".kvmem_obs",
    qdrant_url: str = "localhost",
    qdrant_port: int = 6333,
):
    from fastapi import FastAPI, HTTPException, Query

    store = ObservabilityStore(obs_dir)
    vector_db = VectorDB(url=qdrant_url, port=qdrant_port)
    app = FastAPI(title="KV Memory Dashboard API")

    @app.get("/api/runs")
    def list_runs(
        status: Optional[str] = None,
        model: Optional[str] = None,
        dtype: Optional[str] = None,
        dataset: Optional[str] = None,
        synthetic: Optional[bool] = None,
        n: Optional[int] = None,
        start_after: Optional[str] = None,
        start_before: Optional[str] = None,
    ):
        runs = store.list_runs(
            status=status,
            model=model,
            dtype=dtype,
            dataset=dataset,
            synthetic=synthetic,
            n=n,
            start_after=_parse_time(start_after),
            start_before=_parse_time(start_before),
        )
        return {"runs": runs}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.get("/api/runs/{run_id}/events")
    def get_events(
        run_id: str,
        event_type: Optional[str] = Query(default=None, alias="type"),
        phase: Optional[str] = None,
        question_id: Optional[str] = None,
        level: Optional[str] = None,
        since_ts: Optional[str] = None,
        since_seq: Optional[int] = None,
        search: Optional[str] = None,
        limit: Optional[int] = 1000,
    ):
        if not store.get_run(run_id):
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "events": store.get_events(
                run_id,
                event_type=event_type,
                phase=phase,
                question_id=question_id,
                level=level,
                since_ts=_parse_time(since_ts),
                since_seq=since_seq,
                search=search,
                limit=limit,
            )
        }

    @app.get("/api/live/{run_id}")
    def get_live(
        run_id: str,
        since_ts: Optional[str] = None,
        since_seq: Optional[int] = None,
        limit: Optional[int] = 500,
    ):
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return {
            "run": run,
            "events": store.get_live_events(
                run_id,
                since_ts=_parse_time(since_ts),
                since_seq=since_seq,
                limit=limit,
            ),
        }

    @app.get("/api/qdrant/collections")
    def list_collections():
        return {"collections": vector_db.list_collections()}

    @app.get("/api/qdrant/points")
    def list_points(
        collection: str,
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
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        text_contains: Optional[str] = None,
        layer: Optional[int] = None,
    ):
        return vector_db.scroll_points(
            collection_name=collection,
            limit=limit,
            offset=offset,
            with_vectors=with_vectors,
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
            created_after=_parse_time(created_after),
            created_before=_parse_time(created_before),
            text_contains=text_contains,
            layer=layer,
        )

    return app


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the local KV Memory dashboard API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--obs-dir", default=".kvmem_obs")
    parser.add_argument("--qdrant-url", default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    args = parser.parse_args()

    app = create_app(
        obs_dir=args.obs_dir,
        qdrant_url=args.qdrant_url,
        qdrant_port=args.qdrant_port,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
