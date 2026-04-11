"""
store.py â€” JSON/JSONL-backed observability store for local dashboarding.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional


def _utc_ts() -> float:
    return time.time()


def _iso_ts(ts: Optional[float] = None) -> str:
    import datetime as _dt

    timestamp = _utc_ts() if ts is None else ts
    return _dt.datetime.fromtimestamp(timestamp, tz=_dt.timezone.utc).isoformat()


def _normalize_dict(values: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not values:
        return {}
    return {k: v for k, v in values.items() if v is not None}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


class RunObserver:
    """Bound helper for emitting events into one run with shared context."""

    def __init__(
        self,
        store: "ObservabilityStore",
        run_id: str,
        context_fields: Optional[dict[str, Any]] = None,
    ) -> None:
        self.store = store
        self.run_id = run_id
        self._context_fields = _normalize_dict(context_fields)

    @property
    def context_fields(self) -> dict[str, Any]:
        return dict(self._context_fields)

    def child(self, **fields: Any) -> "RunObserver":
        merged = dict(self._context_fields)
        merged.update(_normalize_dict(fields))
        return RunObserver(self.store, self.run_id, merged)

    def emit(
        self,
        event_type: str,
        *,
        level: str = "info",
        message: Optional[str] = None,
        **fields: Any,
    ) -> dict[str, Any]:
        payload = dict(self._context_fields)
        payload.update(_normalize_dict(fields))
        return self.store.append_event(
            self.run_id,
            event_type=event_type,
            level=level,
            message=message,
            fields=payload,
        )

    def update_summary(self, **summary: Any) -> dict[str, Any]:
        return self.store.update_run(self.run_id, summary_merge=summary)

    def update_metadata(self, **metadata: Any) -> dict[str, Any]:
        return self.store.update_run(self.run_id, metadata_merge=metadata)

    def finish(self, *, summary: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        self.emit("run_finished", status="completed")
        return self.store.finish_run(self.run_id, status="completed", summary=summary)

    def fail(
        self,
        *,
        error: str,
        summary: Optional[dict[str, Any]] = None,
        error_type: Optional[str] = None,
    ) -> dict[str, Any]:
        self.emit(
            "run_failed",
            level="error",
            status="failed",
            error=error,
            error_type=error_type,
        )
        return self.store.finish_run(
            self.run_id,
            status="failed",
            summary=summary,
            error=error,
            error_type=error_type,
        )


class ObservabilityStore:
    """
    Durable local run/event store.

    Layout:
      .kvmem_obs/
        runs/<run_id>.json
        events/<run_id>.jsonl
        index.jsonl
    """

    def __init__(self, base_path: str | Path = ".kvmem_obs") -> None:
        self.base_path = Path(base_path)
        self.runs_dir = self.base_path / "runs"
        self.events_dir = self.base_path / "events"
        self.index_path = self.base_path / "index.jsonl"
        self._lock = threading.Lock()

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.touch(exist_ok=True)

    def create_run(
        self,
        *,
        config: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> RunObserver:
        now = _utc_ts()
        resolved_run_id = run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        doc = {
            "run_id": resolved_run_id,
            "status": "running",
            "created_at": now,
            "created_at_iso": _iso_ts(now),
            "started_at": now,
            "started_at_iso": _iso_ts(now),
            "updated_at": now,
            "updated_at_iso": _iso_ts(now),
            "finished_at": None,
            "finished_at_iso": None,
            "config": _normalize_dict(config),
            "metadata": _normalize_dict(metadata),
            "summary": {},
            "event_count": 0,
            "error_count": 0,
            "next_seq": 1,
            "last_event_at": None,
            "last_event_at_iso": None,
        }
        with self._lock:
            self._write_json(self._run_path(resolved_run_id), doc)
            self._append_jsonl(self.index_path, self._index_entry(doc))
        return RunObserver(self, resolved_run_id)

    def update_run(
        self,
        run_id: str,
        *,
        status: Optional[str] = None,
        summary_merge: Optional[dict[str, Any]] = None,
        metadata_merge: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        with self._lock:
            doc = self._read_run_doc(run_id)
            if status:
                doc["status"] = status
            if summary_merge:
                doc.setdefault("summary", {}).update(_normalize_dict(summary_merge))
            if metadata_merge:
                doc.setdefault("metadata", {}).update(_normalize_dict(metadata_merge))

            now = _utc_ts()
            doc["updated_at"] = now
            doc["updated_at_iso"] = _iso_ts(now)
            self._write_json(self._run_path(run_id), doc)
            return doc

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        summary: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        error_type: Optional[str] = None,
    ) -> dict[str, Any]:
        with self._lock:
            doc = self._read_run_doc(run_id)
            now = _utc_ts()
            doc["status"] = status
            doc["finished_at"] = now
            doc["finished_at_iso"] = _iso_ts(now)
            doc["updated_at"] = now
            doc["updated_at_iso"] = _iso_ts(now)
            if summary:
                doc.setdefault("summary", {}).update(_normalize_dict(summary))
            if error:
                doc.setdefault("summary", {})["error"] = error
            if error_type:
                doc.setdefault("summary", {})["error_type"] = error_type
            self._write_json(self._run_path(run_id), doc)
            self._append_jsonl(self.index_path, self._index_entry(doc))
            return doc

    def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        level: str = "info",
        message: Optional[str] = None,
        fields: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        with self._lock:
            doc = self._read_run_doc(run_id)
            seq = int(doc.get("next_seq", 1))
            now = _utc_ts()
            event = {
                "run_id": run_id,
                "seq": seq,
                "ts": now,
                "ts_iso": _iso_ts(now),
                "type": event_type,
                "level": level,
            }
            if message:
                event["message"] = message
            event.update(_normalize_dict(fields))

            self._append_jsonl(self._event_path(run_id), event)

            doc["next_seq"] = seq + 1
            doc["event_count"] = int(doc.get("event_count", 0)) + 1
            if level == "error" or event_type == "error":
                doc["error_count"] = int(doc.get("error_count", 0)) + 1
            doc["last_event_at"] = now
            doc["last_event_at_iso"] = _iso_ts(now)
            doc["updated_at"] = now
            doc["updated_at_iso"] = _iso_ts(now)
            self._write_json(self._run_path(run_id), doc)
            return event

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            return self._read_run_doc(run_id)

    def list_runs(
        self,
        *,
        status: Optional[str] = None,
        model: Optional[str] = None,
        dtype: Optional[str] = None,
        dataset: Optional[str] = None,
        synthetic: Optional[bool] = None,
        n: Optional[int] = None,
        start_after: Optional[float] = None,
        start_before: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        with self._lock:
            for path in self.runs_dir.glob("*.json"):
                doc = self._read_json(path)
                if not doc:
                    continue
                if status and doc.get("status") != status:
                    continue
                config = doc.get("config", {})
                metadata = doc.get("metadata", {})
                if model and config.get("model") != model and config.get("model_id") != model:
                    continue
                if dtype and config.get("dtype") != dtype:
                    continue
                if dataset and metadata.get("dataset_source") != dataset and config.get("dataset") != dataset:
                    continue
                if synthetic is not None and bool(config.get("synthetic")) != synthetic:
                    continue
                if n is not None and config.get("n") != n:
                    continue
                started_at = doc.get("started_at")
                if start_after is not None and started_at is not None and started_at < start_after:
                    continue
                if start_before is not None and started_at is not None and started_at > start_before:
                    continue
                # Mark runs that appear stuck as stale (no disk mutation)
                if doc.get("status") == "running":
                    last_event_at = doc.get("last_event_at")
                    if last_event_at is not None and time.time() - last_event_at > 7200:
                        doc["stale"] = True
                runs.append(doc)

        runs.sort(key=lambda item: item.get("started_at") or 0.0, reverse=True)
        return runs

    def get_events(
        self,
        run_id: str,
        *,
        event_type: Optional[str] = None,
        phase: Optional[str] = None,
        question_id: Optional[str] = None,
        level: Optional[str] = None,
        since_ts: Optional[float] = None,
        since_seq: Optional[int] = None,
        search: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        path = self._event_path(run_id)
        if not path.exists():
            return []

        events: list[dict[str, Any]] = []
        needle = search.lower() if search else None
        with self._lock:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    if event_type and event.get("type") != event_type:
                        continue
                    if phase and event.get("phase") != phase:
                        continue
                    if question_id and event.get("question_id") != question_id:
                        continue
                    if level and event.get("level") != level:
                        continue
                    if since_ts is not None and float(event.get("ts", 0.0)) < since_ts:
                        continue
                    if since_seq is not None and int(event.get("seq", 0)) <= since_seq:
                        continue
                    if needle:
                        haystack = json.dumps(event, ensure_ascii=False, default=_json_default).lower()
                        if needle not in haystack:
                            continue
                    events.append(event)

        if limit is not None:
            return events[:limit]
        return events

    def get_live_events(
        self,
        run_id: str,
        *,
        since_ts: Optional[float] = None,
        since_seq: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        return self.get_events(
            run_id,
            since_ts=since_ts,
            since_seq=since_seq,
            limit=limit,
        )

    def _index_entry(self, doc: dict[str, Any]) -> dict[str, Any]:
        config = doc.get("config", {})
        metadata = doc.get("metadata", {})
        summary = doc.get("summary", {})
        return {
            "run_id": doc.get("run_id"),
            "status": doc.get("status"),
            "started_at": doc.get("started_at"),
            "started_at_iso": doc.get("started_at_iso"),
            "finished_at": doc.get("finished_at"),
            "finished_at_iso": doc.get("finished_at_iso"),
            "updated_at": doc.get("updated_at"),
            "updated_at_iso": doc.get("updated_at_iso"),
            "model": config.get("model") or config.get("model_id"),
            "dtype": config.get("dtype"),
            "dataset_source": metadata.get("dataset_source"),
            "synthetic": config.get("synthetic"),
            "n": config.get("n"),
            "overall_accuracy": summary.get("kv_metrics", {}).get("overall_accuracy"),
            "overall_f1": summary.get("kv_metrics", {}).get("overall_f1"),
        }

    def _run_path(self, run_id: str) -> Path:
        return self.runs_dir / f"{run_id}.json"

    def _event_path(self, run_id: str) -> Path:
        return self.events_dir / f"{run_id}.jsonl"

    def _read_run_doc(self, run_id: str) -> dict[str, Any]:
        path = self._run_path(run_id)
        doc = self._read_json(path)
        if doc is None:
            raise FileNotFoundError(f"Unknown run_id: {run_id}")
        return doc

    @staticmethod
    def _read_json(path: Path) -> Optional[dict[str, Any]]:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default))
            handle.write("\n")
