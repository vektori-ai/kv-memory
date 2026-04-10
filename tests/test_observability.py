from __future__ import annotations

import uuid
from pathlib import Path

from kvmemory.observability import ObservabilityStore


def _make_store_root() -> Path:
    root = Path(".test_obs_tmp") / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    return root


def test_create_run_and_append_events():
    store = ObservabilityStore(_make_store_root() / ".kvmem_obs")
    observer = store.create_run(
        config={"model": "test-model", "dtype": "float16", "synthetic": True, "n": 4},
        metadata={"dataset_source": "synthetic"},
    )

    observer.emit("phase_started", phase="dataset_load")
    observer.emit("question_started", phase="answer_questions", question_id="q1")
    observer.finish(summary={"kv_metrics": {"overall_accuracy": 1.0}})

    run = store.get_run(observer.run_id)
    assert run["status"] == "completed"
    assert run["event_count"] == 3
    assert run["summary"]["kv_metrics"]["overall_accuracy"] == 1.0

    events = store.get_events(observer.run_id)
    assert [event["type"] for event in events] == [
        "phase_started",
        "question_started",
        "run_finished",
    ]


def test_run_and_event_filters():
    store = ObservabilityStore(_make_store_root() / ".kvmem_obs")

    observer_a = store.create_run(
        config={"model": "model-a", "dtype": "float16", "synthetic": True, "n": 3},
        metadata={"dataset_source": "synthetic"},
    )
    observer_b = store.create_run(
        config={"model": "model-b", "dtype": "bfloat16", "synthetic": False, "n": 8},
        metadata={"dataset_source": "jsonl"},
    )

    observer_a.emit("question_started", phase="answer_questions", question_id="q-a")
    observer_a.emit("error", level="error", phase="answer_questions", question_id="q-a")
    observer_b.emit("question_started", phase="answer_questions", question_id="q-b")

    synthetic_runs = store.list_runs(dataset="synthetic")
    assert [run["run_id"] for run in synthetic_runs] == [observer_a.run_id]

    model_b_runs = store.list_runs(model="model-b")
    assert [run["run_id"] for run in model_b_runs] == [observer_b.run_id]

    filtered_events = store.get_events(
        observer_a.run_id,
        phase="answer_questions",
        question_id="q-a",
        level="error",
    )
    assert len(filtered_events) == 1
    assert filtered_events[0]["type"] == "error"
