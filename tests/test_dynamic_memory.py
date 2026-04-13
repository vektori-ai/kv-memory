from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from kvmemory.core.dynamic_memory import (
    annotate_claim_conflicts,
    classify_change_type,
    extract_fact_claims,
    find_conflicts,
    resolve_claims,
)
from kvmemory.storage.vector_db import VectorDB


def test_extract_deadline_claim_with_temporal_fields():
    claims = extract_fact_claims(
        "The project deadline was originally March 29. Actually, the project deadline is March 31.",
        observed_at="2026-04-13T00:00:00+00:00",
    )

    assert len(claims) == 2
    assert claims[0].claim_key == "project.deadline"
    assert claims[0].value == "March 29"
    assert claims[0].value_time == "March 29"
    assert claims[0].observed_at == "2026-04-13T00:00:00+00:00"
    assert claims[0].change_type == "historical_addition"
    assert claims[1].value == "March 31"
    assert claims[1].change_type == "correction"


def test_conflict_detection_same_key_different_value():
    old, new = extract_fact_claims(
        "The project deadline was originally March 29. Actually, the project deadline is March 31.",
        observed_at="2026-04-13T00:00:00+00:00",
    )

    conflicts = find_conflicts(new, [old])
    assert conflicts == [{
        "claim_key": "project.deadline",
        "new_value": "March 31",
        "prior_value": "March 29",
        "new_change_type": "correction",
        "prior_change_type": "historical_addition",
        "new_observed_at": "2026-04-13T00:00:00+00:00",
        "prior_observed_at": "2026-04-13T00:00:00+00:00",
    }]


def test_conflict_detection_accepts_payload_with_conflicts_field():
    old, new = extract_fact_claims(
        "The project deadline was originally March 29. Actually, the project deadline is March 31.",
        observed_at="2026-04-13T00:00:00+00:00",
    )
    old_payload = old.to_payload()
    old_payload["conflicts"] = []

    conflicts = find_conflicts(new, [old_payload])

    assert conflicts[0]["prior_value"] == "March 29"
    assert conflicts[0]["new_value"] == "March 31"


def test_resolution_prefers_correction_over_older_history():
    claims = extract_fact_claims(
        "The project deadline was originally March 29. Actually, the project deadline is March 31.",
        observed_at="2026-04-13T00:00:00+00:00",
    )

    winners = resolve_claims(claims)
    assert winners["project.deadline"].value == "March 31"


def test_resolution_prefers_newer_update_for_same_key():
    old = extract_fact_claims(
        "The project deadline moved to March 29.",
        observed_at="2026-04-12T00:00:00+00:00",
    )[0]
    new = extract_fact_claims(
        "The project deadline moved to March 31.",
        observed_at="2026-04-13T00:00:00+00:00",
    )[0]

    winners = resolve_claims([old, new])
    assert winners["project.deadline"].value == "March 31"


def test_uncertain_claim_has_low_confidence():
    assert classify_change_type("I think the project deadline is March 31.") == "uncertain"
    claim = extract_fact_claims("I think the project deadline is March 31.")[0]
    assert claim.confidence < 0.5


def test_annotate_claim_conflicts_marks_later_conflict():
    claims = extract_fact_claims(
        "The project deadline is March 29. The project deadline moved to March 31.",
        observed_at="2026-04-13T00:00:00+00:00",
    )
    annotated = annotate_claim_conflicts(claims)

    assert annotated[0]["conflicts"] == []
    assert annotated[1]["conflicts"][0]["prior_value"] == "March 29"


def test_vector_db_find_fact_claims_filters_payload_claims():
    db = VectorDB.__new__(VectorDB)
    db.client = MagicMock()
    db.client.scroll.return_value = ([
        SimpleNamespace(payload={
            "fact_claims": [
                {"claim_key": "project.deadline", "value": "March 31"},
                {"claim_key": "project.budget", "value": "$15"},
            ],
        })
    ], None)

    claims = db.find_fact_claims("test-model", "project.deadline", session_id="session-1")

    assert claims == [{"claim_key": "project.deadline", "value": "March 31"}]
    _, kwargs = db.client.scroll.call_args
    assert kwargs["collection_name"] == "test-model"
    assert kwargs["limit"] == 100
