"""
dynamic_memory.py - lightweight claim extraction and conflict resolution.

This is intentionally heuristic. KV blocks remain the append-only evidence
payload; these claims are a small structured index that helps identify updates,
conflicts, and temporal facts without scanning every stored chunk at query time.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional


MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
)

DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}-\d{2}-\d{2}|"
    r"\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:" + "|".join(MONTHS) + r")\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s*\d{4})?"
    r")\b",
    re.IGNORECASE,
)

PRICE_RE = re.compile(
    r"\b(?P<attr>subscription price|price|cost|budget)\b[^.\n]{0,60}?"
    r"(?P<value>\$?\d+(?:\.\d+)?(?:\s*/\s*[a-zA-Z]+)?)",
    re.IGNORECASE,
)

DATE_ATTRIBUTES = ("deadline", "due date", "meeting", "launch", "release", "appointment")
ENTITY_STOPWORDS = {
    "a", "an", "and", "as", "at", "but", "for", "from", "in", "is", "it",
    "of", "on", "or", "the", "this", "to", "was", "were",
}


@dataclass(frozen=True)
class FactClaim:
    entity: str
    attribute: str
    value: str
    claim_key: str
    change_type: str
    confidence: float
    observed_at: str
    event_time: Optional[str]
    value_time: Optional[str]
    source_text: str

    def to_payload(self) -> dict:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def extract_fact_claims(text: str, observed_at: Optional[str] = None) -> list[FactClaim]:
    """Extract simple entity.attribute=value claims from a memory chunk."""
    observed_at = observed_at or utc_now_iso()
    claims: list[FactClaim] = []
    seen: set[tuple[str, str, str]] = set()

    for sentence in _sentences(text):
        change_type = classify_change_type(sentence)
        confidence = confidence_for_change_type(change_type)

        for attr in DATE_ATTRIBUTES:
            if attr not in sentence.lower():
                continue
            for date_match in DATE_RE.finditer(sentence):
                value = date_match.group(0).strip()
                entity = _entity_for_attribute(sentence, attr)
                claim = _make_claim(
                    entity=entity,
                    attribute=attr,
                    value=value,
                    change_type=change_type,
                    confidence=confidence,
                    observed_at=observed_at,
                    event_time=normalize_date(value),
                    source_text=sentence,
                )
                dedup_key = (claim.claim_key, _norm_value(claim.value), claim.change_type)
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    claims.append(claim)

        for match in PRICE_RE.finditer(sentence):
            attr = _canonical(match.group("attr"))
            value = " ".join(match.group("value").replace(" ", "").split())
            entity = _entity_for_attribute(sentence, match.group("attr"))
            claim = _make_claim(
                entity=entity,
                attribute=attr,
                value=value,
                change_type=change_type,
                confidence=confidence,
                observed_at=observed_at,
                event_time=_first_date(sentence),
                source_text=sentence,
            )
            dedup_key = (claim.claim_key, _norm_value(claim.value), claim.change_type)
            if dedup_key not in seen:
                seen.add(dedup_key)
                claims.append(claim)

    return claims


def classify_change_type(text: str) -> str:
    """Classify how a claim should behave during conflict resolution."""
    lower = text.lower()
    if re.search(r"\b(i think|might|maybe|probably|possibly|unsure)\b", lower):
        return "uncertain"
    if re.search(r"\b(actually|correction|corrected|instead|not .{0,40} but)\b", lower):
        return "correction"
    if re.search(r"\b(moved|changed|updated|latest|now|new deadline|rescheduled)\b", lower):
        return "update"
    if re.search(r"\b(originally|previously|initially|formerly|used to)\b", lower):
        return "historical_addition"
    return "assertion"


def confidence_for_change_type(change_type: str) -> float:
    return {
        "correction": 0.95,
        "update": 0.85,
        "assertion": 0.70,
        "historical_addition": 0.65,
        "uncertain": 0.35,
    }.get(change_type, 0.50)


def find_conflicts(new_claim: FactClaim, existing_claims: Iterable[FactClaim | dict]) -> list[dict]:
    """Return prior same-key claims with a different normalized value."""
    conflicts = []
    new_value = _norm_value(new_claim.value)
    for prior_raw in existing_claims:
        prior = _coerce_claim(prior_raw)
        if prior.claim_key != new_claim.claim_key:
            continue
        if _norm_value(prior.value) == new_value:
            continue
        conflicts.append({
            "claim_key": new_claim.claim_key,
            "new_value": new_claim.value,
            "prior_value": prior.value,
            "new_change_type": new_claim.change_type,
            "prior_change_type": prior.change_type,
            "new_observed_at": new_claim.observed_at,
            "prior_observed_at": prior.observed_at,
        })
    return conflicts


def resolve_claims(claims: Iterable[FactClaim | dict]) -> dict[str, FactClaim]:
    """Resolve one winning current claim per claim_key using the project policy."""
    winners: dict[str, FactClaim] = {}
    for raw_claim in claims:
        claim = _coerce_claim(raw_claim)
        current = winners.get(claim.claim_key)
        if current is None or _resolution_rank(claim) > _resolution_rank(current):
            winners[claim.claim_key] = claim
    return winners


def annotate_claim_conflicts(claims: list[FactClaim]) -> list[dict]:
    """Return claim payloads with same-key conflict metadata."""
    prior_by_key: dict[str, list[FactClaim]] = {}
    annotated = []
    for claim in claims:
        prior = prior_by_key.get(claim.claim_key, [])
        payload = claim.to_payload()
        payload["conflicts"] = find_conflicts(claim, prior)
        prior_by_key.setdefault(claim.claim_key, []).append(claim)
        annotated.append(payload)
    return annotated


def normalize_date(value: str) -> str:
    cleaned = value.strip().rstrip(".,")
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)\b", r"\1", cleaned, flags=re.IGNORECASE)
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            pass
    return re.sub(r"\s+", " ", cleaned)


def _first_date(text: str) -> Optional[str]:
    match = DATE_RE.search(text)
    return normalize_date(match.group(0)) if match else None


def _make_claim(
    entity: str,
    attribute: str,
    value: str,
    change_type: str,
    confidence: float,
    observed_at: str,
    event_time: Optional[str],
    source_text: str,
) -> FactClaim:
    entity = _canonical(entity) or "global"
    attribute = _canonical(attribute)
    return FactClaim(
        entity=entity,
        attribute=attribute,
        value=value.strip(),
        claim_key=f"{entity}.{attribute}",
        change_type=change_type,
        confidence=confidence,
        observed_at=observed_at,
        event_time=event_time,
        value_time=event_time,
        source_text=source_text.strip(),
    )


def _entity_for_attribute(sentence: str, attribute: str) -> str:
    lower = sentence.lower()
    idx = lower.find(attribute.lower())
    left = sentence[:idx] if idx >= 0 else ""
    words = [
        word
        for word in re.findall(r"[A-Za-z0-9_-]+", left.lower())
        if word not in ENTITY_STOPWORDS
    ]
    words = [word for word in words if word not in {"actually", "originally", "previously"}]
    return " ".join(words[-4:]) or "global"


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _norm_value(value: str) -> str:
    return _canonical(normalize_date(value))


def _resolution_rank(claim: FactClaim) -> tuple[int, str, float]:
    type_rank = {
        "correction": 4,
        "update": 3,
        "assertion": 2,
        "historical_addition": 1,
        "uncertain": 0,
    }.get(claim.change_type, 0)
    return (type_rank, claim.observed_at, claim.confidence)


def _coerce_claim(raw: FactClaim | dict) -> FactClaim:
    if isinstance(raw, FactClaim):
        return raw
    return FactClaim(**raw)
