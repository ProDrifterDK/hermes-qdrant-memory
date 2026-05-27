from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_REVIEW_HISTORY_TERMS = (
    "review",
    "history",
    "historical",
    "stale",
    "deprecated",
    "superseded",
    "disputed",
    "conflict",
    "conflicting",
    "audit",
    "provenance",
)
_PENALIZED_FACT_STATUSES = {"stale", "review_required", "disputed", "superseded", "deprecated"}


@dataclass(frozen=True)
class RankingContext:
    query: str = ""
    include_fact_history: bool = False
    source_filter: str | None = None
    file_path_filter: str | None = None
    project_path_filter: str | None = None
    subject: str | None = None
    fact_key: str | None = None


@dataclass(frozen=True)
class RankingPolicy:
    enabled: bool = True
    stale_penalty: float = 0.6
    requires_review_penalty: float = 0.7
    fact_status_penalty: float = 0.65
    derivation_depth_penalty: float = 0.02
    canonical_boost: float = 1.08
    source_hash_current_boost: float = 1.04
    exact_filter_boost: float = 1.04
    exact_subject_boost: float = 1.10
    exact_fact_key_boost: float = 1.12


@dataclass(frozen=True)
class RankedScore:
    score: float
    base_score: float
    vector_score: float
    debug: dict[str, Any]


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _normalize_key(value: Any) -> str:
    return str(value or "").strip().casefold()


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_depth(payload: dict[str, Any]) -> int:
    raw_depth = payload.get("derivation_depth")
    if raw_depth is not None:
        try:
            return max(0, int(raw_depth))
        except Exception:
            return 0
    derived_from = payload.get("derived_from")
    if isinstance(derived_from, list):
        return len(derived_from)
    if derived_from:
        return 1
    return 0


def query_requests_review_history(query: str) -> bool:
    normalized = _normalize_text(query)
    if not normalized:
        return False
    return any(term in normalized for term in _REVIEW_HISTORY_TERMS)


def _matches_filter(payload: dict[str, Any], key: str, expected: str | None) -> bool:
    if not expected:
        return False
    return _normalize_text(payload.get(key)) == _normalize_text(expected)


def _source_filter_matches(payload: dict[str, Any], expected: str | None) -> bool:
    if not expected:
        return False
    normalized = _normalize_text(expected)
    return any(_normalize_text(payload.get(key)) == normalized for key in ("source", "source_uri", "file_path"))


def rank_memory_candidate(
    *,
    base_score: float,
    vector_score: float,
    payload: dict[str, Any] | None,
    context: RankingContext | None = None,
    policy: RankingPolicy | None = None,
    recency_decay: float | None = None,
) -> RankedScore:
    """Apply provenance-aware ranking adjustments while preserving audit metadata.

    ``base_score`` is the retriever's existing score after vector normalization, importance,
    and recency decay. This policy only adjusts ordering with provenance signals and keeps
    the raw Qdrant/vector score available in ``debug["vector_score"]``.
    """

    payload = payload or {}
    context = context or RankingContext()
    policy = policy or RankingPolicy()
    score = float(base_score)
    vector = float(vector_score)
    memory_kind = str(payload.get("memory_kind") or "")
    fact_status = str(payload.get("fact_status") or "")
    derivation_depth = _as_depth(payload)
    review_history_requested = bool(context.include_fact_history) or query_requests_review_history(context.query)
    boosts: dict[str, float] = {}
    penalties: dict[str, float] = {}

    debug: dict[str, Any] = {
        "enabled": policy.enabled,
        "base_score": float(base_score),
        "score": score,
        "vector_score": vector,
        "importance": payload.get("importance"),
        "recency_decay": recency_decay,
        "memory_kind": memory_kind,
        "fact_status": fact_status,
        "canonical": payload.get("canonical"),
        "stale": payload.get("stale"),
        "requires_review": payload.get("requires_review"),
        "source_hash_current": payload.get("source_hash_current"),
        "derivation_depth": derivation_depth,
        "review_history_requested": review_history_requested,
        "boosts": boosts,
        "penalties": penalties,
    }
    if not policy.enabled:
        return RankedScore(score=score, base_score=float(base_score), vector_score=vector, debug=debug)

    def boost(name: str, factor: float) -> None:
        nonlocal score
        score *= float(factor)
        boosts[name] = float(factor)

    def penalize(name: str, factor: float) -> None:
        nonlocal score
        score *= float(factor)
        penalties[name] = float(factor)

    if not review_history_requested:
        if _is_true(payload.get("stale")):
            penalize("stale", policy.stale_penalty)
        if _is_true(payload.get("requires_review")):
            penalize("requires_review", policy.requires_review_penalty)
        if fact_status in _PENALIZED_FACT_STATUSES:
            penalize(f"fact_status:{fact_status}", policy.fact_status_penalty)

    if _is_true(payload.get("canonical")):
        boost("canonical", policy.canonical_boost)
    if _is_true(payload.get("source_hash_current")):
        boost("source_hash_current", policy.source_hash_current_boost)
    if _source_filter_matches(payload, context.source_filter):
        boost("source_filter", policy.exact_filter_boost)
    if _matches_filter(payload, "file_path", context.file_path_filter):
        boost("file_path_filter", policy.exact_filter_boost)
    if _matches_filter(payload, "project_path", context.project_path_filter):
        boost("project_path_filter", policy.exact_filter_boost)

    subject_target = context.subject or context.query
    if payload.get("subject") and _normalize_text(payload.get("subject")) == _normalize_text(subject_target):
        boost("exact_subject", policy.exact_subject_boost)
    fact_key_target = context.fact_key or context.query
    if payload.get("fact_key") and _normalize_key(payload.get("fact_key")) == _normalize_key(fact_key_target):
        boost("exact_fact_key", policy.exact_fact_key_boost)

    if derivation_depth > 0:
        factor = max(0.5, 1.0 - derivation_depth * float(policy.derivation_depth_penalty))
        penalize("derivation_depth", factor)

    debug["score"] = score
    return RankedScore(score=score, base_score=float(base_score), vector_score=vector, debug=debug)
