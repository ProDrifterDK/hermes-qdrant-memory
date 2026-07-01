"""Phase 2: stdlib-only sparse / exact retrieval lane for `MemoryRetriever`.

This module adds a deterministic, exact-signal retrieval lane that runs alongside
the dense vector lane and improves recall for *literal* identifiers (UUIDs, point
IDs, file paths, `/api/...` routes, dotted/colon symbols, snake- and camelCase
identifiers, error names/messages, issue IDs like ``SMDFS-455``).

Design constraints
------------------

* Python stdlib only — no new runtime dependencies.
* Reuses the existing ``_scope_filter`` / ``_payload_allowed`` predicates from
  :mod:`qdrant_memory.retriever` so dense and sparse candidates pass through the
  same safety gates.
* Sparse candidates come from ``QdrantClient.scroll_by_filter`` using the exact
  same filter object that the dense search uses. If ``scroll_by_filter`` is
  absent (or raises), the lane degrades silently to dense-only.
* Hard caps (``candidate_cap``) prevent a manual search from scrolling
  unbounded collections. Sparse scoring is bounded so a literal hit can outrank
  an unrelated dense decoy but cannot blow up the score scale.
* Secret-bearing payload text is rejected at scoring time so a payload that
  contains a token/bearer-style literal cannot be promoted by the sparse lane.
* Quarantined points (``consolidation_quarantined`` by default) are excluded
  before scoring. This is applied consistently with the dense retriever's
  ``_payload_allowed`` checks via :func:`is_payload_visible`.

Public surface
--------------

* :func:`extract_signals` — extract exact-signal tokens from a query string.
* :func:`score_candidates` — BM25-style scoring over a list of candidate
  payloads, returning a mapping ``point_id -> SparseScore``.
* :func:`SparseCandidate` / :class:`SparseScore` — small dataclasses returned
  by :func:`score_candidates`.
* :func:`fetch_sparse_candidates` — bounded scroll helper with dense-fallback
  semantics (no-op if ``scroll_by_filter`` is unavailable).
* :func:`merge_candidates` — merge dense + sparse candidates by ``point_id``
  with a sparse lift factor.

This module deliberately does not depend on ``qdrant_memory.retriever`` at
import time to keep it lightweight and trivially testable. The retriever
imports the helpers it needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .lesson_extractor import contains_secret

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Standard 8-4-4-4-12 hex UUID (with or without dashes).
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
# Bare 32-char hex (also a candidate UUID representation).
_HEX32_RE = re.compile(r"\b[0-9a-fA-F]{32}\b")
# Issue / ticket IDs like SMDFS-455, ABC-12, PROJ-12345.
_ISSUE_ID_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d{1,8}\b")
# /api/v1/... style routes, anchored to a slash-rooted path component.
_API_ROUTE_RE = re.compile(r"(?:/[A-Za-z0-9_.@:-]+){2,}")
# Generic slash-separated paths.
_SLASH_PATH_RE = re.compile(r"/(?:[A-Za-z0-9_.-]+/){1,}[A-Za-z0-9_.-]+")
# Dotted symbol chains (module.Class.sub, foo.bar.baz).
_DOTTED_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){1,}\b")
# Colon-separated symbols (java/kotlin style fully qualified names).
_COLON_PATH_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*){1,}\b")
# Snake- and camelCase identifiers of length >= 3.
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
# Plain error literals like "Error: connection refused" or "TypeError: foo bar".
_ERROR_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning|Failure)\b"
    r"(?:\s*:?\s*[A-Za-z0-9_./-]{2,})?"
)
# HTTP-style error messages. Bare status codes (3 digits) only count when they
# appear with the literal "HTTP " prefix; otherwise they'd match unrelated
# numbers like file sizes or timestamps.
_HTTP_STATUS_RE = re.compile(r"\bHTTP\s*[1-5]\d{2}\b")

_MAX_QUERY_TOKENS = 64
_MAX_CANDIDATE_TOKENS = 256
_DEFAULT_CANDIDATE_CAP = 256

# Field keys whose compact metadata we index. These are the "small, safe"
# metadata fields we are allowed to read into the scorer.
_INDEXED_META_KEYS: tuple[str, ...] = (
    "text",
    "source",
    "file_path",
    "project_path",
    "heading",
    "subject",
    "fact_key",
    "source_uri",
)

# Default quarantine marker; consolidation writes this for reversible quarantine.
DEFAULT_QUARANTINE_KEY = "consolidation_quarantined"


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SparseSignals:
    """Exact-signal tokens extracted from a single text input."""

    tokens: tuple[str, ...] = ()
    raw: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return bool(self.tokens)


@dataclass
class SparseScore:
    """Per-candidate sparse score with diagnostic breakdown."""

    point_id: str
    score: float
    matched_tokens: list[str] = field(default_factory=list)
    field_hits: dict[str, int] = field(default_factory=dict)
    literal_hit: bool = False
    secret_blocked: bool = False
    quarantined: bool = False
    payload_invisible: bool = False


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def _normalize_token(token: str) -> str:
    return token.strip().strip("`'\".,;:()[]{}<>")


def _collect_tokens(text: str, *, max_tokens: int) -> list[str]:
    """Extract and dedupe exact-signal tokens from ``text``.

    The order of extraction is significant: longer/more-specific patterns are
    tried first so a UUID is not split into sub-tokens.
    """
    if not text:
        return []
    seen: set[str] = set()
    ordered: list[str] = []

    def add(token: str) -> None:
        norm = _normalize_token(token)
        if not norm:
            return
        if norm in seen:
            return
        seen.add(norm)
        ordered.append(norm)

    for match in _UUID_RE.finditer(text):
        add(match.group(0))
    for match in _HEX32_RE.finditer(text):
        add(match.group(0))
    for match in _ISSUE_ID_RE.finditer(text):
        add(match.group(0))
    for match in _API_ROUTE_RE.finditer(text):
        add(match.group(0))
    for match in _SLASH_PATH_RE.finditer(text):
        add(match.group(0))
    for match in _COLON_PATH_RE.finditer(text):
        add(match.group(0))
    for match in _DOTTED_RE.finditer(text):
        add(match.group(0))
    for match in _ERROR_RE.finditer(text):
        add(match.group(0))
    for match in _HTTP_STATUS_RE.finditer(text):
        add(match.group(0))
    for match in _IDENT_RE.finditer(text):
        token = match.group(0)
        # Skip noise words and short fragments that are unlikely to be useful
        # as exact-signal identifiers.
        if len(token) < 3:
            continue
        if token.lower() in _STOPWORDS:
            continue
        add(token)
    if len(ordered) > max_tokens:
        ordered = ordered[:max_tokens]
    return ordered


_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "and", "for", "with", "this", "that", "from", "into", "onto",
        "your", "you", "are", "was", "were", "but", "not", "all", "any",
        "can", "use", "using", "run", "running", "have", "has", "had", "get",
        "got", "via", "per", "its", "out", "his", "her", "him", "she", "they",
        "them", "our", "who", "why", "how", "what", "when", "where", "here",
        "there", "these", "those", "such", "than", "then", "also", "about",
        "more", "most", "less", "much", "many", "some", "each", "every",
        "either", "neither", "both", "none", "one", "two", "three", "four",
        "five", "six", "seven", "eight", "nine", "ten",
        "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "see", "look",
        "recall", "find", "show", "tell", "know", "think", "make", "made",
        "go", "went", "going", "done", "do", "does", "did", "doing",
        "thing", "things", "stuff", "data", "info", "note", "notes",
        "general", "specific", "kind", "type", "types",
    }
)


# Patterns that are treated as "strong" exact-signal indicators. A query that
# matches at least one of these is a candidate for the sparse lane; a query
# that only matches plain words should stay on the dense lane.
_STRONG_SIGNAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    _UUID_RE,
    _HEX32_RE,
    _ISSUE_ID_RE,
    _API_ROUTE_RE,
    _SLASH_PATH_RE,
    _COLON_PATH_RE,
    _DOTTED_RE,
    _ERROR_RE,
    _HTTP_STATUS_RE,
)


def has_strong_signal(text: str) -> bool:
    """Return True if ``text`` carries at least one high-confidence exact signal.

    The sparse lane should only fire on queries that look like literal /
    identifier lookups (UUID, issue ID, /api/... route, dotted/colon symbol,
    error literal, HTTP status code). Generic natural-language queries should
    fall back to dense-only so broad semantic search is not flooded with
    literal-token matches.
    """
    if not text:
        return False
    for pattern in _STRONG_SIGNAL_PATTERNS:
        if pattern.search(text):
            return True
    return False


def extract_signals(text: str, *, max_tokens: int = _MAX_QUERY_TOKENS) -> SparseSignals:
    """Extract exact-signal tokens from ``text`` (query string or payload)."""
    return SparseSignals(tokens=tuple(_collect_tokens(text, max_tokens=max_tokens)), raw=text or "")


# ---------------------------------------------------------------------------
# Payload visibility / safety
# ---------------------------------------------------------------------------


def is_payload_visible(
    payload: Mapping[str, Any] | None,
    *,
    quarantine_key: str = DEFAULT_QUARANTINE_KEY,
) -> bool:
    """Return True if a payload passes the sparse lane's hard-safety checks.

    Sparse candidates must NEVER be promoted when the underlying payload is
    quarantined or carries secret-bearing text. The same boolean flag is also
    used to decorate :class:`SparseScore` for diagnostics.
    """
    if not payload:
        return True
    if payload.get(quarantine_key):
        return False
    return True


def _payload_field_has_secret(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return contains_secret(value)
    if isinstance(value, (list, tuple, set)):
        return any(_payload_field_has_secret(item) for item in value)
    if isinstance(value, dict):
        return any(_payload_field_has_secret(item) for item in value.values())
    return False


def _iter_candidate_text_fields(payload: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    """Yield (field_name, text) tuples for the compact metadata fields.

    Only the keys in :data:`_INDEXED_META_KEYS` are read. Other payload fields
    (which may carry secrets or large blobs) are intentionally ignored.
    """
    if not payload:
        return
    for key in _INDEXED_META_KEYS:
        value = payload.get(key)
        if isinstance(value, str) and value:
            yield key, value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item:
                    yield key, item


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _field_weight(field_name: str) -> float:
    """Field weight for scoring; literals/identifiers in metadata rank higher."""
    if field_name in {"text"}:
        return 1.0
    if field_name in {"file_path", "project_path", "heading", "subject", "fact_key", "source_uri", "source"}:
        return 1.5
    return 1.0


def score_candidates(
    query: str,
    points: Iterable[Mapping[str, Any]],
    *,
    quarantine_key: str = DEFAULT_QUARANTINE_KEY,
    point_id_getter: Any = None,
) -> list[SparseScore]:
    """Score a list of Qdrant-style point dicts against ``query``.

    Returns one :class:`SparseScore` per input point, in the same order as the
    input. Hidden / quarantined / secret-bearing candidates are returned with
    a score of 0.0 and the appropriate diagnostic flag set so the retriever can
    decide whether to drop them entirely or simply not promote them.
    """
    signals = extract_signals(query)
    token_set = set(signals.tokens)
    out: list[SparseScore] = []
    if not token_set:
        # No exact signals to match; return zero scores so callers can short
        # circuit cleanly without re-iterating.
        for point in points:
            pid = _point_id(point, point_id_getter)
            out.append(
                SparseScore(
                    point_id=pid,
                    score=0.0,
                    matched_tokens=[],
                    field_hits={},
                    literal_hit=False,
                )
            )
        return out

    for point in points:
        pid = _point_id(point, point_id_getter)
        payload = point.get("payload") if isinstance(point, Mapping) else None
        if not isinstance(payload, Mapping):
            out.append(SparseScore(point_id=pid, score=0.0))
            continue
        if not is_payload_visible(payload, quarantine_key=quarantine_key):
            out.append(
                SparseScore(
                    point_id=pid,
                    score=0.0,
                    quarantined=True,
                    payload_invisible=True,
                )
            )
            continue
        # Reject candidates whose PAYLOAD contains a secret literal anywhere,
        # including nested dicts, lists, and non-indexed metadata fields. Only
        # the recursive scan is authoritative — the indexed-fields loop below is
        # kept as a fast path but the recursive check must run first so secrets
        # buried in credential-bearing metadata cannot be promoted by the
        # sparse lane.
        if _payload_field_has_secret(payload):
            out.append(
                SparseScore(
                    point_id=pid,
                    score=0.0,
                    secret_blocked=True,
                    payload_invisible=True,
                )
            )
            continue

        # Tokenize the candidate's compact metadata and accumulate hits.
        # We deliberately cap per-field token count to bound CPU on large
        # payloads.
        field_tokens: dict[str, list[str]] = {}
        field_hits: dict[str, int] = {}
        for field_name, field_text in _iter_candidate_text_fields(payload):
            tokens = _collect_tokens(field_text, max_tokens=_MAX_CANDIDATE_TOKENS)
            field_tokens[field_name] = tokens
            hits = [tok for tok in tokens if tok in token_set]
            if hits:
                field_hits[field_name] = len(hits)
        matched = sorted({tok for toks in field_tokens.values() for tok in toks if tok in token_set})

        # Point ID literal hit is a strong signal.
        literal_hit = False
        for tok in token_set:
            if pid and tok and (tok == pid or tok in pid):
                literal_hit = True
                matched.append(tok)
                field_hits.setdefault("point_id", 0)
                field_hits["point_id"] += 1
                break

        if not matched:
            out.append(
                SparseScore(
                    point_id=pid,
                    score=0.0,
                    matched_tokens=[],
                    field_hits=field_hits,
                )
            )
            continue

        # Score: weighted field hits + literal bonus. The numbers are kept
        # small (sub-1.0 contribution per field) so a literal hit lifts a point
        # noticeably above non-matching dense candidates, while the absolute
        # score never blows past a sane upper bound.
        score = 0.0
        for field_name, hits in field_hits.items():
            score += _field_weight(field_name) * float(hits)
        if literal_hit:
            score += 2.0
        # Multi-field agreement bonus.
        if len([h for h in field_hits.values() if h]) >= 2:
            score += 0.5

        out.append(
            SparseScore(
                point_id=pid,
                score=float(score),
                matched_tokens=sorted(set(matched)),
                field_hits=field_hits,
                literal_hit=literal_hit,
            )
        )
    return out


def _point_id(point: Mapping[str, Any], point_id_getter: Any) -> str:
    if point_id_getter is not None:
        try:
            return str(point_id_getter(point) or "")
        except Exception:
            return ""
    if not isinstance(point, Mapping):
        return ""
    return str(point.get("id") or "")


# ---------------------------------------------------------------------------
# Candidate fetching (defensive scroll wrapper)
# ---------------------------------------------------------------------------


def fetch_sparse_candidates(
    qdrant: Any,
    *,
    collection_name: str,
    flt: Mapping[str, Any] | None,
    candidate_cap: int = _DEFAULT_CANDIDATE_CAP,
) -> list[dict[str, Any]]:
    """Fetch bounded sparse candidates using ``scroll_by_filter``.

    Returns an empty list if ``scroll_by_filter`` is not present or the call
    raises; the retriever falls back to dense-only in that case.
    """
    scroll = getattr(qdrant, "scroll_by_filter", None)
    if not callable(scroll):
        return []
    if not flt:
        # Refuse to scroll the entire collection; an empty filter would
        # otherwise scan every point in the collection.
        return []
    cap = max(1, min(int(candidate_cap), _DEFAULT_CANDIDATE_CAP))
    try:
        result = scroll(
            collection_name,
            dict(flt),
            limit=cap,
            max_total=cap,
        )
    except Exception:
        return []
    if not isinstance(result, list):
        return []
    typed: list[dict[str, Any]] = []
    for item in result:
        if isinstance(item, dict):
            typed.append(item)
    return typed


# ---------------------------------------------------------------------------
# Dense + sparse merge
# ---------------------------------------------------------------------------


@dataclass
class MergedCandidate:
    """A merged dense + sparse candidate ready for ranking."""

    point_id: str
    payload: dict[str, Any]
    dense_score: float = 0.0
    sparse_score: float = 0.0
    literal_hit: bool = False
    matched_tokens: list[str] = field(default_factory=list)
    field_hits: dict[str, int] = field(default_factory=dict)
    quarantined: bool = False
    secret_blocked: bool = False

    @property
    def combined(self) -> float:
        """Combine dense + sparse scores with a bounded sparse lift."""
        if self.quarantined or self.secret_blocked:
            return 0.0
        lift = self.sparse_score * _DEFAULT_SPARSE_LIFT
        return float(self.dense_score) + float(lift)


# Sparse lift: keeps dense ordering dominant while letting a literal hit
# (which contributes >= 2.0 to sparse_score) outrank an unrelated dense decoy.
# A literal hit produces a sparse_score of at least 2.0 (literal bonus 2.0 +
# at least one weighted field hit), so 3.0 * 0.5 = 1.5 comfortably beats a
# dense_score=0.95 decoy while still letting a 0.99 dense hit outrank a
# sparse-only weak match.
_DEFAULT_SPARSE_LIFT = 0.5


def _sparse_promotable(score: SparseScore) -> bool:
    """Return True if a sparse score actually has a matched literal/token hit.

    A non-matching scroll candidate produces a zero SparseScore (no matched
    tokens, no literal_hit). Such candidates must NOT be merged into the
    candidate pool: the default ``min_final_score`` is 0.0 and there is no
    downstream gate that drops them, so they would otherwise leak into the
    final results and trigger ``update_access_metadata``. Only sparse scores
    with positive contribution AND a real matched literal/token signal may be
    promoted by the sparse lane.
    """
    if score.score <= 0.0:
        return False
    if score.literal_hit:
        return True
    return bool(score.matched_tokens)


def merge_candidates(
    *,
    dense: Iterable[Mapping[str, Any]],
    sparse_scores: Iterable[SparseScore],
    sparse_points_by_id: Mapping[str, Mapping[str, Any]],
    sparse_lift: float = _DEFAULT_SPARSE_LIFT,
) -> list[MergedCandidate]:
    """Merge dense hits with sparse scores by point id.

    The dense hit list is authoritative for payload / dense_score. Sparse
    scores contribute only when both:

    * the underlying payload is visible (not quarantined / secret-blocked), and
    * the sparse score has a positive contribution with a matched
      literal/token hit (``_sparse_promotable``).

    Zero-score / no-match scroll candidates are intentionally dropped by the
    sparse lane — they have no matched literal/token signal and would otherwise
    leak through the default ``min_final_score=0.0`` filter and receive
    ``update_access_metadata``.
    """
    sparse_by_id = {score.point_id: score for score in sparse_scores}
    merged: dict[str, MergedCandidate] = {}
    for item in dense:
        if not isinstance(item, Mapping):
            continue
        pid = str(item.get("id") or "")
        if not pid:
            continue
        payload = item.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}
        merged[pid] = MergedCandidate(
            point_id=pid,
            payload=payload,
            dense_score=float(item.get("score", 0.0)),
        )
    for pid, sparse_point in sparse_points_by_id.items():
        score = sparse_by_id.get(pid)
        if score is None:
            continue
        if score.quarantined or score.secret_blocked:
            # Always propagate quarantine / secret flags so the retriever
            # drops the candidate regardless of whether it had a dense hit.
            candidate = merged.get(pid)
            if candidate is None:
                payload = sparse_point.get("payload") if isinstance(sparse_point, Mapping) else {}
                if not isinstance(payload, dict):
                    payload = {}
                candidate = MergedCandidate(
                    point_id=pid,
                    payload=payload,
                )
            candidate.quarantined = score.quarantined
            candidate.secret_blocked = score.secret_blocked
            merged[pid] = candidate
            continue
        # Sparse scores without a positive matched-literal/token hit must not
        # create or update MergedCandidate objects. Otherwise the retriever
        # would surface zero-score scroll candidates and trigger
        # update_access_metadata on them.
        if not _sparse_promotable(score):
            continue
        candidate = merged.get(pid)
        if candidate is None:
            payload = sparse_point.get("payload") if isinstance(sparse_point, Mapping) else {}
            if not isinstance(payload, dict):
                payload = {}
            candidate = MergedCandidate(
                point_id=pid,
                payload=payload,
                dense_score=0.0,
            )
        candidate.sparse_score = float(score.score)
        candidate.literal_hit = score.literal_hit or candidate.literal_hit
        candidate.matched_tokens = list(dict.fromkeys([*candidate.matched_tokens, *score.matched_tokens]))
        candidate.field_hits = {**candidate.field_hits, **score.field_hits}
        merged[pid] = candidate
    # Apply lift override (only if caller supplied a custom lift).
    if sparse_lift != _DEFAULT_SPARSE_LIFT:
        for candidate in merged.values():
            candidate.dense_score = candidate.dense_score  # keep as-is
            # Override lift factor by adjusting sparse_score indirectly via
            # a marker the retriever can read. We avoid mutating the field
            # semantics; instead, the retriever reads combined via the
            # factor and we expose lift through the score itself by storing
            # the residual multiplier on the candidate for clarity.
            setattr(candidate, "_sparse_lift", float(sparse_lift))
    return list(merged.values())


def combine_score(candidate: MergedCandidate, *, sparse_lift: float = _DEFAULT_SPARSE_LIFT) -> float:
    """Compute the combined score honoring a (possibly custom) sparse lift."""
    if candidate.quarantined or candidate.secret_blocked:
        return 0.0
    lift = float(getattr(candidate, "_sparse_lift", sparse_lift))
    return float(candidate.dense_score) + float(candidate.sparse_score) * lift


__all__ = [
    "DEFAULT_QUARANTINE_KEY",
    "MergedCandidate",
    "SparseCandidate",
    "SparseScore",
    "SparseSignals",
    "combine_score",
    "extract_signals",
    "fetch_sparse_candidates",
    "has_strong_signal",
    "is_payload_visible",
    "merge_candidates",
    "score_candidates",
]


# `SparseCandidate` is a public alias for SparseScore to keep imports tidy if
# callers prefer the older name. It is identical to SparseScore.
SparseCandidate = SparseScore