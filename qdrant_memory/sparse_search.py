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


# Trailing file-extension pattern matched against the LAST component of a
# slash-path candidate when computing component-aligned substring matches.
# The pattern intentionally matches common short extensions (``.md``,
# ``.markdown`` …) and avoids stripping hyphenated suffixes like
# ``Nucleogenesis-extra``, which would otherwise let a broad directory
# prefix promote unrelated sibling paths.
_PATH_EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,10}$")


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


def _is_code_like_identifier(token: str) -> bool:
    """Heuristic: is ``token`` a code-like identifier rather than plain prose?

    Promotes snake_case (``CODEGRAPH_PROJECT``, ``active_notes``), camelCase /
    PascalCase boundaries (``userId``, ``MemoryRetriever``) and digit-bearing
    identifiers (``Qwen3``, ``bge_m3``). Does NOT promote plain prose words or
    acronyms such as ``CMPC``, ``Nucleogenesis``, ``mcp``, ``home``,
    ``projects``, ``Documentos`` — those carry no structural signal that a
    candidate is the exact identifier the user is looking for, so promoting on
    them alone floods the sparse lane with same-scope but unrelated hits.
    """
    if "_" in token:
        return True
    # digit adjacent to a letter (e.g. Qwen3, v2, utf8)
    if re.search(r"[A-Za-z]\d|\d[A-Za-z]", token):
        return True
    # camelCase boundary: lowercase immediately followed by uppercase
    if re.search(r"[a-z][A-Z]", token):
        return True
    return False


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

# Strong patterns split by the matching strategy the scorer uses.
#
# ``_EXACT_SCORING_PATTERNS`` produce tokens matched by exact token equality
# (UUIDs, issue IDs, error literals, HTTP statuses). Code-like identifiers are
# also matched by equality (added separately by ``_collect_scoring_signals``).
#
# ``_SUBSTRING_SCORING_PATTERNS`` produce compound path-like tokens that are
# matched by safe substring containment inside indexed fields, so a query for
# ``/api/v1/projects`` can match a longer indexed file path such as
# ``/repo/docs/api/v1/projects.md``. These tokens all contain structural
# delimiters (``/``, ``.``, ``::``) which keeps substring matching specific and
# prevents broad single-word collateral hits.
_EXACT_SCORING_PATTERNS: tuple[re.Pattern[str], ...] = (
    _UUID_RE,
    _HEX32_RE,
    _ISSUE_ID_RE,
    _ERROR_RE,
    _HTTP_STATUS_RE,
)
_SUBSTRING_SCORING_PATTERNS: tuple[re.Pattern[str], ...] = (
    _API_ROUTE_RE,
    _SLASH_PATH_RE,
    _DOTTED_RE,
    _COLON_PATH_RE,
)


def has_strong_signal(text: str) -> bool:
    """Return True if ``text`` carries at least one high-confidence exact signal.

    The sparse lane should only fire on queries that look like literal /
    identifier lookups (UUID, issue ID, /api/... route, dotted/colon symbol,
    error literal, HTTP status code, or a code-like identifier such as
    ``CODEGRAPH_PROJECT`` / ``active_notes``). Generic natural-language queries
    should fall back to dense-only so broad semantic search is not flooded with
    literal-token matches.
    """
    if not text:
        return False
    for pattern in _STRONG_SIGNAL_PATTERNS:
        if pattern.search(text):
            return True
    # Code-like identifiers (snake_case, camelCase, digit-bearing) are also
    # strong-signal indicators worth a sparse lookup even when no other
    # structural pattern matches.
    for match in _IDENT_RE.finditer(text):
        token = match.group(0)
        if len(token) < 3:
            continue
        if token.lower() in _STOPWORDS:
            continue
        if _is_code_like_identifier(token):
            return True
    return False


def extract_signals(text: str, *, max_tokens: int = _MAX_QUERY_TOKENS) -> SparseSignals:
    """Extract exact-signal tokens from ``text`` (query string or payload)."""
    return SparseSignals(tokens=tuple(_collect_tokens(text, max_tokens=max_tokens)), raw=text or "")


@dataclass(frozen=True)
class ScoringSignals:
    """Tokens the scorer uses to decide whether a candidate is a literal hit.

    ``exact`` are matched by token equality (UUIDs, issue IDs, error literals,
    HTTP statuses, and code-like identifiers). ``substring`` are compound
    path-like tokens matched by safe substring containment inside indexed
    fields. Broad plain words / path components are deliberately excluded so
    the sparse lane can no longer promote candidates based solely on generic
    identifier overlap (e.g. ``CMPC``, ``mcp``, ``projects``).
    """

    exact: tuple[str, ...] = ()
    substring: tuple[str, ...] = ()

    @property
    def token_set(self) -> set[str]:
        return set(self.exact)

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return bool(self.exact) or bool(self.substring)


def _collect_scoring_signals(text: str, *, max_tokens: int = _MAX_QUERY_TOKENS) -> ScoringSignals:
    """Extract the subset of tokens the sparse scorer may match on.

    Unlike :func:`extract_signals` (which returns every token including broad
    plain words), this helper returns only high-confidence tokens:

    * strong exact tokens from :data:`_EXACT_SCORING_PATTERNS` (UUID, issue
      ID, error literal, HTTP status);
    * compound path-like tokens from :data:`_SUBSTRING_SCORING_PATTERNS`
      (``/api/...`` routes, slash paths, dotted symbols, colon paths);
    * code-like identifiers (snake_case, camelCase, digit-bearing) such as
      ``CODEGRAPH_PROJECT``, ``active_notes``, ``userId``, ``Qwen3``.

    Broad plain words / path components (``CMPC``, ``Nucleogenesis``, ``mcp``,
    ``home``, ``projects``, ``Documentos``, ``implementation``) are dropped
    because matching on them alone floods the sparse lane with same-scope but
    unrelated candidates.
    """
    if not text:
        return ScoringSignals()

    seen_exact: set[str] = set()
    exact: list[str] = []
    seen_sub: set[str] = set()
    substring: list[str] = []

    def add_exact(token: str) -> None:
        norm = _normalize_token(token)
        if norm and norm not in seen_exact:
            seen_exact.add(norm)
            exact.append(norm)

    def add_sub(token: str) -> None:
        norm = _normalize_token(token)
        if norm and norm not in seen_sub:
            seen_sub.add(norm)
            substring.append(norm)

    for pattern in _EXACT_SCORING_PATTERNS:
        for match in pattern.finditer(text):
            add_exact(match.group(0))

    for pattern in _SUBSTRING_SCORING_PATTERNS:
        for match in pattern.finditer(text):
            add_sub(match.group(0))

    # Code-like identifiers from the generic identifier regex. These are
    # matched by exact equality; broad plain words are skipped.
    for match in _IDENT_RE.finditer(text):
        token = match.group(0)
        if len(token) < 3:
            continue
        if token.lower() in _STOPWORDS:
            continue
        if _is_code_like_identifier(token):
            add_exact(token)

    # Bound the combined token list to keep scoring CPU bounded.
    if len(exact) + len(substring) > max_tokens:
        keep = max(1, max_tokens // 2)
        exact = exact[:keep]
        substring = substring[: (max_tokens - keep)]

    return ScoringSignals(exact=tuple(exact), substring=tuple(substring))


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


# ---------------------------------------------------------------------------
# Structural substring matching for compound query tokens
# ---------------------------------------------------------------------------


def _split_structural_components(token: str, delim: str) -> list[str]:
    """Split ``token`` on ``delim`` and drop empty parts (e.g. leading ``/``).

    Used to compare slash paths, dotted symbols, and ``::``-separated FQNs
    at a structural-component level rather than at the raw-character level.
    """
    if not token or not delim:
        return []
    return [part for part in token.split(delim) if part]


def _strip_trailing_extension(component: str) -> str:
    """Strip a trailing file-extension-like suffix from a single path component.

    ``foo.md`` → ``foo``; ``foo.json`` → ``foo``; ``Nucleogenesis-extra`` is
    returned unchanged (the ``-`` separator is intentionally preserved so
    a broad hyphen-delimited sibling segment does not collapse onto a
    shorter query segment). Returns ``component`` unchanged when stripping
    would leave an empty fragment.
    """
    if not component or "." not in component:
        return component
    match = _PATH_EXT_RE.search(component)
    if not match:
        return component
    base = component[: match.start()]
    return base if base else component


def _candidate_structural_components(
    field_text: str, delim: str
) -> list[list[str]]:
    """Return component lists for every structural token in ``field_text``.

    ``delim`` selects the matching regex family (``/`` for slash paths and
    API routes, ``.`` for dotted symbols, ``::`` for FQNs). The LAST path
    component of each token has its trailing extension stripped so a query
    like ``/api/v1/projects`` matches a candidate ``/repo/api/v1/projects.md``.
    Hyphenated suffixes are NOT stripped, so ``Nucleogenesis-extra`` stays
    distinct from ``Nucleogenesis``.
    """
    if not field_text:
        return []
    tokens: list[list[str]] = []
    if delim == "/":
        seen: set[tuple[str, ...]] = set()
        for match in _SLASH_PATH_RE.finditer(field_text):
            comps = _split_structural_components(match.group(0), "/")
            if not comps:
                continue
            comps[-1] = _strip_trailing_extension(comps[-1])
            if comps[-1] and tuple(comps) not in seen:
                seen.add(tuple(comps))
                tokens.append(comps)
        for match in _API_ROUTE_RE.finditer(field_text):
            comps = _split_structural_components(match.group(0), "/")
            if not comps:
                continue
            comps[-1] = _strip_trailing_extension(comps[-1])
            if comps[-1] and tuple(comps) not in seen:
                seen.add(tuple(comps))
                tokens.append(comps)
        return tokens
    if delim == "::":
        for match in _COLON_PATH_RE.finditer(field_text):
            comps = _split_structural_components(match.group(0), "::")
            if comps:
                tokens.append(comps)
        return tokens
    if delim == ".":
        for match in _DOTTED_RE.finditer(field_text):
            comps = _split_structural_components(match.group(0), ".")
            if comps:
                tokens.append(comps)
        return tokens
    return []


def _substring_token_aligned(query_sub: str, field_text: str) -> bool:
    """Return True when ``query_sub`` is a structural-aligned substring of ``field_text``.

    The query token is decomposed into its own structural components
    (slash-delimited path, dot-delimited symbol, or ``::``-delimited FQN).
    The candidate's indexed structural tokens are decomposed the same way.
    A match is recorded when the query's components appear as a contiguous
    slice — in order, but not necessarily at the very start — of some
    candidate token's component list.

    Comparison is case-insensitive on both sides so ``/api/v1/projects``
    continues to match ``/repo/docs/API/V1/PROJECTS.md`` (a candidate
    indexed in uppercase or with mixed casing).

    This replaces raw ``sub.lower() in lowered`` substring containment,
    which suffered from these near-prefix false positives:

    * ``/api/v1/project`` matching ``/repo/docs/api/v1/projects.md``;
    * ``pkg.mod`` matching ``pkg.module.Class``;
    * ``/home/.../Nucleogenesis`` matching
      ``/home/.../Nucleogenesis-extra/README.md``.

    Tokens with a single structural component (e.g. bare ``CMPC`` or
    ``projects`` with no delimiter) cannot be aligned structurally and fall
    back to the existing exact-equality branch, so broad plain words never
    promote unrelated candidates.
    """
    if not query_sub or not field_text:
        return False
    # Pick the strongest structural delimiter the query carries; the
    # ordering prefers slash, then FQN ``::``, then dotted, because each
    # regex set may produce overlapping matches and we want slash-path
    # tokens evaluated against slash-path tokens.
    if "/" in query_sub:
        delim = "/"
    elif "::" in query_sub:
        delim = "::"
    elif "." in query_sub:
        delim = "."
    else:
        return False

    qcomps = _split_structural_components(query_sub, delim)
    if len(qcomps) < 2:
        # A single-component query cannot anchor a substring match: it
        # would either match the entire candidate (handled by exact
        # equality elsewhere) or nothing. Returning False keeps broad
        # plain words from promoting unrelated paths.
        return False
    # Lowercase the query components for case-insensitive comparison.
    qcomps_lc = [c.lower() for c in qcomps]
    # Also strip trailing extension on the query's last component so a
    # user-typed ``/api/v1/projects.md`` matches the same files as
    # ``/api/v1/projects``.
    if delim == "/":
        qcomps_lc[-1] = _strip_trailing_extension(qcomps_lc[-1])
        if not qcomps_lc[-1]:
            return False

    qlen = len(qcomps_lc)
    candidates = _candidate_structural_components(field_text, delim)
    for ccomps in candidates:
        ccomps_lc = [c.lower() for c in ccomps]
        if len(ccomps_lc) < qlen:
            continue
        for start in range(0, len(ccomps_lc) - qlen + 1):
            if ccomps_lc[start : start + qlen] == qcomps_lc:
                return True
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

    Phase 6D hardening: scoring only matches high-confidence exact/literal
    signals (UUIDs, issue IDs, error literals, HTTP statuses, code-like
    identifiers, and compound path-like tokens by structural-component
    alignment). Broad plain words / generic path components (``CMPC``,
    ``mcp``, ``projects``, ``home``, ``Documentos``) are NOT scored on
    their own, so a candidate can no longer be promoted based solely on
    generic identifier overlap with the query.
    """
    signals = _collect_scoring_signals(query)
    exact_set = signals.token_set
    substring_tokens = signals.substring
    out: list[SparseScore] = []
    if not exact_set and not substring_tokens:
        # No scoring signals to match; return zero scores so callers can short
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

        # Exact-equality matches come from tokenizing the candidate's compact
        # metadata (e.g. an issue ID appearing in ``heading``). Substring
        # matches come from a structural-component alignment between a
        # compound path-like query token (e.g. ``/api/v1/projects``) and the
        # candidate's indexed structural tokens (e.g. the slash-path token
        # ``/repo/docs/api/v1/projects.md`` inside the candidate's
        # ``file_path`` field).
        field_hits: dict[str, int] = {}
        matched: set[str] = set()
        for field_name, field_text in _iter_candidate_text_fields(payload):
            field_hit_count = 0
            # Exact-equality: tokenize the field and intersect with exact_set.
            tokens = _collect_tokens(field_text, max_tokens=_MAX_CANDIDATE_TOKENS)
            for tok in tokens:
                if tok in exact_set:
                    field_hit_count += 1
                    matched.add(tok)
            # Substring containment: structural alignment of compound query
            # tokens against the candidate's structural tokens. Replaces the
            # previous raw ``sub.lower() in lowered`` check, which produced
            # near-prefix false positives on path-like tokens.
            if substring_tokens:
                for sub in substring_tokens:
                    if _substring_token_aligned(sub, field_text):
                        field_hit_count += 1
                        matched.add(sub)
            if field_hit_count:
                field_hits[field_name] = field_hit_count

        # Point ID literal hit is a strong signal.
        literal_hit = False
        for tok in exact_set:
            if pid and tok and (tok == pid or tok in pid):
                literal_hit = True
                matched.add(tok)
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
                matched_tokens=sorted(matched),
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
    "ScoringSignals",
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