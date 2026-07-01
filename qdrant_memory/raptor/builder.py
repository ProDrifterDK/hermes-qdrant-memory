"""Deterministic dry-run RAPTOR builder (Phase 3).

Consumes plain Qdrant-style leaf point dicts of the form
``{"id": "<uuid>", "payload": {...}}`` and returns a
:class:`~qdrant_memory.raptor.schema.RaptorBuildManifest` describing the
candidate tree and node payloads.

Hard guarantees:

1. **No Qdrant mutation.** The builder accepts plain Python dicts only. It
   never imports or touches ``qdrant_client`` and has no I/O surface. The
   ``dry_run`` / ``mutations_performed`` flags on the manifest are pinned
   to ``True`` / ``False``.
2. **MVP summaries are extractive.** Each cluster's summary text is built
   from the child leaf snippets (one snippet per leaf, joined). No LLM
   call, no novel claims, no abstractive rewriting.
3. **Unsafe leaves are skipped.** Leaves are skipped (and recorded in
   ``skipped_leaves``) when any of the following holds:

   - missing point ``id``
   - missing or empty ``payload.text`` / ``payload.lesson``
   - payload/text contains a known secret shape (see
     :func:`qdrant_memory.lesson_extractor.contains_secret`)
   - payload flagged ``consolidation_quarantined=True``
   - payload ``stale=True`` or ``requires_review=True``
   - payload ``fact_status`` in ``{stale, deprecated, superseded,
     disputed, review_required}``

4. **Cross-scope leaves never cluster together.** Different
   ``profile_id`` / ``user_id_hash`` / ``chat_id_hash`` tuples split into
   separate RAPTOR trees. Scope fields are propagated only when all
   leaves in a cluster agree.
5. **Determinism.** Leaves are sorted by point id before clustering.
   Cluster ids, node ids, tree ids, build ids, and the manifest digest
   are all sha256-based and stable across runs given the same input and
   config.
6. **Clustering is intentionally simple.** Group leaves by
   ``(source_type, project_path or file_path or heading or token_bucket)``
   and chunk by ``max_cluster_size``. No embeddings, no KMeans, no LLM.
7. **Payload safety.** Caller-supplied ``extra`` keys are filtered through
   :func:`qdrant_memory.raptor.schema._safe_extra` so reserved / secret
   shapes can never re-enter the candidate payload via the metadata path.

The builder returns a fully deterministic manifest. Repeating the build
over the same leaves + config yields byte-identical JSON.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from qdrant_memory.lesson_extractor import contains_secret
from qdrant_memory.raptor.schema import (
    DEFAULT_PROMPT_VERSION,
    RAPTOR_DERIVATION_TYPE,
    RAPTOR_LEVEL_LEAF,
    RaptorBuildManifest,
    RaptorCluster,
    RaptorNode,
    RaptorScope,
    RaptorTree,
    _safe_extra,
    _sorted_unique,
    _stringify,
    compute_build_id,
    compute_manifest_digest,
    compute_node_id,
    compute_root_id,
    compute_tree_id,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_CLUSTER_SIZE = 8
DEFAULT_SNIPPET_CHARS = 240

_UNSAFE_FACT_STATUSES: frozenset[str] = frozenset(
    {
        "stale",
        "deprecated",
        "superseded",
        "disputed",
        "review_required",
    }
)

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "have",
        "has",
        "had",
        "was",
        "were",
        "are",
        "but",
        "not",
        "you",
        "your",
        "out",
        "all",
        "any",
        "can",
        "use",
        "used",
        "uses",
        "using",
        "via",
        "per",
        "one",
        "two",
        "three",
        "its",
        "also",
        "such",
        "these",
        "those",
        "when",
        "then",
        "than",
        "over",
        "more",
        "most",
        "some",
        "other",
        "into",
        "onto",
        "off",
        "our",
        "their",
        "them",
        "they",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "shall",
        "must",
        "need",
        "needs",
        "needed",
        "each",
        "every",
        "where",
        "what",
        "which",
        "while",
        "about",
        "because",
    }
)


# ---------------------------------------------------------------------------
# Leaf normalization
# ---------------------------------------------------------------------------


@dataclass
class _Leaf:
    point_id: str
    payload: dict[str, Any]
    text: str
    content_hash: str
    bucket: str
    scope: RaptorScope
    fact_status: str
    stale: bool
    requires_review: bool
    quarantined: bool
    secret: bool
    accepted: bool
    skip_reason: str = ""


def _normalize_scope(payload: Mapping[str, Any]) -> RaptorScope:
    return RaptorScope(
        profile_id=_stringify(payload.get("profile_id")),
        user_id_hash=_stringify(payload.get("user_id_hash")),
        chat_id_hash=_stringify(payload.get("chat_id_hash")),
    )


def _fact_status(payload: Mapping[str, Any]) -> str:
    text = _stringify(payload.get("fact_status")).lower()
    if text:
        return text
    if payload.get("stale") is True:
        return "stale"
    return "active"


def _normalize_text(text: str) -> str:
    """Squash whitespace for stable hashing / comparison."""
    return " ".join(str(text or "").split())


def _payload_content_hash(point_id: str, payload: Mapping[str, Any]) -> str:
    """Stable content hash per leaf, deterministic across runs."""
    blob = {
        "id": point_id,
        "text": _stringify(payload.get("text") or payload.get("lesson") or ""),
        "source_uri": _stringify(payload.get("source_uri")),
        "source_type": _stringify(payload.get("source_type")),
        "locator": payload.get("locator") if isinstance(payload.get("locator"), Mapping) else None,
        "tags": sorted(_stringify(t) for t in payload.get("tags", []) if isinstance(payload.get("tags"), list)),
    }
    encoded = json.dumps(blob, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _bucket_key(payload: Mapping[str, Any]) -> str:
    """Deterministic primary bucket for a leaf.

    Order of preference:

    1. explicit ``raptor_bucket_key`` if present (test override path)
    2. ``source_type``
    3. ``project_path``
    4. ``file_path``
    5. ``heading`` (truncated)
    6. first significant token bucket
    7. literal ``"default"``
    """
    for key in ("raptor_bucket_key", "source_type", "project_path", "file_path"):
        value = _stringify(payload.get(key))
        if value:
            return value
    heading = _stringify(payload.get("heading"))
    if heading:
        normalized = unicodedata.normalize("NFKD", heading).strip().lower()
        normalized = " ".join(normalized.split())
        if normalized:
            return f"heading:{normalized[:60]}"
    text = _stringify(payload.get("text") or payload.get("lesson") or "")
    tokens = [tok for tok in _TOKEN_RE.findall(text.lower()) if tok and tok not in _STOPWORDS and len(tok) > 2]
    if tokens:
        return f"tokens:{tokens[0]}"
    return "default"


def _is_safe_leaf(point: Mapping[str, Any]) -> tuple[bool, str, dict[str, Any], str, str]:
    """Validate a leaf point. Returns ``(accepted, reason, payload, text, point_id)``.

    *accepted=False* means the leaf must be skipped entirely (unsafe).
    *reason* is one of the stable skip-reason codes.
    """
    point_id = _stringify(point.get("id"))
    raw_payload = point.get("payload")
    payload: dict[str, Any] = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}

    if not point_id:
        return False, "missing_point_id", payload, "", ""
    # Reject secret-shaped point ids *before* the shape check. Many
    # secret-scanner patterns (``-----BEGIN ... PRIVATE KEY-----``,
    # ``https://user:pass@host`` basic-auth URLs, ``bearer <token>``
    # contextual strings with spaces/slashes) intentionally fail
    # :func:`_valid_point_id_shape`, and they must still classify as
    # ``secret_id_bearing`` so the build loop applies the
    # :func:`_safe_handle_for_point_id` redaction. Letting the shape
    # check fire first would copy the raw secret verbatim into
    # ``manifest.skipped_leaves[].point_id``.
    if contains_secret(point_id):
        return False, "secret_id_bearing", payload, "", point_id
    if not _valid_point_id_shape(point_id):
        return False, "invalid_point_id", payload, "", point_id

    text = _stringify(payload.get("text") or payload.get("lesson"))
    if not text:
        return False, "missing_text", payload, "", point_id

    payload_blob = json.dumps(payload, sort_keys=True, default=str)
    if contains_secret(text) or contains_secret(payload_blob):
        return False, "secret_bearing", payload, text, point_id

    if payload.get("consolidation_quarantined") is True:
        return False, "quarantined", payload, text, point_id

    if payload.get("stale") is True or payload.get("requires_review") is True:
        return False, "unsafe_flag", payload, text, point_id

    status = _fact_status(payload)
    if status in _UNSAFE_FACT_STATUSES:
        return False, f"fact_status:{status}", payload, text, point_id

    return True, "ok", payload, text, point_id


_POINT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")


def _valid_point_id_shape(point_id: str) -> bool:
    """Match the schema module's point-id regex."""
    return bool(point_id) and bool(_POINT_ID_RE.match(point_id))


# When a leaf is skipped because its point id itself matches the secret
# scanner we must not echo the raw id into manifests, summary text, or
# skipped-leaf handles. ``_safe_handle_for_point_id`` returns a deterministic
# sha256-prefixed handle that contains no portion of the original value and
# is safe to log. The 16-hex prefix keeps it compact while still uniquely
# identifying the skipped leaf across runs.
def _safe_handle_for_point_id(point_id: str) -> str:
    raw = (point_id or "").strip()
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"redacted:{digest}"


def _normalize_leaf(point: Mapping[str, Any]) -> _Leaf | None:
    accepted, reason, payload, text, point_id = _is_safe_leaf(point)
    leaf = _Leaf(
        point_id=point_id or "",
        payload=payload,
        text=text or "",
        content_hash=_payload_content_hash(point_id, payload) if accepted else "",
        bucket=_bucket_key(payload),
        scope=_normalize_scope(payload),
        fact_status=_fact_status(payload),
        stale=bool(payload.get("stale")),
        requires_review=bool(payload.get("requires_review")),
        quarantined=bool(payload.get("consolidation_quarantined")),
        secret=False,
        accepted=accepted,
        skip_reason=reason,
    )
    return leaf


# ---------------------------------------------------------------------------
# Summary text generation (extractive only)
# ---------------------------------------------------------------------------


def _snippet(text: str, max_chars: int = DEFAULT_SNIPPET_CHARS) -> str:
    text = _normalize_text(text)
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _leaf_snippet(leaf: _Leaf, max_chars: int = DEFAULT_SNIPPET_CHARS) -> str:
    text = _normalize_text(leaf.text)
    if not text:
        return ""
    if contains_secret(text):
        return "[redacted: possible secret-bearing memory]"
    return _snippet(text, max_chars=max_chars)


def _cluster_summary_text(leaves: Sequence[_Leaf], *, max_chars: int = 1024) -> str:
    """Constrained, extractive summary from leaf snippets only.

    Output is a deterministic, child-anchored digest:

    * one ``- <point_id>: <snippet>`` line per leaf (in point-id order)
    * bounded to ``max_chars`` to avoid bloating the manifest

    No novel claims are generated.
    """
    lines: list[str] = []
    ordered = sorted(leaves, key=lambda leaf: leaf.point_id)
    for leaf in ordered:
        snippet = _leaf_snippet(leaf, max_chars=DEFAULT_SNIPPET_CHARS)
        if not snippet:
            continue
        lines.append(f"- {leaf.point_id}: {snippet}")
    body = "\n".join(lines)
    if len(body) > max_chars:
        body = body[: max(0, max_chars - 1)].rstrip() + "…"
    return body


# ---------------------------------------------------------------------------
# Cluster assignment
# ---------------------------------------------------------------------------


def _chunk(items: Sequence[_Leaf], chunk_size: int) -> Iterable[list[_Leaf]]:
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(items), chunk_size):
        yield list(items[start : start + chunk_size])


def _make_cluster_id(*, build_id: str, level: int, bucket: str, leaf_ids: Sequence[str]) -> str:
    sorted_ids = sorted(leaf_ids)
    raw = f"{build_id}|lvl={int(level)}|bucket={bucket}|leaves={','.join(sorted_ids)}"
    return f"raptor-cluster-{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _bucket_leaves(leaves: Sequence[_Leaf]) -> list[tuple[str, list[_Leaf]]]:
    """Group leaves by (bucket, scope) for stable cluster assignment.

    Returns a list of (bucket_key, leaves) tuples sorted by bucket key for
    determinism.
    """
    grouped: dict[tuple[str, str], list[_Leaf]] = {}
    for leaf in leaves:
        scope_key = f"{leaf.scope.profile_id}|{leaf.scope.user_id_hash}|{leaf.scope.chat_id_hash}"
        grouped.setdefault((leaf.bucket, scope_key), []).append(leaf)
    out: list[tuple[str, list[_Leaf]]] = []
    for (bucket, _scope_key), group_leaves in sorted(grouped.items(), key=lambda item: item[0][0]):
        ordered = sorted(group_leaves, key=lambda leaf: leaf.point_id)
        out.append((bucket, ordered))
    return out


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


@dataclass
class RaptorBuilder:
    """Configurable dry-run RAPTOR builder.

    The builder is intentionally stateless: you can call
    :meth:`build` multiple times with different inputs and config and
    each call returns a fresh manifest. No global state is mutated.
    """

    max_cluster_size: int = DEFAULT_MAX_CLUSTER_SIZE
    prompt_version: str = DEFAULT_PROMPT_VERSION
    summary_max_chars: int = 1024
    config: dict[str, Any] = field(default_factory=dict)

    def build(
        self,
        points: Iterable[Mapping[str, Any]],
        *,
        build_id: str = "",
    ) -> RaptorBuildManifest:
        """Build a deterministic dry-run manifest from *points*.

        *points* is an iterable of plain Qdrant-style dicts
        (``{"id": ..., "payload": {...}}``). The builder never reaches out
        to Qdrant.
        """
        # Step 1 — normalize and validate leaves.
        all_leaves: list[_Leaf] = []
        skipped: list[dict[str, Any]] = []
        for raw in points:
            if not isinstance(raw, Mapping):
                skipped.append(
                    {
                        "point_id": "",
                        "reason": "non_dict_point",
                        "snippet": "",
                    }
                )
                continue
            leaf = _normalize_leaf(raw)
            if leaf is None:
                skipped.append({"point_id": "", "reason": "normalize_failed", "snippet": ""})
                continue
            if not leaf.accepted:
                # For any skipped leaf whose original point_id matches the
                # secret scanner, never echo the raw id into the manifest:
                # only a stable, deterministic redacted handle is recorded.
                # This covers both the explicit ``secret_id_bearing`` reason
                # (e.g. ``sk-...`` point ids that pass the shape regex) and
                # the secret-shaped invalid point ids that fail
                # ``_valid_point_id_shape`` (``-----BEGIN PRIVATE KEY-----``
                # shaped, basic-auth URL, bearer/contextual strings with
                # spaces or slashes). Other skip reasons keep the original
                # id so reviewers can correlate inputs.
                if leaf.skip_reason == "secret_id_bearing" or contains_secret(leaf.point_id or ""):
                    skipped_point_id = _safe_handle_for_point_id(leaf.point_id)
                else:
                    skipped_point_id = leaf.point_id
                skipped.append(
                    {
                        "point_id": skipped_point_id,
                        "reason": leaf.skip_reason,
                        "snippet": "",
                    }
                )
                continue
            all_leaves.append(leaf)

        # Step 2 — sort leaves deterministically (already point-id sorted
        # inside the bucket, but enforce a global ordering for the manifest
        # and ID computation). ``skipped`` is sorted the same way so the
        # manifest digest (and serialized manifest) is stable under input
        # reordering of unsafe leaves.
        all_leaves.sort(key=lambda leaf: leaf.point_id)
        skipped.sort(key=lambda entry: (str(entry.get("point_id") or ""), str(entry.get("reason") or "")))

        if not all_leaves:
            return _empty_manifest(
                prompt_version=self.prompt_version,
                config=self.config,
                skipped=skipped,
            )

        # Step 3 — derive build id if caller didn't supply one.
        prompt_version = self.prompt_version or DEFAULT_PROMPT_VERSION
        leaf_ids_for_build = [leaf.point_id for leaf in all_leaves]
        resolved_build_id = build_id or compute_build_id(
            prompt_version=prompt_version,
            leaves=[{"id": pid} for pid in leaf_ids_for_build],
            config=self.config,
        )

        # Step 4 — group leaves by scope first; one tree per scope group.
        scope_groups = _group_by_scope(all_leaves)

        all_nodes: list[RaptorNode] = []
        tree_ids: list[str] = []
        root_ids: list[str] = []
        warnings: list[str] = []
        if skipped:
            warnings.append(f"skipped_leaves:{len(skipped)}")
        if len(scope_groups) > 1:
            warnings.append("scope_disagreement_across_clusters")

        for _scope_key, scope_leaves in scope_groups:
            cluster_records: list[_ClusterRecord] = []
            for bucket_key, bucket_leaves in _bucket_leaves(scope_leaves):
                for chunk in _chunk(bucket_leaves, self.max_cluster_size):
                    cluster_records.append(
                        _ClusterRecord(
                            level=RAPTOR_LEVEL_LEAF + 1,
                            bucket_key=bucket_key,
                            leaves=chunk,
                        )
                    )
            cluster_records.sort(
                key=lambda record: (record.bucket_key, [leaf.point_id for leaf in record.leaves])
            )

            cluster_nodes: list[RaptorNode] = []
            for record in cluster_records:
                cluster_id = _make_cluster_id(
                    build_id=resolved_build_id,
                    level=record.level,
                    bucket=record.bucket_key,
                    leaf_ids=[leaf.point_id for leaf in record.leaves],
                )
                cluster_scope = _unanimous_scope(record.leaves)
                summary_text = _cluster_summary_text(record.leaves, max_chars=self.summary_max_chars)
                source_hashes = _sorted_unique(leaf.content_hash for leaf in record.leaves if leaf.content_hash)
                leaf_ids = sorted(leaf.point_id for leaf in record.leaves)
                cluster_node = RaptorNode(
                    node_id=compute_node_id(
                        tree_id="pending",
                        level=record.level,
                        cluster_id=cluster_id,
                        parent_ids=(),
                    ),
                    tree_id="pending",
                    root_id="pending",
                    build_id=resolved_build_id,
                    cluster_id=cluster_id,
                    level=record.level,
                    parent_ids=[],
                    child_ids=leaf_ids,
                    summary_of=leaf_ids,
                    source_hashes=source_hashes,
                    prompt_version=prompt_version,
                    scope=cluster_scope,
                    bucket_key=record.bucket_key,
                    summary_text=summary_text,
                    extra={"raptor_leaf_count": len(record.leaves)},
                )
                cluster_nodes.append(cluster_node)

            # Build the root above the cluster summaries.
            root_cluster_id = _make_cluster_id(
                build_id=resolved_build_id,
                level=RAPTOR_LEVEL_LEAF + 2,
                bucket="__root__",
                leaf_ids=[node.cluster_id for node in cluster_nodes],
            )
            root_scope = _unanimous_scope(scope_leaves)
            root_source_hashes = _sorted_unique(
                hash_id for node in cluster_nodes for hash_id in node.source_hashes
            )
            tree_id = compute_tree_id(
                build_id=resolved_build_id,
                prompt_version=prompt_version,
                root_id=f"pending::{root_cluster_id}",
            )
            root_id = compute_root_id(build_id=resolved_build_id, cluster_id=root_cluster_id)

            # Patch tree_id + root_id into every cluster node now that
            # they are stable, then recompute their node_id with the real
            # tree_id (so node ids are stable + tree-scoped).
            for node in cluster_nodes:
                node.tree_id = tree_id
                node.root_id = root_id
                node.node_id = compute_node_id(
                    tree_id=tree_id,
                    level=node.level,
                    cluster_id=node.cluster_id,
                    parent_ids=(),
                )

            root_summary_text = _build_root_summary(cluster_nodes, max_chars=self.summary_max_chars)
            all_leaf_ids: list[str] = []
            for node in cluster_nodes:
                all_leaf_ids.extend(node.summary_of)
            root_node = RaptorNode(
                node_id=compute_node_id(
                    tree_id=tree_id,
                    level=RAPTOR_LEVEL_LEAF + 2,
                    cluster_id=root_cluster_id,
                    parent_ids=(),
                ),
                tree_id=tree_id,
                root_id=root_id,
                build_id=resolved_build_id,
                cluster_id=root_cluster_id,
                level=RAPTOR_LEVEL_LEAF + 2,
                parent_ids=[],
                child_ids=sorted(node.node_id for node in cluster_nodes),
                summary_of=_sorted_unique(all_leaf_ids),
                source_hashes=root_source_hashes,
                prompt_version=prompt_version,
                scope=root_scope,
                bucket_key="__root__",
                summary_text=root_summary_text,
                extra={"raptor_cluster_count": len(cluster_nodes)},
            )
            all_nodes.extend(cluster_nodes)
            all_nodes.append(root_node)
            tree_ids.append(tree_id)
            root_ids.append(root_id)

        all_nodes.sort(key=lambda node: (node.level, node.node_id))

        # Step 5 — produce the manifest.
        candidate_payloads = [node.to_payload() for node in all_nodes]
        candidate_payloads.sort(key=lambda payload: payload.get("raptor_node_id") or "")

        # Top-level scope is the scope of the *first* tree (scope
        # disagreement is reported via warnings). Empty if scopes disagree.
        scope_dict: dict[str, str] = {}
        if len(root_ids) == 1:
            first_scope = next(
                (
                    node.scope
                    for node in all_nodes
                    if node.level == RAPTOR_LEVEL_LEAF + 2 and isinstance(node.scope, RaptorScope)
                ),
                None,
            )
            if isinstance(first_scope, RaptorScope):
                scope_dict = first_scope.as_dict()

        manifest_tree_id = tree_ids[0] if len(tree_ids) == 1 else "|".join(tree_ids)
        manifest_root_id = root_ids[0] if len(root_ids) == 1 else "|".join(root_ids)

        manifest = RaptorBuildManifest(
            build_id=resolved_build_id,
            prompt_version=prompt_version,
            tree_id=manifest_tree_id,
            root_id=manifest_root_id,
            config=dict(self.config),
            leaf_count=len(all_leaves),
            node_count=len(all_nodes),
            skipped_leaves=skipped,
            warnings=warnings,
            candidate_node_payloads=candidate_payloads,
            scope=scope_dict,
        )
        # Compute deterministic digest after payload list is settled.
        manifest.manifest_digest = compute_manifest_digest(manifest.to_dict())
        return manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _ClusterRecord:
    level: int
    bucket_key: str
    leaves: list[_Leaf] = field(default_factory=list)


def _group_by_scope(leaves: Sequence[_Leaf]) -> list[tuple[str, list[_Leaf]]]:
    grouped: dict[str, list[_Leaf]] = {}
    for leaf in leaves:
        scope_key = f"{leaf.scope.profile_id}|{leaf.scope.user_id_hash}|{leaf.scope.chat_id_hash}"
        grouped.setdefault(scope_key, []).append(leaf)
    out: list[tuple[str, list[_Leaf]]] = []
    for scope_key, group_leaves in sorted(grouped.items()):
        ordered = sorted(group_leaves, key=lambda leaf: leaf.point_id)
        out.append((scope_key, ordered))
    return out


def _unanimous_scope(leaves: Iterable[_Leaf]) -> RaptorScope | None:
    leaves_list = list(leaves)
    if not leaves_list:
        return None
    first = leaves_list[0].scope
    if all(leaf.scope == first for leaf in leaves_list):
        return first
    return None


def _build_root_summary(nodes: Sequence[RaptorNode], *, max_chars: int) -> str:
    """Root summary enumerates the cluster ids; it does not synthesize text.

    Each line anchors to the cluster id and the first line of the
    cluster's extractive summary. Child node ids appear inline as
    provenance anchors so callers can trace back to specific summaries.
    """
    lines = ["RAPTOR root — clusters:"]
    for node in nodes:
        snippet = _normalize_text(node.summary_text)
        first_line = snippet.splitlines()[0] if snippet else ""
        first_line = first_line[:120]
        if node.node_id:
            lines.append(f"- {node.cluster_id} ({node.node_id}): {first_line}")
        else:
            lines.append(f"- {node.cluster_id}: {first_line}")
    body = "\n".join(lines)
    if len(body) > max_chars:
        body = body[: max(0, max_chars - 1)].rstrip() + "…"
    return body


def _empty_manifest(
    *,
    prompt_version: str,
    config: Mapping[str, Any],
    skipped: list[dict[str, Any]],
) -> RaptorBuildManifest:
    build_id = compute_build_id(prompt_version=prompt_version, leaves=[], config=config)
    tree_id = compute_tree_id(
        build_id=build_id,
        prompt_version=prompt_version,
        root_id="empty",
    )
    root_id = compute_root_id(build_id=build_id, cluster_id="empty")
    manifest = RaptorBuildManifest(
        build_id=build_id,
        prompt_version=prompt_version,
        tree_id=tree_id,
        root_id=root_id,
        config=dict(config),
        leaf_count=0,
        node_count=0,
        skipped_leaves=skipped,
        warnings=["no_safe_leaves"],
        candidate_node_payloads=[],
        scope={},
    )
    manifest.manifest_digest = compute_manifest_digest(manifest.to_dict())
    return manifest


# ---------------------------------------------------------------------------
# Functional helper
# ---------------------------------------------------------------------------


def build_raptor_dry_run(
    points: Iterable[Mapping[str, Any]],
    *,
    max_cluster_size: int = DEFAULT_MAX_CLUSTER_SIZE,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    config: Mapping[str, Any] | None = None,
    build_id: str = "",
) -> RaptorBuildManifest:
    """Convenience wrapper around :class:`RaptorBuilder`."""
    builder = RaptorBuilder(
        max_cluster_size=max_cluster_size,
        prompt_version=prompt_version,
        config=dict(config or {}),
    )
    return builder.build(points, build_id=build_id)


__all__ = [
    "DEFAULT_MAX_CLUSTER_SIZE",
    "RaptorBuilder",
    "build_raptor_dry_run",
]