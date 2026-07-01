"""RAPTOR apply — validation, planning, and audit helpers (Phase 4).

This module is **pure**: it performs no Qdrant I/O. All validation, digest
verification, dry-run plan construction, and audit artifact persistence live
here. The actual live ``upsert`` is performed by the provider in
``__init__.py``, which calls into :func:`plan_apply` first and then
:func:`persist_apply_record` after a successful mutation.

Hard guarantees
---------------

1. **Exact-ID only.** Every candidate node is addressed by its
   ``raptor_node_id``. No delete-by-filter, no broad update filter, and no
   ``delete_ids`` are issued by this module or by the provider's RAPTOR apply
   path.

2. **Digest-gated.** ``report_id``, ``build_id``, and ``manifest_digest``
   must all match exactly before any live mutation. A stale or altered
   manifest is rejected with :class:`RaptorApplyError`.

3. **Write-gate validation.** Every ``candidate_node_payload`` is validated
   through :func:`evaluate_raptor_summary_write` both before *and* after
   provider metadata enrichment.

4. **Audit trail.** A live application writes a JSON record under
   ``~/.hermes/qdrant_memory/raptor_applied/`` containing the exact
   ``report_id``, ``manifest_digest``, ``applied_node_ids``, and timestamps.

Manifest shape
--------------

The module accepts either:

- A **raw Phase 3 manifest** dict (output of
  :meth:`RaptorBuildManifest.to_dict`) with ``report_id`` injected by the
  caller, **or**
- An **artifact wrapper** dict of the form::

      {"report_id": "raptor-<12hex>", "manifest": { ... raw manifest ... }}

The artifact-wrapper shape is what :func:`persist_manifest_report` and
:func:`load_manifest_report` produce/consume on disk.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from qdrant_memory.lesson_extractor import contains_secret
from qdrant_memory.raptor.schema import (
    RAPTOR_DERIVATION_TYPE,
    RAPTOR_REQUIRED_NODE_FIELDS,
    compute_manifest_digest,
)
from qdrant_memory.write_gate import (
    WriteDecision,
    evaluate_raptor_summary_write,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Strict canonical report ID: raptor-<12 hex chars>
REPORT_ID_RE = re.compile(r"^raptor-[a-f0-9]{12}$")

# Canonical build/node id shapes (Phase 3 builder uses longer hex but we
# just need to validate they are non-empty, hex-ish, and not malformed).
_BUILD_ID_RE = re.compile(r"^raptor-build-[a-f0-9]+$")
_NODE_ID_RE = re.compile(r"^raptor-node-[a-f0-9]+$")

# Status markers that make a source/leaf unsafe for RAPTOR parent status.
_UNSAFE_LEAF_STATUSES: frozenset[str] = frozenset(
    {
        "stale",
        "deprecated",
        "superseded",
        "disputed",
        "review_required",
    }
)

# Status markers that mark a leaf as excluded (forgotten/quarantined/secret).
_EXCLUDED_LEAF_MARKERS: tuple[str, ...] = (
    "consolidation_quarantined",
    "raptor_excluded",
    "raptor_forgotten",
)


class RaptorApplyError(Exception):
    """Raised when a RAPTOR apply fails validation or digest verification."""


# ---------------------------------------------------------------------------
# Manifest loading / persistence
# ---------------------------------------------------------------------------


# Subdirectory names used when a single base ``raptor_artifact_dir`` is
# configured. They keep manifest reports and apply records on disk in
# distinct locations so live apply cannot overwrite a manifest report,
# and so a manifest wrapper cannot be loaded as an apply record.
_REPORT_SUBDIR = "raptor_reports"
_APPLIED_SUBDIR = "raptor_applied"

# Discriminator written into every persisted apply record. Used to refuse
# manifest wrappers, stale files, or anything else under the applied dir
# that does not match the exact apply-record schema.
_APPLY_RECORD_TYPE = "raptor_apply"


def _report_dir(hermes_home: str, configured_dir: str = "") -> Path:
    if configured_dir:
        return Path(configured_dir) / _REPORT_SUBDIR
    return Path(hermes_home) / "qdrant_memory" / _REPORT_SUBDIR


def _applied_dir(hermes_home: str, configured_dir: str = "") -> Path:
    if configured_dir:
        return Path(configured_dir) / _APPLIED_SUBDIR
    return Path(hermes_home) / "qdrant_memory" / _APPLIED_SUBDIR


def wrap_manifest(manifest: Mapping[str, Any], *, report_id: str = "") -> dict[str, Any]:
    """Wrap a raw Phase 3 manifest dict into the artifact-wrapper shape.

    If *report_id* is empty, a deterministic one is derived from the manifest
    digest.
    """
    manifest_dict = dict(manifest)
    digest = manifest_dict.get("manifest_digest") or compute_manifest_digest(manifest_dict)
    manifest_dict["manifest_digest"] = digest
    rid = report_id or f"raptor-{digest[:12]}"
    return {"report_id": rid, "manifest": manifest_dict}


def persist_manifest_report(
    wrapper: dict[str, Any],
    *,
    hermes_home: str,
    configured_dir: str = "",
) -> dict[str, Any]:
    """Persist an artifact-wrapper report to disk.

    Returns ``{"path": ..., "report_id": ...}``.
    """
    report_id = wrapper.get("report_id", "")
    if not REPORT_ID_RE.match(report_id):
        raise ValueError(f"Refusing to persist report with non-canonical report_id: {report_id!r}")
    dir_path = _report_dir(hermes_home, configured_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"{report_id}.json"
    file_path.write_text(
        json.dumps(wrapper, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"path": str(file_path), "report_id": report_id}


def load_manifest_report(
    report_id: str,
    *,
    hermes_home: str,
    configured_dir: str = "",
) -> dict[str, Any] | None:
    """Load an artifact-wrapper report by exact *report_id*.

    Returns ``None`` if the ID is non-canonical or the file is not found.
    """
    if not REPORT_ID_RE.match(report_id):
        return None
    dir_path = _report_dir(hermes_home, configured_dir)
    file_path = dir_path / f"{report_id}.json"
    if not file_path.exists():
        return None
    try:
        loaded = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(loaded, dict):
        return None
    # Require exact report_id match after loading.
    if loaded.get("report_id") != report_id:
        return None
    return loaded


def extract_manifest(wrapper_or_raw: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the raw manifest dict from either shape.

    Accepts:
    - ``{"report_id": ..., "manifest": {...}}`` (artifact wrapper)
    - ``{... raw Phase 3 manifest ...}`` with optional ``report_id`` key
    """
    if not isinstance(wrapper_or_raw, Mapping):
        raise RaptorApplyError("manifest input must be a dict")
    if "manifest" in wrapper_or_raw and isinstance(wrapper_or_raw["manifest"], Mapping):
        return dict(wrapper_or_raw["manifest"])
    return dict(wrapper_or_raw)


def extract_report_id(wrapper_or_raw: Mapping[str, Any]) -> str:
    """Extract ``report_id`` from either shape."""
    report_id = str(wrapper_or_raw.get("report_id") or "").strip()
    return report_id


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_report_id(report_id: str) -> None:
    if not report_id:
        raise RaptorApplyError("report_id is required")
    if report_id != report_id.strip():
        raise RaptorApplyError("report_id must match canonical format raptor-<12hex>")
    if not REPORT_ID_RE.match(report_id):
        raise RaptorApplyError("report_id must match canonical format raptor-<12hex>")


def _validate_build_id(build_id: str) -> None:
    if not build_id:
        raise RaptorApplyError("build_id is required")
    if not _BUILD_ID_RE.match(build_id):
        raise RaptorApplyError(f"build_id must match canonical format raptor-build-<hex>, got: {build_id[:30]}...")


def _validate_node_id(node_id: str) -> None:
    if not node_id:
        raise RaptorApplyError("raptor_node_id is required")
    if not _NODE_ID_RE.match(node_id):
        raise RaptorApplyError(f"raptor_node_id must match canonical format raptor-node-<hex>, got: {node_id[:30]}...")


def verify_manifest_digest(
    manifest: Mapping[str, Any],
    *,
    expected_digest: str,
) -> None:
    """Recompute the manifest digest and fail closed on mismatch.

    Raises :class:`RaptorApplyError` if the recomputed digest does not match
    *expected_digest*.
    """
    recomputed = compute_manifest_digest(dict(manifest))
    if recomputed != expected_digest:
        raise RaptorApplyError(
            "manifest digest mismatch: manifest may be stale or altered"
        )


# Regex for a 64-char lowercase hex SHA-256 hash (matches the builder's
# ``hashlib.sha256(...).hexdigest()`` output).
_SHA256_HEX_RE = re.compile(r"^[a-f0-9]{64}$")


def _is_sha256_hex(value: Any) -> bool:
    """Return True iff *value* is a 64-char lowercase hex SHA-256 string."""
    return isinstance(value, str) and bool(_SHA256_HEX_RE.match(value))


def _is_raptor_provenance_edge(edge: Any) -> bool:
    """Return True iff *edge* is a structurally valid RAPTOR provenance edge.

    A RAPTOR provenance edge is a mapping with:
    - non-empty string ``source_uri``
    - non-empty string ``child_node_id``
    - ``derivation_type`` exactly equal to :data:`RAPTOR_DERIVATION_TYPE`
    - ``relation_type`` exactly equal to ``"SUMMARIZES"``
    """
    if not isinstance(edge, Mapping):
        return False
    source_uri = edge.get("source_uri")
    child_node_id = edge.get("child_node_id")
    derivation_type = edge.get("derivation_type")
    relation_type = edge.get("relation_type")
    return (
        isinstance(source_uri, str)
        and bool(source_uri.strip())
        and isinstance(child_node_id, str)
        and bool(child_node_id.strip())
        and derivation_type == RAPTOR_DERIVATION_TYPE
        and relation_type == "SUMMARIZES"
    )


def _validate_candidate_payload(payload: Mapping[str, Any]) -> WriteDecision:
    """Validate a single candidate node payload through the write gate.

    Returns the :class:`WriteDecision` so callers can include it in the plan.
    Raises :class:`RaptorApplyError` on structural problems the write gate
    does not cover (malformed node id, missing required RAPTOR fields,
    canonical=True, requires_review=False, secret-bearing, etc.).
    """
    if not isinstance(payload, Mapping):
        raise RaptorApplyError("candidate_node_payload must be a dict")

    node_id = str(payload.get("raptor_node_id") or "")
    _validate_node_id(node_id)

    # Check required RAPTOR fields are present by KEY (not truthiness).
    # Several required fields have legitimate falsy values:
    #   - ``raptor_parent_ids`` is ``[]`` for root-level (level 1) nodes.
    #   - ``raptor_summary_of`` is ``[]`` for non-summary cluster rows.
    #   - ``canonical`` is exactly ``False`` (enforced below via ``is True``).
    #   - ``requires_review`` is exactly ``True`` (enforced below via ``is False``).
    missing = [f for f in RAPTOR_REQUIRED_NODE_FIELDS if f not in payload]
    if missing:
        raise RaptorApplyError(f"candidate node {node_id} missing required fields: {missing}")

    # canonical must be exactly the boolean False. Reject every other value
    # (strings, ints, None, True) — type-loose checks let tampered
    # digest-consistent manifests bypass the trust flag invariant.
    if payload.get("canonical") is not False:
        raise RaptorApplyError(
            f"candidate node {node_id} must have canonical exactly equal to False"
        )

    # requires_review must be exactly the boolean True. Reject every other
    # value so RAPTOR summaries cannot be silently marked auto-store.
    if payload.get("requires_review") is not True:
        raise RaptorApplyError(
            f"candidate node {node_id} must have requires_review exactly equal to True"
        )

    # derivation_type must be raptor_summary
    if str(payload.get("derivation_type") or "") != RAPTOR_DERIVATION_TYPE:
        raise RaptorApplyError(
            f"candidate node {node_id} derivation_type must be {RAPTOR_DERIVATION_TYPE}"
        )

    # child IDs must be non-empty
    child_ids = payload.get("raptor_child_ids")
    if not child_ids or not isinstance(child_ids, list) or not all(str(c) for c in child_ids):
        raise RaptorApplyError(f"candidate node {node_id} must have non-empty raptor_child_ids")

    # source_hashes must be a non-empty list of non-empty 64-char lowercase
    # hex SHA-256 strings (the builder uses hashlib.sha256().hexdigest()).
    source_hashes = payload.get("source_hashes")
    if not source_hashes or not isinstance(source_hashes, list):
        raise RaptorApplyError(
            f"candidate node {node_id} must have non-empty source_hashes list"
        )
    if not all(_is_sha256_hex(h) for h in source_hashes):
        raise RaptorApplyError(
            f"candidate node {node_id} source_hashes must each be a 64-char "
            "lowercase hex SHA-256 string"
        )

    # derived_from (provenance) must be a non-empty list of structurally
    # valid RAPTOR provenance edges. Each edge must have:
    #   - non-empty string source_uri
    #   - non-empty string child_node_id
    #   - derivation_type == "raptor_summary"
    #   - relation_type == "SUMMARIZES"
    derived_from = payload.get("derived_from")
    if not derived_from or not isinstance(derived_from, list):
        raise RaptorApplyError(
            f"candidate node {node_id} must have non-empty derived_from provenance"
        )
    for edge in derived_from:
        if not _is_raptor_provenance_edge(edge):
            raise RaptorApplyError(
                f"candidate node {node_id} derived_from entries must be mappings "
                "with non-empty source_uri and child_node_id, "
                f"derivation_type={RAPTOR_DERIVATION_TYPE!r}, "
                "and relation_type='SUMMARIZES'"
            )

    # Secret check on payload text + full payload blob
    text = str(payload.get("text") or "")
    if contains_secret(text):
        raise RaptorApplyError(f"candidate node {node_id} text appears to contain a secret")
    payload_blob = json.dumps(payload, sort_keys=True, default=str)
    if contains_secret(payload_blob):
        raise RaptorApplyError(f"candidate node {node_id} payload appears to contain a secret")

    # Run through the write gate for a formal decision
    decision = evaluate_raptor_summary_write(
        text=text,
        metadata=dict(payload),
        confidence=1.0,
    )
    return decision


def _check_duplicate_node_ids(payloads: list[Mapping[str, Any]]) -> None:
    """Fail if any two candidate payloads share the same raptor_node_id."""
    seen: set[str] = set()
    for payload in payloads:
        node_id = str(payload.get("raptor_node_id") or "")
        if node_id in seen:
            raise RaptorApplyError(f"duplicate raptor_node_id: {node_id}")
        seen.add(node_id)


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    report_id: str,
    build_id: str,
    manifest_digest: str,
) -> list[WriteDecision]:
    """Full validation of a RAPTOR manifest before any live mutation.

    Validates:
    - ``report_id`` matches canonical format and the manifest's own report_id
      (if present).
    - ``build_id`` matches the manifest's ``build_id``.
    - ``manifest_digest`` matches a recomputed digest (anti-stale/alteration).
    - No duplicate node IDs.
    - Each candidate payload passes structural + write-gate validation.

    Returns the list of :class:`WriteDecision` per candidate (in payload
    order).
    Raises :class:`RaptorApplyError` on any validation failure.
    """
    # 1. Canonical IDs
    _validate_report_id(report_id)
    _validate_build_id(build_id)

    # 2. build_id must match manifest
    manifest_build_id = str(manifest.get("build_id") or "")
    if manifest_build_id != build_id:
        raise RaptorApplyError(
            f"build_id mismatch: caller={build_id} manifest={manifest_build_id}"
        )

    # 3. report_id consistency (if manifest carries one)
    manifest_report_id = str(manifest.get("report_id") or "")
    if manifest_report_id and manifest_report_id != report_id:
        raise RaptorApplyError(
            f"report_id mismatch: caller={report_id} manifest={manifest_report_id}"
        )

    # 4. Manifest digest
    verify_manifest_digest(manifest, expected_digest=manifest_digest)

    # 5. Manifest must have candidate payloads
    payloads = manifest.get("candidate_node_payloads")
    if not payloads or not isinstance(payloads, list):
        raise RaptorApplyError("manifest has no candidate_node_payloads")

    # 6. Check for stale/excluded markers in the manifest's skipped_leaves
    skipped = manifest.get("skipped_leaves") or []
    if isinstance(skipped, list):
        for entry in skipped:
            if isinstance(entry, Mapping):
                reason = str(entry.get("reason") or "")
                if reason.startswith("secret") or reason.startswith("quarantin"):
                    raise RaptorApplyError(
                        f"manifest contains skipped leaf with unsafe reason: {reason}; "
                        "cannot apply a build derived from secret/quarantined leaves"
                    )

    # 7. Duplicate node IDs
    _check_duplicate_node_ids(payloads)

    # 8. Validate each candidate
    decisions: list[WriteDecision] = []
    for payload in payloads:
        decisions.append(_validate_candidate_payload(payload))

    return decisions


# ---------------------------------------------------------------------------
# Dry-run plan
# ---------------------------------------------------------------------------


def plan_apply(
    manifest: Mapping[str, Any],
    *,
    report_id: str,
    build_id: str,
    manifest_digest: str,
    existing_node_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Produce a dry-run apply plan without any mutation.

    Calls :func:`validate_manifest` first, so any validation failure raises
    :class:`RaptorApplyError`.

    The plan includes:
    - ``would_upsert_ids``: node IDs that would be newly upserted.
    - ``already_present_ids``: node IDs already present in Qdrant with
      matching RAPTOR metadata (idempotent).
    - ``blocked_ids``: node IDs that exist in Qdrant but would conflict
      (fail-closed, caller should not proceed).
    - ``write_decisions``: per-node write-gate decisions.
    - ``warnings``: non-fatal advisory messages.
    """
    decisions = validate_manifest(
        manifest,
        report_id=report_id,
        build_id=build_id,
        manifest_digest=manifest_digest,
    )

    existing = existing_node_ids or set()

    payloads = manifest.get("candidate_node_payloads") or []
    would_upsert_ids: list[str] = []
    already_present_ids: list[str] = []
    blocked_ids: list[str] = []
    write_decisions: list[dict[str, Any]] = []
    warnings: list[str] = []

    for payload, decision in zip(payloads, decisions):
        node_id = str(payload.get("raptor_node_id") or "")
        write_decisions.append(
            {
                "raptor_node_id": node_id,
                "decision": decision.decision,
                "reasons": list(decision.reasons),
                "requires_review": decision.requires_review,
            }
        )

        if node_id in existing:
            # Idempotent — already present
            already_present_ids.append(node_id)
        elif decision.decision == "reject":
            blocked_ids.append(node_id)
            warnings.append(f"node {node_id} rejected by write gate: {', '.join(decision.reasons)}")
        else:
            would_upsert_ids.append(node_id)

    # Check for stale/excluded leaves referenced by parent summaries
    child_id_sets = _collect_child_ids(payloads)
    if child_id_sets:
        warnings.append(
            "parent summaries reference child leaf IDs; status of those "
            "leaves should be verified via qdrant_memory_raptor_status before "
            "live apply"
        )

    return {
        "dry_run": True,
        "report_id": report_id,
        "build_id": build_id,
        "manifest_digest": manifest_digest,
        "would_upsert_ids": sorted(would_upsert_ids),
        "already_present_ids": sorted(already_present_ids),
        "blocked_ids": sorted(blocked_ids),
        "write_decisions": write_decisions,
        "warnings": warnings,
        "node_count": len(payloads),
    }


def _collect_child_ids(payloads: list[Mapping[str, Any]]) -> set[str]:
    """Collect all child IDs referenced by candidate payloads."""
    child_ids: set[str] = set()
    for payload in payloads:
        for cid in (payload.get("raptor_child_ids") or []):
            child_ids.add(str(cid))
    return child_ids


# ---------------------------------------------------------------------------
# Audit persistence
# ---------------------------------------------------------------------------


def persist_apply_record(
    *,
    report_id: str,
    build_id: str,
    manifest_digest: str,
    applied_node_ids: list[str],
    hermes_home: str,
    configured_dir: str = "",
    profile_id: str = "",
) -> dict[str, Any]:
    """Persist a live application audit record.

    Returns the record dict with ``application_id`` and ``artifact_path``.
    """
    _validate_report_id(report_id)

    applied_at = datetime.utcnow().isoformat() + "Z"
    record = {
        "record_type": _APPLY_RECORD_TYPE,
        "report_id": report_id,
        "build_id": build_id,
        "manifest_digest": manifest_digest,
        "applied_node_ids": sorted(applied_node_ids),
        "applied_at": applied_at,
        "profile_id": profile_id,
        "schema_version": 1,
    }
    # Seed for deterministic application_id
    seed = f"{applied_at}:{report_id}:{build_id}:{manifest_digest}"
    application_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    record["application_id"] = application_id

    dir_path = _applied_dir(hermes_home, configured_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"{report_id}.json"
    record["artifact_path"] = str(file_path)
    file_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return record


def _validate_apply_record(
    record: Any,
    *,
    report_id: str,
    build_id: str,
    manifest_digest: str,
    expected_node_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Strictly validate a loaded apply record against the expected context.

    A record is accepted only when ALL of the following match exactly:

    - ``record_type`` equals :data:`_APPLY_RECORD_TYPE`
    - ``report_id``, ``build_id``, ``manifest_digest`` match the caller
    - ``applied_node_ids`` is a list of strings
    - If *expected_node_ids* is provided, the stored applied node IDs
      must equal the expected set (same manifest, same node set)

    Any deviation (manifest wrapper, stale build/digest, missing fields,
    non-list applied_node_ids, mismatched node set) raises
    :class:`RaptorApplyError` so live apply fails closed.
    """
    if not isinstance(record, Mapping):
        raise RaptorApplyError(
            "apply record is not a dict; refusing to treat as already applied"
        )
    if record.get("record_type") != _APPLY_RECORD_TYPE:
        raise RaptorApplyError(
            f"apply record has wrong record_type {record.get('record_type')!r}; "
            f"expected {_APPLY_RECORD_TYPE!r}"
        )
    if str(record.get("report_id") or "") != report_id:
        raise RaptorApplyError(
            "apply record report_id does not match caller; refusing stale idempotency"
        )
    if str(record.get("build_id") or "") != build_id:
        raise RaptorApplyError(
            "apply record build_id does not match caller; refusing stale idempotency"
        )
    if str(record.get("manifest_digest") or "") != manifest_digest:
        raise RaptorApplyError(
            "apply record manifest_digest does not match caller; refusing stale idempotency"
        )
    applied_ids_raw = record.get("applied_node_ids")
    if not isinstance(applied_ids_raw, list) or not all(
        isinstance(x, str) and x for x in applied_ids_raw
    ):
        raise RaptorApplyError(
            "apply record applied_node_ids is not a non-empty list of strings"
        )
    if expected_node_ids is not None:
        stored = {str(x) for x in applied_ids_raw}
        if stored != expected_node_ids:
            raise RaptorApplyError(
                "apply record applied_node_ids do not match the expected "
                "manifest node set; refusing to treat as already applied"
            )
    return dict(record)


def load_apply_record(
    report_id: str,
    *,
    hermes_home: str,
    configured_dir: str = "",
    build_id: str = "",
    manifest_digest: str = "",
    expected_node_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Load a previously persisted apply record for *report_id*.

    Returns ``None`` if the file does not exist or the *report_id* is not
    canonical. Raises :class:`RaptorApplyError` if the file exists but
    fails strict validation (wrong record_type, mismatched report/build/
    digest, malformed applied_node_ids). Malformed records MUST fail
    closed so live apply cannot be silently skipped.

    The optional *build_id*, *manifest_digest*, and *expected_node_ids*
    arguments are only used for strict validation. When supplied, the
    loaded record must match all of them exactly to be returned.
    """
    if not REPORT_ID_RE.match(report_id):
        return None
    dir_path = _applied_dir(hermes_home, configured_dir)
    file_path = dir_path / f"{report_id}.json"
    if not file_path.exists():
        return None
    try:
        loaded = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RaptorApplyError(
            f"apply record at {file_path} is malformed JSON: {exc}"
        ) from exc
    # Strict validation: if the caller provided any of the strict-match
    # fields, require exact match. Without them, only require the
    # record_type discriminator and structural sanity, but never return
    # a manifest wrapper silently.
    if build_id or manifest_digest or expected_node_ids is not None:
        return _validate_apply_record(
            loaded,
            report_id=report_id,
            build_id=build_id,
            manifest_digest=manifest_digest,
            expected_node_ids=expected_node_ids,
        )
    if not isinstance(loaded, Mapping):
        raise RaptorApplyError(
            "apply record is not a dict; refusing to treat as already applied"
        )
    if loaded.get("record_type") != _APPLY_RECORD_TYPE:
        raise RaptorApplyError(
            f"apply record has wrong record_type {loaded.get('record_type')!r}; "
            f"expected {_APPLY_RECORD_TYPE!r}"
        )
    return dict(loaded)


def is_already_applied(
    report_id: str,
    *,
    hermes_home: str,
    configured_dir: str = "",
    build_id: str = "",
    manifest_digest: str = "",
    expected_node_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Check if a report has already been applied.

    Returns the apply record dict if applied, or ``None`` when no record
    exists. Raises :class:`RaptorApplyError` on malformed or mismatched
    records so callers fail closed instead of silently treating the
    manifest as not applied.
    """
    return load_apply_record(
        report_id,
        hermes_home=hermes_home,
        configured_dir=configured_dir,
        build_id=build_id,
        manifest_digest=manifest_digest,
        expected_node_ids=expected_node_ids,
    )


# ---------------------------------------------------------------------------
# Status helpers (read-only, conservative)
# ---------------------------------------------------------------------------

# Unsafe status markers for leaves that would make a parent stale/excluded
_UNSAFE_LEAVE_STATUS_MARKERS: tuple[str, ...] = (
    "fact_status",
    "stale",
    "consolidation_quarantined",
    "requires_review",
    "raptor_excluded",
    "raptor_forgotten",
)


def assess_leaf_safety(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Assess whether a leaf payload is safe for RAPTOR parent status.

    Returns a dict with ``safe`` (bool) and ``reasons`` (list[str]).

    Conservative: any unsafe marker makes the leaf unsafe.
    """
    if not isinstance(payload, Mapping):
        return {"safe": False, "reasons": ["payload_not_dict"]}

    reasons: list[str] = []
    text = str(payload.get("text") or payload.get("lesson") or "")

    # Secret-bearing
    if text and contains_secret(text):
        reasons.append("secret_bearing")
    payload_blob = json.dumps(payload, sort_keys=True, default=str)
    if contains_secret(payload_blob):
        reasons.append("secret_bearing_payload")

    # Quarantined
    if payload.get("consolidation_quarantined") is True:
        reasons.append("quarantined")

    # Explicit exclusion markers
    for marker in ("raptor_excluded", "raptor_forgotten"):
        if payload.get(marker) is True:
            reasons.append(marker)

    # fact_status
    fact_status = str(payload.get("fact_status") or "").strip().lower()
    if not fact_status and payload.get("stale") is True:
        fact_status = "stale"
    if fact_status in _UNSAFE_LEAF_STATUSES:
        reasons.append(f"fact_status:{fact_status}")

    # requires_review flag (leaf that itself needs review)
    if payload.get("requires_review") is True:
        reasons.append("requires_review")

    return {
        "safe": len(reasons) == 0,
        "reasons": reasons,
    }


def assess_parent_status(
    child_payloads: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assess RAPTOR parent status given a list of child leaf payloads.

    If any child is unsafe, the parent is marked ``stale`` or ``excluded``
    conservatively.

    Returns a dict with:
    - ``parent_status``: ``"active"``, ``"stale"``, or ``"excluded"``.
    - ``unsafe_children``: list of safety assessment dicts.
    - ``safe_children_count``: int.
    """
    unsafe_children: list[dict[str, Any]] = []
    safe_count = 0
    has_secret = False
    has_quarantined = False
    has_stale = False
    has_other_unsafe = False

    for i, payload in enumerate(child_payloads):
        assessment = assess_leaf_safety(payload)
        if assessment["safe"]:
            safe_count += 1
            continue
        reasons = list(assessment["reasons"])
        unsafe_children.append(
            {
                "child_index": i,
                "reasons": reasons,
            }
        )
        if any("secret" in r for r in reasons):
            has_secret = True
        elif "quarantined" in reasons or "raptor_excluded" in reasons or "raptor_forgotten" in reasons:
            has_quarantined = True
        elif any("stale" in r or "fact_status:" in r for r in reasons):
            has_stale = True
        else:
            # Any other unsafe reason (e.g. requires_review, payload_not_dict,
            # missing/deletion) must still make the parent non-active.
            # We use the existing "stale" vocabulary unless secret/quarantine
            # is present, in which case "excluded" wins.
            has_other_unsafe = True

    # Conservative priority: secret > quarantined/excluded > stale (incl.
    # any other unsafe reason such as requires_review).
    if has_secret:
        parent_status = "excluded"
    elif has_quarantined:
        parent_status = "excluded"
    elif has_stale or has_other_unsafe:
        parent_status = "stale"
    else:
        parent_status = "active"

    return {
        "parent_status": parent_status,
        "unsafe_children": unsafe_children,
        "safe_children_count": safe_count,
        "total_children": len(child_payloads),
    }


__all__ = [
    "REPORT_ID_RE",
    "RaptorApplyError",
    "assess_leaf_safety",
    "assess_parent_status",
    "extract_manifest",
    "extract_report_id",
    "is_already_applied",
    "load_apply_record",
    "load_manifest_report",
    "persist_apply_record",
    "persist_manifest_report",
    "plan_apply",
    "validate_manifest",
    "verify_manifest_digest",
    "wrap_manifest",
]
