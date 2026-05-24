from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from qdrant_memory.client import QdrantClient

SCHEMA_VERSION = 1
BACKUP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SCOPES = ("memory", "learning")


class BackupError(ValueError):
    """Raised when backup/export/restore safety validation fails."""


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _backup_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def generate_backup_id() -> str:
    return f"backup-{_backup_timestamp()}-{uuid.uuid4().hex[:8]}"


def backup_root(config: dict[str, Any], hermes_home: str | Path) -> Path:
    configured = str(config.get("backup_artifact_dir") or "").strip()
    return Path(configured).expanduser() if configured else Path(hermes_home) / "qdrant_memory" / "backups"


def ensure_private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.chmod(0o700)
    except PermissionError:
        pass
    return path


def validate_backup_id(backup_id: str) -> str:
    value = str(backup_id or "").strip()
    if not value or value in {".", ".."} or "/" in value or "\\" in value or not BACKUP_ID_RE.match(value):
        raise BackupError("invalid backup_id")
    return value


def backup_dir_for_id(root: Path, backup_id: str) -> Path:
    safe_id = validate_backup_id(backup_id)
    root_resolved = root.expanduser().resolve()
    backup_dir = (root_resolved / safe_id).resolve()
    if backup_dir.parent != root_resolved:
        raise BackupError("invalid backup_id path")
    return backup_dir


def qdrant_client_from_config(config: dict[str, Any]) -> QdrantClient:
    return QdrantClient(str(config.get("qdrant_url") or ""), api_key=str(config.get("qdrant_api_key") or ""))


def collection_specs(config: dict[str, Any], scope: str = "both") -> list[dict[str, str]]:
    normalized = str(scope or "both").strip().lower()
    if normalized not in {"memory", "learning", "both"}:
        raise BackupError("scope must be one of: memory, learning, both")
    specs: list[dict[str, str]] = []
    if normalized in {"memory", "both"}:
        specs.append({"scope": "memory", "collection_name": str(config.get("collection_name") or "hermes_memory"), "file": "memory.jsonl"})
    if normalized in {"learning", "both"}:
        specs.append({"scope": "learning", "collection_name": str(config.get("learning_collection_name") or "hermes_learnings"), "file": "learning.jsonl"})
    return specs


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_text(data: str) -> str:
    return _sha256_bytes(data.encode("utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise BackupError(f"artifact already exists: {path}")
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(tmp, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            path.chmod(0o600)
        except PermissionError:
            pass
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


_SECRET_URL_KEYWORDS = (
    "api_key",
    "apikey",
    "token",
    "password",
    "passwd",
    "secret",
    "authorization",
    "bearer",
    "credential",
    "private_key",
)


def _is_secret_url_key(key: str) -> bool:
    normalized = str(key or "").lower().replace("-", "_")
    return any(keyword in normalized for keyword in _SECRET_URL_KEYWORDS)


def _redact_url_component_secrets(component: str) -> str:
    if not component:
        return component
    try:
        pairs = parse_qsl(component, keep_blank_values=True)
    except Exception:
        return "<redacted>" if _looks_sensitive_url(component) else component
    if pairs:
        return urlencode([(key, "<redacted>" if _is_secret_url_key(key) else value) for key, value in pairs])
    return "<redacted>" if _is_secret_url_key(component) else component


def _looks_sensitive_url(value: str) -> bool:
    if not value:
        return False
    return bool(re.search(r"://[^\s/:@]+:[^\s/@]+@", value) or re.search(r"(?:api[_-]?key|token|password|passwd|secret|authorization|bearer|credential|private_key)=", value, flags=re.IGNORECASE))


def _redact_credentialed_url_fail_closed(value: Any) -> str:
    raw = str(value or "")
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
        if not parts.scheme or not parts.netloc:
            return "<redacted>" if _looks_sensitive_url(raw) else raw
        netloc = parts.netloc
        if parts.username is not None or parts.password is not None:
            host = parts.hostname or ""
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            netloc = f"<redacted>@{host}"
            try:
                if parts.port is not None:
                    netloc = f"{netloc}:{parts.port}"
            except ValueError:
                return "<redacted>"
        query = _redact_url_component_secrets(parts.query)
        fragment = _redact_url_component_secrets(parts.fragment)
        redacted = urlunsplit((parts.scheme, netloc, parts.path, query, fragment))
        if _looks_sensitive_url(redacted):
            return "<redacted>"
        return redacted
    except Exception:
        return "<redacted>"


def _redacted_qdrant_url(config: dict[str, Any]) -> str:
    return _redact_credentialed_url_fail_closed(config.get("qdrant_url"))


def _parse_int_field(value: Any, *, field: str, default: int | None = None) -> int:
    if value is None or value == "":
        if default is not None:
            return default
        raise BackupError(f"{field} is missing")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise BackupError(f"{field} must be an integer") from exc


def _parse_optional_int_field(value: Any, *, field: str) -> int | None:
    if value is None or value == "":
        return None
    return _parse_int_field(value, field=field)


def point_checksum(record: dict[str, Any]) -> str:
    material = {"id": record.get("id"), "payload": record.get("payload") or {}, "vector": record.get("vector")}
    return _sha256_text(_canonical_json(material))


def normalize_point(point: Any) -> dict[str, Any]:
    if isinstance(point, dict):
        point_id = point.get("id")
        payload = point.get("payload")
        vector = point.get("vector")
        if vector is None and "vectors" in point:
            vector = point.get("vectors")
    else:
        point_id = getattr(point, "id", None)
        payload = getattr(point, "payload", None)
        vector = getattr(point, "vector", None)
        if vector is None:
            vector = getattr(point, "vectors", None)
    if point_id is None or str(point_id) == "":
        raise BackupError("Qdrant point is missing an explicit id")
    if not isinstance(payload, dict):
        payload = {}
    record = {"id": point_id, "vector": vector if vector is not None else [], "payload": payload}
    record["point_sha256"] = point_checksum(record)
    return record


def _point_for_upsert(record: dict[str, Any]) -> dict[str, Any]:
    return {"id": record["id"], "vector": record.get("vector") if record.get("vector") is not None else [], "payload": record.get("payload") or {}}


def _vector_size_from_client(qdrant: Any, collection_name: str, config: dict[str, Any]) -> int | None:
    try:
        size = qdrant.collection_vector_size(collection_name)
        if size is not None:
            return int(size)
    except Exception:
        pass
    try:
        return int(config.get("vector_size"))
    except Exception:
        return None


def _validate_vector_size(record: dict[str, Any], expected_size: int | None, *, scope: str) -> None:
    if expected_size is None:
        return
    vector = record.get("vector")
    if isinstance(vector, list):
        if len(vector) != expected_size:
            raise BackupError(f"{scope} point vector size mismatch for id {record.get('id')}")
        return
    if isinstance(vector, dict):
        for item in vector.values():
            if isinstance(item, list) and len(item) != expected_size:
                raise BackupError(f"{scope} point named vector size mismatch for id {record.get('id')}")
        return
    if vector in (None, []):
        return
    raise BackupError(f"{scope} point vector has unsupported shape for id {record.get('id')}")


def _jsonl_text(header: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [_canonical_json(header)]
    lines.extend(_canonical_json(record) for record in records)
    return "\n".join(lines) + "\n"


def _export_collection_to_file(
    qdrant: Any,
    config: dict[str, Any],
    *,
    scope: str,
    collection_name: str,
    out_path: Path,
    overwrite: bool,
    artifact_type: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    records = [normalize_point(point) for point in qdrant.scroll_by_filter(collection_name, {}, limit=256, with_payload=True, with_vector=True)]
    vector_size = _vector_size_from_client(qdrant, collection_name, config)
    for record in records:
        _validate_vector_size(record, vector_size, scope=scope)
    created = created_at or utc_timestamp()
    header = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": artifact_type,
        "scope": scope,
        "collection_name": collection_name,
        "created_at": created,
        "record_count": len(records),
        "vector_size": vector_size,
        "contains_raw_memory_text": True,
        "contains_vectors": True,
    }
    _safe_atomic_write_text(out_path, _jsonl_text(header, records), overwrite=overwrite)
    return {
        "scope": scope,
        "collection_name": collection_name,
        "path": str(out_path),
        "file": out_path.name,
        "count": len(records),
        "sha256": file_sha256(out_path),
        "vector_size": vector_size,
    }


def export_collection(qdrant: Any, config: dict[str, Any], *, scope: str, out_file: str | Path, overwrite: bool = False) -> dict[str, Any]:
    specs = collection_specs(config, scope)
    if len(specs) != 1:
        raise BackupError("export scope must be memory or learning")
    out_path = Path(out_file).expanduser()
    if out_path.exists() and not overwrite:
        raise BackupError(f"artifact already exists: {out_path}")
    summary = _export_collection_to_file(
        qdrant,
        config,
        scope=specs[0]["scope"],
        collection_name=specs[0]["collection_name"],
        out_path=out_path,
        overwrite=overwrite,
        artifact_type="qdrant_memory_export",
    )
    summary.update({"artifact_type": "qdrant_memory_export", "contains_raw_memory_text": True, "contains_vectors": True})
    return summary


def create_backup(qdrant: Any, config: dict[str, Any], *, hermes_home: str | Path, scope: str = "both", backup_id: str | None = None) -> dict[str, Any]:
    root = ensure_private_dir(backup_root(config, hermes_home))
    backup_id = validate_backup_id(backup_id or generate_backup_id())
    backup_dir = backup_dir_for_id(root, backup_id)
    if backup_dir.exists():
        raise BackupError(f"backup already exists: {backup_id}")
    ensure_private_dir(backup_dir)
    created_at = utc_timestamp()
    collections: dict[str, Any] = {}
    for spec in collection_specs(config, scope):
        item = _export_collection_to_file(
            qdrant,
            config,
            scope=spec["scope"],
            collection_name=spec["collection_name"],
            out_path=backup_dir / spec["file"],
            overwrite=False,
            artifact_type="qdrant_memory_backup_collection",
            created_at=created_at,
        )
        collections[spec["scope"]] = {
            "collection_name": item["collection_name"],
            "file": item["file"],
            "count": item["count"],
            "sha256": item["sha256"],
            "vector_size": item.get("vector_size"),
        }
    redacted_url = _redacted_qdrant_url(config)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "qdrant_memory_backup",
        "backup_id": backup_id,
        "created_at": created_at,
        "scope": scope,
        "qdrant_url": redacted_url,
        "redacted_qdrant_url": redacted_url,
        "contains_raw_memory_text": True,
        "contains_vectors": True,
        "collections": collections,
    }
    _safe_atomic_write_text(backup_dir / "manifest.json", _canonical_json(manifest) + "\n", overwrite=False)
    return sanitize_manifest(manifest, backup_dir=backup_dir)


def _load_manifest_from_dir(backup_dir: Path) -> dict[str, Any]:
    manifest_path = backup_dir / "manifest.json"
    if not manifest_path.exists():
        raise BackupError("backup manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BackupError(f"backup manifest is invalid: {exc}") from exc
    if not isinstance(manifest, dict):
        raise BackupError("backup manifest must be an object")
    schema_version = _parse_int_field(manifest.get("schema_version"), field="backup schema_version", default=0)
    if schema_version != SCHEMA_VERSION:
        raise BackupError("unsupported backup schema_version")
    if str(manifest.get("artifact_type") or "") != "qdrant_memory_backup":
        raise BackupError("unsupported backup artifact_type")
    validate_backup_id(str(manifest.get("backup_id") or ""))
    return manifest


def _safe_collection_file(backup_dir: Path, file_name: str) -> Path:
    if not file_name or "/" in file_name or "\\" in file_name or file_name in {".", ".."}:
        raise BackupError("invalid backup collection file path")
    path = (backup_dir / file_name).resolve()
    if path.parent != backup_dir.resolve():
        raise BackupError("invalid backup collection file path")
    return path


def _read_collection_file(path: Path, *, expected_sha256: str, scope: str, expected_vector_size: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise BackupError(f"backup collection file is missing: {path.name}")
    actual_sha = file_sha256(path)
    if actual_sha != expected_sha256:
        raise BackupError(f"backup collection file checksum mismatch: {path.name}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise BackupError(f"backup collection file is empty: {path.name}")
    try:
        header = json.loads(lines[0])
    except Exception as exc:
        raise BackupError(f"backup collection header is invalid: {path.name}: {exc}") from exc
    if not isinstance(header, dict):
        raise BackupError(f"backup collection header must be an object: {path.name}")
    header_schema_version = _parse_int_field(header.get("schema_version"), field=f"backup collection header schema_version ({path.name})", default=0)
    if header_schema_version != SCHEMA_VERSION:
        raise BackupError(f"backup collection schema_version is unsupported: {path.name}")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except Exception as exc:
            raise BackupError(f"backup point record is invalid at {path.name}:{line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise BackupError(f"backup point record must be an object at {path.name}:{line_number}")
        normalized = {"id": record.get("id"), "vector": record.get("vector"), "payload": record.get("payload") or {}}
        if normalized["id"] is None or str(normalized["id"]) == "":
            raise BackupError(f"backup point record is missing id at {path.name}:{line_number}")
        seen_key = str(normalized["id"])
        if seen_key in seen:
            raise BackupError(f"duplicate point id in backup: {normalized['id']}")
        seen.add(seen_key)
        expected_point_sha = str(record.get("point_sha256") or "")
        actual_point_sha = point_checksum(normalized)
        if expected_point_sha != actual_point_sha:
            raise BackupError(f"backup point checksum mismatch for id {normalized['id']}")
        normalized["point_sha256"] = actual_point_sha
        _validate_vector_size(normalized, expected_vector_size, scope=scope)
        records.append(normalized)
    return records


def load_backup(backup_id: str, config: dict[str, Any], *, hermes_home: str | Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], Path]:
    root = ensure_private_dir(backup_root(config, hermes_home))
    backup_dir = backup_dir_for_id(root, backup_id)
    if not backup_dir.exists() or not backup_dir.is_dir():
        raise BackupError(f"unknown backup: {backup_id}")
    manifest = _load_manifest_from_dir(backup_dir)
    collections = manifest.get("collections")
    if not isinstance(collections, dict) or not collections:
        raise BackupError("backup manifest has no collections")
    records_by_scope: dict[str, list[dict[str, Any]]] = {}
    for scope, details in collections.items():
        if scope not in SCOPES:
            raise BackupError(f"unsupported backup collection scope: {scope}")
        if not isinstance(details, dict):
            raise BackupError(f"backup collection manifest is invalid: {scope}")
        file_path = _safe_collection_file(backup_dir, str(details.get("file") or ""))
        expected_vector_size = _parse_optional_int_field(details.get("vector_size"), field=f"backup collection vector_size ({scope})")
        records = _read_collection_file(
            file_path,
            expected_sha256=str(details.get("sha256") or ""),
            scope=scope,
            expected_vector_size=expected_vector_size,
        )
        expected_count = _parse_int_field(details.get("count"), field=f"backup collection count ({scope})", default=0)
        if len(records) != expected_count:
            raise BackupError(f"backup collection count mismatch: {scope}")
        records_by_scope[scope] = records
    return manifest, records_by_scope, backup_dir


def sanitize_manifest(manifest: dict[str, Any], *, backup_dir: Path | None = None, checksum_ok: bool | None = None) -> dict[str, Any]:
    collections: dict[str, Any] = {}
    for scope, details in (manifest.get("collections") or {}).items():
        if isinstance(details, dict):
            collections[str(scope)] = {
                "collection_name": details.get("collection_name"),
                "file": details.get("file"),
                "count": _parse_int_field(details.get("count"), field=f"backup collection count ({scope})", default=0),
                "sha256": details.get("sha256"),
                "vector_size": details.get("vector_size"),
            }
    redacted_url = _redact_credentialed_url_fail_closed(manifest.get("qdrant_url") or manifest.get("redacted_qdrant_url"))
    result = {
        "schema_version": manifest.get("schema_version"),
        "artifact_type": manifest.get("artifact_type"),
        "backup_id": manifest.get("backup_id"),
        "created_at": manifest.get("created_at"),
        "scope": manifest.get("scope"),
        "qdrant_url": redacted_url,
        "redacted_qdrant_url": redacted_url,
        "contains_raw_memory_text": bool(manifest.get("contains_raw_memory_text")),
        "contains_vectors": bool(manifest.get("contains_vectors")),
        "collections": collections,
    }
    if backup_dir is not None:
        result["backup_dir"] = str(backup_dir)
    if checksum_ok is not None:
        result["checksum_ok"] = bool(checksum_ok)
    return result


def list_backups(config: dict[str, Any], *, hermes_home: str | Path) -> dict[str, Any]:
    root = ensure_private_dir(backup_root(config, hermes_home))
    backups: list[dict[str, Any]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            manifest = _load_manifest_from_dir(child)
            backups.append(sanitize_manifest(manifest, backup_dir=child))
        except BackupError:
            continue
    backups.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return {"backup_root": str(root), "count": len(backups), "backups": backups}


def inspect_backup(backup_id: str, config: dict[str, Any], *, hermes_home: str | Path) -> dict[str, Any]:
    manifest, _records, backup_dir = load_backup(backup_id, config, hermes_home=hermes_home)
    return sanitize_manifest(manifest, backup_dir=backup_dir, checksum_ok=True)


def _chunks(items: list[Any], size: int = 256) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _collection_name_for_scope(config: dict[str, Any], scope: str, manifest_details: dict[str, Any]) -> str:
    configured = "collection_name" if scope == "memory" else "learning_collection_name"
    return str(config.get(configured) or manifest_details.get("collection_name") or "")


def preflight_restore_targets(qdrant: Any, config: dict[str, Any], manifest: dict[str, Any], records_by_scope: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Validate all restore target collections before any live upsert can run."""

    manifest_collections = manifest.get("collections") if isinstance(manifest.get("collections"), dict) else {}
    target_sizes: dict[str, Any] = {}
    errors: list[str] = []
    for scope, records in records_by_scope.items():
        details = manifest_collections.get(scope) if isinstance(manifest_collections, dict) else {}
        details = details if isinstance(details, dict) else {}
        collection_name = _collection_name_for_scope(config, scope, details)
        if not collection_name:
            errors.append(f"{scope} target collection name is missing")
            continue
        try:
            target_size_raw = qdrant.collection_vector_size(collection_name)
        except Exception as exc:
            raise BackupError(f"{scope} target collection preflight failed: {collection_name}") from exc
        target_size = _parse_optional_int_field(target_size_raw, field=f"{scope} target vector_size")
        if target_size is None:
            errors.append(f"{scope} target vector size is unavailable: {collection_name}")
            continue
        manifest_size = _parse_optional_int_field(details.get("vector_size"), field=f"backup collection vector_size ({scope})")
        if manifest_size is not None and manifest_size != target_size:
            errors.append(f"{scope} target vector size mismatch for {collection_name}: backup={manifest_size} target={target_size}")
            continue
        for record in records:
            try:
                _validate_vector_size(record, target_size, scope=scope)
            except BackupError as exc:
                errors.append(str(exc))
                break
        target_sizes[scope] = {"collection_name": collection_name, "vector_size": target_size}
    if errors:
        raise BackupError("; ".join(errors))
    return target_sizes


def plan_restore(qdrant: Any, config: dict[str, Any], manifest: dict[str, Any], records_by_scope: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    plan_collections: dict[str, Any] = {}
    upserts_by_scope: dict[str, list[dict[str, Any]]] = {}
    manifest_collections = manifest.get("collections") or {}
    for scope, records in records_by_scope.items():
        details = manifest_collections.get(scope) if isinstance(manifest_collections, dict) else {}
        details = details if isinstance(details, dict) else {}
        collection_name = _collection_name_for_scope(config, scope, details)
        backup_by_id = {str(record["id"]): record for record in records}
        existing_by_id: dict[str, dict[str, Any]] = {}
        ids = [record["id"] for record in records]
        for batch in _chunks(ids):
            for point in qdrant.retrieve(collection_name, batch, with_payload=True, with_vector=True):
                existing = normalize_point(point)
                existing_by_id[str(existing["id"])] = existing
        same = 0
        changed: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        for point_id, record in backup_by_id.items():
            existing = existing_by_id.get(point_id)
            if existing is None:
                missing.append(record)
            elif existing.get("point_sha256") == record.get("point_sha256"):
                same += 1
            else:
                changed.append(record)
        to_upsert = changed + missing
        upserts_by_scope[scope] = [_point_for_upsert(record) for record in to_upsert]
        plan_collections[scope] = {
            "collection_name": collection_name,
            "total": len(records),
            "same": same,
            "changed": len(changed),
            "missing": len(missing),
            "would_upsert": len(to_upsert),
            "upserted": 0,
        }
    return plan_collections, upserts_by_scope


def restore_backup(
    qdrant: Any,
    config: dict[str, Any],
    *,
    hermes_home: str | Path,
    backup_id: str,
    dry_run: bool = True,
    backup_first: bool = False,
) -> dict[str, Any]:
    manifest, records_by_scope, _backup_dir = load_backup(backup_id, config, hermes_home=hermes_home)
    preflight_restore_targets(qdrant, config, manifest, records_by_scope)
    collections, upserts_by_scope = plan_restore(qdrant, config, manifest, records_by_scope)
    result: dict[str, Any] = {
        "backup_id": manifest.get("backup_id"),
        "dry_run": bool(dry_run),
        "validated": True,
        "collections": collections,
    }
    if dry_run:
        result["applied"] = False
        return result
    pre_restore = create_backup(qdrant, config, hermes_home=hermes_home, scope="both")
    pre_restore_backup_id = str(pre_restore.get("backup_id") or "")
    result["pre_restore_backup_id"] = pre_restore_backup_id
    manifest_collections = manifest.get("collections") or {}
    total_upserted = 0
    for scope, points in upserts_by_scope.items():
        if not points:
            continue
        details = manifest_collections.get(scope) if isinstance(manifest_collections, dict) else {}
        collection_name = _collection_name_for_scope(config, scope, details if isinstance(details, dict) else {})
        qdrant.upsert(collection_name, points)
        collections[scope]["upserted"] = len(points)
        total_upserted += len(points)
    result.update({"applied": True, "upserted": total_upserted})
    return result
