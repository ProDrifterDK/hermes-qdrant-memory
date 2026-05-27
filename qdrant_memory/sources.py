from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import unquote, urlsplit

from .schema import sanitize_point_id_links, valid_fact_status, valid_memory_kind, valid_relation_type


DEFAULT_MAX_CHARS = 8000
HARD_MAX_CHARS = 100_000
DEFAULT_MAX_FILE_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_LINE_SPAN = 2000
EXPAND_METADATA_STRING_CAP = 512
EXPAND_METADATA_TOKEN_CAP = 128
EXPAND_LOCATOR_HEADING_CAP = 200
MAX_SAFE_LINE_NUMBER = 2_147_483_647
_BINARY_EXTENSIONS = {
    ".7z",
    ".bin",
    ".bmp",
    ".bz2",
    ".class",
    ".dll",
    ".dmg",
    ".doc",
    ".docx",
    ".dylib",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".o",
    ".pdf",
    ".png",
    ".pyc",
    ".pyo",
    ".rar",
    ".so",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".wasm",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}


class SourceResolver(Protocol):
    schemes: set[str]

    def stat(
        self,
        source_uri: str,
        locator: dict[str, Any] | None = None,
        *,
        content_hash: str | None = None,
        source_modified_at: str | None = None,
    ) -> dict[str, Any]:
        ...

    def expand(
        self,
        source_uri: str,
        locator: dict[str, Any] | None = None,
        *,
        mode: str = "excerpt",
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> dict[str, Any]:
        ...


def _scheme(source_uri: str) -> str:
    try:
        return urlsplit(str(source_uri or "")).scheme.lower()
    except Exception:
        return ""


def _normalize_max_chars(max_chars: int | None) -> int:
    try:
        parsed = int(max_chars if max_chars is not None else DEFAULT_MAX_CHARS)
    except Exception:
        parsed = DEFAULT_MAX_CHARS
    return max(0, min(parsed, HARD_MAX_CHARS))


def _bounded_text(text: str, max_chars: int | None) -> tuple[str, bool]:
    budget = _normalize_max_chars(max_chars)
    if len(text) <= budget:
        return text, False
    return text[:budget], True


def _unsupported(source_uri: str, *, reason: str = "unsupported_scheme", message: str | None = None) -> dict[str, Any]:
    scheme = _scheme(source_uri)
    if not message:
        message = f"Unsupported source URI scheme: {scheme or '<missing>'}"
    return {
        "source_uri": source_uri,
        "scheme": scheme,
        "supported": False,
        "status": "unsupported",
        "reason": reason,
        "message": message,
    }


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _parse_source_modified_at(value: str | None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except Exception:
        return None


def _point_payload(point: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(point, dict):
        return {}
    payload = point.get("payload")
    return payload if isinstance(payload, dict) else {}


def _point_text(point: dict[str, Any] | None) -> str:
    payload = _point_payload(point)
    for key in ("text", "lesson", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _point_id(point: dict[str, Any] | None, fallback: str = "") -> str:
    if isinstance(point, dict) and point.get("id") is not None:
        return str(point.get("id"))
    return fallback


class SourceResolverRegistry:
    """Resolve source URI schemes to bounded, read-only source resolvers."""

    def __init__(self, resolvers: list[SourceResolver] | None = None):
        self._resolvers: dict[str, SourceResolver] = {}
        for resolver in resolvers or []:
            self.register(resolver)

    def register(self, resolver: SourceResolver) -> None:
        for scheme in getattr(resolver, "schemes", set()):
            normalized = str(scheme).lower().strip()
            if normalized:
                self._resolvers[normalized] = resolver

    def resolver_for(self, source_uri: str) -> SourceResolver | None:
        return self._resolvers.get(_scheme(source_uri))

    def stat(
        self,
        source_uri: str,
        locator: dict[str, Any] | None = None,
        *,
        content_hash: str | None = None,
        source_modified_at: str | None = None,
    ) -> dict[str, Any]:
        resolver = self.resolver_for(source_uri)
        if not resolver:
            return _compact_status_response(_unsupported(source_uri))
        try:
            return _compact_status_response(resolver.stat(source_uri, locator, content_hash=content_hash, source_modified_at=source_modified_at))
        except Exception:
            return _compact_status_response(
                {
                    "source_uri": source_uri,
                    "scheme": _scheme(source_uri),
                    "supported": True,
                    "status": "unknown",
                    "reason": "resolver_error",
                    "message": "Resolver failed while checking source status.",
                }
            )

    def expand(
        self,
        source_uri: str,
        locator: dict[str, Any] | None = None,
        *,
        mode: str = "excerpt",
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> dict[str, Any]:
        budget = _normalize_max_chars(max_chars)
        resolver = self.resolver_for(source_uri)
        if not resolver:
            result = _unsupported(source_uri)
            result.update({"mode": mode, "text": "", "chars": 0, "max_chars": budget, "truncated": False})
            return _compact_expand_response(result)
        try:
            result = resolver.expand(source_uri, locator, mode=mode, max_chars=budget)
        except Exception:
            return _compact_expand_response(
                {
                    "source_uri": source_uri,
                    "scheme": _scheme(source_uri),
                    "supported": True,
                    "status": "unknown",
                    "reason": "resolver_error",
                    "message": "Resolver failed while expanding source.",
                    "mode": mode,
                    "text": "",
                    "chars": 0,
                    "max_chars": budget,
                    "truncated": False,
                }
            )
        text = str(result.get("text") or "")
        bounded, truncated = _bounded_text(text, budget)
        if bounded != text or truncated:
            result = dict(result)
            result["text"] = bounded
            result["truncated"] = True
        result.setdefault("max_chars", budget)
        result["chars"] = len(str(result.get("text") or ""))
        return _compact_expand_response(result)


class FileSourceResolver:
    schemes = {"file"}

    def __init__(self, *, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES, max_line_span: int = DEFAULT_MAX_LINE_SPAN):
        self.max_file_bytes = max(1, int(max_file_bytes))
        self.max_line_span = max(1, int(max_line_span))

    def _path_error(self, source_uri: str, reason: str, message: str) -> dict[str, Any]:
        return {
            "source_uri": source_uri,
            "scheme": "file",
            "supported": False,
            "status": "unsupported",
            "exists": False,
            "changed": None,
            "reason": reason,
            "message": message,
        }

    def _has_dot_segments(self, path: str) -> bool:
        return any(part in {".", ".."} for part in path.replace("\\", "/").split("/"))

    def _path_result_from_uri(self, source_uri: str) -> tuple[Path | None, dict[str, Any] | None]:
        try:
            parts = urlsplit(str(source_uri or ""))
        except Exception as exc:
            return None, self._path_error(source_uri, "malformed_file_uri", f"Malformed file URI: {exc}")
        if parts.scheme.lower() != "file":
            return None, self._path_error(source_uri, "unsupported_scheme", "Expected a file:// source URI")
        if parts.netloc and parts.netloc not in {"localhost", "127.0.0.1"}:
            return None, self._path_error(source_uri, "non_local_file_uri", "Only local file:// sources are supported")
        raw_path = parts.path or ""
        decoded_path = unquote(raw_path)
        if not decoded_path.startswith("/"):
            return None, self._path_error(source_uri, "relative_file_uri", "file:// sources must use an absolute path")
        if self._has_dot_segments(raw_path) or self._has_dot_segments(decoded_path):
            return None, self._path_error(source_uri, "unsafe_file_uri", "file:// sources cannot contain dot-segment path traversal")
        path = Path(decoded_path).expanduser()
        if not path.is_absolute():
            return None, self._path_error(source_uri, "relative_file_uri", "file:// sources must use an absolute path")
        return path, None

    def _path_from_uri(self, source_uri: str) -> Path | None:
        path, _error = self._path_result_from_uri(source_uri)
        return path

    def _base(self, source_uri: str, path: Path | None = None, *, include_path: bool = True) -> dict[str, Any]:
        payload = {"source_uri": source_uri, "scheme": "file"}
        if include_path and path is not None:
            payload["path"] = str(path)
        return payload

    def _binary_reason(self, path: Path) -> str | None:
        if path.suffix.lower() in _BINARY_EXTENSIONS:
            return "binary_source"
        try:
            with path.open("rb") as handle:
                sample = handle.read(4096)
        except Exception:
            return "unreadable_source"
        if b"\x00" in sample:
            return "binary_source"
        return None

    def _line_range(self, locator: dict[str, Any] | None) -> tuple[int | None, int | None, bool]:
        if not isinstance(locator, dict):
            return None, None, False
        try:
            start = int(locator.get("line_start")) if locator.get("line_start") is not None else None
        except Exception:
            start = None
        try:
            end = int(locator.get("line_end")) if locator.get("line_end") is not None else None
        except Exception:
            end = None
        if start is None and end is None:
            return None, None, False
        if start is None:
            start = end or 1
        if end is None:
            end = start
        start = max(1, int(start))
        end = max(start, int(end))
        capped = False
        if end - start + 1 > self.max_line_span:
            end = start + self.max_line_span - 1
            capped = True
        return start, end, capped

    def _read_excerpt(self, path: Path, locator: dict[str, Any] | None, *, mode: str, max_chars: int) -> tuple[str, dict[str, Any], bool]:
        start, end, capped_lines = self._line_range(locator)
        locator_out = expand_locator_metadata(locator) if isinstance(locator, dict) else {}
        text = ""
        truncated = bool(capped_lines)
        char_cap = max(max_chars, 0) + 1
        if start is not None and end is not None:
            locator_out.update({"line_start": start, "line_end": end})
            parts: list[str] = []
            remaining_cap = char_cap
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    line_no = 1
                    stop = False
                    while line_no <= end and not stop:
                        while True:
                            read_size = 4096 if line_no < start else max(1, min(4096, remaining_cap))
                            line_part = handle.readline(read_size)
                            if line_part == "":
                                stop = True
                                break
                            line_complete = line_part.endswith("\n") or len(line_part) < read_size
                            if line_no >= start:
                                parts.append(line_part)
                                remaining_cap -= len(line_part)
                                if remaining_cap <= 0:
                                    truncated = True
                                    stop = True
                                    break
                            if line_complete:
                                break
                        line_no += 1
            except Exception:
                raise
            text = "".join(parts)
        else:
            if mode == "source" and path.stat().st_size > self.max_file_bytes:
                # Still bounded; never read unbounded huge files.
                truncated = True
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                text = handle.read(char_cap)
                if len(text) > max_chars:
                    truncated = True
        bounded, bounded_truncated = _bounded_text(text, max_chars)
        return bounded, locator_out, bool(truncated or bounded_truncated)

    def stat(
        self,
        source_uri: str,
        locator: dict[str, Any] | None = None,
        *,
        content_hash: str | None = None,
        source_modified_at: str | None = None,
    ) -> dict[str, Any]:
        path, uri_error = self._path_result_from_uri(source_uri)
        base = self._base(source_uri, path)
        if uri_error:
            return {**uri_error, **base}
        if path is None:
            return {
                **base,
                "supported": False,
                "status": "unsupported",
                "exists": False,
                "changed": None,
                "reason": "non_local_file_uri",
                "message": "Only local file:// sources are supported",
            }
        if not path.exists() or not path.is_file():
            return {
                **base,
                "supported": True,
                "status": "missing",
                "exists": False,
                "changed": None,
                "message": "Source file is missing",
            }
        binary_reason = self._binary_reason(path)
        if binary_reason:
            return {
                **base,
                "supported": False,
                "status": "unsupported",
                "exists": True,
                "changed": None,
                "reason": binary_reason,
                "message": "Binary or unreadable file source is not expanded",
            }

        stat = path.stat()
        result: dict[str, Any] = {
            **base,
            "supported": True,
            "status": "unknown",
            "exists": True,
            "changed": None,
            "file_size": stat.st_size,
            "source_modified_at": source_modified_at,
        }
        expected = str(content_hash or "").strip()
        if expected:
            result.update({"status": "exists", "changed": False})
            if stat.st_size > self.max_file_bytes:
                result.update(
                    {
                        "status": "unknown",
                        "changed": None,
                        "reason": "source_too_large",
                        "message": "Source file exceeds the expansion size budget",
                    }
                )
                return result
            try:
                text, locator_out, truncated = self._read_excerpt(path, locator, mode="excerpt", max_chars=HARD_MAX_CHARS)
            except Exception:
                return {**result, "status": "unknown", "changed": None, "reason": "read_error", "message": "Source file could not be read."}
            result["locator"] = locator_out
            if truncated:
                result.update(
                    {
                        "status": "unknown",
                        "changed": None,
                        "reason": "hash_budget_exceeded",
                        "message": "Source excerpt exceeded verification budget",
                    }
                )
                return result
            actual = _sha256_text(text.strip())
            result["expected_hash"] = expected
            result["actual_hash"] = actual
            if actual != expected:
                result["status"] = "changed"
                result["changed"] = True
            return result

        expected_mtime = _parse_source_modified_at(source_modified_at)
        if source_modified_at and expected_mtime is None:
            result.update({"reason": "invalid_source_modified_at", "message": "Source modified timestamp is invalid"})
            return result
        if expected_mtime is not None:
            result.update({"status": "exists", "changed": False})
            if abs(stat.st_mtime - expected_mtime) > 1e-6:
                result["status"] = "changed"
                result["changed"] = True
            return result

        result.update(
            {
                "reason": "missing_verification_metadata",
                "message": "No content_hash or source_modified_at metadata is available to verify freshness",
            }
        )
        return result

    def expand(
        self,
        source_uri: str,
        locator: dict[str, Any] | None = None,
        *,
        mode: str = "excerpt",
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> dict[str, Any]:
        budget = _normalize_max_chars(max_chars)
        path, uri_error = self._path_result_from_uri(source_uri)
        base = self._base(source_uri, path, include_path=False)
        if mode not in {"excerpt", "source"}:
            return _compact_expand_response(
                {
                    **base,
                    "supported": False,
                    "status": "unsupported",
                    "reason": "unsupported_mode",
                    "mode": mode,
                    "text": "",
                    "chars": 0,
                    "max_chars": budget,
                    "truncated": False,
                    "message": f"file:// resolver does not support mode={mode}",
                }
            )
        if uri_error:
            return _compact_expand_response(
                {
                    **uri_error,
                    **base,
                    "mode": mode,
                    "text": "",
                    "chars": 0,
                    "max_chars": budget,
                    "truncated": False,
                }
            )
        if path is None:
            return _compact_expand_response(
                {
                    **base,
                    "supported": False,
                    "status": "unsupported",
                    "reason": "non_local_file_uri",
                    "mode": mode,
                    "text": "",
                    "chars": 0,
                    "max_chars": budget,
                    "truncated": False,
                    "message": "Only local file:// sources are supported",
                }
            )
        if not path.exists() or not path.is_file():
            return _compact_expand_response(
                {
                    **base,
                    "supported": True,
                    "status": "missing",
                    "exists": False,
                    "mode": mode,
                    "text": "",
                    "chars": 0,
                    "max_chars": budget,
                    "truncated": False,
                    "message": "Source file is missing",
                }
            )
        file_stat = path.stat()
        binary_reason = self._binary_reason(path)
        if binary_reason:
            return _compact_expand_response(
                {
                    **base,
                    "supported": False,
                    "status": "unsupported",
                    "exists": True,
                    "reason": binary_reason,
                    "mode": mode,
                    "text": "",
                    "chars": 0,
                    "max_chars": budget,
                    "truncated": False,
                    "message": "Binary or unreadable file source is not expanded",
                }
            )
        if file_stat.st_size > self.max_file_bytes:
            return _compact_expand_response(
                {
                    **base,
                    "supported": False,
                    "status": "unsupported",
                    "exists": True,
                    "reason": "source_too_large",
                    "mode": mode,
                    "text": "",
                    "chars": 0,
                    "max_chars": budget,
                    "truncated": False,
                    "file_size": file_stat.st_size,
                    "message": "Source file exceeds the expansion size budget",
                }
            )
        try:
            text, locator_out, truncated = self._read_excerpt(path, locator, mode=mode, max_chars=budget)
        except Exception:
            return _compact_expand_response(
                {
                    **base,
                    "supported": True,
                    "status": "unknown",
                    "reason": "read_error",
                    "mode": mode,
                    "text": "",
                    "chars": 0,
                    "max_chars": budget,
                    "truncated": False,
                    "message": "Source file could not be read.",
                }
            )
        return _compact_expand_response(
            {
                **base,
                "supported": True,
                "status": "exists",
                "exists": True,
                "mode": mode,
                "locator": expand_locator_metadata(locator_out),
                "text": text,
                "chars": len(text),
                "max_chars": budget,
                "truncated": truncated,
                "file_size": file_stat.st_size,
            }
        )


class MemoryPointResolver:
    schemes = {"memory"}

    def __init__(self, lookup: Callable[[str], dict[str, Any] | None] | None = None):
        self.lookup = lookup

    def _parse_point_id(self, source_uri: str) -> str:
        parts = urlsplit(source_uri)
        if parts.scheme.lower() != "memory":
            return ""
        if parts.netloc == "point":
            return unquote(parts.path.lstrip("/"))
        path = parts.path.strip("/")
        if path.startswith("point/"):
            return unquote(path.split("/", 1)[1])
        return ""

    def _base(self, source_uri: str, point_id: str = "") -> dict[str, Any]:
        payload = {"source_uri": source_uri, "scheme": "memory"}
        if point_id:
            payload["point_id"] = point_id
        return payload

    def _lookup(self, point_id: str) -> dict[str, Any] | None:
        if not self.lookup:
            return None
        result = self.lookup(point_id)
        return result if isinstance(result, dict) else None

    def stat(
        self,
        source_uri: str,
        locator: dict[str, Any] | None = None,
        *,
        content_hash: str | None = None,
        source_modified_at: str | None = None,
    ) -> dict[str, Any]:
        point_id = self._parse_point_id(source_uri)
        base = self._base(source_uri, point_id)
        if not point_id:
            return {**base, "supported": False, "status": "unsupported", "exists": False, "reason": "invalid_memory_uri", "message": "Expected memory://point/<id>"}
        if not self.lookup:
            return {**base, "supported": True, "status": "unknown", "exists": None, "reason": "lookup_unavailable", "message": "No exact point lookup is configured"}
        point = self._lookup(point_id)
        if not point:
            return {**base, "supported": True, "status": "missing", "exists": False, "changed": None, "message": "Memory point is missing"}
        return {**base, "supported": True, "status": "exists", "exists": True, "changed": False}

    def expand(
        self,
        source_uri: str,
        locator: dict[str, Any] | None = None,
        *,
        mode: str = "excerpt",
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> dict[str, Any]:
        budget = _normalize_max_chars(max_chars)
        if mode not in {"excerpt", "source"}:
            point_id = self._parse_point_id(source_uri)
            return _compact_expand_response({**self._base(source_uri, point_id), "supported": False, "status": "unsupported", "reason": "unsupported_mode", "mode": mode, "text": "", "chars": 0, "max_chars": budget, "truncated": False, "message": f"memory:// resolver does not support mode={mode}"})
        point_id = self._parse_point_id(source_uri)
        base = self._base(source_uri, point_id)
        if not point_id:
            return _compact_expand_response({**base, "supported": False, "status": "unsupported", "reason": "invalid_memory_uri", "mode": mode, "text": "", "chars": 0, "max_chars": budget, "truncated": False, "message": "Expected memory://point/<id>"})
        if not self.lookup:
            return _compact_expand_response({**base, "supported": True, "status": "unknown", "reason": "lookup_unavailable", "mode": mode, "text": "", "chars": 0, "max_chars": budget, "truncated": False, "message": "No exact point lookup is configured"})
        point = self._lookup(point_id)
        if not point:
            return _compact_expand_response({**base, "supported": True, "status": "missing", "exists": False, "mode": mode, "text": "", "chars": 0, "max_chars": budget, "truncated": False, "message": "Memory point is missing"})
        payload = _point_payload(point)
        text, truncated = _bounded_text(_point_text(point), budget)
        result = {
            **base,
            "supported": True,
            "status": "exists",
            "exists": True,
            "mode": mode,
            "text": text,
            "chars": len(text),
            "max_chars": budget,
            "truncated": truncated,
        }
        src = expand_source_metadata(payload)
        if src:
            result["source"] = src
        return _compact_expand_response(result)


def retrieve_point(qdrant: Any, collection_name: str, point_id: str, *, with_payload: bool = True, with_vector: bool = False) -> dict[str, Any] | None:
    if not qdrant or not callable(getattr(qdrant, "retrieve", None)):
        return None
    points = qdrant.retrieve(collection_name, [point_id], with_payload=with_payload, with_vector=with_vector)
    if not isinstance(points, list) or not points:
        return None
    for point in points:
        if isinstance(point, dict) and str(point.get("id")) == str(point_id):
            return point
    return None


def source_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return expand_source_metadata(payload)


def expand_locator_metadata(locator: dict[str, Any] | None) -> dict[str, Any]:
    """Return locator metadata safe for bounded expand responses."""
    if not isinstance(locator, dict):
        return {}
    result: dict[str, Any] = {}
    for key in ("line_start", "line_end"):
        line = _safe_line_number(locator.get(key))
        if line is not None:
            result[key] = line
    heading = _compact_expand_string(locator.get("heading"), max_chars=EXPAND_LOCATOR_HEADING_CAP)
    if heading is not None:
        result["heading"] = heading
    return result


def _compact_expand_string(value: Any, *, max_chars: int = EXPAND_METADATA_STRING_CAP) -> str | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[: max(0, max_chars)]


def _compact_expand_response(result: dict[str, Any]) -> dict[str, Any]:
    """Return an expand response with bounded top-level metadata fields."""
    compacted = dict(result)
    field_caps = {
        "source_uri": EXPAND_METADATA_STRING_CAP,
        "point_id": EXPAND_METADATA_STRING_CAP,
        "message": EXPAND_METADATA_STRING_CAP,
        "reason": EXPAND_METADATA_STRING_CAP,
        "scheme": EXPAND_METADATA_TOKEN_CAP,
    }
    for key, max_chars in field_caps.items():
        if key not in compacted:
            continue
        value = _compact_expand_string(compacted.get(key), max_chars=max_chars)
        if value is None:
            compacted.pop(key, None)
        else:
            compacted[key] = value
    return compacted


def _compact_status_response(result: dict[str, Any]) -> dict[str, Any]:
    """Return inspect/trace/status metadata safe for public tool and CLI output."""
    compacted = dict(result)
    field_caps = {
        "source_uri": EXPAND_METADATA_STRING_CAP,
        "point_id": EXPAND_METADATA_STRING_CAP,
        "message": EXPAND_METADATA_STRING_CAP,
        "reason": EXPAND_METADATA_STRING_CAP,
        "scheme": EXPAND_METADATA_TOKEN_CAP,
        "path": EXPAND_METADATA_STRING_CAP,
        "collection": EXPAND_METADATA_TOKEN_CAP,
        "collection_name": EXPAND_METADATA_STRING_CAP,
        "direction": EXPAND_METADATA_TOKEN_CAP,
        "memory_kind": EXPAND_METADATA_TOKEN_CAP,
        "derivation_type": EXPAND_METADATA_TOKEN_CAP,
        "relation_type": EXPAND_METADATA_TOKEN_CAP,
        "source_type": EXPAND_METADATA_TOKEN_CAP,
        "expand_hint": EXPAND_METADATA_STRING_CAP,
        "content_hash": EXPAND_METADATA_STRING_CAP,
        "expected_hash": EXPAND_METADATA_STRING_CAP,
        "actual_hash": EXPAND_METADATA_STRING_CAP,
        "source_modified_at": EXPAND_METADATA_STRING_CAP,
        "fact_status": EXPAND_METADATA_TOKEN_CAP,
        "observed_at": EXPAND_METADATA_STRING_CAP,
        "valid_from": EXPAND_METADATA_STRING_CAP,
        "valid_until": EXPAND_METADATA_STRING_CAP,
    }
    for key, max_chars in field_caps.items():
        if key not in compacted:
            continue
        value = _compact_expand_string(compacted.get(key), max_chars=max_chars)
        if value is None:
            compacted.pop(key, None)
        else:
            compacted[key] = value
    if isinstance(compacted.get("locator"), dict):
        locator = expand_locator_metadata(compacted.get("locator"))
        if locator:
            compacted["locator"] = locator
        else:
            compacted.pop("locator", None)
    return compacted


def _compact_expand_token(value: Any) -> str | None:
    text = _compact_expand_string(value, max_chars=EXPAND_METADATA_TOKEN_CAP)
    if text is None:
        return None
    safe_chars = set("abcdefghijklmnopqrstuvwxyz0123456789_.-:/")
    if all(char in safe_chars for char in text):
        return text
    return None


def _compact_expand_hash(value: Any) -> str | None:
    text = _compact_expand_string(value)
    if text is None:
        return None
    safe_chars = set("abcdefghijklmnopqrstuvwxyz0123456789_.-:")
    if all(char in safe_chars for char in text):
        return text
    return None


def _safe_line_number(value: Any) -> int | None:
    if isinstance(value, bool) or isinstance(value, (dict, list, tuple, set)):
        return None
    text = str(value).strip()
    if not text or len(text) > 16:
        return None
    try:
        line = int(text)
    except Exception:
        return None
    if line < 1 or line > MAX_SAFE_LINE_NUMBER:
        return None
    return line


def expand_source_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Return compact source metadata safe for bounded expand responses.

    Expand returns bounded text. Keep inspect/trace as the metadata-oriented
    surfaces and avoid returning legacy or attacker-controlled nested values,
    local file hashes/paths, and derivation edges here.
    """
    metadata: dict[str, Any] = {}
    for key in ("source_uri", "source_modified_at"):
        value = _compact_expand_string(payload.get(key))
        if value is not None:
            metadata[key] = value
    memory_kind = valid_memory_kind(payload.get("memory_kind"))
    if memory_kind is not None:
        metadata["memory_kind"] = memory_kind
    for key in ("source_type", "derivation_type"):
        value = _compact_expand_token(payload.get(key))
        if value is not None:
            metadata[key] = value
    content_hash = _compact_expand_hash(payload.get("content_hash"))
    if content_hash is not None:
        metadata["content_hash"] = content_hash
    raw_locator = payload.get("locator")
    locator = expand_locator_metadata(raw_locator if isinstance(raw_locator, dict) else None)
    if locator:
        metadata["locator"] = locator
    for key in ("canonical", "stale", "requires_review"):
        value = payload.get(key)
        if isinstance(value, bool):
            metadata[key] = value
    return metadata


def _temporal_fact_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    status = valid_fact_status(payload.get("fact_status"))
    if status:
        metadata["fact_status"] = status
    for key in ("observed_at", "valid_from", "valid_until"):
        value = _compact_expand_string(payload.get(key))
        if value is not None:
            metadata[key] = value
    for key in ("supersedes", "superseded_by", "invalidated_by"):
        links = sanitize_point_id_links(payload.get(key))
        if links:
            metadata[key] = links
    return metadata


def _compact_assertion_edge(edge: Any) -> dict[str, Any]:
    if isinstance(edge, str):
        source_uri = _compact_expand_string(edge)
        if source_uri is not None and "://" in source_uri:
            return {"source_uri": source_uri}
        return {}
    if not isinstance(edge, dict):
        return {}
    compacted: dict[str, Any] = {}
    for key in ("source_uri", "point_id", "source_modified_at"):
        value = _compact_expand_string(edge.get(key))
        if value is not None:
            compacted[key] = value
    for key in ("source_type", "derivation_type"):
        value = _compact_expand_token(edge.get(key))
        if value is not None:
            compacted[key] = value
    relation_type = valid_relation_type(edge.get("relation_type"))
    if relation_type is not None:
        compacted["relation_type"] = relation_type
    content_hash = _compact_expand_hash(edge.get("content_hash"))
    if content_hash is not None:
        compacted["content_hash"] = content_hash
    locator = expand_locator_metadata(edge.get("locator") if isinstance(edge.get("locator"), dict) else None)
    if locator:
        compacted["locator"] = locator
    return compacted


def _compact_assertion_edge_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        candidates = [value]
    elif isinstance(value, (list, tuple)):
        candidates = list(value)
    else:
        return []
    compacted: list[dict[str, Any]] = []
    for edge in candidates[:8]:
        item = _compact_assertion_edge(edge)
        if item:
            compacted.append(item)
    return compacted


def _assertion_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    if valid_memory_kind(payload.get("memory_kind")) != "assertion":
        return {}
    metadata: dict[str, Any] = {}
    for key in ("claim_text", "subject", "predicate", "object"):
        value = _compact_expand_string(payload.get(key))
        if value is not None:
            metadata[key] = value
    raw_confidence = payload.get("confidence")
    try:
        confidence = float(raw_confidence) if raw_confidence is not None else None
    except Exception:
        confidence = None
    if confidence is not None and math.isfinite(confidence):
        metadata["confidence"] = confidence
    for key in ("evidence", "derived_from"):
        edges = _compact_assertion_edge_list(payload.get(key))
        if edges:
            metadata[key] = edges
    return metadata


def inspect_point(qdrant: Any, collection_name: str, point_id: str, *, collection: str = "memory") -> dict[str, Any]:
    point = retrieve_point(qdrant, collection_name, point_id, with_payload=True, with_vector=False)
    result: dict[str, Any] = {
        "found": point is not None,
        "point_id": point_id,
        "collection": collection,
        "collection_name": collection_name,
    }
    if point is None:
        return _compact_status_response(result)
    payload = _point_payload(point)
    src = source_metadata(payload)
    temporal = _temporal_fact_metadata(payload)
    if temporal:
        src.update(temporal)
    assertion = _assertion_metadata(payload)
    if assertion:
        src.update(assertion)
    if src:
        result["source"] = src
    text = _point_text(point)
    if text:
        snippet, _ = _bounded_text(" ".join(text.split()), 240)
        result["snippet"] = snippet
    return _compact_status_response(result)


def default_registry(memory_lookup: Callable[[str], dict[str, Any] | None] | None = None, config: Mapping[str, Any] | None = None) -> SourceResolverRegistry:
    resolvers: list[SourceResolver] = [FileSourceResolver(), MemoryPointResolver(memory_lookup)]
    if config:
        from qdrant_memory.adapters import adapter_resolvers

        resolvers.extend(adapter_resolvers(config))
    return SourceResolverRegistry(resolvers)


def _lookup_for(qdrant: Any, collection_name: str) -> Callable[[str], dict[str, Any] | None]:
    def lookup(point_id: str) -> dict[str, Any] | None:
        return retrieve_point(qdrant, collection_name, point_id, with_payload=True, with_vector=False)

    return lookup


def _fallback_expansion(point: dict[str, Any], *, point_id: str, collection_name: str, mode: str, max_chars: int, reason: str = "missing_source_uri") -> dict[str, Any]:
    text, truncated = _bounded_text(_point_text(point), max_chars)
    return _compact_expand_response(
        {
            "point_id": point_id,
            "collection_name": collection_name,
            "supported": True,
            "status": "unknown",
            "reason": reason,
            "fallback": "point_text",
            "mode": mode,
            "text": text,
            "chars": len(text),
            "max_chars": _normalize_max_chars(max_chars),
            "truncated": truncated,
            "message": "Source URI is unavailable; returned bounded point text instead.",
        }
    )


def expand_point(
    qdrant: Any,
    collection_name: str,
    point_id: str,
    *,
    collection: str = "memory",
    mode: str = "excerpt",
    max_chars: int = DEFAULT_MAX_CHARS,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    point = retrieve_point(qdrant, collection_name, point_id, with_payload=True, with_vector=False)
    if point is None:
        return _compact_expand_response({"found": False, "point_id": point_id, "collection": collection, "collection_name": collection_name})
    budget = _normalize_max_chars(max_chars)
    if mode == "neighbors":
        fallback = _fallback_expansion(point, point_id=point_id, collection_name=collection_name, mode=mode, max_chars=budget, reason="neighbors_unsupported")
        fallback.update({"status": "unsupported", "supported": False, "neighbors": []})
        return fallback
    payload = _point_payload(point)
    source_uri = str(payload.get("source_uri") or "").strip()
    if not source_uri:
        return _fallback_expansion(point, point_id=point_id, collection_name=collection_name, mode=mode, max_chars=budget)
    registry = default_registry(_lookup_for(qdrant, collection_name), config)
    expansion = registry.expand(source_uri, payload.get("locator") if isinstance(payload.get("locator"), dict) else None, mode=mode, max_chars=budget)
    expansion.update({"found": True, "point_id": point_id, "collection": collection, "collection_name": collection_name})
    return _compact_expand_response(expansion)


def source_status_for_point(qdrant: Any, collection_name: str, point_id: str, *, collection: str = "memory", config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    point = retrieve_point(qdrant, collection_name, point_id, with_payload=True, with_vector=False)
    if point is None:
        return _compact_status_response({"found": False, "point_id": point_id, "collection": collection, "collection_name": collection_name})
    payload = _point_payload(point)
    source_uri = str(payload.get("source_uri") or "").strip()
    if not source_uri:
        return _compact_status_response(
            {
                "found": True,
                "point_id": point_id,
                "collection": collection,
                "collection_name": collection_name,
                "supported": False,
                "status": "unknown",
                "reason": "missing_source_uri",
                "message": "Point has no source_uri metadata.",
            }
        )
    registry = default_registry(_lookup_for(qdrant, collection_name), config)
    status = registry.stat(
        source_uri,
        payload.get("locator") if isinstance(payload.get("locator"), dict) else None,
        content_hash=str(payload.get("content_hash") or "") or None,
        source_modified_at=str(payload.get("source_modified_at") or "") or None,
    )
    status.update({"found": True, "point_id": point_id, "collection": collection, "collection_name": collection_name})
    return _compact_status_response(status)


def _trace_edge(edge: Any, registry: SourceResolverRegistry) -> dict[str, Any]:
    if not isinstance(edge, dict):
        return {"status": "unknown", "reason": "invalid_edge"}
    result: dict[str, Any] = {}
    source_uri = str(edge.get("source_uri") or "").strip()
    if not source_uri and edge.get("point_id"):
        source_uri = f"memory://point/{edge.get('point_id')}"
    if source_uri:
        result["source_uri"] = source_uri
    derivation_type = _compact_expand_token(edge.get("derivation_type"))
    if derivation_type:
        result["derivation_type"] = derivation_type
    relation_type = valid_relation_type(edge.get("relation_type"))
    if relation_type:
        result["relation_type"] = relation_type
    locator = expand_locator_metadata(edge.get("locator") if isinstance(edge.get("locator"), dict) else None)
    if locator:
        result["locator"] = locator
    if not source_uri:
        result.update({"status": "unknown", "reason": "missing_source_uri"})
        return _compact_status_response(result)
    status = registry.stat(source_uri, edge.get("locator") if isinstance(edge.get("locator"), dict) else None)
    for key in ("supported", "status", "exists", "changed", "reason", "message", "point_id", "scheme"):
        if key in status:
            result[key] = status[key]
    if source_uri.startswith("memory://") and status.get("status") == "exists" and status.get("point_id"):
        # Return a compact hint only; expansion remains a separate explicit step.
        result["expand_hint"] = source_uri
    return _compact_status_response(result)


def _trace_supersession_link(qdrant: Any, collection_name: str, point_id: str) -> dict[str, Any]:
    result: dict[str, Any] = {"point_id": _compact_expand_string(point_id) or point_id}
    point = retrieve_point(qdrant, collection_name, point_id, with_payload=True, with_vector=False)
    if point is None:
        result["status"] = "missing"
        return _compact_status_response(result)
    result["status"] = "exists"
    status = valid_fact_status(_point_payload(point).get("fact_status"))
    if status:
        result["fact_status"] = status
    return _compact_status_response(result)


def _supersession_trace(qdrant: Any, collection_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    history: dict[str, Any] = {}
    for key in ("supersedes", "superseded_by", "invalidated_by"):
        links = sanitize_point_id_links(payload.get(key))
        if links:
            history[key] = [_trace_supersession_link(qdrant, collection_name, point_id) for point_id in links]
    return history


def trace_point(qdrant: Any, collection_name: str, point_id: str, *, collection: str = "memory", direction: str = "upstream", config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    direction = str(direction or "upstream").strip().lower()
    if direction not in {"upstream", "downstream", "both"}:
        direction = "upstream"
    point = retrieve_point(qdrant, collection_name, point_id, with_payload=True, with_vector=False)
    if point is None:
        return _compact_status_response({"found": False, "point_id": point_id, "collection": collection, "collection_name": collection_name, "direction": direction})
    payload = _point_payload(point)
    registry = default_registry(_lookup_for(qdrant, collection_name), config)
    result: dict[str, Any] = {
        "found": True,
        "point_id": point_id,
        "collection": collection,
        "collection_name": collection_name,
        "direction": direction,
    }
    temporal = _temporal_fact_metadata(payload)
    for key in ("fact_status", "observed_at", "valid_from", "valid_until"):
        if key in temporal:
            result[key] = temporal[key]
    supersession = _supersession_trace(qdrant, collection_name, payload)
    if supersession:
        result["supersession"] = supersession
    if direction in {"upstream", "both"}:
        upstream = payload.get("derived_from") if isinstance(payload.get("derived_from"), list) else []
        result["upstream"] = [_trace_edge(edge, registry) for edge in upstream]
    if direction in {"downstream", "both"}:
        result["downstream"] = {
            "supported": False,
            "status": "unsupported",
            "reason": "downstream_trace_unsupported",
            "message": "Downstream trace requires an indexed reverse derivation lookup and is not enabled in this phase.",
        }
    return _compact_status_response(result)
