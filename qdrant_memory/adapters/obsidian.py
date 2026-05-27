from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from qdrant_memory.sources import DEFAULT_MAX_CHARS, FileSourceResolver


class ObsidianSourceResolver:
    schemes = {"obsidian"}

    def __init__(self, *, vault_root: str):
        self.vault_root = Path(vault_root).expanduser().resolve()
        self.file_resolver = FileSourceResolver()

    def _error(self, source_uri: str, reason: str, message: str) -> dict[str, Any]:
        return {
            "source_uri": source_uri,
            "scheme": "obsidian",
            "supported": False,
            "status": "unsupported",
            "exists": False,
            "changed": None,
            "reason": reason,
            "message": message,
        }

    def _has_dot_segments(self, value: str) -> bool:
        return any(part in {".", ".."} for part in value.replace("\\", "/").split("/"))

    def _file_uri(self, source_uri: str) -> tuple[str | None, dict[str, Any] | None]:
        try:
            parts = urlsplit(str(source_uri or ""))
        except Exception:
            return None, self._error(source_uri, "malformed_obsidian_uri", "Malformed obsidian:// source URI")
        if parts.scheme.lower() != "obsidian":
            return None, self._error(source_uri, "unsupported_scheme", "Expected an obsidian:// source URI")
        raw_rel = "/".join(part.strip("/") for part in (parts.netloc, parts.path) if part.strip("/"))
        decoded_rel = unquote(raw_rel)
        if not decoded_rel:
            return None, self._error(source_uri, "missing_obsidian_note", "obsidian:// source URI has no note path")
        if decoded_rel.startswith("/") or self._has_dot_segments(raw_rel) or self._has_dot_segments(decoded_rel):
            return None, self._error(source_uri, "unsafe_obsidian_uri", "obsidian:// source URI cannot escape the configured vault root")
        candidate = (self.vault_root / decoded_rel).resolve()
        try:
            candidate.relative_to(self.vault_root)
        except ValueError:
            return None, self._error(source_uri, "unsafe_obsidian_uri", "obsidian:// source URI cannot escape the configured vault root")
        return candidate.as_uri(), None

    def _rewrite(self, result: dict[str, Any], source_uri: str) -> dict[str, Any]:
        rewritten = dict(result)
        rewritten["source_uri"] = source_uri
        rewritten["scheme"] = "obsidian"
        return rewritten

    def stat(
        self,
        source_uri: str,
        locator: dict[str, Any] | None = None,
        *,
        content_hash: str | None = None,
        source_modified_at: str | None = None,
    ) -> dict[str, Any]:
        file_uri, error = self._file_uri(source_uri)
        if error:
            return error
        assert file_uri is not None
        return self._rewrite(self.file_resolver.stat(file_uri, locator, content_hash=content_hash, source_modified_at=source_modified_at), source_uri)

    def expand(
        self,
        source_uri: str,
        locator: dict[str, Any] | None = None,
        *,
        mode: str = "excerpt",
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> dict[str, Any]:
        file_uri, error = self._file_uri(source_uri)
        if error:
            return {**error, "mode": mode, "text": "", "chars": 0, "max_chars": max_chars, "truncated": False}
        assert file_uri is not None
        return self._rewrite(self.file_resolver.expand(file_uri, locator, mode=mode, max_chars=max_chars), source_uri)
