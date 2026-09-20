from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .fact_metadata import derive_fact_metadata
from .lineage import (
    CHUNKER_VERSION,
    apply_capture_plan,
    make_scope_key,
    make_source_key,
    make_version_identity_digest,
    plan_file_lineage,
    read_back_exact_records,
    source_logical_id,
    storage_point_id,
    version_logical_id,
)
from .schema import build_payload, score_importance

DEFAULT_EXCLUDE_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "dist",
    "build",
    "target",
    ".next",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".hg",
    ".svn",
}

BINARYISH_EXTENSIONS = {
    ".7z", ".a", ".bin", ".bmp", ".bz2", ".class", ".dll", ".dmg", ".doc",
    ".docx", ".dylib", ".exe", ".gif", ".gz", ".ico", ".jar", ".jpeg",
    ".jpg", ".lock", ".o", ".pdf", ".png", ".pyc", ".pyo", ".rar", ".so",
    ".sqlite", ".sqlite3", ".tar", ".wasm", ".webp", ".xls", ".xlsx", ".zip",
}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_DATE_HEADING_RE = re.compile(r"^(#{1,6})\s+(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b.*$")
_HR_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_TAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9_/-]+)")
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.S)


@dataclass
class FileChunk:
    id: str
    text: str
    source: str
    source_type: str
    file_path: str
    file_mtime: float
    file_size: int
    file_sha256: str
    chunk_index: int
    chunk_count: int
    chunk_hash: str
    heading: str = ""
    line_start: int | None = None
    line_end: int | None = None
    source_uri: str = ""
    tags: list[str] = field(default_factory=list)
    file_mtime_ns: int | None = None
    manifest_version: int = 1
    lineage_schema_version: int | None = None
    lineage_entity_id: str = ""
    file_version_id: str = ""
    file_version_entity_id: str = ""
    chunker_version: str = CHUNKER_VERSION
    lineage_pending: bool | None = None
    derived_from: list[dict[str, Any]] = field(default_factory=list)

    def locator(self) -> dict[str, Any]:
        locator: dict[str, Any] = {}
        if self.line_start is not None:
            locator["line_start"] = self.line_start
        if self.line_end is not None:
            locator["line_end"] = self.line_end
        if self.heading:
            locator["heading"] = self.heading
        return locator

    def payload(
        self,
        *,
        profile_id: str = "default",
        platform: str = "cli",
        session_id: str = "",
        user_id_hash: str = "",
        chat_id_hash: str = "",
        project_path: str = "",
        model: str = "",
    ) -> dict[str, Any]:
        fact_metadata = derive_fact_metadata(
            text=self.text,
            source_type=self.source_type,
            chunk_type="file_chunk",
            tags=self.tags,
            heading=self.heading,
            file_path=self.file_path,
            project_path=project_path,
        )
        locator = self.locator()
        payload = build_payload(
            text=self.text,
            source=self.source,
            source_type=self.source_type,
            chunk_type="file_chunk",
            importance=score_importance(self.text, self.source_type),
            tags=self.tags,
            profile_id=profile_id,
            platform=platform,
            user_id_hash=user_id_hash,
            chat_id_hash=chat_id_hash,
            session_id=session_id,
            project_path=project_path,
            model=model,
            memory_kind="source_chunk",
            fact_metadata=fact_metadata,
            source_uri=self.source_uri or file_uri(self.file_path),
            locator=locator,
            content_hash=f"sha256:{self.chunk_hash}",
            source_modified_at=source_modified_at_iso(self.file_mtime),
            derivation_type="indexed_chunk",
            canonical=True,
            stale=False,
            requires_review=False,
        )
        payload.update(
            {
                "file_path": self.file_path,
                "file_mtime": self.file_mtime,
                "file_size": self.file_size,
                "file_sha256": self.file_sha256,
                "manifest_version": self.manifest_version,
                "chunk_id": self.id,
                "chunk_hash": self.chunk_hash,
                "chunk_index": self.chunk_index,
                "chunk_count": self.chunk_count,
                "heading": self.heading,
            }
        )
        if self.file_mtime_ns is not None:
            payload["file_mtime_ns"] = self.file_mtime_ns
        if self.lineage_schema_version is not None:
            payload["lineage_schema_version"] = self.lineage_schema_version
        for key, value in (
            ("lineage_entity_id", self.lineage_entity_id),
            ("file_version_id", self.file_version_id),
            ("file_version_entity_id", self.file_version_entity_id),
            ("chunker_version", self.chunker_version if self.file_version_id else ""),
        ):
            if value:
                payload[key] = value
        if self.lineage_pending is not None:
            payload["lineage_pending"] = self.lineage_pending
        if self.derived_from:
            payload["derived_from"] = list(self.derived_from)
        return payload


def expand_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def normalize_extensions(values: Iterable[str]) -> set[str]:
    out = set()
    for value in values:
        ext = str(value).strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        out.add(ext)
    return out or {".md", ".txt"}


def make_file_chunk_id(file_path: str, chunk_index: int) -> str:
    digest = hashlib.sha256(f"indexed-file\n{file_path}\n{chunk_index}".encode("utf-8")).hexdigest()
    return str(uuid.UUID(digest[:32]))


def sha256_hex(value: bytes | str) -> str:
    data = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(data).hexdigest()


def source_modified_at_iso(mtime: float) -> str:
    return datetime.fromtimestamp(float(mtime), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def file_uri(file_path: str) -> str:
    try:
        return Path(file_path).resolve().as_uri()
    except Exception:
        return ""


def _locate_text_span_whitespace_tolerant(source_text: str, clean: str, *, start_at: int = 0) -> tuple[int, int] | None:
    pattern = "".join(
        r"\s+" if part.isspace() else re.escape(part)
        for part in re.split(r"(\s+)", clean)
        if part
    )
    if not pattern:
        return None
    try:
        compiled = re.compile(pattern)
    except re.error:
        return None
    match = compiled.search(source_text, max(0, start_at)) or compiled.search(source_text)
    if not match:
        return None
    return match.start(), match.end()


def locate_text_lines(source_text: str, chunk_text_value: str, *, start_at: int = 0) -> tuple[int | None, int | None, int]:
    clean = (chunk_text_value or "").strip()
    if not clean:
        return None, None, start_at
    start = source_text.find(clean, max(0, start_at))
    if start >= 0:
        end = start + len(clean)
    else:
        start = source_text.find(clean)
        if start >= 0:
            end = start + len(clean)
        else:
            span = _locate_text_span_whitespace_tolerant(source_text, clean, start_at=start_at)
            if not span:
                return None, None, start_at
            start, end = span
    line_start = source_text.count("\n", 0, start) + 1
    line_end = line_start + source_text.count("\n", start, end)
    return line_start, line_end, end


def normalize_newlines(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def file_path_filter(
    file_path: str,
    *,
    profile_id: str | None = None,
    user_id_hash: str = "",
    chat_id_hash: str = "",
) -> dict[str, Any]:
    must = [
        {"key": "file_path", "match": {"value": file_path}},
        {"key": "chunk_type", "match": {"value": "file_chunk"}},
    ]
    if profile_id is not None:
        must.append({"key": "profile_id", "match": {"value": profile_id}})
    if user_id_hash:
        must.append({"key": "user_id_hash", "match": {"value": user_id_hash}})
    if chat_id_hash:
        must.append({"key": "chat_id_hash", "match": {"value": chat_id_hash}})
    return {"must": must}


def file_chunk_filter(
    *, profile_id: str | None = None, user_id_hash: str = "", chat_id_hash: str = ""
) -> dict[str, Any]:
    must = [{"key": "chunk_type", "match": {"value": "file_chunk"}}]
    if profile_id is not None:
        must.append({"key": "profile_id", "match": {"value": profile_id}})
    if user_id_hash:
        must.append({"key": "user_id_hash", "match": {"value": user_id_hash}})
    if chat_id_hash:
        must.append({"key": "chat_id_hash", "match": {"value": chat_id_hash}})
    return {"must": must}


def _owned_file_chunks(
    points: Iterable[dict[str, Any]], *, profile_id: str, user_id_hash: str, chat_id_hash: str
) -> tuple[list[dict[str, Any]], list[str]]:
    owned, errors = [], []
    for point in points:
        payload = point.get("payload") if isinstance(point, dict) else None
        if not isinstance(payload, dict) or not payload.get("profile_id"):
            errors.append("profile_id")
            continue
        if payload.get("profile_id") != profile_id:
            continue
        if (str(payload.get("user_id_hash") or ""), str(payload.get("chat_id_hash") or "")) != (
            user_id_hash, chat_id_hash
        ):
            continue
        owned.append(point)
    return owned, errors


def is_path_within(path: str, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def extract_tags(text: str) -> list[str]:
    tags: set[str] = set(_TAG_RE.findall(text or ""))
    match = _FRONTMATTER_RE.match(text or "")
    if match:
        frontmatter = match.group(1)
        in_tags_list = False
        for raw in frontmatter.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("tags:"):
                in_tags_list = True
                rest = line.split(":", 1)[1].strip()
                if rest.startswith("[") and rest.endswith("]"):
                    for item in rest.strip("[]").split(","):
                        item = item.strip().strip("'\"")
                        if item:
                            tags.add(item.lstrip("#"))
                elif rest:
                    tags.add(rest.strip("'\"").lstrip("#"))
                continue
            if in_tags_list and line.startswith("-"):
                item = line[1:].strip().strip("'\"")
                if item:
                    tags.add(item.lstrip("#"))
            elif not raw.startswith((" ", "\t", "-")):
                in_tags_list = False
    return sorted(t for t in tags if t)


def strip_frontmatter(text: str) -> str:
    return _FRONTMATTER_RE.sub("", text or "", count=1).strip()


def classify_source_type(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    path_s = str(path).lower()
    if "skills" in parts or "/.hermes/skills/" in path_s:
        return "skill_doc"
    if any(p in parts for p in ("docs", "plans", "project", "projects")):
        return "project_doc"
    return "indexed_file"


def _split_oversized(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    current = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                parts.append(current.strip())
                current = ""
            for i in range(0, len(para), max_chars):
                chunk = para[i : i + max_chars].strip()
                if chunk:
                    parts.append(chunk)
            continue
        candidate = para if not current else current + "\n\n" + para
        if len(candidate) > max_chars and current:
            parts.append(current.strip())
            current = para
        else:
            current = candidate
    if current.strip():
        parts.append(current.strip())
    return parts


def _is_heading_only_chunk(lines: list[str]) -> bool:
    meaningful = [line.strip() for line in lines if line.strip()]
    return bool(meaningful) and all(_HEADING_RE.match(line) for line in meaningful)


def chunk_markdown(text: str, *, max_chars: int) -> list[tuple[str, str]]:
    text = strip_frontmatter(text)
    if not text:
        return []
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if body and not _is_heading_only_chunk(current_lines):
            sections.append((current_heading, current_lines[:]))
        current_lines = []

    for line in text.splitlines():
        heading_match = _HEADING_RE.match(line)
        date_heading_match = _DATE_HEADING_RE.match(line)
        if heading_match or date_heading_match:
            flush()
            current_heading = (heading_match.group(2) if heading_match else line.lstrip("# ")).strip()
            current_lines = [line]
            continue
        if _HR_RE.match(line) and current_lines and len("\n".join(current_lines)) >= max_chars // 2:
            flush()
            current_heading = current_heading
            continue
        current_lines.append(line)
    flush()

    if not sections:
        fallback_lines = text.splitlines()
        if _is_heading_only_chunk(fallback_lines):
            return []
        sections = [("", fallback_lines)]

    chunks: list[tuple[str, str]] = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        for piece in _split_oversized(body, max_chars):
            chunks.append((heading, piece))
    return chunks


def chunk_text(text: str, *, max_chars: int) -> list[tuple[str, str]]:
    text = (text or "").strip()
    if not text:
        return []
    return [("", piece) for piece in _split_oversized(text, max_chars)]


class FileIndexer:
    def __init__(
        self,
        *,
        qdrant=None,
        embeddings=None,
        collection_name: str = "hermes_memory",
        config: dict[str, Any] | None = None,
        profile_id: str = "default",
        platform: str = "cli",
        session_id: str = "",
        user_id_hash: str = "",
        chat_id_hash: str = "",
        project_path: str = "",
        model: str = "",
    ):
        self.qdrant = qdrant
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.config = config or {}
        self.profile_id = profile_id
        self.platform = platform
        self.session_id = session_id
        self.user_id_hash = user_id_hash
        self.chat_id_hash = chat_id_hash
        self.project_path = project_path
        self.model = model

    @property
    def max_chars(self) -> int:
        try:
            tokens = int(self.config.get("max_chunk_tokens", 512))
        except Exception:
            tokens = 512
        return max(400, tokens * 4)

    @property
    def extensions(self) -> set[str]:
        return normalize_extensions(self.config.get("index_extensions", [".md", ".txt"]))

    @property
    def exclude_dirs(self) -> set[str]:
        configured = self.config.get("index_exclude_dirs", list(DEFAULT_EXCLUDE_DIRS))
        return {str(x) for x in configured} | DEFAULT_EXCLUDE_DIRS

    def should_skip_file(self, path: Path) -> bool:
        if path.suffix.lower() in BINARYISH_EXTENSIONS:
            return True
        if path.suffix.lower() not in self.extensions:
            return True
        if any(part in self.exclude_dirs for part in path.parts):
            return True
        try:
            with path.open("rb") as handle:
                sample = handle.read(2048)
            if b"\x00" in sample:
                return True
        except Exception:
            return True
        return False

    def iter_files(self, paths: Iterable[str | Path], *, max_files: int | None = None) -> tuple[list[Path], list[dict[str, str]]]:
        files: list[Path] = []
        skipped: list[dict[str, str]] = []
        limit = int(max_files or self.config.get("index_max_files", 500) or 500)
        for raw in paths:
            root = expand_path(str(raw))
            if not root.exists():
                skipped.append({"path": str(root), "reason": "missing"})
                continue
            candidates = [root] if root.is_file() else []
            if root.is_dir():
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = [d for d in dirnames if d not in self.exclude_dirs and d.lower() not in self.exclude_dirs]
                    for filename in sorted(filenames):
                        candidates.append(Path(dirpath) / filename)
                        if len(files) + len(skipped) >= limit * 10:
                            break
                    if len(files) >= limit:
                        break
            for candidate in candidates:
                if len(files) >= limit:
                    skipped.append({"path": str(candidate), "reason": "max_files"})
                    continue
                if self.should_skip_file(candidate):
                    skipped.append({"path": str(candidate), "reason": "excluded"})
                    continue
                files.append(candidate.resolve())
        return files, skipped

    def _prepare_snapshot(self, path: Path) -> tuple[dict[str, Any], list[FileChunk]]:
        path = path.resolve()
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        identity_before = (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino, before.st_dev)
        identity_after = (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino, after.st_dev)
        if identity_before != identity_after or len(raw) != after.st_size:
            raise RuntimeError("file changed during snapshot read")
        text = normalize_newlines(raw.decode("utf-8", errors="replace"))
        tags, max_chars = extract_tags(text), self.max_chars
        raw_chunks = chunk_markdown(text, max_chars=max_chars) if path.suffix.lower() == ".md" else chunk_text(text, max_chars=max_chars)
        source, source_uri = str(path), path.as_uri()
        file_sha256 = sha256_hex(raw)
        manifest = {
            "file_path": source,
            "file_size": after.st_size,
            "file_mtime": after.st_mtime,
            "file_mtime_ns": after.st_mtime_ns,
            "file_mtime_iso": source_modified_at_iso(after.st_mtime),
            "file_sha256": file_sha256,
            "source_uri": source_uri,
        }
        chunks: list[FileChunk] = []
        search_offset = 0
        for index, (heading, chunk_text_value) in enumerate(raw_chunks):
            clean = chunk_text_value.strip()
            if not clean:
                continue
            line_start, line_end, search_offset = locate_text_lines(text, clean, start_at=search_offset)
            chunks.append(FileChunk(
                id=make_file_chunk_id(source, index), text=clean, source=source,
                source_type=classify_source_type(path), file_path=source,
                file_mtime=after.st_mtime, file_size=after.st_size,
                file_sha256=file_sha256, chunk_index=index,
                chunk_count=len(raw_chunks), chunk_hash=sha256_hex(clean),
                heading=heading, line_start=line_start, line_end=line_end,
                source_uri=source_uri, tags=tags, file_mtime_ns=after.st_mtime_ns,
                chunker_version=f"{CHUNKER_VERSION}:{max_chars}",
            ))
        return manifest, chunks

    def prepare_file(self, path: Path) -> list[FileChunk]:
        return self._prepare_snapshot(path)[1]

    def prepare(self, paths: Iterable[str | Path], *, max_files: int | None = None) -> dict[str, Any]:
        files, skipped = self.iter_files(paths, max_files=max_files)
        errors: list[dict[str, str]] = []
        chunks: list[FileChunk] = []
        indexed_files = 0
        file_manifests: list[dict[str, Any]] = []
        for path in files:
            try:
                manifest, file_chunks = self._prepare_snapshot(path)
                file_manifests.append(manifest)
                if file_chunks:
                    indexed_files += 1
                    chunks.extend(file_chunks)
                else:
                    skipped.append({"path": str(path), "reason": "empty"})
            except Exception as exc:
                errors.append({"path": str(path), "error": str(exc)})
        return {
            "files": files, "file_manifests": file_manifests,
            "files_seen": len(files) + len(skipped), "files_indexed": indexed_files,
            "files_skipped": len(skipped), "skipped": skipped[:50],
            "chunks": chunks, "chunks_prepared": len(chunks), "errors": errors[:20],
            "max_files_truncated": any(item.get("reason") == "max_files" for item in skipped),
        }

    def index(self, paths: Iterable[str | Path], *, dry_run: bool = True, force: bool = False, max_files: int | None = None) -> dict[str, Any]:
        input_paths = list(paths)
        prepared = self.prepare(input_paths, max_files=max_files)
        chunks: list[FileChunk] = prepared.pop("chunks")
        manifests: list[dict[str, Any]] = prepared.get("file_manifests", [])
        chunks_by_file = {str(item["file_path"]): [] for item in manifests}
        for chunk in chunks:
            chunks_by_file.setdefault(chunk.file_path, []).append(chunk)
        lineage_mode = str(self.config.get("lineage_mode") or "off")
        capture = lineage_mode == "capture"
        summary: dict[str, Any] = {
            "dry_run": bool(dry_run), "files_seen": prepared["files_seen"],
            "files_indexed": prepared["files_indexed"], "files_skipped": prepared["files_skipped"],
            "chunks_prepared": len(chunks), "chunks_upserted": 0, "chunks_deleted": 0,
            "stale_ids": [], "stale_count": 0, "files_with_stale_chunks": 0,
            "manifest_checked": False, "directory_manifest_checked": False,
            "directory_roots_checked": [], "deleted_file_paths": [], "deleted_file_ids": [],
            "delete_mode": "none", "errors": list(prepared.get("errors", []))[:20],
            "partial_failure": False,
            "paths": [str(p) for p in input_paths], "force": bool(force),
            "lineage_mode": lineage_mode, "lineage_files": [], "lineage_entity_ids": [],
            "lineage_edge_ids": [], "lineage_point_ids": [], "lineage_existing_ids": [],
            "lineage_repair_ids": [], "lineage_blocked_files": [],
            "lineage_acknowledged_ids": [], "lineage_read_back_ids": [],
            "refusals": [], "foreign_scope_chunks": [],
            "lineage_coverage": {"selected_files": len(manifests), "selected_chunks": len(chunks),
                                 "eligible_files": 0, "eligible_chunks": 0,
                                 "captured_files": 0, "captured_chunks": 0,
                                 "legacy_blocked_files": 0, "legacy_blocked_chunks": 0},
            "filter_delete_paths": [], "filter_delete_candidates": 0,
            "filter_delete_receipts": [], "filter_deletes_issued": 0,
        }
        if prepared.get("errors"):
            summary["partial_failure"] = True
        foreign_ids_by_file: dict[str, set[str]] = {}
        requested_paths = [expand_path(str(path)) for path in input_paths]
        requested_directory_roots = [path for path in requested_paths if path.is_dir()]
        requested_file_paths = {str(path) for path in requested_paths if not path.is_dir()}
        plans: list[dict[str, Any]] = []
        desired_ids_by_file: dict[str, set[str]] = {path: set() for path in chunks_by_file}
        existing_ids_by_file: dict[str, list[str] | None] = {path: None for path in chunks_by_file}
        off_blocked_files: set[str] = set()
        inventory_unavailable_files: set[str] = set()

        def finish() -> dict[str, Any]:
            if summary.get("refused"):
                summary["partial_failure"] = True
            foreign: list[dict[str, Any]] = []
            for path, ids in sorted(foreign_ids_by_file.items()):
                overwritten = desired_ids_by_file.get(path, set()) if not capture and path not in off_blocked_files else set()
                leftover_ids = ids - overwritten
                if (leftover_ids and path not in inventory_unavailable_files
                        and (path in requested_file_paths
                             or any(is_path_within(path, root) for root in requested_directory_roots))):
                    foreign.append({"file_path": path, "count": len(leftover_ids)})
            summary["foreign_scope_chunks"] = foreign
            return summary
        stale_ids: list[str] = []
        manifest_capable = bool(self.qdrant and callable(getattr(self.qdrant, "scroll_by_filter", None)))
        manifest_by_path = {str(item["file_path"]): item for item in manifests}
        if manifest_capable:
            summary["manifest_checked"] = True
            for file_path in sorted(chunks_by_file):
                try:
                    existing = self.qdrant.scroll_by_filter(
                        self.collection_name,
                        file_path_filter(file_path, profile_id=None),
                        limit=256, with_payload=True, with_vector=False,
                    )
                    owned, ownership_errors = _owned_file_chunks(
                        existing, profile_id=self.profile_id,
                        user_id_hash=self.user_id_hash, chat_id_hash=self.chat_id_hash,
                    )
                    owned_ids = {str(point["id"]) for point in owned if point.get("id") is not None}
                    existing_ids_by_file[file_path] = sorted(owned_ids)
                    foreign_ids_by_file.setdefault(file_path, set()).update(
                        str(point["id"]) for point in existing
                        if point.get("id") is not None and str(point["id"]) not in owned_ids
                    )
                    if capture:
                        scope_key = make_scope_key(
                            collection_name=self.collection_name, profile_id=self.profile_id,
                            user_id_hash=self.user_id_hash, chat_id_hash=self.chat_id_hash)
                        source_key = make_source_key(scope_key=scope_key, resolved_file_path=file_path)
                        source_id = storage_point_id(source_logical_id(source_key=source_key, profile_id=self.profile_id))
                        version_digest = make_version_identity_digest(
                            source_key=source_key,
                            file_sha256=str(manifest_by_path[file_path]["file_sha256"]),
                        )
                        version_id = storage_point_id(version_logical_id(
                            version_identity_digest=version_digest, profile_id=self.profile_id))
                        graph_records = read_back_exact_records(
                            self.qdrant, self.collection_name, [source_id, version_id])
                        plan = plan_file_lineage(
                            collection_name=self.collection_name, profile_id=self.profile_id,
                            user_id_hash=self.user_id_hash, chat_id_hash=self.chat_id_hash,
                            manifest=manifest_by_path[file_path], chunks=chunks_by_file[file_path],
                            existing_points=owned, existing_source=graph_records.get(source_id),
                            existing_version=graph_records.get(version_id),
                            ownership_errors=ownership_errors,
                        )
                        plans.append(plan)
                        summary["lineage_files"].append({
                            key: plan[key] for key in ("file_path", "lineage_baseline_basis",
                                                       "lineage_missing_fields", "lineage_blocked_reason",
                                                       "lineage_history_complete")
                        })
                        for key in ("lineage_entity_ids", "lineage_edge_ids", "lineage_point_ids", "lineage_existing_ids"):
                            summary[key].extend(plan[key])
                        if plan["capturable"]:
                            summary["lineage_coverage"]["eligible_files"] += 1
                            summary["lineage_coverage"]["eligible_chunks"] += len(chunks_by_file[file_path])
                            graph_existing = read_back_exact_records(self.qdrant, self.collection_name, plan["lineage_point_ids"])
                            plan["lineage_repair_ids"] = (
                                [] if plan["is_new"]
                                else sorted(set(plan["lineage_point_ids"]) - set(graph_existing))
                            )
                            if dry_run:
                                summary["lineage_repair_ids"].extend(plan["lineage_repair_ids"])
                        else:
                            blocked = summary["lineage_files"][-1]
                            summary["lineage_blocked_files"].append(blocked)
                            summary["refused"] = True
                            summary["refusals"].append({
                                "file_path": file_path,
                                "reason": str(plan["lineage_blocked_reason"]),
                            })
                            if plan["lineage_baseline_basis"] == "missing_indexed_file_sha256":
                                summary["lineage_coverage"]["legacy_blocked_files"] += 1
                                summary["lineage_coverage"]["legacy_blocked_chunks"] += len(owned)
                    desired = {chunk.id for chunk in chunks_by_file[file_path]}
                    desired_ids_by_file[file_path] = desired
                    if not capture:
                        lineage_managed = any(
                            any((point.get("payload") or {}).get(key) not in (None, "")
                                for key in ("file_version_id", "lineage_entity_id", "lineage_schema_version"))
                            for point in existing
                        )
                        if lineage_managed:
                            blocked = {
                                "file_path": file_path,
                                "lineage_baseline_basis": "indexed_file_sha256",
                                "lineage_missing_fields": [],
                                "lineage_blocked_reason": "lineage_managed_requires_capture_or_retirement",
                                "lineage_history_complete": False,
                            }
                            off_blocked_files.add(file_path)
                            summary["lineage_files"].append(blocked)
                            summary["lineage_blocked_files"].append(blocked)
                            summary["refused"] = True
                            summary["refusals"].append({
                                "file_path": file_path,
                                "reason": "lineage_managed_requires_capture_or_retirement",
                            })
                        else:
                            file_stale = [str(point["id"]) for point in owned
                                          if point.get("id") is not None and str(point["id"]) not in desired]
                            if file_stale:
                                summary["files_with_stale_chunks"] += 1
                                stale_ids.extend(file_stale)
                except Exception as exc:
                    summary["errors"].append({"file_path": file_path, "error": f"manifest sync failed: {exc}"})
                    summary["partial_failure"] = True
                    if not capture:
                        reason = "lineage_inventory_unavailable_refused_fail_closed"
                        blocked = {
                            "file_path": file_path,
                            "lineage_baseline_basis": "inventory_unavailable",
                            "lineage_missing_fields": [],
                            "lineage_blocked_reason": reason,
                            "lineage_history_complete": False,
                        }
                        off_blocked_files.add(file_path)
                        inventory_unavailable_files.add(file_path)
                        foreign_ids_by_file.pop(file_path, None)
                        summary["lineage_files"].append(blocked)
                        summary["lineage_blocked_files"].append(blocked)
                        summary["refused"] = True
                        summary["refusals"].append({"file_path": file_path, "reason": reason})

            roots = requested_directory_roots
            if roots and not prepared.get("max_files_truncated"):
                summary["directory_manifest_checked"] = True
                summary["directory_roots_checked"] = [str(root) for root in roots]
                try:
                    all_existing_chunks = self.qdrant.scroll_by_filter(
                        self.collection_name, file_chunk_filter(profile_id=None),
                        limit=256, with_payload=True, with_vector=False,
                    )
                    existing_chunks, _ = _owned_file_chunks(
                        all_existing_chunks, profile_id=self.profile_id,
                        user_id_hash=self.user_id_hash, chat_id_hash=self.chat_id_hash,
                    )
                    owned_directory_ids = {
                        str(point["id"]) for point in existing_chunks if point.get("id") is not None
                    }
                    for point in all_existing_chunks:
                        payload, point_id = point.get("payload") or {}, point.get("id")
                        path = str(payload.get("file_path") or "")
                        if (path and point_id is not None
                                and str(point_id) not in owned_directory_ids
                                and any(is_path_within(path, root) for root in roots)):
                            foreign_ids_by_file.setdefault(path, set()).add(str(point_id))
                    deleted: dict[str, list[str]] = {}
                    for point in existing_chunks:
                        payload = point.get("payload") or {}
                        path, point_id = str(payload.get("file_path") or ""), point.get("id")
                        if not path or point_id is None or path in desired_ids_by_file or Path(path).exists():
                            continue
                        if any(is_path_within(path, root) for root in roots):
                            if not capture and any(payload.get(key) not in (None, "") for key in (
                                "file_version_id", "lineage_entity_id", "lineage_schema_version"
                            )):
                                if path not in off_blocked_files:
                                    blocked = {
                                        "file_path": path,
                                        "lineage_baseline_basis": "indexed_file_sha256",
                                        "lineage_missing_fields": [],
                                        "lineage_blocked_reason": "lineage_managed_requires_capture_or_retirement",
                                        "lineage_history_complete": False,
                                    }
                                    off_blocked_files.add(path)
                                    summary["lineage_files"].append(blocked)
                                    summary["lineage_blocked_files"].append(blocked)
                                    summary["refused"] = True
                                    summary["refusals"].append({
                                        "file_path": path,
                                        "reason": "lineage_managed_requires_capture_or_retirement",
                                    })
                                continue
                            deleted.setdefault(path, []).append(str(point_id))
                    for path, ids in sorted(deleted.items()):
                        summary["deleted_file_paths"].append(path)
                        summary["deleted_file_ids"].extend(ids)
                        if capture:
                            blocked = {"file_path": path, "lineage_baseline_basis": "indexed_file_sha256",
                                       "lineage_missing_fields": [],
                                       "lineage_blocked_reason": "retirement_requires_reconcile",
                                       "lineage_history_complete": False}
                            summary["lineage_files"].append(blocked)
                            summary["lineage_blocked_files"].append(blocked)
                            summary["lineage_coverage"]["selected_files"] += 1
                            summary["lineage_coverage"]["selected_chunks"] += len(ids)
                            summary["refused"] = True
                            summary["refusals"].append({
                                "file_path": path,
                                "reason": "retirement_requires_reconcile",
                            })
                        else:
                            stale_ids.extend(ids)
                    summary["files_with_stale_chunks"] += len(deleted)
                except Exception as exc:
                    summary["errors"].append({"error": f"directory manifest sync failed: {exc}"})
                    summary["partial_failure"] = True

        if capture and not manifest_capable:
            summary["refused"] = True
            summary["refusals"].append({"reason": "lineage_capture_requires_exact_scroll_support"})
        if not capture and force and manifest_capable:
            for file_path in sorted(set(desired_ids_by_file) - off_blocked_files):
                stale_ids.extend(existing_ids_by_file[file_path] or [])
        stale_ids = list(dict.fromkeys(stale_ids))
        summary["stale_ids"], summary["stale_count"] = stale_ids, len(stale_ids)
        force_filter_paths: list[str] = []
        if stale_ids:
            summary["delete_mode"] = "ids"
        elif (not capture and force and not manifest_capable and desired_ids_by_file
              and hasattr(self.qdrant, "delete_filter")):
            candidates = sorted(set(desired_ids_by_file) - off_blocked_files)
            if candidates and (not self.user_id_hash or not self.chat_id_hash):
                summary["refused"] = True
                summary["refusals"].append({"reason": "force_filter_delete_requires_exact_scope"})
            elif candidates:
                force_filter_paths = candidates
                summary["delete_mode"] = "filter"
                summary["filter_delete_paths"] = force_filter_paths
                summary["filter_delete_candidates"] = None
        for key in ("lineage_entity_ids", "lineage_edge_ids", "lineage_point_ids",
                    "lineage_existing_ids", "lineage_repair_ids"):
            summary[key] = sorted(set(summary[key]))
        if dry_run:
            summary["errors"] = summary["errors"][:20]
            return finish()
        if not self.qdrant or (not capture and chunks and not self.embeddings):
            summary["errors"].append({"error": "qdrant and embeddings are required when dry_run is false"})
            summary["partial_failure"] = True
            return finish()

        if capture:
            for plan in plans:
                if not plan["capturable"]:
                    continue
                plan_chunks = chunks_by_file[plan["file_path"]]
                points: list[dict[str, Any]] = []
                try:
                    for chunk in plan_chunks:
                        if chunk.id not in plan["old_ids"] and not self.embeddings:
                            raise RuntimeError("embeddings are required for new lineage chunks")
                        payload = chunk.payload(
                            profile_id=self.profile_id, platform=self.platform,
                            session_id=self.session_id, user_id_hash=self.user_id_hash,
                            chat_id_hash=self.chat_id_hash, project_path=self.project_path,
                            model=self.model,
                        )
                        point = {"id": chunk.id, "payload": payload}
                        if chunk.id not in plan["old_ids"]:
                            point["vector"] = self.embeddings.embed_document(chunk.text)
                        points.append(point)
                    applied = apply_capture_plan(
                        qdrant=self.qdrant, collection_name=self.collection_name,
                        plan=plan, chunk_points=points,
                        read_owned_chunks=lambda file_path=plan["file_path"]: _owned_file_chunks(
                            self.qdrant.scroll_by_filter(
                                self.collection_name,
                                file_path_filter(file_path, profile_id=None),
                                limit=256, with_payload=True, with_vector=False,
                            ),
                            profile_id=self.profile_id,
                            user_id_hash=self.user_id_hash, chat_id_hash=self.chat_id_hash,
                        )[0],
                        lock_timeout=float(self.config.get("lineage_lock_timeout_seconds", 5.0)),
                        lock_dir=str(self.config.get("lineage_lock_dir") or ""),
                    )
                    summary["lineage_acknowledged_ids"].extend(applied["acknowledged_ids"])
                    summary["lineage_read_back_ids"].extend(applied["read_back_ids"])
                    summary["lineage_repair_ids"] = sorted(set(summary["lineage_repair_ids"]) | set(applied["repair_ids"]))
                    summary["lineage_coverage"]["captured_files"] += 1
                    summary["lineage_coverage"]["captured_chunks"] += len(plan_chunks)
                    summary["chunks_upserted"] += sum(chunk.id not in plan["old_ids"] for chunk in plan_chunks)
                except Exception as exc:
                    summary["errors"].append({"file_path": plan["file_path"], "error": f"lineage capture failed: {exc}"})
                    summary["partial_failure"] = True
            summary["lineage_acknowledged_ids"] = sorted(set(summary["lineage_acknowledged_ids"]))
            summary["lineage_read_back_ids"] = sorted(set(summary["lineage_read_back_ids"]))
            summary["errors"] = summary["errors"][:20]
            return finish()

        points: list[dict[str, Any]] = []
        for chunk in chunks:
            if chunk.file_path in off_blocked_files:
                continue
            try:
                payload = chunk.payload(profile_id=self.profile_id, platform=self.platform,
                                        session_id=self.session_id, user_id_hash=self.user_id_hash,
                                        chat_id_hash=self.chat_id_hash, project_path=self.project_path,
                                        model=self.model)
                points.append({"id": chunk.id, "vector": self.embeddings.embed_document(chunk.text), "payload": payload})
            except Exception as exc:
                summary["errors"].append({"id": chunk.id, "error": str(exc)})
                summary["partial_failure"] = True
        if stale_ids and hasattr(self.qdrant, "delete_ids"):
            try:
                self.qdrant.delete_ids(self.collection_name, stale_ids)
                summary["chunks_deleted"], summary["delete_mode"] = len(stale_ids), "ids"
            except Exception as exc:
                summary["errors"].append({"error": f"delete stale ids failed: {exc}"})
                summary["partial_failure"] = True
        elif force_filter_paths:
            for file_path in force_filter_paths:
                try:
                    self.qdrant.delete_filter(self.collection_name, file_path_filter(
                        file_path, profile_id=self.profile_id, user_id_hash=self.user_id_hash,
                        chat_id_hash=self.chat_id_hash))
                    observed = existing_ids_by_file[file_path]
                    summary["filter_deletes_issued"] += 1
                    summary["filter_delete_receipts"].append({
                        "file_path": file_path,
                        "status": "issued",
                        "observed_matches": None if observed is None else len(observed),
                    })
                    summary["errors"].append({
                        "file_path": file_path,
                        "error": "filter delete issued; exact deletion count unavailable",
                    })
                    summary["partial_failure"] = True
                except Exception as exc:
                    summary["errors"].append({"file_path": file_path, "error": f"delete stale chunks failed: {exc}"})
                    summary["partial_failure"] = True
            if summary["filter_deletes_issued"]:
                summary["chunks_deleted"] = None
        for start in range(0, len(points), 64):
            batch = points[start:start + 64]
            if batch:
                try:
                    self.qdrant.upsert(self.collection_name, batch)
                    summary["chunks_upserted"] += len(batch)
                except Exception as exc:
                    summary["errors"].append({"error": f"upsert failed: {exc}"})
                    summary["partial_failure"] = True
        summary["errors"] = summary["errors"][:20]
        return finish()
