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
        locator: dict[str, Any] = {}
        if self.line_start is not None:
            locator["line_start"] = self.line_start
        if self.line_end is not None:
            locator["line_end"] = self.line_end
        if self.heading:
            locator["heading"] = self.heading
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
                "manifest_version": 1,
                "chunk_id": self.id,
                "chunk_hash": self.chunk_hash,
                "chunk_index": self.chunk_index,
                "chunk_count": self.chunk_count,
                "heading": self.heading,
            }
        )
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


def file_path_filter(file_path: str) -> dict[str, Any]:
    return {"must": [{"key": "file_path", "match": {"value": file_path}}]}


def file_chunk_filter() -> dict[str, Any]:
    return {"must": [{"key": "chunk_type", "match": {"value": "file_chunk"}}]}


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

    def prepare_file(self, path: Path) -> list[FileChunk]:
        path = path.resolve()
        raw = path.read_bytes()
        text = normalize_newlines(raw.decode("utf-8", errors="replace"))
        tags = extract_tags(text)
        max_chars = self.max_chars
        if path.suffix.lower() == ".md":
            raw_chunks = chunk_markdown(text, max_chars=max_chars)
        else:
            raw_chunks = chunk_text(text, max_chars=max_chars)
        source = str(path)
        source_type = classify_source_type(path)
        stat = path.stat()
        mtime = stat.st_mtime
        file_size = stat.st_size
        file_sha256 = sha256_hex(raw)
        count = len(raw_chunks)
        chunks: list[FileChunk] = []
        search_offset = 0
        source_uri = path.as_uri()
        for index, (heading, chunk_text_value) in enumerate(raw_chunks):
            clean = chunk_text_value.strip()
            if not clean:
                continue
            line_start, line_end, search_offset = locate_text_lines(text, clean, start_at=search_offset)
            chunks.append(
                FileChunk(
                    id=make_file_chunk_id(source, index),
                    text=clean,
                    source=source,
                    source_type=source_type,
                    file_path=source,
                    file_mtime=mtime,
                    file_size=file_size,
                    file_sha256=file_sha256,
                    chunk_index=index,
                    chunk_count=count,
                    chunk_hash=sha256_hex(clean),
                    heading=heading,
                    line_start=line_start,
                    line_end=line_end,
                    source_uri=source_uri,
                    tags=tags,
                )
            )
        return chunks

    def prepare(self, paths: Iterable[str | Path], *, max_files: int | None = None) -> dict[str, Any]:
        files, skipped = self.iter_files(paths, max_files=max_files)
        errors: list[dict[str, str]] = []
        chunks: list[FileChunk] = []
        indexed_files = 0
        file_manifests: list[dict[str, Any]] = []
        for path in files:
            resolved = path.resolve()
            try:
                stat = resolved.stat()
                raw = resolved.read_bytes()
                file_manifests.append(
                    {
                        "file_path": str(resolved),
                        "file_size": stat.st_size,
                        "file_mtime": stat.st_mtime,
                        "file_sha256": sha256_hex(raw),
                    }
                )
                file_chunks = self.prepare_file(resolved)
                if file_chunks:
                    indexed_files += 1
                    chunks.extend(file_chunks)
                else:
                    skipped.append({"path": str(path), "reason": "empty"})
            except Exception as exc:
                errors.append({"path": str(path), "error": str(exc)})
        return {
            "files": files,
            "file_manifests": file_manifests,
            "files_seen": len(files) + len(skipped),
            "files_indexed": indexed_files,
            "files_skipped": len(skipped),
            "skipped": skipped[:50],
            "chunks": chunks,
            "chunks_prepared": len(chunks),
            "errors": errors[:20],
            "max_files_truncated": any(item.get("reason") == "max_files" for item in skipped),
        }

    def index(self, paths: Iterable[str | Path], *, dry_run: bool = True, force: bool = False, max_files: int | None = None) -> dict[str, Any]:
        input_paths = list(paths)
        prepared = self.prepare(input_paths, max_files=max_files)
        chunks: list[FileChunk] = prepared.pop("chunks")
        file_manifests: list[dict[str, Any]] = prepared.get("file_manifests", [])
        desired_ids_by_file: dict[str, set[str]] = {str(item["file_path"]): set() for item in file_manifests}
        for chunk in chunks:
            desired_ids_by_file.setdefault(chunk.file_path, set()).add(chunk.id)

        summary: dict[str, Any] = {
            "dry_run": bool(dry_run),
            "files_seen": prepared["files_seen"],
            "files_indexed": prepared["files_indexed"],
            "files_skipped": prepared["files_skipped"],
            "chunks_prepared": len(chunks),
            "chunks_upserted": 0,
            "chunks_deleted": 0,
            "stale_ids": [],
            "stale_count": 0,
            "files_with_stale_chunks": 0,
            "manifest_checked": False,
            "directory_manifest_checked": False,
            "directory_roots_checked": [],
            "deleted_file_paths": [],
            "deleted_file_ids": [],
            "delete_mode": "none",
            "errors": list(prepared.get("errors", []))[:20],
            "paths": [str(p) for p in input_paths],
            "force": bool(force),
        }

        stale_ids: list[str] = []
        stale_seen: set[str] = set()
        manifest_capable = bool(self.qdrant and callable(getattr(self.qdrant, "scroll_by_filter", None)))
        if manifest_capable:
            summary["manifest_checked"] = True
            for file_path in sorted(desired_ids_by_file):
                try:
                    existing = self.qdrant.scroll_by_filter(
                        self.collection_name,
                        file_path_filter(file_path),
                        limit=256,
                        with_payload=True,
                        with_vector=False,
                    )
                    desired = desired_ids_by_file.get(file_path, set())
                    file_stale = [str(point.get("id")) for point in existing if point.get("id") is not None and str(point.get("id")) not in desired]
                    if file_stale:
                        summary["files_with_stale_chunks"] += 1
                        for point_id in file_stale:
                            if point_id not in stale_seen:
                                stale_seen.add(point_id)
                                stale_ids.append(point_id)
                except Exception as exc:
                    summary["errors"].append({"file_path": file_path, "error": f"manifest sync failed: {exc}"})

            directory_roots = [expand_path(str(path)) for path in input_paths if expand_path(str(path)).is_dir()]
            max_file_truncated = bool(prepared.get("max_files_truncated"))
            if directory_roots and not max_file_truncated:
                summary["directory_manifest_checked"] = True
                summary["directory_roots_checked"] = [str(root) for root in directory_roots]
                try:
                    existing_file_chunks = self.qdrant.scroll_by_filter(
                        self.collection_name,
                        file_chunk_filter(),
                        limit=256,
                        with_payload=True,
                        with_vector=False,
                    )
                    current_file_paths = set(desired_ids_by_file)
                    deleted_ids_by_path: dict[str, list[str]] = {}
                    for point in existing_file_chunks:
                        payload = point.get("payload", {}) or {}
                        existing_path = str(payload.get("file_path") or "")
                        point_id = point.get("id")
                        if not existing_path or point_id is None:
                            continue
                        if existing_path in current_file_paths:
                            continue
                        if Path(existing_path).exists():
                            continue
                        if not any(is_path_within(existing_path, root) for root in directory_roots):
                            continue
                        deleted_ids_by_path.setdefault(existing_path, []).append(str(point_id))
                    for deleted_path in sorted(deleted_ids_by_path):
                        summary["deleted_file_paths"].append(deleted_path)
                        for point_id in deleted_ids_by_path[deleted_path]:
                            summary["deleted_file_ids"].append(point_id)
                            if point_id not in stale_seen:
                                stale_seen.add(point_id)
                                stale_ids.append(point_id)
                    if deleted_ids_by_path:
                        summary["files_with_stale_chunks"] += len(deleted_ids_by_path)
                except Exception as exc:
                    summary["errors"].append({"error": f"directory manifest sync failed: {exc}"})

            summary["stale_ids"] = stale_ids
            summary["stale_count"] = len(stale_ids)
            if stale_ids:
                summary["delete_mode"] = "ids"

        if dry_run:
            summary["errors"] = summary["errors"][:20]
            return summary
        if not self.qdrant or (chunks and not self.embeddings):
            summary["errors"].append({"error": "qdrant and embeddings are required when dry_run is false"})
            return summary

        points: list[dict[str, Any]] = []
        for chunk in chunks:
            try:
                payload = chunk.payload(
                    profile_id=self.profile_id,
                    platform=self.platform,
                    session_id=self.session_id,
                    user_id_hash=self.user_id_hash,
                    chat_id_hash=self.chat_id_hash,
                    project_path=self.project_path,
                    model=self.model,
                )
                points.append({"id": chunk.id, "vector": self.embeddings.embed_document(chunk.text), "payload": payload})
            except Exception as exc:
                summary["errors"].append({"id": chunk.id, "error": str(exc)})

        if stale_ids and hasattr(self.qdrant, "delete_ids"):
            try:
                self.qdrant.delete_ids(self.collection_name, stale_ids)
                summary["chunks_deleted"] = len(stale_ids)
                summary["delete_mode"] = "ids"
            except Exception as exc:
                summary["errors"].append({"error": f"delete stale ids failed: {exc}"})
        elif force and desired_ids_by_file and hasattr(self.qdrant, "delete_filter"):
            summary["delete_mode"] = "filter"
            for file_path in sorted(desired_ids_by_file):
                try:
                    self.qdrant.delete_filter(self.collection_name, file_path_filter(file_path))
                except Exception as exc:
                    summary["errors"].append({"file_path": file_path, "error": f"delete stale chunks failed: {exc}"})

        for i in range(0, len(points), 64):
            batch = points[i : i + 64]
            if not batch:
                continue
            try:
                self.qdrant.upsert(self.collection_name, batch)
                summary["chunks_upserted"] += len(batch)
            except Exception as exc:
                summary["errors"].append({"error": f"upsert failed: {exc}"})
        summary["errors"] = summary["errors"][:20]
        return summary
