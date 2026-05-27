from __future__ import annotations

import hashlib
import uuid
from typing import Any

from .fact_metadata import derive_fact_metadata
from .schema import build_payload, clean_text_for_memory, now_iso
from .scoring import final_memory_score, normalize_minmax
from .retriever import RetrievedMemory, _extend_search_filter_conditions

LEARNING_TYPES = {
    "tool_failure_lesson",
    "user_correction",
    "workflow_lesson",
    "environment_quirk",
}


def make_learning_id(learning_type: str, trigger: str, mistake: str = "", correction: str = "") -> str:
    digest = hashlib.sha256(
        f"learning\n{learning_type}\n{trigger}\n{mistake}\n{correction}".encode("utf-8")
    ).hexdigest()
    return str(uuid.UUID(digest[:32]))


def classify_learning_type(trigger: str = "", correction: str = "", *, tool_name: str = "") -> str:
    text = f"{trigger}\n{correction}\n{tool_name}".lower()
    if any(word in text for word in ("correction", "corrected", "corrig", "correg", "spelled", "surname", "prefer")):
        return "user_correction"
    if any(word in text for word in ("workflow", "playbook", "procedure", "dry-run", "dry run", "before live", "steps")):
        return "workflow_lesson"
    if any(word in text for word in ("env", "environment", "local", "venv", "path", "installed", "not found", "quirk")):
        return "environment_quirk"
    if tool_name or any(word in text for word in ("tool", "failed", "failure", "error", "exception", "traceback", "http 500", "exit code")):
        return "tool_failure_lesson"
    return "workflow_lesson"


def _normalize_learning_type(value: str) -> str:
    value = str(value or "").strip()
    return value if value in LEARNING_TYPES else "workflow_lesson"


def _learning_text(*, lesson: str, trigger: str = "", mistake: str = "", correction: str = "", evidence: str = "") -> str:
    parts = [f"Lesson: {lesson.strip()}"]
    if trigger.strip():
        parts.append(f"Trigger: {trigger.strip()}")
    if mistake.strip():
        parts.append(f"Mistake: {mistake.strip()}")
    if correction.strip():
        parts.append(f"Correction: {correction.strip()}")
    if evidence.strip():
        parts.append(f"Evidence: {evidence.strip()}")
    return "\n".join(parts)


def build_learning_payload(
    *,
    lesson: str,
    learning_type: str = "workflow_lesson",
    trigger: str = "",
    mistake: str = "",
    correction: str = "",
    evidence: str = "",
    tool_name: str = "",
    command: str = "",
    project_path: str = "",
    profile_id: str = "default",
    platform: str = "cli",
    user_id_hash: str = "",
    chat_id_hash: str = "",
    session_id: str = "",
    model: str = "",
    importance: int = 7,
    confidence: float = 0.8,
    tags: list[str] | None = None,
    promote_to_skill_candidate: bool = False,
) -> dict[str, Any]:
    learning_type = _normalize_learning_type(learning_type)
    clean_lesson = clean_text_for_memory(lesson)[:12000]
    clean_trigger = clean_text_for_memory(trigger)[:4000]
    clean_mistake = clean_text_for_memory(mistake)[:4000]
    clean_correction = clean_text_for_memory(correction)[:4000]
    clean_evidence = clean_text_for_memory(evidence)[:4000]
    text = _learning_text(
        lesson=clean_lesson,
        trigger=clean_trigger,
        mistake=clean_mistake,
        correction=clean_correction,
        evidence=clean_evidence,
    )
    fact_metadata = derive_fact_metadata(
        text=text,
        source_type="learning",
        chunk_type=learning_type,
        tags=tags or [],
        learning_type=learning_type,
        tool_name=tool_name,
        command=command,
        project_path=project_path,
    )
    payload = build_payload(
        text=text,
        source="hermes_learning",
        source_type="learning",
        chunk_type=learning_type,
        importance=importance,
        confidence=confidence,
        tags=tags or [],
        profile_id=profile_id,
        platform=platform,
        user_id_hash=user_id_hash,
        chat_id_hash=chat_id_hash,
        session_id=session_id,
        project_path=project_path,
        model=model,
        memory_kind="learning",
        fact_metadata=fact_metadata,
    )
    payload.update(
        {
            "learning_type": learning_type,
            "trigger": clean_trigger,
            "mistake": clean_mistake,
            "correction": clean_correction,
            "evidence": clean_evidence,
            "tool_name": str(tool_name or "")[:200],
            "command": str(command or "")[:4000],
            "promote_to_skill_candidate": bool(promote_to_skill_candidate),
        }
    )
    return payload


class LearningStore:
    def __init__(
        self,
        *,
        qdrant,
        embeddings,
        collection_name: str,
        profile_id: str = "default",
        platform: str = "cli",
        session_id: str = "",
        user_id_hash: str = "",
        chat_id_hash: str = "",
        project_path: str = "",
        model: str = "",
        scope: dict[str, str] | None = None,
        search_candidates: int = 20,
        decay_rate: float = 0.001,
        min_raw_score: float = 0.0,
        min_final_score: float = 0.0,
    ):
        self.qdrant = qdrant
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.profile_id = profile_id
        self.platform = platform
        self.session_id = session_id
        self.user_id_hash = user_id_hash
        self.chat_id_hash = chat_id_hash
        self.project_path = project_path
        self.model = model
        self.scope = scope or {"profile_id": profile_id}
        self.search_candidates = int(search_candidates)
        self.decay_rate = float(decay_rate)
        self.min_raw_score = float(min_raw_score)
        self.min_final_score = float(min_final_score)

    def store(
        self,
        *,
        lesson: str,
        learning_type: str = "",
        trigger: str = "",
        mistake: str = "",
        correction: str = "",
        evidence: str = "",
        tool_name: str = "",
        command: str = "",
        importance: int = 7,
        confidence: float = 0.8,
        tags: list[str] | None = None,
        promote_to_skill_candidate: bool = False,
    ) -> str:
        clean_lesson = clean_text_for_memory(lesson)
        if not clean_lesson:
            return ""
        ltype = _normalize_learning_type(learning_type) if learning_type else classify_learning_type(trigger, correction, tool_name=tool_name)
        point_id = make_learning_id(ltype, trigger or clean_lesson[:200], mistake, correction or clean_lesson)
        payload = build_learning_payload(
            lesson=clean_lesson,
            learning_type=ltype,
            trigger=trigger,
            mistake=mistake,
            correction=correction,
            evidence=evidence,
            tool_name=tool_name,
            command=command,
            project_path=self.project_path,
            profile_id=self.profile_id,
            platform=self.platform,
            user_id_hash=self.user_id_hash,
            chat_id_hash=self.chat_id_hash,
            session_id=self.session_id,
            model=self.model,
            importance=max(1, min(10, int(importance))),
            confidence=float(confidence),
            tags=tags or [],
            promote_to_skill_candidate=promote_to_skill_candidate,
        )
        vector = self.embeddings.embed_document(payload["text"])
        self.qdrant.upsert(self.collection_name, [{"id": point_id, "vector": vector, "payload": payload}])
        return point_id

    def _filter(
        self,
        learning_type: str | None = None,
        scope: dict[str, str] | None = None,
        *,
        tags: list[str] | None = None,
        source: str | None = None,
        file_path: str | None = None,
        project_path: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> dict[str, Any]:
        must = [{"key": "source_type", "match": {"value": "learning"}}]
        active_scope = self.scope.copy()
        if scope:
            active_scope.update(scope)
        for key, value in active_scope.items():
            if value:
                must.append({"key": key, "match": {"value": value}})
        if learning_type:
            must.append({"key": "learning_type", "match": {"value": _normalize_learning_type(learning_type)}})
        _extend_search_filter_conditions(
            must,
            tags=tags,
            source=source,
            file_path=file_path,
            project_path=project_path,
            since=since,
            until=until,
        )
        return {"must": must}

    def find_semantic_duplicate(
        self,
        query: str,
        *,
        learning_type: str | None = None,
        threshold: float = 0.9,
        top_k: int = 3,
        scope: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Return a likely duplicate learning without mutating access metadata."""
        clean_query = clean_text_for_memory(query)
        if not clean_query:
            return None
        vector = self.embeddings.embed_query(clean_query)
        raw = self.qdrant.search(
            self.collection_name,
            vector,
            limit=max(1, min(10, int(top_k))),
            filter=self._filter(learning_type=learning_type, scope=scope),
            with_payload=True,
            with_vector=False,
        )
        best: dict[str, Any] | None = None
        best_score = float("-inf")
        for item in raw:
            try:
                score = float(item.get("score", 0.0))
            except Exception:
                score = 0.0
            if score > best_score:
                best = item
                best_score = score
        if best is not None and best_score >= float(threshold):
            return best
        return None

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        learning_type: str | None = None,
        scope: dict[str, str] | None = None,
        tags: list[str] | None = None,
        source: str | None = None,
        file_path: str | None = None,
        project_path: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[RetrievedMemory]:
        vector = self.embeddings.embed_query(query)
        raw = self.qdrant.search(
            self.collection_name,
            vector,
            limit=max(int(top_k), self.search_candidates),
            filter=self._filter(
                learning_type=learning_type,
                scope=scope,
                tags=tags,
                source=source,
                file_path=file_path,
                project_path=project_path,
                since=since,
                until=until,
            ),
            with_payload=True,
            with_vector=False,
        )
        scores = normalize_minmax([float(item.get("score", 0.0)) for item in raw])
        chunks: list[RetrievedMemory] = []
        for item, norm_score in zip(raw, scores):
            raw_score = float(item.get("score", 0.0))
            if raw_score < self.min_raw_score:
                continue
            payload = item.get("payload") or {}
            text = str(payload.get("text") or "")
            final = final_memory_score(norm_score, payload.get("importance", 7), payload.get("created_at", ""), self.decay_rate)
            if final < self.min_final_score:
                continue
            chunks.append(RetrievedMemory(str(item.get("id", "")), text, payload, raw_score, final))
        chunks.sort(key=lambda c: c.final_score, reverse=True)
        selected = chunks[: max(1, min(20, int(top_k)))]
        self.update_access_metadata(selected)
        return selected

    def update_access_metadata(self, chunks: list[RetrievedMemory]) -> None:
        for chunk in chunks:
            try:
                count = int((chunk.payload or {}).get("access_count", 0)) + 1
                self.qdrant.update_payload(self.collection_name, chunk.id, {"last_accessed": now_iso(), "access_count": count})
            except Exception:
                pass
