from __future__ import annotations

from typing import Any

from .fact_metadata import derive_fact_metadata
from .schema import build_payload, clean_text_for_memory, make_point_id, score_importance


def strip_injected_context(text: str) -> str:
    return clean_text_for_memory(text)

class ConversationWriter:
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

    def preview_text(
        self,
        text: str,
        *,
        source_type: str = "manual",
        chunk_type: str = "fact",
        importance: int | None = None,
        tags: list[str] | None = None,
        source: str = "hermes_tool",
    ) -> dict[str, Any]:
        clean = clean_text_for_memory(text)
        if not clean:
            return {"id": "", "text": "", "payload": {}}
        clean = clean[:12000]
        point_id = make_point_id(source, clean)
        fact_metadata = derive_fact_metadata(
            text=clean,
            source_type=source_type,
            chunk_type=chunk_type,
            tags=tags or [],
            project_path=self.project_path,
        )
        payload = build_payload(
            text=clean,
            source=source,
            source_type=source_type,
            chunk_type=chunk_type,
            importance=importance if importance is not None else score_importance(clean, source_type),
            tags=tags or [],
            profile_id=self.profile_id,
            platform=self.platform,
            user_id_hash=self.user_id_hash,
            chat_id_hash=self.chat_id_hash,
            session_id=self.session_id,
            project_path=self.project_path,
            model=self.model,
            fact_metadata=fact_metadata,
        )
        return {"id": point_id, "text": clean, "payload": payload}

    def find_semantic_duplicate(
        self,
        text: str,
        *,
        source_type: str | None = "manual",
        threshold: float = 0.92,
        top_k: int = 3,
    ) -> dict[str, Any] | None:
        clean = clean_text_for_memory(text)
        if not clean:
            return None
        clean = clean[:12000]
        embed_query = getattr(self.embeddings, "embed_query", None)
        vector = embed_query(clean) if callable(embed_query) else self.embeddings.embed_document(clean)
        must: list[dict[str, Any]] = []
        if self.profile_id:
            must.append({"key": "profile_id", "match": {"value": self.profile_id}})
        if source_type:
            must.append({"key": "source_type", "match": {"value": source_type}})
        raw = self.qdrant.search(
            self.collection_name,
            vector,
            limit=max(1, int(top_k or 3)),
            filter={"must": must} if must else None,
            with_payload=True,
            with_vector=False,
        )
        for item in raw:
            score = float(item.get("score", 0.0))
            if score < float(threshold):
                continue
            payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
            return {
                "id": str(item.get("id", "")),
                "score": score,
                "text": str(payload.get("text") or ""),
                "payload": payload,
            }
        return None

    def store_text(self, text: str, *, source_type: str = "manual", chunk_type: str = "fact", importance: int | None = None, tags: list[str] | None = None, source: str = "hermes_tool") -> str:
        preview = self.preview_text(text, source_type=source_type, chunk_type=chunk_type, importance=importance, tags=tags, source=source)
        clean = preview["text"]
        if not clean:
            return ""
        point_id = preview["id"]
        vector = self.embeddings.embed_document(clean)
        self.qdrant.upsert(self.collection_name, [{"id": point_id, "vector": vector, "payload": preview["payload"]}])
        return point_id

    def store_turn(self, user_content: str, assistant_content: str) -> str:
        clean_user = strip_injected_context(user_content)
        clean_assistant = strip_injected_context(assistant_content)
        if not clean_user or not clean_assistant:
            return ""
        if len(clean_user) + len(clean_assistant) < 20:
            return ""
        text = f"User: {clean_user}\nAssistant: {clean_assistant}"
        return self.store_text(text, source_type="conversation", chunk_type="turn", source="hermes_conversation_turn")
