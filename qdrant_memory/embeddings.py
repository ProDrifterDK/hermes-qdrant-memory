from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
from typing import Any

_TOO_LARGE_RE = re.compile(r"current batch size:\s*(\d+)", re.IGNORECASE)
_TOO_LARGE_MARKERS = (
    "too large to process",
    "input is too large",
    "increase the physical batch size",
)


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        query_prefix: str = "search_query: ",
        document_prefix: str = "search_document: ",
        timeout: float = 20.0,
        api_key: str = "",
        max_input_chars: int = 12000,
        max_chunks: int = 16,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.timeout = timeout
        self.api_key = api_key or ""
        self.max_input_chars = max(0, int(max_input_chars or 0))
        self.max_chunks = max(1, int(max_chunks or 1))

    def _request(self, method: str, path: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Embedding HTTP {exc.code}: {text}") from exc

    def models(self) -> Any:
        return self._request("GET", "/models")

    def health(self) -> bool:
        try:
            self.models()
            return True
        except Exception:
            return False

    def _parse_current_batch_size(self, error: RuntimeError) -> int:
        match = _TOO_LARGE_RE.search(str(error))
        if not match:
            return 0
        try:
            return max(0, int(match.group(1)))
        except Exception:
            return 0

    def _is_too_large_error(self, error: RuntimeError) -> bool:
        lowered = str(error).lower()
        return any(marker in lowered for marker in _TOO_LARGE_MARKERS)

    def _split_text(self, text: str, max_chars: int) -> list[str]:
        if max_chars <= 0 or len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        current = ""
        parts = re.split(r"(\s+)", text)
        for part in parts:
            if not part:
                continue
            if len(part) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                for start in range(0, len(part), max_chars):
                    chunks.append(part[start : start + max_chars])
                continue
            if current and len(current) + len(part) > max_chars:
                chunks.append(current)
                current = part
            else:
                current += part
        if current or not chunks:
            chunks.append(current)
        return chunks

    def _embedding_from_response(self, data: Any) -> list[float]:
        items = data.get("data", []) if isinstance(data, dict) else []
        if not items or "embedding" not in items[0]:
            raise RuntimeError("Embedding response did not include data[0].embedding")
        return [float(x) for x in items[0]["embedding"]]

    def _embed_single(self, text: str) -> list[float]:
        body = {"model": self.model, "input": text}
        return self._embedding_from_response(self._request("POST", "/embeddings", body))

    def _average_embeddings(self, vectors: list[list[float]], weights: list[int]) -> list[float]:
        if not vectors:
            raise RuntimeError("No embeddings produced for input")
        if len(vectors) == 1:
            return vectors[0]
        dim = len(vectors[0])
        if any(len(vector) != dim for vector in vectors):
            raise RuntimeError("Embedding chunks returned inconsistent vector dimensions")
        total_weight = float(sum(max(1, weight) for weight in weights))
        averaged = [0.0] * dim
        for vector, weight in zip(vectors, weights):
            factor = max(1, weight) / total_weight
            for idx, value in enumerate(vector):
                averaged[idx] += value * factor
        norm = math.sqrt(sum(value * value for value in averaged))
        if norm <= 0:
            return averaged
        return [value / norm for value in averaged]

    def _embed_input_with_fallback(self, text: str) -> list[float]:
        try:
            return self._embed_single(text)
        except RuntimeError as exc:
            if not self._is_too_large_error(exc) or len(text) <= 1:
                raise
            batch_size = self._parse_current_batch_size(exc)
            fallback_chars = batch_size if batch_size > 0 else max(1, len(text) // 2)
            if fallback_chars >= len(text):
                fallback_chars = max(1, len(text) // 2)
            chunks = self._split_text(text, fallback_chars)[: self.max_chunks]
            vectors = [self._embed_input_with_fallback(chunk) for chunk in chunks]
            return self._average_embeddings(vectors, [len(chunk) for chunk in chunks])

    def _embed_prefixed(self, text: str, prefix: str) -> list[float]:
        raw_text = text or ""
        if self.max_input_chars > 0:
            chunk_budget = max(1, self.max_input_chars - len(prefix))
        else:
            chunk_budget = 0
        chunks = self._split_text(raw_text, chunk_budget)[: self.max_chunks]
        vectors = [self._embed_input_with_fallback(f"{prefix}{chunk}") for chunk in chunks]
        return self._average_embeddings(vectors, [len(chunk) for chunk in chunks])

    def embed(self, text: str) -> list[float]:
        return self._embed_prefixed(text, "")

    def embed_query(self, text: str) -> list[float]:
        return self._embed_prefixed(text, self.query_prefix)

    def embed_document(self, text: str) -> list[float]:
        return self._embed_prefixed(text, self.document_prefix)
