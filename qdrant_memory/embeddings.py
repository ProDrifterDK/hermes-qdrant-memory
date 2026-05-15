from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class EmbeddingClient:
    def __init__(self, base_url: str, model: str, *, query_prefix: str = "search_query: ", document_prefix: str = "search_document: ", timeout: float = 20.0, api_key: str = ""):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.timeout = timeout
        self.api_key = api_key or ""

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

    def embed(self, text: str) -> list[float]:
        body = {"model": self.model, "input": text}
        data = self._request("POST", "/embeddings", body)
        items = data.get("data", [])
        if not items or "embedding" not in items[0]:
            raise RuntimeError("Embedding response did not include data[0].embedding")
        return [float(x) for x in items[0]["embedding"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(f"{self.query_prefix}{text}")

    def embed_document(self, text: str) -> list[float]:
        return self.embed(f"{self.document_prefix}{text}")
