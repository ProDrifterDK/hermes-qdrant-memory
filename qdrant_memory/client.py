from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class QdrantClient:
    def __init__(self, base_url: str, *, timeout: float = 5.0, api_key: str = ""):
        self.base_url = (base_url or "").rstrip("/")
        self.timeout = timeout
        self.api_key = api_key or ""

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path if path.startswith('/') else '/' + path}"

    def _request(self, method: str, path: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["api-key"] = self.api_key
        req = urllib.request.Request(self._url(path), data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qdrant HTTP {exc.code}: {text}") from exc

    def health(self) -> bool:
        try:
            self._request("GET", "/")
            return True
        except Exception:
            try:
                self._request("GET", "/collections")
                return True
            except Exception:
                return False

    def get_collections(self) -> list[str]:
        data = self._request("GET", "/collections")
        items = data.get("result", {}).get("collections", [])
        return [str(item.get("name")) for item in items if item.get("name")]

    def ensure_collection(self, name: str, vector_size: int, distance: str) -> dict[str, Any]:
        if name in self.get_collections():
            return {"exists": True, "name": name}
        body = {"vectors": {"size": int(vector_size), "distance": str(distance)}}
        result = self._request("PUT", f"/collections/{urllib.parse.quote(name)}", body)
        return {"created": True, "name": name, "result": result}

    def count(self, name: str) -> int:
        data = self._request("POST", f"/collections/{urllib.parse.quote(name)}/points/count", {"exact": True})
        return int(data.get("result", {}).get("count", 0))

    def upsert(self, name: str, points: list[dict[str, Any]]) -> Any:
        return self._request("PUT", f"/collections/{urllib.parse.quote(name)}/points?wait=true", {"points": points})

    def search(
        self,
        name: str,
        vector: list[float],
        limit: int,
        filter: dict[str, Any] | None = None,
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "vector": vector,
            "limit": int(limit),
            "with_payload": with_payload,
            "with_vector": with_vector,
        }
        if filter:
            body["filter"] = filter
        data = self._request("POST", f"/collections/{urllib.parse.quote(name)}/points/search", body)
        return data.get("result", []) or []

    def update_payload(self, name: str, point_id: str, payload: dict[str, Any]) -> Any:
        body = {"payload": payload, "points": [point_id]}
        return self._request("POST", f"/collections/{urllib.parse.quote(name)}/points/payload?wait=true", body)

    def delete_ids(self, name: str, ids: list[str]) -> Any:
        body = {"points": ids}
        return self._request("POST", f"/collections/{urllib.parse.quote(name)}/points/delete?wait=true", body)

    def delete_filter(self, name: str, filter: dict[str, Any]) -> Any:
        body = {"filter": filter}
        return self._request("POST", f"/collections/{urllib.parse.quote(name)}/points/delete?wait=true", body)

    def scroll_by_filter(
        self,
        name: str,
        filter: dict[str, Any],
        *,
        limit: int = 256,
        with_payload: bool = True,
        with_vector: bool = False,
        max_total: int | None = None,
    ) -> list[dict[str, Any]]:
        points: list[dict[str, Any]] = []
        offset: Any = None
        page_limit = int(limit)
        while True:
            if max_total is not None:
                remaining = int(max_total) - len(points)
                if remaining <= 0:
                    break
                page_limit = min(page_limit, remaining)
            body: dict[str, Any] = {
                "filter": filter,
                "limit": page_limit,
                "with_payload": with_payload,
                "with_vector": with_vector,
            }
            if offset is not None:
                body["offset"] = offset
            data = self._request("POST", f"/collections/{urllib.parse.quote(name)}/points/scroll", body)
            result = data.get("result", {}) or {}
            batch = result.get("points", []) or []
            points.extend(batch)
            offset = result.get("next_page_offset")
            if offset is None or not batch:
                break
        return points
