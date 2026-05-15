#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdrant_memory.config import load_config  # noqa: E402
from qdrant_memory.client import QdrantClient  # noqa: E402
from qdrant_memory.embeddings import EmbeddingClient  # noqa: E402


def main() -> int:
    cfg = load_config()
    qdrant = QdrantClient(cfg["qdrant_url"], api_key=cfg.get("qdrant_api_key", ""))
    embeddings = EmbeddingClient(
        cfg["embedding_url"],
        cfg["embedding_model"],
        query_prefix=cfg["query_prefix"],
        document_prefix=cfg["document_prefix"],
        api_key=cfg.get("embedding_api_key", ""),
    )
    payload = {
        "qdrant_url": cfg["qdrant_url"],
        "embedding_url": cfg["embedding_url"],
        "embedding_model": cfg["embedding_model"],
        "collection_name": cfg["collection_name"],
        "vector_size": cfg["vector_size"],
        "qdrant_ok": qdrant.health(),
        "embedding_ok": embeddings.health(),
    }
    try:
        payload["point_count"] = qdrant.count(cfg["collection_name"])
    except Exception as exc:
        payload["point_count_error"] = str(exc)
    print(json.dumps(payload, indent=2))
    return 0 if payload["qdrant_ok"] and payload["embedding_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
