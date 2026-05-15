#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdrant_memory.config import load_config  # noqa: E402
from qdrant_memory.client import QdrantClient  # noqa: E402
from qdrant_memory.embeddings import EmbeddingClient  # noqa: E402
from qdrant_memory.indexer import FileIndexer  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Index files into Hermes Qdrant memory outside a Hermes chat session.")
    parser.add_argument("paths", nargs="*", help="Files or directories to index. Defaults to configured index_dirs.")
    parser.add_argument("--live", action="store_true", help="Actually upsert points. Default is dry-run.")
    parser.add_argument("--force", action="store_true", help="Delete existing chunks for each file_path before upserting.")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum files to scan/index.")
    args = parser.parse_args()

    cfg = load_config()
    qdrant = QdrantClient(cfg["qdrant_url"], api_key=cfg.get("qdrant_api_key", ""))
    embeddings = EmbeddingClient(
        cfg["embedding_url"],
        cfg["embedding_model"],
        query_prefix=cfg["query_prefix"],
        document_prefix=cfg["document_prefix"],
        api_key=cfg.get("embedding_api_key", ""),
    )
    indexer = FileIndexer(
        qdrant=qdrant,
        embeddings=embeddings,
        collection_name=cfg["collection_name"],
        config=cfg,
        profile_id="default",
        platform="cli",
        session_id="script-index",
        model=cfg["embedding_model"],
    )
    result = indexer.index_paths(
        args.paths or cfg.get("index_dirs") or [],
        dry_run=not args.live,
        force=args.force,
        max_files=args.max_files,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not result.get("errors") else 2


if __name__ == "__main__":
    raise SystemExit(main())
