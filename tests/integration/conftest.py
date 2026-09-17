"""Shared live-service fixtures for Qdrant integration tests.

Only disposable collections created by these fixtures are ever written to.
Production collections (`hermes_memory`, `hermes_learnings`) are never a test
target. Fixtures fail (not skip) when integration is explicitly enabled but a
service is unavailable, and every created collection is deleted and verified
absent on teardown.
"""

from __future__ import annotations

import os
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pytest

from qdrant_memory.client import QdrantClient
from qdrant_memory.embeddings import EmbeddingClient

TRUTHY = {"1", "true", "yes", "on"}
PROFILE_ID = "integration-profile"
PLATFORM = "pytest"

# Fixed prefix for lineage integration collections (approved plan, section 5).
LINEAGE_TEST_PREFIX = "hermes_qdrant_lineage_itest"
# Names that must never be touched by tests.
_FORBIDDEN_COLLECTIONS = {"hermes_memory", "hermes_learnings"}


def integration_enabled() -> bool:
    return os.environ.get("RUN_QDRANT_INTEGRATION", "").strip().lower() in TRUTHY


def _qdrant_test_settings() -> dict[str, Any]:
    return {
        "qdrant_url": os.environ.get("QDRANT_TEST_URL", "http://127.0.0.1:6333"),
        "api_key": os.environ.get("QDRANT_TEST_API_KEY", ""),
        "embedding_url": os.environ.get("QDRANT_TEST_EMBEDDING_URL", "http://127.0.0.1:8080/v1"),
        "embedding_model": os.environ.get("QDRANT_TEST_EMBEDDING_MODEL", "bge-m3"),
        "vector_size": int(os.environ.get("QDRANT_TEST_VECTOR_SIZE", "1024")),
        "distance": os.environ.get("QDRANT_TEST_DISTANCE", "Cosine"),
    }


def _delete_and_verify(qdrant: QdrantClient, created: list[str], prefix: str) -> None:
    """Delete exactly the tracked collections and confirm absence."""
    errors: list[str] = []
    for collection_name in reversed(created):
        if not collection_name.startswith(prefix):
            errors.append(f"refused to delete non-prefixed collection {collection_name!r}")
            continue
        try:
            qdrant._request(
                "DELETE", f"/collections/{urllib.parse.quote(collection_name, safe='')}"
            )
        except Exception as exc:  # pragma: no cover - depends on live service state
            errors.append(f"failed to delete {collection_name!r}: {exc}")
    if errors:
        pytest.fail("collection cleanup failed: " + "; ".join(errors))
    if created:
        try:
            remaining = set(qdrant.get_collections())
        except Exception as exc:  # pragma: no cover
            pytest.fail(f"could not verify collection cleanup: {exc}")
        still_present = sorted(set(created) & remaining)
        if still_present:
            pytest.fail(f"collections still present after cleanup: {still_present}")


@dataclass(frozen=True)
class LiveContext:
    qdrant: QdrantClient
    embeddings: EmbeddingClient
    memory_collection: str
    learning_collection: str
    prefix: str
    vector_size: int
    distance: str
    scope: dict[str, str]


@pytest.fixture
def live_context() -> Iterator[LiveContext]:
    """Generic two-collection fixture (memory + learnings) for live tests."""
    prefix = os.environ.get("QDRANT_TEST_COLLECTION_PREFIX", "hermes_qdrant_itest").strip()
    if not prefix:
        pytest.fail("QDRANT_TEST_COLLECTION_PREFIX must not be empty")
    if prefix in _FORBIDDEN_COLLECTIONS:
        pytest.fail(f"refusing production collection prefix: {prefix!r}")

    settings = _qdrant_test_settings()
    qdrant = QdrantClient(settings["qdrant_url"], timeout=10.0, **{"api_key": settings["api_key"]})
    embeddings = EmbeddingClient(settings["embedding_url"], settings["embedding_model"], timeout=30.0)

    try:
        qdrant._request("GET", "/collections")
    except Exception as exc:  # pragma: no cover - depends on live service state
        pytest.fail(f"Qdrant health check failed for {settings['qdrant_url']}: {exc}")
    try:
        probe_vector = embeddings.embed_query("Hermes Qdrant live integration health probe")
    except Exception as exc:  # pragma: no cover - depends on live service state
        pytest.fail(f"Embedding health check failed for {settings['embedding_url']}: {exc}")
    if len(probe_vector) != settings["vector_size"]:
        pytest.fail(
            f"Embedding vector size mismatch for {settings['embedding_url']}: "
            f"expected {settings['vector_size']}, got {len(probe_vector)}"
        )

    random_suffix = uuid.uuid4().hex[:12]
    memory_collection = f"{prefix}_{random_suffix}_memory"
    learning_collection = f"{prefix}_{random_suffix}_learnings"
    for collection_name in (memory_collection, learning_collection):
        if not collection_name.startswith(prefix):
            pytest.fail(f"Refusing to use non-prefixed test collection: {collection_name!r}")

    created: list[str] = []
    try:
        existing_collections = set(qdrant.get_collections())
        for collection_name in (memory_collection, learning_collection):
            if collection_name in existing_collections:
                pytest.fail(f"Refusing to reuse existing test collection: {collection_name!r}")
        for collection_name in (memory_collection, learning_collection):
            result = qdrant.ensure_collection(collection_name, settings["vector_size"], settings["distance"])
            if result.get("exists"):
                pytest.fail(f"Refusing to reuse concurrently created test collection: {collection_name!r}")
            created.append(collection_name)
        yield LiveContext(
            qdrant=qdrant,
            embeddings=embeddings,
            memory_collection=memory_collection,
            learning_collection=learning_collection,
            prefix=prefix,
            vector_size=settings["vector_size"],
            distance=settings["distance"],
            scope={"profile_id": PROFILE_ID, "platform": PLATFORM},
        )
    finally:
        _delete_and_verify(qdrant, created, prefix)


@dataclass(frozen=True)
class LineageContext:
    qdrant: QdrantClient
    embeddings: EmbeddingClient
    primary_collection: str
    restore_collection: str
    prefix: str
    vector_size: int
    distance: str
    created: tuple[str, ...]


@pytest.fixture
def lineage_context() -> Iterator[LineageContext]:
    """Disposable collections under the fixed lineage test prefix.

    Enforces the exact `hermes_qdrant_lineage_itest` prefix, refuses existing
    names, tracks created collections, and verifies absence after teardown.
    """
    prefix = os.environ.get("QDRANT_TEST_COLLECTION_PREFIX", LINEAGE_TEST_PREFIX).strip()
    if not prefix:
        pytest.fail("QDRANT_TEST_COLLECTION_PREFIX must not be empty")
    if prefix != LINEAGE_TEST_PREFIX:
        pytest.fail(
            f"lineage integration tests require the fixed test prefix "
            f"{LINEAGE_TEST_PREFIX!r}, got {prefix!r}"
        )
    if prefix in _FORBIDDEN_COLLECTIONS:
        pytest.fail(f"refusing production collection prefix: {prefix!r}")

    settings = _qdrant_test_settings()
    qdrant = QdrantClient(settings["qdrant_url"], timeout=10.0, **{"api_key": settings["api_key"]})
    embeddings = EmbeddingClient(settings["embedding_url"], settings["embedding_model"], timeout=30.0)

    try:
        qdrant._request("GET", "/collections")
    except Exception as exc:  # pragma: no cover - depends on live service state
        pytest.fail(f"Qdrant health check failed for {settings['qdrant_url']}: {exc}")
    try:
        probe_vector = embeddings.embed_query("Hermes Qdrant live integration health probe")
    except Exception as exc:  # pragma: no cover - depends on live service state
        pytest.fail(f"Embedding health check failed for {settings['embedding_url']}: {exc}")
    if len(probe_vector) != settings["vector_size"]:
        pytest.fail(
            f"Embedding vector size mismatch for {settings['embedding_url']}: "
            f"expected {settings['vector_size']}, got {len(probe_vector)}"
        )

    suffix = uuid.uuid4().hex[:12]
    primary = f"{prefix}_{suffix}_a"
    restore = f"{prefix}_{suffix}_b"
    for collection_name in (primary, restore):
        if not collection_name.startswith(prefix):
            pytest.fail(f"Refusing to use non-prefixed test collection: {collection_name!r}")

    created: list[str] = []
    try:
        existing = set(qdrant.get_collections())
        for collection_name in (primary, restore):
            if collection_name in existing:
                pytest.fail(f"Refusing to reuse existing test collection: {collection_name!r}")
        for collection_name in (primary, restore):
            result = qdrant.ensure_collection(collection_name, settings["vector_size"], settings["distance"])
            if result.get("exists"):
                pytest.fail(f"Refusing to reuse concurrently created test collection: {collection_name!r}")
            created.append(collection_name)
        yield LineageContext(
            qdrant=qdrant,
            embeddings=embeddings,
            primary_collection=primary,
            restore_collection=restore,
            prefix=prefix,
            vector_size=settings["vector_size"],
            distance=settings["distance"],
            created=tuple(created),
        )
    finally:
        _delete_and_verify(qdrant, created, prefix)
