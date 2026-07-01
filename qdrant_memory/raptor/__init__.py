"""RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) support.

Phase 3 — schema + deterministic dry-run builder.

This subpackage proposes RAPTOR trees and per-build manifests **without**
mutating Qdrant. It exposes:

- :mod:`qdrant_memory.raptor.schema` — dataclasses for RAPTOR nodes, trees,
  and build manifests, plus deterministic ID/digest helpers.
- :mod:`qdrant_memory.raptor.builder` — deterministic cluster/summary builder
  that consumes plain Qdrant-style leaf point dicts and returns dry-run
  artifacts (manifest + candidate node payloads).

Safety guarantees:

- No Qdrant mutations. The builder accepts plain point dicts only.
- Every candidate summary node is ``canonical=False`` and
  ``requires_review=True`` with full child IDs, source hashes, and
  provenance/``derived_from`` chains.
- MVP summaries are extractive (snippets from child leaves) — no LLM call,
  no abstractive freeform claims.
- Skip unsafe leaves: missing IDs, missing payload text, secret-bearing
  payloads, quarantined payloads, and leaves flagged with
  ``fact_status`` in ``{stale, deprecated, superseded, disputed,
  review_required}``.
- Cross-profile / cross-user / cross-chat leaves are never merged into a
  single cluster.
- Manifest digest is deterministic and excludes volatile timestamps.

Phase 4 (apply/status) is intentionally **not** exposed here.
"""

from .schema import (
    DEFAULT_PROMPT_VERSION,
    RAPTOR_DERIVATION_TYPE,
    RAPTOR_LEVEL_LEAF,
    RAPTOR_LEVEL_ROOT,
    RAPTOR_REQUIRED_NODE_FIELDS,
    RaptorBuildManifest,
    RaptorCluster,
    RaptorNode,
    RaptorScope,
    RaptorTree,
    compute_build_id,
    compute_manifest_digest,
    compute_node_id,
    compute_root_id,
    compute_tree_id,
)
from .builder import (
    DEFAULT_MAX_CLUSTER_SIZE,
    RaptorBuilder,
    build_raptor_dry_run,
)
from .search import (
    HARD_CONTEXT_CHAR_BUDGET,
    HARD_MAX_CHILDREN,
    HARD_MAX_DEPTH,
    HARD_MAX_SOURCE_CHARS,
    HARD_SEED_TOP_K,
    RaptorLeafHit,
    RaptorSearcher,
    RaptorSearchResult,
    RaptorSummaryHit,
)

__all__ = [
    "DEFAULT_MAX_CLUSTER_SIZE",
    "DEFAULT_PROMPT_VERSION",
    "HARD_CONTEXT_CHAR_BUDGET",
    "HARD_MAX_CHILDREN",
    "HARD_MAX_DEPTH",
    "HARD_MAX_SOURCE_CHARS",
    "HARD_SEED_TOP_K",
    "RAPTOR_DERIVATION_TYPE",
    "RAPTOR_LEVEL_LEAF",
    "RAPTOR_LEVEL_ROOT",
    "RAPTOR_REQUIRED_NODE_FIELDS",
    "RaptorBuilder",
    "RaptorBuildManifest",
    "RaptorCluster",
    "RaptorLeafHit",
    "RaptorNode",
    "RaptorScope",
    "RaptorSearcher",
    "RaptorSearchResult",
    "RaptorSummaryHit",
    "RaptorTree",
    "build_raptor_dry_run",
    "compute_build_id",
    "compute_manifest_digest",
    "compute_node_id",
    "compute_root_id",
    "compute_tree_id",
]
