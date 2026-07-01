"""Phase 5 hybrid retrieve packer.

The packer is intentionally thin. It gathers three read-only result
fragments (dense+sparse seed hits, graph-aware expansions, and RAPTOR
search/zoom hits) and produces a single stable, dedupe-by-point-id output
shape with per-bucket provenance. No Qdrant mutation is reachable from
this module — every caller is required to pass objects whose own ``search``
methods are read-only.
"""

from .router import HybridRouter, HybridRouteResult
from .fusion import (
    deduplicate_by_point_id,
    rrf_fuse,
)

__all__ = [
    "HybridRouter",
    "HybridRouteResult",
    "deduplicate_by_point_id",
    "rrf_fuse",
]
