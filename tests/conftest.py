from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
HERMES_AGENT = Path.home() / ".hermes" / "hermes-agent"
for path in (PLUGIN_ROOT, HERMES_AGENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(scope="session", autouse=True)
def _isolated_default_lineage_lock_dir(tmp_path_factory):
    """Redirect the default lineage collection lock directory.

    Several call sites pass an empty ``lock_dir`` (``tests/test_lineage.py``
    through the indexer/restore paths, and provider writers when
    ``lineage_lock_dir`` is unset), which would otherwise create and flock
    ``/tmp/hermes-qdrant-lineage-<uid>`` - a path shared with the live runtime.
    Point the documented ``HERMES_QDRANT_MEMORY_LINEAGE_LOCK_DIR`` override at a
    per-session temporary directory so the suite never touches the shared path.
    """
    lock_dir = tmp_path_factory.mktemp("lineage-locks")
    previous = os.environ.get("HERMES_QDRANT_MEMORY_LINEAGE_LOCK_DIR")
    os.environ["HERMES_QDRANT_MEMORY_LINEAGE_LOCK_DIR"] = str(lock_dir)
    try:
        yield lock_dir
    finally:
        if previous is None:
            os.environ.pop("HERMES_QDRANT_MEMORY_LINEAGE_LOCK_DIR", None)
        else:
            os.environ["HERMES_QDRANT_MEMORY_LINEAGE_LOCK_DIR"] = previous
