from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
HERMES_AGENT = Path.home() / ".hermes" / "hermes-agent"
for path in (PLUGIN_ROOT, HERMES_AGENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# --- HERMES_HOME isolation ---------------------------------------------------------------
#
# Code under test writes watcher state, the watcher log, consolidation reports and proposal
# drafts under ``$HERMES_HOME`` (falling back to ``~/.hermes``). A test that forgets its own
# ``monkeypatch.setenv("HERMES_HOME", ...)`` used to append to the real
# ``qdrant_memory/consolidation/watcher.log`` of whoever ran the suite (rows with
# ``"report_id": "r1"``). The whole session therefore runs with HERMES_HOME pointed at a
# throwaway directory, set before collection, and the session fails if the real watcher files
# changed anyway (a write through a hard-coded ``~/.hermes`` path).

_GUARDED_FILES = ("qdrant_memory/consolidation/watcher.log", "qdrant_memory/consolidation/watcher_state.json")
_REAL_HOMES: list[Path] = []
_BEFORE: dict[str, str | None] = {}
_PREVIOUS_HERMES_HOME: str | None = None
_ISOLATED_HOME: list[Path] = []


def _digest(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _guarded_paths() -> list[Path]:
    return [home / rel for home in _REAL_HOMES for rel in _GUARDED_FILES]


def pytest_configure(config):
    global _PREVIOUS_HERMES_HOME
    _PREVIOUS_HERMES_HOME = os.environ.get("HERMES_HOME")
    homes = {Path.home() / ".hermes"}
    if _PREVIOUS_HERMES_HOME:
        homes.add(Path(_PREVIOUS_HERMES_HOME))
    _REAL_HOMES[:] = sorted(homes)
    _BEFORE.clear()
    _BEFORE.update({str(path): _digest(path) for path in _guarded_paths()})
    isolated = Path(tempfile.mkdtemp(prefix="qdrant-memory-tests-home-"))
    _ISOLATED_HOME[:] = [isolated]
    os.environ["HERMES_HOME"] = str(isolated)


def pytest_sessionfinish(session, exitstatus):
    changed = [path for path, before in _BEFORE.items() if _digest(Path(path)) != before]
    if changed:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        message = ("the test session changed real Hermes files it must never write: " + ", ".join(changed)
                   + " (a test bypassed the isolated HERMES_HOME; if a live watcher run wrote them during "
                   "the session, re-run)")
        if reporter is not None:
            reporter.write_line("ERROR: " + message, red=True, bold=True)
        else:
            print("ERROR: " + message, file=sys.stderr)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_unconfigure(config):
    if _PREVIOUS_HERMES_HOME is None:
        os.environ.pop("HERMES_HOME", None)
    else:
        os.environ["HERMES_HOME"] = _PREVIOUS_HERMES_HOME
    for isolated in _ISOLATED_HOME:
        shutil.rmtree(isolated, ignore_errors=True)


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
