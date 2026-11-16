"""Capture CP1: the session-hook capture paths are gone and stay gone.

Failure modes this file exists for (design CP1):
- a removed hook still runs: ``on_pre_compress`` / ``on_session_end`` call an extractor,
  keep candidates in provider state, or feed text back into the compression summary;
- the deprecated config keys break loading an existing install, change behaviour, or warn
  more than once per process;
- a tool whose only data source was the removed in-memory queue is still offered;
- the improve ``pending_session`` scope (fed by that queue) silently returns an empty report.

The fourth CP1 failure mode, tests writing the real ``watcher.log``, is guarded for the whole
session in ``tests/conftest.py``.
"""
from __future__ import annotations

import json
import logging

import pytest

from qdrant_memory import config as config_module
from qdrant_memory import source_extraction

REMOVED_TOOLS = (
    "qdrant_learning_preview",
    "qdrant_learning_approve",
    "qdrant_memory_extraction_preview",
    "qdrant_memory_extraction_approve",
)

DEPRECATED_SETTINGS = {
    "learning_auto_extract_enabled": True,
    "learning_auto_extract_mode": "store",
    "learning_auto_extract_min_confidence": 0.1,
    "learning_auto_extract_semantic_dedupe_enabled": False,
    "source_extraction_enabled": True,
    "source_extraction_mode": "store",
    "source_extraction_max_candidates_per_session": 50,
}

# Every shape either removed producer used to fire on (English line signals, the "Actually, ... not
# ..." correction, a tool failure followed by a correction), so a reinstated producer has input.
SIGNAL_TRANSCRIPT = [
    {"role": "user", "content": "Actually, always restart the worker with the priority queue, not the default one."},
    {"role": "assistant", "content": "Decision: route staging traffic through the blue proxy pool.\nInvariant: x."},
    {"role": "tool", "name": "terminal", "content": "pytest: command not found"},
    {"role": "assistant", "content": "Correction: use venv/bin/python -m pytest."},
    {"role": "user", "content": "Remember this: I prefer exact point ids."},
]


class RecordingQdrant:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append(name)
            return []
        return record


def _provider(tmp_path, **config):
    from __init__ import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    provider._active = True
    provider._hermes_home = str(tmp_path / "home")
    provider._session_id = "cp1-session"
    provider._qdrant = RecordingQdrant()
    provider._config.update({"collection_name": "memory", "learning_collection_name": "learnings", **config})
    return provider


def _state(provider):
    return {key: repr(value) for key, value in vars(provider).items()}


def test_session_hooks_collect_nothing_even_with_the_old_settings_on(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a session hook ran the source-line extractor")

    monkeypatch.setattr(source_extraction, "_line_signal", refuse)
    provider = _provider(tmp_path, **DEPRECATED_SETTINGS)
    before = _state(provider)

    summary_text = provider.on_pre_compress(SIGNAL_TRANSCRIPT)
    ended = provider.on_session_end(SIGNAL_TRANSCRIPT)

    assert summary_text == ""
    assert ended is None
    assert _state(provider) == before, "a session hook changed provider state (a candidate queue?)"
    assert provider._qdrant.calls == []
    assert not hasattr(provider, "_pending_learning_candidates")
    assert not hasattr(provider, "_pending_extraction_candidates")


def test_deprecated_keys_load_ignored_and_warn_once_per_process(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(config_module, "_deprecation_warned", False)
    (tmp_path / "qdrant_memory.json").write_text(json.dumps({"source_extraction_min_confidence": 0.2}))
    monkeypatch.setenv("HERMES_QDRANT_MEMORY_LEARNING_AUTO_EXTRACT_REQUIRE_EVIDENCE", "false")
    caplog.set_level(logging.WARNING, logger=config_module.logger.name)

    first = config_module.load_config(hermes_home=str(tmp_path), hermes_config={"qdrant_memory": DEPRECATED_SETTINGS})
    second = config_module.load_config(hermes_home=str(tmp_path), hermes_config={"qdrant_memory": DEPRECATED_SETTINGS})

    assert first == second
    assert not [key for key in first if key.startswith(config_module.DEPRECATED_KEY_PREFIXES)]
    warnings = [r for r in caplog.records if "removed capture settings" in r.getMessage()]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    for key in (*DEPRECATED_SETTINGS, "source_extraction_min_confidence", "learning_auto_extract_require_evidence"):
        assert key in message


def test_a_config_without_deprecated_keys_does_not_warn(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(config_module, "_deprecation_warned", False)
    caplog.set_level(logging.WARNING, logger=config_module.logger.name)

    config_module.load_config(hermes_home=str(tmp_path), hermes_config={"qdrant_memory": {"sync_turns": False}})

    assert not [r for r in caplog.records if "removed capture settings" in r.getMessage()]


@pytest.mark.parametrize("surface", ["full", "recall"])
@pytest.mark.parametrize("origin_mode", ["off", "recorded"])
@pytest.mark.parametrize("expand", [False, True])
def test_queue_only_tools_are_on_no_surface(tmp_path, surface, origin_mode, expand):
    from qdrant_memory.tools import ALL_TOOL_NAMES

    provider = _provider(tmp_path, tool_surface=surface, session_origin_mode=origin_mode,
                         session_expand_enabled=expand)

    offered = {schema["name"] for schema in provider.get_tool_schemas()}

    assert offered.isdisjoint(REMOVED_TOOLS)
    assert ALL_TOOL_NAMES.isdisjoint(REMOVED_TOOLS)


@pytest.mark.parametrize("tool", REMOVED_TOOLS)
def test_calling_a_removed_tool_is_an_unknown_tool_error(tmp_path, tool):
    provider = _provider(tmp_path)

    answer = json.loads(provider.handle_tool_call(tool, {"candidate_id": "anything"}))

    assert "error" in answer
    assert provider._qdrant.calls == []


@pytest.mark.parametrize("args", [{}, {"source_scope": "pending_session"}])
def test_improve_preview_refuses_the_removed_pending_session_scope(tmp_path, args):
    provider = _provider(tmp_path)

    answer = json.loads(provider.handle_tool_call("qdrant_memory_improve_preview", {**args, "persist": False}))

    assert "error" in answer
    assert "report_id" not in answer
