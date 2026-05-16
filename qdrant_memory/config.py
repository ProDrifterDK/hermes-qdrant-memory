from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "qdrant_url": "http://127.0.0.1:6333",
    "embedding_url": "http://127.0.0.1:8080/v1",
    "embedding_model": "bge-m3",
    "vector_size": 1024,
    "distance": "Cosine",
    "collection_name": "hermes_memory",
    "learning_collection_name": "hermes_learnings",
    "auto_recall": True,
    "auto_recall_top_k": 5,
    "search_candidates": 20,
    "decay_rate": 0.001,
    "max_chunk_tokens": 512,
    "display_tokens": 300,
    "sync_turns": True,
    "sync_subagents": False,
    "learning_enabled": True,
    "consolidation_enabled": False,
    "reconsolidation_enabled": False,
    "query_prefix": "search_query: ",
    "document_prefix": "search_document: ",
    "scope_mode": "profile",
    "min_raw_score": 0.0,
    "min_final_score": 0.0,
    "qdrant_api_key": "",
    "embedding_api_key": "",
    "index_dirs": [],
    "index_extensions": [".md", ".txt"],
    "index_exclude_dirs": [
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        "dist",
        "build",
        "target",
        ".next",
        ".cache",
    ],
    "index_max_files": 500,
    "index_dry_run_default": True,
}

_BOOL_KEYS = {
    "enabled",
    "auto_recall",
    "sync_turns",
    "sync_subagents",
    "learning_enabled",
    "consolidation_enabled",
    "reconsolidation_enabled",
    "index_dry_run_default",
}
_INT_KEYS = {"vector_size", "auto_recall_top_k", "search_candidates", "max_chunk_tokens", "display_tokens", "index_max_files"}
_FLOAT_KEYS = {"decay_rate", "min_raw_score", "min_final_score"}
_LIST_KEYS = {"index_dirs", "index_extensions", "index_exclude_dirs"}


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _coerce(key: str, value: Any) -> Any:
    default = DEFAULTS[key]
    if value is None:
        return default
    if key in _BOOL_KEYS:
        return _as_bool(value, bool(default))
    if key in _INT_KEYS:
        try:
            return int(value)
        except Exception:
            return default
    if key in _FLOAT_KEYS:
        try:
            return float(value)
        except Exception:
            return default
    if key in _LIST_KEYS:
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, tuple):
            return [str(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except Exception:
                pass
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return list(default)
    if isinstance(default, str):
        return str(value)
    return value


def _load_json_config(hermes_home: str | None) -> dict[str, Any]:
    if not hermes_home:
        return {}
    path = Path(hermes_home) / "qdrant_memory.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _load_hermes_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config  # type: ignore

        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def load_config(*, hermes_home: str | None = None, hermes_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load qdrant_memory config with safe defaults.

    Precedence: DEFAULTS < $HERMES_HOME/qdrant_memory.json < config.yaml qdrant_memory section <
    environment variables named HERMES_QDRANT_MEMORY_<KEY>.
    """
    merged: dict[str, Any] = dict(DEFAULTS)
    merged.update({k: v for k, v in _load_json_config(hermes_home).items() if k in DEFAULTS})
    root_cfg = dict(hermes_config) if hermes_config is not None else _load_hermes_config()
    section = root_cfg.get("qdrant_memory", {}) if isinstance(root_cfg, Mapping) else {}
    if isinstance(section, Mapping):
        merged.update({k: v for k, v in section.items() if k in DEFAULTS})
    for key in DEFAULTS:
        env_key = f"HERMES_QDRANT_MEMORY_{key.upper()}"
        if env_key in os.environ:
            merged[key] = os.environ[env_key]
    return {key: _coerce(key, merged.get(key)) for key in DEFAULTS}
