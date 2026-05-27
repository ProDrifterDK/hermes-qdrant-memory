from __future__ import annotations

from typing import Any, Mapping

from qdrant_memory.sources import SourceResolver


def adapter_resolvers(config: Mapping[str, Any] | None = None) -> list[SourceResolver]:
    config = config or {}
    if not config.get("obsidian_adapter_enabled"):
        return []
    vault_root = str(config.get("obsidian_vault_root") or "").strip()
    if not vault_root:
        return []
    from qdrant_memory.adapters.obsidian import ObsidianSourceResolver

    return [ObsidianSourceResolver(vault_root=vault_root)]
