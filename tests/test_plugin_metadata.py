from pathlib import Path

from __init__ import QdrantMemoryProvider, register


ROOT = Path(__file__).resolve().parents[1]


def _parse_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def test_plugin_yaml_exists_and_has_required_metadata():
    plugin_yaml = ROOT / "plugin.yaml"

    assert plugin_yaml.exists()

    metadata = _parse_simple_yaml(plugin_yaml)

    assert metadata["name"] == "qdrant"
    assert metadata["kind"] == "exclusive"
    assert metadata["category"] == "memory"
    assert metadata["license"] == "MIT"
    assert metadata.get("version")
    assert metadata.get("homepage", "").startswith("https://")

    description = metadata.get("description", "")
    assert "Qdrant" in description
    assert "Hermes Agent" in description


class FakeContext:
    def __init__(self):
        self.providers = []

    def register_memory_provider(self, provider):
        self.providers.append(provider)


def test_register_registers_qdrant_memory_provider_without_initializing_clients():
    ctx = FakeContext()

    register(ctx)

    assert len(ctx.providers) == 1
    provider = ctx.providers[0]
    assert isinstance(provider, QdrantMemoryProvider)
    assert provider.name == "qdrant"
    assert provider._qdrant is None
    assert provider._embeddings is None
