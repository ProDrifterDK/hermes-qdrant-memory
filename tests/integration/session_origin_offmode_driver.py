"""Drive one manual-store journey in session_origin_mode=off and write every observable.

Used by ``test_live_session_origin.test_off_mode_is_byte_identical_to_base``: the same
journey runs once against a snapshot of the pre-C'2 tree and once against this tree, each
in its own interpreter, and the two outputs must be equal byte for byte. The clock that
stamps payloads is frozen so the comparison needs no normalisation.

argv: tree_root hermes_agent_dir qdrant_url memory_collection learning_collection
      state_db hermes_home cite_json out_path
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys

(tree, hermes_agent, qdrant_url, memory, learnings, state_db, home, cite_json, out_path) = sys.argv[1:10]
sys.path[:0] = [tree, hermes_agent]

import qdrant_memory.schema as schema  # noqa: E402

schema.now_iso = lambda: "2026-11-15T12:00:00+00:00"

from __init__ import QdrantMemoryProvider  # noqa: E402
from qdrant_memory.client import QdrantClient  # noqa: E402
from qdrant_memory.config import load_config  # noqa: E402
from qdrant_memory.writer import ConversationWriter  # noqa: E402

TEXT = "The staging proxy listens on port 8443."
assert memory.startswith("tmp_c2_") and learnings.startswith("tmp_c2_"), (memory, learnings)


class Embedding:
    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * 64
        for token in re.findall(r"[a-z0-9_-]+", text.casefold()):
            vector[int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 64] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    embed_document = _embed
    embed_query = _embed


qdrant = QdrantClient(qdrant_url, timeout=10.0)
provider = QdrantMemoryProvider()
provider._active = True
provider._hermes_home = home
provider._profile_id = "c2-oracle"
provider._platform = "pytest"
provider._session_id = "20261115_090000_aaaa01"
provider._config = load_config(hermes_home=home, hermes_config={"qdrant_memory": {
    "collection_name": memory,
    "learning_collection_name": learnings,
    "session_recall_collection": memory + "_windows_unused",
    "session_recall_state_db": state_db,
    "session_origin_mode": "off",
    "vector_size": 64,
    "lineage_lock_dir": home + "/locks",
}})
assert provider._config["collection_name"] == memory, provider._config["collection_name"]
provider._qdrant = qdrant
provider._embeddings = Embedding()
provider._writer = ConversationWriter(qdrant=qdrant, embeddings=provider._embeddings, collection_name=memory,
                                      profile_id="c2-oracle", platform="pytest", session_id=provider._session_id)

cite = json.loads(cite_json)
observed: dict[str, str] = {}
observed["schemas"] = json.dumps(provider.get_tool_schemas(), sort_keys=True)
observed["preview"] = provider.handle_tool_call("qdrant_memory_store", {"text": TEXT, "cite": cite})
observed["approve_a"] = provider.handle_tool_call(
    "qdrant_memory_store", {"text": TEXT, "cite": cite, "dry_run": False, "approve": True})
point_id = json.loads(observed["approve_a"])["id"]
observed["payload_a"] = json.dumps(qdrant.retrieve(memory, [point_id], with_payload=True), sort_keys=True)
provider.on_session_switch("20261115_100000_bbbb02")
observed["approve_b"] = provider.handle_tool_call(
    "qdrant_memory_store", {"text": TEXT, "cite": cite, "dry_run": False, "approve": True})
observed["payload_b"] = json.dumps(qdrant.retrieve(memory, [point_id], with_payload=True), sort_keys=True)
observed["inspect"] = provider.handle_tool_call("qdrant_memory_inspect", {"point_id": point_id})
observed["forget"] = provider.handle_tool_call("qdrant_memory_forget", {"ids": [point_id], "dry_run": False})
observed["after_forget"] = json.dumps(qdrant.retrieve(memory, [point_id], with_payload=True), sort_keys=True)

with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(observed, handle, sort_keys=True, indent=1)
