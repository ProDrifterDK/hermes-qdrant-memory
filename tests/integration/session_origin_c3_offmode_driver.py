"""Drive one hook-candidate journey in session_origin_mode=off and write every observable.

Used by ``test_live_session_origin_c3.test_off_mode_is_byte_identical_to_base``: the same
journey (hooks in session A, a compression switch to A', learning and extraction preview
and approval, a learning store and re-store from session B) runs once against a snapshot of
the pre-C'3 tree and once against this tree, each in its own interpreter, and the outputs
must be equal byte for byte. Every clock that stamps payloads or outputs is frozen.

argv: tree_root hermes_agent_dir qdrant_url memory_collection learning_collection
      state_db hermes_home out_path
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import re
import sys

(tree, hermes_agent, qdrant_url, memory, learnings, state_db, home, out_path) = sys.argv[1:9]
sys.path[:0] = [tree, hermes_agent]

import qdrant_memory.schema as schema  # noqa: E402

schema.now_iso = lambda: "2026-11-15T12:00:00+00:00"

import __init__ as plugin  # noqa: E402
from qdrant_memory.client import QdrantClient  # noqa: E402
from qdrant_memory.config import load_config  # noqa: E402
from qdrant_memory.writer import ConversationWriter  # noqa: E402


class _FrozenDatetime(_dt.datetime):
    @classmethod
    def utcnow(cls):  # type: ignore[override]
        return cls(2026, 11, 15, 12, 0, 0)


plugin.datetime = _FrozenDatetime
assert memory.startswith("tmp_c3_") and learnings.startswith("tmp_c3_"), (memory, learnings)

SID_A, SID_A2, SID_B = "20261115_120000_c3aaaa", "20261115_130000_c3aaa2", "20261115_140000_c3bbbb"
HOOK_MESSAGES = [
    {"role": "assistant", "content": "Decision: route staging traffic through the blue proxy pool.\nI will note it."},
    {"role": "user", "content": "Actually, use the blue proxy pool, not the green one."},
    {"role": "assistant", "content": "Understood, the blue pool."},
]


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
provider = plugin.QdrantMemoryProvider()
provider._active = True
provider._hermes_home = home
provider._profile_id = "c3-oracle"
provider._platform = "pytest"
provider._session_id = SID_A
provider._config = load_config(hermes_home=home, hermes_config={"qdrant_memory": {
    "collection_name": memory,
    "learning_collection_name": learnings,
    "session_recall_collection": memory + "_windows_unused",
    "session_recall_state_db": state_db,
    "session_origin_mode": "off",
    "vector_size": 64,
    "lineage_lock_dir": home + "/locks",
    "learning_auto_extract_enabled": True,
    "learning_auto_extract_semantic_dedupe_enabled": False,
    "source_extraction_enabled": True,
}})
assert provider._config["collection_name"] == memory, provider._config["collection_name"]
provider._qdrant = qdrant
provider._embeddings = Embedding()
provider._writer = ConversationWriter(qdrant=qdrant, embeddings=provider._embeddings, collection_name=memory,
                                      profile_id="c3-oracle", platform="pytest", session_id=SID_A)


def call(tool: str, args: dict) -> str:
    return provider.handle_tool_call(tool, args)


def points(collection: str, point_id: str) -> str:
    return json.dumps(qdrant.retrieve(collection, [point_id], with_payload=True), sort_keys=True)


observed: dict[str, str] = {}
observed["schemas"] = json.dumps(provider.get_tool_schemas(), sort_keys=True)
observed["pre_compress"] = provider.on_pre_compress(HOOK_MESSAGES)
provider.on_session_switch(SID_A2, parent_session_id=SID_A, reset=False, reason="compression")

observed["learning_preview"] = call("qdrant_learning_preview", {"include_metadata": True})
learning_id = json.loads(observed["learning_preview"])["candidates"][0]["candidate_id"]
observed["learning_approve_dry"] = call("qdrant_learning_approve", {"candidate_id": learning_id})
learning_args = json.loads(observed["learning_approve_dry"])["learning_args"]
observed["learning_approve"] = call("qdrant_learning_approve", {"candidate_id": learning_id, "dry_run": False})
learning_point = json.loads(observed["learning_approve"])["id"]
observed["learning_payload"] = points(learnings, learning_point)

observed["extraction_preview"] = call("qdrant_memory_extraction_preview", {})
decision = next(item for item in json.loads(observed["extraction_preview"])["candidates"]
                if item["proposed_payload"].get("memory_kind") == "decision")
observed["extraction_approve_dry"] = call("qdrant_memory_extraction_approve", {"candidate_id": decision["candidate_id"]})
observed["extraction_approve"] = call("qdrant_memory_extraction_approve",
                                      {"candidate_id": decision["candidate_id"], "dry_run": False, "approve": True})
observed["extraction_payload"] = points(memory, decision["candidate_id"])

provider.on_session_switch(SID_B, parent_session_id="", reset=True)
observed["learning_restore"] = call("qdrant_learning_store", learning_args)
observed["learning_payload_after_restore"] = points(learnings, learning_point)
observed["learning_restore_again"] = call("qdrant_learning_store", {**learning_args, "importance": 9})
observed["learning_payload_final"] = points(learnings, learning_point)

with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(observed, handle, sort_keys=True, indent=1)
