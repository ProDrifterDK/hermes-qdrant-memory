"""Drive one improve / trace / inspect journey in session_origin_mode=off and write every observable.

Used by ``test_live_session_origin_c4.test_off_mode_is_byte_identical_to_base_except_the_new_tool``:
the same journey (tool schemas and prompt blocks on both surfaces, a legacy conversation turn,
improve preview / dry-run / apply over source text, improve preview over the legacy turn's
point id, trace and inspect of both points) runs once against a snapshot of the pre-C'4 tree
and once against this tree, each in its own interpreter, and the outputs must be equal byte
for byte apart from the deltas the test names. Every clock that stamps payloads or outputs is
frozen.

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
assert memory.startswith("tmp_c4_") and learnings.startswith("tmp_c4_"), (memory, learnings)

SID_I = "local-c4-intact"
MARKERS = "Please note this.\nGraph entity: project: Bluefin\nGraph entity: service: Kestrel"


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


def make_provider(**extra):
    provider = plugin.QdrantMemoryProvider()
    provider._active = True
    provider._hermes_home = home
    provider._profile_id = "c4-oracle"
    provider._platform = "pytest"
    provider._session_id = SID_I
    provider._config = load_config(hermes_home=home, hermes_config={"qdrant_memory": {
        "collection_name": memory,
        "learning_collection_name": learnings,
        "session_recall_collection": memory + "_windows_unused",
        "session_recall_state_db": state_db,
        "session_origin_mode": "off",
        "vector_size": 64,
        "lineage_lock_dir": home + "/locks",
        **extra,
    }})
    assert provider._config["collection_name"] == memory, provider._config["collection_name"]
    provider._qdrant = qdrant
    provider._embeddings = Embedding()
    provider._writer = ConversationWriter(qdrant=qdrant, embeddings=provider._embeddings, collection_name=memory,
                                          profile_id="c4-oracle", platform="pytest", session_id=SID_I)
    return provider


provider = make_provider()
recall = make_provider(tool_surface="recall")


def call(tool: str, args: dict) -> str:
    return provider.handle_tool_call(tool, args)


def points(point_id: str) -> str:
    return json.dumps(qdrant.retrieve(memory, [point_id], with_payload=True), sort_keys=True)


observed: dict[str, str] = {}
observed["schemas_full"] = json.dumps(provider.get_tool_schemas(), sort_keys=True)
observed["schemas_recall"] = json.dumps(recall.get_tool_schemas(), sort_keys=True)
observed["prompt_full"] = provider.system_prompt_block()
observed["prompt_recall"] = recall.system_prompt_block()

legacy_id = provider._writer.store_turn(MARKERS, "Noted both.")
observed["legacy_id"] = legacy_id

observed["improve_text_preview"] = call("qdrant_memory_improve_preview",
                                        {"source_text": "Graph entity: project: Bluefin", "persist": False})
report = json.loads(observed["improve_text_preview"])
candidate = next(c for c in report["candidates"] if c["candidate_type"] == "graph_entity_candidate")
apply_args = {"report_id": report["report_id"], "candidate_id": candidate["candidate_id"]}
observed["improve_apply_dry"] = call("qdrant_memory_improve_apply", apply_args)
observed["improve_apply"] = call("qdrant_memory_improve_apply", {**apply_args, "dry_run": False, "approve": True})
improve_id = json.loads(observed["improve_apply"])["id"]
observed["improve_payload"] = points(improve_id)
observed["improve_point_ids_preview"] = call("qdrant_memory_improve_preview",
                                             {"point_ids": [legacy_id], "persist": False})

for label, point_id in (("improve", improve_id), ("legacy", legacy_id)):
    observed[f"trace_{label}"] = call("qdrant_memory_trace", {"point_id": point_id, "direction": "both"})
    observed[f"inspect_{label}"] = call("qdrant_memory_inspect", {"point_id": point_id})

if "qdrant_memory_origin_check" in {s["name"] for s in provider.get_tool_schemas()}:
    observed["origin_check"] = call("qdrant_memory_origin_check", {"collection": "memory"})

with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(observed, handle, sort_keys=True, indent=1)
