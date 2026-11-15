#!/usr/bin/env python3
"""Live E2E for the local/sovereign profile lane (read_sources + tool_surface: recall).

Runs against the real local services and only ever writes collections it creates
itself, all named ``tmp_lanetest_<run>_*``; they are deleted at the end, even on
failure. A temporary HERMES_HOME holds the profile config, written in the exact shape
of the README "Local or sovereign profile" example with the tmp collection names.

    python3 scripts/e2e_local_lane.py [--qdrant-url URL] [--embedding-url URL] [--out FILE]

Exit status 0 only when every check passes and cleanup is verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import traceback
import urllib.parse
import uuid
from pathlib import Path

PREFIX = "tmp_lanetest_"
FORBIDDEN = {"hermes_memory", "hermes_learnings", "hermes_session_windows", "resystbot_learnings", "picoclaw_memory"}
PLUGIN_ROOT = Path(__file__).resolve().parents[1]


def snapshot(qdrant, name: str) -> dict:
    points = qdrant.scroll_by_filter(name, {}, limit=256, with_payload=True)
    canonical = json.dumps(sorted(([str(p["id"]), p.get("payload")] for p in points), key=lambda x: x[0]), sort_keys=True)
    return {"count": qdrant.count(name), "payload_sha256": hashlib.sha256(canonical.encode()).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--embedding-url", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    run = uuid.uuid4().hex[:10]
    names = {
        "shared_memory": f"{PREFIX}{run}_shared_memory",
        "shared_learning": f"{PREFIX}{run}_shared_learnings",
        "own_memory": f"{PREFIX}{run}_own_memory",
        "own_learning": f"{PREFIX}{run}_own_learnings",
        "own_sessions": f"{PREFIX}{run}_own_session_windows",
    }
    assert all(n.startswith(PREFIX) and n not in FORBIDDEN for n in names.values())

    home = Path(tempfile.mkdtemp(prefix="lanetest-home-", dir=os.environ.get("TMPDIR") or None))
    # The profile's config, exactly the README example with tmp collection names.
    (home / "config.yaml").write_text(
        "memory:\n"
        "  provider: qdrant\n"
        "qdrant_memory:\n"
        f"  qdrant_url: {args.qdrant_url}\n"
        f"  embedding_url: {args.embedding_url}\n"
        f"  collection_name: {names['own_memory']}\n"
        f"  learning_collection_name: {names['own_learning']}\n"
        f"  session_recall_collection: {names['own_sessions']}\n"
        "  tool_surface: recall\n"
        "  sync_turns: false\n"
        "  read_sources:\n"
        f"    - {{collection: {names['shared_memory']}, profile_id: default, kind: memory}}\n"
        f"    - {{collection: {names['shared_learning']}, profile_id: default, kind: learning}}\n",
        encoding="utf-8",
    )
    os.environ["HERMES_HOME"] = str(home)
    for key in list(os.environ):
        if key.startswith("HERMES_QDRANT_MEMORY_") and key != "HERMES_QDRANT_MEMORY_LINEAGE_LOCK_DIR":
            del os.environ[key]
    os.environ["HERMES_QDRANT_MEMORY_LINEAGE_LOCK_DIR"] = str(home / "locks")
    sys.path.insert(0, str(PLUGIN_ROOT))

    from qdrant_memory.client import QdrantClient, ReadOnlyCollectionError
    from qdrant_memory.embeddings import EmbeddingClient
    from qdrant_memory.learning import LearningStore
    from qdrant_memory.writer import ConversationWriter

    admin = QdrantClient(args.qdrant_url, timeout=30.0)  # unguarded: setup and teardown only
    report: dict = {"run": run, "collections": names, "hermes_home": str(home), "checks": {}}
    checks = report["checks"]
    provider = None
    try:
        existing = set(admin.get_collections())
        clash = sorted(set(names.values()) & existing)
        if clash:
            raise SystemExit(f"refusing to reuse existing collections {clash}")
        embeddings = EmbeddingClient(args.embedding_url, "bge-m3", timeout=60.0)
        vector_size = len(embeddings.embed_query("probe"))
        for key in ("shared_memory", "shared_learning", "own_memory", "own_learning"):
            admin.ensure_collection(names[key], vector_size, "Cosine")

        # Seed "main's" shared memory through the plugin's own writers (profile default),
        # plus one point of another profile that the local lane must not see.
        main_writer = ConversationWriter(qdrant=admin, embeddings=embeddings, collection_name=names["shared_memory"], profile_id="default")
        main_writer.store_text("The desk band lives in ~/.hermes/deskband; start it with deskband.sh from that directory.")
        main_writer.store_text("Alan's GPU box runs the 27B model at about 30 tokens per second.")
        ConversationWriter(qdrant=admin, embeddings=embeddings, collection_name=names["shared_memory"], profile_id="other").store_text(
            "Other profile note: the desk band lives in /srv/elsewhere."
        )
        LearningStore(qdrant=admin, embeddings=embeddings, collection_name=names["shared_learning"], profile_id="default").store(
            lesson="Restart the desk band with systemctl --user restart deskband before debugging its output.",
            learning_type="workflow_lesson",
        )
        ConversationWriter(qdrant=admin, embeddings=embeddings, collection_name=names["own_memory"], profile_id="local27b").store_text(
            "Local profile note: the desk band reads its layout from deskband.toml."
        )

        before = {key: snapshot(admin, names[key]) for key in ("shared_memory", "shared_learning")}
        report["shared_before"] = before

        import importlib

        plugin = importlib.import_module("__init__")
        provider = plugin.QdrantMemoryProvider()
        provider.initialize(f"lanetest-{run}", hermes_home=str(home), agent_identity="local27b", platform="cli")
        cfg = provider._config
        assert cfg["collection_name"] == names["own_memory"], cfg["collection_name"]
        assert cfg["tool_surface"] == "recall" and cfg["sync_turns"] is False
        assert provider._qdrant.read_only_collections == {names["shared_memory"], names["shared_learning"]}
        report["effective_config"] = {k: cfg[k] for k in ("collection_name", "learning_collection_name", "session_recall_collection", "tool_surface", "sync_turns", "read_sources")}

        schemas = sorted(s["name"] for s in provider.get_tool_schemas())
        report["exposed_tools"] = schemas
        checks["six_tools_exposed"] = len(schemas) == 6

        prompt = provider.prefetch("where does the desk band live?")
        report["prefetch_prompt"] = prompt
        checks["prefetch_has_shared_fact"] = "~/.hermes/deskband" in prompt
        checks["prefetch_hides_other_profile"] = "/srv/elsewhere" not in prompt

        search = json.loads(provider.handle_tool_call("qdrant_memory_search", {"query": "where does the desk band live?", "top_k": 5}))
        report["search"] = search
        labels = {r.get("source_collection") for r in search.get("results", [])}
        checks["search_merged_from_both"] = {names["own_memory"], names["shared_memory"]} <= labels
        checks["search_sorted_by_score"] = [r["score"] for r in search["results"]] == sorted((r["score"] for r in search["results"]), reverse=True)
        checks["search_hides_other_profile"] = all("/srv/elsewhere" not in r["text"] for r in search["results"])
        # One scale across collections: the shared desk band note is the closer match
        # (higher vector score) and must rank above the own note, with a lower score.
        order = [(r.get("source_collection"), r["score"], r["vector_score"], r["text"][:50]) for r in search["results"]]
        report["search_rank_order"] = order
        shared_i = next((i for i, r in enumerate(search["results"]) if "~/.hermes/deskband" in r["text"]), None)
        own_i = next((i for i, r in enumerate(search["results"]) if r.get("source_collection") == names["own_memory"]), None)
        checks["search_ranks_closer_shared_note_above_own"] = (
            shared_i is not None
            and own_i is not None
            and search["results"][shared_i]["vector_score"] > search["results"][own_i]["vector_score"]
            and shared_i < own_i
            and search["results"][shared_i]["score"] > search["results"][own_i]["score"]
        )

        learning = json.loads(provider.handle_tool_call("qdrant_learning_search", {"query": "how do I restart the desk band", "top_k": 3}))
        report["learning_search"] = learning
        checks["learning_from_shared"] = any(r.get("source_collection") == names["shared_learning"] for r in learning.get("results", []))

        shared_id = next((r["id"] for r in search["results"] if r.get("source_collection") == names["shared_memory"]), "")
        expanded = json.loads(provider.handle_tool_call("qdrant_memory_expand", {"point_id": shared_id}))
        report["expand_shared"] = expanded
        checks["expand_reads_shared"] = expanded.get("collection_name") == names["shared_memory"] and "deskband" in str(expanded.get("text"))

        context = json.loads(provider.handle_tool_call("qdrant_memory_context", {"topic": "desk band location"}))
        context_labels = sorted({e.get("source_collection") for s in context.get("sections", {}).values() for e in s})
        report["context_labels"] = context_labels
        checks["context_labelled"] = names["shared_memory"] in context_labels

        refused_store = json.loads(provider.handle_tool_call("qdrant_memory_store", {"text": "local write attempt", "dry_run": False, "approve": True}))
        refused_forget = json.loads(provider.handle_tool_call("qdrant_memory_forget", {"point_id": shared_id}))
        report["hidden_tool_refusals"] = [refused_store, refused_forget]
        checks["hidden_tools_refused"] = all("tool_surface" in r.get("error", "") for r in (refused_store, refused_forget))

        try:
            provider._qdrant.update_payload(names["shared_memory"], shared_id, {"access_count": 999})
            checks["guard_refuses_direct_write"] = False
        except ReadOnlyCollectionError as exc:
            report["guard_refusal"] = str(exc)
            checks["guard_refuses_direct_write"] = True

        recall = provider._ensure_session_recall()
        report["session_recall_settings"] = {"collection": recall.settings.collection, "state_db": recall.settings.state_db}
        checks["session_recall_isolated"] = recall.settings.collection == names["own_sessions"] and recall.settings.state_db == str(home / "state.db")

        after = {key: snapshot(admin, names[key]) for key in ("shared_memory", "shared_learning")}
        report["shared_after"] = after
        checks["shared_unchanged"] = before == after
    except BaseException as exc:  # report, then clean up
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        if provider is not None:
            try:
                provider.shutdown()
            except Exception:
                pass
        deleted, cleanup_errors = [], []
        for name in sorted(n for n in admin.get_collections() if n.startswith(f"{PREFIX}{run}_")):
            try:
                admin._request("DELETE", f"/collections/{urllib.parse.quote(name, safe='')}")
                deleted.append(name)
            except Exception as exc:
                cleanup_errors.append(f"{name}: {exc}")
        leftover = sorted(n for n in admin.get_collections() if n.startswith(PREFIX))
        report["cleanup"] = {"deleted": deleted, "errors": cleanup_errors, "leftover_tmp_lanetest": leftover}
        checks["cleanup_verified"] = not cleanup_errors and not leftover

    report["ok"] = bool(checks) and all(checks.values()) and "error" not in report
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
