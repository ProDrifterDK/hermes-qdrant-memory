#!/usr/bin/env python3
"""C'5 gate measurement: legacy auto-recall vs legacy plus the session-recall lane.

Read-only against live services. Every Qdrant request goes through a client that
refuses anything but reads, for every collection; ``state.db`` is opened ``mode=ro``.

Arms (auto-recall budget: top 5, legacy formatter cap ~1,200 characters):

  A          today's legacy prefetch (dense ``hermes_memory`` + read sources, top 5).
  B          A without conversation-turn points (``memory_kind=conversation_turn`` or
             ``source_type=conversation``+``chunk_type=turn`` excluded at query time).
  C_<s>_rr   A plus the session lane (scope ``s``), cross-encoder rerank on.
  C_<s>_dn   the same, dense order only.

The lane gets its own 1,200-character cap inside a 2,400-character budget shared with
legacy (the shipped defaults). ``*_eq`` rows re-score C at an equal total of 1,200
characters (600 legacy + 600 lane) as a sensitivity check.

Scoring is the raw-RAG A/B rule: the question's answer-key regex (case-insensitive) must
appear in the arm's injected text, in returned order, within the arm's budget.

Time hygiene: each question has a cutoff (frozen set: the A/B corpus cut; continuity
set: the tip session's last message). Legacy searches ``created_at <= cutoff`` and the
lane ``start_ts < cutoff``, so no arm can read text written after the question was
(notionally) asked, including the files that define the questions.

Usage (Hermes venv, from the repo root):

  python scripts/c5_gate_harness.py --frozen questions.json --continuity continuity.json \\
      --out outputs/ [--hermes-home ~/.hermes] [--frozen-cutoff 1793030539]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
TURN_EXCLUSION = [
    {"key": "memory_kind", "match": {"value": "conversation_turn"}},
    {"must": [
        {"key": "source_type", "match": {"value": "conversation"}},
        {"key": "chunk_type", "match": {"value": "turn"}},
    ]},
]
GPU_DIR = Path("/sys/class/drm/card1/device")


def _load_plugin():
    spec = importlib.util.spec_from_file_location("qdrant_memory_plugin_c5", REPO / "__init__.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gpu() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("mem_info_vram_used", "gpu_busy_percent"):
        try:
            out[name] = int((GPU_DIR / name).read_text().strip())
        except Exception:
            pass
    return out


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return round(ordered[index], 3)


def build_provider(plugin, hermes_home: str):
    from qdrant_memory.client import QdrantClient

    class ReadOnlyQdrant(QdrantClient):
        """Refuses every mutating request for every collection; can hide turn points."""

        exclude_turns_in: frozenset[str] = frozenset()
        mutations_refused = 0

        def _request(self, method, path, body=None):
            rest = path.split("?", 1)[0].rstrip("/")
            read = method == "GET" or (method == "POST" and rest.endswith(
                ("/points/search", "/points/scroll", "/points/count", "/points")))
            if not read:
                ReadOnlyQdrant.mutations_refused += 1
                raise RuntimeError(f"c5 harness refused a mutating request: {method} {rest}")
            return super()._request(method, path, body)

        def _with_exclusion(self, name, flt):
            if name not in self.exclude_turns_in:
                return flt
            flt = dict(flt or {})
            flt["must_not"] = list(flt.get("must_not") or []) + TURN_EXCLUSION
            return flt

        def search(self, name, vector, limit, filter=None, with_payload=True, with_vector=False):
            return super().search(name, vector, limit, filter=self._with_exclusion(name, filter),
                                  with_payload=with_payload, with_vector=with_vector)

        def scroll_page(self, name, filter, **kwargs):
            return super().scroll_page(name, self._with_exclusion(name, filter), **kwargs)

    plugin.QdrantClient = lambda *a, **kw: ReadOnlyQdrant(*a, **kw)
    # The live config may enable shadow telemetry; the harness must not write its artifacts.
    plugin.ShadowRecorder = None  # initialize() catches the TypeError and leaves it unset
    provider = plugin.QdrantMemoryProvider()
    provider.initialize("c5-gate-harness", hermes_home=hermes_home, agent_context="flush")
    provider._config["auto_recall_shadow_enabled"] = False
    provider._config["session_prefetch_enabled"] = False
    provider._shadow_recorder = None
    return provider, ReadOnlyQdrant


def legacy(provider, plugin, query: str, until_iso: str | None) -> str:
    """Exactly prefetch's legacy branch, plus the time cutoff."""
    recall = provider._memory_recall() or provider._retriever
    kwargs: dict[str, Any] = {"update_access": False}
    if until_iso:
        kwargs["until"] = until_iso
    chunks = recall.search(query, top_k=int(provider._config["auto_recall_top_k"]), **kwargs)
    return plugin.format_for_prompt(chunks, int(provider._config["display_tokens"]))


def lane_block(provider, query: str, session_id: str, legacy_text: str, *, scope: str, rerank: bool,
               before_iso: str | None, candidates: int, top_k: int, max_chars: int, total_chars: int) -> str:
    from qdrant_memory.session_prefetch import PrefetchSettings, SessionPrefetchLane

    settings = PrefetchSettings(scope=scope, top_k=top_k, max_chars=max_chars, total_chars=total_chars,
                                rerank=rerank, candidates=candidates, timeout_s=30.0)
    lane = SessionPrefetchLane(provider._ensure_session_recall(), settings)
    return lane.build(query, session_id, settings.lane_budget(len(legacy_text)), before=before_iso)


def hit(key: str, text: str, budget: int | None = None) -> bool:
    return bool(re.search(key, text[:budget] if budget else text, re.I))


def run(args) -> dict[str, Any]:
    plugin = _load_plugin()
    provider, client_cls = build_provider(plugin, args.hermes_home)
    from qdrant_memory.session_prefetch import compose_prefetch
    from qdrant_memory.session_recall import TOP_K_MAX

    if args.display_tokens > 0:
        # The D2 auto-recall budget: format_for_prompt caps legacy at max(800, tokens * 4).
        provider._config["display_tokens"] = args.display_tokens
    legacy_cap = max(800, int(provider._config["display_tokens"]) * 4)
    memory_collections = {provider._config["collection_name"]} | {
        s["collection"] for s in provider._read_sources("memory")}
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
    cont_doc = json.loads(Path(args.continuity).read_text(encoding="utf-8"))
    continuity = cont_doc["questions"] if isinstance(cont_doc, dict) else cont_doc
    cut = float(args.frozen_cutoff)
    items = [("frozen", q, "", cut) for q in frozen] + [
        ("continuity", q, q["session_id"], float(q["before_ts"])) for q in continuity]

    # Harness check: with no cutoff, the replicated legacy branch is byte-identical to
    # the provider's own prefetch (lane disabled).
    parity = []
    for q in frozen[:3]:
        parity.append(legacy(provider, plugin, q["q"], None) == provider.prefetch(q["q"], session_id=""))

    # Shipped defaults: lane cap 1,200; shared total = legacy cap + lane cap.
    lane_kw = dict(candidates=args.candidates, top_k=5, max_chars=1200, total_chars=legacy_cap + 1200 + 2)
    half = legacy_cap // 2
    rows = []
    started_at = time.time()
    for set_name, q, sid, cutoff in items:
        until_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
        before_iso = datetime.fromtimestamp(cutoff).isoformat(timespec="seconds")
        row: dict[str, Any] = {"set": set_name, "id": q["id"], "key": q["key"], "gpu_before": _gpu()}
        text: dict[str, str] = {}
        lat: dict[str, float] = {}

        t0 = time.monotonic()
        text["A"] = legacy(provider, plugin, q["q"], until_iso)
        lat["A"] = time.monotonic() - t0

        client = provider._qdrant
        client.exclude_turns_in = frozenset(memory_collections)
        try:
            t0 = time.monotonic()
            text["B"] = legacy(provider, plugin, q["q"], until_iso)
            lat["B"] = time.monotonic() - t0
        finally:
            client.exclude_turns_in = frozenset()

        scopes = ["all"] if set_name == "frozen" else ["current", "all"]
        for scope in scopes:
            for rerank in (True, False):
                arm = f"C_{'cur' if scope == 'current' else 'all'}_{'rr' if rerank else 'dn'}"
                t0 = time.monotonic()
                try:
                    block = lane_block(provider, q["q"], sid, text["A"], scope=scope, rerank=rerank,
                                       before_iso=before_iso, **lane_kw)
                    row.setdefault("lane_errors", {})
                except Exception as exc:  # fail-open, as in prefetch
                    block = ""
                    row.setdefault("lane_errors", {})[arm] = f"{type(exc).__name__}: {exc}"[:300]
                lat[f"{arm}_lane"] = time.monotonic() - t0
                lat[arm] = lat["A"] + lat[f"{arm}_lane"]
                text[arm] = compose_prefetch(text["A"], block)
                text[f"{arm}_eq"] = compose_prefetch(text["A"][:half], block[: legacy_cap - half])
                text[f"{arm}_laneonly"] = block

        if set_name == "frozen" and args.sanity:
            # The shipped tool path at the recorded A/B budget (top 15, 4,000 chars, rerank on).
            recall = provider._ensure_session_recall()
            t0 = time.monotonic()
            found = recall.search(q["q"], scope="all", chain=[], top_k=min(15, TOP_K_MAX), before=before_iso)
            lat["S4K"] = time.monotonic() - t0
            text["S4K"] = "\n".join(r["text"] for r in found["results"])[:4000]

        row["gpu_after"] = _gpu()
        row["hits"] = {arm: hit(q["key"], value) for arm, value in text.items()}
        row["chars"] = {arm: len(value) for arm, value in text.items()}
        row["latency_s"] = {k: round(v, 4) for k, v in lat.items()}
        row["evidence"] = text
        rows.append(row)
        print(f"{set_name:10s} {q['id']:>3} " + " ".join(
            f"{a}={'Y' if row['hits'][a] else '.'}" for a in sorted(row["hits"]) if not a.endswith("laneonly")),
            flush=True)

    finished_at = time.time()
    arms = sorted({a for r in rows for a in r["hits"]})
    table: dict[str, dict[str, Any]] = {}
    for set_name in ("frozen", "continuity"):
        subset = [r for r in rows if r["set"] == set_name]
        table[set_name] = {"n": len(subset)}
        for arm in arms:
            scored = [r for r in subset if arm in r["hits"]]
            if scored:
                table[set_name][arm] = sum(r["hits"][arm] for r in scored)
    latency: dict[str, dict[str, Any]] = {}
    for key in sorted({k for r in rows for k in r["latency_s"]}):
        values = [r["latency_s"][key] for r in rows if key in r["latency_s"]]
        latency[key] = {"n": len(values), "p50": _pct(values, 0.5), "p95": _pct(values, 0.95),
                        "max": round(max(values), 3), "mean": round(statistics.mean(values), 3)}
    return {
        "started_at": datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
        "finished_at": datetime.fromtimestamp(finished_at).isoformat(timespec="seconds"),
        "inputs": {
            "frozen": {"path": str(args.frozen), "sha256": _sha256(Path(args.frozen)), "n": len(frozen)},
            "continuity": {"path": str(args.continuity), "sha256": _sha256(Path(args.continuity)),
                           "n": len(continuity)},
            "frozen_cutoff": cut,
        },
        "config": {
            "plugin_code": str(Path(str(sys.modules["qdrant_memory.session_prefetch"].__file__)).parent.parent),
            "collection_name": provider._config["collection_name"],
            "read_sources": provider._read_sources("memory"),
            "auto_recall_top_k": provider._config["auto_recall_top_k"],
            "display_tokens": provider._config["display_tokens"],
            "legacy_cap_chars": legacy_cap,
            "session_recall_collection": provider._config["session_recall_collection"],
            "session_recall_rerank_url": provider._config.get("session_recall_rerank_url"),
            "lane": lane_kw,
        },
        "legacy_parity_with_prefetch": parity,
        "mutations_refused": client_cls.mutations_refused,
        "table": table,
        "latency": latency,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--continuity", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hermes-home", default=str(Path.home() / ".hermes"))
    parser.add_argument("--frozen-cutoff", default="1793030539")
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--display-tokens", type=int, default=300,
                        help="legacy display_tokens (300 = the D2 ~1,200-char budget; 0 = the live config)")
    parser.add_argument("--sanity", action="store_true", help="also run the tool path at 15/4,000 chars")
    parser.add_argument("--tag", default="run")
    args = parser.parse_args()
    # The config (and so the collections and reranker) of the Hermes home being measured.
    os.environ["HERMES_HOME"] = args.hermes_home
    result = run(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"c5_gate_{args.tag}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {k: result[k] for k in ("started_at", "finished_at", "inputs", "config", "table", "latency",
                                      "legacy_parity_with_prefetch", "mutations_refused")}
    (out / f"c5_gate_{args.tag}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({"table": result["table"], "latency": result["latency"],
                      "parity": result["legacy_parity_with_prefetch"],
                      "mutations_refused": result["mutations_refused"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
