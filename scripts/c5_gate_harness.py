#!/usr/bin/env python3
"""C'5 gate measurement: legacy auto-recall vs legacy plus the session-recall lane.

Read-only against live services. Every Qdrant request goes through a client that
refuses anything but reads, for every collection; ``state.db`` is opened ``mode=ro``.

Arms (auto-recall budget: top 5, legacy formatter cap ``max(800, display_tokens * 4)``):

  A          today's legacy prefetch (dense ``hermes_memory`` + read sources, top 5).
  B          A without conversation-turn points (``memory_kind=conversation_turn`` or
             ``source_type=conversation``+``chunk_type=turn`` excluded at query time).
  C_<s>_rr   A plus the session lane (scope ``s``), cross-encoder rerank on.
  C_<s>_dn   the same, dense order only.

The lane gets its own 1,200-character cap inside a budget shared with legacy (the
shipped defaults). ``*_eq`` rows re-score C at an equal total of the legacy cap (half
legacy, half lane) as a sensitivity check.

Scoring is the raw-RAG A/B rule: the question's answer-key regex (case-insensitive) must
appear in the arm's injected text, in returned order, within the arm's budget.

Noise set (``--noise``): real short user turns (confirmations, status pings) that
should inject nothing or very little. Each carries its own session id and timestamp,
used as the caller session and the time cutoff, like the live turn. Scored by injected
characters and blocks, for arms A, C_all_rr and C_all_dn.

Relevance floors (``--grid``): every arm's candidates are recorded once with their
scores (legacy: the top-k chunks with final and raw scores; lane: every entry of the
window search with its rerank and dense scores, floors off). The floors are then applied
offline with the plugin's own functions (``above_auto_recall_floor``,
``PrefetchSettings.admits``, the formatters), which is exact because a floor cuts a
suffix of the rank order. ``--analyze-only run.json`` recomputes the grid from a run.

Host scaffolding: every injected legacy memory and lane excerpt is checked with
``qdrant_memory/host_markers.py`` of this checkout (``host_hits`` per arm), also when
``--plugin-root`` points at another checkout (for example main, to count before a fix;
a checkout whose lane has no ``collect`` is measured at its own behaviour, floors off).

Time hygiene: each question has a cutoff (frozen set: the A/B corpus cut; continuity
set: the tip session's last message; noise: the turn's own timestamp). Legacy searches
``created_at <= cutoff`` and the lane ``start_ts < cutoff``, so no arm can read text
written after the question was (notionally) asked.

Usage (Hermes venv, from the repo root):

  python scripts/c5_gate_harness.py --frozen questions.json --continuity continuity.json \\
      --noise noise.json --out outputs/ [--display-tokens 0] [--grid grid.json] [--plugin-root DIR]
  python scripts/c5_gate_harness.py --analyze-only outputs/c5_gate_run.json --grid grid.json --out outputs/
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent.parent
TURN_EXCLUSION = [
    {"key": "memory_kind", "match": {"value": "conversation_turn"}},
    {"must": [
        {"key": "source_type", "match": {"value": "conversation"}},
        {"key": "chunk_type", "match": {"value": "turn"}},
    ]},
]
GPU_DIR = Path("/sys/class/drm/card1/device")
# Whitespace is collapsed in the lane block, so headers are found anywhere in a line.
_INLINE_HEADER = re.compile(r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2} (?:user|assistant)\] ")
# Body text of a compaction handoff a window may start inside of (agent/context_compressor.py).
_SUMMARY_BODY = ("## Historical Task Snapshot", "## User Messages (verbatim, newest first)", "## Context Recovery")
_ENTRY = re.compile(r"(?m)^\d+\. \[[^\n]*\]\n   (.*)$")
_PAYLOAD_KEYS = ("created_at", "importance", "source_type", "memory_kind", "fact_status", "file_path", "source",
                 "heading", "chunk_type")
DEFAULT_GRID = {"legacy_final": [0.0], "legacy_raw": [0.0], "lane_rerank": [None], "lane_dense": [None],
                "min_query_word_chars": [0]}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
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


# --- host scaffolding detector (this checkout's, whatever plugin is measured) ---------

_MARKERS = _load_module("c5_host_markers", HERE / "qdrant_memory" / "host_markers.py")


def host_text(text: str) -> bool:
    """True when an injected memory or excerpt carries host scaffolding."""
    text = str(text or "")
    if any(marker in text for marker in _SUMMARY_BODY):
        return True
    starts = [m.start() for m in _INLINE_HEADER.finditer(text)]
    if not starts or starts[0] != 0:
        starts.insert(0, 0)
    for a, b in zip(starts, starts[1:] + [len(text)]):
        segment = _INLINE_HEADER.sub("", text[a:b], count=1)
        parts = [segment]
        if segment.startswith("User: ") and " Assistant: " in segment:
            parts = segment[len("User: "):].split(" Assistant: ", 1)
        if any(_MARKERS.is_host_injected(part) for part in parts):
            return True
    return False


def block_entries(block: str) -> list[str]:
    return [m.group(1) for m in _ENTRY.finditer(block or "")]


# --- the measured plugin ---------------------------------------------------------------


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


def legacy_chunks(provider, query: str, until_iso: str | None) -> list[Any]:
    """Exactly prefetch's legacy search, plus the time cutoff (floors applied later)."""
    recall = provider._memory_recall() or provider._retriever
    kwargs: dict[str, Any] = {"update_access": False}
    if until_iso:
        kwargs["until"] = until_iso
    return recall.search(query, top_k=int(provider._config["auto_recall_top_k"]), **kwargs)


def chunk_record(chunk) -> dict[str, Any]:
    payload = chunk.payload or {}
    return {"id": chunk.id, "text": chunk.text, "final_score": chunk.final_score, "qdrant_score": chunk.qdrant_score,
            "payload": {k: payload[k] for k in _PAYLOAD_KEYS if k in payload}}


class Simulator:
    """Floors and formatting, with the measured plugin's own functions."""

    def __init__(self, display_tokens: int, lane_kw: dict[str, Any], expand_hint: bool = False):
        from qdrant_memory import retriever, session_prefetch

        self.retriever, self.sp = retriever, session_prefetch
        self.display_tokens = display_tokens
        self.lane_kw = lane_kw
        # The gate runs format the lane without the expand-tool hint, like every earlier C'5 run.
        self.expand_hint = expand_hint
        self.has_floors = hasattr(retriever, "above_auto_recall_floor") and hasattr(
            session_prefetch.PrefetchSettings, "admits")

    def legacy(self, chunks: list[dict[str, Any]], final_floor: float = 0.0, raw_floor: float = 0.0) -> str:
        items = [self.retriever.RetrievedMemory(id=c["id"], text=c["text"], payload=c["payload"],
                                                qdrant_score=c["qdrant_score"], final_score=c["final_score"])
                 for c in chunks]
        if self.has_floors:
            items = self.retriever.above_auto_recall_floor(items, final_floor, raw_floor)
        return self.retriever.format_for_prompt(items, self.display_tokens)

    def settings(self, rerank_floor=None, dense_floor=None):
        kw: dict[str, Any] = dict(scope="all", top_k=self.lane_kw["top_k"], max_chars=self.lane_kw["max_chars"],
                  total_chars=self.lane_kw["total_chars"], rerank=True, candidates=self.lane_kw["candidates"],
                  timeout_s=30.0)
        if self.has_floors:
            kw.update(min_rerank_score=rerank_floor, min_dense_score=dense_floor)
        return self.sp.PrefetchSettings(**kw)

    def lane(self, lane: dict[str, Any], legacy_text: str, rerank_floor=None, dense_floor=None, budget=None) -> str:
        settings = self.settings(rerank_floor, dense_floor)
        if budget is None:
            budget = settings.lane_budget(len(legacy_text))
        if "block" in lane:  # a checkout without collect: its own block, floors off
            return lane["block"] if budget >= self.sp.MIN_LANE_CHARS else ""
        entries = [self.sp._Entry(e["label"], e["msg_id"], e["text"], e["dense_score"], e["rerank_score"])
                   for e in lane["entries"]]
        entries = [e for e in entries if settings.admits(e.rerank_score, e.dense_score)][: settings.top_k]
        return self.sp.format_session_block(entries, budget, expand_hint=self.expand_hint)


def lane_record(provider, query: str, session_id: str, legacy_text: str, *, scope: str, rerank: bool,
                before_iso: str | None, candidates: int, top_k: int, max_chars: int, total_chars: int,
                exclude_machine: bool = False) -> dict:
    import dataclasses

    from qdrant_memory.session_prefetch import PrefetchSettings, SessionPrefetchLane

    settings = PrefetchSettings(scope=scope, top_k=top_k, max_chars=max_chars, total_chars=total_chars,
                                rerank=rerank, candidates=candidates, timeout_s=30.0)
    if exclude_machine:  # capture CP2 opt-in; a checkout without the field cannot run this arm
        settings = dataclasses.replace(settings, exclude_machine=True)
    lane = SessionPrefetchLane(provider._ensure_session_recall(), settings)
    if not hasattr(lane, "collect"):
        return {"block": lane.build(query, session_id, settings.lane_budget(len(legacy_text)), before=before_iso)}
    entries = lane.collect(query, session_id, before=before_iso, floors=False)
    return {"rerank": lane.last_rerank, "excluded": [list(span) for span in getattr(lane, "last_excluded", [])],
            "entries": [
        {"label": e.label, "msg_id": e.msg_id, "text": e.text, "dense_score": e.dense_score,
         "rerank_score": e.rerank_score} for e in entries]}


def hit(key: str, text: str, budget: int | None = None) -> bool:
    return bool(re.search(key, text[:budget] if budget else text, re.I))


def _word_chars(text: str) -> int:
    return len(re.findall(r"\w", text))


def arm_texts(sim: Simulator, row: dict[str, Any], point: dict[str, Any], legacy_cap: int) -> dict[str, str]:
    """Every arm's injected text for one row at one floor setting."""
    if _word_chars(row["q"]) < int(point.get("min_query_word_chars") or 0):
        return {arm: "" for arm in ["A", "B", *row["lane"]]}
    text = {
        "A": sim.legacy(row["legacy"]["A"], point["legacy_final"], point["legacy_raw"]),
        "B": sim.legacy(row["legacy"]["B"], point["legacy_final"], point["legacy_raw"]),
    }
    half = legacy_cap // 2
    for arm, lane in row["lane"].items():
        floors = (point["lane_rerank"], point["lane_dense"])
        block = sim.lane(lane, text["A"], *floors)
        text[arm] = sim.sp.compose_prefetch(text["A"], block)
        if row["set"] != "noise":
            text[f"{arm}_laneonly"] = block
            text[f"{arm}_eq"] = sim.sp.compose_prefetch(text["A"][:half], block[: legacy_cap - half])
    return text


def score(sim: Simulator, rows: list[dict[str, Any]], point: dict[str, Any], legacy_cap: int) -> dict[str, Any]:
    out: dict[str, Any] = {"point": point, "frozen": {}, "continuity": {}, "noise": {}, "host_hits": {}}
    for row in rows:
        text = arm_texts(sim, row, point, legacy_cap)
        for arm, value in text.items():
            if arm.endswith(("_eq", "_laneonly")):
                continue
            legacy_part = text["A"] if arm != "B" else text["B"]
            lane_part = value[len(text["A"]):] if arm.startswith("C_") else ""
            injected = block_entries(legacy_part) + block_entries(lane_part)
            hh = sum(host_text(e) for e in injected)
            out["host_hits"][arm] = out["host_hits"].get(arm, 0) + hh
            bucket = out[row["set"]].setdefault(arm, {"n": 0, "hits": 0, "chars": 0, "blocks": 0, "entries": 0,
                                                      "nonempty": 0})
            bucket["n"] += 1
            if row["set"] == "noise":
                bucket["chars"] += len(value)
                bucket["blocks"] += bool(legacy_part) + bool(lane_part.strip())
                bucket["entries"] += len(injected)
                bucket["nonempty"] += bool(value)
            else:
                bucket["hits"] += hit(row["key"], value)
        if row["set"] != "noise":
            for arm in [a for a in text if a.endswith(("_eq", "_laneonly"))]:
                bucket = out[row["set"]].setdefault(arm, {"n": 0, "hits": 0})
                bucket["n"] += 1
                bucket["hits"] += hit(row["key"], text[arm])
    return out


def grid_points(grid: dict[str, Any]) -> list[dict[str, Any]]:
    """Each floor varied on its own around the others' defaults (index 0), plus the explicit
    ``points`` (combinations) listed in the grid file."""
    base = {k: grid[k][0] for k in DEFAULT_GRID}
    points = [dict(base)]
    for key in DEFAULT_GRID:
        for value in grid[key][1:]:
            points.append({**base, key: value})
    for extra in grid.get("points") or []:
        points.append({**base, **extra})
    return points


def analyze(rows, legacy_cap, display_tokens, lane_kw, grid) -> list[dict[str, Any]]:
    sim = Simulator(display_tokens, lane_kw)
    return [score(sim, rows, point, legacy_cap) for point in grid_points(grid)]


def score_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Top-1 scores per set: what the floors have to separate."""
    out: dict[str, Any] = {}
    for set_name in ("frozen", "continuity", "noise"):
        subset = [r for r in rows if r["set"] == set_name]
        dist: dict[str, list[float]] = {"legacy_final": [], "legacy_raw": []}
        for r in subset:
            if r["legacy"]["A"]:
                dist["legacy_final"].append(r["legacy"]["A"][0]["final_score"])
                dist["legacy_raw"].append(r["legacy"]["A"][0]["qdrant_score"])
            for arm, lane in r["lane"].items():
                entries = lane.get("entries") or []
                if entries:
                    key = "rerank" if entries[0]["rerank_score"] is not None else "dense"
                    dist.setdefault(f"{arm}_{key}", []).append(
                        entries[0]["rerank_score"] if key == "rerank" else entries[0]["dense_score"])
        out[set_name] = {k: {"n": len(v), "min": _pct(v, 0), "p10": _pct(v, 0.1), "p25": _pct(v, 0.25),
                             "p50": _pct(v, 0.5), "p75": _pct(v, 0.75), "p90": _pct(v, 0.9), "max": _pct(v, 1)}
                         for k, v in dist.items()}
    return out


def run(args) -> dict[str, Any]:
    root = Path(args.plugin_root).resolve()
    sys.path.insert(0, str(root))
    plugin = _load_module("qdrant_memory_plugin_c5", root / "__init__.py")
    provider, client_cls = build_provider(plugin, args.hermes_home)
    from qdrant_memory.session_prefetch import compose_prefetch

    if args.display_tokens > 0:
        provider._config["display_tokens"] = args.display_tokens
    display_tokens = int(provider._config["display_tokens"])
    legacy_cap = max(800, display_tokens * 4)
    memory_collections = {provider._config["collection_name"]} | {
        s["collection"] for s in provider._read_sources("memory")}
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
    cont_doc = json.loads(Path(args.continuity).read_text(encoding="utf-8"))
    continuity = cont_doc["questions"] if isinstance(cont_doc, dict) else cont_doc
    noise: list[dict[str, Any]] = []
    if args.noise:
        noise_doc = json.loads(Path(args.noise).read_text(encoding="utf-8"))
        noise = noise_doc["questions"] if isinstance(noise_doc, dict) else noise_doc
    cut = float(args.frozen_cutoff)
    items = ([("frozen", q, "", cut) for q in frozen]
             + [("continuity", q, q["session_id"], float(q["before_ts"])) for q in continuity]
             + [("noise", q, q["session_id"], float(q["before_ts"])) for q in noise])

    sim = Simulator(display_tokens, {})
    # Harness check: with no cutoff, the replicated legacy branch is byte-identical to the
    # provider's own prefetch (lane disabled, the live config's floors).
    parity = []
    for q in frozen[:3]:
        chunks = [chunk_record(c) for c in legacy_chunks(provider, q["q"], None)]
        mine = sim.legacy(chunks, provider._config.get("auto_recall_min_final_score"),
                          provider._config.get("auto_recall_min_raw_score"))
        parity.append(mine == provider.prefetch(q["q"], session_id=""))

    lane_kw = dict(candidates=args.candidates, top_k=5, max_chars=1200, total_chars=legacy_cap + 1200 + 2,
                   exclude_machine=bool(args.exclude_machine))
    rows = []
    started_at = time.time()
    for set_name, q, sid, cutoff in items:
        until_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
        before_iso = datetime.fromtimestamp(cutoff).isoformat(timespec="seconds")
        row: dict[str, Any] = {"set": set_name, "id": q["id"], "q": q["q"], "key": q.get("key"),
                               "gpu_before": _gpu(), "legacy": {}, "lane": {}, "lane_errors": {}}
        lat: dict[str, float] = {}

        t0 = time.monotonic()
        row["legacy"]["A"] = [chunk_record(c) for c in legacy_chunks(provider, q["q"], until_iso)]
        lat["A"] = time.monotonic() - t0
        text_a = sim.legacy(row["legacy"]["A"])

        client = provider._qdrant
        client.exclude_turns_in = frozenset(memory_collections)
        try:
            t0 = time.monotonic()
            row["legacy"]["B"] = [chunk_record(c) for c in legacy_chunks(provider, q["q"], until_iso)]
            lat["B"] = time.monotonic() - t0
        finally:
            client.exclude_turns_in = frozenset()

        scopes = ["current", "all"] if set_name == "continuity" else ["all"]
        for scope in scopes:
            for rerank in (True, False):
                arm = f"C_{'cur' if scope == 'current' else 'all'}_{'rr' if rerank else 'dn'}"
                t0 = time.monotonic()
                try:
                    row["lane"][arm] = lane_record(provider, q["q"], sid, text_a, scope=scope, rerank=rerank,
                                                   before_iso=before_iso, **lane_kw)
                except Exception as exc:  # fail-open, as in prefetch
                    row["lane"][arm] = {"entries": [], "rerank": "error"}
                    row["lane_errors"][arm] = f"{type(exc).__name__}: {exc}"[:300]
                lat[f"{arm}_lane"] = time.monotonic() - t0
                lat[arm] = lat["A"] + lat[f"{arm}_lane"]

        row["gpu_after"] = _gpu()
        row["latency_s"] = {k: round(v, 4) for k, v in lat.items()}
        rows.append(row)
        print(f"{set_name:10s} {q['id']:>4} lanes={','.join(row['lane'])} "
              f"rerank={[v.get('rerank', '')[:20] for v in row['lane'].values()]}", flush=True)
    finished_at = time.time()

    latency: dict[str, dict[str, Any]] = {}
    for key in sorted({k for r in rows for k in r["latency_s"]}):
        values = [r["latency_s"][key] for r in rows if key in r["latency_s"]]
        latency[key] = {"n": len(values), "p50": _pct(values, 0.5), "p95": _pct(values, 0.95),
                        "max": round(max(values), 3), "mean": round(statistics.mean(values), 3)}
    del compose_prefetch
    return {
        "started_at": datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
        "finished_at": datetime.fromtimestamp(finished_at).isoformat(timespec="seconds"),
        "inputs": {
            "frozen": {"path": str(args.frozen), "sha256": _sha256(Path(args.frozen)), "n": len(frozen)},
            "continuity": {"path": str(args.continuity), "sha256": _sha256(Path(args.continuity)),
                           "n": len(continuity)},
            "noise": ({"path": str(args.noise), "sha256": _sha256(Path(args.noise)), "n": len(noise)}
                      if args.noise else None),
            "frozen_cutoff": cut,
        },
        "config": {
            "plugin_code": str(Path(str(sys.modules["qdrant_memory.session_prefetch"].__file__)).parent.parent),
            "has_floors": sim.has_floors,
            "collection_name": provider._config["collection_name"],
            "read_sources": provider._read_sources("memory"),
            "auto_recall_top_k": provider._config["auto_recall_top_k"],
            "display_tokens": display_tokens,
            "legacy_cap_chars": legacy_cap,
            "session_recall_collection": provider._config["session_recall_collection"],
            "session_recall_rerank_url": provider._config.get("session_recall_rerank_url"),
            "lane": lane_kw,
        },
        "legacy_parity_with_prefetch": parity,
        "mutations_refused": client_cls.mutations_refused,
        "latency": latency,
        "rows": rows,
    }


def summarize(result: dict[str, Any], grid: dict[str, Any]) -> dict[str, Any]:
    cfg = result["config"]
    points = analyze(result["rows"], cfg["legacy_cap_chars"], cfg["display_tokens"], cfg["lane"], grid)
    floor0 = points[0]
    table = {s: {"n": sum(1 for r in result["rows"] if r["set"] == s),
                 **{arm: v["hits"] for arm, v in floor0[s].items()}} for s in ("frozen", "continuity")}
    return {"table": table, "noise_floor0": floor0["noise"], "host_hits_floor0": floor0["host_hits"],
            "score_distribution": score_distribution(result["rows"]), "grid": points}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frozen")
    parser.add_argument("--continuity")
    parser.add_argument("--noise", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--hermes-home", default=str(Path.home() / ".hermes"))
    parser.add_argument("--plugin-root", default=str(HERE), help="checkout to measure (default: this one)")
    parser.add_argument("--frozen-cutoff", default="1793030539")
    parser.add_argument("--candidates", type=int, default=32)
    parser.add_argument("--display-tokens", type=int, default=300,
                        help="legacy display_tokens (300 = the D2 ~1,200-char budget; 0 = the live config)")
    parser.add_argument("--grid", default="", help="JSON: lists legacy_final, legacy_raw, lane_rerank, lane_dense, "
                        "min_query_word_chars (index 0 = the base point), optional points")
    parser.add_argument("--analyze-only", default="", help="recompute the grid from a recorded run")
    parser.add_argument("--tag", default="run")
    parser.add_argument("--exclude-machine", action="store_true",
                        help="lane arms with session_prefetch_exclude_machine on (capture CP2)")
    args = parser.parse_args()
    grid = dict(DEFAULT_GRID)
    if args.grid:
        grid.update(json.loads(Path(args.grid).read_text(encoding="utf-8")))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        result = json.loads(Path(args.analyze_only).read_text(encoding="utf-8"))
        root = Path(args.plugin_root).resolve()
        sys.path.insert(0, str(root))
    else:
        if not (args.frozen and args.continuity):
            parser.error("--frozen and --continuity are required unless --analyze-only")
        # The config (and so the collections and reranker) of the Hermes home being measured.
        os.environ["HERMES_HOME"] = args.hermes_home
        result = run(args)
        (out / f"c5_gate_{args.tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")
    summary = {k: result[k] for k in ("started_at", "finished_at", "inputs", "config", "latency",
                                      "legacy_parity_with_prefetch", "mutations_refused")}
    summary.update(summarize(result, grid))
    (out / f"c5_gate_{args.tag}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("table", "noise_floor0", "host_hits_floor0",
                                              "legacy_parity_with_prefetch", "mutations_refused")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
