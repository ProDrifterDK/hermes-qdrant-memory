"""Auto-recall prompt formatters: never header-only, never over budget.

``format_for_prompt`` (legacy) and ``format_hybrid_for_prompt`` (hybrid) build the
memory block that automatic recall injects. Their budget is ``max(800, display_tokens
* 4)`` characters, and the session-recall lane sizes its share from that same cap.

Ways they can fail:

1. Header only: the top-ranked entry alone exceeds what is left of the budget, the loop
   stops, and the prompt gets the header with no memory (the C'5 defect: 23/30 frozen
   questions at ``display_tokens: 300``). A header-only string is also truthy, so the
   prefetch cache and the hybrid-to-legacy fallback treat it as a hit.
2. Over budget: separators are not counted, so the block exceeds the cap and eats the
   lane's share of the shared total.
3. Rank order broken: a lower-ranked entry is shown before a higher-ranked one.
4. Silent truncation: an entry is cut with no visible marker.
5. Nothing showable but a non-empty string: callers rely on ``""`` meaning "no recall".

Tests 1 and 2 are covered here; 3 to 5 are asserted inside the same scenarios.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pytest

from qdrant_memory.retriever import RetrievedMemory, format_for_prompt, format_hybrid_for_prompt

TRUNCATION_MARKER = "[truncated]"


def _chunk(text: str, rank: int) -> RetrievedMemory:
    payload = {
        "created_at": "2026-01-01T00:00:00+00:00",
        "importance": 6,
        "source_type": "conversation",
        "source": "session/turn",
    }
    return RetrievedMemory(id=str(rank), text=text, payload=payload, qdrant_score=0.9, final_score=0.9 - rank / 100)


@dataclass
class _HybridResult:
    summaries: list = field(default_factory=list)
    exact_hits: list = field(default_factory=list)
    cited_leaves: list = field(default_factory=list)
    graph_relations: list = field(default_factory=list)


def _legacy(texts: list[str], display_tokens: int) -> str:
    return format_for_prompt([_chunk(t, i) for i, t in enumerate(texts)], display_tokens)


def _hybrid(texts: list[str], display_tokens: int) -> str:
    return format_hybrid_for_prompt(_HybridResult(exact_hits=[{"text": t} for t in texts]), display_tokens)


def _shown_entries(kind: str, out: str) -> list[str]:
    pattern = r"(?m)^\d+\. \[[^\n]*\]\n   (.*)$" if kind == "legacy" else r"(?m)^- (.*)$"
    return re.findall(pattern, out)


FORMATTERS = {"legacy": _legacy, "hybrid": _hybrid}


@pytest.mark.parametrize("kind", sorted(FORMATTERS))
def test_oversized_first_entry_still_yields_a_memory(kind):
    top = "TOPKEY " + "conversation turn detail " * 240  # ~6,000 chars, far over a 1,200 cap
    second = "SECONDKEY short memory that fits"
    out = FORMATTERS[kind]([top, second], 300)

    entries = _shown_entries(kind, out)
    assert out and entries, f"header-only or empty output: {out!r}"
    # The most relevant entry reaches the prompt, visibly cut, ahead of anything lower ranked.
    assert entries[0].startswith("TOPKEY"), entries
    assert entries[0].endswith(TRUNCATION_MARKER), entries[0][-40:]
    assert len(out) <= 1200

    # Every entry oversized: still the top one, never a bare header.
    out_all = FORMATTERS[kind]([top, top.replace("TOPKEY", "OTHERKEY")], 300)
    entries_all = _shown_entries(kind, out_all)
    assert entries_all and entries_all[0].startswith("TOPKEY"), out_all[:400]


@pytest.mark.parametrize("kind", sorted(FORMATTERS))
@pytest.mark.parametrize("display_tokens", [0, 50, 300, 700])
def test_output_never_exceeds_budget(kind, display_tokens):
    cap = max(800, display_tokens * 4)
    for sizes in ([10], [40] * 30, [cap // 3] * 6, [cap - 300, 50, 50], [cap * 3, 20], [5, cap * 2, 5, cap]):
        texts = [f"m{i} " + "y" * size for i, size in enumerate(sizes)]
        out = FORMATTERS[kind](texts, display_tokens)
        assert len(out) <= cap, (sizes, len(out), cap)
        entries = _shown_entries(kind, out)
        assert entries, (sizes, out[:300])
        # Rank order among shown entries.
        ranks = [int(entry.split(" ", 1)[0][1:]) for entry in entries]
        assert ranks == sorted(ranks), (sizes, ranks)
