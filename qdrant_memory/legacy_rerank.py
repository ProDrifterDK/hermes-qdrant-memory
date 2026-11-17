"""Opt-in cross-encoder rerank of the legacy automatic-recall block (``auto_recall_rerank``).

``final_score`` is min-max normalized within each search, so a floor on it
(``auto_recall_min_final_score``) cuts by rank inside that search, not by relevance: it can
drop the chunk that answers the question and keep an off-topic one. With the mode on, the
legacy block takes a wider pool of the same search, scores every candidate's text against
the message with the reranker the session lane uses (``session_recall_rerank_url``), orders
the pool by that score and admits a chunk when it clears ``auto_recall_min_rerank_score``
(absolute, on the reranker's scale; ``off`` admits every chunk, reordered).

The pool never exceeds what the dense search fetches anyway (``search_candidates``, at most
20), so the Qdrant request, the scores and the quarantine and turn filters are those of the
mode-off block, and the pool's first ``auto_recall_top_k`` chunks are exactly the mode-off
candidates. Fail-open: no reranker URL, an error, an invalid response or the timeout give
the mode-off block, byte for byte.

Each chunk is sent as the block can show it: whitespace collapsed (as the formatter prints it)
and cut to the block's own character cap (``prompt_char_cap(display_tokens)``, at most
``DOC_MAX_CHARS``). Stored conversation turns reach 12,000 characters, past the context of a
typical reranker server, which then rejects the whole request; text past the cap never reaches
the prompt anyway.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .config import relevance_floor
from .retriever import prompt_char_cap
from .session_recall import MIN_RERANK_S, rerank

logger = logging.getLogger(__name__)

# One search returns at most this many results (MemoryRetriever.search).
POOL_MAX = 20
# Ceiling on the text sent per chunk, whatever display_tokens is (about 1,000 tokens).
DOC_MAX_CHARS = 4000


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    return default if number != number else number


@dataclass(frozen=True)
class LegacyRerankSettings:
    enabled: bool = False
    url: str = ""
    top_k: int = 5
    candidates: int = 20
    timeout_s: float = 1.5
    doc_chars: int = 1200
    # None = off: every pooled chunk is admitted, in rerank order. 0 is a real threshold.
    min_rerank_score: float | None = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LegacyRerankSettings":
        top_k = max(1, min(POOL_MAX, int(_number(config.get("auto_recall_top_k"), 5))))
        # The widest pool whose dense request is the mode-off block's: the retriever fetches
        # max(top_k, search_candidates) hits whatever the pool size below that.
        widest = max(top_k, min(POOL_MAX, int(_number(config.get("search_candidates"), 20))))
        wanted = int(_number(config.get("auto_recall_rerank_candidates"), 20))
        url = str(config.get("session_recall_rerank_url") or "").strip()
        return cls(
            enabled=bool(config.get("auto_recall_rerank")) and bool(url),
            url=url,
            top_k=top_k,
            candidates=max(top_k, min(widest, wanted)),
            timeout_s=max(0.1, min(30.0, _number(config.get("auto_recall_rerank_timeout_s"), 1.5))),
            doc_chars=min(DOC_MAX_CHARS, prompt_char_cap(max(0, int(_number(config.get("display_tokens"), 300))))),
            min_rerank_score=relevance_floor(config.get("auto_recall_min_rerank_score")),
        )


def rerank_documents(chunks: Sequence[Any], settings: LegacyRerankSettings) -> list[str]:
    """What the reranker scores for each chunk: its text as the block prints it, cut to the cap."""
    return [" ".join(str(getattr(c, "text", "") or "").split())[: settings.doc_chars] for c in chunks]


def rerank_pool(query: str, chunks: Sequence[Any], settings: LegacyRerankSettings, *,
                deadline: float) -> list[tuple[int, float]] | None:
    """``(index, score)`` best first for ``chunks``' texts, or None (fail-open).

    ``deadline`` is the ``time.monotonic()`` instant the call must end by: the HTTP timeout is
    the time left, and with less than ``MIN_RERANK_S`` left the reranker is not asked.
    """
    left = deadline - time.monotonic()
    if not settings.enabled or not chunks or left < MIN_RERANK_S:
        return None
    try:
        return rerank(settings.url, query, rerank_documents(chunks, settings), left)
    except Exception:
        logger.debug("Legacy recall rerank unavailable; the block keeps the dense order", exc_info=True)
        return None


def rerank_order(chunks: Sequence[Any], ranking: Sequence[tuple[int, float]], top_k: int,
                 min_rerank_score: float | None) -> list[Any]:
    """The chunks in ``ranking`` order, those below ``min_rerank_score`` left out, at most ``top_k``.

    A reranked chunk is judged by its rerank score alone: the final-score and raw-score floors
    of the dense order do not apply to it.
    """
    out: list[Any] = []
    for index, score in ranking:
        if min_rerank_score is not None and float(score) < min_rerank_score:
            continue
        out.append(chunks[index])
        if len(out) >= top_k:
            break
    return out
