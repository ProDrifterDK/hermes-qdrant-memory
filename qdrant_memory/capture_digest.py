"""``hermes qdrant capture digest``: a short plain-text summary of the capture queue.

Meant for a no-agent Hermes cron job that delivers stdout to a chat (decision P5): what
happened since the last digest (counts per route and decision), up to ``max_items`` pending
review items as one-line facts with their short ids (oldest first, at most 10), the items that
expire within three days, the facts stored automatically (lane A auto route) with the point id
to forget, and the exact commands or chat phrasing to approve or reject.

When nothing happened since the last digest it returns ``""`` so the cron job delivers
nothing. It reads the queue and writes only its own marker
(``$HERMES_HOME/qdrant_memory/capture/digest.json``: how many queue events the last digest
covered). Fact text is shown here on purpose: it is the owner's own review channel.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from typing import Any

from qdrant_memory.capture_queue import CaptureQueue, utc_iso, write_json_private

DIGEST_SCHEMA = "qdrant-capture-digest-v1"
MAX_ITEMS = 10
EXPIRING_DAYS = 3
FACT_CHARS = 140

_TEXT = {
    "en": {
        "title": "Qdrant capture digest",
        "since": "Since the last digest: {review} queued for review, {mirror} mirrored, {auto} auto-applied, "
                 "{approved} approved, {rejected} rejected, {expired} expired, {merged} merged.",
        "pending": "Pending review: {total} (showing {shown}, oldest first):",
        "none_pending": "Pending review: 0.",
        "more": "(+{more} more: hermes qdrant capture review)",
        "expiring": "Expiring within 3 days: {items}",
        "auto": "Stored automatically (undo: hermes qdrant forget <point_id> --no-dry-run --approve):",
        "approve": "Approve: hermes qdrant capture approve <id> --no-dry-run --approve "
                   "[--fact \"one sentence to store instead\"]",
        "reject": "Reject: hermes qdrant capture reject <id> --no-dry-run --approve",
        "chat": "In chat: \"approve capture <id>\" (the agent proposes one sentence to store; correct it before "
                "approving) or \"reject capture <id>\"",
    },
    "es": {
        "title": "Resumen de captura de memoria (Qdrant)",
        "since": "Desde el último resumen: {review} en cola por revisar, {mirror} espejadas, {auto} aplicadas "
                 "automáticamente, {approved} aprobadas, {rejected} rechazadas, {expired} vencidas, "
                 "{merged} fusionadas.",
        "pending": "Por revisar: {total} (se muestran {shown}, las más antiguas primero):",
        "none_pending": "Por revisar: 0.",
        "more": "(+{more} más: hermes qdrant capture review)",
        "expiring": "Vencen en 3 días o menos: {items}",
        "auto": "Guardadas automáticamente (deshacer: hermes qdrant forget <point_id> --no-dry-run --approve):",
        "approve": "Aprobar: hermes qdrant capture approve <id> --no-dry-run --approve "
                   "[--fact \"una frase para guardar en su lugar\"]",
        "reject": "Rechazar: hermes qdrant capture reject <id> --no-dry-run --approve",
        "chat": "En el chat: \"aprueba la captura <id>\" (el agente propone una frase para guardar; corrígela antes "
                "de aprobar) o \"rechaza la captura <id>\"",
    },
}


def _one_line(text: str, limit: int = FACT_CHARS) -> str:
    flat = " ".join(str(text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _marker_path(queue: CaptureQueue):
    return queue.dir / "digest.json"


def _covered(queue: CaptureQueue) -> int:
    try:
        data = json.loads(_marker_path(queue).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    value = data.get("events_covered") if isinstance(data, dict) else None
    return int(value) if isinstance(value, int) and value >= 0 else 0


def digest(hermes_home: str, *, max_items: int = MAX_ITEMS, language: str = "en", now: float | None = None,
           commit: bool = True) -> str:
    """The digest text, or ``""`` when nothing happened since the last one. Advances the marker."""
    clock = time.time() if now is None else float(now)
    text = _TEXT.get(str(language or "en").lower()[:2], _TEXT["en"])
    queue = CaptureQueue(hermes_home)
    events = queue.events()
    covered = _covered(queue)
    if covered > len(events):  # the queue was replaced: everything is new
        covered = 0
    new = events[covered:]
    if not new:
        return ""
    items = queue.items(clock, events)
    counts: Counter = Counter()
    auto_ids: list[str] = []
    for event in new:
        if event["event"] == "queued":
            counts[{"review": "review", "auto": "auto_queued", "mirror": "mirror"}.get(event.get("route"), "other")] += 1
        elif event["event"] == "applied" and event.get("route") == "auto":
            counts["auto"] += 1
            auto_ids.append(str(event.get("candidate_id")))
        elif event["event"] in ("approved", "rejected", "expired", "merged"):
            counts[event["event"]] += 1
    pending = sorted((item for item in items.values() if item.status == "pending"),
                     key=lambda item: (str(item.queued.get("at")), item.candidate_id))
    shown = pending[: max(0, min(int(max_items), MAX_ITEMS))]
    lines = [text["title"], text["since"].format(review=counts["review"], mirror=counts["mirror"], auto=counts["auto"],
                                                 approved=counts["approved"], rejected=counts["rejected"],
                                                 expired=counts["expired"], merged=counts["merged"])]
    if pending:
        lines.append(text["pending"].format(total=len(pending), shown=len(shown)))
        lines.extend(f"- [{item.short_id}] {item.queued.get('kind')}: {_one_line(item.fact)}" for item in shown)
        if len(pending) > len(shown):
            lines.append(text["more"].format(more=len(pending) - len(shown)))
        soon = [item for item in pending if item.expires_at is not None
                and item.expires_at - clock <= EXPIRING_DAYS * 86400]
        if soon:
            lines.append(text["expiring"].format(items=", ".join(
                f"[{item.short_id}] {time.strftime('%Y-%m-%d', time.localtime(item.expires_at))}" for item in soon)))
    else:
        lines.append(text["none_pending"])
    if auto_ids:
        lines.append(text["auto"])
        for cid in auto_ids:
            item = items.get(cid)
            if item is not None:
                lines.append(f"- [{item.short_id}] point {item.point_id}: {_one_line(item.fact)}")
    if pending:
        lines.extend([text["approve"], text["reject"], text["chat"]])
    if commit:
        write_json_private(_marker_path(queue), {"schema": DIGEST_SCHEMA, "events_covered": len(events),
                                                 "at": utc_iso(clock)})
    return "\n".join(lines)


def digest_language(config: dict[str, Any], requested: str | None) -> str:
    return str(requested or config.get("capture_digest_language") or "en")
