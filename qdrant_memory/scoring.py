from __future__ import annotations

import math
from datetime import datetime, timezone


def _parse_dt(value: str) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def recency_score(created_at: str, decay_rate: float, now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    created = _parse_dt(created_at)
    hours = max(0.0, (now - created).total_seconds() / 3600.0)
    return math.exp(-float(decay_rate) * hours)


def normalize_minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    nums = [float(s) for s in scores]
    lo, hi = min(nums), max(nums)
    if math.isclose(lo, hi):
        return [1.0 for _ in nums]
    return [(s - lo) / (hi - lo) for s in nums]


def final_memory_score(qdrant_score: float, importance: int | float, created_at: str, decay_rate: float) -> float:
    imp = max(0.0, min(10.0, float(importance))) / 10.0
    return max(0.0, float(qdrant_score)) * imp * recency_score(created_at, decay_rate)
