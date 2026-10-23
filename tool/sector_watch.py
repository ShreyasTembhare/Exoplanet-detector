"""
Watch MAST for newly released TESS sectors.

A sector that just landed has had no community search yet, so it is where a
new planet is most likely to be found first. The continuous loop asks this
module for the latest released sector (cached in the state store so MAST is
hit at most every ``ttl_hours``), queues sectors released since the previous
check ahead of the backlog, and stops the sequential walk at the release
frontier instead of requesting sectors that do not exist yet.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

KV_KEY = "sector_watch"
DEFAULT_TTL_HOURS = 12.0
NEW_SECTOR_PRIORITY = 1000.0


def query_latest_sector() -> Optional[int]:
    """Highest TESS sector with released SPOC 2-minute light curves."""
    from astroquery.mast import Observations

    table = Observations.query_criteria(
        obs_collection="TESS", dataproduct_type="timeseries", provenance_name="SPOC",
    )
    if table is None or len(table) == 0 or "sequence_number" not in table.colnames:
        return None
    values = []
    for v in table["sequence_number"]:
        try:
            values.append(int(v))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


def latest_sector(store, ttl_hours: float = DEFAULT_TTL_HOURS,
                  fetcher: Optional[Callable[[], Optional[int]]] = None,
                  now: Optional[float] = None) -> Optional[int]:
    """Cached latest released sector; None when it has never been resolved."""
    now = time.time() if now is None else now
    cached = store.get_json(KV_KEY, {}) or {}
    if cached.get("latest") and now - float(cached.get("checked_at", 0)) < ttl_hours * 3600:
        return int(cached["latest"])
    try:
        latest = (fetcher or query_latest_sector)()
    except Exception as exc:
        logger.warning("sector watch: MAST query failed: %s", exc)
        latest = None
    if latest is None:
        return int(cached["latest"]) if cached.get("latest") else None
    store.set_json(KV_KEY, {**cached, "latest": int(latest), "checked_at": now})
    return int(latest)


def enqueue_new_sectors(store, plan: str, latest: Optional[int]) -> list:
    """Queue sectors released since the last call, newest first.

    The first call only records the frontier: everything before it is the
    backlog the sequential walk already covers.
    """
    if not latest:
        return []
    cached = store.get_json(KV_KEY, {}) or {}
    announced = cached.get("announced")
    store.set_json(KV_KEY, {**cached, "announced": max(int(latest), int(announced or 0))})
    if announced is None or int(latest) <= int(announced):
        return []
    seen = {int(row.get("sector") or 0) for row in store.items(plan=plan, limit=100000)}
    fresh = [s for s in range(int(announced) + 1, int(latest) + 1) if s not in seen]
    if fresh:
        store.enqueue(plan, [
            {"kind": "sector", "target": f"sector {s}", "sector": s,
             "priority": NEW_SECTOR_PRIORITY + s}
            for s in fresh
        ])
        logger.info("sector watch: queued newly released sectors %s", fresh)
    return fresh


def next_walk_sector(start: int, seen: set, latest: Optional[int]) -> Optional[int]:
    """First sector >= ``start`` not yet in the plan, or None past the frontier."""
    s = max(int(start), 1)
    while s in seen:
        s += 1
    if latest and s > int(latest):
        return None
    return s
