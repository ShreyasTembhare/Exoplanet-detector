"""Find the TESS sectors MAST currently publishes, and the next unseen stars."""

from __future__ import annotations

import json
import time
from pathlib import Path


def count_sector(sector: int) -> int:
    """How many TESS timeseries MAST lists for one sector. Zero means it is not published."""
    from astroquery.mast import Observations

    count = Observations.query_criteria_count(
        obs_collection="TESS",
        sequence_number=int(sector),
        dataproduct_type="timeseries",
    )
    return int(count or 0)


def discover_sector_max(count_fn=count_sector, *, high_cap: int = 200) -> int:
    """Highest sector number that still has timeseries, using a binary search.

    ``count_fn`` is injectable so tests never call MAST.
    """
    if count_fn(1) <= 0:
        raise RuntimeError("MAST returned no TESS timeseries for sector 1.")
    low = 1
    high = 1
    while high < high_cap and count_fn(high) > 0:
        low = high
        high = min(high_cap, high * 2 if high > 1 else 2)
        if high == low:
            break
    if count_fn(high) > 0:
        return high
    best = low
    left, right = low + 1, high - 1
    while left <= right:
        mid = (left + right) // 2
        if count_fn(mid) > 0:
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    return best


def cached_sector_max(root: Path, *, count_fn=count_sector, max_age: float = 12 * 3600, force: bool = False) -> int:
    """Remember the latest sector so the survey does not re-query MAST every star."""
    path = Path(root) / "mast_sectors.json"
    now = time.time()
    if path.is_file() and not force:
        try:
            saved = json.loads(path.read_text())
            age = now - float(saved.get("fetched_at") or 0)
            sector_max = int(saved.get("sector_max") or 0)
            if sector_max > 0 and age < max_age:
                return sector_max
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    sector_max = discover_sector_max(count_fn)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sector_max": sector_max, "fetched_at": now}))
    return sector_max


def seen_path(root: Path, sector: int) -> Path:
    return Path(root) / "mast_seen" / f"s{int(sector):03d}.txt"


def read_seen(root: Path, sector: int) -> set:
    path = seen_path(root, sector)
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def remember_seen(root: Path, sector: int, tic_ids: list) -> None:
    path = seen_path(root, sector)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_seen(root, sector)
    with path.open("a", encoding="utf-8") as handle:
        for tic in tic_ids:
            text = str(tic).strip()
            if not text or text in existing:
                continue
            handle.write(text + "\n")
            existing.add(text)


def next_unseen_batch(sector: int, limit: int, seen: set, survey_pass: int = 1) -> list:
    """Best stars not yet searched in this sector.

    Later passes pull a deeper MAST slice, so a finished best-N does not
    repeat the same stars forever.
    """
    from hunter import _attach_toi_dispositions, _fetch_sector_records
    from pipeline.target_rank import rank_targets
    from pipeline.tic_enrich import enrich_records

    pull = max(int(limit) * 20 * max(int(survey_pass), 1), int(limit), 1)
    records = _fetch_sector_records(int(sector), cap=pull)
    if not records:
        return []
    records = _attach_toi_dispositions(records)
    enrich_records(records)
    ranked = rank_targets(
        records,
        limit=max(len(records), 1),
        period_min=0.5,
        period_max=15.0,
    )
    fresh = []
    for record in ranked.search_order():
        tic = record.tic_norm
        if not tic or tic in seen:
            continue
        fresh.append(tic)
        if len(fresh) >= int(limit):
            break
    return fresh
