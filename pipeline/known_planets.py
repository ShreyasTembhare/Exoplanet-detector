"""
Known-planet ephemerides and orbit-level matching.

A candidate is only "known" when its period (or a simple harmonic of it)
and, when both are available, its transit phase line up with a catalogued
TOI or confirmed planet on the same star. A new period on a known host is
reported as ``NEW_ON_KNOWN_HOST``; multi-planet systems are common, so those
are some of the most valuable candidates the hunt can produce.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .atomic import atomic_write_json, read_json

logger = logging.getLogger(__name__)

BTJD_OFFSET = 2457000.0
CACHE_PATH = Path("cache") / "known_planets.json"
CACHE_TTL_SECONDS = 24 * 3600
HARMONICS = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)

KnownTable = Dict[str, List[dict]]


def _norm_tic(tic) -> str:
    text = str(tic).strip().upper()
    if text.startswith("TIC"):
        text = text[3:].strip()
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        return text


def _num(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number


def to_btjd(epoch: float) -> float:
    """Catalog epochs are BJD; light curves are BTJD."""
    epoch = _num(epoch)
    if epoch != epoch:
        return epoch
    return epoch - BTJD_OFFSET if epoch > 2_000_000 else epoch


def build_known_table(rows) -> KnownTable:
    table: KnownTable = {}
    for row in rows:
        tic = _norm_tic(row.get("tid", row.get("tic_id", "")))
        if not tic:
            continue
        table.setdefault(tic, []).append({
            "period": _num(row.get("pl_orbper", row.get("period"))),
            "epoch_btjd": to_btjd(row.get("pl_tranmid", row.get("epoch"))),
            "toi": str(row.get("toi", "") or ""),
            "disposition": str(row.get("tfopwg_disp", row.get("disposition", "")) or "").strip().upper(),
        })
    return table


def load_known_ephemerides(cache_path: Path = CACHE_PATH, ttl: float = CACHE_TTL_SECONDS) -> KnownTable:
    """TOI table keyed by TIC, cached on disk for a day. Empty when offline."""
    cached = read_json(cache_path)
    if isinstance(cached, dict) and time.time() - float(cached.get("fetched_at", 0)) < ttl:
        return cached.get("table", {})
    try:
        from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

        toi = NasaExoplanetArchive.query_criteria(
            table="toi", select="tid,toi,tfopwg_disp,pl_orbper,pl_tranmid",
        )
        table = build_known_table(toi.to_pandas().to_dict(orient="records"))
    except Exception as exc:
        logger.warning("Known-planet catalog unavailable (%s).", exc)
        return cached.get("table", {}) if isinstance(cached, dict) else {}
    atomic_write_json(cache_path, {"fetched_at": time.time(), "table": table})
    return table


def match_orbit(
    period: float, epoch: Optional[float], known: List[dict],
    period_tol: float = 0.01, phase_tol: float = 0.05,
) -> Optional[dict]:
    """First catalogued orbit matching ``period`` (up to a harmonic) and phase."""
    period = _num(period)
    if not (period == period and period > 0):
        return None
    epoch = _num(epoch)
    for entry in known:
        p_known = _num(entry.get("period"))
        if not (p_known == p_known and p_known > 0):
            continue
        for harmonic in HARMONICS:
            if abs(period / p_known - harmonic) / harmonic >= period_tol:
                continue
            e_known = _num(entry.get("epoch_btjd"))
            if harmonic == 1.0 and epoch == epoch and e_known == e_known:
                cycles = (epoch - e_known) / p_known
                offset = abs(cycles - round(cycles))
                # Ephemeris drift grows with elapsed cycles; only reject on phase
                # when the catalog period is precise enough to predict it.
                if offset > phase_tol and abs(cycles) < 200:
                    continue
            return {**entry, "harmonic": harmonic}
    return None


def classify_candidate(tic_id, period, epoch, known_table: KnownTable) -> Tuple[str, Optional[dict]]:
    """``KNOWN``, ``NEW_ON_KNOWN_HOST`` or ``NEW_CANDIDATE`` plus the matched entry."""
    entries = known_table.get(_norm_tic(tic_id), []) if known_table else []
    if not entries:
        return "NEW_CANDIDATE", None
    match = match_orbit(period, epoch, entries)
    if match is not None:
        return "KNOWN", match
    return "NEW_ON_KNOWN_HOST", None
