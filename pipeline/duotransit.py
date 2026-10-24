"""
Duotransit search: two single transits, usually in sectors years apart.

Long-period planets (P > ~30 d) often show one transit per TESS sector, and
BLS cannot fold two events separated by a two-year gap into a peak. Here the
two strongest box events are found, checked for consistent depth and
duration, and the allowed periods are the aliases ``P_n = dt / n``. An alias
is ruled out when it predicts a transit inside observed data where no dip is
seen. The survivors are what a follow-up observer needs (cf. Hawthorn et al.
2024 on TESS duotransits).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .single_event import DEFAULT_MES_THRESHOLD, _robust_sigma, monotransit_durations, single_event_statistic

MAX_ALIASES = 200


def _mask_event(time, flux, epoch, duration_days, pad=1.5):
    f = np.asarray(flux, dtype=float).copy()
    f[np.abs(np.asarray(time) - epoch) <= pad * duration_days] = np.nan
    return f


def _depth_at(time, flux, centre, duration_days, baseline, min_points=3) -> Optional[float]:
    sel = np.abs(time - centre) <= 0.5 * duration_days
    if sel.sum() < min_points:
        return None
    return float(baseline - np.nanmean(flux[sel]))


def surviving_aliases(
    time, flux, epoch1: float, epoch2: float, duration_days: float, depth: float,
    min_period: float, sigma: Optional[float] = None, max_aliases: int = MAX_ALIASES,
) -> List[dict]:
    """Aliases ``dt/n`` not contradicted by observed flat data."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ok = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[ok], flux[ok]
    baseline = float(np.median(flux))
    sigma = sigma if sigma is not None else _robust_sigma(flux)
    dt = abs(epoch2 - epoch1)
    first = min(epoch1, epoch2)
    out = []
    n_max = int(min(max_aliases, np.floor(dt / max(min_period, 1e-3))))
    for n in range(1, n_max + 1):
        period = dt / n
        k_lo = int(np.floor((time.min() - first) / period)) - 1
        k_hi = int(np.ceil((time.max() - first) / period)) + 1
        contradicted = False
        covered = 0
        for k in range(k_lo, k_hi + 1):
            if k in (0, n):
                continue
            centre = first + k * period
            d = _depth_at(time, flux, centre, duration_days, baseline)
            if d is None:
                continue
            covered += 1
            n_in = max(int(np.sum(np.abs(time - centre) <= 0.5 * duration_days)), 1)
            # A real transit there would give `depth`; reject when the data sit
            # well above half of it.
            if d < 0.5 * depth and (0.5 * depth - d) > 2.0 * sigma / np.sqrt(n_in):
                contradicted = True
                break
        if not contradicted:
            out.append({"n": n, "period": float(period), "covered_epochs": covered})
    return out


def find_duotransit(
    time, flux, *,
    min_separation_days: float = 20.0,
    min_alias_period: float = 15.0,
    rho_star_solar: float = 1.0,
    mes_threshold: float = DEFAULT_MES_THRESHOLD,
    depth_ratio_max: float = 2.0,
) -> Optional[dict]:
    """Return the duotransit dict (epochs, depth, duration, aliases) or None."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    durations = list(monotransit_durations(min_alias_period, 1000.0, rho_star_solar))
    first = single_event_statistic(time, flux, durations)
    if first["mes"] < mes_threshold or not np.isfinite(first["epoch"]):
        return None
    masked = _mask_event(time, flux, first["epoch"], first["duration_days"])
    second = single_event_statistic(time, masked, [first["duration_days"]])
    if second["mes"] < 0.6 * mes_threshold or not np.isfinite(second["epoch"]):
        return None
    if abs(second["epoch"] - first["epoch"]) < min_separation_days:
        return None
    d1, d2 = first["depth"], second["depth"]
    if not (d1 > 0 and d2 > 0) or max(d1, d2) / min(d1, d2) > depth_ratio_max:
        return None
    depth = 0.5 * (d1 + d2)
    duration = first["duration_days"]
    aliases = surviving_aliases(
        time, flux, first["epoch"], second["epoch"], duration, depth, min_period=min_alias_period,
    )
    if not aliases:
        return None
    epochs = sorted([float(first["epoch"]), float(second["epoch"])])
    return {
        "kind": "duotransit",
        "epoch": epochs[0],
        "epochs": epochs,
        "separation_days": float(epochs[1] - epochs[0]),
        "depth": float(depth),
        "duration_days": float(duration),
        "mes": float(min(first["mes"], second["mes"])),
        "mes_events": [float(first["mes"]), float(second["mes"])],
        "aliases": aliases[:50],
        "n_aliases": len(aliases),
        "n_in": int(first["n_in"]),
    }
