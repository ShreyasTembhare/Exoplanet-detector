"""
Injection-recovery completeness maps.

For each (period, planet radius, Tmag) cell, transits of a Sun-like star are
injected into simulated single-sector TESS noise (white noise at the CDPP
expected for that Tmag), flattened with the same biweight detrender and
searched with BLS on an Ofir grid. The recovered fraction (period within 1%
or a 2:1 alias, SDE >= 7) is the completeness.

The map answers "what could this pipeline have found here?" and feeds back
into target selection: :func:`detection_probability` translates any star
into the single-sector Sun-like frame (planet radius scaled by R*, Tmag
brightened by 2.5 log10(n_sectors) for the extra transits), so the
expected-yield planner uses measured recovery instead of an analytic curve.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PERIODS = (0.7, 1.5, 3.0, 5.0, 8.0, 12.0)
DEFAULT_RADII = (1.0, 1.5, 2.0, 3.0, 4.0, 8.0)
DEFAULT_TMAGS = (8.0, 10.0, 12.0, 14.0)
CADENCE_DAYS = 10.0 / 1440.0
SECTOR_DAYS = 27.0
R_EARTH_OVER_R_SUN = 0.0091577
MAP_NAME = "completeness.json"


def _simulate(rng, period, radius_earth, tmag, cadence=CADENCE_DAYS, baseline=SECTOR_DAYS):
    from pipeline.target_rank import cdpp_ppm_from_tmag

    time_arr = np.arange(0.0, baseline, cadence)
    # Mid-sector downlink gap, as in real TESS sectors.
    gap = (time_arr > baseline / 2 - 0.5) & (time_arr < baseline / 2 + 0.5)
    time_arr = time_arr[~gap]
    sigma = cdpp_ppm_from_tmag(tmag) * 1e-6 * np.sqrt(1.0 / 24.0 / cadence)
    flux = 1.0 + rng.normal(0.0, sigma, time_arr.size)
    flux *= 1.0 + 0.002 * np.sin(2 * np.pi * time_arr / rng.uniform(3.0, 12.0))
    depth = (radius_earth * R_EARTH_OVER_R_SUN) ** 2
    duration = 13.0 / 24.0 * (period / 365.25) ** (1 / 3)
    epoch = rng.uniform(0, period)
    phase = (time_arr - epoch + 0.5 * period) % period - 0.5 * period
    flux[np.abs(phase) < duration / 2] *= 1.0 - depth
    return time_arr, flux, duration


def _recovered(time_arr, flux, period, duration, sde_min=7.0, n_periods=3000) -> bool:
    from astropy.timeseries import BoxLeastSquares

    from pipeline.detrend import biweight_detrend, window_for_duration
    from pipeline.period_grid import ofir_periods

    flat, _ = biweight_detrend(time_arr, flux, window_days=window_for_duration(duration))
    ok = np.isfinite(flat)
    periods = ofir_periods(0.5, min(15.0, np.ptp(time_arr) / 2.0), n_periods)
    durations = np.clip(np.array([0.6, 1.0, 1.5]) * duration, 0.02, 0.4)
    res = BoxLeastSquares(time_arr[ok], flat[ok]).power(periods, durations)
    power = np.asarray(res.power)
    i = int(np.argmax(power))
    sde = (power[i] - power.mean()) / (power.std() or 1.0)
    best = float(res.period[i])
    match = any(abs(best / (period * k) - 1) < 0.01 for k in (0.5, 1.0, 2.0))
    return bool(match and sde >= sde_min)


def run_injection_recovery(
    periods=DEFAULT_PERIODS, radii=DEFAULT_RADII, tmags=DEFAULT_TMAGS,
    n_per_cell: int = 3, seed: int = 0, budget_s: Optional[float] = None,
) -> dict:
    """Recovery fraction per cell, shaped ``[tmag][period][radius]``.

    ``budget_s`` stops early (unfinished cells stay null) so a worker can
    refresh the map inside a fixed time slice.
    """
    rng = np.random.default_rng(seed)
    start = time.time()
    grid = [[[None for _ in radii] for _ in periods] for _ in tmags]
    trials = 0
    for ti, tmag in enumerate(tmags):
        for pi, period in enumerate(periods):
            for ri, radius in enumerate(radii):
                if budget_s is not None and time.time() - start > budget_s:
                    break
                hits = 0
                for _ in range(n_per_cell):
                    t, f, dur = _simulate(rng, period, radius, tmag)
                    hits += _recovered(t, f, period, dur)
                    trials += 1
                grid[ti][pi][ri] = hits / n_per_cell
    return {
        "periods": list(map(float, periods)), "radii": list(map(float, radii)),
        "tmags": list(map(float, tmags)), "recovery": grid, "n_per_cell": n_per_cell,
        "trials": trials, "seconds": round(time.time() - start, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frame": "single sector, Sun-like star, 10-min cadence",
    }


def save_map(data: dict, path: str | Path) -> Path:
    from pipeline.atomic import atomic_write_json

    atomic_write_json(path, data)
    return Path(path)


def load_map(path: Optional[str | Path] = None) -> Optional[dict]:
    import json

    path = Path(path) if path else default_path()
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def default_path() -> Optional[Path]:
    env = os.environ.get("HUNT_COMPLETENESS_MAP")
    if env:
        return Path(env)
    data = os.environ.get("DESK_DATA")
    return Path(data) / MAP_NAME if data else None


def map_age_hours(path: str | Path) -> float:
    data = load_map(path)
    if not data or not data.get("generated_at"):
        return float("inf")
    try:
        made = datetime.fromisoformat(data["generated_at"])
    except ValueError:
        return float("inf")
    return (datetime.now(timezone.utc) - made).total_seconds() / 3600.0


def refresh_if_stale(path: str | Path, max_age_hours: float = 24.0,
                     budget_s: float = 300.0, **kwargs) -> bool:
    """Recompute the map when missing or older than ``max_age_hours``."""
    if max_age_hours <= 0 or map_age_hours(path) < max_age_hours:
        return False
    logger.info("Refreshing completeness map at %s (budget %.0fs)", path, budget_s)
    save_map(run_injection_recovery(budget_s=budget_s, **kwargs), path)
    return True


def _interp_axis(axis, value, log=True):
    axis = np.asarray(axis, dtype=float)
    x = np.log(axis) if log else axis
    v = np.log(max(value, 1e-6)) if log else value
    v = float(np.clip(v, x[0], x[-1]))
    j = int(np.clip(np.searchsorted(x, v) - 1, 0, len(x) - 2)) if len(x) > 1 else 0
    w = 0.0 if len(x) == 1 else (v - x[j]) / (x[j + 1] - x[j])
    return j, w


def lookup(data: dict, period: float, radius_earth: float, tmag: float) -> Optional[float]:
    """Trilinear interpolation of the recovery fraction; None if cells are empty."""
    cached = data.get("_array")
    if cached is not None and cached[0] is data["recovery"]:
        rec = cached[1]
    else:
        rec = np.array(
            [[[np.nan if c is None else c for c in row] for row in plane] for plane in data["recovery"]],
            dtype=float,
        )
        data["_array"] = (data["recovery"], rec)
    ti, tw = _interp_axis(data["tmags"], tmag, log=False)
    pi, pw = _interp_axis(data["periods"], period)
    ri, rw = _interp_axis(data["radii"], radius_earth)
    total, weight = 0.0, 0.0
    for dt, wt in ((0, 1 - tw), (1, tw)):
        for dp, wp in ((0, 1 - pw), (1, pw)):
            for dr, wr in ((0, 1 - rw), (1, rw)):
                w = wt * wp * wr
                t_i = min(ti + dt, rec.shape[0] - 1)
                p_i = min(pi + dp, rec.shape[1] - 1)
                r_i = min(ri + dr, rec.shape[2] - 1)
                val = rec[t_i, p_i, r_i]
                if w > 0 and np.isfinite(val):
                    total += w * val
                    weight += w
    return float(total / weight) if weight > 0 else None


def detection_probability(data: dict, period: float, radius_earth: float, tmag: float,
                          stellar_radius: float = 1.0, n_sectors: float = 1.0) -> Optional[float]:
    """Map lookup after translating the star into the map's frame."""
    r_eff = radius_earth / max(stellar_radius, 0.05)
    tmag_eff = tmag - 2.5 * np.log10(max(n_sectors, 1.0))
    return lookup(data, period, r_eff, tmag_eff)


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run injection-recovery and write a completeness map")
    parser.add_argument("--out", default=MAP_NAME)
    parser.add_argument("--n-per-cell", type=int, default=3)
    parser.add_argument("--budget-seconds", type=float, default=None)
    args = parser.parse_args(argv)
    data = run_injection_recovery(n_per_cell=args.n_per_cell, budget_s=args.budget_seconds)
    save_map(data, args.out)
    print(f"wrote {args.out}: {data['trials']} trials in {data['seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
