"""
Single-event (monotransit) statistic on an unfolded light curve.

A TESS sector is ~27 days, so a 100-day period cannot show two transits.
This is a box MES: depth over the out-of-transit robust sigma, times
sqrt(n_in). It is the stand-in for a CETRA-style linear search. The
unfolded transformer is not on this path.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Kepler pipeline's multiple-event threshold is ~7.1; a single event uses
# the same scale so a quiet dip does not page a human.
DEFAULT_MES_THRESHOLD = 7.1


def _robust_sigma(flux: np.ndarray) -> float:
    med = float(np.median(flux))
    mad = float(np.median(np.abs(flux - med))) * 1.4826
    if mad > 0:
        return mad
    sigma = float(np.std(flux))
    return sigma if sigma > 0 else 1.0


def single_event_statistic(
    time: np.ndarray,
    flux: np.ndarray,
    durations_days: Sequence[float],
    *,
    min_samples: int = 3,
) -> dict:
    """Slide a box over the unfolded curve and return the strongest dip.

    The result dict has ``mes``, ``epoch`` (mid-event), ``duration_days``,
    ``depth`` (fractional), and ``n_in``.
    """
    t = np.asarray(time, dtype=np.float64)
    f = np.asarray(flux, dtype=np.float64)
    finite = np.isfinite(t) & np.isfinite(f)
    t, f = t[finite], f[finite]
    empty = {
        "mes": 0.0,
        "epoch": float("nan"),
        "duration_days": float("nan"),
        "depth": float("nan"),
        "n_in": 0,
    }
    if len(t) < min_samples * 2:
        return empty

    order = np.argsort(t)
    t, f = t[order], f[order]
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return empty

    baseline = float(np.median(f))
    sigma = _robust_sigma(f)
    best = dict(empty)
    cumsum = np.cumsum(f)

    for dur in durations_days:
        dur = float(dur)
        if not np.isfinite(dur) or dur <= 0:
            continue
        n_in = max(int(min_samples), int(round(dur / dt)))
        if n_in >= len(f) // 2 or n_in < min_samples:
            continue
        # Window sum ending at index k+n_in-1 equals cumsum[k+n_in-1] - cumsum[k-1].
        window_sum = cumsum[n_in - 1:] - np.concatenate(([0.0], cumsum[: len(f) - n_in]))
        window_mean = window_sum / n_in
        depth = baseline - window_mean
        mes = depth / sigma * np.sqrt(n_in)
        k = int(np.argmax(mes))
        if float(mes[k]) > float(best["mes"]):
            t_start = float(t[k])
            best = {
                "mes": float(mes[k]),
                "epoch": t_start + 0.5 * dur,
                "duration_days": dur,
                "depth": float(depth[k]),
                "n_in": int(n_in),
            }
    return best


def monotransit_durations(
    period_min: float,
    period_max: float,
    rho_star_solar: float = 1.0,
) -> np.ndarray:
    """A few physical transit lengths (days) inside the period window."""
    from .physical_priors import transit_duration_hours

    lo = max(float(period_min), 1e-3)
    hi = max(float(period_max), lo * 1.01)
    periods = np.geomspace(lo, hi, 4)
    hours = []
    for period in periods:
        base = transit_duration_hours(float(period), rho_star_solar)
        hours.extend([base * 0.5, base, base * 1.5])
    days = np.array([h / 24.0 for h in hours if h > 0], dtype=np.float64)
    if len(days) == 0:
        return np.array([0.05, 0.1, 0.2], dtype=np.float64)
    # Unique after rounding so the slide is not repeated.
    rounded = np.unique(np.round(days, 5))
    return rounded


def looks_like_monotransit(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period_min: float = 1.0,
    period_max: float = 100.0,
    rho_star_solar: float = 1.0,
    mes_threshold: float = DEFAULT_MES_THRESHOLD,
    durations_days: Optional[Sequence[float]] = None,
) -> Optional[dict]:
    """Return the single-event dict when MES clears the threshold, else None."""
    durations = (
        list(durations_days)
        if durations_days is not None
        else list(monotransit_durations(period_min, period_max, rho_star_solar))
    )
    event = single_event_statistic(time, flux, durations)
    if event["mes"] >= mes_threshold and event["n_in"] >= 3:
        event["kind"] = "monotransit"
        return event
    return None
