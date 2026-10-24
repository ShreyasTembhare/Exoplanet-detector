"""
Compact plot data saved with each candidate so the desk can draw evidence
panels without re-downloading anything: the raw (detrended) light curve with
transit times, the BLS periodogram, a binned phase fold, odd vs even transits
and the phase-0.5 window where a secondary eclipse would sit.

Everything is binned or max-pooled to a few thousand numbers per candidate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

MAX_LC_POINTS = 2500
MAX_PERIODOGRAM_POINTS = 1500
PHASE_BINS = 200
WINDOW_BINS = 80


def _finite_pairs(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _bin_mean(x, y, edges):
    idx = np.digitize(x, edges) - 1
    n = len(edges) - 1
    sums = np.bincount(idx[(idx >= 0) & (idx < n)], weights=y[(idx >= 0) & (idx < n)], minlength=n)
    counts = np.bincount(idx[(idx >= 0) & (idx < n)], minlength=n)
    centers = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    keep = counts > 0
    return centers[keep], means[keep]


def _round(values, digits=6):
    return [round(float(v), digits) for v in values]


def light_curve(time, flux, max_points: int = MAX_LC_POINTS) -> dict:
    t, f = _finite_pairs(time, flux)
    if len(t) == 0:
        return {"time": [], "flux": []}
    if len(t) > max_points:
        order = np.argsort(t)
        t, f = t[order], f[order]
        chunks = np.array_split(np.arange(len(t)), max_points)
        t = np.array([t[c].mean() for c in chunks if len(c)])
        f = np.array([f[c].mean() for c in chunks if len(c)])
    return {"time": _round(t, 5), "flux": _round(f)}


def periodogram(periods, power, max_points: int = MAX_PERIODOGRAM_POINTS) -> dict:
    p, w = _finite_pairs(periods, power)
    if len(p) == 0:
        return {"period": [], "power": []}
    order = np.argsort(p)
    p, w = p[order], w[order]
    if len(p) > max_points:
        # Max-pool so narrow peaks survive downsampling.
        chunks = np.array_split(np.arange(len(p)), max_points)
        best = [c[np.argmax(w[c])] for c in chunks if len(c)]
        p, w = p[best], w[best]
    return {"period": _round(p, 6), "power": _round(w, 8)}


def phase_fold(time, flux, period: float, epoch: float, bins: int = PHASE_BINS) -> dict:
    t, f = _finite_pairs(time, flux)
    if len(t) == 0 or not (period and np.isfinite(period) and period > 0):
        return {"phase": [], "flux": []}
    phase = ((t - epoch) / period + 0.5) % 1.0 - 0.5
    x, y = _bin_mean(phase, f, np.linspace(-0.5, 0.5, bins + 1))
    return {"phase": _round(x), "flux": _round(y)}


def odd_even(time, flux, period: float, epoch: float, duration_days: float, bins: int = WINDOW_BINS) -> dict:
    """Binned flux around odd and even transits, in hours from mid-transit."""
    t, f = _finite_pairs(time, flux)
    if len(t) == 0 or not (period and np.isfinite(period) and period > 0):
        return {}
    window = max(3.0 * float(duration_days or 0.1), 0.1)
    cycle = np.round((t - epoch) / period)
    dt = t - (epoch + cycle * period)
    near = np.abs(dt) <= window
    edges = np.linspace(-window, window, bins + 1) * 24.0
    out = {}
    for name, parity in (("odd", 1), ("even", 0)):
        sel = near & ((cycle.astype(int) % 2) == parity)
        x, y = _bin_mean(dt[sel] * 24.0, f[sel], edges)
        out[name] = {"hours": _round(x, 3), "flux": _round(y)}
    return out


def secondary_window(time, flux, period: float, epoch: float, duration_days: float,
                     bins: int = WINDOW_BINS) -> dict:
    """Binned flux around phase 0.5 (hours from the expected secondary)."""
    t, f = _finite_pairs(time, flux)
    if len(t) == 0 or not (period and np.isfinite(period) and period > 0):
        return {}
    window = max(3.0 * float(duration_days or 0.1), 0.1)
    centre = epoch + 0.5 * period
    cycle = np.round((t - centre) / period)
    dt = t - (centre + cycle * period)
    sel = np.abs(dt) <= window
    x, y = _bin_mean(dt[sel] * 24.0, f[sel], np.linspace(-window, window, bins + 1) * 24.0)
    return {"hours": _round(x, 3), "flux": _round(y)}


def transit_times(time, period: float, epoch: float) -> list:
    t = np.asarray(time, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) == 0 or not (period and np.isfinite(period) and period > 0):
        return [float(epoch)] if epoch is not None and np.isfinite(epoch) else []
    first = np.ceil((t.min() - epoch) / period)
    last = np.floor((t.max() - epoch) / period)
    if last - first > 500:
        return []
    return _round(epoch + np.arange(first, last + 1) * period, 5)


def evidence_curves(
    time, flux, period: float, epoch: float, duration_days: Optional[float] = None,
    periods=None, power=None,
) -> dict:
    duration = float(duration_days) if duration_days and np.isfinite(duration_days) else 0.1
    periodic = bool(period and np.isfinite(period) and period > 0)
    out = {
        "light_curve": {**light_curve(time, flux), "transits": transit_times(time, period, epoch)},
        "periodogram": periodogram(periods, power) if periods is not None and power is not None
        else {"period": [], "power": []},
    }
    if periodic:
        out["phase_curve"] = phase_fold(time, flux, period, epoch)
        out["odd_even_curve"] = odd_even(time, flux, period, epoch, duration)
        out["secondary_curve"] = secondary_window(time, flux, period, epoch, duration)
    return out
