"""
Robust detrending for transit search.

A time-windowed Tukey biweight (Hippke et al. 2019, "wotan") preserves
transit depth better than Savitzky-Golay or a median filter for the same
window and is the recommended default for TESS. When ``wotan`` is installed
it is used; otherwise a NumPy implementation evaluates the biweight location
on a coarse grid and interpolates, which is accurate to well under the
photometric noise for windows of hours to a day.

``masked_redetrend`` refits the trend with in-transit cadences excluded once
a candidate ephemeris is known, so the trend cannot eat part of the dip and
the measured depth (and SNR) are not biased low.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_DAYS = 0.75
BIWEIGHT_C = 5.0


def wotan_available() -> bool:
    try:
        import wotan  # noqa: F401
        return True
    except Exception:
        return False


def _biweight_location(values: np.ndarray, iterations: int = 5) -> float:
    if len(values) == 0:
        return float("nan")
    location = float(np.median(values))
    for _ in range(iterations):
        mad = float(np.median(np.abs(values - location)))
        if mad == 0:
            return location
        u = (values - location) / (BIWEIGHT_C * mad)
        w = (1 - u ** 2) ** 2
        w[np.abs(u) >= 1] = 0
        total = w.sum()
        if total == 0:
            return location
        new = float(np.sum(w * values) / total)
        if abs(new - location) < 1e-10:
            return new
        location = new
    return location


def biweight_trend(
    time: np.ndarray,
    flux: np.ndarray,
    window_days: float = DEFAULT_WINDOW_DAYS,
    mask: Optional[np.ndarray] = None,
    grid_step_days: Optional[float] = None,
) -> np.ndarray:
    """Biweight trend at every cadence. ``mask`` True = exclude from the fit."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    use = np.isfinite(time) & np.isfinite(flux)
    if mask is not None:
        use &= ~np.asarray(mask, dtype=bool)
    if use.sum() < 3:
        return np.full_like(flux, np.nanmedian(flux) if np.isfinite(flux).any() else 1.0)

    if wotan_available() and mask is None:
        try:
            from wotan import flatten

            _, trend = flatten(time, flux, method="biweight", window_length=window_days, return_trend=True)
            trend = np.asarray(trend, dtype=float)
            if np.isfinite(trend).sum() > 0.5 * len(trend):
                return _fill(time, trend)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.info("wotan failed (%s); using the NumPy biweight", exc)

    t_fit = time[use]
    f_fit = flux[use]
    order = np.argsort(t_fit)
    t_fit, f_fit = t_fit[order], f_fit[order]
    half = window_days / 2.0
    step = grid_step_days or max(window_days / 30.0, np.median(np.diff(t_fit)) if len(t_fit) > 1 else window_days)
    grid = np.arange(t_fit[0], t_fit[-1] + step, step)
    lo = np.searchsorted(t_fit, grid - half, side="left")
    hi = np.searchsorted(t_fit, grid + half, side="right")
    values = np.full(len(grid), np.nan)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b - a >= 3:
            values[i] = _biweight_location(f_fit[a:b])
    good = np.isfinite(values)
    if good.sum() < 2:
        return np.full_like(flux, np.median(f_fit))
    trend = np.interp(time, grid[good], values[good])
    return trend


def _fill(time: np.ndarray, trend: np.ndarray) -> np.ndarray:
    good = np.isfinite(trend) & np.isfinite(time)
    if good.all():
        return trend
    out = trend.copy()
    out[~good] = np.interp(time[~good], time[good], trend[good]) if good.sum() >= 2 else np.nanmedian(trend)
    return out


def biweight_detrend(time, flux, window_days: float = DEFAULT_WINDOW_DAYS) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(flattened_flux, trend)``."""
    trend = biweight_trend(time, flux, window_days=window_days)
    with np.errstate(invalid="ignore", divide="ignore"):
        flat = np.asarray(flux, dtype=float) / trend
    return flat, trend


def transit_mask(time, period: float, epoch: float, duration_days: float, factor: float = 1.5) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    if not (period and np.isfinite(period) and period > 0):
        return np.abs(time - epoch) < factor * duration_days / 2.0
    phase = (time - epoch + 0.5 * period) % period - 0.5 * period
    return np.abs(phase) < factor * duration_days / 2.0


def masked_redetrend(
    time, flat_flux, trend, period: float, epoch: float, duration_days: float,
    window_days: float = DEFAULT_WINDOW_DAYS,
) -> np.ndarray:
    """Re-flatten with the transits excluded from the trend fit.

    ``flat_flux * trend`` recovers the raw curve, so only the trend is refit.
    """
    raw = np.asarray(flat_flux, dtype=float) * np.asarray(trend, dtype=float)
    mask = transit_mask(time, period, epoch, duration_days)
    window = max(window_days, 3.0 * float(duration_days))
    new_trend = biweight_trend(time, raw, window_days=window, mask=mask)
    with np.errstate(invalid="ignore", divide="ignore"):
        return raw / new_trend


def window_for_duration(max_duration_days: float) -> float:
    """Hippke et al.: a window ~3x the longest transit sought keeps its depth."""
    return float(np.clip(3.0 * max_duration_days, 0.3, 2.0))
