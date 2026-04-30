"""
Phase 3: Fold and feature vectors (global + local + local-centered views,
periodogram view, centroid + difference-image vetting).

Output for the ResNet path:
  * global_view: 2048 bins of the period-folded flux.
  * local_view : 256 bins zoomed into the transit segment.

Output for the ExoMiner++ path (later phases):
  * local_centered: 128 bins centered exactly on the transit (depth normalized).
  * periodogram_view: log-period vs power, resampled to 1024 bins.
  * scalar features (depth, duration, SDE, odd/even, secondary, V/U, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np

from .vetting import VettingMetrics, compute_vetting_metrics

logger = logging.getLogger(__name__)

GLOBAL_BINS = 2048
LOCAL_BINS = 256
LOCAL_CENTERED_BINS = 128
PERIODOGRAM_BINS = 1024


def fold_phase(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    return ((np.asarray(time, dtype=np.float64) - epoch) % period) / period


def bin_phase_flux(phase: np.ndarray, flux: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin (phase, flux) into ``n_bins`` over [0, 1). Empty bins get linearly
    interpolated from neighbors so a missed transit dip doesn't get filled in
    with the global mean."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binned = np.full(n_bins, np.nan, dtype=np.float64)
    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        if np.any(mask):
            binned[i] = np.nanmean(flux[mask])
    if np.all(np.isnan(binned)):
        return np.full(n_bins, np.nanmean(flux), dtype=np.float64)

    finite = np.isfinite(binned)
    if not np.all(finite):
        idx = np.arange(n_bins)
        binned[~finite] = np.interp(idx[~finite], idx[finite], binned[finite])
    return binned


def transit_phase_window(phase: np.ndarray, flux: np.ndarray, n_bins: int = 64) -> Tuple[float, float]:
    coarse = bin_phase_flux(phase, flux, n_bins)
    min_idx = int(np.nanargmin(coarse))
    center = (min_idx + 0.5) / n_bins
    return center, 0.05


def global_view(time: np.ndarray, flux: np.ndarray, period: float, epoch: float) -> np.ndarray:
    return bin_phase_flux(fold_phase(time, period, epoch), flux, GLOBAL_BINS)


def local_view(
    time: np.ndarray, flux: np.ndarray, period: float, epoch: float,
    transit_center: Optional[float] = None,
    transit_half_width: float = 0.05,
    n_bins: int = LOCAL_BINS,
) -> np.ndarray:
    phase = fold_phase(time, period, epoch)
    if transit_center is None:
        transit_center, transit_half_width = transit_phase_window(phase, flux)
    low, high = transit_center - transit_half_width, transit_center + transit_half_width
    if low < 0:
        mask = (phase >= (low + 1.0)) | (phase < high)
    elif high > 1:
        mask = (phase >= low) | (phase < (high - 1.0))
    else:
        mask = (phase >= low) & (phase < high)
    if np.sum(mask) < 10:
        return bin_phase_flux(phase, flux, n_bins)
    phase_seg = phase[mask].copy()
    flux_seg = flux[mask]
    if low < 0:
        phase_seg = np.where(phase_seg >= (low + 1.0), phase_seg - 1.0, phase_seg)
    if high > 1:
        phase_seg = np.where(phase_seg < (high - 1.0), phase_seg + 1.0, phase_seg)
    p_min, p_max = float(np.min(phase_seg)), float(np.max(phase_seg))
    if p_max > p_min:
        phase_norm = (phase_seg - p_min) / (p_max - p_min)
    else:
        phase_norm = np.zeros_like(phase_seg)
    return bin_phase_flux(phase_norm, flux_seg, n_bins)


def local_centered_view(
    time: np.ndarray, flux: np.ndarray, period: float, epoch: float,
    half_width: float = 0.04, n_bins: int = LOCAL_CENTERED_BINS,
) -> np.ndarray:
    """Always center on phase=0 (transit). For ExoMiner++ multi-input."""
    phase = fold_phase(time, period, epoch)
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    mask = np.abs(phase) < half_width
    if np.sum(mask) < 5:
        return np.zeros(n_bins, dtype=np.float64)
    p = phase[mask]
    f = flux[mask]
    p_norm = (p + half_width) / (2 * half_width)
    return bin_phase_flux(p_norm, f, n_bins)


def periodogram_view(
    periods: np.ndarray, power: np.ndarray, n_bins: int = PERIODOGRAM_BINS,
) -> np.ndarray:
    """Resample log-period vs power onto a fixed-length 1D vector."""
    if periods is None or len(periods) == 0:
        return np.zeros(n_bins, dtype=np.float64)
    finite = np.isfinite(periods) & np.isfinite(power) & (periods > 0)
    if np.sum(finite) < 8:
        return np.zeros(n_bins, dtype=np.float64)
    p = periods[finite]
    pw = power[finite]
    order = np.argsort(p)
    p, pw = p[order], pw[order]
    log_p = np.log(p)
    target = np.linspace(log_p.min(), log_p.max(), n_bins)
    return np.interp(target, log_p, pw)


def run_phase3(
    time: np.ndarray,
    flux: np.ndarray,
    best_period: float,
    epoch: float,
    tic_id: str,
    sector: str,
    tpf: Optional[Any] = None,
    use_cache: bool = True,
    *,
    bls_sde: float = float("nan"),
    bls_periods: Optional[np.ndarray] = None,
    bls_power: Optional[np.ndarray] = None,
    tpf_time: Optional[np.ndarray] = None,
    tpf_flux_cube: Optional[np.ndarray] = None,
    ra: float = float("nan"),
    dec: float = float("nan"),
    toi_table: Optional[object] = None,
) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    """
    Returns ``(global_view, local_view, centroid_offset, extras)`` where
    ``extras`` contains the rest of the multi-input features and the
    :class:`VettingMetrics` payload as a dict.
    """
    from .cache_io import get_phase3, set_phase3

    if use_cache:
        cached = get_phase3(tic_id, sector, best_period)
        if cached is not None:
            return cached  # may be 3-tuple from older caches

    phase = fold_phase(time, best_period, epoch)
    transit_center, transit_half_width = transit_phase_window(phase, flux)

    global_vec = global_view(time, flux, best_period, epoch)
    local_vec = local_view(
        time, flux, best_period, epoch,
        transit_center=transit_center,
        transit_half_width=transit_half_width,
    )
    centered_vec = local_centered_view(
        time, flux, best_period, epoch, half_width=transit_half_width,
    )
    periodogram_vec = periodogram_view(
        bls_periods if bls_periods is not None else np.array([]),
        bls_power if bls_power is not None else np.array([]),
    )

    # Centroid + difference image: prefer cached arrays from Phase 1.
    if (tpf_time is None or tpf_flux_cube is None) and tpf is not None:
        try:
            tpf_time = np.asarray(getattr(tpf.time, "value", tpf.time))
            tpf_flux_cube = np.asarray(tpf.flux, dtype=np.float32)
        except Exception:
            tpf_time, tpf_flux_cube = None, None

    metrics: VettingMetrics = compute_vetting_metrics(
        time, flux, best_period, epoch,
        bls_sde=bls_sde,
        tpf_time=tpf_time, tpf_flux_cube=tpf_flux_cube,
        ra=ra, dec=dec, tic_id=tic_id, toi_table=toi_table,
        half_width=transit_half_width,
    )

    extras = {
        "vetting": metrics.as_dict(),
        "local_centered": centered_vec,
        "periodogram_view": periodogram_vec,
        "transit_center": transit_center,
        "transit_half_width": transit_half_width,
    }
    centroid_offset = float(metrics.centroid_offset_px)

    if use_cache:
        set_phase3(
            tic_id, sector, best_period,
            global_vec, local_vec, centroid_offset, extras=extras,
        )

    return global_vec, local_vec, centroid_offset, extras
