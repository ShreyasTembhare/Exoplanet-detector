"""
Phase 3: Fold and feature vectors (global + local views, centroid vetting).
- Global view: fold on best period, bin into 2048 points.
- Local view: transit segment only, bin into 256 points (transit shape: U vs V).
- Centroid offset: in-transit vs out-of-transit centroid from TPF (scalar); NaN if no TPF.
"""

import logging
from typing import Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)

GLOBAL_BINS = 2048
LOCAL_BINS = 256


def fold_phase(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """Phase in [0, 1): (time - epoch) % period / period."""
    phase = ((time - epoch) % period) / period
    return np.asarray(phase, dtype=np.float64)


def bin_phase_flux(phase: np.ndarray, flux: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin (phase, flux) into n_bins over [0, 1). Returns binned flux (median or mean per bin)."""
    bins = np.linspace(0, 1, n_bins + 1)
    binned = np.full(n_bins, np.nan, dtype=np.float64)
    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        if np.sum(mask) > 0:
            binned[i] = np.nanmean(flux[mask])
        else:
            binned[i] = np.nanmean(flux)  # fallback
    # Fill any remaining NaN with global mean
    if np.any(np.isnan(binned)):
        binned[np.isnan(binned)] = np.nanmean(flux)
    return binned


def transit_phase_window(phase: np.ndarray, flux: np.ndarray, n_bins: int = 50) -> Tuple[float, float]:
    """
    Find transit window: phase range where the folded flux is lowest (dip).
    Returns (center_phase, half_width) so window = [center - half_width, center + half_width].
    """
    # Coarse bin to find minimum
    coarse = bin_phase_flux(phase, flux, n_bins)
    min_idx = np.nanargmin(coarse)
    center = (min_idx + 0.5) / n_bins
    # Use ~0.1 in phase as default half-width (transit duration)
    half_width = 0.05
    return center, half_width


def global_view(time: np.ndarray, flux: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """Fold and bin into 2048 points (global view)."""
    phase = fold_phase(time, period, epoch)
    return bin_phase_flux(phase, flux, GLOBAL_BINS)


def local_view(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    transit_center: Optional[float] = None,
    transit_half_width: float = 0.05,
) -> np.ndarray:
    """
    Extract transit segment and bin into 256 points (local view).
    If transit_center is None, find it from the folded curve.
    """
    phase = fold_phase(time, period, epoch)
    if transit_center is None:
        transit_center, transit_half_width = transit_phase_window(phase, flux)
    # Window [center - half_width, center + half_width] with wrap
    low, high = transit_center - transit_half_width, transit_center + transit_half_width
    if low < 0:
        mask = (phase >= (low + 1.0)) | (phase < high)
    elif high > 1:
        mask = (phase >= low) | (phase < (high - 1.0))
    else:
        mask = (phase >= low) & (phase < high)
    if np.sum(mask) < 10:
        # Fallback: use full phase and bin to LOCAL_BINS
        return bin_phase_flux(phase, flux, LOCAL_BINS)
    phase_seg = phase[mask]
    flux_seg = flux[mask]
    # Normalize phase segment to [0, 1) for binning
    if low < 0:
        phase_seg = np.where(phase_seg >= (low + 1.0), phase_seg - 1.0, phase_seg)
    if high > 1:
        phase_seg = np.where(phase_seg < (high - 1.0), phase_seg + 1.0, phase_seg)
    p_min, p_max = np.min(phase_seg), np.max(phase_seg)
    if p_max > p_min:
        phase_norm = (phase_seg - p_min) / (p_max - p_min)
    else:
        phase_norm = np.zeros_like(phase_seg)
    return bin_phase_flux(phase_norm, flux_seg, LOCAL_BINS)


def centroid_offset_from_tpf(
    tpf: Any,
    period: float,
    epoch: float,
    transit_center: float = 0.5,
    transit_half_width: float = 0.05,
) -> float:
    """
    Compute centroid offset: distance between in-transit and out-of-transit
    flux-weighted centroid from TPF. Returns offset in pixels (or normalized).
    If TPF unavailable or error, returns np.nan.
    """
    try:
        # tpf: lightkurve TargetPixelFile; need time, flux per pixel
        time = np.asarray(tpf.time.value) if hasattr(tpf.time, "value") else np.asarray(tpf.time)
        phase = fold_phase(time, period, epoch)
        in_transit = (phase >= (transit_center - transit_half_width)) & (
            phase < (transit_center + transit_half_width)
        )
        if transit_center + transit_half_width > 1:
            in_transit = in_transit | (phase < (transit_center + transit_half_width - 1.0))
        if transit_center - transit_half_width < 0:
            in_transit = in_transit | (phase >= (transit_center - transit_half_width + 1.0))
        n_in = np.sum(in_transit)
        n_out = np.sum(~in_transit)
        if n_in < 2 or n_out < 2:
            return np.nan
        # Flux-weighted centroid: (sum col * flux) / sum(flux), (sum row * flux) / sum(flux)
        flux_cube = np.asarray(tpf.flux)  # (n_cadence, row, col)
        n_cadence, n_row, n_col = flux_cube.shape
        cols = np.arange(n_col, dtype=np.float64)
        rows = np.arange(n_row, dtype=np.float64)
        flux_in = flux_cube[in_transit].sum(axis=0)  # (row, col)
        flux_out = flux_cube[~in_transit].sum(axis=0)
        sum_flux_in = flux_in.sum()
        sum_flux_out = flux_out.sum()
        if sum_flux_in <= 0 or sum_flux_out <= 0:
            return np.nan
        col_in = np.sum(cols[None, :] * flux_in) / sum_flux_in
        row_in = np.sum(rows[:, None] * flux_in) / sum_flux_in
        col_out = np.sum(cols[None, :] * flux_out) / sum_flux_out
        row_out = np.sum(rows[:, None] * flux_out) / sum_flux_out
        offset = np.sqrt((col_in - col_out) ** 2 + (row_in - row_out) ** 2)
        return float(offset)
    except Exception as e:
        logger.warning(f"Centroid from TPF failed: {e}")
        return np.nan


def run_phase3(
    time: np.ndarray,
    flux: np.ndarray,
    best_period: float,
    epoch: float,
    tic_id: str,
    sector: str,
    tpf: Optional[Any] = None,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run Phase 3: global view (2048), local view (256), centroid offset.
    Returns (global_view, local_view, centroid_offset).
    """
    from .cache_io import get_phase3, set_phase3

    if use_cache:
        cached = get_phase3(tic_id, sector, best_period)
        if cached is not None:
            return cached

    global_vec = global_view(time, flux, best_period, epoch)
    transit_center, transit_half_width = transit_phase_window(
        fold_phase(time, best_period, epoch), flux
    )
    local_vec = local_view(
        time, flux, best_period, epoch,
        transit_center=transit_center,
        transit_half_width=transit_half_width,
    )
    if tpf is not None:
        centroid_offset = centroid_offset_from_tpf(
            tpf, best_period, epoch,
            transit_center=transit_center,
            transit_half_width=transit_half_width,
        )
    else:
        centroid_offset = np.nan

    if use_cache:
        set_phase3(tic_id, sector, best_period, global_vec, local_vec, centroid_offset)

    return global_vec, local_vec, float(centroid_offset)
