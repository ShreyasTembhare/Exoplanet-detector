"""
Phase 3 vetting metrics. Computes the standard transit-vetting suite that the
classifier downstream consumes as scalar features and that humans use to decide
which candidates are worth follow-up.

Metrics included:
  * SDE (Signal Detection Efficiency from the BLS power spectrum).
  * Transit depth and (best-effort) duration.
  * Number of observed transits in the LC.
  * Odd-vs-even transit-depth ratio (eclipsing-binary indicator).
  * Secondary-eclipse depth at phase=0.5 (EB indicator).
  * V-shape vs U-shape metric (trapezoid fit chi^2 ratio).
  * Centroid in/out-of-transit offset and difference image (if TPF arrays
    provided).
  * Gaia DR3 nearby-source contamination count (if astroquery available).
  * Ephemeris match against the TOI catalog (if a TOI table is provided).

All metrics are robust to NaNs/missing data and degrade gracefully when their
inputs are absent.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VettingMetrics:
    sde: float = float("nan")
    depth: float = float("nan")
    duration_phase: float = float("nan")
    duration_hours: float = float("nan")
    n_transits: int = 0
    odd_even_ratio: float = float("nan")
    secondary_depth: float = float("nan")
    secondary_significance: float = float("nan")
    v_shape_score: float = float("nan")
    centroid_offset_px: float = float("nan")
    diff_image_peak_px: float = float("nan")
    gaia_neighbors_within_30arcsec: int = -1
    matches_toi: bool = False
    matched_toi_id: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def fold_phase(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    return ((np.asarray(time, dtype=np.float64) - epoch) % period) / period


def _phase_in_transit(phase: np.ndarray, center: float, half_width: float) -> np.ndarray:
    low, high = center - half_width, center + half_width
    if low < 0:
        return (phase >= low + 1.0) | (phase < high)
    if high > 1.0:
        return (phase >= low) | (phase < high - 1.0)
    return (phase >= low) & (phase < high)


def transit_depth_and_duration(
    time: np.ndarray, flux: np.ndarray, period: float, epoch: float,
    base_half_width: float = 0.05,
) -> Tuple[float, float]:
    """Return (depth, duration_phase). Depth is in normalized flux."""
    phase = fold_phase(time, period, epoch)
    half_widths = np.linspace(0.005, base_half_width * 2, 16)
    out_mask = ~_phase_in_transit(phase, 0.0, base_half_width * 2)
    if np.sum(out_mask) < 10:
        return float("nan"), float("nan")
    out_level = float(np.nanmedian(flux[out_mask]))
    best_depth = 0.0
    best_dur = float(half_widths[0]) * 2
    for hw in half_widths:
        in_mask = _phase_in_transit(phase, 0.0, hw)
        if np.sum(in_mask) < 3:
            continue
        in_level = float(np.nanmedian(flux[in_mask]))
        depth = out_level - in_level
        if depth > best_depth:
            best_depth = depth
            best_dur = float(hw) * 2
    return float(max(best_depth, 0.0)), float(best_dur)


def count_transits(time: np.ndarray, period: float, epoch: float) -> int:
    """How many transit *events* fall within the time series."""
    t = np.asarray(time, dtype=np.float64)
    if len(t) == 0 or period <= 0:
        return 0
    n_min = int(np.floor((np.nanmin(t) - epoch) / period))
    n_max = int(np.ceil((np.nanmax(t) - epoch) / period))
    return max(0, n_max - n_min)


def odd_even_depth_ratio(
    time: np.ndarray, flux: np.ndarray, period: float, epoch: float,
    half_width: float = 0.02,
) -> float:
    """
    Eclipsing binaries often have alternating odd / even depths. Return
    ``min(d_odd, d_even) / max(d_odd, d_even)``. Values close to 1 are
    consistent with a planet.
    """
    t = np.asarray(time, dtype=np.float64)
    if len(t) == 0 or period <= 0:
        return float("nan")
    n = np.floor((t - epoch) / period).astype(int)
    even_mask = (n % 2 == 0)

    folded = fold_phase(time, period, epoch)
    folded[folded > 0.5] -= 1.0

    in_transit = np.abs(folded) < half_width
    out_transit = (np.abs(folded) > half_width * 2) & (np.abs(folded) < 0.25)

    def depth(mask):
        in_m = in_transit & mask
        out_m = out_transit & mask
        if np.sum(in_m) < 3 or np.sum(out_m) < 5:
            return float("nan")
        return float(np.nanmedian(flux[out_m]) - np.nanmedian(flux[in_m]))

    d_even = depth(even_mask)
    d_odd = depth(~even_mask)
    if not (np.isfinite(d_even) and np.isfinite(d_odd)):
        return float("nan")
    if d_even <= 0 or d_odd <= 0:
        return float("nan")
    return float(min(d_even, d_odd) / max(d_even, d_odd))


def secondary_eclipse(
    time: np.ndarray, flux: np.ndarray, period: float, epoch: float,
    half_width: float = 0.02,
) -> Tuple[float, float]:
    """
    Search for a secondary eclipse near phase=0.5. Returns
    ``(depth, significance_in_sigma)``. Significance > ~3 is suspicious
    (likely an EB).
    """
    phase = fold_phase(time, period, epoch)
    sec_mask = _phase_in_transit(phase, 0.5, half_width)
    base_mask = _phase_in_transit(phase, 0.25, half_width) | _phase_in_transit(phase, 0.75, half_width)
    if np.sum(sec_mask) < 3 or np.sum(base_mask) < 5:
        return float("nan"), float("nan")
    base_level = float(np.nanmedian(flux[base_mask]))
    sec_level = float(np.nanmedian(flux[sec_mask]))
    sec_depth = base_level - sec_level
    base_sigma = float(np.nanstd(flux[base_mask])) or 1e-6
    significance = sec_depth / (base_sigma / np.sqrt(max(np.sum(sec_mask), 1)))
    return float(max(sec_depth, 0.0)), float(significance)


def v_shape_score(
    time: np.ndarray, flux: np.ndarray, period: float, epoch: float,
    half_width: float = 0.05,
) -> float:
    """
    Fit a trapezoid (U-shape) and a V-shape to the in-transit binned flux.
    Return ``chi2_v / chi2_trapezoid``. Values < 1 favor a V-shape (grazing
    eclipsing binary), > 1 favor a U-shape (planet).
    """
    phase = fold_phase(time, period, epoch)
    folded = phase.copy()
    folded[folded > 0.5] -= 1.0
    in_mask = np.abs(folded) < half_width
    if np.sum(in_mask) < 20:
        return float("nan")
    x = folded[in_mask]
    y = np.asarray(flux)[in_mask]
    bins = np.linspace(-half_width, half_width, 33)
    centers = 0.5 * (bins[:-1] + bins[1:])
    bin_y = np.array([
        np.nanmedian(y[(x >= bins[i]) & (x < bins[i + 1])])
        if np.any((x >= bins[i]) & (x < bins[i + 1])) else np.nan
        for i in range(len(bins) - 1)
    ])
    finite = np.isfinite(bin_y)
    if np.sum(finite) < 12:
        return float("nan")
    bx = centers[finite]
    by = bin_y[finite]
    base = float(np.nanmax(by))
    depth = base - float(np.nanmin(by))
    if depth <= 0:
        return float("nan")
    # V-shape model: linear from edges to minimum
    v_model = base - depth * (1 - np.abs(bx) / half_width)
    chi2_v = float(np.nansum((by - v_model) ** 2))
    # Trapezoidal model: flat bottom for inner half, linear ramp on edges
    inner = np.abs(bx) < half_width * 0.4
    trap_model = np.where(inner, base - depth, base - depth * (1 - (np.abs(bx) - half_width * 0.4) / (half_width * 0.6)))
    trap_model = np.where(np.abs(bx) > half_width, base, trap_model)
    chi2_t = float(np.nansum((by - trap_model) ** 2)) or 1e-12
    return float(chi2_v / chi2_t)


def centroid_metrics_from_tpf_arrays(
    tpf_time: np.ndarray, flux_cube: np.ndarray,
    period: float, epoch: float, half_width: float = 0.05,
) -> Tuple[float, float]:
    """
    Returns (centroid_offset_px, diff_image_peak_px). Both NaN if not enough
    in/out cadences.
    """
    if flux_cube is None or flux_cube.size == 0:
        return float("nan"), float("nan")
    phase = fold_phase(tpf_time, period, epoch)
    in_mask = _phase_in_transit(phase, 0.0, half_width)
    out_mask = ~in_mask
    if np.sum(in_mask) < 3 or np.sum(out_mask) < 10:
        return float("nan"), float("nan")
    flux_in = np.nanmean(flux_cube[in_mask], axis=0)
    flux_out = np.nanmean(flux_cube[out_mask], axis=0)
    diff = flux_out - flux_in
    n_row, n_col = flux_in.shape[-2:]
    rows = np.arange(n_row, dtype=np.float64)
    cols = np.arange(n_col, dtype=np.float64)

    def centroid(img):
        s = float(np.nansum(img))
        if s <= 0:
            return float("nan"), float("nan")
        col_c = float(np.nansum(cols[None, :] * img) / s)
        row_c = float(np.nansum(rows[:, None] * img) / s)
        return row_c, col_c

    rin, cin = centroid(flux_in)
    rout, cout = centroid(flux_out)
    if not (np.isfinite(rin) and np.isfinite(rout)):
        return float("nan"), float("nan")
    offset = float(np.hypot(rin - rout, cin - cout))

    diff_pos = np.where(diff > 0, diff, 0.0)
    rd, cd = centroid(diff_pos)
    if np.isfinite(rd) and np.isfinite(cd):
        diff_peak = float(np.hypot(rd - rout, cd - cout))
    else:
        diff_peak = float("nan")
    return offset, diff_peak


def gaia_neighbors_within(arcsec: float, ra: float, dec: float) -> int:  # pragma: no cover
    try:
        from astroquery.gaia import Gaia
    except Exception:
        return -1
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        coord = SkyCoord(ra, dec, unit="deg")
        radius = arcsec * u.arcsec
        job = Gaia.cone_search_async(coordinate=coord, radius=radius)
        result = job.get_results()
        return max(0, len(result) - 1)  # exclude target itself
    except Exception as exc:
        logger.warning("Gaia query failed: %s", exc)
        return -1


def match_toi(
    tic_id: str, period: float, toi_table: Optional[object] = None,
    period_tol: float = 0.01,
) -> Tuple[bool, str]:
    """Match (TIC, period) against a TOI catalog row."""
    if toi_table is None:
        return False, ""
    try:
        norm = str(tic_id).strip().upper().replace("TIC", "").strip()
        for row in toi_table:
            tid = str(row.get("tid", row.get("TID", ""))).strip()
            if tid != norm:
                continue
            P = float(row.get("pl_orbper", row.get("PL_ORBPER", float("nan"))))
            if np.isfinite(P) and abs(period - P) / P < period_tol:
                return True, str(row.get("toi", row.get("TOI", "")))
        return False, ""
    except Exception:
        return False, ""


def compute_vetting_metrics(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    bls_sde: float = float("nan"),
    tpf_time: Optional[np.ndarray] = None,
    tpf_flux_cube: Optional[np.ndarray] = None,
    ra: float = float("nan"),
    dec: float = float("nan"),
    tic_id: str = "",
    toi_table: Optional[object] = None,
    half_width: float = 0.05,
    skip_gaia: bool = True,
) -> VettingMetrics:
    m = VettingMetrics()
    m.sde = float(bls_sde)

    depth, dur_phase = transit_depth_and_duration(time, flux, period, epoch, base_half_width=half_width)
    m.depth = depth
    m.duration_phase = dur_phase
    m.duration_hours = float(dur_phase * period * 24.0) if np.isfinite(dur_phase) else float("nan")
    m.n_transits = count_transits(time, period, epoch)
    m.odd_even_ratio = odd_even_depth_ratio(time, flux, period, epoch, half_width=half_width)
    m.secondary_depth, m.secondary_significance = secondary_eclipse(time, flux, period, epoch, half_width=half_width)
    m.v_shape_score = v_shape_score(time, flux, period, epoch, half_width=half_width)

    if tpf_time is not None and tpf_flux_cube is not None:
        offset, diff_peak = centroid_metrics_from_tpf_arrays(
            tpf_time, tpf_flux_cube, period, epoch, half_width=half_width,
        )
        m.centroid_offset_px = offset
        m.diff_image_peak_px = diff_peak

    if not skip_gaia and np.isfinite(ra) and np.isfinite(dec):
        m.gaia_neighbors_within_30arcsec = gaia_neighbors_within(30.0, ra, dec)

    if toi_table is not None and tic_id:
        m.matches_toi, m.matched_toi_id = match_toi(tic_id, period, toi_table)

    return m
