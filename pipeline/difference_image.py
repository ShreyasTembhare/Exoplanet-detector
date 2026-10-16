"""
One difference-image score for a shortlisted candidate.

ExoMiner++ showed that a difference image separates planets from nearby
eclipsing binaries. This is that one number — peak offset of the
in-minus-out image from the out-of-transit centroid, plus its significance —
not an 11-branch clone inside the per-star loop.
"""

from __future__ import annotations

import numpy as np

from .vetting import fold_phase


def difference_image_score(
    tpf_time: np.ndarray,
    flux_cube: np.ndarray,
    period: float,
    epoch: float,
    half_width: float = 0.05,
) -> dict:
    """Score one TPF cube. Missing or tiny cubes return NaNs and ``available=False``."""
    empty = {
        "available": False,
        "diff_image_offset_px": float("nan"),
        "diff_image_significance": float("nan"),
    }
    if tpf_time is None or flux_cube is None:
        return empty
    cube = np.asarray(flux_cube, dtype=np.float64)
    times = np.asarray(tpf_time, dtype=np.float64)
    if cube.ndim != 3 or len(times) != cube.shape[0] or cube.shape[0] < 8:
        return empty

    phase = fold_phase(times, period, epoch)
    half = float(half_width) if half_width and half_width > 0 else 0.05
    # Distance to phase 0, wrapping.
    folded = np.where(phase > 0.5, phase - 1.0, phase)
    in_mask = np.abs(folded) <= half
    out_mask = ~in_mask
    if int(np.sum(in_mask)) < 3 or int(np.sum(out_mask)) < 5:
        return empty

    flux_in = np.nanmean(cube[in_mask], axis=0)
    flux_out = np.nanmean(cube[out_mask], axis=0)
    diff = flux_out - flux_in  # positive where flux dropped in transit

    n_row, n_col = diff.shape
    rows = np.arange(n_row, dtype=np.float64)
    cols = np.arange(n_col, dtype=np.float64)

    def _centroid(img: np.ndarray):
        positive = np.where(np.isfinite(img) & (img > 0), img, 0.0)
        total = float(np.sum(positive))
        if total <= 0:
            return float("nan"), float("nan")
        col_c = float(np.sum(cols[None, :] * positive) / total)
        row_c = float(np.sum(rows[:, None] * positive) / total)
        return row_c, col_c

    out_row, out_col = _centroid(flux_out)
    diff_row, diff_col = _centroid(diff)
    if not all(np.isfinite(v) for v in (out_row, out_col, diff_row, diff_col)):
        offset = float("nan")
    else:
        offset = float(np.hypot(diff_row - out_row, diff_col - out_col))

    finite = diff[np.isfinite(diff)]
    if len(finite) < 4:
        significance = float("nan")
    else:
        med = float(np.median(finite))
        mad = float(np.median(np.abs(finite - med))) * 1.4826
        sigma = mad if mad > 0 else float(np.std(finite))
        peak = float(np.nanmax(diff))
        if sigma > 0:
            significance = (peak - med) / sigma
        elif peak > med:
            significance = float("inf")
        else:
            significance = float("nan")

    return {
        "available": True,
        "diff_image_offset_px": offset,
        "diff_image_significance": significance,
    }
