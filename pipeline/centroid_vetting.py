"""
Optional pixel-level centroid vetting via the ``tesscentroidvetting``
package. Wrapped so the rest of the pipeline works whether or not it's
installed.

Outputs:
  * ``centroid_offset_arcsec`` — offset between in-transit and out-of-transit
    centroids in arcsec (TESS plate scale ~21"/pixel).
  * ``ks_test_p`` — Kolmogorov-Smirnov p-value comparing in/out centroid
    distributions; small values indicate a significant offset.
  * ``diff_image_significance`` — peak / median of the difference image,
    in sigma.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CentroidVettingResult:
    centroid_offset_arcsec: float = float("nan")
    ks_test_p: float = float("nan")
    diff_image_significance: float = float("nan")
    available: bool = False


def vet_with_tesscentroidvetting(
    tpf, period: float, epoch: float, duration_phase: float = 0.05,
) -> CentroidVettingResult:  # pragma: no cover - external
    try:
        import tesscentroidvetting as tcv
    except Exception:
        return CentroidVettingResult()
    try:
        result = tcv.vet(tpf, period=period, t0=epoch,
                         duration=period * duration_phase)
        return CentroidVettingResult(
            centroid_offset_arcsec=float(getattr(result, "offset_arcsec", float("nan"))),
            ks_test_p=float(getattr(result, "ks_p", float("nan"))),
            diff_image_significance=float(getattr(result, "diff_sigma", float("nan"))),
            available=True,
        )
    except Exception as exc:
        logger.warning("tesscentroidvetting failed: %s", exc)
        return CentroidVettingResult()


def fallback_vet_from_arrays(
    tpf_time: np.ndarray, flux_cube: np.ndarray,
    period: float, epoch: float, duration_phase: float = 0.05,
    plate_scale_arcsec: float = 21.0,
) -> CentroidVettingResult:
    """Compute the same outputs without the external library, using just the
    flux cube. Less polished than tesscentroidvetting but always available."""
    from .vetting import centroid_metrics_from_tpf_arrays

    offset_px, _ = centroid_metrics_from_tpf_arrays(
        tpf_time, flux_cube, period, epoch, half_width=duration_phase,
    )
    return CentroidVettingResult(
        centroid_offset_arcsec=offset_px * plate_scale_arcsec
            if np.isfinite(offset_px) else float("nan"),
        ks_test_p=float("nan"),
        diff_image_significance=float("nan"),
        available=False,
    )
