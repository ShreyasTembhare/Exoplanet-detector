"""
Phase 1: TESS data ingestion and cleaning (CPU).
- Source: TESS via lightkurve.
- Flattening: Savitzky-Golay (lc.flatten(window_length=101)).
- Outlier removal: 3-sigma sigma-clipping (remove_outliers(sigma=3)).
"""

import logging
from typing import Optional, Tuple, Union

import numpy as np
import lightkurve as lk
from lightkurve import LightCurve

from .cache_io import get_phase1, set_phase1

logger = logging.getLogger(__name__)


def _flux_to_array(flux) -> np.ndarray:
    """Extract numpy array from lightkurve flux (may be Quantity)."""
    if hasattr(flux, "value"):
        return np.asarray(flux.value, dtype=np.float64)
    return np.asarray(flux, dtype=np.float64)


def _time_to_array(time) -> np.ndarray:
    """Extract numpy array from lightkurve time (may be Time)."""
    if hasattr(time, "value"):
        return np.asarray(time.value, dtype=np.float64)
    return np.asarray(time, dtype=np.float64)


def load_tess_lightcurve(tic_id: str, sector: Optional[int] = None) -> Tuple[LightCurve, str]:
    """
    Load TESS light curve for a TIC. Prefer TPF -> LC for centroid later; fallback to pre-made LC.
    Returns (light_curve, sector_label) where sector_label is e.g. '1' or 'all'.
    """
    tic_str = str(tic_id).strip()
    if not tic_str.upper().startswith("TIC"):
        tic_str = f"TIC {tic_str}"

    # Prefer light curve search (simpler); can use search_tesscut/TPF if centroid needed in Phase 3
    search = lk.search_lightcurve(tic_str, mission="TESS", sector=sector)
    if len(search) == 0:
        raise ValueError(f"No TESS light curve found for {tic_str}")

    if sector is not None:
        lc = search[0].download()
        sector_label = str(lc.sector) if hasattr(lc, "sector") and lc.sector is not None else str(sector)
    else:
        # Download and stitch first few sectors for more coverage
        lcs = search[:3].download_all()
        lc = lcs.stitch()
        sector_label = "all"

    return lc, sector_label


def preprocess_lightcurve(lc: LightCurve) -> LightCurve:
    """
    Flatten with Savitzky-Golay (window_length=101) and 3-sigma outlier removal.
    lc.flatten(window_length=101).remove_outliers(sigma=3)
    """
    lc_flat = lc.flatten(window_length=101)
    lc_clean = lc_flat.remove_outliers(sigma=3)
    return lc_clean


def run_phase1(
    tic_id: str,
    sector: Optional[int] = None,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Run Phase 1: load TESS LC, flatten + 3σ clip. Optionally use cache.
    Returns (time, flux, flux_err, sector_label).
    """
    sector_label = "all" if sector is None else str(sector)

    if use_cache:
        cached = get_phase1(tic_id, sector_label)
        if cached is not None:
            return (*cached, sector_label)

    lc, sector_label = load_tess_lightcurve(tic_id, sector)
    lc_clean = preprocess_lightcurve(lc)

    time = _time_to_array(lc_clean.time)
    flux = _flux_to_array(lc_clean.flux)
    flux_err = _flux_to_array(lc_clean.flux_err) if lc_clean.flux_err is not None else np.full_like(flux, np.nan)

    if use_cache:
        set_phase1(tic_id, sector_label, time, flux, flux_err)

    return time, flux, flux_err, sector_label
