"""
Phase 1: TESS data ingestion and cleaning (CPU).

Improvements over the previous version:
  * Quality bitmask passed to ``download`` so bad cadences (momentum dumps,
    scattered light, etc.) are dropped before flattening — otherwise the
    sigma-clip under-clips because they inflate the noise estimate.
  * Same-author stitching: filter the search table to a single author
    (``SPOC`` first, fallback ``TESS-SPOC`` then ``QLP``) before downloading
    multi-sector products. Mixing authors leaves flux discontinuities at
    sector boundaries that BLS picks up as spurious 27-day power.
  * Cadence-aware Savitzky-Golay window: the previous fixed
    ``window_length=101`` corresponded to 3.3 hours at 2-min cadence and
    silently flattened away hot-Jupiter ingresses. Window now scales as
    ``ceil(12 hr / cadence_minutes)`` and is rounded to the nearest odd
    value.
  * ``run_phase1`` now also returns optional metadata (cadence_minutes,
    crowdsap, tmag, author, n_sectors) so prefilters and vetting can use it.
  * Optional TPF download for centroid vetting (``with_tpf=True``).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Tuple

import numpy as np

from .cache_io import (
    get_phase1,
    get_tpf_arrays,
    set_phase1,
    set_tpf_arrays,
)

logger = logging.getLogger(__name__)


PREFERRED_AUTHORS = ("SPOC", "TESS-SPOC", "QLP")


def _flux_to_array(flux) -> np.ndarray:
    if hasattr(flux, "value"):
        return np.asarray(flux.value, dtype=np.float64)
    return np.asarray(flux, dtype=np.float64)


def _time_to_array(time) -> np.ndarray:
    if hasattr(time, "value"):
        return np.asarray(time.value, dtype=np.float64)
    return np.asarray(time, dtype=np.float64)


def _cadence_minutes(time: np.ndarray) -> float:
    if len(time) < 2:
        return 2.0
    diffs = np.diff(np.asarray(time, dtype=np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 2.0
    return float(np.median(diffs) * 24.0 * 60.0)


def cadence_aware_window(cadence_minutes: float, target_hours: float = 12.0) -> int:
    """Window length (in cadences) for Savitzky-Golay flatten."""
    if cadence_minutes <= 0:
        cadence_minutes = 2.0
    n = int(math.ceil(target_hours * 60.0 / cadence_minutes))
    n = max(n, 51)  # safety floor for noisy short LCs
    if n % 2 == 0:
        n += 1
    return n


def _filter_same_author(search):
    """Return a search restricted to a single preferred author, or the original
    if no preferred author is present."""
    if search is None or len(search) == 0:
        return search
    try:
        authors = search.table["author"]
    except Exception:
        return search
    for a in PREFERRED_AUTHORS:
        mask = np.array([str(x).strip() == a for x in authors])
        if np.any(mask):
            return search[mask]
    return search


def load_tess_lightcurve(
    tic_id: str,
    sector: Optional[int] = None,
    quality_bitmask: str = "default",
    max_sectors: int = 6,
) -> Tuple[Any, str, dict]:
    """
    Load TESS light curve(s) for a TIC. Returns (lc, sector_label, meta).
    ``meta`` carries cadence_minutes, author, crowdsap, tmag, n_sectors.
    """
    import lightkurve as lk

    tic_str = str(tic_id).strip()
    if not tic_str.upper().startswith("TIC"):
        tic_str = f"TIC {tic_str}"

    search = lk.search_lightcurve(tic_str, mission="TESS", sector=sector)
    if len(search) == 0:
        raise ValueError(f"No TESS light curve found for {tic_str}")
    search = _filter_same_author(search)

    meta: dict = {}
    if sector is not None:
        lc = search[0].download(quality_bitmask=quality_bitmask)
        meta["n_sectors"] = 1
        sector_label = str(lc.sector) if hasattr(lc, "sector") and lc.sector is not None else str(sector)
    else:
        lcs = search[:max_sectors].download_all(quality_bitmask=quality_bitmask)
        lc = lcs.stitch()
        meta["n_sectors"] = len(lcs)
        sector_label = "all"

    try:
        meta["author"] = str(getattr(lc, "author", search.table["author"][0]))
    except Exception:
        meta["author"] = "unknown"
    for header_key, meta_key in [
        ("CROWDSAP", "crowdsap"),
        ("TESSMAG", "tmag"),
        ("TEFF", "teff"),
        ("LOGG", "logg"),
        ("RADIUS", "radius"),
    ]:
        try:
            meta[meta_key] = float(lc.meta.get(header_key, lc.meta.get(header_key.lower(), float("nan"))))
        except Exception:
            meta[meta_key] = float("nan")

    return lc, sector_label, meta


def preprocess_lightcurve(lc, cadence_minutes: Optional[float] = None):
    """
    Drop NaN cadences, flatten with cadence-aware Savitzky-Golay window,
    and remove 3-sigma outliers.
    """
    lc = lc.remove_nans()
    if cadence_minutes is None:
        cadence_minutes = _cadence_minutes(_time_to_array(lc.time))
    window = cadence_aware_window(cadence_minutes)
    try:
        lc_flat = lc.flatten(window_length=window)
    except Exception as exc:
        logger.warning("flatten(window=%d) failed (%s); retrying with 101", window, exc)
        lc_flat = lc.flatten(window_length=101)
    return lc_flat.remove_outliers(sigma=3)


def _maybe_download_tpf(tic_id: str, sector: Optional[int]):  # pragma: no cover - network
    import lightkurve as lk
    tic_str = str(tic_id).strip()
    if not tic_str.upper().startswith("TIC"):
        tic_str = f"TIC {tic_str}"
    search = lk.search_targetpixelfile(tic_str, mission="TESS", sector=sector)
    if search is None or len(search) == 0:
        return None
    search = _filter_same_author(search)
    return search[0].download()


def fetch_tpf_arrays(
    tic_id: str, sector: Optional[int] = None, use_cache: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns (time, flux_cube, aperture) for the TPF, with caching.
    Used by Phase 3 centroid + difference-image vetting.
    """
    sector_label = "all" if sector is None else str(sector)
    if use_cache:
        cached = get_tpf_arrays(tic_id, sector_label)
        if cached is not None:
            return cached
    try:
        tpf = _maybe_download_tpf(tic_id, sector)
    except Exception as exc:
        logger.warning("TPF download failed for %s sector=%s: %s", tic_id, sector, exc)
        return None
    if tpf is None:
        return None
    try:
        time = _time_to_array(tpf.time)
        flux_cube = np.asarray(tpf.flux, dtype=np.float32)
        aperture = np.asarray(tpf.pipeline_mask, dtype=bool) if hasattr(tpf, "pipeline_mask") else None
    except Exception as exc:
        logger.warning("TPF array extraction failed: %s", exc)
        return None
    if use_cache:
        set_tpf_arrays(tic_id, sector_label, time, flux_cube, aperture)
    return time, flux_cube, aperture


def run_phase1(
    tic_id: str,
    sector: Optional[int] = None,
    use_cache: bool = True,
    quality_bitmask: str = "default",
    with_tpf: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, dict]:
    """
    Run Phase 1 and return ``(time, flux, flux_err, sector_label, meta)``.

    ``meta`` includes ``cadence_minutes``, ``crowdsap``, ``tmag``, ``author``,
    ``n_sectors``, and ``with_tpf`` (bool indicating cache had TPF).
    """
    sector_label = "all" if sector is None else str(sector)

    if use_cache:
        cached = get_phase1(tic_id, sector_label)
        if cached is not None:
            time, flux, flux_err, meta = cached
            if with_tpf:
                fetch_tpf_arrays(tic_id, sector, use_cache=True)
            return time, flux, flux_err, sector_label, meta

    lc, sector_label, meta = load_tess_lightcurve(
        tic_id, sector, quality_bitmask=quality_bitmask,
    )
    time_full = _time_to_array(lc.time)
    cadence_min = _cadence_minutes(time_full)
    meta["cadence_minutes"] = cadence_min

    lc_clean = preprocess_lightcurve(lc, cadence_minutes=cadence_min)

    time = _time_to_array(lc_clean.time)
    flux = _flux_to_array(lc_clean.flux)
    if lc_clean.flux_err is not None:
        flux_err = _flux_to_array(lc_clean.flux_err)
    else:
        flux_err = np.full_like(flux, np.nan)

    if use_cache:
        set_phase1(tic_id, sector_label, time, flux, flux_err, meta=meta)

    if with_tpf:
        try:
            fetch_tpf_arrays(tic_id, sector, use_cache=True)
        except Exception as exc:  # pragma: no cover
            logger.warning("TPF fetch skipped: %s", exc)

    return time, flux, flux_err, sector_label, meta
