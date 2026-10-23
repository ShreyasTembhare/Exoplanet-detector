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


FLUX_COLUMNS = {
    "SPOC": "pdcsap_flux",
    "TESS-SPOC": "pdcsap_flux",
    "QLP": "kspsap_flux",
}


def _column(table, name, n):
    try:
        return list(table[name])
    except Exception:
        return [None] * n


def select_products(table, sector: Optional[int] = None, max_sectors: Optional[int] = None) -> Tuple[list, str]:
    """Pick which rows of a ``search_lightcurve`` table to download.

    * One author for the whole stitch (mixing pipelines leaves steps at sector
      boundaries). The author covering the most sectors wins, so a star with
      two SPOC 2-min sectors and ten QLP/TESS-SPOC FFI sectors is searched on
      the long baseline; ties go to ``PREFERRED_AUTHORS`` order.
    * One product per sector, the shortest cadence (20 s < 120 s < FFI).
    * Sectors in ascending order; ``max_sectors=None`` keeps them all.

    Returns ``(row_indices, author)``.
    """
    n = len(table)
    if n == 0:
        return [], ""
    authors = [str(a or "").strip() for a in _column(table, "author", n)]
    exptimes = []
    for value in _column(table, "exptime", n):
        try:
            exptimes.append(float(getattr(value, "value", value)))
        except (TypeError, ValueError):
            exptimes.append(float("inf"))
    sectors = []
    for i, value in enumerate(_column(table, "sequence_number", n)):
        try:
            sectors.append(int(value))
        except (TypeError, ValueError):
            sectors.append(-1 - i)

    def rank(author: str) -> int:
        return PREFERRED_AUTHORS.index(author) if author in PREFERRED_AUTHORS else len(PREFERRED_AUTHORS)

    rows = [i for i in range(n) if sector is None or sectors[i] == int(sector)] or list(range(n))
    coverage: dict = {}
    for i in rows:
        coverage.setdefault(authors[i], set()).add(sectors[i])
    best_author = min(coverage, key=lambda a: (-len(coverage[a]), rank(a)))
    best_per_sector: dict = {}
    for i in rows:
        if authors[i] != best_author:
            continue
        current = best_per_sector.get(sectors[i])
        if current is None or exptimes[i] < exptimes[current]:
            best_per_sector[sectors[i]] = i
    chosen = [best_per_sector[s] for s in sorted(best_per_sector)]
    if max_sectors:
        chosen = chosen[:max_sectors]
    return chosen, best_author


def _download(product, quality_bitmask: str, author: str):
    column = FLUX_COLUMNS.get(author)
    if column:
        try:
            return product.download(quality_bitmask=quality_bitmask, flux_column=column)
        except Exception as exc:
            logger.info("%s flux column %s unavailable (%s); using the default column", author, column, exc)
    return product.download(quality_bitmask=quality_bitmask)


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
    max_sectors: Optional[int] = None,
) -> Tuple[Any, str, dict]:
    """
    Load TESS light curve(s) for a TIC. Returns (lc, sector_label, meta).
    ``meta`` carries cadence_minutes, author, crowdsap, tmag, n_sectors,
    sectors and exptime_s. See :func:`select_products` for which products
    are used; FFI high-level science products (TESS-SPOC, QLP) count, so
    stars without 2-min data are searched too.
    """
    import lightkurve as lk

    tic_str = str(tic_id).strip()
    if not tic_str.upper().startswith("TIC"):
        tic_str = f"TIC {tic_str}"

    search = lk.search_lightcurve(tic_str, mission="TESS", sector=sector)
    if len(search) == 0:
        raise ValueError(f"No TESS light curve found for {tic_str}")
    chosen, author = select_products(search.table, sector=sector, max_sectors=max_sectors)
    if not chosen:
        raise ValueError(f"No usable TESS light curve product for {tic_str}")

    meta: dict = {"author": author or "unknown"}
    lcs = []
    for index in chosen:
        try:
            lcs.append(_download(search[index], quality_bitmask, author))
        except Exception as exc:
            logger.warning("Download failed for %s row %d: %s", tic_str, index, exc)
    if not lcs:
        raise ValueError(f"Every light-curve download failed for {tic_str}")
    if len(lcs) == 1:
        lc = lcs[0].normalize()
    else:
        lc = lk.LightCurveCollection(lcs).stitch()
    meta["n_sectors"] = len(lcs)
    meta["sectors"] = [int(getattr(item, "sector", 0) or 0) for item in lcs]
    try:
        meta["exptime_s"] = float(getattr(search.table["exptime"][chosen[0]], "value",
                                          search.table["exptime"][chosen[0]]))
    except Exception:
        meta["exptime_s"] = float("nan")
    if sector is not None:
        sector_label = str(meta["sectors"][0] or sector)
    else:
        sector_label = "all"
    for header_key, meta_key in [
        ("CROWDSAP", "crowdsap"),
        ("TESSMAG", "tmag"),
        ("TEFF", "teff"),
        ("LOGG", "logg"),
        ("RADIUS", "radius"),
        ("RA_OBJ", "ra"),
        ("DEC_OBJ", "dec"),
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
