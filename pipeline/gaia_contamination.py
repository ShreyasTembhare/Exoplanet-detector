"""
Gaia DR3 contamination around a TESS target.

TESS pixels are 21 arcsec wide, so a transit-like dip is often a background
eclipsing binary blended into the aperture. Instead of only counting Gaia
neighbours, this estimates how much of the flux in a ~2-pixel aperture comes
from them (a Gaussian PSF weight per source) and the depth a neighbour would
need to fake the signal.

``gaia_flux_ratio`` = neighbour flux / target flux inside the aperture. Above
~0.5 the blend scenario is plausible for shallow signals; the graded demotion
in :mod:`pipeline.hard_flags` uses it.
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

TESS_PIXEL_ARCSEC = 21.0
PSF_SIGMA_ARCSEC = 0.65 * TESS_PIXEL_ARCSEC
SEARCH_RADIUS_ARCSEC = 2.5 * TESS_PIXEL_ARCSEC
TARGET_MATCH_ARCSEC = 2.0

_cache: dict = {}


def contamination_from_sources(
    separations_arcsec: Sequence[float],
    gmags: Sequence[float],
    target_gmag: Optional[float] = None,
    depth: Optional[float] = None,
) -> dict:
    """Pure part: PSF-weighted flux of neighbours relative to the target.

    The target is the brightest source within ``TARGET_MATCH_ARCSEC`` unless
    ``target_gmag`` is given. ``depth`` (fractional) adds the eclipse depth a
    neighbour would need (``depth * total / neighbour``); above 1 that
    neighbour cannot cause the signal.
    """
    sep = np.asarray(separations_arcsec, dtype=float)
    mag = np.asarray(gmags, dtype=float)
    keep = np.isfinite(sep) & np.isfinite(mag)
    sep, mag = sep[keep], mag[keep]
    out = {
        "gaia_neighbors_within_30arcsec": -1,
        "gaia_flux_ratio": float("nan"),
        "gaia_brightest_neighbor_dmag": float("nan"),
        "gaia_max_neighbor_depth_needed": float("nan"),
    }
    if len(sep) == 0:
        return out
    order = np.argsort(sep)
    sep, mag = sep[order], mag[order]
    target_index = None
    if target_gmag is None:
        close = np.where(sep <= TARGET_MATCH_ARCSEC)[0]
        if len(close):
            target_index = int(close[np.argmin(mag[close])])
            target_gmag = float(mag[target_index])
    if target_gmag is None or not math.isfinite(target_gmag):
        return out
    neighbours = np.ones(len(sep), dtype=bool)
    if target_index is not None:
        neighbours[target_index] = False
    else:
        neighbours &= ~((sep <= TARGET_MATCH_ARCSEC) & (np.abs(mag - target_gmag) < 0.5))
    weight = np.exp(-0.5 * (sep / PSF_SIGMA_ARCSEC) ** 2)
    flux = 10 ** (-0.4 * (mag - target_gmag)) * weight
    neighbour_flux = float(np.sum(flux[neighbours]))
    out["gaia_neighbors_within_30arcsec"] = int(np.sum(neighbours & (sep <= 30.0)))
    out["gaia_flux_ratio"] = neighbour_flux
    if np.any(neighbours):
        out["gaia_brightest_neighbor_dmag"] = float(np.min(mag[neighbours]) - target_gmag)
        if depth is not None and math.isfinite(depth) and depth > 0:
            total = 1.0 + neighbour_flux
            strongest = float(np.max(flux[neighbours]))
            out["gaia_max_neighbor_depth_needed"] = float(depth * total / strongest) if strongest > 0 else float("inf")
    return out


def query_gaia_sources(ra: float, dec: float, radius_arcsec: float = SEARCH_RADIUS_ARCSEC):  # pragma: no cover
    """Return ``(separations_arcsec, gmags)`` from Gaia DR3, or ``None`` offline."""
    if not (math.isfinite(ra) and math.isfinite(dec)):
        return None
    try:
        from astroquery.gaia import Gaia
    except Exception:
        return None
    query = (
        "SELECT DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec})) * 3600 AS sep, phot_g_mean_mag "
        "FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {r}))"
    ).format(ra=float(ra), dec=float(dec), r=float(radius_arcsec) / 3600.0)
    try:
        table = Gaia.launch_job(query).get_results()
    except Exception as exc:
        logger.info("Gaia DR3 query failed: %s", exc)
        return None
    return np.asarray(table["sep"], dtype=float), np.asarray(table["phot_g_mean_mag"], dtype=float)


def gaia_contamination(ra: float, dec: float, depth: Optional[float] = None) -> dict:
    """Contamination for one star; the Gaia query is cached per position."""
    if not (isinstance(ra, (int, float)) and isinstance(dec, (int, float))
            and math.isfinite(ra) and math.isfinite(dec)):
        return contamination_from_sources([], [])
    key = (round(float(ra), 5), round(float(dec), 5))
    if key not in _cache:
        if len(_cache) > 4096:
            _cache.clear()
        _cache[key] = query_gaia_sources(float(ra), float(dec))
    sources = _cache[key]
    if sources is None:
        return contamination_from_sources([], [])
    return contamination_from_sources(sources[0], sources[1], depth=depth)
