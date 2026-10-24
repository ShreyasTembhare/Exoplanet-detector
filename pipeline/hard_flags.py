"""
Hard vetting flags that run before any classifier.

Odd/even depth, a significant secondary eclipse, a centroid shift, and too
few transits already live in :mod:`pipeline.vetting` and
:mod:`pipeline.centroid_vetting`. This module turns those numbers into a
stable list of reason codes so every SDE-pass peak is flagged the same way.
"""

from __future__ import annotations

from typing import List, Mapping, Optional

# Literature-style cuts used as an automated filter in front of a human
# (LEO-Vetter-like), not as a statistical validation.
ODD_EVEN_MIN_RATIO = 0.65
SECONDARY_SIGMA_MAX = 5.0
CENTROID_OFFSET_PX_MAX = 1.0
MIN_TRANSITS = 2


def _finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number


def hard_flag_reasons(vetting: Optional[Mapping]) -> List[str]:
    """Return reason codes for a vetting dict. Empty means no hard flag.

    Keys match :meth:`pipeline.vetting.VettingMetrics.as_dict`.
    """
    vetting = vetting or {}
    reasons: List[str] = []

    odd_even = vetting.get("odd_even_ratio")
    if _finite(odd_even) and float(odd_even) < ODD_EVEN_MIN_RATIO:
        reasons.append("odd_even")

    secondary = vetting.get("secondary_significance")
    if _finite(secondary) and float(secondary) > SECONDARY_SIGMA_MAX:
        reasons.append("secondary")

    centroid = vetting.get("centroid_offset_px")
    if centroid is None:
        centroid = vetting.get("centroid_offset")
    if _finite(centroid) and float(centroid) > CENTROID_OFFSET_PX_MAX:
        reasons.append("centroid")

    n_transits = vetting.get("n_transits")
    if n_transits is not None and _finite(n_transits) and int(float(n_transits)) < MIN_TRANSITS:
        reasons.append("n_transits")

    return reasons


GAIA_FLUX_RATIO_MAX = 0.5
FPP_SOFT_MAX = 0.5
DEMOTION_FLOOR = 0.05


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def demotion(vetting: Optional[Mapping]) -> dict:
    """Graded score multiplier instead of a pass/fail flag.

    Each test multiplies ``factor`` by how far the candidate is past its cut,
    so an odd/even ratio of 0.6 costs little and 0.2 costs a lot. Reasons
    keep the per-test multiplier so a human can see why a candidate sank.
    A neighbour that would need an eclipse deeper than 100% cannot cause the
    signal, so the contamination penalty is skipped in that case.
    """
    vetting = vetting or {}
    reasons = []

    def add(code: str, multiplier: float, value) -> None:
        if multiplier < 0.999:
            reasons.append({"code": code, "multiplier": round(multiplier, 3), "value": value})

    odd_even = vetting.get("odd_even_ratio")
    if _finite(odd_even) and float(odd_even) < ODD_EVEN_MIN_RATIO:
        add("odd_even", _clip(float(odd_even) / ODD_EVEN_MIN_RATIO, 0.2, 1.0), float(odd_even))

    secondary = vetting.get("secondary_significance")
    if _finite(secondary) and float(secondary) > SECONDARY_SIGMA_MAX:
        add("secondary", _clip(SECONDARY_SIGMA_MAX / float(secondary), 0.15, 1.0), float(secondary))

    centroid = vetting.get("centroid_offset_px")
    if centroid is None:
        centroid = vetting.get("centroid_offset")
    if _finite(centroid) and float(centroid) > CENTROID_OFFSET_PX_MAX:
        add("centroid", _clip(CENTROID_OFFSET_PX_MAX / float(centroid), 0.15, 1.0), float(centroid))

    n_transits = vetting.get("n_transits")
    if _finite(n_transits) and int(float(n_transits)) < MIN_TRANSITS and vetting.get("kind") != "monotransit":
        add("n_transits", 0.5, int(float(n_transits)))

    ratio = vetting.get("gaia_flux_ratio")
    needed = vetting.get("gaia_max_neighbor_depth_needed")
    impossible_blend = _finite(needed) and float(needed) > 1.0
    if _finite(ratio) and float(ratio) > GAIA_FLUX_RATIO_MAX and not impossible_blend:
        add("gaia_contamination", _clip(1.0 - 0.5 * (float(ratio) - GAIA_FLUX_RATIO_MAX), 0.3, 1.0), float(ratio))

    fpp = vetting.get("fpp")
    if _finite(fpp) and float(fpp) > FPP_SOFT_MAX:
        add("fpp", _clip(1.5 - float(fpp), 0.3, 1.0), float(fpp))

    factor = 1.0
    for reason in reasons:
        factor *= reason["multiplier"]
    return {"factor": round(max(DEMOTION_FLOOR, factor), 4), "reasons": reasons}


def accepts_for_reseach(vetting: Optional[Mapping], sde: float, sde_threshold: float) -> bool:
    """A peak is peeled off and masked when it is a repeating transit.

    Hard odd/even, secondary, and centroid flags still travel with the
    candidate, but a single event is left to the monotransit statistic
    instead of being treated as a periodic planet.
    """
    if not _finite(sde) or float(sde) < float(sde_threshold):
        return False
    reasons = hard_flag_reasons(vetting)
    if "n_transits" in reasons:
        return False
    return True
