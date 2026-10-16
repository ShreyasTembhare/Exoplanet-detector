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
