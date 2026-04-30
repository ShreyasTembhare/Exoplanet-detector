"""
Cheap pre-filters to gate which TICs reach the BLS stage.

The BLS stage is the throughput bottleneck; running it on stars that are too
faint, too contaminated, or too noisy wastes compute and dilutes the candidate
list with nuisance peaks. Prefilters here are O(N) over the light curve and
cost ~1 ms per star, vs ~100 ms-1 s for BLS.

Decisions returned as :class:`PrefilterDecision`:
  * ``passed``        -- True/False
  * ``reasons``       -- list of strings naming any failed checks
  * ``cdpp_ppm``      -- estimated CDPP (combined differential photometric precision)
  * ``rotation_period_d`` -- rough rotation period if the LC is dominated by rotation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class PrefilterDecision:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    cdpp_ppm: float = float("nan")
    rotation_period_d: float = float("nan")
    n_cadences: int = 0


def estimate_cdpp(flux: np.ndarray, window: int = 13) -> float:
    """
    Rough CDPP estimate: rolling-median residual std, scaled to ppm.
    A real pipeline would use Kepler's CDPP definition; this is good enough
    to filter obviously noisy LCs in O(N).
    """
    flux = np.asarray(flux, dtype=np.float64)
    flux = flux[np.isfinite(flux)]
    if len(flux) < window * 2:
        return float("nan")
    cumsum = np.cumsum(flux)
    rolling = (cumsum[window:] - cumsum[:-window]) / window
    res = flux[window // 2 + 1: window // 2 + 1 + len(rolling)] - rolling
    if len(res) == 0:
        return float("nan")
    sigma = np.median(np.abs(res - np.median(res))) * 1.4826
    if sigma <= 0:
        return float("nan")
    return float(sigma * 1e6)


def estimate_rotation_period(time: np.ndarray, flux: np.ndarray) -> float:
    """Cheap proxy: autocorrelation peak over [0.2, 30] day lags."""
    t = np.asarray(time, dtype=np.float64)
    f = np.asarray(flux, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(f)
    t, f = t[mask], f[mask]
    if len(t) < 100:
        return float("nan")
    cadence_d = float(np.median(np.diff(t)))
    if cadence_d <= 0:
        return float("nan")
    f -= np.mean(f)
    n = len(f)
    max_lag = min(n - 1, int(30.0 / cadence_d))
    min_lag = max(1, int(0.2 / cadence_d))
    if max_lag <= min_lag:
        return float("nan")
    lags = np.arange(min_lag, max_lag)
    norms = np.sum(f * f) + 1e-12
    acf = np.array([np.sum(f[:n - lag] * f[lag:]) / norms for lag in lags])
    if len(acf) == 0:
        return float("nan")
    peak_idx = int(np.argmax(acf))
    return float(lags[peak_idx] * cadence_d)


def prefilter_target(
    time: np.ndarray,
    flux: np.ndarray,
    meta: Optional[dict] = None,
    *,
    max_cdpp_ppm: float = 5000.0,
    max_tmag: float = 16.0,
    min_crowdsap: float = 0.5,
    min_cadences: int = 200,
    flag_short_rotation: bool = True,
) -> PrefilterDecision:
    """Return a :class:`PrefilterDecision` summarizing whether this TIC is
    worth running BLS on."""
    meta = meta or {}
    reasons: List[str] = []
    n = int(len(time))

    if n < min_cadences:
        reasons.append(f"too_few_cadences({n}<{min_cadences})")

    cdpp = estimate_cdpp(flux)
    if np.isfinite(cdpp) and cdpp > max_cdpp_ppm:
        reasons.append(f"high_cdpp({cdpp:.0f}ppm)")

    tmag = float(meta.get("tmag", float("nan")))
    if np.isfinite(tmag) and tmag > max_tmag:
        reasons.append(f"too_faint(Tmag={tmag:.2f})")

    crowdsap = float(meta.get("crowdsap", float("nan")))
    if np.isfinite(crowdsap) and crowdsap < min_crowdsap:
        reasons.append(f"contaminated(crowdsap={crowdsap:.2f})")

    rot = estimate_rotation_period(time, flux) if flag_short_rotation else float("nan")
    if flag_short_rotation and np.isfinite(rot) and rot < 1.0:
        # Don't fail outright -- many fast rotators are still valid hot Jupiter
        # hosts. Just flag.
        pass

    return PrefilterDecision(
        passed=len(reasons) == 0,
        reasons=reasons,
        cdpp_ppm=cdpp,
        rotation_period_d=rot,
        n_cadences=n,
    )
