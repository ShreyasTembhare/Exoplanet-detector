"""
Physical priors for the BLS / TLS search.

The naive search uses a single fixed transit duration (``duration_phase=0.05``)
across all periods and all stars. That's wasteful — for any ``period`` the
realistic transit duration is bounded above by Kepler's third law given the
star's mean density ``rho_star``:

    T_dur (hours) ~= 13 * (P / 365 d)^(1/3) * (rho_sun / rho_star)^(1/3)

So a 1-day-period planet around a sun-like star has ``T_dur < ~3 hr``,
giving ``duration_phase ~ 0.13``, while a 30-day planet has
``T_dur ~ 11 hr`` -> ``duration_phase ~ 0.015``. Searching durations
appropriate to each period saves ~3-10x on grid size.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# rho_sun in g / cm^3
RHO_SUN = 1.408


def transit_duration_hours(period_days: float, rho_star_solar: float = 1.0) -> float:
    """Maximum realistic transit duration via Kepler's 3rd law (central transit)."""
    if rho_star_solar <= 0:
        rho_star_solar = 1.0
    return 13.0 * (period_days / 365.25) ** (1 / 3) / rho_star_solar ** (1 / 3)


def duration_phase_for_period(period_days: float, rho_star_solar: float = 1.0,
                              safety_factor: float = 1.5) -> float:
    """Return duration_phase = T_dur / P, with a safety factor to allow
    grazing geometry / impact parameter > 0 transits."""
    dur_hours = transit_duration_hours(period_days, rho_star_solar) * safety_factor
    dur_phase = dur_hours / 24.0 / max(period_days, 1e-6)
    return float(np.clip(dur_phase, 0.005, 0.25))


def derive_rho_star(meta: Optional[dict]) -> float:
    """Pull stellar density from TIC catalog metadata; fallback to solar."""
    meta = meta or {}
    rho = meta.get("rho_star", float("nan"))
    if isinstance(rho, (int, float)) and np.isfinite(rho) and rho > 0:
        return float(rho)
    logg = meta.get("logg", float("nan"))
    radius = meta.get("radius", float("nan"))
    # Cheap approximation: rho ~ 10^logg / R (in solar units). Falls back to solar.
    if (
        isinstance(logg, (int, float))
        and isinstance(radius, (int, float))
        and np.isfinite(logg)
        and np.isfinite(radius)
        and radius > 0
    ):
        # log g = log(M / R^2) + const; rho ~ M / R^3 = 10^logg / R.
        try:
            return float(10 ** logg / radius / 27400.0)  # rough normalization
        except Exception:
            pass
    return 1.0


def adaptive_duration_grid(
    period_min: float, period_max: float, rho_star_solar: float = 1.0,
    n: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return ``(periods_log, durations_phase)`` arrays where each period gets a
    matched duration_phase. ``n`` durations are sampled per period to allow
    for impact-parameter variation.
    """
    log_p = np.exp(np.linspace(np.log(period_min), np.log(period_max), 64))
    base = np.array([duration_phase_for_period(p, rho_star_solar) for p in log_p])
    if n <= 1:
        return log_p, base
    return log_p, np.outer(base, np.linspace(0.5, 1.5, n))
