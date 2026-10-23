"""
Period grids for BLS.

A log-uniform grid spends as many trial periods between 10 and 15 days as
between 0.5 and 0.75 days, but a transit at short period occupies a much
larger phase fraction... of a much narrower peak in frequency. Ofir (2014)
showed the optimal sampling is uniform in ``f**(1/3)``: the frequency step
that keeps a transit from drifting by more than ~one duration across the
baseline scales as ``f**(2/3)``. For a fixed budget this puts more trials at
short periods (where peaks are narrow) and fewer at long ones, recovering
low-SNR short-period signals the log grid steps over.
"""

from __future__ import annotations

import numpy as np

G_SI = 6.674e-11
M_SUN = 1.989e30
R_SUN = 6.957e8
DAY = 86400.0


def ofir_spacing(baseline_days: float, rho_star_solar: float = 1.0, oversampling: float = 3.0) -> float:
    """Step in ``f**(1/3)`` (f in 1/day) from Ofir (2014), eq. 5."""
    rho = max(float(rho_star_solar), 1e-3)
    # R/(G M)^(1/3) for a star of solar mass and radius R = rho^(-1/3) R_sun.
    r_over = (rho ** (-1.0 / 3.0)) * R_SUN / (G_SI * M_SUN) ** (1.0 / 3.0)
    span = max(float(baseline_days), 1e-3) * DAY
    a_si = (2.0 * np.pi) ** (2.0 / 3.0) / np.pi * r_over / (span * max(float(oversampling), 1e-3))
    # f^(1/3) in Hz^(1/3) -> day^-1 units: multiply by DAY^(1/3).
    return float(a_si * DAY ** (1.0 / 3.0))


def ofir_count(baseline_days: float, period_min: float, period_max: float,
               rho_star_solar: float = 1.0, oversampling: float = 3.0) -> int:
    """Number of trial periods the optimal grid needs."""
    x_lo = (1.0 / float(period_max)) ** (1.0 / 3.0)
    x_hi = (1.0 / float(period_min)) ** (1.0 / 3.0)
    step = ofir_spacing(baseline_days, rho_star_solar, oversampling) / 3.0
    return int(np.ceil((x_hi - x_lo) / step)) + 1


def ofir_periods(period_min: float, period_max: float, n: int) -> np.ndarray:
    """``n`` periods spaced uniformly in ``f**(1/3)``, ascending in period."""
    n = max(int(n), 2)
    x = np.linspace((1.0 / float(period_max)) ** (1.0 / 3.0), (1.0 / float(period_min)) ** (1.0 / 3.0), n)
    return np.sort(1.0 / x ** 3)


def log_periods(period_min: float, period_max: float, n: int) -> np.ndarray:
    return np.exp(np.linspace(np.log(period_min), np.log(period_max), max(int(n), 2)))


def make_grid(kind: str, period_min: float, period_max: float, n: int) -> np.ndarray:
    if kind == "ofir":
        return ofir_periods(period_min, period_max, n)
    return log_periods(period_min, period_max, n)
