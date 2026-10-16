"""
Batman injection-recovery on quiet stars.

Completeness versus period, radius, Tmag, and CDPP is the dial for the SDE
cut and the duration grid. This is a small harness, not a full campaign:
inject a Mandel–Agol transit with ``batman`` into a quiet residual curve,
run BLS, and record whether the period came back.

Injections belong on quiet stars. They are not a way to label FP/FA light
curves as planets.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

R_EARTH_OVER_R_SUN = 0.0091577
# Quadratic limb darkening near a solar-type TESS star. Good enough for a harness.
DEFAULT_U = (0.4, 0.2)


def batman_available() -> bool:
    try:
        import batman  # noqa: F401
        return True
    except Exception:
        return False


def inject_batman(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period: float,
    epoch: float,
    rp_rearth: float,
    rstar_rsun: float = 1.0,
    inclination_deg: float = 90.0,
    a_over_rstar: Optional[float] = None,
    limb_darkening: Sequence[float] = DEFAULT_U,
) -> np.ndarray:
    """Return a copy of ``flux`` with a batman transit multiplied in.

    ``a_over_rstar`` defaults to the circular stellar-density estimate at
    solar density, which sets the duration.
    """
    import batman

    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    rstar = float(rstar_rsun) if rstar_rsun and rstar_rsun > 0 else 1.0
    rp_rstar = float(rp_rearth) * R_EARTH_OVER_R_SUN / rstar
    if a_over_rstar is None:
        # a/R* ≈ 215 * (P/365.25)^(2/3) at solar density.
        a_over_rstar = 215.032 * (float(period) / 365.25) ** (2.0 / 3.0)
    params = batman.TransitParams()
    params.t0 = float(epoch)
    params.per = float(period)
    params.rp = float(np.clip(rp_rstar, 1e-4, 0.5))
    params.a = float(max(a_over_rstar, 2.0))
    params.inc = float(inclination_deg)
    params.ecc = 0.0
    params.w = 90.0
    params.u = list(limb_darkening)
    params.limb_dark = "quadratic"
    model = batman.TransitModel(params, time)
    injected = flux * model.light_curve(params)
    return np.asarray(injected, dtype=np.float64)


def recover_injection(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    true_period: float,
    period_min: float,
    period_max: float,
    sde_threshold: float = 6.0,
    duration_phase: float = 0.05,
    nperiods: int = 80,
    backend: str = "numpy",
    period_tol: float = 0.05,
) -> dict:
    """Run a small BLS and report whether the injected period was recovered."""
    from .bls_gpu import compute_bls_periodogram

    res = compute_bls_periodogram(
        time, flux,
        period_min=period_min, period_max=period_max,
        nperiods=nperiods, two_pass=False,
        backend=backend, duration_phase=duration_phase, topk=3,
        downsample_limit=20000,
    )
    rel = abs(res.best_period - true_period) / true_period if true_period else 1.0
    recovered = bool(rel <= period_tol and res.sde >= sde_threshold)
    return {
        "recovered": recovered,
        "best_period": float(res.best_period),
        "epoch": float(res.epoch),
        "sde": float(res.sde),
        "relative_period_error": float(rel),
    }


def estimate_cdpp_ppm(flux: np.ndarray) -> float:
    from .prefilter import estimate_cdpp

    value = float(estimate_cdpp(flux))
    # A perfectly flat residual is quiet, not missing. CDPP is 0 ppm.
    if not np.isfinite(value):
        return 0.0
    return value


def completeness_on_quiet_star(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    periods: Sequence[float],
    radii_rearth: Sequence[float],
    tmag: float,
    rstar_rsun: float = 1.0,
    sde_threshold: float = 6.0,
    backend: str = "numpy",
    nperiods: int = 60,
    rng_seed: int = 0,
) -> list:
    """Inject a small grid and return one row per (period, radius).

    Rows carry ``recovered``, ``period``, ``radius_rearth``, ``tmag``, and
    ``cdpp_ppm``. The input curve should already be a quiet residual; this
    function does not inject onto known false positives.
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    cdpp = estimate_cdpp_ppm(flux)
    rng = np.random.default_rng(rng_seed)
    t0 = float(np.nanmin(time))
    rows = []
    p_lo = min(periods) * 0.8
    p_hi = max(periods) * 1.2
    for period in periods:
        for radius in radii_rearth:
            epoch = t0 + float(rng.uniform(0.0, float(period)))
            injected = inject_batman(
                time, flux, period=float(period), epoch=epoch,
                rp_rearth=float(radius), rstar_rsun=rstar_rsun,
            )
            # Duration phase from the stellar-density prior, one width is enough
            # for the harness (the search itself uses the full grid).
            from .physical_priors import duration_phase_for_period
            dur_phase = duration_phase_for_period(float(period), 1.0)
            outcome = recover_injection(
                time, injected,
                true_period=float(period),
                period_min=p_lo, period_max=p_hi,
                sde_threshold=sde_threshold,
                duration_phase=dur_phase,
                nperiods=nperiods,
                backend=backend,
            )
            rows.append({
                "recovered": outcome["recovered"],
                "period": float(period),
                "radius_rearth": float(radius),
                "tmag": float(tmag),
                "cdpp_ppm": cdpp,
                "sde": outcome["sde"],
                "best_period": outcome["best_period"],
                "relative_period_error": outcome["relative_period_error"],
            })
    return rows


def completeness_summary(rows: Sequence[dict]) -> dict:
    """Fraction recovered, plus the axes the harness is supposed to vary."""
    if not rows:
        return {"n": 0, "completeness": float("nan"), "by_period": {}, "by_radius": {}}
    recovered = [bool(r["recovered"]) for r in rows]
    by_period: dict = {}
    by_radius: dict = {}
    for row in rows:
        by_period.setdefault(row["period"], []).append(bool(row["recovered"]))
        by_radius.setdefault(row["radius_rearth"], []).append(bool(row["recovered"]))
    return {
        "n": len(rows),
        "completeness": float(np.mean(recovered)),
        "tmag": rows[0].get("tmag"),
        "cdpp_ppm": rows[0].get("cdpp_ppm"),
        "by_period": {p: float(np.mean(v)) for p, v in by_period.items()},
        "by_radius": {r: float(np.mean(v)) for r, v in by_radius.items()},
    }
