"""
Yield-stage search shared by the sequential and parallel hunters.

After BLS:

* fold and vet every SDE-pass peak (hard flags, Gaia on this shortlist only)
* TLS-refine those peaks only — never the stars that failed the SDE gate
* one difference-image score and an optional TRICERATOPS call, shortlist only
* when a repeating transit is accepted, mask it and search again
* a single-event MES for monotransits on the unfolded curve

The unfolded transformer is intentionally not imported here.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .bls_gpu import BLSResult, _epoch_for_period, compute_bls_periodogram, compute_sde
from .hard_flags import accepts_for_reseach, hard_flag_reasons

logger = logging.getLogger(__name__)


def mask_transits(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration_days: float,
    pad: float = 1.5,
) -> np.ndarray:
    """Replace in-transit cadences with a linear interpolation of the wings."""
    t = np.asarray(time, dtype=np.float64)
    f = np.asarray(flux, dtype=np.float64).copy()
    if len(t) == 0 or period <= 0 or not np.isfinite(period):
        return f
    half = 0.5 * float(duration_days) * float(pad)
    if not np.isfinite(half) or half <= 0:
        half = 0.05 * period
    phase = np.mod(t - float(epoch), period)
    dist = np.minimum(phase, period - phase)
    mask = dist <= half
    if int(np.sum(mask)) == 0 or int(np.sum(~mask)) < 5:
        return f
    f[mask] = np.nan
    good = np.isfinite(f)
    f[~good] = np.interp(t[~good], t[good], f[good])
    return f


def mask_known_signal(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: Optional[float] = None,
    duration_phase: float = 0.05,
) -> np.ndarray:
    """Mask a catalog ephemeris before searching a known host for another planet.

    When the catalog epoch is missing, the epoch is the depth-maximizing phase
    at the known period so the known signal is what gets removed.
    """
    t = np.asarray(time, dtype=np.float64)
    f = np.asarray(flux, dtype=np.float64)
    dur_phase = float(duration_phase) if duration_phase and duration_phase > 0 else 0.05
    if epoch is None or not (isinstance(epoch, (int, float)) and np.isfinite(epoch)):
        epoch = _epoch_for_period(t, f, float(period), dur_phase)
    return mask_transits(t, f, float(period), float(epoch), dur_phase * float(period))


def _duration_days(period: float, duration_phase: float) -> float:
    if duration_phase and np.isfinite(duration_phase) and duration_phase > 0:
        return float(duration_phase) * float(period)
    return 0.05 * float(period)


def _peak_dict(peak, vetting: dict, extras: dict) -> dict:
    flags = hard_flag_reasons(vetting)
    return {
        "period": float(peak.period),
        "epoch": float(peak.epoch),
        "power": float(peak.power),
        "sde": float(peak.sde),
        "duration_phase": float(getattr(peak, "duration_phase", 0.05) or 0.05),
        "vetting": vetting,
        "hard_flags": flags,
        "global_view": extras.get("global_view"),
        "local_view": extras.get("local_view"),
        "centroid_offset": extras.get("centroid_offset"),
        "diff_image": extras.get("diff_image"),
        "triceratops": extras.get("triceratops"),
        "kind": "periodic",
    }


def _vet_peak(
    time, flux, peak, *,
    tic_id, sector, tpf_time, tpf_flux_cube, ra, dec, bls_result: BLSResult,
    run_triceratops: bool,
) -> dict:
    from .difference_image import difference_image_score
    from .fold_features import run_phase3

    global_view, local_view, centroid, phase3_extras = run_phase3(
        time, flux, peak.period, peak.epoch,
        tic_id=tic_id, sector=str(sector), use_cache=False,
        bls_sde=peak.sde, bls_periods=bls_result.periods, bls_power=bls_result.power,
        tpf_time=tpf_time, tpf_flux_cube=tpf_flux_cube,
        ra=ra, dec=dec,
        skip_gaia=False,
    )
    vetting = {}
    if isinstance(phase3_extras, dict):
        vetting = dict(phase3_extras.get("vetting") or {})
    vetting.setdefault("transit_center_btjd", float(peak.epoch))
    half = float(getattr(peak, "duration_phase", 0.05) or 0.05) / 2.0
    diff = {}
    if tpf_time is not None and tpf_flux_cube is not None:
        diff = difference_image_score(tpf_time, tpf_flux_cube, peak.period, peak.epoch, half_width=max(half, 0.01))
        if diff.get("available"):
            vetting["diff_image_offset_px"] = diff["diff_image_offset_px"]
            vetting["diff_image_significance"] = diff["diff_image_significance"]
    triceratops = {"available": False}
    if run_triceratops:
        from .triceratops_validate import validate_fpp
        depth = vetting.get("depth")
        depth_ppm = float(depth) * 1e6 if isinstance(depth, (int, float)) and depth == depth else float("nan")
        triceratops = validate_fpp(
            tic_id, peak.period, peak.epoch,
            depth_ppm=depth_ppm,
            duration_hours=float(vetting.get("duration_hours", float("nan"))),
        )
        vetting["fpp"] = triceratops.get("fpp")
        vetting["nfpp"] = triceratops.get("nfpp")
    return _peak_dict(peak, vetting, {
        "global_view": global_view,
        "local_view": local_view,
        "centroid_offset": centroid,
        "diff_image": diff,
        "triceratops": triceratops,
    })


def tls_on_sde_peaks(time, flux, peaks, sde_threshold: float, max_peaks: int = 3) -> list:
    """Refine only peaks that already cleared the SDE gate."""
    passing = [p for p in peaks if getattr(p, "sde", 0.0) >= sde_threshold]
    if not passing:
        return []
    from .tls_refine import refine_peaks
    return refine_peaks(time, flux, passing, max_peaks=max_peaks)


def cached_bls(
    time, flux, *,
    tic_id: str, sector: str,
    period_min: float, period_max: float, nperiods: int,
    backend: Optional[str], rho_star_solar: float,
    adaptive_duration: bool, use_cache: bool,
    topk: int = 5,
    two_pass: bool = True,
    coarse_nperiods: int = 2000,
    refine_nperiods: int = 3000,
) -> BLSResult:
    """Phase-2 cache lookup. A second call with the same TIC/sector/grid is free."""
    from .cache_io import get_phase2, set_phase2

    if use_cache:
        cached = get_phase2(tic_id, sector, period_min, period_max, nperiods)
        if cached is not None:
            periods, power, best_period, epoch, peaks = cached
            sde_arr = compute_sde(power)
            return BLSResult(
                periods=periods, power=power, best_period=best_period,
                epoch=epoch, sde=float(np.max(sde_arr)) if len(sde_arr) else 0.0,
                peaks=peaks,
            )
    res = compute_bls_periodogram(
        time, flux,
        period_min=period_min, period_max=period_max, nperiods=nperiods,
        duration_phase=None if adaptive_duration else 0.05,
        backend=backend, rho_star_solar=rho_star_solar, topk=topk,
        two_pass=two_pass, coarse_nperiods=coarse_nperiods, refine_nperiods=refine_nperiods,
    )
    if use_cache:
        set_phase2(
            tic_id, sector, period_min, period_max, nperiods,
            res.periods, res.power, res.best_period, res.epoch, peaks=res.peaks,
        )
    return res


def search_with_masking(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    tic_id: str,
    sector: str,
    period_min: float = 0.5,
    period_max: float = 15.0,
    nperiods: int = 400,
    backend: Optional[str] = "numpy",
    rho_star_solar: float = 1.0,
    adaptive_duration: bool = True,
    sde_threshold: float = 7.0,
    use_cache: bool = True,
    use_tls: bool = True,
    max_planets: int = 3,
    tpf_time=None,
    tpf_flux_cube=None,
    ra: float = float("nan"),
    dec: float = float("nan"),
    known_period: Optional[float] = None,
    known_epoch: Optional[float] = None,
    run_triceratops: bool = True,
    run_monotransit: bool = True,
    two_pass: bool = True,
    coarse_nperiods: int = 2000,
    refine_nperiods: int = 3000,
) -> dict:
    """BLS, vet every SDE-pass peak, mask accepted transits, then MES.

    Returns ``signals`` (periodic, possibly several), ``monotransit`` or
    None, and the first :class:`BLSResult` (``bls``).
    """
    work = np.asarray(flux, dtype=np.float64).copy()
    if known_period is not None and np.isfinite(known_period) and known_period > 0:
        work = mask_known_signal(time, work, float(known_period), known_epoch)
        use_cache = False  # the masked curve is not the cached original

    signals: List[dict] = []
    first_bls: Optional[BLSResult] = None
    for planet_index in range(max(int(max_planets), 1)):
        res = cached_bls(
            time, work,
            tic_id=tic_id,
            sector=sector if planet_index == 0 else f"{sector}_p{planet_index}",
            period_min=period_min, period_max=period_max, nperiods=nperiods,
            backend=backend, rho_star_solar=rho_star_solar,
            adaptive_duration=adaptive_duration,
            use_cache=use_cache and planet_index == 0,
            two_pass=two_pass,
            coarse_nperiods=coarse_nperiods,
            refine_nperiods=refine_nperiods,
        )
        if first_bls is None:
            first_bls = res
        passing = [p for p in res.peaks if p.sde >= sde_threshold]
        if res.sde < sde_threshold or not passing:
            break
        if use_tls:
            refinements = tls_on_sde_peaks(time, work, passing, sde_threshold, max_peaks=3)
            for peak, refined in zip(passing, refinements):
                if getattr(refined, "available", False) and refined.sde > peak.sde:
                    peak.period = float(refined.period)
                    peak.epoch = float(refined.epoch)
                    peak.sde = float(refined.sde)
        accepted = None
        for peak in passing:
            signal = _vet_peak(
                time, work, peak,
                tic_id=tic_id, sector=sector,
                tpf_time=tpf_time, tpf_flux_cube=tpf_flux_cube,
                ra=ra, dec=dec, bls_result=res,
                run_triceratops=run_triceratops,
            )
            signals.append(signal)
            if accepted is None and accepts_for_reseach(signal["vetting"], peak.sde, sde_threshold):
                accepted = signal
        if accepted is None:
            break
        dur_days = _duration_days(accepted["period"], accepted["duration_phase"])
        work = mask_transits(time, work, accepted["period"], accepted["epoch"], dur_days)
        use_cache = False

    mono = None
    if run_monotransit:
        from .single_event import looks_like_monotransit
        # Don't re-report an event that the periodic search already accepted.
        mono = looks_like_monotransit(
            time, work,
            period_min=period_min, period_max=max(period_max, 100.0),
            rho_star_solar=rho_star_solar,
        )
        if mono is not None and signals:
            # If the monotransit epoch sits on an accepted periodic transit, drop it.
            for sig in signals:
                dist = abs((mono["epoch"] - sig["epoch"]) % sig["period"])
                dist = min(dist, sig["period"] - dist)
                if dist < _duration_days(sig["period"], sig["duration_phase"]):
                    mono = None
                    break

    return {
        "signals": signals,
        "monotransit": mono,
        "bls": first_bls,
    }
