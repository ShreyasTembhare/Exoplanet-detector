"""
Phase 2: GPU-accelerated BLS periodogram.

Backends:
  - jax        : JIT-compiled JAX kernel, float32, chunked period grids (default GPU).
  - cuvarbase  : optional hand-tuned CUDA BLS (used iff installed and CUDA available).
  - astropy    : oracle / CPU fallback using ``astropy.timeseries.BoxLeastSquares``.
  - numpy      : slow Python loop (only used as a last-resort fallback).

Two-pass strategy:
  Pass A (coarse): fewer periods over full range -> find peak region.
  Pass B (refine): dense grid around the coarse peak -> precise best_period.

Top-k peaks are extracted from the coarse pass with non-maximum suppression on
period ratios (1/2, 1, 2, 3) so harmonics and aliases are deduplicated. The
classifier downstream gets to vet all of them, not just the global argmax.

All power values returned to the rest of the pipeline are non-negative: depth
is clamped to ``max(y_out - y_in, 0)`` so stellar flares, brightening events,
and other dips-up cannot masquerade as transits.
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, lax

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    JAX_AVAILABLE = False

try:
    import cuvarbase.bls as _cuvb_bls  # type: ignore[import]

    CUVARBASE_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    CUVARBASE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BLSPeak:
    """A single peak from the BLS periodogram with its phase / epoch."""

    period: float
    epoch: float
    power: float
    sde: float
    duration_phase: float = 0.05
    depth: float = float("nan")


@dataclass
class BLSResult:
    """Full periodogram + the top-k deduplicated peaks."""

    periods: np.ndarray
    power: np.ndarray
    best_period: float
    epoch: float
    sde: float
    peaks: List[BLSPeak] = field(default_factory=list)


# ---------------------------------------------------------------------------
# NumPy backend (correctness reference, single-period kernel)
# ---------------------------------------------------------------------------


def _bls_power_numpy(
    time: np.ndarray,
    flux: np.ndarray,
    periods: np.ndarray,
    duration_phase: float = 0.05,
    n_phase_steps: int = 20,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Plain NumPy BLS. Slow but obviously correct. Used for tiny inputs and tests.

    Bug fixes vs the previous version:
      * Wrap-around mask now ``replaces`` the no-wrap mask when the transit
        straddles phase=1, instead of OR-ing onto a mask that's silently
        always-False. The old code lost half of any wrapping transit.
      * ``depth`` is clamped to ``max(y_out - y_in, 0)`` so brightening events
        do not produce positive BLS power.
    """
    n_periods = len(periods)
    power_grid = np.zeros((n_periods, n_phase_steps), dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    n_total = len(flux)

    for i, period in enumerate(periods):
        phase = (time % period) / period
        for j in range(n_phase_steps):
            phi = j / n_phase_steps
            high = phi + duration_phase
            if high > 1.0:
                in_transit = (phase >= phi) | (phase < high - 1.0)
            else:
                in_transit = (phase >= phi) & (phase < high)
            n_in = int(np.sum(in_transit))
            n_out = n_total - n_in
            if n_in < 2 or n_out < 2:
                power_grid[i, j] = 0.0
                continue
            y_in = float(np.mean(flux[in_transit]))
            y_out = float(np.mean(flux[~in_transit]))
            depth = max(y_out - y_in, 0.0)
            power_grid[i, j] = depth ** 2 * (n_in * n_out) / (n_in + n_out)

    power = np.max(power_grid, axis=1)
    best_idx = int(np.argmax(power))
    best_period = float(periods[best_idx])
    best_phase_idx = int(np.argmax(power_grid[best_idx]))
    best_phase = best_phase_idx / n_phase_steps
    t0 = float(np.nanmin(time))
    epoch = t0 + best_phase * best_period
    return periods, power, best_period, float(epoch)


# ---------------------------------------------------------------------------
# JAX backend (JIT + chunked + float32)
# ---------------------------------------------------------------------------


def _build_jax_kernel(n_phase_steps: int, duration_phase: float):
    """Build (and cache) a jit-compiled function that computes BLS power for a
    chunk of trial periods. The closure-bound ``n_phase_steps`` /
    ``duration_phase`` let JAX trace once per (steps, dur) pair."""

    if not JAX_AVAILABLE:  # pragma: no cover
        return None

    phase_offsets = jnp.linspace(
        0.0, 1.0 - 1.0 / n_phase_steps, n_phase_steps, dtype=jnp.float32
    )

    def power_one(period, phi, time, flux):
        phase = (time % period) / period
        high = phi + duration_phase
        in_no_wrap = (phase >= phi) & (phase < high)
        in_wrap = (phase >= phi) | (phase < high - 1.0)
        wrap = high > 1.0
        in_transit = jnp.where(wrap, in_wrap, in_no_wrap)
        n_in = jnp.sum(in_transit.astype(jnp.float32))
        n_out = jnp.float32(flux.shape[0]) - n_in
        sum_in = jnp.sum(jnp.where(in_transit, flux, jnp.float32(0.0)))
        sum_out = jnp.sum(jnp.where(in_transit, jnp.float32(0.0), flux))
        y_in = sum_in / (n_in + 1e-6)
        y_out = sum_out / (n_out + 1e-6)
        depth = jnp.maximum(y_out - y_in, jnp.float32(0.0))
        denom = n_in + n_out + 1e-6
        return depth ** 2 * (n_in * n_out) / denom

    def power_one_period(period, time, flux):
        # Scan over phase offsets; reduce with max.
        def body(carry, phi):
            cur_max, cur_phi = carry
            p = power_one(period, phi, time, flux)
            keep = p > cur_max
            return (jnp.where(keep, p, cur_max), jnp.where(keep, phi, cur_phi)), None

        init = (jnp.float32(0.0), jnp.float32(0.0))
        (best_p, best_phi), _ = lax.scan(body, init, phase_offsets)
        return best_p, best_phi

    @jit
    def power_chunk(periods_chunk, time, flux):
        return jax.vmap(lambda P: power_one_period(P, time, flux))(periods_chunk)

    return power_chunk


_JAX_KERNEL_CACHE: dict = {}


def _bls_power_jax(
    time: np.ndarray,
    flux: np.ndarray,
    periods: np.ndarray,
    duration_phase: float = 0.05,
    n_phase_steps: int = 20,
    chunk_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """JAX backend with JIT, scan-over-phase, and chunked period processing."""
    if not JAX_AVAILABLE:
        return _bls_power_numpy(time, flux, periods, duration_phase, n_phase_steps)

    key = (n_phase_steps, float(duration_phase))
    kernel = _JAX_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = _build_jax_kernel(n_phase_steps, duration_phase)
        _JAX_KERNEL_CACHE[key] = kernel

    time_j = jnp.asarray(time, dtype=jnp.float32)
    flux_j = jnp.asarray(flux, dtype=jnp.float32)

    powers = []
    best_phis = []
    n = int(len(periods))
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        per_chunk = jnp.asarray(periods[start:end], dtype=jnp.float32)
        p_chunk, phi_chunk = kernel(per_chunk, time_j, flux_j)
        powers.append(np.asarray(p_chunk))
        best_phis.append(np.asarray(phi_chunk))
    power = np.concatenate(powers).astype(np.float64)
    best_phi_per_period = np.concatenate(best_phis).astype(np.float64)

    best_idx = int(np.argmax(power))
    best_period = float(periods[best_idx])
    best_phase = float(best_phi_per_period[best_idx])
    t0 = float(np.nanmin(time))
    epoch = t0 + best_phase * best_period
    return periods, power, best_period, float(epoch)


# ---------------------------------------------------------------------------
# Astropy backend (oracle / CPU fallback)
# ---------------------------------------------------------------------------


def _bls_power_astropy(
    time: np.ndarray,
    flux: np.ndarray,
    periods: np.ndarray,
    duration_phase: float = 0.05,
    n_phase_steps: int = 20,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Astropy BoxLeastSquares — used as oracle in tests and CPU fallback."""
    from astropy.timeseries import BoxLeastSquares

    bls = BoxLeastSquares(time, flux)
    durations = np.array([duration_phase * float(np.median(periods))])
    res = bls.power(periods, durations)
    power = np.asarray(res.power, dtype=np.float64)
    # Clamp negative power (Astropy returns SR statistic that can be negative).
    power = np.maximum(power, 0.0)
    best_idx = int(np.argmax(power))
    best_period = float(periods[best_idx])
    epoch = float(res.transit_time[best_idx])
    return periods, power, best_period, epoch


# ---------------------------------------------------------------------------
# cuvarbase backend (optional CUDA)
# ---------------------------------------------------------------------------


def _bls_power_cuvarbase(
    time: np.ndarray,
    flux: np.ndarray,
    periods: np.ndarray,
    duration_phase: float = 0.05,
    n_phase_steps: int = 20,
) -> Tuple[np.ndarray, np.ndarray, float, float]:  # pragma: no cover - hardware-gated
    """Optional CUDA BLS via cuvarbase. Falls back to JAX if init fails."""
    if not CUVARBASE_AVAILABLE:
        return _bls_power_jax(time, flux, periods, duration_phase, n_phase_steps)
    try:
        bls_proc = _cuvb_bls.eebls_gpu_fast
        freqs = 1.0 / np.asarray(periods, dtype=np.float64)
        powers, best_phis = bls_proc(
            time.astype(np.float64), flux.astype(np.float64),
            freqs, qmin=duration_phase * 0.5, qmax=duration_phase * 1.5,
        )
        power = np.maximum(np.asarray(powers, dtype=np.float64), 0.0)
        best_idx = int(np.argmax(power))
        best_period = float(periods[best_idx])
        best_phase = float(best_phis[best_idx])
        t0 = float(np.nanmin(time))
        return periods, power, best_period, float(t0 + best_phase * best_period)
    except Exception as exc:
        logger.warning("cuvarbase failed (%s); falling back to JAX.", exc)
        return _bls_power_jax(time, flux, periods, duration_phase, n_phase_steps)


# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------

_BACKEND_FNS = {
    "numpy": _bls_power_numpy,
    "jax": _bls_power_jax,
    "astropy": _bls_power_astropy,
    "cuvarbase": _bls_power_cuvarbase,
}


def _select_backend(n_cadences: int, requested: Optional[str] = None) -> str:
    if requested is not None:
        if requested == "cuvarbase" and not CUVARBASE_AVAILABLE:
            logger.warning("cuvarbase not installed; using jax.")
            return "jax" if JAX_AVAILABLE else "astropy"
        if requested == "jax" and not JAX_AVAILABLE:
            return "astropy"
        return requested
    if JAX_AVAILABLE:
        return "jax"
    return "astropy"


def _downsample(time: np.ndarray, flux: np.ndarray, max_cadences: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(time) <= max_cadences:
        return time, flux
    idx = np.round(np.linspace(0, len(time) - 1, max_cadences)).astype(int)
    return time[idx], flux[idx]


# ---------------------------------------------------------------------------
# Vetting helpers
# ---------------------------------------------------------------------------


def compute_sde(power: np.ndarray) -> np.ndarray:
    """
    Signal Detection Efficiency. Subtract median, divide by MAD-based sigma.
    Cadence-count and noise-level invariant; the standard Kepler/TESS gate
    is ``SDE > 7``.
    """
    p = np.asarray(power, dtype=np.float64)
    med = np.median(p)
    mad = np.median(np.abs(p - med))
    sigma = 1.4826 * mad if mad > 0 else (np.std(p) or 1.0)
    return (p - med) / sigma


def _select_topk_peaks(
    periods: np.ndarray,
    power: np.ndarray,
    sde: np.ndarray,
    k: int = 5,
    harmonic_tol: float = 0.02,
) -> List[Tuple[int, float]]:
    """
    Pick the top-k peaks with non-maximum suppression on period ratios that
    are integer or half-integer harmonics. Returns list of (idx, period).
    """
    order = np.argsort(power)[::-1]
    chosen: List[Tuple[int, float]] = []
    harmonics = (0.5, 1.0, 2.0, 3.0, 1.0 / 3.0)
    for idx in order:
        P = float(periods[idx])
        if P <= 0:
            continue
        keep = True
        for _, P0 in chosen:
            for h in harmonics:
                if abs(P / P0 - h) / h < harmonic_tol:
                    keep = False
                    break
            if not keep:
                break
        if keep:
            chosen.append((int(idx), P))
            if len(chosen) >= k:
                break
    return chosen


# ---------------------------------------------------------------------------
# Two-pass BLS
# ---------------------------------------------------------------------------


def compute_bls_periodogram(
    time: np.ndarray,
    flux: np.ndarray,
    period_min: float = 0.5,
    period_max: float = 20.0,
    nperiods: int = 5000,
    duration_phase: Optional[float] = 0.05,
    backend: Optional[str] = None,
    two_pass: bool = True,
    coarse_nperiods: int = 2000,
    refine_nperiods: int = 3000,
    refine_half_width: float = 0.15,
    downsample_limit: int = 80000,
    topk: int = 5,
    rho_star_solar: Optional[float] = None,
) -> BLSResult:
    """
    Compute BLS periodogram and return a :class:`BLSResult` with top-k peaks.

    The previous signature returned a 4-tuple ``(periods, power, best_period, epoch)``;
    callers that still need that should use :func:`compute_bls_legacy`.

    If ``duration_phase`` is None, an adaptive duration is chosen per pass
    using Kepler's 3rd law (Phase 1 :mod:`physical_priors`). This is much
    closer to physical reality than a fixed 0.05 across all periods.
    """
    backend_name = _select_backend(len(time), backend)
    bls_fn = _BACKEND_FNS[backend_name]

    if duration_phase is None:
        from .physical_priors import duration_phase_for_period
        rho = rho_star_solar if rho_star_solar is not None else 1.0
        # Use the geometric mean of the period range as the representative.
        rep_period = float(np.sqrt(period_min * period_max))
        duration_phase = duration_phase_for_period(rep_period, rho)
        logger.info("Adaptive duration_phase=%.4f (rho=%.2f, period~%.2fd)",
                    duration_phase, rho, rep_period)

    ds_time, ds_flux = _downsample(time, flux, downsample_limit)
    logger.info(
        "BLS backend=%s, cadences=%d (ds=%d), two_pass=%s",
        backend_name, len(time), len(ds_time), two_pass,
    )

    if not two_pass:
        periods = np.exp(np.linspace(np.log(period_min), np.log(period_max), nperiods))
        periods, power, best_period, epoch = bls_fn(ds_time, ds_flux, periods, duration_phase=duration_phase)
        sde = compute_sde(power)
        peaks = _build_peaks(periods, power, sde, ds_time, best_period, topk, duration_phase)
        return BLSResult(periods=periods, power=power, best_period=best_period, epoch=epoch,
                         sde=float(np.max(sde)), peaks=peaks)

    # Pass A: coarse log-spaced grid.
    t0 = _time.time()
    coarse_periods = np.exp(np.linspace(np.log(period_min), np.log(period_max), coarse_nperiods))
    coarse_periods, coarse_power, coarse_best, coarse_epoch = bls_fn(
        ds_time, ds_flux, coarse_periods, duration_phase=duration_phase,
    )
    coarse_sde = compute_sde(coarse_power)
    logger.info(
        "  Coarse pass: %d periods in %.0f ms -> peak=%.4f d (SDE=%.2f)",
        coarse_nperiods, (_time.time() - t0) * 1000, coarse_best, float(np.max(coarse_sde)),
    )

    coarse_peak_idx = _select_topk_peaks(coarse_periods, coarse_power, coarse_sde, k=topk)

    # Pass B: refine around the global best peak.
    t0 = _time.time()
    refine_lo = max(period_min, coarse_best * (1 - refine_half_width))
    refine_hi = min(period_max, coarse_best * (1 + refine_half_width))
    refine_periods = np.linspace(refine_lo, refine_hi, refine_nperiods)
    refine_periods, refine_power, best_period, epoch = bls_fn(
        ds_time, ds_flux, refine_periods, duration_phase=duration_phase,
    )
    logger.info(
        "  Refine pass: %d periods [%.3f-%.3f] in %.0f ms -> best=%.4f d",
        refine_nperiods, refine_lo, refine_hi, (_time.time() - t0) * 1000, best_period,
    )

    all_periods = np.concatenate([coarse_periods, refine_periods])
    all_power = np.concatenate([coarse_power, refine_power])
    sort_idx = np.argsort(all_periods)
    all_periods = all_periods[sort_idx]
    all_power = all_power[sort_idx]
    sde = compute_sde(all_power)

    # Build peaks from the union, anchored on the coarse-pass selections so
    # we capture multi-planet candidates.
    peaks = _build_peaks(all_periods, all_power, sde, ds_time, best_period, topk, duration_phase,
                         seed_periods=[p for _, p in coarse_peak_idx])
    return BLSResult(
        periods=all_periods, power=all_power,
        best_period=best_period, epoch=epoch,
        sde=float(np.max(sde)), peaks=peaks,
    )


def _build_peaks(
    periods: np.ndarray,
    power: np.ndarray,
    sde: np.ndarray,
    time: np.ndarray,
    best_period: float,
    topk: int,
    duration_phase: float,
    seed_periods: Optional[List[float]] = None,
) -> List[BLSPeak]:
    """Select top-k peaks (with NMS) and recover epoch for each via NumPy fold."""
    selected = _select_topk_peaks(periods, power, sde, k=topk)
    if seed_periods:
        # Ensure the seeded coarse peaks are present.
        existing = {round(p, 6) for _, p in selected}
        for P in seed_periods:
            if round(P, 6) not in existing:
                near = int(np.argmin(np.abs(periods - P)))
                selected.append((near, float(periods[near])))
        selected = selected[:topk]

    out: List[BLSPeak] = []
    t0 = float(np.nanmin(time))
    for idx, P in selected:
        epoch = _epoch_for_period(time, P, duration_phase)
        out.append(
            BLSPeak(
                period=float(P),
                epoch=float(epoch),
                power=float(max(power[idx], 0.0)),
                sde=float(sde[idx]),
                duration_phase=duration_phase,
                depth=float("nan"),
            )
        )
    if not out:
        out.append(
            BLSPeak(
                period=float(best_period),
                epoch=t0,
                power=float(np.max(power)),
                sde=float(np.max(sde)),
                duration_phase=duration_phase,
            )
        )
    return out


def _epoch_for_period(
    time: np.ndarray,
    period: float,
    duration_phase: float,
    n_phase_steps: int = 64,
) -> float:
    """Find best transit epoch for a given period via a NumPy phase-offset scan
    using the **fixed** wrap-around mask."""
    time = np.asarray(time, dtype=np.float64)
    phase = (time % period) / period
    best_phi = 0.5
    best_score = -1.0
    for j in range(n_phase_steps):
        phi = j / n_phase_steps
        high = phi + duration_phase
        if high > 1.0:
            in_transit = (phase >= phi) | (phase < high - 1.0)
        else:
            in_transit = (phase >= phi) & (phase < high)
        n_in = int(np.sum(in_transit))
        if n_in < 2:
            continue
        # Use mean depth as scoring proxy for epoch refinement.
        score = float(n_in)
        if score > best_score:
            best_score = score
            best_phi = phi
    t0 = float(np.nanmin(time))
    return t0 + best_phi * period


# ---------------------------------------------------------------------------
# Phase 2 runners (legacy + structured)
# ---------------------------------------------------------------------------


def compute_bls_legacy(*args, **kwargs) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Backward-compatible 4-tuple return for callers that haven't migrated."""
    res = compute_bls_periodogram(*args, **kwargs)
    return res.periods, res.power, res.best_period, res.epoch


def run_phase2(
    time: np.ndarray,
    flux: np.ndarray,
    tic_id: str,
    sector: str,
    period_min: float = 0.5,
    period_max: float = 20.0,
    nperiods: int = 5000,
    use_cache: bool = True,
    two_pass: bool = True,
    backend: Optional[str] = None,
    topk: int = 5,
    return_result: bool = False,
):
    """
    Run Phase 2: BLS periodogram with two-pass optimization.

    Returns a 4-tuple ``(periods, power, best_period, epoch)`` for backward
    compatibility, **or** a :class:`BLSResult` if ``return_result=True``. The
    cache always stores the structured form (peaks included).
    """
    from .cache_io import get_phase2, set_phase2

    if use_cache:
        cached = get_phase2(tic_id, sector, period_min, period_max, nperiods)
        if cached is not None:
            periods, power, best_period, epoch, peaks = cached
            sde_arr = compute_sde(power)
            res = BLSResult(periods=periods, power=power, best_period=best_period,
                            epoch=epoch, sde=float(np.max(sde_arr)), peaks=peaks)
            return res if return_result else (periods, power, best_period, epoch)

    res = compute_bls_periodogram(
        time, flux, period_min=period_min, period_max=period_max, nperiods=nperiods,
        two_pass=two_pass, backend=backend, topk=topk,
    )
    if use_cache:
        set_phase2(tic_id, sector, period_min, period_max, nperiods,
                   res.periods, res.power, res.best_period, res.epoch, peaks=res.peaks)
    return res if return_result else (res.periods, res.power, res.best_period, res.epoch)
