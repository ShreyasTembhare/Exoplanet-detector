"""
Phase 2: GPU-accelerated BLS periodogram.
Broadcasts time vs a grid of trial periods; peaks in power indicate candidate periods.
Backend: JAX (GPU/CPU) with NumPy fallback if JAX not installed.

Two-pass strategy:
  Pass A (coarse): fewer periods over full range -> find peak region.
  Pass B (refine): dense grid around the coarse peak -> precise best_period.
Auto-policy: selects JAX vs NumPy based on cadence count and available memory.
Optional downsampling before BLS to speed up long light curves.
"""

import logging
import time as _time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import jax.numpy as jnp
    from jax import jit, vmap

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Backend kernels
# ---------------------------------------------------------------------------

def _bls_power_numpy(
    time: np.ndarray,
    flux: np.ndarray,
    periods: np.ndarray,
    duration_phase: float = 0.05,
    n_phase_steps: int = 20,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    n_periods = len(periods)
    power_grid = np.zeros((n_periods, n_phase_steps))
    flux = np.asarray(flux, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)

    for i, period in enumerate(periods):
        phase = (time % period) / period
        for j in range(n_phase_steps):
            phi = j / n_phase_steps
            in_transit = (phase >= phi) & (phase < (phi + duration_phase) % 1.0)
            if phi + duration_phase > 1:
                in_transit = in_transit | (phase < (phi + duration_phase - 1.0))
            n_in = np.sum(in_transit)
            n_out = len(flux) - n_in
            if n_in < 2 or n_out < 2:
                power_grid[i, j] = 0.0
                continue
            y_in = np.mean(flux[in_transit])
            y_out = np.mean(flux[~in_transit])
            depth = y_out - y_in
            power_grid[i, j] = depth ** 2 * (n_in * n_out) / (n_in + n_out)

    power = np.max(power_grid, axis=1)
    best_idx = np.argmax(power)
    best_period = float(periods[best_idx])
    best_phase_idx = np.argmax(power_grid[best_idx])
    best_phase = best_phase_idx / n_phase_steps
    t0 = np.nanmin(time)
    epoch = t0 + best_phase * best_period
    return periods, power, best_period, float(epoch)


def _bls_power_jax(
    time: np.ndarray,
    flux: np.ndarray,
    periods: np.ndarray,
    duration_phase: float = 0.05,
    n_phase_steps: int = 20,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    time = jnp.asarray(time)
    flux = jnp.asarray(flux)
    periods = jnp.asarray(periods)

    def power_one(period: float, phi: float) -> float:
        phase = (time % period) / period
        in_no_wrap = (phase >= phi) & (phase < phi + duration_phase)
        in_wrap = (phase >= phi) | (phase < phi + duration_phase - 1.0)
        wrap = (phi + duration_phase > 1.0)
        in_transit = jnp.where(wrap, in_wrap, in_no_wrap)
        n_in = jnp.sum(in_transit.astype(jnp.float32))
        n_out = len(flux) - n_in
        sum_in = jnp.sum(jnp.where(in_transit, flux, 0.0))
        sum_out = jnp.sum(jnp.where(~in_transit, flux, 0.0))
        y_in = sum_in / (n_in + 1e-10)
        y_out = sum_out / (n_out + 1e-10)
        depth = y_out - y_in
        return depth ** 2 * (n_in * n_out) / (n_in + n_out + 1e-10)

    phase_offsets = jnp.linspace(0, 1 - 1 / n_phase_steps, n_phase_steps)

    def max_power_per_period(period: float) -> float:
        powers = vmap(lambda phi: power_one(period, phi))(phase_offsets)
        return jnp.max(powers)

    power = vmap(max_power_per_period)(periods)
    power = np.asarray(power)
    periods_np = np.asarray(periods)
    best_idx = int(np.argmax(power))
    best_period = float(periods_np[best_idx])

    time_np = np.asarray(time)
    flux_np = np.asarray(flux)
    phase_np = (time_np % best_period) / best_period
    t0 = float(np.nanmin(time_np))
    best_phase = 0.5
    best_depth = -1.0
    for j in range(n_phase_steps):
        phi = j / n_phase_steps
        in_transit = (phase_np >= phi) & (phase_np < (phi + duration_phase) % 1.0)
        if phi + duration_phase > 1:
            in_transit = in_transit | (phase_np < (phi + duration_phase - 1.0))
        if np.sum(in_transit) < 2:
            continue
        y_in = np.mean(flux_np[in_transit])
        y_out = np.mean(flux_np[~in_transit])
        d = y_out - y_in
        if d > best_depth:
            best_depth = d
            best_phase = phi
    epoch = t0 + best_phase * best_period
    return periods_np, power, best_period, float(epoch)


# ---------------------------------------------------------------------------
# Auto-policy
# ---------------------------------------------------------------------------

def _select_backend(n_cadences: int, use_jax: Optional[bool] = None) -> str:
    """Pick JAX or NumPy based on cadence count and availability."""
    if use_jax is True and JAX_AVAILABLE:
        return "jax"
    if use_jax is False or not JAX_AVAILABLE:
        return "numpy"
    if n_cadences <= 50000:
        return "jax" if JAX_AVAILABLE else "numpy"
    return "numpy"


def _downsample(time: np.ndarray, flux: np.ndarray, max_cadences: int) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform downsample to max_cadences if longer."""
    if len(time) <= max_cadences:
        return time, flux
    idx = np.round(np.linspace(0, len(time) - 1, max_cadences)).astype(int)
    return time[idx], flux[idx]


# ---------------------------------------------------------------------------
# Two-pass BLS
# ---------------------------------------------------------------------------

def compute_bls_periodogram(
    time: np.ndarray,
    flux: np.ndarray,
    period_min: float = 0.5,
    period_max: float = 20.0,
    nperiods: int = 5000,
    duration_phase: float = 0.05,
    use_jax: Optional[bool] = None,
    two_pass: bool = True,
    coarse_nperiods: int = 2000,
    refine_nperiods: int = 3000,
    refine_half_width: float = 0.15,
    downsample_limit: int = 80000,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute BLS periodogram with optional two-pass strategy.
    Pass A: coarse grid -> find peak region.
    Pass B: dense grid around peak -> precise period.
    Returns (periods, power, best_period, epoch).
    """
    backend = _select_backend(len(time), use_jax)
    bls_fn = _bls_power_jax if backend == "jax" else _bls_power_numpy

    ds_time, ds_flux = _downsample(time, flux, downsample_limit)
    logger.info("BLS backend=%s, cadences=%d (ds=%d), two_pass=%s",
                backend, len(time), len(ds_time), two_pass)

    if not two_pass:
        periods = np.exp(np.linspace(np.log(period_min), np.log(period_max), nperiods))
        return bls_fn(ds_time, ds_flux, periods, duration_phase=duration_phase)

    # Pass A: coarse
    t0 = _time.time()
    coarse_periods = np.exp(np.linspace(np.log(period_min), np.log(period_max), coarse_nperiods))
    _, coarse_power, coarse_best, _ = bls_fn(ds_time, ds_flux, coarse_periods, duration_phase=duration_phase)
    coarse_ms = (_time.time() - t0) * 1000
    logger.info("  Coarse pass: %d periods in %.0f ms -> peak=%.4f d", coarse_nperiods, coarse_ms, coarse_best)

    # Pass B: refine around peak
    t0 = _time.time()
    refine_lo = max(period_min, coarse_best * (1 - refine_half_width))
    refine_hi = min(period_max, coarse_best * (1 + refine_half_width))
    refine_periods = np.linspace(refine_lo, refine_hi, refine_nperiods)
    _, refine_power, best_period, epoch = bls_fn(ds_time, ds_flux, refine_periods, duration_phase=duration_phase)
    refine_ms = (_time.time() - t0) * 1000
    logger.info("  Refine pass: %d periods [%.3f-%.3f] in %.0f ms -> best=%.4f d",
                refine_nperiods, refine_lo, refine_hi, refine_ms, best_period)

    all_periods = np.concatenate([coarse_periods, refine_periods])
    all_power = np.concatenate([coarse_power, refine_power])
    sort_idx = np.argsort(all_periods)
    return all_periods[sort_idx], all_power[sort_idx], best_period, epoch


# ---------------------------------------------------------------------------
# Phase 2 runner
# ---------------------------------------------------------------------------

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
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Run Phase 2: BLS periodogram with two-pass optimization.
    Returns (periods, power, best_period, epoch).
    """
    from .cache_io import get_phase2, set_phase2

    if use_cache:
        cached = get_phase2(tic_id, sector, period_min, period_max, nperiods)
        if cached is not None:
            return cached

    periods, power, best_period, epoch = compute_bls_periodogram(
        time, flux, period_min=period_min, period_max=period_max, nperiods=nperiods,
        two_pass=two_pass,
    )
    if use_cache:
        set_phase2(tic_id, sector, period_min, period_max, nperiods, periods, power, best_period, epoch)
    return periods, power, best_period, epoch
