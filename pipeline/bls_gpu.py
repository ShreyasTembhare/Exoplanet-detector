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


def _astropy_available() -> bool:
    try:
        import astropy.timeseries  # noqa: F401
        return True
    except Exception:
        return False


def _fastest_backend() -> str:
    """JAX when installed, otherwise Astropy, otherwise the NumPy loop.

    Astropy remains the reference oracle: pass ``backend="astropy"`` in tests.
    """
    if JAX_AVAILABLE:
        return "jax"
    if _astropy_available():
        return "astropy"
    return "numpy"


def _select_backend(n_cadences: int, requested: Optional[str] = None) -> str:
    """Pick the fastest installed backend unless the caller names one."""
    del n_cadences  # kept for call-site compatibility
    if requested is None:
        return _fastest_backend()
    if requested == "cuvarbase" and not CUVARBASE_AVAILABLE:
        logger.warning("cuvarbase not installed; using the fastest built-in backend.")
        return _fastest_backend()
    if requested == "jax" and not JAX_AVAILABLE:
        logger.warning("jax not installed; using the fastest built-in backend.")
        return _fastest_backend()
    if requested == "astropy" and not _astropy_available():
        logger.warning("astropy not installed; using the NumPy BLS loop.")
        return "numpy"
    return requested


def _downsample(
    time: np.ndarray,
    flux: np.ndarray,
    max_cadences: int,
    min_duration_days: Optional[float] = None,
    min_samples_in_transit: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample only when the new cadence still resolves the shortest duration.

    A blind cap (historically 80_000 samples) smears ultra-short transits on a
    long baseline. If ``max_cadences`` would leave fewer than
    ``min_samples_in_transit`` samples inside ``min_duration_days``, the cap is
    raised so that box stays resolved.
    """
    time = np.asarray(time)
    flux = np.asarray(flux)
    n = int(len(time))
    if n <= max_cadences:
        return time, flux
    limit = int(max_cadences)
    if (
        min_duration_days is not None
        and min_duration_days > 0
        and n > 1
        and min_samples_in_transit > 0
    ):
        span = float(np.nanmax(time) - np.nanmin(time))
        if span > 0:
            max_dt = float(min_duration_days) / float(min_samples_in_transit)
            n_needed = int(np.ceil(span / max_dt)) + 1
            limit = max(limit, min(n, n_needed))
    if n <= limit:
        return time, flux
    idx = np.unique(np.round(np.linspace(0, n - 1, limit)).astype(int))
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


def _shortest_duration_days(
    period_min: float,
    duration_phase: Optional[float],
    rho_star_solar: float,
) -> float:
    """Absolute length (days) of the shortest box the search will try."""
    if duration_phase is not None:
        return float(duration_phase) * float(period_min)
    from .physical_priors import duration_phase_for_period

    # 0.5× is the short edge of adaptive_duration_grid.
    return 0.5 * duration_phase_for_period(float(period_min), rho_star_solar) * float(period_min)


def _power_on_grid(
    bls_fn,
    time: np.ndarray,
    flux: np.ndarray,
    periods: np.ndarray,
    duration_phase: Optional[float],
    rho_star_solar: float,
    n_durations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(power, duration_phase_used)`` aligned with ``periods``.

    A fixed ``duration_phase`` is one box for the whole grid (tests and the
    Astropy reference). ``duration_phase is None`` calls
    :func:`pipeline.physical_priors.adaptive_duration_grid` and scores about
    3–4 durations per period (0.5–1.5× the stellar-density estimate), in
    log-period bins so a short period and a long period do not share a width.
    """
    periods = np.asarray(periods, dtype=np.float64)
    if len(periods) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    if duration_phase is not None:
        _, power, _, _ = bls_fn(time, flux, periods, duration_phase=float(duration_phase))
        power = np.maximum(np.asarray(power, dtype=np.float64), 0.0)
        return power, np.full(len(periods), float(duration_phase))

    from .physical_priors import adaptive_duration_grid, duration_phase_for_period

    positive = periods[periods > 0]
    pmin = float(np.min(positive))
    pmax = float(np.max(positive))
    n_dur = int(np.clip(n_durations, 1, 4))
    rep_periods, dur_grid = adaptive_duration_grid(pmin, pmax, rho_star_solar, n=n_dur)
    base0 = duration_phase_for_period(float(rep_periods[0]), rho_star_solar)
    scales = np.asarray(dur_grid[0], dtype=np.float64) / max(base0, 1e-8)

    log_span = np.log(pmax / max(pmin, 1e-8))
    n_bins = int(np.clip(np.round(log_span / np.log(1.8)) if log_span > 0 else 1, 1, 8))
    order = np.argsort(periods)
    chunks = np.array_split(order, n_bins)
    power = np.zeros(len(periods), dtype=np.float64)
    used = np.full(len(periods), duration_phase_for_period(float(np.median(positive)), rho_star_solar))
    for ch in chunks:
        if len(ch) == 0:
            continue
        p_chunk = periods[ch]
        p_med = float(np.median(p_chunk[p_chunk > 0])) if np.any(p_chunk > 0) else pmin
        base = duration_phase_for_period(p_med, rho_star_solar)
        best_pw = np.full(len(ch), -1.0)
        best_d = np.full(len(ch), base)
        for scale in scales:
            dur = float(np.clip(base * float(scale), 0.005, 0.25))
            _, pw, _, _ = bls_fn(time, flux, p_chunk, duration_phase=dur)
            pw = np.maximum(np.asarray(pw, dtype=np.float64), 0.0)
            better = pw > best_pw
            best_pw = np.where(better, pw, best_pw)
            best_d = np.where(better, dur, best_d)
        power[ch] = best_pw
        used[ch] = best_d
    return power, used


def _refine_period(
    bls_fn,
    time: np.ndarray,
    flux: np.ndarray,
    period_seed: float,
    period_min: float,
    period_max: float,
    n_refine: int,
    half_width: float,
    duration_phase: Optional[float],
    rho_star_solar: float,
    n_durations: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fine grid around one coarse peak. Returns periods, power, durations."""
    seed = float(period_seed)
    lo = max(float(period_min), seed * (1.0 - half_width))
    hi = min(float(period_max), seed * (1.0 + half_width))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = max(float(period_min), seed * 0.99)
        hi = min(float(period_max), max(lo * 1.01, seed * 1.01))
    grid = np.linspace(lo, hi, max(int(n_refine), 2))
    power, used = _power_on_grid(
        bls_fn, time, flux, grid, duration_phase, rho_star_solar, n_durations,
    )
    return grid, power, used


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
    n_durations: int = 4,
) -> BLSResult:
    """
    Compute BLS periodogram and return a :class:`BLSResult` with top-k peaks.

    The previous signature returned a 4-tuple ``(periods, power, best_period, epoch)``;
    callers that still need that should use :func:`compute_bls_legacy`.

    If ``duration_phase`` is None, each period is scored at a few durations from
    :func:`pipeline.physical_priors.adaptive_duration_grid` (about 3–4), not one
    geometric-mean box for the whole grid.

    Pass B refines every kept top-k peak, not only the global best. Epochs
    maximize mean in-transit depth.
    """
    backend_name = _select_backend(len(time), backend)
    bls_fn = _BACKEND_FNS[backend_name]
    rho = float(rho_star_solar) if rho_star_solar is not None else 1.0
    n_durations = int(np.clip(n_durations, 1, 4))

    if duration_phase is None:
        logger.info(
            "Per-period duration grid: %d durations (rho=%.2f, P=%.3f–%.3f d)",
            n_durations, rho, period_min, period_max,
        )

    min_dur = _shortest_duration_days(period_min, duration_phase, rho)
    ds_time, ds_flux = _downsample(
        time, flux, downsample_limit, min_duration_days=min_dur,
    )
    logger.info(
        "BLS backend=%s, cadences=%d (ds=%d), two_pass=%s",
        backend_name, len(time), len(ds_time), two_pass,
    )

    def _result(periods, power, used, seed_periods=None) -> BLSResult:
        sde = compute_sde(power)
        best_idx = int(np.argmax(power)) if len(power) else 0
        best_period = float(periods[best_idx]) if len(periods) else float(period_min)
        dur_best = float(used[best_idx]) if len(used) else 0.05
        epoch = _epoch_for_period(ds_time, ds_flux, best_period, dur_best)
        peaks = _build_peaks(
            periods, power, sde, ds_time, ds_flux, best_period, topk, used,
            seed_periods=seed_periods,
        )
        return BLSResult(
            periods=periods, power=power, best_period=best_period, epoch=epoch,
            sde=float(np.max(sde)) if len(sde) else 0.0, peaks=peaks,
        )

    if not two_pass:
        periods = np.exp(np.linspace(np.log(period_min), np.log(period_max), nperiods))
        power, used = _power_on_grid(
            bls_fn, ds_time, ds_flux, periods, duration_phase, rho, n_durations,
        )
        return _result(periods, power, used)

    # Pass A: coarse log-spaced grid.
    t0 = _time.time()
    coarse_periods = np.exp(np.linspace(np.log(period_min), np.log(period_max), coarse_nperiods))
    coarse_power, coarse_used = _power_on_grid(
        bls_fn, ds_time, ds_flux, coarse_periods, duration_phase, rho, n_durations,
    )
    coarse_sde = compute_sde(coarse_power)
    coarse_best = float(coarse_periods[int(np.argmax(coarse_power))])
    logger.info(
        "  Coarse pass: %d periods in %.0f ms -> peak=%.4f d (SDE=%.2f)",
        coarse_nperiods, (_time.time() - t0) * 1000, coarse_best, float(np.max(coarse_sde)),
    )

    coarse_peaks = _select_topk_peaks(coarse_periods, coarse_power, coarse_sde, k=topk)

    # Pass B: fine grid around every kept top-k peak (and the global best).
    t0 = _time.time()
    seeds: List[float] = []
    seen = set()
    for pseed in [coarse_best] + [p for _, p in coarse_peaks]:
        key = round(float(pseed), 6)
        if key in seen or not np.isfinite(pseed) or pseed <= 0:
            continue
        seen.add(key)
        seeds.append(float(pseed))
        if len(seeds) >= max(int(topk), 1):
            break

    refine_periods_all: List[np.ndarray] = []
    refine_power_all: List[np.ndarray] = []
    refine_used_all: List[np.ndarray] = []
    for pseed in seeds:
        grid, pw, used = _refine_period(
            bls_fn, ds_time, ds_flux, pseed,
            period_min, period_max, refine_nperiods, refine_half_width,
            duration_phase, rho, n_durations,
        )
        refine_periods_all.append(grid)
        refine_power_all.append(pw)
        refine_used_all.append(used)
    logger.info(
        "  Refine pass: %d peaks × %d periods in %.0f ms",
        len(seeds), refine_nperiods, (_time.time() - t0) * 1000,
    )

    refine_periods = np.concatenate(refine_periods_all) if refine_periods_all else coarse_periods[:0]
    refine_power = np.concatenate(refine_power_all) if refine_power_all else coarse_power[:0]
    refine_used = np.concatenate(refine_used_all) if refine_used_all else coarse_used[:0]

    all_periods = np.concatenate([coarse_periods, refine_periods])
    all_power = np.concatenate([coarse_power, refine_power])
    all_used = np.concatenate([coarse_used, refine_used])
    sort_idx = np.argsort(all_periods)
    all_periods = all_periods[sort_idx]
    all_power = all_power[sort_idx]
    all_used = all_used[sort_idx]

    return _result(
        all_periods, all_power, all_used,
        seed_periods=[p for _, p in coarse_peaks],
    )


def _build_peaks(
    periods: np.ndarray,
    power: np.ndarray,
    sde: np.ndarray,
    time: np.ndarray,
    flux: np.ndarray,
    best_period: float,
    topk: int,
    duration_used: np.ndarray,
    seed_periods: Optional[List[float]] = None,
) -> List[BLSPeak]:
    """Select top-k peaks (with NMS) and recover a depth-maximizing epoch for each."""
    duration_used = np.asarray(duration_used, dtype=np.float64)
    selected = _select_topk_peaks(periods, power, sde, k=topk)
    if seed_periods:
        existing = {round(p, 6) for _, p in selected}
        for P in seed_periods:
            if round(P, 6) not in existing:
                near = int(np.argmin(np.abs(periods - P)))
                selected.append((near, float(periods[near])))
        selected = selected[:topk]

    out: List[BLSPeak] = []
    for idx, P in selected:
        dur = float(duration_used[idx]) if idx < len(duration_used) else 0.05
        epoch = _epoch_for_period(time, flux, float(P), dur)
        out.append(
            BLSPeak(
                period=float(P),
                epoch=float(epoch),
                power=float(max(power[idx], 0.0)),
                sde=float(sde[idx]),
                duration_phase=dur,
                depth=float("nan"),
            )
        )
    if not out:
        dur = float(duration_used[int(np.argmax(power))]) if len(duration_used) else 0.05
        out.append(
            BLSPeak(
                period=float(best_period),
                epoch=_epoch_for_period(time, flux, float(best_period), dur),
                power=float(np.max(power)) if len(power) else 0.0,
                sde=float(np.max(sde)) if len(sde) else 0.0,
                duration_phase=dur,
            )
        )
    return out


def _epoch_for_period(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    duration_phase: float,
    n_phase_steps: int = 64,
) -> float:
    """Mid-transit epoch that maximizes mean in-transit depth.

    Scoring by in-transit point count (``n_in``) is almost flat on a uniform
    cadence, so the epoch was arbitrary. Folds, odd/even, and secondary tests
    all use this epoch.
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if len(time) == 0 or not np.isfinite(period) or period <= 0:
        return 0.0
    if len(flux) != len(time):
        raise ValueError("time and flux must have the same length")
    phase = np.mod(time, period) / period
    n_steps = max(int(n_phase_steps), 1)
    dur = float(duration_phase) if duration_phase and duration_phase > 0 else 0.05
    best_phi = 0.0
    best_score = -np.inf
    for j in range(n_steps):
        phi = j / n_steps
        high = phi + dur
        if high > 1.0:
            in_transit = (phase >= phi) | (phase < (high - 1.0))
        else:
            in_transit = (phase >= phi) & (phase < high)
        n_in = int(np.sum(in_transit))
        if n_in < 2:
            continue
        y_in = float(np.mean(flux[in_transit]))
        y_out = float(np.mean(flux[~in_transit]))
        score = y_out - y_in
        if score > best_score:
            best_score = score
            best_phi = phi
    mid_phase = (best_phi + 0.5 * dur) % 1.0
    t_min = float(np.nanmin(time))
    phase_of_tmin = (t_min % period) / period
    dphi = (mid_phase - phase_of_tmin) % 1.0
    return float(t_min + dphi * period)


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
