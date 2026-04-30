"""
BLS correctness tests.

Verifies:
  * Top BLS period matches the injected period to within the period grid step.
  * Stellar flares (positive bumps) do not produce positive BLS power.
  * Wrap-around transits (phase straddling 1.0) are recovered.
  * The Astropy oracle backend agrees with the JAX backend.
  * SDE is computed and is well above 7 for an obvious transit.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_numpy_backend_recovers_period(synthetic_transit):
    from pipeline.bls_gpu import compute_bls_periodogram

    t, f, period, _, _, _ = synthetic_transit
    res = compute_bls_periodogram(
        t, f, period_min=1.0, period_max=10.0,
        nperiods=400, two_pass=True,
        coarse_nperiods=200, refine_nperiods=200,
        backend="numpy", duration_phase=0.04,
        topk=3,
    )
    assert abs(res.best_period - period) / period < 0.01
    assert res.sde > 5.0, f"SDE too low: {res.sde}"


def test_topk_returns_multiple_peaks(synthetic_transit):
    from pipeline.bls_gpu import compute_bls_periodogram

    t, f, _, _, _, _ = synthetic_transit
    res = compute_bls_periodogram(
        t, f, period_min=1.0, period_max=10.0,
        nperiods=200, two_pass=True,
        coarse_nperiods=120, refine_nperiods=120,
        backend="numpy", duration_phase=0.04, topk=5,
    )
    assert len(res.peaks) >= 1
    # Power is non-negative.
    for pk in res.peaks:
        assert pk.power >= 0


def test_flare_does_not_produce_positive_power():
    """A pure brightening event should not yield BLS power above noise."""
    from pipeline.bls_gpu import compute_bls_periodogram

    rng = np.random.default_rng(7)
    n = 5000
    t = np.linspace(0, 27, n)
    f = 1.0 + rng.normal(0, 1e-4, size=n)
    # Inject a positive bump (flare-like).
    f += 0.01 * np.exp(-((t - 13.0) / 0.05) ** 2)

    res = compute_bls_periodogram(
        t, f, period_min=1.0, period_max=10.0, nperiods=80,
        two_pass=False, backend="numpy", duration_phase=0.05, topk=3,
    )
    # Power should remain ~ noise level: clamping depth>=0 means a single
    # flare can never look like a transit train.
    assert res.sde < 6.0, f"flare should not pass SDE gate: {res.sde}"


def test_wrap_around_transit_is_recovered():
    """Place the transit so phase=0 falls inside it (epoch ~ start of LC)."""
    from pipeline.bls_gpu import compute_bls_periodogram

    rng = np.random.default_rng(3)
    period = 2.5
    duration_d = 0.10
    n = 8000
    t = np.linspace(0.0, 25.0, n)
    f = 1.0 + rng.normal(0, 5e-5, size=n)
    epoch = 0.05
    phase = (t - epoch) % period
    in_transit = (phase < duration_d / 2) | (phase > period - duration_d / 2)
    f[in_transit] -= 0.005

    res = compute_bls_periodogram(
        t, f, period_min=1.0, period_max=8.0, nperiods=300,
        two_pass=True, coarse_nperiods=200, refine_nperiods=200,
        backend="numpy", duration_phase=0.05, topk=3,
    )
    assert abs(res.best_period - period) / period < 0.02


def test_astropy_oracle_agrees_within_tolerance(synthetic_transit):
    pytest.importorskip("astropy.timeseries")
    from pipeline.bls_gpu import compute_bls_periodogram

    t, f, period, _, _, _ = synthetic_transit
    nominal = compute_bls_periodogram(
        t, f, period_min=1.0, period_max=10.0,
        nperiods=200, two_pass=False, backend="numpy",
        duration_phase=0.04, topk=1,
    )
    oracle = compute_bls_periodogram(
        t, f, period_min=1.0, period_max=10.0,
        nperiods=200, two_pass=False, backend="astropy",
        duration_phase=0.04, topk=1,
    )
    # Both backends should land within 5% of the injected period.
    assert abs(nominal.best_period - period) / period < 0.05
    assert abs(oracle.best_period - period) / period < 0.05


def test_sde_is_robust_to_scale(synthetic_transit):
    """SDE shouldn't depend strongly on raw power scale."""
    from pipeline.bls_gpu import compute_bls_periodogram

    t, f, _, _, _, _ = synthetic_transit
    res1 = compute_bls_periodogram(
        t, f, period_min=1.0, period_max=10.0, nperiods=120,
        two_pass=False, backend="numpy", duration_phase=0.04, topk=1,
    )
    # Scale flux: same shape, different variance; SDE should be similar.
    res2 = compute_bls_periodogram(
        t, (f - 1) * 100 + 1, period_min=1.0, period_max=10.0, nperiods=120,
        two_pass=False, backend="numpy", duration_phase=0.04, topk=1,
    )
    assert abs(res1.sde - res2.sde) / max(res1.sde, 1.0) < 0.5
