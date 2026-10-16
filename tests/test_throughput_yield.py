"""Throughput and yield: downsample, duration grid, refine-all, cache, MES, mask."""

from __future__ import annotations

import numpy as np
import pytest


def test_downsample_keeps_short_transit_samples():
    from pipeline.bls_gpu import _downsample

    time = np.linspace(0.0, 365.0, 400_000)
    flux = np.ones_like(time)
    short = 5.0 / (24.0 * 60.0)  # 5 minutes
    kept_t, kept_f = _downsample(time, flux, 8_000, min_duration_days=short)
    assert len(kept_t) > 8_000
    assert len(kept_t) == len(kept_f)
    dt = float(np.median(np.diff(kept_t)))
    assert short / dt >= 3.0 - 0.05

    long_t, _ = _downsample(time, flux, 8_000, min_duration_days=2.0)
    assert len(long_t) <= 8_000


def test_adaptive_duration_grid_is_used_per_period(monkeypatch):
    import pipeline.physical_priors as priors
    from pipeline.bls_gpu import compute_bls_periodogram

    calls = {"n": 0}
    real = priors.adaptive_duration_grid

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(priors, "adaptive_duration_grid", wrapped)
    rng = np.random.default_rng(0)
    time = np.linspace(0.0, 20.0, 400)
    flux = 1.0 + rng.normal(0, 1e-4, size=time.shape)
    compute_bls_periodogram(
        time, flux, period_min=1.0, period_max=10.0, nperiods=24,
        two_pass=False, backend="numpy", duration_phase=None,
        rho_star_solar=1.0, n_durations=4, topk=2,
    )
    assert calls["n"] >= 1

    calls["n"] = 0
    compute_bls_periodogram(
        time, flux, period_min=1.0, period_max=10.0, nperiods=24,
        two_pass=False, backend="numpy", duration_phase=0.05, topk=1,
    )
    assert calls["n"] == 0


def test_refine_runs_for_each_kept_peak(monkeypatch):
    import pipeline.bls_gpu as bls

    calls = []
    real = bls._refine_period

    def wrapped(*args, **kwargs):
        calls.append(args[3])
        return real(*args, **kwargs)

    monkeypatch.setattr(bls, "_refine_period", wrapped)
    rng = np.random.default_rng(4)
    time = np.linspace(0.0, 27.0, 1500)
    flux = 1.0 + rng.normal(0, 1e-4, size=time.shape)
    period = 3.2
    phase = (time - 1.1) % period
    flux[(phase < 0.06) | (phase > period - 0.06)] -= 0.01
    res = bls.compute_bls_periodogram(
        time, flux, period_min=1.0, period_max=12.0,
        two_pass=True, coarse_nperiods=40, refine_nperiods=20,
        backend="numpy", duration_phase=0.04, topk=3,
    )
    assert len(calls) >= 2
    # The injected period or its 2:1 alias should be among the refined peaks.
    periods = [pk.period for pk in res.peaks] + [res.best_period]
    assert any(min(abs(p / period - 1.0), abs(p / period - 2.0), abs(p / (2 * period) - 1.0)) < 0.08 for p in periods)


def test_phase2_cache_skips_second_compute(tmp_cache_dir, monkeypatch):
    import pipeline.planet_search as search

    calls = {"n": 0}
    real = search.compute_bls_periodogram

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(search, "compute_bls_periodogram", wrapped)
    time = np.linspace(0.0, 10.0, 300)
    flux = np.ones_like(time)
    kwargs = dict(
        tic_id="TIC 15", sector="15",
        period_min=1.0, period_max=5.0, nperiods=20,
        backend="numpy", rho_star_solar=1.0,
        adaptive_duration=False, use_cache=True,
        two_pass=False,
    )
    first = search.cached_bls(time, flux, **kwargs)
    second = search.cached_bls(time, flux, **kwargs)
    assert calls["n"] == 1
    assert second.best_period == pytest.approx(first.best_period)


def test_default_backend_is_fastest_installed():
    from pipeline.bls_gpu import JAX_AVAILABLE, _fastest_backend, _select_backend

    chosen = _select_backend(1000, None)
    assert chosen == _fastest_backend()
    if JAX_AVAILABLE:
        assert chosen == "jax"
    else:
        assert chosen in {"astropy", "numpy"}
    # Astropy remains selectable as the reference when it imports.
    try:
        import astropy.timeseries  # noqa: F401
    except Exception:
        return
    assert _select_backend(1000, "astropy") == "astropy"


def test_single_event_recovers_one_dip():
    from pipeline.single_event import single_event_statistic

    time = np.linspace(0.0, 27.0, 4000)
    flux = np.ones_like(time)
    flux[(time > 10.0) & (time < 10.15)] -= 0.01
    event = single_event_statistic(time, flux, [0.15])
    assert event["mes"] > 7.0
    assert abs(event["epoch"] - 10.075) <= 0.5 * 0.15 + 0.02

    quiet = single_event_statistic(time, np.ones_like(time), [0.15])
    assert quiet["mes"] < 1.0


def test_mask_removes_injected_transit():
    from pipeline.bls_gpu import compute_bls_periodogram
    from pipeline.planet_search import mask_transits

    time = np.linspace(0.0, 27.0, 2000)
    flux = np.ones_like(time)
    period, epoch, duration = 4.0, 1.2, 0.12
    phase = (time - epoch) % period
    flux[(phase < duration / 2) | (phase > period - duration / 2)] -= 0.02
    before = compute_bls_periodogram(
        time, flux, period_min=2.0, period_max=8.0, nperiods=40,
        two_pass=False, backend="numpy", duration_phase=0.04, topk=1,
    )
    masked = mask_transits(time, flux, period, epoch, duration)
    after = compute_bls_periodogram(
        time, masked, period_min=2.0, period_max=8.0, nperiods=40,
        two_pass=False, backend="numpy", duration_phase=0.04, topk=1,
    )
    assert before.sde > after.sde


def test_multiplanet_mask_and_research():
    from pipeline.planet_search import search_with_masking

    rng = np.random.default_rng(9)
    time = np.linspace(0.0, 30.0, 2500)
    flux = 1.0 + rng.normal(0, 1e-4, size=time.shape)
    planets = [(3.1, 0.8, 0.10, 0.015), (8.4, 2.2, 0.16, 0.012)]
    for period, epoch, duration, depth in planets:
        phase = (time - epoch) % period
        flux[(phase < duration / 2) | (phase > period - duration / 2)] -= depth

    found = search_with_masking(
        time, flux, tic_id="TIC 99", sector="1",
        period_min=2.0, period_max=12.0, nperiods=60,
        backend="numpy", adaptive_duration=False,
        sde_threshold=5.0, use_cache=False, use_tls=False,
        max_planets=2, run_triceratops=False, run_monotransit=False,
        two_pass=False,
    )
    recovered = [sig["period"] for sig in found["signals"]]
    assert len(recovered) >= 2
    for period, _, _, _ in planets:
        assert any(abs(p - period) / period < 0.1 for p in recovered)


def test_tls_only_on_sde_pass_peaks(monkeypatch):
    import pipeline.planet_search as search

    calls = []

    def fake_tls(time, flux, peaks, sde_threshold, max_peaks=3):
        calls.append([p.sde for p in peaks])
        return []

    monkeypatch.setattr(search, "tls_on_sde_peaks", fake_tls)
    time = np.linspace(0.0, 12.0, 400)
    flux = np.ones_like(time)
    search.search_with_masking(
        time, flux, tic_id="TIC 1", sector="1",
        period_min=1.0, period_max=6.0, nperiods=20,
        backend="numpy", adaptive_duration=False,
        sde_threshold=50.0, use_cache=False, use_tls=True,
        run_triceratops=False, run_monotransit=False, two_pass=False,
    )
    assert calls == []

    phase = (time - 0.4) % 2.5
    flux[(phase < 0.05) | (phase > 2.45)] -= 0.03
    search.search_with_masking(
        time, flux, tic_id="TIC 1", sector="1b",
        period_min=1.0, period_max=6.0, nperiods=30,
        backend="numpy", adaptive_duration=False,
        sde_threshold=4.0, use_cache=False, use_tls=True,
        run_triceratops=False, run_monotransit=False, two_pass=False,
        max_planets=1,
    )
    assert calls, "TLS should run once a peak clears the SDE gate"
    assert all(sde >= 4.0 for batch in calls for sde in batch)
