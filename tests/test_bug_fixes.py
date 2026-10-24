"""Regression tests for pipeline correctness fixes."""

from __future__ import annotations

import numpy as np
import pytest


def test_refine_peaks_is_index_aligned():
    from pipeline.tls_refine import refine_peaks

    t = np.linspace(0, 10, 500)
    f = np.ones_like(t)
    peaks = [{"period": None, "epoch": 1.0}, {"period": 2.0, "epoch": 0.5}]
    out = refine_peaks(t, f, peaks, max_peaks=5)
    assert len(out) == 2
    assert not out[0].available
    assert out[1].period == pytest.approx(2.0) or out[1].available


def test_count_transits_ignores_gaps():
    from pipeline.vetting import count_transits

    t = np.concatenate([np.arange(0, 5, 0.01), np.arange(20, 27, 0.01)])
    # P=2 over a 27 d span would be ~14 by span/period; only ~6 are observed.
    n = count_transits(t, period=2.0, epoch=0.5, duration_days=0.1)
    assert 5 <= n <= 7


def test_phase2_cache_key_depends_on_variant(tmp_cache_dir):
    from pipeline import cache_io

    periods = np.linspace(1, 2, 10)
    cache_io.set_phase2("TIC 1", "5", 0.5, 15, 100, periods, periods, 1.5, 0.1, variant="backend=jax")
    assert cache_io.get_phase2("TIC 1", "5", 0.5, 15, 100, variant="backend=jax") is not None
    assert cache_io.get_phase2("TIC 1", "5", 0.5, 15, 100, variant="backend=numpy") is None


def test_astropy_backend_scales_duration_with_period(synthetic_transit):
    from pipeline.bls_gpu import _bls_power_astropy

    t, f, period, *_ = synthetic_transit
    periods = np.exp(np.linspace(np.log(0.5), np.log(12.0), 400))
    _, power, best, _ = _bls_power_astropy(t, f, periods, duration_phase=0.035)
    assert len(power) == len(periods)
    ratio = best / period
    assert min(abs(ratio - h) for h in (0.5, 1.0, 2.0)) < 0.05


def test_errors_are_retried_then_parked():
    from hunter import MAX_ERROR_ATTEMPTS, _should_skip_stage

    progress = {"1": {"last_stage": "DONE", "status": "ERROR: timeout", "error_attempts": 1}}
    assert not _should_skip_stage(progress, "1", "predict")
    progress["1"]["error_attempts"] = MAX_ERROR_ATTEMPTS
    assert _should_skip_stage(progress, "1", "predict")
    progress["2"] = {"last_stage": "DONE", "status": "NO_SIGNAL"}
    assert _should_skip_stage(progress, "2", "predict")


def test_load_progress_counts_error_attempts(tmp_path):
    from hunter import _append_log, _ensure_setup, _load_progress

    log = str(tmp_path / "log.txt")
    _ensure_setup(log, str(tmp_path / "c"))
    _append_log(log, "7", "DONE", "ERROR: a")
    _append_log(log, "7", "DONE", "ERROR: b")
    assert _load_progress(log)["7"]["error_attempts"] == 2


def test_sector_outcome():
    from autopilot import sector_outcome

    assert sector_outcome({"completed": 0, "candidates": [], "error": "boom"}) == "retry"
    assert sector_outcome({"completed": 0, "candidates": []}) == "retry"
    assert sector_outcome({"completed": 0, "skipped": 10}) == "done"
    assert sector_outcome({"completed": 0, "no_targets": True}) == "done"
    assert sector_outcome({"completed": 5}) == "done"


def test_known_planet_orbit_matching():
    from pipeline.known_planets import build_known_table, classify_candidate

    table = build_known_table([
        {"tid": 100, "toi": "123.01", "tfopwg_disp": "CP", "pl_orbper": 3.0, "pl_tranmid": 2459000.5},
    ])
    assert classify_candidate("TIC 100", 3.001, 2000.5, table)[0] == "KNOWN"
    assert classify_candidate("TIC 100", 6.0, None, table)[0] == "KNOWN"  # harmonic
    assert classify_candidate("TIC 100", 7.3, None, table)[0] == "NEW_ON_KNOWN_HOST"
    assert classify_candidate("TIC 999", 3.0, None, table)[0] == "NEW_CANDIDATE"


def test_candidate_score_without_classifier_uses_sde():
    from strategy_profiles import compute_candidate_score

    weak = compute_candidate_score(None, 0.0, float("nan"), sde=7.5)
    strong = compute_candidate_score(None, 0.0, float("nan"), sde=25.0)
    assert 0.0 <= weak < strong <= 1.0


def test_monotransit_views_have_a_dip():
    from pipeline.planet_search import monotransit_views

    t = np.linspace(0, 27, 5000)
    f = np.ones_like(t)
    f[np.abs(t - 13.0) < 0.1] -= 0.01
    gv, lv = monotransit_views(t, f, 13.0, 0.2)
    assert gv.shape == (2048,) and lv.shape == (256,)
    assert lv.min() < 0.995


def test_sky_coords_prefers_catalog_record():
    from hunter import sky_coords
    from pipeline.target_rank import TargetRecord

    assert sky_coords(TargetRecord(tic_id="1", ra=10.0, dec=-5.0), {}) == (10.0, -5.0)
    assert sky_coords(TargetRecord(tic_id="1"), {"ra": 1.0, "dec": 2.0}) == (1.0, 2.0)
