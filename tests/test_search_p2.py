"""Phase-2 search upgrades: detrending, period grid, duotransits, batching, planner, sector watch."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline import detrend, period_grid
from pipeline.bls_gpu import _bls_power_numpy, bls_power_batch
from pipeline.duotransit import find_duotransit
from pipeline.target_rank import TargetRecord, expected_yield, rank_targets


def _box(t, period, epoch, dur, depth):
    phase = (t - epoch + 0.5 * period) % period - 0.5 * period
    return np.where(np.abs(phase) < dur / 2, -depth, 0.0)


def test_biweight_keeps_transit_depth_where_savgol_halves_it():
    from scipy.signal import savgol_filter

    rng = np.random.default_rng(0)
    t = np.arange(0, 27, 2 / 1440)
    trend = 1 + 0.005 * np.sin(2 * np.pi * t / 10.0)
    flux = trend * (1 + _box(t, 4.0, 1.0, 0.15, 0.003)) + rng.normal(0, 2e-4, t.size)
    flat, fit = detrend.biweight_detrend(t, flux, window_days=0.6)
    assert np.nanstd(fit / trend - 1) < 1e-3
    in_tr = _box(t, 4.0, 1.0, 0.15, 1.0) < 0
    assert 1 - np.nanmedian(flat[in_tr]) == pytest.approx(0.003, rel=0.1)
    savgol = flux / savgol_filter(flux, int(0.6 / (2 / 1440)) | 1, 2)
    assert 1 - np.nanmedian(savgol[in_tr]) < 0.6 * 0.003


def test_masked_redetrend_recovers_more_depth_than_unmasked():
    t = np.arange(0, 20, 2 / 1440)
    dur = 0.4
    flux = 1 + _box(t, 5.0, 2.0, dur, 0.004)
    flat, trend = detrend.biweight_detrend(t, flux, window_days=0.5)
    redone = detrend.masked_redetrend(t, flat, trend, 5.0, 2.0, dur, window_days=0.5)
    in_tr = _box(t, 5.0, 2.0, dur * 0.5, 1.0) < 0
    assert 1 - np.nanmedian(redone[in_tr]) > 1 - np.nanmedian(flat[in_tr])
    assert 1 - np.nanmedian(redone[in_tr]) == pytest.approx(0.004, rel=0.1)


def test_ofir_grid_is_uniform_in_cube_root_frequency():
    p = period_grid.ofir_periods(0.5, 15.0, 500)
    assert p[0] == pytest.approx(0.5) and p[-1] == pytest.approx(15.0)
    x = (1 / p) ** (1 / 3)
    steps = np.diff(np.sort(x))
    assert np.allclose(steps, steps[0])
    # More trials at short period than a log grid of the same size.
    log = period_grid.log_periods(0.5, 15.0, 500)
    assert np.sum(p < 2.0) > np.sum(log < 2.0)
    assert period_grid.ofir_count(27.0, 0.5, 15.0) > period_grid.ofir_count(10.0, 0.5, 15.0)


def test_duotransit_lists_aliases_and_rejects_covered_ones():
    rng = np.random.default_rng(3)
    t = np.concatenate([np.arange(0, 27, 10 / 1440), np.arange(52, 79, 10 / 1440)])
    dur, depth = 0.25, 0.004
    flux = 1 + rng.normal(0, 3e-4, t.size)
    for centre in (10.0, 60.0):
        flux[np.abs(t - centre) < dur / 2] -= depth
    result = find_duotransit(t, flux, min_alias_period=12.0)
    assert result is not None and result["kind"] == "duotransit"
    assert sorted(round(e) for e in result["epochs"]) == [10, 60]
    periods = [round(a["period"], 1) for a in result["aliases"]]
    assert 50.0 in periods and 25.0 in periods
    assert 16.7 not in periods  # would put a transit at t=26.7, inside data


def test_bls_epoch_is_absolute_time():
    t = np.arange(3000.0, 3027.0, 10 / 1440)
    flux = 1 + _box(t, 3.0, 3001.2, 0.12, 0.01)
    periods = np.linspace(2.9, 3.1, 41)
    _, _, best_p, epoch = _bls_power_numpy(t, flux, periods, duration_phase=0.04, n_phase_steps=75)
    assert best_p == pytest.approx(3.0, abs=0.01)
    phase = (epoch - 3001.2 + 0.5 * best_p) % best_p - 0.5 * best_p
    assert abs(phase) < 0.15


def test_batch_bls_matches_single_star():
    rng = np.random.default_rng(1)
    curves = []
    for period in (2.0, 3.3):
        t = np.arange(1500, 1527, 20 / 1440)
        curves.append((t, 1 + _box(t, period, 1500.5, 0.1, 0.01) + rng.normal(0, 5e-4, t.size)))
    periods = period_grid.log_periods(1.0, 5.0, 300)
    results = bls_power_batch(curves, periods, duration_phase=0.03, n_phase_steps=40)
    assert [round(r["best_period"], 1) for r in results] == [2.0, 3.3]
    assert all(r["sde"] > 7 for r in results)


def test_triage_keeps_transiting_star_and_drops_noise():
    from parallel_hunter import triage_keep

    rng = np.random.default_rng(2)
    t = np.arange(1500, 1527, 10 / 1440)
    planet = {"time": t, "flux": 1 + _box(t, 2.5, 1500.3, 0.1, 0.01) + rng.normal(0, 5e-4, t.size)}
    noise = {"time": t, "flux": 1 + rng.normal(0, 5e-4, t.size)}
    keep = triage_keep([planet, noise], 0.5, 10.0, min_sde=9.0, n_periods=1500)
    assert keep == [True, False]


def test_yield_prefers_unexplored_baseline_and_small_stars():
    fresh = TargetRecord(tic_id="fresh", radius=0.5, n_sectors=4, tmag=10, cdpp_ppm=150)
    exhausted = TargetRecord(tic_id="done", radius=0.5, n_sectors=4, tmag=10, cdpp_ppm=150,
                             extra={"searched_sectors": 4})
    big = TargetRecord(tic_id="big", radius=2.0, n_sectors=4, tmag=10, cdpp_ppm=150)
    assert expected_yield(fresh) > expected_yield(big) > 0
    assert expected_yield(exhausted) == pytest.approx(0.0, abs=1e-12)
    partly = TargetRecord(tic_id="part", radius=0.5, n_sectors=4, tmag=10, cdpp_ppm=150,
                          extra={"searched_sectors": 1})
    assert 0 < expected_yield(partly) < expected_yield(fresh)
    queue = rank_targets([exhausted, big, fresh], limit=3, objective="yield")
    assert [r.tic_id for r in queue.primary][0] == "fresh"


def test_continuous_walk_puts_new_release_first_without_moving_the_walk(tmp_path):
    from tool.desk import Desk

    desk = Desk(tmp_path / "desk", enable_auto_start=False)
    cont = desk.continuous
    latest = {"value": 80}
    cont.sector_fetcher = lambda: latest["value"]
    cont.state.update({"mode": "sectors", "plan": "sectors", "sector": 10, "walk_sector": 10,
                       "status": "running", "config": {}})
    cont.store.enqueue("sectors", [{"kind": "sector", "target": "sector 10", "sector": 10}])

    first = cont._next_item()
    assert first["sector"] == 10
    cont.store.complete(first["id"])

    latest["value"] = 82
    desk.store.set_json("sector_watch", {**desk.store.get_json("sector_watch"), "checked_at": 0})
    nxt = cont._next_item()
    assert nxt["sector"] == 82
    assert cont.state["walk_sector"] == 10


class _Store:
    def __init__(self):
        self.kv, self.rows = {}, []

    def get_json(self, key, default=None):
        return self.kv.get(key, default)

    def set_json(self, key, value):
        self.kv[key] = value

    def items(self, plan=None, limit=0):
        return self.rows

    def enqueue(self, plan, items):
        self.rows.extend(items)


def test_sector_watch_caches_and_queues_only_new_releases():
    from tool import sector_watch

    store = _Store()
    calls = []

    def fetch():
        calls.append(1)
        return 80

    assert sector_watch.latest_sector(store, fetcher=fetch, now=0) == 80
    assert sector_watch.latest_sector(store, fetcher=fetch, now=60) == 80
    assert len(calls) == 1
    store.rows = [{"sector": 10}]
    assert sector_watch.enqueue_new_sectors(store, "sectors", 80) == []  # first call sets frontier
    assert sector_watch.enqueue_new_sectors(store, "sectors", 82) == [81, 82]
    assert all(r["priority"] >= sector_watch.NEW_SECTOR_PRIORITY for r in store.rows[1:])
    assert sector_watch.enqueue_new_sectors(store, "sectors", 82) == []
    assert sector_watch.latest_sector(store, fetcher=lambda: None, now=10**9) == 80
    assert sector_watch.next_walk_sector(10, {10, 11, 81, 82}, 82) == 12
    assert sector_watch.next_walk_sector(83, {81, 82}, 82) is None
