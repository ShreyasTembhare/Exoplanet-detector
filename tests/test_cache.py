"""
Cache round-trip tests.

Catches regressions of the historical bug where ``np.savez`` silently
appended ``.npz`` to a temp filename ending in ``.npz.tmp``, causing every
on-disk cache file to be 0 bytes. We verify that:
  * The on-disk file is non-empty.
  * The same arrays come back out.
  * No orphan ``*.tmp.npz`` files are left behind.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_phase1_round_trip(tmp_cache_dir):
    from pipeline import cache_io

    time = np.linspace(0, 10, 1000, dtype=np.float64)
    flux = np.ones_like(time) + 0.001 * np.sin(time)
    flux_err = np.full_like(flux, 1e-4)
    meta = {"cadence_minutes": 2.0, "tmag": 11.5, "crowdsap": 0.95}

    cache_io.set_phase1("TIC 12345", "15", time, flux, flux_err, meta=meta)

    out_path = tmp_cache_dir / "phase1" / "TIC_12345_15_phase1.npz"
    assert out_path.exists()
    assert out_path.stat().st_size > 0, "phase1 cache file is empty (regression!)"

    cached = cache_io.get_phase1("TIC 12345", "15")
    assert cached is not None
    cached_time, cached_flux, cached_err, cached_meta = cached
    np.testing.assert_allclose(cached_time, time)
    np.testing.assert_allclose(cached_flux, flux)
    np.testing.assert_allclose(cached_err, flux_err)
    assert cached_meta.get("tmag") == pytest.approx(11.5)

    # No leftover .tmp.npz files.
    leftovers = list((tmp_cache_dir / "phase1").glob("*tmp*"))
    assert leftovers == [], f"leftover tempfiles: {leftovers}"


def test_phase2_round_trip(tmp_cache_dir):
    from pipeline import cache_io
    from pipeline.bls_gpu import BLSPeak

    periods = np.linspace(0.5, 20.0, 200)
    power = np.random.RandomState(0).rand(200)
    peaks = [BLSPeak(period=3.5, epoch=1.7, power=0.05, sde=12.0, duration_phase=0.05)]

    cache_io.set_phase2(
        "TIC 99", "all", 0.5, 20.0, 200,
        periods, power, best_period=3.5, epoch=1.7, peaks=peaks,
    )
    cached = cache_io.get_phase2("TIC 99", "all", 0.5, 20.0, 200)
    assert cached is not None
    cp, cw, cb, ce, cpeaks = cached
    np.testing.assert_allclose(cp, periods)
    np.testing.assert_allclose(cw, power)
    assert cb == pytest.approx(3.5)
    assert ce == pytest.approx(1.7)
    assert len(cpeaks) == 1
    assert cpeaks[0].period == pytest.approx(3.5)


def test_phase3_round_trip(tmp_cache_dir):
    from pipeline import cache_io

    g = np.linspace(-0.01, 0.01, 2048).astype(np.float64)
    l = np.linspace(-0.01, 0.01, 256).astype(np.float64)
    extras = {"vetting": {"sde": 12.0, "depth": 0.005, "duration_hours": 2.0}}

    cache_io.set_phase3("TIC 77", "all", 3.5, g, l, 0.42, extras=extras)
    cached = cache_io.get_phase3("TIC 77", "all", 3.5)
    assert cached is not None
    cg, cl, co, cextras = cached
    np.testing.assert_allclose(cg, g)
    np.testing.assert_allclose(cl, l)
    assert co == pytest.approx(0.42)
    assert cextras["vetting"]["sde"] == pytest.approx(12.0)


def test_corrupt_zero_byte_file_returns_none(tmp_cache_dir):
    """Backwards-compat: legacy 0-byte files should miss the cache, not crash."""
    from pipeline import cache_io

    p = tmp_cache_dir / "phase1" / "broken.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    assert cache_io._read_npz(p) is None
