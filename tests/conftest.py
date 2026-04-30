"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make repo root importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def synthetic_transit():
    """A clean synthetic light curve with a known box-transit injected.

    Returns ``(time, flux, period, epoch, depth, duration)``.
    """
    rng = np.random.default_rng(42)
    period = 3.5
    epoch = 1.7
    duration_d = 0.12
    depth = 0.005

    cadence_min = 2.0  # 2-minute TESS cadence
    n = int(27 * 24 * 60 / cadence_min)  # one sector
    t = np.linspace(1700.0, 1700.0 + 27.0, n)

    flux = 1.0 + rng.normal(0.0, 1e-4, size=n)

    # Inject box transits.
    phase = (t - epoch) % period
    in_transit = (phase < duration_d / 2) | (phase > period - duration_d / 2)
    flux[in_transit] -= depth
    return t, flux, period, epoch, depth, duration_d


@pytest.fixture
def tmp_cache_dir(tmp_path, monkeypatch):
    """Run with a private cache dir so tests don't leak."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    from pipeline import cache_io

    monkeypatch.setattr(cache_io, "CACHE_ROOT", cache_dir)
    return cache_dir
