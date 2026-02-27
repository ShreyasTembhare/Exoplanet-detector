"""
Disk cache for pipeline phases 1, 2, 3. Enables resuming from last stage on rerun.
Keys: TIC ID, sector, and phase-specific params. Storage: cache/phase1/, phase2/, phase3/ (.npz).
Uses atomic writes (temp file + rename) to prevent corruption on crash.
"""

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np

CACHE_VERSION = 2
CACHE_ROOT = Path("cache")

logger = logging.getLogger(__name__)


def _safe_key(s: str) -> str:
    """Sanitize string for use in filenames."""
    return s.replace("/", "_").replace(" ", "_")


def _phase1_key(tic_id: str, sector: str) -> str:
    return _safe_key(f"{tic_id}_{sector}_phase1")


def _phase2_key(tic_id: str, sector: str, period_min: float, period_max: float, nperiods: int) -> str:
    return _safe_key(f"{tic_id}_{sector}_{period_min}_{period_max}_{nperiods}_phase2")


def _phase3_key(tic_id: str, sector: str, best_period: float) -> str:
    raw = f"{tic_id}_{sector}_{best_period:.6f}_phase3"
    return hashlib.md5(raw.encode()).hexdigest()


def _read_npz(path: Path, required_version: int = CACHE_VERSION) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        data = dict(np.load(path, allow_pickle=True))
        if "cache_version" in data:
            v = int(data["cache_version"])
            if v < required_version:
                logger.info(f"Cache version {v} < {required_version}, ignoring {path}")
                return None
        return data
    except Exception as e:
        logger.warning(f"Failed to read cache {path}: {e}")
        return None


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    """Write .npz atomically: write to temp file in same dir, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npz.tmp", dir=str(path.parent))
    try:
        os.close(fd)
        np.savez(tmp, cache_version=CACHE_VERSION, **arrays)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --- Phase 1: cleaned light curve (time, flux, flux_err) ---


def get_phase1(tic_id: str, sector: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    key = _phase1_key(tic_id, sector)
    path = CACHE_ROOT / "phase1" / f"{key}.npz"
    data = _read_npz(path)
    if data is None:
        return None
    if "time" not in data or "flux" not in data:
        return None
    time = data["time"]
    flux = data["flux"]
    flux_err = data.get("flux_err")
    if flux_err is None:
        flux_err = np.full_like(flux, np.nan)
    logger.info(f"Cache hit: phase1 {key}")
    return time, flux, flux_err


def set_phase1(tic_id: str, sector: str, time: np.ndarray, flux: np.ndarray, flux_err: Optional[np.ndarray] = None) -> None:
    key = _phase1_key(tic_id, sector)
    path = CACHE_ROOT / "phase1" / f"{key}.npz"
    if flux_err is None:
        flux_err = np.full_like(flux, np.nan)
    _write_npz_atomic(path, time=time, flux=flux, flux_err=flux_err)
    logger.info(f"Cached phase1: {key}")


def has_phase1(tic_id: str, sector: str) -> bool:
    key = _phase1_key(tic_id, sector)
    return (CACHE_ROOT / "phase1" / f"{key}.npz").exists()


# --- Phase 2: periodogram (periods, power, best_period, epoch) ---


def get_phase2(
    tic_id: str, sector: str, period_min: float, period_max: float, nperiods: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, float, float]]:
    key = _phase2_key(tic_id, sector, period_min, period_max, nperiods)
    path = CACHE_ROOT / "phase2" / f"{key}.npz"
    data = _read_npz(path)
    if data is None:
        return None
    if "periods" not in data or "power" not in data:
        return None
    periods = data["periods"]
    power = data["power"]
    best_period = float(data.get("best_period", np.nan))
    epoch = float(data.get("epoch", np.nan))
    logger.info(f"Cache hit: phase2 {key}")
    return periods, power, best_period, epoch


def set_phase2(
    tic_id: str, sector: str, period_min: float, period_max: float, nperiods: int,
    periods: np.ndarray, power: np.ndarray, best_period: float, epoch: float,
) -> None:
    key = _phase2_key(tic_id, sector, period_min, period_max, nperiods)
    path = CACHE_ROOT / "phase2" / f"{key}.npz"
    _write_npz_atomic(
        path, periods=periods, power=power,
        best_period=np.array(best_period), epoch=np.array(epoch),
    )
    logger.info(f"Cached phase2: {key}")


def has_phase2(tic_id: str, sector: str, period_min: float = 0.5, period_max: float = 20.0, nperiods: int = 5000) -> bool:
    key = _phase2_key(tic_id, sector, period_min, period_max, nperiods)
    return (CACHE_ROOT / "phase2" / f"{key}.npz").exists()


# --- Phase 3: global_view (2048), local_view (256), centroid_offset ---


def get_phase3(tic_id: str, sector: str, best_period: float) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    key = _phase3_key(tic_id, sector, best_period)
    path = CACHE_ROOT / "phase3" / f"{key}.npz"
    data = _read_npz(path)
    if data is None:
        return None
    if "global_view" not in data or "local_view" not in data:
        return None
    global_view = data["global_view"]
    local_view = data["local_view"]
    centroid_offset = float(data.get("centroid_offset", np.nan))
    logger.info(f"Cache hit: phase3 {key}")
    return global_view, local_view, centroid_offset


def set_phase3(
    tic_id: str, sector: str, best_period: float,
    global_view: np.ndarray, local_view: np.ndarray, centroid_offset: float,
) -> None:
    key = _phase3_key(tic_id, sector, best_period)
    path = CACHE_ROOT / "phase3" / f"{key}.npz"
    _write_npz_atomic(
        path, global_view=global_view, local_view=local_view,
        centroid_offset=np.array(centroid_offset),
    )
    logger.info(f"Cached phase3: {key}")
