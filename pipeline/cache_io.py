"""
Disk cache for pipeline phases 1, 2, 3. Enables resuming from last stage on rerun.
Keys: TIC ID, sector, and phase-specific params. Storage: cache/phase1/, phase2/, phase3/ (.npz).
Uses atomic writes (temp file + rename) to prevent corruption on crash.
"""

import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

CACHE_VERSION = 3
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


def _tpf_key(tic_id: str, sector: str) -> str:
    return _safe_key(f"{tic_id}_{sector}_tpf")


def _read_npz(path: Path, required_version: int = CACHE_VERSION) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        if path.stat().st_size == 0:
            return None
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
    """
    Write .npz atomically: write to a temp file with explicit .npz suffix, then rename.

    Subtle bug being avoided: ``np.savez`` silently appends ``.npz`` to filenames
    that don't end in ``.npz``. Using ``tempfile.mkstemp(suffix=".npz.tmp")`` would
    therefore write to ``<tmp>.npz.tmp.npz`` while the empty ``<tmp>.npz.tmp`` got
    renamed onto the destination — leaving every cache file zero bytes on disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez(str(tmp), cache_version=CACHE_VERSION, **arrays)
        os.replace(str(tmp), str(path))
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


# --- Phase 1: cleaned light curve (time, flux, flux_err, meta) ---


def get_phase1(
    tic_id: str, sector: str
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, dict]]:
    """
    Returns (time, flux, flux_err, meta) where meta is a dict of cadence_minutes,
    crowdsap, tmag, author, and other per-LC metadata. Older cache entries without
    meta return an empty dict.
    """
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
    meta_arr = data.get("meta")
    if meta_arr is not None:
        try:
            meta = meta_arr.item() if hasattr(meta_arr, "item") else dict(meta_arr)
            if not isinstance(meta, dict):
                meta = {}
        except Exception:
            meta = {}
    else:
        meta = {}
    logger.info(f"Cache hit: phase1 {key}")
    return time, flux, flux_err, meta


def set_phase1(
    tic_id: str,
    sector: str,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: Optional[np.ndarray] = None,
    meta: Optional[dict] = None,
) -> None:
    key = _phase1_key(tic_id, sector)
    path = CACHE_ROOT / "phase1" / f"{key}.npz"
    if flux_err is None:
        flux_err = np.full_like(flux, np.nan)
    _write_npz_atomic(
        path,
        time=time,
        flux=flux,
        flux_err=flux_err,
        meta=np.array(meta or {}, dtype=object),
    )
    logger.info(f"Cached phase1: {key}")


def has_phase1(tic_id: str, sector: str) -> bool:
    key = _phase1_key(tic_id, sector)
    return (CACHE_ROOT / "phase1" / f"{key}.npz").exists()


# --- Phase 2: periodogram (periods, power, best_period, epoch) ---


def get_phase2(
    tic_id: str, sector: str, period_min: float, period_max: float, nperiods: int,
):
    """
    Returns ``(periods, power, best_period, epoch, peaks)`` where peaks is a
    list of :class:`pipeline.bls_gpu.BLSPeak`. Older cache entries without
    peaks return an empty list.
    """
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
    peaks = []
    peaks_arr = data.get("peaks")
    if peaks_arr is not None:
        try:
            from .bls_gpu import BLSPeak
            if isinstance(peaks_arr, np.ndarray):
                raw = peaks_arr.tolist() if peaks_arr.ndim > 0 else [peaks_arr.item()]
            else:
                raw = list(peaks_arr)
            for d in raw:
                if isinstance(d, dict):
                    peaks.append(BLSPeak(**d))
        except Exception:
            peaks = []
    logger.info(f"Cache hit: phase2 {key}")
    return periods, power, best_period, epoch, peaks


def set_phase2(
    tic_id: str, sector: str, period_min: float, period_max: float, nperiods: int,
    periods: np.ndarray, power: np.ndarray, best_period: float, epoch: float,
    peaks: Optional[list] = None,
) -> None:
    key = _phase2_key(tic_id, sector, period_min, period_max, nperiods)
    path = CACHE_ROOT / "phase2" / f"{key}.npz"
    peaks_serial = []
    if peaks:
        for pk in peaks:
            try:
                if hasattr(pk, "__dataclass_fields__"):
                    peaks_serial.append({k: getattr(pk, k) for k in pk.__dataclass_fields__})
                elif isinstance(pk, dict):
                    peaks_serial.append(pk)
            except Exception:
                continue
    _write_npz_atomic(
        path, periods=periods, power=power,
        best_period=np.array(best_period), epoch=np.array(epoch),
        peaks=np.array(peaks_serial, dtype=object),
    )
    logger.info(f"Cached phase2: {key}")


def has_phase2(tic_id: str, sector: str, period_min: float = 0.5, period_max: float = 20.0, nperiods: int = 5000) -> bool:
    key = _phase2_key(tic_id, sector, period_min, period_max, nperiods)
    return (CACHE_ROOT / "phase2" / f"{key}.npz").exists()


# --- Phase 3: global_view (2048), local_view (256), centroid_offset ---


def get_phase3(tic_id: str, sector: str, best_period: float):
    """
    Returns ``(global_view, local_view, centroid_offset, extras)``.
    Older cache entries without ``extras`` get an empty dict.
    """
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
    extras_arr = data.get("extras")
    extras: dict = {}
    if extras_arr is not None:
        try:
            payload = extras_arr.item() if hasattr(extras_arr, "item") else dict(extras_arr)
            if isinstance(payload, dict):
                extras = payload
        except Exception:
            extras = {}
    logger.info(f"Cache hit: phase3 {key}")
    return global_view, local_view, centroid_offset, extras


def set_phase3(
    tic_id: str, sector: str, best_period: float,
    global_view: np.ndarray, local_view: np.ndarray, centroid_offset: float,
    extras: Optional[dict] = None,
) -> None:
    """
    extras: optional vetting metric dict (depth, duration, n_transits, sde,
    odd_even_ratio, secondary_depth, v_shape, gaia_neighbors, ...).
    """
    key = _phase3_key(tic_id, sector, best_period)
    path = CACHE_ROOT / "phase3" / f"{key}.npz"
    _write_npz_atomic(
        path, global_view=global_view, local_view=local_view,
        centroid_offset=np.array(centroid_offset),
        extras=np.array(extras or {}, dtype=object),
    )
    logger.info(f"Cached phase3: {key}")


# --- TPF cache: store a small tarball-like blob via numpy savez ---


def has_tpf(tic_id: str, sector: str) -> bool:
    key = _tpf_key(tic_id, sector)
    return (CACHE_ROOT / "tpf" / f"{key}.npz").exists()


def get_tpf_path(tic_id: str, sector: str) -> Path:
    key = _tpf_key(tic_id, sector)
    return CACHE_ROOT / "tpf" / f"{key}.npz"


def set_tpf_arrays(
    tic_id: str,
    sector: str,
    time: np.ndarray,
    flux_cube: np.ndarray,
    aperture: Optional[np.ndarray] = None,
) -> None:
    """
    Persist a TPF as raw arrays so we don't need to re-download for centroid /
    difference-image vetting.
    """
    key = _tpf_key(tic_id, sector)
    path = CACHE_ROOT / "tpf" / f"{key}.npz"
    _write_npz_atomic(
        path,
        time=time,
        flux_cube=flux_cube,
        aperture=np.array([] if aperture is None else aperture),
    )
    logger.info(f"Cached tpf: {key}")


def get_tpf_arrays(
    tic_id: str, sector: str
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    key = _tpf_key(tic_id, sector)
    path = CACHE_ROOT / "tpf" / f"{key}.npz"
    data = _read_npz(path)
    if data is None:
        return None
    if "time" not in data or "flux_cube" not in data:
        return None
    return data["time"], data["flux_cube"], data.get("aperture")
