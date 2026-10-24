"""
Training set that matches what the hunter sees at inference.

* **Triage-v2 labels** (Tey et al. 2023, Zenodo 7411579): the 3 MB
  ``tces_with_labels_v3.csv`` holds ~25k TESS TCEs with BLS ephemerides and
  E/J/B/S/N votes. The 6.5 GB light-curve tarball is not needed: light
  curves are fetched per TIC through our own Phase 1, so detrending and
  cadence match the live pipeline.
* **BLS-recovered folds**: each labelled signal is re-found with a narrow
  BLS on our light curve and folded at *that* ephemeris, exactly like a
  hunter candidate, instead of at the catalog value. Signals our search
  cannot recover are dropped for positives (the model would never see them)
  and kept for negatives.
* **Injections** onto quiet light curves (``J`` labels, no detected signal)
  add physically shaped positives across depth and period, and run through
  the same BLS recovery. Injections never go onto EB/FP stars.
* ``sector`` (from the epoch) and the published ``split`` are written to the
  manifest so validation holds out whole sectors.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from pipeline.fold_features import (
    fold_phase,
    global_view,
    local_centered_view,
    local_view,
    periodogram_view,
    transit_phase_window,
)

logger = logging.getLogger(__name__)

TRIAGE_V2_RECORD = "7411579"
TRIAGE_TABLE = "tces_with_labels_v3.csv"
TRIAGE_URL = f"https://zenodo.org/api/records/{TRIAGE_V2_RECORD}/files/{TRIAGE_TABLE}/content"
TESS_S1_START_BTJD = 1325.3
SECTOR_DAYS = 27.4
POSITIVE_VOTES = {"E"}
NEGATIVE_VOTES = {"J", "B", "S"}


def sector_from_btjd(btjd: float) -> int:
    if btjd is None or not np.isfinite(btjd):
        return -1
    return max(int((float(btjd) - TESS_S1_START_BTJD) // SECTOR_DAYS) + 1, 1)


def download_triage_table(out_dir: str | Path, opener: Optional[Callable] = None,
                          timeout: float = 60.0) -> Optional[Path]:
    """Fetch the labelled TCE table (cached). None when Zenodo is unreachable."""
    out = Path(out_dir) / TRIAGE_TABLE
    if out.is_file() and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    opener = opener or (lambda url: urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "tess-exoplanet-detector"}), timeout=timeout,
    ))
    try:
        with opener(TRIAGE_URL) as resp:
            data = resp.read()
    except Exception as exc:
        logger.info("Triage-v2 table not reachable (%s).", exc)
        return None
    tmp = out.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(out)
    return out


def vote_label(row: dict) -> Optional[int]:
    """Consensus E -> 1, J/B/S -> 0, N -> None; else majority of labeller votes."""
    consensus = str(row.get("Consensus Label") or "").strip().upper()
    if consensus in POSITIVE_VOTES:
        return 1
    if consensus in NEGATIVE_VOTES:
        return 0
    if consensus == "N":
        return None
    votes = [str(row.get(f"L{i}") or "").strip().upper() for i in range(1, 9)]
    votes = [v for v in votes if v in POSITIVE_VOTES | NEGATIVE_VOTES]
    if not votes:
        return None
    frac = sum(v in POSITIVE_VOTES for v in votes) / len(votes)
    if frac >= 0.75:
        return 1
    if frac <= 0.25:
        return 0
    return None


def load_triage_table(path: str | Path):
    """Normalised DataFrame: tic, period, epoch, duration_days, depth_ppm, tmag,
    radius, mass, split, sector, label, quiet (J vote)."""
    import pandas as pd

    df = pd.read_csv(path, comment="#")
    df = df[pd.to_numeric(df["TIC ID"], errors="coerce").notna()].copy()
    rows = []
    for row in df.to_dict("records"):
        label = vote_label(row)
        if label is None:
            continue
        epoch = float(row["Epoch"])
        rows.append({
            "tic": str(int(float(row["TIC ID"]))),
            "period": float(row["Period"]),
            "epoch": epoch,
            "duration_days": float(row["Duration"]),
            "depth_ppm": float(row.get("Depth") or np.nan),
            "tmag": float(row.get("Tmag") or np.nan),
            "radius": float(row.get("SRad") or row.get("SRadEst") or np.nan),
            "mass": float(row.get("SMass") or np.nan),
            "split": str(row.get("Split") or ""),
            "sector": sector_from_btjd(epoch),
            "label": int(label),
            "quiet": str(row.get("Consensus Label") or "").strip().upper() == "J",
        })
    return pd.DataFrame(rows)


def recover_ephemeris(time, flux, period: float, epoch: float, duration_days: float,
                      tol: float = 0.01, n_periods: int = 400) -> Tuple[float, float, float, bool]:
    """Narrow BLS around ``period``. Returns (period, epoch, depth_snr, recovered).

    SDE is meaningless inside a +-1% window (every trial sits on the peak),
    so the strength of the recovered dip is its BLS depth SNR.
    """
    from astropy.timeseries import BoxLeastSquares

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ok = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[ok], flux[ok]
    if len(time) < 50 or not (period > 0):
        return period, epoch, float("nan"), False
    periods = np.linspace(period * (1 - tol), period * (1 + tol), n_periods)
    durations = np.clip(np.array([0.5, 1.0, 1.5]) * duration_days, 0.01, 0.45 * period)
    sigma = 1.4826 * float(np.median(np.abs(flux - np.median(flux)))) or float(np.std(flux)) or 1e-3
    res = BoxLeastSquares(time, flux, dy=np.full_like(flux, sigma)).power(periods, durations)
    i = int(np.argmax(np.asarray(res.power, dtype=float)))
    snr = float(res.depth_snr[i])
    best_p = float(res.period[i])
    best_t0 = float(res.transit_time[i])
    # Recovered when the dip is where the catalog says (within a duration).
    off = abs(((best_t0 - epoch) + 0.5 * best_p) % best_p - 0.5 * best_p)
    recovered = bool(off < max(duration_days, 0.05) and float(res.depth[i]) > 0)
    return best_p, best_t0, snr, recovered


def fold_views(time, flux, period: float, epoch: float, periods=None, power=None) -> dict:
    """The same views ``run_phase3`` builds for a live candidate."""
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    phase = fold_phase(time, period, epoch)
    centre, half = transit_phase_window(phase, flux)
    return {
        "global_view": global_view(time, flux, period, epoch).astype(np.float32),
        "local_view": local_view(time, flux, period, epoch, transit_center=centre,
                                 transit_half_width=half).astype(np.float32),
        "local_centered": local_centered_view(time, flux, period, epoch, half_width=half).astype(np.float32),
        "periodogram": periodogram_view(
            periods if periods is not None else np.array([]),
            power if power is not None else np.array([]),
        ).astype(np.float32),
    }


def inject_transit(time, flux, period: float, epoch: float, depth: float, duration_days: float,
                   limb_u: float = 0.5) -> np.ndarray:
    """Limb-darkened transit shape (quadratic-ish profile, no external deps)."""
    time = np.asarray(time, dtype=float)
    phase = (time - epoch + 0.5 * period) % period - 0.5 * period
    x = np.clip(np.abs(phase) / (0.5 * duration_days), 0.0, 1.0)
    inside = np.abs(phase) < 0.5 * duration_days
    mu = np.sqrt(np.clip(1.0 - x ** 2, 0.0, 1.0))
    shape = (1.0 - limb_u * (1.0 - mu)) / (1.0 - limb_u / 3.0)
    out = np.asarray(flux, dtype=float).copy()
    out[inside] *= 1.0 - depth * shape[inside]
    return out


def _duration_for(period: float, radius: float, mass: float) -> float:
    r = radius if radius and np.isfinite(radius) and radius > 0 else 1.0
    m = mass if mass and np.isfinite(mass) and mass > 0 else 1.0
    return float(13.0 / 24.0 * (period / 365.25) ** (1 / 3) * r * m ** (-1 / 3))


def scalar_features(row: dict, snr: float, depth: float, duration_days: float, period: float,
                    n_points: int, baseline: float) -> dict:
    return {
        "depth": depth, "duration_hours": duration_days * 24.0,
        "n_transits": baseline / period if period > 0 else 0.0,
        "depth_to_noise": snr, "tmag": row.get("tmag"), "radius": row.get("radius"),
        "period_days": period, "duration_to_period": duration_days / period if period > 0 else 0.0,
    }


def build_training_set(
    table,
    load_lc: Callable[[str], Optional[Tuple[np.ndarray, np.ndarray]]],
    out_dir: str | Path,
    *,
    n_injections: int = 0,
    seed: int = 0,
    limit: Optional[int] = None,
    min_snr: float = 7.0,
) -> Optional[str]:
    """Write ``{out_dir}/folds/*.npz`` and ``manifest.csv``; return the manifest path.

    ``load_lc(tic) -> (time, flux)`` returns a flattened light curve (Phase 1
    output) or None. ``table`` is :func:`load_triage_table` output (or any
    frame with the same columns).
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    folds = out / "folds"
    folds.mkdir(parents=True, exist_ok=True)
    rows = []
    quiet_curves = []
    records = table.to_dict("records")
    if limit:
        records = records[: int(limit)]
    for i, row in enumerate(records):
        try:
            lc = load_lc(row["tic"])
        except Exception as exc:
            logger.info("light curve for TIC %s failed: %s", row["tic"], exc)
            lc = None
        if lc is None:
            continue
        time, flux = (np.asarray(a, dtype=float) for a in lc)
        if row.get("quiet") and row["label"] == 0:
            quiet_curves.append((row, time, flux))
        period, epoch, snr, recovered = recover_ephemeris(
            time, flux, row["period"], row["epoch"], row["duration_days"],
        )
        if row["label"] == 1 and (not recovered or not (snr >= min_snr)):
            continue
        if not recovered:
            period, epoch = row["period"], row["epoch"]
        views = fold_views(time, flux, period, epoch)
        depth = row["depth_ppm"] * 1e-6 if np.isfinite(row.get("depth_ppm", np.nan)) else 0.0
        scalars = scalar_features(row, snr, depth, row["duration_days"], period,
                                  len(time), float(np.ptp(time)))
        path = folds / f"tce_{i:06d}_{row['tic']}.npz"
        np.savez_compressed(path, label=row["label"], centroid_offset=np.nan,
                            scalars=np.array([scalars.get(k) or 0.0 for k in sorted(scalars)], dtype=np.float32),
                            scalar_names=np.array(sorted(scalars)), **views)
        rows.append({"path": str(path), "label": row["label"], "sector": row["sector"],
                     "split": row.get("split", ""), "tic": row["tic"],
                     "source": "recovered" if recovered else "catalog"})

    made = 0
    attempts = 0
    while quiet_curves and made < n_injections and attempts < n_injections * 4:
        attempts += 1
        row, time, flux = quiet_curves[int(rng.integers(len(quiet_curves)))]
        baseline = float(np.ptp(time))
        period = float(np.exp(rng.uniform(np.log(0.5), np.log(max(min(baseline / 2.0, 20.0), 0.6)))))
        epoch = float(time.min() + rng.uniform(0, period))
        duration = _duration_for(period, row.get("radius"), row.get("mass"))
        noise = float(np.nanstd(flux)) or 1e-3
        n_in = max(len(time) * duration / period, 1.0)
        # Target SNR 7-30 so positives span the hard end near the detection limit.
        depth = float(rng.uniform(7.0, 30.0) * noise / np.sqrt(n_in))
        injected = inject_transit(time, flux, period, epoch, depth, duration)
        p_rec, t_rec, snr, recovered = recover_ephemeris(time, injected, period, epoch, duration)
        if not recovered or not (snr >= min_snr):
            continue
        views = fold_views(time, injected, p_rec, t_rec)
        scalars = scalar_features(row, snr, depth, duration, p_rec, len(time), baseline)
        path = folds / f"inj_{made:06d}_{row['tic']}.npz"
        np.savez_compressed(path, label=1, centroid_offset=0.0,
                            scalars=np.array([scalars.get(k) or 0.0 for k in sorted(scalars)], dtype=np.float32),
                            scalar_names=np.array(sorted(scalars)), **views)
        rows.append({"path": str(path), "label": 1, "sector": row["sector"],
                     "split": row.get("split", ""), "tic": row["tic"], "source": "injection"})
        made += 1

    if not rows:
        return None
    manifest = out / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    logger.info("Training manifest %s: %d rows (%d injections)", manifest, len(rows), made)
    return str(manifest)


def phase1_loader(tic: str):
    """Default ``load_lc``: our Phase 1 (MAST, all sectors, biweight-flattened)."""
    from pipeline.phase1_preprocess import run_phase1

    time, flux, _err, _label, _meta = run_phase1(f"TIC {tic}")
    return time, flux
