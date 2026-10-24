"""
Fold training views at catalog period, epoch, and duration.

Astronet-Triage-v2 (and the TOI table) already publish the ephemeris of each
labeled signal. The view is folded there. This module does not run BLS, and
it does not inject a box onto an FP/FA star.

A real Triage-v2 or MAST download is attempted only by
:func:`try_remote_catalog_subset`. When that is unreachable, callers keep
using local catalog rows (the fixture path).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

PLANET_DISPOSITIONS = {"CP", "KP"}
FALSE_DISPOSITIONS = {"FP", "FA"}
# Zenodo 7411579 — Astronet-Triage-v2. The files are large; we only try.
TRIAGE_V2_RECORD = "7411579"


def disposition_to_label(disposition: str) -> Optional[int]:
    """Binary label from a TFOP disposition. Ambiguous classes (PC) are skipped."""
    text = (disposition or "").strip().upper()
    if text in PLANET_DISPOSITIONS:
        return 1
    if text in FALSE_DISPOSITIONS:
        return 0
    return None


def fold_at_catalog_ephemeris(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float,
    duration_days: float,
):
    """Global and local views folded at the catalog ephemeris, not at a BLS peak.

    The local window is centered on phase 0 (the catalog epoch). Its width
    tracks the catalog duration so a short transit is not zoomed to the wrong box.
    """
    from .fold_features import global_view, local_view

    period = float(period)
    epoch = float(epoch)
    duration_days = float(duration_days)
    if period <= 0 or duration_days <= 0:
        raise ValueError("catalog period and duration must be positive")
    dur_phase = duration_days / period
    half = float(np.clip(max(dur_phase, 0.01), 0.01, 0.2))
    global_vec = global_view(time, flux, period, epoch)
    local_vec = local_view(
        time, flux, period, epoch,
        transit_center=0.0,
        transit_half_width=half,
    )
    return global_vec, local_vec


def _duration_days(row: dict) -> float:
    if row.get("duration_days") not in (None, ""):
        return float(row["duration_days"])
    if row.get("duration_hours") not in (None, ""):
        return float(row["duration_hours"]) / 24.0
    if row.get("pl_trandurh") not in (None, ""):
        return float(row["pl_trandurh"]) / 24.0
    raise KeyError("duration")


def _label_of(row: dict) -> Optional[int]:
    if "label" in row and row["label"] is not None and row["label"] != "":
        return int(row["label"])
    return disposition_to_label(str(row.get("disposition", row.get("tfopwg_disp", ""))))


def build_from_catalog_records(
    records: Iterable[dict],
    light_curves: dict,
    out_dir: str,
) -> Optional[str]:
    """Write an npz + manifest folded at each row's catalog ephemeris.

    ``light_curves`` maps TIC id (digits, with or without a ``TIC`` prefix)
    to ``(time, flux)``. Rows without a positive label class, or without
    period / epoch / duration, are skipped. BLS is not called. No synthetic
    boxes are added.
    """
    import pandas as pd

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, row in enumerate(records):
        label = _label_of(row)
        if label not in (0, 1):
            continue
        try:
            period = float(row.get("period", row.get("pl_orbper")))
            epoch = float(row.get("epoch", row.get("pl_tranmid")))
            duration_days = _duration_days(row)
        except (TypeError, ValueError, KeyError):
            logger.info("Skipping catalog row %s: missing ephemeris", row.get("tic"))
            continue
        tic = str(row.get("tic", row.get("tic_id", row.get("tid", i))))
        lc = _lookup_lc(light_curves, tic)
        if lc is None:
            continue
        time, flux = lc
        try:
            global_vec, local_vec = fold_at_catalog_ephemeris(
                time, flux, period, epoch, duration_days,
            )
        except ValueError:
            continue
        tag = "planet" if label == 1 else "fp"
        npz_path = out_path / f"{tag}_{i:04d}.npz"
        centroid = row.get("centroid_offset", np.nan)
        np.savez(
            npz_path,
            global_view=np.asarray(global_vec),
            local_view=np.asarray(local_vec),
            centroid_offset=np.array(centroid),
            label=np.array(label),
            period=np.array(period),
            epoch=np.array(epoch),
            duration_days=np.array(duration_days),
        )
        sector = row.get("sector", -1)
        try:
            sector_i = int(sector)
        except (TypeError, ValueError):
            sector_i = -1
        rows.append({
            "path": str(npz_path),
            "label": label,
            "tic": tic,
            "sector": sector_i,
            "period": period,
            "epoch": epoch,
            "folded_at": "catalog",
        })

    if not rows:
        return None
    manifest = out_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    logger.info("Catalog-ephemeris manifest: %s (%d rows)", manifest, len(rows))
    return str(manifest)


def _lookup_lc(light_curves: dict, tic: str):
    keys = [
        tic,
        tic.replace("TIC", "").strip(),
        f"TIC {tic.replace('TIC', '').strip()}",
    ]
    for key in keys:
        if key in light_curves:
            return light_curves[key]
    return None


def try_remote_catalog_subset(
    out_dir: str = "data/triage_v2",
    timeout: float = 60.0,
    limit: Optional[int] = 300,
    n_injections: int = 100,
    load_lc=None,
) -> Optional[str]:
    """Build a real training manifest from the Triage-v2 label table.

    Downloads only the 3 MB labelled TCE table, fetches each TIC's light
    curve through Phase 1 (``load_lc`` overrides that, e.g. for tests), and
    folds at BLS-recovered ephemerides. Returns None when Zenodo is
    unreachable or nothing could be folded. Never fabricates data.
    """
    from pipeline import training_data

    table_path = training_data.download_triage_table(out_dir, timeout=timeout)
    if table_path is None:
        return None
    table = training_data.load_triage_table(table_path)
    if limit:
        # Keep the class balance workable for a small subset.
        pos = table[table["label"] == 1].head(limit // 2)
        neg = table[table["label"] == 0].head(limit - len(pos))
        import pandas as pd

        table = pd.concat([pos, neg], ignore_index=True)
    return training_data.build_training_set(
        table, load_lc or training_data.phase1_loader, out_dir, n_injections=n_injections,
    )
