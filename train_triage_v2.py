#!/usr/bin/env python3
"""
Catalog-ephemeris Triage-v2 train path.

Folds labeled signals at the published period / epoch / duration (Astronet-
Triage-v2 / TOI style), then trains TwoTowerResNet1D. Never writes a
checkpoint from Gaussian noise.

After you fetch Zenodo 7411579 (or build a local catalog + light curves):

    python train_triage_v2.py --build-from data/triage_v2/labels.csv \\
        --light-curves data/triage_v2/lcs --out-dir data/triage_v2
    python train_triage_v2.py --manifest data/triage_v2/manifest.csv --epochs 30

Offline path proof (CI / fixtures only — not a science checkpoint):

    python train_triage_v2.py --fixture-train --out /tmp/resnet1d_fixture.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_fixture_dataset(out_dir: Path, n_per_class: int = 8) -> str:
    """Synthetic catalog folds for proving the train path. Not science data."""
    from pipeline.catalog_labels import build_from_catalog_records

    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    light_curves = {}
    rng = np.random.default_rng(7)
    for i in range(n_per_class * 2):
        label = 1 if i < n_per_class else 0
        tic = str(1000 + i)
        period = 2.0 + (i % 5) * 0.4
        epoch = 0.3 + 0.05 * (i % 3)
        duration = 0.08 + 0.01 * (i % 2)
        time = np.linspace(0.0, 27.0, 1800)
        flux = np.ones_like(time) + rng.normal(0.0, 2e-4, size=time.shape)
        if label == 1:
            phase = (time - epoch) % period
            flux[(phase < duration / 2) | (phase > period - duration / 2)] -= 0.006
        else:
            # Shallow EB-like secondary for negatives — still folded at catalog ephemeris.
            phase = (time - epoch) % period
            flux[(phase < duration / 2) | (phase > period - duration / 2)] -= 0.002
            flux[((phase - 0.5) % 1.0) < duration / 2] -= 0.0015
        light_curves[tic] = (time, flux)
        records.append({
            "tic": tic,
            "label": label,
            "period": period,
            "epoch": epoch,
            "duration_days": duration,
            "sector": 1 + (i % 4),
        })
    manifest = build_from_catalog_records(records, light_curves, str(out_dir))
    if not manifest:
        raise RuntimeError("Fixture catalog build produced no samples.")
    return manifest


def build_from_label_table(labels_csv: Path, lc_dir: Path, out_dir: Path) -> str:
    """Fold rows from a CSV of catalog ephemerides against local light-curve CSVs.

    Expected label columns: tic, disposition or label, period, epoch,
    duration_days or duration_hours, optional sector.
    Light curves: ``{tic}.csv`` with time,flux under ``lc_dir``.
    """
    from pipeline.catalog_labels import build_from_catalog_records

    records = list(csv.DictReader(labels_csv.open()))
    light_curves = {}
    for path in sorted(lc_dir.glob("*.csv")):
        tic = path.stem.replace("TIC", "").replace("_", "").strip()
        rows = list(csv.DictReader(path.open()))
        if not rows:
            # allow headerless time,flux
            raw = path.read_text().strip().splitlines()
            start = 1 if raw and "time" in raw[0].lower() else 0
            time, flux = [], []
            for line in raw[start:]:
                parts = line.replace(",", " ").split()
                if len(parts) >= 2:
                    time.append(float(parts[0]))
                    flux.append(float(parts[1]))
            if len(time) >= 30:
                light_curves[tic] = (np.asarray(time), np.asarray(flux))
            continue
        time = np.asarray([float(r.get("time") or r.get("TIME")) for r in rows], dtype=np.float64)
        flux = np.asarray([float(r.get("flux") or r.get("FLUX")) for r in rows], dtype=np.float64)
        light_curves[tic] = (time, flux)
        light_curves[path.stem] = (time, flux)
    manifest = build_from_catalog_records(records, light_curves, str(out_dir))
    if not manifest:
        raise RuntimeError("No catalog-ephemeris samples were written. Check labels and light curves.")
    return manifest


def try_fetch_remote(out_dir: Path) -> str | None:
    from pipeline.catalog_labels import try_remote_catalog_subset

    return try_remote_catalog_subset(out_dir=str(out_dir))


def train_manifest(manifest: str, out: Path, epochs: int, batch_size: int) -> dict:
    from train_classifier import run_train

    result = run_train(
        data=manifest,
        epochs=epochs,
        batch_size=batch_size,
        out=str(out),
        patience=max(2, epochs),
        use_injection_recovery=False,
    )
    if isinstance(result, dict) and result.get("refused"):
        raise RuntimeError(f"Training refused: {result.get('reason')}")
    metrics_path = Path(str(out) + ".metrics.json")
    if not metrics_path.is_file():
        # train_classifier may write beside the checkpoint with a different suffix
        alt = out.with_suffix(".metrics.json")
        if alt.is_file():
            metrics_path = alt
    meta = {
        "checkpoint": str(out),
        "manifest": manifest,
        "metrics_path": str(metrics_path) if metrics_path.is_file() else None,
        "result": result if isinstance(result, dict) else {"ok": True},
    }
    side = out.with_suffix(out.suffix + ".triage.json")
    side.write_text(json.dumps(meta, indent=2))
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description="Catalog-ephemeris Triage-v2 train path")
    parser.add_argument("--manifest", type=str, default=None, help="Existing catalog-fold manifest.csv")
    parser.add_argument("--build-from", type=str, default=None, help="Labels CSV to fold at catalog ephemerides")
    parser.add_argument("--light-curves", type=str, default=None, help="Directory of {tic}.csv light curves")
    parser.add_argument("--out-dir", type=str, default="data/triage_v2", help="Where to write folds + manifest")
    parser.add_argument("--fetch", action="store_true", help="Try Zenodo Triage-v2 remote subset first")
    parser.add_argument(
        "--fixture-train",
        action="store_true",
        help="Build a tiny synthetic catalog set and train (path proof only, not science weights)",
    )
    parser.add_argument("--out", type=str, default="models/checkpoints/resnet1d.pt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir)

    manifest = args.manifest
    if args.fixture_train:
        fixture_dir = out_dir / "fixture"
        manifest = build_fixture_dataset(fixture_dir)
        logger.info("Fixture manifest at %s (not a science dataset).", manifest)
    elif args.fetch:
        remote = try_fetch_remote(out_dir)
        if remote:
            manifest = remote
        else:
            logger.warning(
                "Remote Triage-v2 subset unavailable. "
                "Download Zenodo 7411579 locally, then pass --build-from / --manifest."
            )
    if not manifest and args.build_from:
        if not args.light_curves:
            raise SystemExit("--light-curves is required with --build-from")
        manifest = build_from_label_table(Path(args.build_from), Path(args.light_curves), out_dir)

    if not manifest:
        parser.print_help()
        raise SystemExit(
            "\nNo manifest. Use --fixture-train, --manifest, or --build-from with --light-curves "
            "(after a Zenodo/MAST fetch)."
        )

    meta = train_manifest(manifest, out, epochs=args.epochs, batch_size=args.batch_size)
    logger.info("Wrote checkpoint path record: %s", meta.get("checkpoint"))
    print(json.dumps({"checkpoint": meta["checkpoint"], "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
