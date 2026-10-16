"""Catalog-ephemeris Triage-v2 train path (fixture only — no Gaussian science weights)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_fixture_train_writes_checkpoint(tmp_path):
    pytest.importorskip("torch")
    from train_triage_v2 import build_fixture_dataset, train_manifest

    manifest = build_fixture_dataset(tmp_path / "folds", n_per_class=6)
    assert Path(manifest).is_file()
    out = tmp_path / "resnet1d.pt"
    meta = train_manifest(manifest, out, epochs=2, batch_size=4)
    assert out.is_file()
    assert meta["checkpoint"] == str(out)
    # Metrics may be written beside the checkpoint when training improves.
    side = Path(str(out) + ".triage.json")
    assert side.is_file()
    payload = json.loads(side.read_text())
    assert payload["manifest"] == manifest


def test_cli_fixture_train(tmp_path):
    pytest.importorskip("torch")
    from train_triage_v2 import main

    out = tmp_path / "ckpt.pt"
    code = main([
        "--fixture-train",
        "--out-dir", str(tmp_path / "data"),
        "--out", str(out),
        "--epochs", "2",
        "--batch-size", "2",
    ])
    assert code == 0
    assert out.is_file()


def test_build_from_label_table(tmp_path):
    from train_triage_v2 import build_from_label_table

    labels = tmp_path / "labels.csv"
    labels.write_text(
        "tic,label,period,epoch,duration_days,sector\n"
        "1,1,3.0,0.5,0.1,2\n"
        "2,0,4.0,1.0,0.12,2\n"
    )
    lc_dir = tmp_path / "lcs"
    lc_dir.mkdir()
    for tic, depth in (("1", 0.006), ("2", 0.0)):
        rows = ["time,flux"]
        for i in range(200):
            t = 1700 + i * 0.02
            flux = 1.0 - depth if (i % 50) < 3 else 1.0
            rows.append(f"{t:.4f},{flux:.6f}")
        (lc_dir / f"{tic}.csv").write_text("\n".join(rows) + "\n")
    manifest = build_from_label_table(labels, lc_dir, tmp_path / "out")
    assert Path(manifest).is_file()
    text = Path(manifest).read_text()
    assert "catalog" in text
    assert "1" in text and "2" in text
