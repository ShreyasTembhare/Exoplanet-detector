"""Phase-3 ML: Triage-v2 loader, recovered folds + injections, calibration,
sector holdout, ExoMiner stage, desk-label fine-tuning, completeness maps."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from pipeline import training_data

TRIAGE_CSV = """# Astronet-Triage-v2 TCE information
# TIC ID
TIC ID,Epoch,Period,Duration,Depth,SMass,SRad,SRadEst,Tmag,Year,MinT,MaxT,Split,Consensus Label,L1,L2,L3,L4,L5,L6,L7,L8,Notes
,BTJD,days,days,ppm,M_sun,R_sun,R_sun,,,BTJD,BTJD,,,,,,,,,,,
101,1500.40,3.0,0.12,8000,1.0,1.0,1.0,10.0,1,1490,1517,train,E,,,,,,,,,
102,1500.70,2.2,0.10,6000,1.0,1.0,1.0,10.5,1,1490,1517,val,J,,,,,,,,,
103,1501.00,4.0,0.10,5000,1.0,1.0,1.0,11.0,1,1490,1517,train,,E,E,E,J,,,,,
104,1501.00,4.0,0.10,5000,1.0,1.0,1.0,11.0,1,1490,1517,train,N,,,,,,,,,
105,1700.00,5.0,0.10,5000,1.0,1.0,1.0,11.0,2,1690,1717,test,B,,,,,,,,,
106,1501.20,6.0,0.10,7000,1.0,1.0,1.0,11.0,1,1490,1517,train,E,,,,,,,,,
"""


def _box(t, period, epoch, dur, depth):
    phase = (t - epoch + 0.5 * period) % period - 0.5 * period
    return np.where(np.abs(phase) < dur / 2, -depth, 0.0)


def test_triage_table_labels_and_sectors(tmp_path):
    path = tmp_path / training_data.TRIAGE_TABLE
    path.write_text(TRIAGE_CSV)
    table = training_data.load_triage_table(path)
    labels = dict(zip(table["tic"], table["label"]))
    assert labels == {"101": 1, "102": 0, "103": 1, "105": 0, "106": 1}  # N dropped; 3/4 E votes -> 1
    assert table.set_index("tic").loc["102", "quiet"]
    assert table.set_index("tic").loc["105", "sector"] > table.set_index("tic").loc["101", "sector"]


def test_download_uses_cache_and_survives_offline(tmp_path):
    calls = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(url):
        calls.append(url)
        return _Resp(TRIAGE_CSV.encode())

    path = training_data.download_triage_table(tmp_path, opener=opener)
    assert path.read_text().startswith("# Astronet")
    assert training_data.download_triage_table(tmp_path, opener=opener) == path
    assert len(calls) == 1

    def offline(url):
        raise OSError("no network")

    assert training_data.download_triage_table(tmp_path / "other", opener=offline) is None


def _fake_curves():
    rng = np.random.default_rng(5)
    t = np.arange(1490.0, 1517.0, 10 / 1440)
    noise = lambda: 1 + rng.normal(0, 4e-4, t.size)  # noqa: E731
    return {
        "101": (t, noise() + _box(t, 3.0, 1500.40, 0.12, 0.008)),
        "102": (t, noise()),
        "103": (t, noise() + _box(t, 4.0, 1501.0, 0.10, 0.005)),
        "105": (t, noise() + _box(t, 5.0, 1500.0, 0.10, 0.005)),
        "106": (t, noise()),  # labelled E but nothing there: our search cannot recover it
    }


def test_build_training_set_recovers_folds_and_injects(tmp_path):
    path = tmp_path / training_data.TRIAGE_TABLE
    path.write_text(TRIAGE_CSV)
    table = training_data.load_triage_table(path)
    curves = _fake_curves()
    manifest = training_data.build_training_set(table, curves.get, tmp_path / "out", n_injections=3, seed=1)
    import pandas as pd

    rows = pd.read_csv(manifest)
    assert "106" not in rows["tic"].astype(str).tolist()  # unrecoverable positive dropped
    assert set(rows[rows["source"] == "recovered"]["tic"].astype(str)) >= {"101", "103"}
    assert (rows["source"] == "injection").sum() == 3
    assert set(rows[rows["source"] == "injection"]["label"]) == {1}
    sample = np.load(rows.iloc[0]["path"])
    assert sample["global_view"].shape[0] > sample["local_view"].shape[0] > 0
    assert "local_centered" in sample and "periodogram" in sample


def test_recover_ephemeris_finds_the_dip_not_the_catalog_value():
    t = np.arange(1500.0, 1527.0, 10 / 1440)
    flux = 1 + _box(t, 3.01, 1500.52, 0.1, 0.01)
    p, t0, snr, ok = training_data.recover_ephemeris(t, flux, 3.0, 1500.5, 0.1)
    assert ok and abs(p - 3.01) < 0.005 and snr > 7


def test_temperature_scaling_fixes_overconfidence():
    from models.calibration import calibrated_probs, expected_calibration_error, fit_temperature

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 2000)
    true_logit = np.where(labels == 1, 1.0, -1.0) + rng.normal(0, 1.5, labels.size)
    logits = np.stack([np.zeros_like(true_logit), true_logit * 4.0], axis=1)  # 4x overconfident
    temperature = fit_temperature(logits, labels)
    assert 2.5 < temperature < 6.0
    assert expected_calibration_error(calibrated_probs(logits, temperature), labels) < \
        expected_calibration_error(calibrated_probs(logits, 1.0), labels)
    assert fit_temperature(logits[:5], np.ones(5, dtype=int)) == 1.0


def test_default_split_holds_out_latest_sectors():
    from train_classifier import sector_disjoint_split

    sectors = [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10 + [5] * 10
    train, val = sector_disjoint_split(50, sectors, None)
    assert {sectors[i] for i in val} == {5}
    assert {sectors[i] for i in train} == {1, 2, 3, 4}


def test_exominer_blocks_are_residual_and_scalars_nan_safe():
    torch = pytest.importorskip("torch")
    from models.exominer import ExoMinerVetter, _ResBlock1D, make_exominer_inputs, scalar_vector

    block = _ResBlock1D(8, 16)
    for p in block.body.parameters():
        torch.nn.init.zeros_(p)
    x = torch.randn(2, 8, 32)
    block.eval()
    # With a zeroed body the output is the (projected) skip path, not zero.
    assert block(x).abs().sum() > 0
    vec = scalar_vector({"depth": float("nan"), "period_days": 100.0, "sde_bls": None})
    assert np.all(np.isfinite(vec)) and vec.max() <= 2.0
    model = ExoMinerVetter().eval()
    inputs = [torch.from_numpy(np.stack([a, a])).float()
              for a in make_exominer_inputs(np.ones(2048), np.ones(256), scalars_dict={"depth": 1e-3})]
    assert model(*inputs).shape == (2, 2)


def test_exominer_trains_and_scores_as_second_stage(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from pipeline import second_stage

    path = tmp_path / training_data.TRIAGE_TABLE
    path.write_text(TRIAGE_CSV)
    manifest = training_data.build_training_set(
        training_data.load_triage_table(path), _fake_curves().get, tmp_path / "data", n_injections=4,
    )
    out = tmp_path / "exominer.pt"
    result = second_stage.train_exominer(manifest, str(out), epochs=1, batch_size=4)
    assert out.is_file() and result["temperature"] > 0
    monkeypatch.setenv("EXOMINER_CHECKPOINT", str(out))
    bundle = second_stage.load_vetter()
    rows = [{"global_view": np.ones(2048), "local_view": np.ones(256), "vetting": {"depth": 1e-3}, "sde": 9.0},
            {"global_view": None, "local_view": None}]
    probs = second_stage.vet(bundle, rows)
    assert 0.0 <= probs[0] <= 1.0 and probs[1] is None
    assert second_stage.combine(0.81, 0.25) == pytest.approx(0.45)
    assert second_stage.combine(None, 0.3) == 0.3 and second_stage.combine(0.3, None) == 0.3


def test_desk_labels_fine_tune_existing_checkpoint(tmp_path):
    pytest.importorskip("torch")
    from hunter import _save_candidate_evidence
    from tool.desk import Desk
    from train_classifier import desk_label_manifest, run_train
    from train_triage_v2 import build_fixture_dataset, train_manifest

    out = tmp_path / "resnet1d.pt"
    train_manifest(build_fixture_dataset(tmp_path / "fixture", n_per_class=4), out, epochs=1, batch_size=4)

    desk = Desk(tmp_path / "desk", enable_auto_start=False)
    rng = np.random.default_rng(0)
    for i in range(6):
        g = 1 + rng.normal(0, 1e-3, 2048)
        g[1000:1030] -= 0.01 if i % 2 == 0 else 0.0
        _save_candidate_evidence(
            str(desk.candidates), f"TIC {900 + i}", 2.0 + i, None, 0.01, 9.0, 0.0, "",
            vetting={"depth": 0.01}, peaks=[], strategy_profile="balanced", epoch=1500.0,
            views={"global_view": g, "local_view": g[::8], "centroid_offset": 0.0},
        )
    ids = sorted(row["id"] for row in desk.list_candidates())
    assert len(ids) == 6
    for i, cid in enumerate(ids):
        desk.label_candidate(cid, "planet" if i % 2 == 0 else "false_positive")
    desk.label_candidate(ids[0], "planet")

    manifest = desk_label_manifest(str(desk.root), out_dir=str(tmp_path / "labels"))
    assert manifest and len(Path(manifest).read_text().strip().splitlines()) == 1 + len(ids)
    before = out.stat().st_mtime_ns
    result = run_train(labels=str(desk.root), out=str(out), epochs=1, batch_size=2, patience=1)
    assert not (isinstance(result, dict) and result.get("refused"))
    assert out.stat().st_mtime_ns >= before


def test_completeness_map_lookup_and_planner_feedback(tmp_path):
    from pipeline import completeness
    from pipeline.target_rank import TargetRecord, expected_yield

    small = completeness.run_injection_recovery(periods=(1.0, 3.0), radii=(2.0, 8.0), tmags=(9.0,),
                                                n_per_cell=1, seed=0)
    assert len(small["recovery"]) == 1 and len(small["recovery"][0]) == 2
    assert small["recovery"][0][1][1] == 1.0  # 8 Re at Tmag 9 is always found

    data = {"periods": [1.0, 10.0], "radii": [1.0, 4.0], "tmags": [8.0, 12.0],
            "recovery": [[[0.2, 1.0], [0.0, 0.8]], [[0.0, 0.6], [0.0, 0.2]]]}
    assert completeness.lookup(data, 1.0, 4.0, 8.0) == pytest.approx(1.0)
    assert completeness.lookup(data, 1.0, 2.0, 8.0) == pytest.approx(0.6)  # halfway in log R
    # Four sectors look 1.5 mag brighter than one.
    assert completeness.detection_probability(data, 1.0, 4.0, 12.0, n_sectors=1) < \
        completeness.detection_probability(data, 1.0, 4.0, 12.0, n_sectors=4)

    star = TargetRecord(tic_id="1", radius=1.0, n_sectors=1, tmag=10.0, cdpp_ppm=100)
    nothing = {**data, "recovery": [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]}
    assert expected_yield(star, completeness=nothing) == 0.0
    assert expected_yield(star, completeness=data) > 0.0

    path = tmp_path / "completeness.json"
    completeness.save_map(small, path)
    assert completeness.map_age_hours(path) < 1
    assert completeness.refresh_if_stale(path, max_age_hours=24) is False


def test_completeness_endpoint(tmp_path):
    from fastapi.testclient import TestClient

    from tool.server import create_app

    client = TestClient(create_app(tmp_path / "desk", enable_auto_start=False))
    assert client.get("/api/completeness").json()["available"] is False
    (tmp_path / "desk" / "completeness.json").write_text(json.dumps(
        {"periods": [1.0], "radii": [2.0], "tmags": [10.0], "recovery": [[[0.5]]],
         "generated_at": "2026-01-01T00:00:00+00:00", "trials": 1}))
    body = client.get("/api/completeness").json()
    assert body["available"] is True and body["recovery"] == [[[0.5]]]
