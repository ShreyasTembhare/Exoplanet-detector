"""Unblock: depth epoch, two-tower contract, SDE candidates, no Gaussian checkpoint, TFOP t0."""

from __future__ import annotations

import json

import numpy as np
import pytest


def _circular_days(a, b, period):
    delta = abs((a - b) % period)
    return min(delta, period - delta)


def test_epoch_for_period_matches_injected_t0():
    from pipeline.bls_gpu import _epoch_for_period

    period = 3.5
    duration = 0.12
    t0 = 1702.4
    t = np.linspace(1700.0, 1727.0, 6000)
    flux = np.ones_like(t)
    phase = (t - t0) % period
    half = duration / 2
    flux[(phase < half) | (phase > period - half)] -= 0.008

    epoch = _epoch_for_period(t, flux, period, duration / period)
    assert _circular_days(epoch, t0, period) <= 0.5 * duration


def test_two_tower_save_load_and_batch(tmp_path):
    torch = pytest.importorskip("torch")
    from models.resnet1d import TwoTowerResNet1D, load_checkpoint, make_two_tower_inputs

    model = TwoTowerResNet1D()
    path = tmp_path / "resnet1d.pt"
    torch.save({"model": model.state_dict(), "arch": "two_tower"}, path)
    loaded = load_checkpoint(str(path), device="cpu", strict=True)

    rng = np.random.default_rng(0)
    globals_, locals_ = [], []
    for _ in range(2):
        g, l = make_two_tower_inputs(rng.normal(size=2048), rng.normal(size=256))
        globals_.append(g)
        locals_.append(l)
    batch_g = torch.from_numpy(np.stack(globals_, axis=0)).float()
    batch_l = torch.from_numpy(np.stack(locals_, axis=0)).float()
    centroid = torch.zeros(2)
    logits = loaded(batch_g, batch_l, centroid_offset=centroid)
    assert tuple(logits.shape) == (2, 2)


def test_predict_batch_uses_two_tower_forward(tmp_path):
    torch = pytest.importorskip("torch")
    from hunter import _predict_batch
    from models.resnet1d import TwoTowerResNet1D

    model = TwoTowerResNet1D().eval()
    rng = np.random.default_rng(1)
    items = [
        {
            "global_view": rng.normal(size=2048).astype(np.float32),
            "local_view": rng.normal(size=256).astype(np.float32),
            "centroid_offset": 0.2,
        }
    ]
    probs = _predict_batch(model, torch.device("cpu"), items)
    assert len(probs) == 1
    assert 0.0 <= probs[0] <= 1.0


def test_run_pipeline_predict_two_tower(tmp_path):
    pytest.importorskip("torch")
    import torch

    from models.resnet1d import TwoTowerResNet1D
    from run_pipeline import predict

    model = TwoTowerResNet1D()
    path = tmp_path / "resnet1d.pt"
    torch.save({"model": model.state_dict()}, path)
    rng = np.random.default_rng(2)
    result = {
        "global_view": rng.normal(size=2048).astype(np.float32),
        "local_view": rng.normal(size=256).astype(np.float32),
        "centroid_offset": 0.1,
    }
    out = predict(result, checkpoint_path=str(path))
    assert out is not None
    assert 0.0 <= out["prob_planet"] <= 1.0


def test_sde_pass_without_checkpoint_is_candidate(tmp_path):
    from hunter import _save_candidate_evidence, status_for_probability

    assert status_for_probability(None, 0.85) == "CANDIDATE_SDE"
    assert status_for_probability(0.2, 0.85) == "CLEARED"
    assert status_for_probability(0.9, 0.85).startswith("CANDIDATE (")

    _save_candidate_evidence(
        str(tmp_path), "TIC 42", 3.5, None, 0.01, 9.5, 0.1,
        "plot.png",
        vetting={"depth": 0.004, "n_transits": 6, "odd_even_ratio": 0.98},
        peaks=[{"period": 3.5, "epoch": 1711.2, "sde": 9.5}],
        strategy_profile="test",
        epoch=1711.2,
    )
    payload = json.loads(next(tmp_path.glob("TIC_*.json")).read_text())
    assert payload["epoch"] == pytest.approx(1711.2)
    assert payload["vetting"]["depth"] == pytest.approx(0.004)
    assert payload["vetting"]["transit_center_btjd"] == pytest.approx(1711.2)


def test_train_refuses_gaussian_checkpoint(tmp_path):
    pytest.importorskip("torch")
    from train_classifier import run_train

    out = tmp_path / "resnet1d.pt"
    result = run_train(data=None, out=str(out), epochs=1, patience=1)
    assert result["refused"] is True
    assert result["reason"] == "missing_data"
    assert not out.exists()


def test_packet_writes_phase2_epoch_not_zero():
    from exports.tfop_packet import candidate_json_to_tfop, export_packet

    candidate = {
        "tic_id": "TIC 7",
        "best_period": 4.25,
        "epoch": 2459.5,
        "sde": 11.0,
        "prob_planet": None,
        "discovery_status": "NEW_CANDIDATE",
        "vetting": {"transit_center_btjd": 0.0, "depth": 0.002, "n_transits": 4},
        "alt_peaks": [{"period": 8.5, "epoch": 100.0}],
    }
    packet_row = candidate_json_to_tfop(candidate)
    assert packet_row.epoch_btjd == pytest.approx(2459.5)
    assert packet_row.epoch_btjd != 0.0

    # No top-level epoch: fall through to a real alternate peak, never a stored 0.
    missing = {
        "tic_id": "TIC 8",
        "best_period": 2.0,
        "sde": 8.0,
        "vetting": {"transit_center_btjd": 0},
        "alt_peaks": [{"epoch": 1800.25}],
    }
    assert candidate_json_to_tfop(missing).epoch_btjd == pytest.approx(1800.25)

    assert export_packet.__name__ == "export_packet"
