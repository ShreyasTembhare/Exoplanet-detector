"""TFOP discovery-packet export tests."""

from __future__ import annotations

import json

import pytest


def test_export_round_trip(tmp_path):
    from exports.tfop_packet import export_packet

    candidates = [
        {
            "tic_id": "TIC 12345",
            "best_period": 3.5,
            "sde": 12.3,
            "prob_planet": 0.91,
            "centroid_offset": 0.4,
            "discovery_status": "NEW_CANDIDATE",
            "vetting": {
                "depth": 0.005, "duration_hours": 2.5, "n_transits": 6,
                "odd_even_ratio": 0.95, "secondary_significance": 1.0,
                "v_shape_score": 1.2,
            },
            "plot": "candidates/TIC_12345_p3.50.png",
        },
        {
            "tic_id": "TIC 99",
            "best_period": 1.2,
            "sde": 8.0,
            "prob_planet": 0.88,
            "discovery_status": "KNOWN",  # should be filtered out
            "vetting": {},
        },
    ]
    out = tmp_path / "packet.json"
    export_packet(candidates, str(out), profile="balanced")

    packet = json.loads(out.read_text())
    assert packet["schema_version"] == "1.0"
    assert packet["n_candidates"] == 1, "KNOWN should be filtered"
    cand = packet["candidates"][0]
    assert cand["tic_id"] == "TIC 12345"
    assert cand["depth_ppm"] == pytest.approx(5000.0)
    assert cand["sde"] == pytest.approx(12.3)
    assert cand["odd_even_ratio"] == pytest.approx(0.95)


def test_packet_id_stable():
    from exports.tfop_packet import TFOPCandidate

    c1 = TFOPCandidate(
        tic_id="TIC 1", period_days=3.14159, epoch_btjd=0.0,
        depth_ppm=1000.0, duration_hours=2.0, sde=10.0, n_transits=5,
        odd_even_ratio=1.0, secondary_significance=0.5, v_shape_score=1.0,
        centroid_offset_arcsec=0.5, prob_planet=0.9,
        discovery_status="NEW_CANDIDATE",
    )
    c2 = TFOPCandidate(
        tic_id="TIC 1", period_days=3.14159, epoch_btjd=99.0,  # different epoch
        depth_ppm=1000.0, duration_hours=2.0, sde=10.0, n_transits=5,
        odd_even_ratio=1.0, secondary_significance=0.5, v_shape_score=1.0,
        centroid_offset_arcsec=0.5, prob_planet=0.9,
        discovery_status="NEW_CANDIDATE",
    )
    assert c1.packet_id() == c2.packet_id()  # same tic + period -> same id
