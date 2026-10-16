"""Transit desk API routes that do not need MAST."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from tool.server import create_app


def test_health(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["checkpoint"] is False


def test_empty_candidate_list(tmp_path):
    client = TestClient(create_app(tmp_path))
    response = client.get("/api/candidates")
    assert response.status_code == 200
    assert response.json()["candidates"] == []


def test_export_rejects_missing_epoch(tmp_path):
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    (candidates / "TIC_999_p1.00.json").write_text(json.dumps({
        "tic_id": "999",
        "best_period": 1.0,
        "sde": 8.0,
        "vetting": {"depth": 0.001, "n_transits": 3},
    }))
    client = TestClient(create_app(tmp_path))
    listed = client.get("/api/candidates").json()["candidates"]
    assert listed[0]["epoch_btjd"] is None
    response = client.post(f"/api/candidates/{listed[0]['id']}/export")
    assert response.status_code == 422
    assert "t0" in response.json()["detail"]
    assert list((tmp_path / "exports").glob("*.json")) == []

    (candidates / "TIC_1000_p2.50.json").write_text(json.dumps({
        "tic_id": "1000",
        "best_period": 2.5,
        "epoch": 0,
        "sde": 9.1,
        "alt_peaks": [{"epoch": 1710.25}],
        "vetting": {
            "depth": 0.002,
            "duration_hours": 2.4,
            "n_transits": 4,
            "transit_center_btjd": 0,
            "odd_even_ratio": float("nan"),
        },
    }, allow_nan=True))
    listed = client.get("/api/candidates").json()["candidates"]
    good = next(row for row in listed if row["tic_id"] == "1000")
    response = client.post(f"/api/candidates/{good['id']}/export")
    assert response.status_code == 200
    body = response.json()
    assert body["epoch_btjd"] == pytest.approx(1710.25)
    assert body["packet"]["candidates"][0]["epoch_btjd"] == pytest.approx(1710.25)
    assert body["packet"]["candidates"][0]["epoch_btjd"] != 0

    detail = client.get(f"/api/candidates/{good['id']}")
    assert detail.status_code == 200
    assert detail.json()["epoch_btjd"] == pytest.approx(1710.25)


def test_rank_order_on_fixture_tic_table(tmp_path):
    client = TestClient(create_app(tmp_path))
    rows = [
        {"tic_id": "1", "radius": 1.2, "n_sectors": 1, "crowdsap": 0.90, "tmag": 10, "ra": 10, "dec": 20},
        {"tic_id": "2", "radius": 0.35, "n_sectors": 13, "crowdsap": 0.97, "tmag": 11, "ra": 200, "dec": -40},
        {"tic_id": "3", "radius": 0.30, "n_sectors": 4, "crowdsap": 0.90, "tmag": 12,
         "disposition": "CP", "known_period": 5.5, "ra": 5, "dec": 5},
        {"tic_id": "4", "radius": 0.20, "n_sectors": 13, "crowdsap": 0.98, "tmag": 9,
         "cdpp_ppm": 30, "disposition": "FP", "ra": 1, "dec": 1},
        {"tic_id": "5", "radius": 0.50, "n_sectors": 2, "crowdsap": 0.20, "tmag": 12},
        {"tic_id": "6", "radius": 0.80, "n_sectors": 1, "crowdsap": 0.90, "tmag": 17.5},
    ]
    response = client.post("/api/queue/rank", json={
        "rows": rows, "limit": 10, "period_min": 0.5, "period_max": 15,
    })
    assert response.status_code == 200
    body = response.json()
    primary = [row["tic_id"] for row in body["primary"]]
    second = [row["tic_id"] for row in body["second_pass"]]
    assert primary[0] == "2"
    assert primary.index("2") < primary.index("1")
    assert "4" in primary and primary.index("4") > primary.index("1")
    assert "3" in second and "3" not in primary
    assert "5" not in primary and "5" not in second
    assert "6" not in primary and "6" not in second
    assert "ra" not in body["primary"][0] and "dec" not in body["primary"][0]
    assert "Rsun" in body["primary"][0]["why"]
    assert "sectors" in body["primary"][0]["why"]
    assert "CROWDSAP" in body["primary"][0]["why"]

    swapped = json.loads(json.dumps(rows))
    swapped[0]["ra"], swapped[0]["dec"] = 300, 80
    swapped[1]["ra"], swapped[1]["dec"] = 1, -1
    again = client.post("/api/queue/rank", json={"rows": swapped, "limit": 10}).json()
    assert [row["tic_id"] for row in again["primary"]] == primary

    best = client.post("/api/queue/rank", json={"rows": rows, "limit": 1}).json()
    assert [row["tic_id"] for row in best["primary"]] == ["2"]
    assert [row["tic_id"] for row in best["second_pass"]] == ["3"]


def test_sector_search_reports_mast_and_writes_nothing(tmp_path):
    from tool.desk import mast_requirement

    if mast_requirement()["available"]:
        pytest.skip("MAST clients are installed")
    client = TestClient(create_app(tmp_path))
    started = client.post("/api/runs", json={"mode": "sector", "sector": 15, "limit": 5})
    assert started.status_code == 200
    assert started.json()["run"]["status"] == "running"
    body = None
    for _ in range(40):
        body = client.get("/api/runs/current").json()["run"]
        if body["status"] != "running":
            break
        time.sleep(0.05)
    assert body["status"] == "failed"
    assert body["needs_mast"] is True
    assert "MAST" in body["error"]
    assert client.get("/api/candidates").json()["candidates"] == []
