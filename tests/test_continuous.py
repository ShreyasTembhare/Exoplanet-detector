"""Continuous search: log, local files, pause, and refresh. No MAST."""

from __future__ import annotations

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from tool.desk import Desk
from tool.server import create_app


def _curve(path, samples=48):
    rows = ["time,flux"]
    for i in range(samples):
        rows.append(f"{1700 + i * 0.02:.4f},{1.0 + (i % 3) * 1e-4:.6f}")
    path.write_text("\n".join(rows) + "\n")


def _wait(client, pred, timeout=25):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = client.get("/api/continuous").json()
        if pred(last):
            return last
        time.sleep(0.05)
    raise AssertionError(last)


def test_log_lines_append(tmp_path):
    curves = tmp_path / "curves"
    curves.mkdir()
    _curve(curves / "only.csv")
    desk = tmp_path / "desk"
    client = TestClient(create_app(desk))
    started = client.post("/api/continuous/start", json={"local_dir": str(curves), "nperiods": 64})
    assert started.status_code == 200
    first = _wait(client, lambda body: len(body["lines"]) >= 1)
    text1 = (desk / "continuous.log").read_text()
    second = _wait(client, lambda body: len(body["lines"]) >= 2)
    text2 = (desk / "continuous.log").read_text()
    assert text2.startswith(text1)
    assert second["lines"][0] == first["lines"][0]
    assert second["lines"][0]["target"] == "only.csv"
    assert second["lines"][1]["target"] == "only.csv"
    assert second["lines"][1]["phase"] == "result"
    assert "ts" in second["lines"][1]
    assert second["lines"][1]["candidate"] is False


def test_loop_advances_to_next_local_file(tmp_path):
    curves = tmp_path / "curves"
    curves.mkdir()
    _curve(curves / "a.csv")
    _curve(curves / "b.csv")
    client = TestClient(create_app(tmp_path / "desk"))
    response = client.post("/api/continuous/start", json={"local_dir": str(curves), "nperiods": 64})
    assert response.status_code == 200
    body = _wait(
        client,
        lambda item: [line["target"] for line in item["lines"] if line["phase"] == "result"] == ["a.csv", "b.csv"],
    )
    assert body["status"] == "stopped"
    assert client.get("/api/candidates").json()["candidates"] == []


def test_pause_stops_new_work(tmp_path, monkeypatch):
    curves = tmp_path / "curves"
    curves.mkdir()
    _curve(curves / "a.csv")
    _curve(curves / "b.csv")
    started = threading.Event()
    release = threading.Event()
    import tool.continuous as continuous

    real = continuous.run_local_lightcurve
    calls = []

    def wrapped(time, flux, tic_id, candidate_dir, **kwargs):
        calls.append(tic_id)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)
        return real(time, flux, tic_id, candidate_dir, **kwargs)

    monkeypatch.setattr(continuous, "run_local_lightcurve", wrapped)
    client = TestClient(create_app(tmp_path / "desk"))
    response = client.post("/api/continuous/start", json={"local_dir": str(curves), "nperiods": 64})
    assert response.status_code == 200
    assert started.wait(5)
    paused = client.post("/api/continuous/pause")
    assert paused.status_code == 200
    assert paused.json()["status"] == "paused"
    release.set()
    body = _wait(
        client,
        lambda item: any(line["phase"] == "result" and line["target"] == "a.csv" for line in item["lines"]),
    )
    time.sleep(0.4)
    body = client.get("/api/continuous").json()
    assert body["status"] == "paused"
    assert "b.csv" not in [line["target"] for line in body["lines"]]
    assert calls == ["a"]


def test_refresh_reads_the_same_log(tmp_path):
    curves = tmp_path / "curves"
    curves.mkdir()
    _curve(curves / "only.csv")
    desk_dir = tmp_path / "desk"
    client = TestClient(create_app(desk_dir))
    client.post("/api/continuous/start", json={"local_dir": str(curves), "nperiods": 64})
    first = _wait(client, lambda body: any(line["phase"] == "result" for line in body["lines"]))
    second = client.get("/api/continuous").json()
    assert second["lines"][: len(first["lines"])] == first["lines"]
    disk = [json.loads(line) for line in (desk_dir / "continuous.log").read_text().splitlines() if line.strip()]
    assert disk[: len(first["lines"])] == first["lines"]
    finished = _wait(client, lambda body: body["status"] == "stopped")
    again = client.get("/api/continuous").json()
    assert again["lines"] == finished["lines"]
    reloaded = Desk(desk_dir)
    assert reloaded.continuous.snapshot()["lines"] == again["lines"]
    assert reloaded.continuous.snapshot()["status"] == "stopped"


def test_ranked_queue_is_walked_when_mast_is_missing(tmp_path):
    from tool.desk import mast_requirement

    if mast_requirement()["available"]:
        pytest.skip("MAST clients are installed")
    client = TestClient(create_app(tmp_path))
    ranked = client.post("/api/queue/rank", json={
        "rows": [
            {"tic_id": "1", "radius": 1.2, "n_sectors": 1, "crowdsap": 0.9, "tmag": 10},
            {"tic_id": "2", "radius": 0.35, "n_sectors": 13, "crowdsap": 0.97, "tmag": 11},
        ],
        "limit": 10,
    })
    assert ranked.status_code == 200
    client.post("/api/continuous/start", json={"sector": 4, "nperiods": 64})
    body = _wait(client, lambda item: item["status"] == "stopped")
    results = [line for line in body["lines"] if line["phase"] == "result"]
    assert [line["target"] for line in results] == ["2", "1"]
    assert all(line["candidate"] is False for line in results)
    assert all("MAST" in (line.get("error") or "") for line in results)
    assert client.get("/api/candidates").json()["candidates"] == []


def test_missing_mast_does_not_invent_stars(tmp_path):
    from tool.desk import mast_requirement

    if mast_requirement()["available"]:
        pytest.skip("MAST clients are installed")
    client = TestClient(create_app(tmp_path))
    started = client.post("/api/continuous/start", json={"sector": 15, "nperiods": 64})
    assert started.status_code == 200
    body = _wait(client, lambda item: item["status"] == "paused")
    assert body["sector"] == 15
    assert any("MAST" in (line.get("error") or "") for line in body["lines"])
    assert client.get("/api/candidates").json()["candidates"] == []
    time.sleep(0.3)
    later = client.get("/api/continuous").json()
    assert later["sector"] == 15
    assert len(later["lines"]) == len(body["lines"])
