"""Crash-safe resume: leases, retries, idempotent enqueue, restart recovery."""

from __future__ import annotations

import json
import time

from fastapi.testclient import TestClient

from tool.state import StateStore


def test_enqueue_is_idempotent(tmp_path):
    store = StateStore(tmp_path / "s.db")
    items = [{"kind": "queue", "target": str(i), "sector": 5} for i in range(3)]
    assert store.enqueue("p", items) == 3
    assert store.enqueue("p", items) == 0
    assert store.counts("p")["pending"] == 3


def test_expired_lease_is_reclaimed(tmp_path):
    store = StateStore(tmp_path / "s.db")
    store.enqueue("p", [{"kind": "k", "target": "a"}])
    first = store.claim("worker-1", plan="p", lease_seconds=0.05)
    assert first["target"] == "a"
    assert store.claim("worker-2", plan="p") is None
    time.sleep(0.1)
    again = store.claim("worker-2", plan="p")
    assert again["id"] == first["id"]
    assert again["attempts"] == 2


def test_fail_retries_then_parks(tmp_path):
    store = StateStore(tmp_path / "s.db")
    store.enqueue("p", [{"kind": "k", "target": "a"}, {"kind": "k", "target": "b"}], max_attempts=2)
    a = store.claim("w", plan="p")
    assert store.fail(a["id"], "boom") == "pending"
    # The retried item goes to the back; b is next.
    assert store.claim("w", plan="p")["target"] == "b"
    a2 = store.claim("w", plan="p")
    assert a2["target"] == "a"
    assert store.fail(a2["id"], "boom again") == "failed"
    assert store.counts("p")["failed"] == 1
    assert store.retry_item(a2["id"])
    assert store.counts("p")["pending"] == 1


def test_release_stale_frees_dead_owner(tmp_path):
    store = StateStore(tmp_path / "s.db")
    store.enqueue("p", [{"kind": "k", "target": "a"}])
    store.claim("dead-process", plan="p", lease_seconds=3600)
    assert StateStore(tmp_path / "s.db").release_stale("new-process") == 1
    assert store.claim("new-process", plan="p")["target"] == "a"


def test_backup_is_consistent(tmp_path):
    store = StateStore(tmp_path / "s.db")
    store.set_json("k", {"v": 1})
    copy = store.backup_to(tmp_path / "backup" / "s.db")
    assert StateStore(copy).get_json("k") == {"v": 1}


def _curve(path, samples=48):
    rows = ["time,flux"] + [f"{1700 + i * 0.02:.4f},{1.0 + (i % 3) * 1e-4:.6f}" for i in range(samples)]
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


def test_restart_resumes_same_plan_without_replay(tmp_path):
    from tool.server import create_app

    curves = tmp_path / "curves"
    curves.mkdir()
    for name in ("a", "b", "c"):
        _curve(curves / f"{name}.csv")
    desk_dir = tmp_path / "desk"
    store = StateStore(desk_dir / "state.db")
    # Simulate a crash after 'a' finished and while 'b' was leased by a dead process.
    plan = f"local:{curves.resolve()}"
    store.enqueue(plan, [
        {"kind": "local", "target": f"{n}.csv", "payload": {"path": str(curves / f'{n}.csv')}}
        for n in ("a", "b", "c")
    ])
    a = store.claim("old", plan=plan)
    store.complete(a["id"])
    store.claim("old", plan=plan, lease_seconds=3600)
    store.set_json("continuous", {
        "status": "running", "mode": "local", "plan": plan, "local_dir": str(curves.resolve()),
        "config": {"nperiods": 64}, "auto_start": True,
    })
    client = TestClient(create_app(desk_dir, enable_auto_start=False))
    body = _wait(client, lambda s: s["status"] == "stopped")
    searched = [line["target"] for line in body["lines"] if line["phase"] == "result"]
    assert searched == ["b.csv", "c.csv"]
    assert body["progress"]["work"]["done"] == 3


def test_finished_pass_does_not_replay_on_restart(tmp_path):
    from tool.server import create_app

    curves = tmp_path / "curves"
    curves.mkdir()
    _curve(curves / "a.csv")
    desk_dir = tmp_path / "desk"
    client = TestClient(create_app(desk_dir, enable_auto_start=False))
    client.post("/api/continuous/start", json={"local_dir": str(curves), "nperiods": 64})
    _wait(client, lambda s: s["status"] == "stopped")
    again = client.post("/api/continuous/start", json={"local_dir": str(curves), "nperiods": 64})
    assert again.status_code == 200
    body = _wait(client, lambda s: s["status"] == "stopped")
    results = [line for line in body["lines"] if line["phase"] == "result"]
    assert [line["target"] for line in results] == ["a.csv"]


def test_legacy_control_file_is_imported(tmp_path):
    from tool.desk import Desk

    desk_dir = tmp_path / "desk"
    desk_dir.mkdir()
    (desk_dir / "continuous.json").write_text(json.dumps({"status": "stopped", "sector": 12, "auto_start": False}))
    desk = Desk(desk_dir, enable_auto_start=True)
    assert desk.continuous.snapshot()["sector"] == 12
    assert desk.continuous.snapshot()["status"] == "stopped"


def test_one_off_run_survives_restart(tmp_path):
    from tool.desk import Desk

    desk = Desk(tmp_path, enable_auto_start=False)
    desk.store.save_run({"id": "run-1", "status": "running", "mode": "sector"})
    again = Desk(tmp_path, enable_auto_start=False)
    assert again.current_run()["status"] == "failed"
    assert "restart" in again.current_run()["error"]


def test_events_paginate(tmp_path):
    from tool.server import create_app

    curves = tmp_path / "curves"
    curves.mkdir()
    _curve(curves / "a.csv")
    _curve(curves / "b.csv")
    client = TestClient(create_app(tmp_path / "desk", enable_auto_start=False))
    client.post("/api/continuous/start", json={"local_dir": str(curves), "nperiods": 64})
    _wait(client, lambda s: s["status"] == "stopped")
    first = client.get("/api/continuous/events", params={"limit": 2}).json()
    assert len(first["lines"]) == 2
    rest = client.get("/api/continuous/events", params={"after_id": first["lines"][-1]["id"]}).json()
    assert rest["lines"] and rest["lines"][0]["id"] > first["lines"][-1]["id"]


def test_failed_work_item_can_be_retried(tmp_path):
    from tool.server import create_app

    client = TestClient(create_app(tmp_path, enable_auto_start=False))
    store = client.app.state.desk.store
    store.enqueue("p", [{"kind": "queue", "target": "1", "sector": 3}], max_attempts=1)
    item = store.claim("w", plan="p")
    store.fail(item["id"], "MAST timeout")
    failed = client.get("/api/work", params={"status": "failed"}).json()
    assert failed["items"][0]["last_error"] == "MAST timeout"
    assert client.post(f"/api/work/{item['id']}/retry").status_code == 200
    assert store.counts("p")["pending"] == 1
