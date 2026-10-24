"""Bearer token, path jail and opt-in auto-start."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tool.__main__ import desk_options, is_loopback
from tool.server import create_app


def test_token_required_on_api_but_not_health(tmp_path):
    client = TestClient(create_app(tmp_path / "desk", enable_auto_start=False, token="s3cret"))
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health").json()["auth_required"] is True
    assert client.get("/api/candidates").status_code == 401
    assert client.get("/api/candidates", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/candidates", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert client.get("/api/candidates?token=s3cret").status_code == 200
    assert client.post("/api/continuous/stop").status_code == 401


def test_no_token_means_open_local_desk(tmp_path):
    client = TestClient(create_app(tmp_path / "desk", enable_auto_start=False))
    assert client.get("/api/candidates").status_code == 200
    assert client.get("/api/health").json()["auth_required"] is False


def test_path_jail_blocks_reads_outside_allowed_dirs(tmp_path):
    secret = tmp_path / "outside.csv"
    secret.write_text("tic_id\n1\n")
    inside_dir = tmp_path / "desk"
    client = TestClient(create_app(inside_dir, enable_auto_start=False, allowed_dirs=[]))
    blocked = client.post("/api/queue/rank", json={"path": str(secret)})
    assert blocked.status_code == 400
    assert "outside" in blocked.json()["detail"]
    traversal = client.post("/api/queue/rank", json={"path": "../outside.csv"})
    assert traversal.status_code == 400
    assert client.post("/api/continuous/start", json={"local_dir": str(tmp_path)}).status_code == 400


def test_allowed_dir_is_readable(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "tics.csv").write_text("tic_id,tmag\n123,9.5\n")
    client = TestClient(create_app(tmp_path / "desk", enable_auto_start=False, allowed_dirs=[str(shared)]))
    response = client.post("/api/queue/rank", json={"path": str(shared / "tics.csv")})
    assert response.status_code != 400 or "outside" not in response.json().get("detail", "")


def test_exposed_bind_requires_token_and_jails_paths():
    assert is_loopback("127.0.0.1") and is_loopback("localhost") and is_loopback("::1")
    assert not is_loopback("0.0.0.0")
    with pytest.raises(SystemExit):
        desk_options("0.0.0.0", {})
    exposed = desk_options("0.0.0.0", {"DESK_TOKEN": "t"})
    assert exposed["token"] == "t" and exposed["allowed_dirs"] == []
    local = desk_options("127.0.0.1", {})
    assert local["allowed_dirs"] is None and local["token"] is None
    assert desk_options("0.0.0.0", {}, insecure=True)["token"] is None


def test_auto_start_is_opt_in():
    assert desk_options("127.0.0.1", {})["enable_auto_start"] is False
    assert desk_options("127.0.0.1", {"AUTO_START": "1"})["enable_auto_start"] is True
    assert desk_options("127.0.0.1", {"AUTO_START": "1"}, auto_start=False)["enable_auto_start"] is False
    assert desk_options("127.0.0.1", {}, auto_start=True)["enable_auto_start"] is True
