"""Crash recovery: truncated checkpoints, torn writes, corrupt control files."""

from __future__ import annotations

import json

import pytest

from tool import checkpoint
from tool.desk import Desk
from tool.state import StateStore


def _live_state(root):
    store = StateStore(root / "state.db")
    store.set_json("continuous", {"status": "paused", "sector": 7})
    (root / "candidates").mkdir(parents=True, exist_ok=True)
    (root / "candidates" / "TIC_1_p1.00.json").write_text('{"tic_id": "1"}')
    return store


def test_truncated_checkpoint_leaves_live_state_untouched(tmp_path):
    src = tmp_path / "src"
    store = _live_state(src)
    data = checkpoint.build_archive(src, store=store)

    live = tmp_path / "live"
    live_store = StateStore(live / "state.db")
    live_store.set_json("continuous", {"status": "running", "sector": 42})
    before = (live / "state.db").read_bytes()

    with pytest.raises(Exception):
        checkpoint.restore_archive(data[: len(data) // 2], live)
    assert (live / "state.db").read_bytes() == before
    assert StateStore(live / "state.db").get_json("continuous")["sector"] == 42
    assert not [p for p in live.iterdir() if p.name.startswith(".restore-")]


def test_restore_merges_and_keeps_newer_local_candidates(tmp_path):
    src = tmp_path / "src"
    data = checkpoint.build_archive(src, store=_live_state(src))
    live = tmp_path / "live"
    (live / "candidates").mkdir(parents=True)
    (live / "candidates" / "TIC_2_p2.00.json").write_text('{"tic_id": "2"}')
    checkpoint.restore_archive(data, live)
    assert (live / "candidates" / "TIC_1_p1.00.json").is_file()
    assert (live / "candidates" / "TIC_2_p2.00.json").is_file()
    assert StateStore(live / "state.db").get_json("continuous")["sector"] == 7


def test_torn_candidate_json_is_skipped_not_fatal(tmp_path):
    desk = Desk(tmp_path / "desk", enable_auto_start=False)
    good = {"tic_id": "5", "best_period": 2.0, "sde": 9.0, "vetting": {}}
    (desk.candidates / "TIC_5_p2.00.json").write_text(json.dumps(good))
    (desk.candidates / "TIC_6_p3.00.json").write_text('{"tic_id": "6", "best_peri')
    rows = desk.list_candidates()
    assert [row["tic_id"] for row in rows] == ["5"]


def test_corrupt_control_file_starts_idle(tmp_path):
    root = tmp_path / "desk"
    root.mkdir()
    (root / "continuous.json").write_text('{"status": "runn')
    desk = Desk(root, enable_auto_start=False)
    assert desk.continuous.state.get("status") in {"idle", "stopped"}
    assert isinstance(desk.health(), dict)
