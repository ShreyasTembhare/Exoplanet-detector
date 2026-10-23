"""Checkpoint round trip and the headless worker resuming on another machine."""

from __future__ import annotations

import io
import json
import tarfile
from types import SimpleNamespace

import pytest

from tool import checkpoint
from tool.desk import Desk
from tool.state import StateStore
from tool.worker import Worker, parse_args, resume_config


def _curve(path, samples=48):
    rows = ["time,flux"]
    for i in range(samples):
        rows.append(f"{1700 + i * 0.02:.4f},{1.0 + (i % 3) * 1e-4:.6f}")
    path.write_text("\n".join(rows) + "\n")


def test_checkpoint_round_trip(tmp_path):
    root = tmp_path / "a"
    store = StateStore(root / "state.db")
    store.set_json("continuous", {"status": "paused", "sector": 12})
    store.enqueue("p", [{"kind": "queue", "target": "1", "sector": 12}])
    (root / "candidates").mkdir()
    (root / "candidates" / "TIC1.json").write_text('{"tic_id": "1"}')
    (root / "cache").mkdir()
    (root / "cache" / "big.npz").write_bytes(b"x" * 10)

    backend = checkpoint.DirectoryBackend(str(tmp_path / "bucket"))
    assert checkpoint.push(root, backend, store=store) > 0

    other = tmp_path / "b"
    assert checkpoint.pull(other, backend)
    restored = StateStore(other / "state.db")
    assert restored.get_json("continuous")["sector"] == 12
    assert restored.counts("p")["pending"] == 1
    assert json.loads((other / "candidates" / "TIC1.json").read_text())["tic_id"] == "1"
    assert not (other / "cache").exists()


def test_pull_without_checkpoint_is_a_no_op(tmp_path):
    backend = checkpoint.DirectoryBackend(str(tmp_path / "empty"))
    assert checkpoint.pull(tmp_path / "root", backend) is False


def test_restore_rejects_path_traversal(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"hi"))
    with pytest.raises(ValueError):
        checkpoint.restore_archive(buffer.getvalue(), tmp_path / "root")
    assert not (tmp_path / "escape.txt").exists()


def test_backend_from_env():
    assert checkpoint.backend_from_env({}) is None
    assert isinstance(checkpoint.backend_from_env({"STATE_BUCKET": "file:///tmp/x"}), checkpoint.DirectoryBackend)
    git = checkpoint.backend_from_env({"STATE_GIT_BRANCH": "hunt-state"})
    assert isinstance(git, checkpoint.GitBranchBackend) and git.branch == "hunt-state"
    assert checkpoint.backend_from_env({"STATE_GIT_BRANCH": "off", "GITHUB_ACTIONS": "true"}) is None


def test_resume_config_prefers_unfinished_plan():
    args = SimpleNamespace(limit=50, strategy_profile="fast", end_sector=None, sector=1)
    local = resume_config({"mode": "local", "local_dir": "/d", "pass_complete": False}, args)
    assert local["local_dir"] == "/d" and "mode" not in local
    sectors = resume_config({"mode": "sectors", "sector": 14}, args)
    assert sectors["mode"] == "sectors" and sectors["sector"] == 14 and sectors["limit"] == 50
    args.sector = 20
    assert resume_config({"mode": "sectors", "sector": 14}, args)["sector"] == 20


def test_worker_resumes_interrupted_plan_from_checkpoint(tmp_path):
    curves = tmp_path / "curves"
    curves.mkdir()
    _curve(curves / "a.csv")
    _curve(curves / "b.csv")

    # Machine A got through a.csv, then the job was killed.
    first = Desk(tmp_path / "machine-a", enable_auto_start=False)
    cont = first.continuous
    plan = cont._plan({"local_dir": str(curves)})
    first.store.enqueue(plan["plan"], plan["items"])
    done = first.store.claim("machine-a", plan=plan["plan"])
    assert done["target"] == "a.csv"
    first.store.complete(done["id"], {"result": {}})
    cont.state.update(status="paused", mode="local", plan=plan["plan"], local_dir=plan["local_dir"],
                      config={"nperiods": 64, "limit": 20})
    cont._persist()
    backend = checkpoint.DirectoryBackend(str(tmp_path / "bucket"))
    checkpoint.push(first.root, backend, store=first.store)
    first.shutdown()

    # Machine B restores and finishes only b.csv.
    args = parse_args(["--data", str(tmp_path / "machine-b"), "--budget-minutes", "0", "--drain-minutes", "1"])
    worker = Worker(args, backend=backend)
    assert worker.run() == 0
    store = worker.desk.store
    counts = store.counts(plan["plan"])
    assert counts["done"] == 2 and counts["pending"] == 0
    results = [line["target"] for line in store.events(limit=100) if line.get("phase") == "result"]
    assert results == ["b.csv"]
    assert (tmp_path / "machine-b" / "dashboard" / "index.html").is_file()

    # The final checkpoint carries machine B's progress.
    third = tmp_path / "machine-c"
    assert checkpoint.pull(third, backend)
    assert StateStore(third / "state.db").counts(plan["plan"])["done"] == 2
