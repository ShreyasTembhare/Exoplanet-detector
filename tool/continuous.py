"""Server-side loop that keeps searching after the browser closes."""

from __future__ import annotations

import copy
import csv
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from services import HuntConfig, mast_requirement, run_hunt, run_local_lightcurve


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _csv_files(directory: Path) -> list:
    files = [
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".csv" and not path.name.startswith(".")
    ]
    files.sort(key=lambda path: path.name)
    return files


def _sde_text(sde) -> str:
    number = _finite(sde)
    if number is None:
        return "—"
    return f"{number:.2f}"


class ContinuousSearch:
    def __init__(self, desk):
        self.desk = desk
        self.path = desk.root / "continuous.json"
        self.log_path = desk.root / "continuous.log"
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.state = self._load()
        if self.state.get("status") == "running":
            self._ensure_thread()

    def _idle(self) -> dict:
        return {
            "status": "idle",
            "mode": None,
            "cursor": 0,
            "sector": 1,
            "local_dir": None,
            "queue_ids": [],
            "message": "Idle.",
            "config": {},
            "started_at": None,
            "updated_at": None,
        }

    def _load(self) -> dict:
        if not self.path.is_file():
            return self._idle()
        try:
            data = json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return self._idle()
        if not isinstance(data, dict):
            return self._idle()
        base = self._idle()
        base.update(data)
        return base

    def _persist(self):
        self.state["updated_at"] = _now()
        self.path.write_text(json.dumps(_json_ready(self.state), indent=2))

    def read_lines(self) -> list:
        if not self.log_path.is_file():
            return []
        lines = []
        for raw in self.log_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                lines.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return lines

    def snapshot(self) -> dict:
        with self.lock:
            state = copy.deepcopy(self.state)
        state["lines"] = self.read_lines()
        return state

    def is_running(self) -> bool:
        with self.lock:
            return self.state.get("status") == "running"

    def start(self, config: dict) -> dict:
        with self.desk.lock:
            if self.desk.run and self.desk.run.get("status") == "running":
                raise RuntimeError("A search is already running.")
            with self.lock:
                status = self.state.get("status")
                if status == "running":
                    raise RuntimeError("Keep searching is already running.")
                if status == "paused":
                    self.state["status"] = "running"
                    self.state["message"] = "Resumed. The loop continues from the next target."
                    self._persist()
                    self._ensure_thread()
                    state = copy.deepcopy(self.state)
                else:
                    plan = self._plan(config)
                    self.state = {
                        "status": "running",
                        "mode": plan["mode"],
                        "cursor": 0,
                        "sector": int(config.get("sector") or 1),
                        "local_dir": plan.get("local_dir"),
                        "queue_ids": plan.get("queue_ids") or [],
                        "message": plan["message"],
                        "config": {
                            "limit": int(config.get("limit") or 20),
                            "period_min": float(config.get("period_min") or 0.5),
                            "period_max": float(config.get("period_max") or 15.0),
                            "nperiods": int(config.get("nperiods") or 400),
                            "sde_threshold": float(config.get("sde_threshold") or 7.0),
                        },
                        "started_at": _now(),
                        "updated_at": _now(),
                    }
                    self.log_path.write_text("")
                    self._persist()
                    self._ensure_thread()
                    state = copy.deepcopy(self.state)
        state["lines"] = self.read_lines()
        return state

    def pause(self) -> dict:
        with self.lock:
            if self.state.get("status") != "running":
                raise RuntimeError("Keep searching is not running.")
            self.state["status"] = "paused"
            self.state["message"] = "Paused. No new target will start."
            self._append_unlocked({
                "target": "",
                "sector": self.state.get("sector"),
                "sde": None,
                "candidate": False,
                "error": None,
                "phase": "pause",
                "message": "Paused. No new target will start.",
            })
            self._persist()
            state = copy.deepcopy(self.state)
        state["lines"] = self.read_lines()
        return state

    def stop(self) -> dict:
        with self.lock:
            if self.state.get("status") not in {"running", "paused"}:
                raise RuntimeError("Keep searching is not running.")
            self.state["status"] = "stopped"
            self.state["message"] = "Stopped."
            self._append_unlocked({
                "target": "",
                "sector": self.state.get("sector"),
                "sde": None,
                "candidate": False,
                "error": None,
                "phase": "stop",
                "message": "Stopped.",
            })
            self._persist()
            state = copy.deepcopy(self.state)
        state["lines"] = self.read_lines()
        return state

    def _plan(self, config: dict) -> dict:
        local_dir = str(config.get("local_dir") or "").strip()
        if local_dir:
            directory = Path(local_dir).expanduser()
            if not directory.is_dir():
                raise ValueError(f"Local curve directory not found: {local_dir}")
            files = _csv_files(directory)
            if not files:
                raise ValueError("Local curve directory has no time, flux CSV files.")
            return {
                "mode": "local",
                "local_dir": str(directory),
                "message": f"Walking {len(files)} local curves.",
            }
        ids = _queue_ids(self.desk.queue)
        if ids:
            return {
                "mode": "queue",
                "queue_ids": ids,
                "message": f"Walking {len(ids)} ranked targets.",
            }
        sector = int(config.get("sector") or 1)
        return {"mode": "sectors", "message": f"Walking sectors from {sector}."}

    def _ensure_thread(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._loop, name="continuous-search", daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            while True:
                with self.lock:
                    if self.state.get("status") != "running":
                        return
                    try:
                        item = self._peek_item()
                    except Exception as exc:
                        self.state["status"] = "paused"
                        self.state["message"] = "Paused. " + str(exc)
                        self._append_unlocked({
                            "target": "",
                            "sector": self.state.get("sector"),
                            "sde": None,
                            "candidate": False,
                            "error": str(exc),
                            "phase": "result",
                            "message": str(exc),
                        })
                        self._persist()
                        return
                    if item is None:
                        self.state["status"] = "stopped"
                        self.state["message"] = "Stopped. Reached the end of this pass."
                        self._append_unlocked({
                            "target": "",
                            "sector": self.state.get("sector") if self.state.get("mode") == "sectors" else "local",
                            "sde": None,
                            "candidate": False,
                            "error": None,
                            "phase": "end",
                            "message": "Reached the end of this pass.",
                        })
                        self._persist()
                        return
                try:
                    self._search_item(item)
                except Exception as exc:
                    self._append(
                        target=item.get("target") or "",
                        sector=item.get("sector"),
                        sde=None,
                        candidate=False,
                        error=str(exc),
                        phase="result",
                        message=f"{item.get('target') or 'search'}: {exc}",
                    )
                with self.lock:
                    if item.get("advance", True) and self.state.get("status") in {"running", "paused"}:
                        self._advance(item)
                    if self.state.get("status") != "running":
                        return
        finally:
            with self.lock:
                if self.thread is threading.current_thread():
                    self.thread = None
                if self.state.get("status") == "running":
                    self._ensure_thread()

    def _peek_item(self) -> dict | None:
        mode = self.state.get("mode")
        cursor = int(self.state.get("cursor") or 0)
        sector = int(self.state.get("sector") or 1)
        if mode == "local":
            directory = Path(self.state.get("local_dir") or "")
            if not directory.is_dir():
                raise ValueError(f"Local curve directory not found: {directory}")
            files = _csv_files(directory)
            if cursor >= len(files):
                return None
            path = files[cursor]
            return {
                "kind": "local",
                "path": str(path),
                "target": path.name,
                "tic_id": path.stem,
                "sector": "local",
                "advance": True,
            }
        if mode == "queue":
            ids = list(self.state.get("queue_ids") or [])
            if cursor >= len(ids):
                return None
            tic = ids[cursor]
            if not mast_requirement()["available"]:
                return {
                    "kind": "queue_blocked",
                    "target": tic,
                    "sector": sector,
                    "advance": True,
                }
            return {"kind": "queue", "target": tic, "sector": sector, "advance": True}
        if not mast_requirement()["available"]:
            return {
                "kind": "sector_blocked",
                "target": f"sector {sector}",
                "sector": sector,
                "advance": False,
            }
        return {"kind": "sector", "target": f"sector {sector}", "sector": sector, "advance": True}

    def _advance(self, item: dict):
        if self.state.get("mode") == "sectors" and item.get("kind") == "sector":
            self.state["sector"] = int(self.state.get("sector") or 1) + 1
        else:
            self.state["cursor"] = int(self.state.get("cursor") or 0) + 1
        self._persist()

    def _search_item(self, item: dict):
        kind = item.get("kind")
        if kind == "local":
            self._search_local(item)
            return
        if kind in {"queue_blocked", "sector_blocked"}:
            self._log_mast_block(item)
            return
        self._search_hunt(item)

    def _log_mast_block(self, item: dict):
        mast = mast_requirement()
        sector = item.get("sector")
        target = item.get("target") or ""
        message = (
            f"{target} sector {sector} SDE — not a candidate. {mast['message']}"
        )
        self._append(
            target=target,
            sector=sector,
            sde=None,
            candidate=False,
            error=mast["message"],
            phase="result",
            message=message,
        )
        if item.get("kind") == "sector_blocked":
            with self.lock:
                if self.state.get("status") == "running":
                    self.state["status"] = "paused"
                    self.state["message"] = "Paused. " + mast["message"]
                    self._persist()

    def _search_local(self, item: dict):
        name = item["target"]
        self._append(
            target=name,
            sector="local",
            sde=None,
            candidate=False,
            error=None,
            phase="start",
            message=f"Searching {name}.",
        )
        path = Path(item["path"])
        if path.stat().st_size > 20_000_000:
            raise ValueError(f"{name} is larger than 20 MB.")
        from tool.desk import read_time_flux

        time, flux = read_time_flux(path.read_text())
        cfg = self.state.get("config") or {}
        result = run_local_lightcurve(
            time,
            flux,
            item.get("tic_id") or path.stem,
            str(self.desk.candidates),
            period_min=float(cfg.get("period_min") or 0.5),
            period_max=float(cfg.get("period_max") or 15.0),
            nperiods=int(cfg.get("nperiods") or 400),
            sde_threshold=float(cfg.get("sde_threshold") or 7.0),
        )
        saved = bool(result.get("saved"))
        sde = result.get("sde")
        verdict = "candidate" if saved else "not a candidate"
        detail = result.get("message") or ""
        self._append(
            target=name,
            sector="local",
            sde=sde,
            candidate=saved,
            error=None,
            phase="result",
            message=f"{name} sector local SDE {_sde_text(sde)} {verdict}. {detail}".strip(),
        )

    def _search_hunt(self, item: dict):
        target = str(item.get("target") or "")
        sector = int(item.get("sector") or 1)
        self._append(
            target=target,
            sector=sector,
            sde=None,
            candidate=False,
            error=None,
            phase="start",
            message=f"Searching {target} sector {sector}.",
        )
        cfg = self.state.get("config") or {}
        log_file = self.desk.root / "continuous_hunt.csv"
        if log_file.exists():
            log_file.unlink()
        tic_list = None
        limit = int(cfg.get("limit") or 20)
        if item.get("kind") == "queue":
            tic_path = self.desk.root / "continuous_tic.csv"
            tic_path.write_text(f"tic_id\n{target}\n")
            tic_list = str(tic_path)
            limit = 1
        result = run_hunt(HuntConfig(
            sector=sector,
            limit=limit,
            sde_threshold=float(cfg.get("sde_threshold") or 7.0),
            period_min=float(cfg.get("period_min") or 0.5),
            period_max=float(cfg.get("period_max") or 15.0),
            nperiods=int(cfg.get("nperiods") or 400),
            checkpoint="models/checkpoints/resnet1d.pt",
            candidate_dir=str(self.desk.candidates),
            log_file=str(log_file),
            tic_list=tic_list,
            with_tpf=True,
            prefilter=True,
            sequential=False,
        ))
        rows = _read_hunt_log(log_file)
        if rows:
            for row in rows:
                status = str(row.get("STATUS") or "")
                error = str(row.get("ERROR") or "") or None
                if status.upper().startswith("ERROR"):
                    error = status
                candidate = status.startswith("CANDIDATE")
                sde = _finite(row.get("SDE"))
                tic = str(row.get("TIC_ID") or target)
                verdict = "candidate" if candidate else "not a candidate"
                detail = error or status or "hunt finished"
                self._append(
                    target=tic,
                    sector=sector,
                    sde=sde,
                    candidate=candidate,
                    error=error if not candidate else None,
                    phase="result",
                    message=f"TIC {tic} sector {sector} SDE {_sde_text(sde)} {verdict}. {detail}",
                )
            return
        found = result.get("candidates") if isinstance(result, dict) else None
        if found:
            for cand in found:
                if isinstance(cand, dict):
                    tic = str(cand.get("tic_id") or target)
                    sde = _finite(cand.get("sde"))
                    candidate = str(cand.get("status") or "").startswith("CANDIDATE") or cand.get("sde") is not None
                else:
                    tic = str(cand)
                    sde = None
                    candidate = True
                verdict = "candidate" if candidate else "not a candidate"
                self._append(
                    target=tic,
                    sector=sector,
                    sde=sde,
                    candidate=candidate,
                    error=None,
                    phase="result",
                    message=f"TIC {tic} sector {sector} SDE {_sde_text(sde)} {verdict}.",
                )
            return
        self._append(
            target=target,
            sector=sector,
            sde=None,
            candidate=False,
            error=None,
            phase="result",
            message=f"{target} sector {sector} SDE — not a candidate. Hunt finished with no stars logged.",
        )

    def _append(self, **fields):
        with self.lock:
            self._append_unlocked(fields)

    def _append_unlocked(self, fields: dict):
        line = _json_ready({
            "ts": _now(),
            "target": fields.get("target") or "",
            "sector": fields.get("sector"),
            "sde": fields.get("sde"),
            "candidate": bool(fields.get("candidate")),
            "error": fields.get("error"),
            "phase": fields.get("phase") or "result",
            "message": fields.get("message") or "",
        })
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _queue_ids(queue: dict | None) -> list:
    if not queue:
        return []
    ids = []
    for key in ("primary", "second_pass"):
        for row in queue.get(key) or []:
            tic = str(row.get("tic_id") or "").strip()
            if tic:
                ids.append(tic)
    return ids


def _read_hunt_log(path: Path) -> list:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))
