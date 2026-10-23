"""Server-side loop that keeps searching after the browser closes.

Every target is a row in the ``work_items`` table of :mod:`tool.state`. The
loop leases one item at a time, so a crash, redeploy or ``kill -9`` resumes
at the same target: the lease expires (or is released at startup) and the
item is claimed again. Finished items stay finished; starting the same plan
again continues where it stopped instead of replaying from the beginning.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from pipeline.atomic import atomic_write_json
from services import HuntConfig, mast_requirement, run_hunt, run_local_lightcurve
from tool.state import default_owner

STATE_KEY = "continuous"
SNAPSHOT_LINES = 500
LEASE_SECONDS = 3600.0


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


def _parse_iso(value) -> float | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def plan_id(mode: str, *, local_dir: str | None = None, queue_ids: list | None = None) -> str:
    if mode == "local":
        return f"local:{local_dir}"
    if mode == "queue":
        digest = hashlib.sha1(",".join(queue_ids or []).encode()).hexdigest()[:12]
        return f"queue:{digest}"
    return "sectors"


class ContinuousSearch:
    def __init__(self, desk, *, enable_auto_start: bool = True):
        self.desk = desk
        self.store = desk.store
        self.owner = default_owner()
        self.path = desk.root / "continuous.json"
        self.log_path = desk.root / "continuous.log"
        self.progress_path = desk.root / "continuous_progress.json"
        self.hunt_logs = desk.root / "hunt_logs"
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()
        self.state = self._load()
        self._write_progress_unlocked()
        if self.state.get("status") == "running":
            self._ensure_thread()
        elif enable_auto_start:
            self._maybe_auto_start()

    def _idle(self) -> dict:
        return {
            "status": "idle",
            "mode": None,
            "plan": None,
            "cursor": 0,
            "sector": 1,
            "local_dir": None,
            "queue_ids": [],
            "message": "Idle.",
            "config": {},
            "started_at": None,
            "updated_at": None,
            "auto_start": True,
            "pass_complete": False,
            "stars_done": 0,
            "candidates_found": 0,
            "errors": 0,
            "current_target": "",
            "last_sde": None,
            "last_log_line": "",
        }

    def _load(self) -> dict:
        base = self._idle()
        data = self.store.get_json(STATE_KEY)
        if not isinstance(data, dict) and self.path.is_file():
            # One-time import of the pre-SQLite control file.
            try:
                data = json.loads(self.path.read_text())
            except (OSError, ValueError):
                data = None
        if isinstance(data, dict):
            base.update(data)
            if "auto_start" not in data:
                base["auto_start"] = True
        return base

    def _persist(self):
        self.state["updated_at"] = _now()
        self.store.set_json(STATE_KEY, _json_ready(self.state))
        self._write_progress_unlocked()

    def _write_progress_unlocked(self):
        payload = self._progress_payload()
        atomic_write_json(self.progress_path, _json_ready(payload))
        return payload

    def _progress_payload(self) -> dict:
        started = _parse_iso(self.state.get("started_at"))
        elapsed = max(0.0, datetime.now(timezone.utc).timestamp() - started) if started else 0.0
        stars = int(self.state.get("stars_done") or 0)
        stars_per_hour = (stars / elapsed * 3600.0) if elapsed > 1 else 0.0
        plan = self.state.get("plan")
        counts = self.store.counts(plan) if plan else {"pending": 0, "leased": 0, "done": 0, "failed": 0, "total": 0}
        remaining = counts["pending"] + counts["leased"]
        eta = None
        if stars_per_hour > 0 and self.state.get("mode") in {"local", "queue"}:
            eta = round(remaining / stars_per_hour * 3600.0, 1)
        return {
            "status": self.state.get("status") or "idle",
            "mode": self.state.get("mode"),
            "plan": plan,
            "cursor": counts["done"] + counts["failed"],
            "sector": self.state.get("sector"),
            "current_target": self.state.get("current_target") or "",
            "stars_done": stars,
            "candidates_found": int(self.state.get("candidates_found") or 0),
            "errors": int(self.state.get("errors") or 0),
            "last_sde": self.state.get("last_sde"),
            "last_log_line": self.state.get("last_log_line") or "",
            "message": self.state.get("message") or "",
            "auto_start": bool(self.state.get("auto_start", True)),
            "started_at": self.state.get("started_at"),
            "updated_at": self.state.get("updated_at") or _now(),
            "elapsed_seconds": round(elapsed, 1),
            "stars_per_hour": round(stars_per_hour, 2),
            "eta_seconds": eta,
            "work": counts,
            "last_event_id": self.store.last_event_id(),
        }

    def read_lines(self, after_id: int = 0, limit: int = SNAPSHOT_LINES) -> list:
        if after_id:
            return self.store.events(after_id=after_id, limit=limit)
        return self.store.events(limit=limit, tail=True)

    def progress(self) -> dict:
        with self.lock:
            return _json_ready(self._write_progress_unlocked())

    def snapshot(self, after_id: int = 0, limit: int = SNAPSHOT_LINES) -> dict:
        with self.lock:
            state = copy.deepcopy(self.state)
            progress = self._write_progress_unlocked()
        state["lines"] = self.read_lines(after_id=after_id, limit=limit)
        state["progress"] = _json_ready(progress)
        state["cursor"] = progress["cursor"]
        state["total_lines"] = self.store.event_count()
        return state

    def is_running(self) -> bool:
        with self.lock:
            return self.state.get("status") == "running"

    def _maybe_auto_start(self):
        with self.lock:
            if not self.state.get("auto_start", True):
                return
            status = self.state.get("status")
            # Resume an interrupted running loop; stay paused if the operator paused.
            if status in {"running", "paused"}:
                return
            mode = self.state.get("mode")
            resume_plan = mode in {"local", "queue"} and not self.state.get("pass_complete")
            config = dict(self.state.get("config") or {})
            sector = int(self.state.get("sector") or 1)
        try:
            if resume_plan:
                if mode == "local":
                    config["local_dir"] = self.state.get("local_dir")
                self.start(config)
            else:
                self.start({
                    "mode": "sectors",
                    "sector": sector,
                    "limit": int(config.get("limit") or 20),
                    "period_min": float(config.get("period_min") or 0.5),
                    "period_max": float(config.get("period_max") or 15.0),
                    "nperiods": int(config.get("nperiods") or 400),
                    "sde_threshold": float(config.get("sde_threshold") or 7.0),
                })
        except Exception as exc:
            with self.lock:
                self.state["message"] = f"Auto-start deferred: {exc}"
                self._persist()

    def _plan_differs(self, config: dict) -> bool:
        try:
            plan = self._plan(config)
        except ValueError:
            return True
        return plan["plan"] != self.state.get("plan")

    def start(self, config: dict) -> dict:
        with self.desk.lock:
            if self.desk.run and self.desk.run.get("status") == "running":
                raise RuntimeError("A search is already running.")
            with self.lock:
                status = self.state.get("status")
                if status == "running":
                    raise RuntimeError("Keep searching is already running.")
                plan = self._plan(config)
                same_plan = plan["plan"] == self.state.get("plan")
                if config.get("rescan"):
                    self.store.reset_plan(plan["plan"])
                self.store.enqueue(plan["plan"], plan["items"])
                if same_plan and status in {"paused", "stopped", "idle"} and not config.get("rescan"):
                    self.state.update({
                        "status": "running",
                        "auto_start": True,
                        "pass_complete": False,
                        "config": self._config(config, keep=self.state.get("config")),
                        "message": "Resumed. The loop continues from the next unfinished target.",
                    })
                    if not self.state.get("started_at"):
                        self.state["started_at"] = _now()
                    self._append_unlocked({"phase": "resume", "message": self.state["message"]})
                else:
                    sector = int(config.get("sector") or self.state.get("sector") or 1)
                    self.state = {
                        **self._idle(),
                        "status": "running",
                        "mode": plan["mode"],
                        "plan": plan["plan"],
                        "sector": sector,
                        "local_dir": plan.get("local_dir"),
                        "queue_ids": plan.get("queue_ids") or [],
                        "message": plan["message"],
                        "config": self._config(config),
                        "started_at": _now(),
                        "updated_at": _now(),
                        "auto_start": True,
                        "last_log_line": plan["message"],
                    }
                self._persist()
                self._ensure_thread()
                state = copy.deepcopy(self.state)
        state["lines"] = self.read_lines()
        state["progress"] = self.progress()
        return state

    @staticmethod
    def _config(config: dict, keep: dict | None = None) -> dict:
        keep = keep or {}

        def pick(name, default, cast):
            value = config.get(name)
            if value in (None, ""):
                value = keep.get(name, default)
            return cast(value)

        return {
            "limit": pick("limit", 20, int),
            "period_min": pick("period_min", 0.5, float),
            "period_max": pick("period_max", 15.0, float),
            "nperiods": pick("nperiods", 400, int),
            "sde_threshold": pick("sde_threshold", 7.0, float),
            "strategy_profile": pick("strategy_profile", "balanced", str),
            "end_sector": pick("end_sector", None, lambda v: None if v in (None, "") else int(v)),
        }

    def pause(self) -> dict:
        with self.lock:
            if self.state.get("status") != "running":
                raise RuntimeError("Keep searching is not running.")
            self.state["status"] = "paused"
            self.state["message"] = "Paused. No new target will start."
            self._append_unlocked({
                "sector": self.state.get("sector"),
                "phase": "pause",
                "message": "Paused. No new target will start.",
            })
            self._persist()
            state = copy.deepcopy(self.state)
        state["lines"] = self.read_lines()
        state["progress"] = self.progress()
        return state

    def stop(self) -> dict:
        with self.lock:
            if self.state.get("status") not in {"running", "paused"}:
                raise RuntimeError("Keep searching is not running.")
            self.state["status"] = "stopped"
            self.state["auto_start"] = False
            self.state["message"] = "Stopped. Auto-start is off until Start or Resume."
            self._append_unlocked({
                "sector": self.state.get("sector"),
                "phase": "stop",
                "message": "Stopped. Auto-start is off until Start or Resume.",
            })
            self._persist()
            state = copy.deepcopy(self.state)
        state["lines"] = self.read_lines()
        state["progress"] = self.progress()
        return state

    def shutdown(self, timeout: float | None = None):
        """Stop claiming new items; the in-flight item finishes and is recorded."""
        self.shutdown_event.set()
        thread = self.thread
        if thread and thread.is_alive() and timeout:
            thread.join(timeout)
        if thread is None or not thread.is_alive():
            self.store.release_owner(self.owner)

    def _plan(self, config: dict) -> dict:
        local_dir = str(config.get("local_dir") or "").strip()
        if local_dir and config.get("mode") != "sectors":
            directory = self.desk.resolve_user_path(local_dir, must_be_dir=True)
            files = _csv_files(directory)
            if not files:
                raise ValueError("Local curve directory has no time, flux CSV files.")
            resolved = str(directory.resolve())
            return {
                "mode": "local",
                "plan": plan_id("local", local_dir=resolved),
                "local_dir": resolved,
                "message": f"Walking {len(files)} local curves.",
                "items": [
                    {"kind": "local", "target": path.name, "sector": None, "payload": {"path": str(path)}}
                    for path in files
                ],
            }
        ids = _queue_ids(self.desk.queue) if config.get("mode") != "sectors" else []
        if ids:
            sector = int(config.get("sector") or 1)
            return {
                "mode": "queue",
                "plan": plan_id("queue", queue_ids=ids),
                "queue_ids": ids,
                "message": f"Walking {len(ids)} ranked targets.",
                "items": [
                    {"kind": "queue", "target": tic, "sector": sector, "priority": float(len(ids) - i)}
                    for i, tic in enumerate(ids)
                ],
            }
        sector = int(config.get("sector") or 1)
        return {
            "mode": "sectors",
            "plan": plan_id("sectors"),
            "message": f"Walking sectors from {sector}.",
            "items": [{"kind": "sector", "target": f"sector {sector}", "sector": sector}],
        }

    def _ensure_thread(self):
        if self.thread and self.thread.is_alive():
            return
        self.shutdown_event.clear()
        self.thread = threading.Thread(target=self._loop, name="continuous-search")
        self.thread.start()

    def _next_item(self) -> dict | None:
        plan = self.state.get("plan")
        item = self.store.claim(self.owner, plan=plan, lease_seconds=LEASE_SECONDS)
        if item is None and self.state.get("mode") == "sectors":
            done = [row.get("sector") or 0 for row in self.store.items(plan=plan, limit=1000)]
            following = max([int(self.state.get("sector") or 1) - 1] + done) + 1
            end_sector = (self.state.get("config") or {}).get("end_sector")
            if end_sector and following > int(end_sector):
                return None
            self.store.enqueue(plan, [{"kind": "sector", "target": f"sector {following}", "sector": following}])
            item = self.store.claim(self.owner, plan=plan, lease_seconds=LEASE_SECONDS)
        return item

    def _loop(self):
        try:
            while not self.shutdown_event.is_set():
                with self.lock:
                    if self.state.get("status") != "running":
                        return
                    item = self._next_item()
                    if item is None:
                        self.state["status"] = "stopped"
                        self.state["pass_complete"] = True
                        self.state["message"] = "Stopped. Reached the end of this pass."
                        self._append_unlocked({
                            "sector": self.state.get("sector") if self.state.get("mode") == "sectors" else "local",
                            "phase": "end",
                            "message": "Reached the end of this pass.",
                        })
                        self._persist()
                        return
                    if item.get("kind") == "sector":
                        self.state["sector"] = int(item.get("sector") or 1)
                        self._persist()
                self._run_item(item)
        finally:
            with self.lock:
                if self.thread is threading.current_thread():
                    self.thread = None
                restart = self.state.get("status") == "running" and not self.shutdown_event.is_set()
            if self.shutdown_event.is_set():
                self.store.release_owner(self.owner)
            elif restart:
                self._ensure_thread()

    def _run_item(self, item: dict):
        try:
            with self.store.keep_lease(item, self.owner, LEASE_SECONDS):
                outcome = self._search_item(item)
        except Exception as exc:
            status = self.store.fail(item["id"], str(exc), retry=True)
            retry_note = " Will retry." if status == "pending" else " Giving up after repeated errors."
            self._append(
                target=item.get("target") or "",
                sector=item.get("sector"),
                error=str(exc),
                phase="result",
                message=f"{item.get('target') or 'search'}: {exc}.{retry_note}",
            )
            return
        outcome = outcome or {}
        if outcome.get("release"):
            self.store.release(item["id"])
        elif outcome.get("failed"):
            self.store.fail(item["id"], outcome.get("error") or "failed", retry=False)
        else:
            self.store.complete(item["id"], outcome.get("result"))

    def _search_item(self, item: dict) -> dict:
        kind = item.get("kind")
        if kind == "local":
            return self._search_local(item)
        if not mast_requirement()["available"]:
            return self._log_mast_block(item)
        return self._search_hunt(item)

    def _log_mast_block(self, item: dict) -> dict:
        mast = mast_requirement()
        sector = item.get("sector")
        target = item.get("target") or ""
        self._append(
            target=target,
            sector=sector,
            error=mast["message"],
            phase="result",
            message=f"{target} sector {sector} SDE — not a candidate. {mast['message']}",
        )
        if item.get("kind") == "sector":
            with self.lock:
                if self.state.get("status") == "running":
                    self.state["status"] = "paused"
                    self.state["message"] = "Paused. " + mast["message"]
                    self._persist()
            return {"release": True}
        return {"failed": True, "error": mast["message"]}

    def _search_local(self, item: dict) -> dict:
        name = item["target"]
        self._append(target=name, sector="local", phase="start", message=f"Searching {name}.")
        path = Path((item.get("payload") or {}).get("path") or "")
        if path.stat().st_size > 20_000_000:
            raise ValueError(f"{name} is larger than 20 MB.")
        from tool.desk import read_time_flux

        time_arr, flux = read_time_flux(path.read_text())
        cfg = self.state.get("config") or {}
        result = run_local_lightcurve(
            time_arr,
            flux,
            path.stem,
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
            phase="result",
            message=f"{name} sector local SDE {_sde_text(sde)} {verdict}. {detail}".strip(),
        )
        return {"result": {"saved": saved, "sde": _finite(sde)}}

    def _search_hunt(self, item: dict) -> dict:
        target = str(item.get("target") or "")
        sector = int(item.get("sector") or 1)
        if item.get("kind") == "sector":
            start_message = f"Searching sector {sector}."
        else:
            start_message = f"Searching {target} sector {sector}."
        self._append(target=target, sector=sector, phase="start", message=start_message)
        cfg = self.state.get("config") or {}
        self.hunt_logs.mkdir(parents=True, exist_ok=True)
        tic_list = None
        limit = int(cfg.get("limit") or 20)
        if item.get("kind") == "queue":
            # One log per (sector, plan) so star-level resume survives restarts.
            log_file = self.hunt_logs / f"queue_s{sector:03d}.csv"
            tic_path = self.hunt_logs / f"tic_{target}.csv"
            tic_path.write_text(f"tic_id\n{target}\n")
            tic_list = str(tic_path)
            limit = 1
        else:
            log_file = self.hunt_logs / f"sector_{sector:03d}.csv"
        seen_rows = len(_read_hunt_log(log_file))
        result = run_hunt(HuntConfig(
            sector=sector,
            limit=limit,
            sde_threshold=float(cfg.get("sde_threshold") or 7.0),
            period_min=float(cfg.get("period_min") or 0.5),
            period_max=float(cfg.get("period_max") or 15.0),
            nperiods=int(cfg.get("nperiods") or 400),
            strategy_profile=str(cfg.get("strategy_profile") or "balanced"),
            checkpoint="models/checkpoints/resnet1d.pt",
            candidate_dir=str(self.desk.candidates),
            log_file=str(log_file),
            tic_list=tic_list,
            with_tpf=True,
            prefilter=True,
            sequential=False,
        ))
        rows = _read_hunt_log(log_file)[seen_rows:]
        summary = {"completed": 0, "candidates": 0, "errors": 0}
        if isinstance(result, dict):
            summary["completed"] = int(result.get("completed") or 0)
            summary["skipped"] = int(result.get("skipped") or 0)
        for row in rows:
            status = str(row.get("STATUS") or "")
            error = str(row.get("ERROR") or "") or None
            if status.upper().startswith("ERROR"):
                error = status
                summary["errors"] += 1
            candidate = status.startswith("CANDIDATE")
            summary["candidates"] += int(candidate)
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
        if not rows:
            self._append(
                target=target,
                sector=sector,
                phase="result",
                message=f"{target} sector {sector} SDE — not a candidate. Hunt finished with no new stars logged.",
            )
        if item.get("kind") == "queue" and rows and all(
            str(r.get("STATUS") or "").upper().startswith("ERROR") for r in rows
        ):
            raise RuntimeError(str(rows[-1].get("STATUS")))
        return {"result": summary}

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
        line["id"] = self.store.append_event(line)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        phase = line.get("phase")
        if line.get("target"):
            self.state["current_target"] = line["target"]
        if line.get("message"):
            self.state["last_log_line"] = line["message"]
        if phase == "result":
            self.state["stars_done"] = int(self.state.get("stars_done") or 0) + 1
            sde = _finite(line.get("sde"))
            if sde is not None:
                self.state["last_sde"] = sde
            if line.get("candidate"):
                self.state["candidates_found"] = int(self.state.get("candidates_found") or 0) + 1
            if line.get("error"):
                self.state["errors"] = int(self.state.get("errors") or 0) + 1
        self._persist()

    def wait_for_event(self, after_id: int, timeout: float = 15.0) -> int:
        """Block until an event newer than ``after_id`` exists (for SSE)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            last = self.store.last_event_id()
            if last > after_id:
                return last
            time.sleep(0.25)
        return self.store.last_event_id()


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
