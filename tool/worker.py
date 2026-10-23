"""
Headless hunt worker: restore state, search for a time budget, checkpoint, exit.

Built for hosts that kill jobs on a timer (GitHub Actions: 6 h) or on
redeploy (Fly, Railway, Kubernetes). It always resumes from the same spot:

1. pull the last checkpoint (``tool.checkpoint``) into ``--data``;
2. resume the continuous loop from ``state.db`` (unfinished plan, next
   sector, retry counters, per-star hunter logs);
3. push a checkpoint every ``--checkpoint-minutes``;
4. on SIGTERM/SIGINT or when the budget is nearly spent, stop claiming new
   work, give the in-flight item ``--drain-minutes`` to finish, release its
   lease if it did not, write the static dashboard and push a final
   checkpoint.

An item interrupted mid-flight is retried next run; the hunter's per-star
log means the stars it already finished are skipped.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from pathlib import Path

log = logging.getLogger("tool.worker")


def _env_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Headless resumable hunt worker")
    parser.add_argument("--data", default=os.environ.get("DESK_DATA", "var/desk"))
    parser.add_argument("--budget-minutes", type=float, default=float(os.environ.get("HUNT_BUDGET_MINUTES", 0) or 0),
                        help="Stop after this many minutes (0 = run until signalled)")
    parser.add_argument("--drain-minutes", type=float, default=float(os.environ.get("HUNT_DRAIN_MINUTES", 20)),
                        help="Time reserved at the end for the in-flight item and the final checkpoint")
    parser.add_argument("--checkpoint-minutes", type=float,
                        default=float(os.environ.get("HUNT_CHECKPOINT_MINUTES", 15)))
    parser.add_argument("--sector", type=int, default=_env_int("HUNT_START_SECTOR", 1))
    parser.add_argument("--end-sector", type=int, default=_env_int("HUNT_END_SECTOR", None))
    parser.add_argument("--limit", type=int, default=_env_int("HUNT_LIMIT", 200), help="Stars per sector item")
    parser.add_argument("--strategy-profile", default=os.environ.get("HUNT_PROFILE", "balanced"))
    parser.add_argument("--dashboard", default=os.environ.get("HUNT_DASHBOARD_DIR", ""),
                        help="Also write the static status page here (default: <data>/dashboard)")
    parser.add_argument("--no-pull", action="store_true", help="Do not restore a checkpoint first")
    parser.add_argument("--no-push", action="store_true", help="Do not upload checkpoints")
    return parser.parse_args(argv)


def resume_config(state: dict, args) -> dict:
    """Continue an unfinished local/queue plan, otherwise walk sectors from the saved one."""
    config = dict(state.get("config") or {})
    config.update({
        "limit": args.limit,
        "strategy_profile": args.strategy_profile,
    })
    if args.end_sector:
        config["end_sector"] = args.end_sector
    mode = state.get("mode")
    if mode in {"local", "queue"} and not state.get("pass_complete"):
        if mode == "local":
            config["local_dir"] = state.get("local_dir")
        return config
    config["mode"] = "sectors"
    config["sector"] = max(int(state.get("sector") or 0), int(args.sector or 1))
    return config


class Worker:
    def __init__(self, args, backend=None):
        from tool import checkpoint

        self.args = args
        self.root = Path(args.data)
        self.stop = threading.Event()
        self.backend = backend if backend is not None else (None if args.no_push and args.no_pull
                                                            else checkpoint.backend_from_env())
        self.checkpoint = checkpoint
        self.desk = None

    def handle_signal(self, signum, _frame):
        log.warning("Signal %s received; draining.", signum)
        self.stop.set()

    def restore(self) -> bool:
        if self.args.no_pull or self.backend is None:
            return False
        try:
            found = self.checkpoint.pull(self.root, self.backend)
        except Exception as exc:
            log.error("Checkpoint restore failed (%s); starting from local state.", exc)
            return False
        log.info("%s %s", "Restored checkpoint from" if found else "No checkpoint yet at", self.backend.describe())
        return found

    def save(self, final: bool = False) -> None:
        if self.desk is None:
            return
        try:
            from tool.dashboard import render

            render(self.desk, self.root / "dashboard")
            if self.args.dashboard:
                render(self.desk, Path(self.args.dashboard))
        except Exception as exc:
            log.warning("Dashboard render failed: %s", exc)
        if self.args.no_push or self.backend is None:
            return
        try:
            size = self.checkpoint.push(self.root, self.backend, store=self.desk.store)
            log.info("%s checkpoint (%s bytes) to %s", "Final" if final else "Periodic", size,
                     self.backend.describe())
        except Exception as exc:
            log.error("Checkpoint push failed: %s", exc)

    def start_loop(self) -> None:
        from tool.desk import Desk

        self.desk = Desk(self.root, enable_auto_start=False)
        cont = self.desk.continuous
        if cont.is_running():
            log.info("Resuming the running plan %s at sector %s.", cont.state.get("plan"), cont.state.get("sector"))
            return
        config = resume_config(cont.state, self.args)
        state = cont.start(config)
        log.info("Started %s: %s", state.get("plan"), state.get("message"))

    def run(self) -> int:
        started = time.monotonic()
        budget = self.args.budget_minutes * 60.0
        drain = self.args.drain_minutes * 60.0
        soft_deadline = started + max(60.0, budget - drain) if budget else None
        hard_deadline = started + max(120.0, budget - 120.0) if budget else None
        interval = max(60.0, self.args.checkpoint_minutes * 60.0)

        self.restore()
        self.start_loop()
        cont = self.desk.continuous
        next_checkpoint = time.monotonic() + interval
        while not self.stop.is_set():
            now = time.monotonic()
            if soft_deadline and now >= soft_deadline:
                log.info("Budget reached; draining.")
                break
            if not cont.is_running() and (cont.thread is None or not cont.thread.is_alive()):
                log.info("Loop finished: %s", cont.state.get("message"))
                break
            if now >= next_checkpoint:
                self.save()
                next_checkpoint = now + interval
            wait = min(5.0, next_checkpoint - now)
            if soft_deadline:
                wait = min(wait, soft_deadline - now)
            self.stop.wait(max(0.5, wait))

        timeout = None
        if hard_deadline:
            timeout = max(5.0, hard_deadline - time.monotonic())
        elif self.args.drain_minutes:
            timeout = drain
        cont.shutdown(timeout=timeout)
        thread = cont.thread
        stuck = bool(thread and thread.is_alive())
        if stuck:
            log.warning("In-flight item did not finish in time; releasing its lease for the next run.")
            self.desk.store.release_owner(cont.owner)
        self.save(final=True)
        return 1 if stuck else 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    worker = Worker(args)
    signal.signal(signal.SIGTERM, worker.handle_signal)
    signal.signal(signal.SIGINT, worker.handle_signal)
    code = worker.run()
    if code:
        # A search thread is still inside a long download/BLS call; state is
        # already checkpointed, so do not wait for it.
        logging.shutdown()
        os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
