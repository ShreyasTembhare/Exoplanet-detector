"""
SQLite state store for the desk and the headless worker.

One file (``state.db``, WAL mode) holds everything needed to resume after a
crash, a redeploy, or a move to another machine:

* ``work_items``  - what to search, with leases, attempts and last error
* ``kv``          - small JSON control documents (continuous loop state)
* ``events``      - the live log, paginated by id
* ``runs``        - one-off desk runs, so their status survives a restart
* ``labels``      - operator labels for active learning

A worker claims an item with a lease. A crash leaves the lease to expire and
the item is claimed again; success marks it done; an error puts it back to
pending until ``max_attempts`` is reached.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS work_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan TEXT NOT NULL,
    kind TEXT NOT NULL,
    target TEXT NOT NULL,
    sector INTEGER NOT NULL DEFAULT 0,
    seq INTEGER NOT NULL DEFAULT 0,
    priority REAL NOT NULL DEFAULT 0,
    payload TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    lease_owner TEXT,
    lease_expires REAL,
    last_error TEXT,
    result TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE (plan, kind, target, sector)
);
CREATE INDEX IF NOT EXISTS work_items_claim ON work_items (plan, status, priority DESC, seq, id);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS labels (
    candidate_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL
);
"""

TERMINAL = ("done", "failed")


def default_owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"


class StateStore:
    def __init__(self, path: os.PathLike | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    # -- connection --------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        yield self._conn()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            conn = self._conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def backup_to(self, dest: os.PathLike | str) -> Path:
        """Consistent snapshot of the live database (safe while writers run)."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(str(dest))
        try:
            self._conn().backup(target)
        finally:
            target.close()
        return dest

    # -- key/value ---------------------------------------------------------

    def get_json(self, key: str, default: Any = None) -> Any:
        row = self._conn().execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except ValueError:
            return default

    def set_json(self, key: str, value: Any) -> None:
        text = json.dumps(value, default=str)
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO kv (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key, text, time.time()),
            )

    # -- events ------------------------------------------------------------

    def append_event(self, payload: dict) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO events (ts, payload) VALUES (?, ?)",
                (str(payload.get("ts", "")), json.dumps(payload, default=str)),
            )
            return int(cur.lastrowid)

    def events(self, after_id: int = 0, limit: int = 500, tail: bool = False,
               before_id: Optional[int] = None) -> List[dict]:
        conn = self._conn()
        if before_id is not None:
            rows = conn.execute(
                "SELECT id, payload FROM (SELECT id, payload FROM events WHERE id > ? AND id < ? "
                "ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                (int(after_id), int(before_id), int(limit)),
            ).fetchall()
        elif tail:
            rows = conn.execute(
                "SELECT id, payload FROM (SELECT id, payload FROM events WHERE id > ? "
                "ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                (int(after_id), int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, payload FROM events WHERE id > ? ORDER BY id ASC LIMIT ?",
                (int(after_id), int(limit)),
            ).fetchall()
        out = []
        for row in rows:
            try:
                item = json.loads(row["payload"])
            except ValueError:
                continue
            item["id"] = int(row["id"])
            out.append(item)
        return out

    def last_event_id(self) -> int:
        row = self._conn().execute("SELECT COALESCE(MAX(id), 0) AS m FROM events").fetchone()
        return int(row["m"])

    def event_count(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"])

    # -- work items --------------------------------------------------------

    def enqueue(self, plan: str, items: Iterable[dict], max_attempts: int = 3) -> int:
        """Insert items idempotently. Existing (plan, kind, target, sector) rows are kept."""
        now = time.time()
        added = 0
        with self.transaction() as conn:
            base = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) AS m FROM work_items WHERE plan = ?", (plan,),
            ).fetchone()["m"]
            for offset, item in enumerate(items):
                cur = conn.execute(
                    "INSERT OR IGNORE INTO work_items "
                    "(plan, kind, target, sector, seq, priority, payload, max_attempts, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        # NULLs are distinct in SQLite UNIQUE constraints; 0 means "no sector".
                        plan, str(item.get("kind", "item")), str(item.get("target", "")),
                        int(item.get("sector") or 0), int(base) + 1 + offset, float(item.get("priority", 0.0)),
                        json.dumps(item.get("payload") or {}, default=str),
                        int(item.get("max_attempts", max_attempts)), now, now,
                    ),
                )
                added += cur.rowcount
        return added

    def claim(self, owner: str, plan: Optional[str] = None, lease_seconds: float = 3600.0) -> Optional[dict]:
        """Lease the next pending (or lease-expired) item, highest priority first."""
        now = time.time()
        with self.transaction() as conn:
            params: list = [now]
            where = "(status = 'pending' OR (status = 'leased' AND lease_expires < ?))"
            if plan is not None:
                where += " AND plan = ?"
                params.append(plan)
            row = conn.execute(
                f"SELECT * FROM work_items WHERE {where} ORDER BY priority DESC, seq ASC, id ASC LIMIT 1",
                params,
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "UPDATE work_items SET status = 'leased', lease_owner = ?, lease_expires = ?, "
                "attempts = attempts + 1, updated_at = ? WHERE id = ?",
                (owner, now + float(lease_seconds), now, row["id"]),
            )
            claimed = self._row(row)
            claimed.update(status="leased", lease_owner=owner, attempts=claimed["attempts"] + 1)
            return claimed

    def peek(self, plan: str) -> Optional[dict]:
        now = time.time()
        row = self._conn().execute(
            "SELECT * FROM work_items WHERE plan = ? AND "
            "(status = 'pending' OR (status = 'leased' AND lease_expires < ?)) "
            "ORDER BY priority DESC, seq ASC, id ASC LIMIT 1",
            (plan, now),
        ).fetchone()
        return self._row(row) if row else None

    def heartbeat(self, item_id: int, owner: str, lease_seconds: float = 3600.0) -> bool:
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE work_items SET lease_expires = ?, updated_at = ? "
                "WHERE id = ? AND lease_owner = ? AND status = 'leased'",
                (time.time() + float(lease_seconds), time.time(), int(item_id), owner),
            )
            return cur.rowcount == 1

    def complete(self, item_id: int, result: Optional[dict] = None) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE work_items SET status = 'done', lease_owner = NULL, lease_expires = NULL, "
                "result = ?, last_error = NULL, updated_at = ? WHERE id = ?",
                (json.dumps(result or {}, default=str), time.time(), int(item_id)),
            )

    def fail(self, item_id: int, error: str, retry: bool = True) -> str:
        """Record an error. Returns the new status (``pending`` to retry, or ``failed``).

        A retried item moves to the back of its plan so one bad target cannot
        block the rest of the queue.
        """
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT plan, attempts, max_attempts FROM work_items WHERE id = ?", (int(item_id),),
            ).fetchone()
            if row is None:
                return "missing"
            status = "pending" if retry and row["attempts"] < row["max_attempts"] else "failed"
            seq = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS s FROM work_items WHERE plan = ?", (row["plan"],),
            ).fetchone()["s"]
            conn.execute(
                "UPDATE work_items SET status = ?, lease_owner = NULL, lease_expires = NULL, "
                "last_error = ?, seq = ?, updated_at = ? WHERE id = ?",
                (status, str(error)[:2000], int(seq), time.time(), int(item_id)),
            )
            return status

    def release(self, item_id: int, refund_attempt: bool = True) -> None:
        """Hand a leased item back without counting it as a failure."""
        with self.transaction() as conn:
            conn.execute(
                "UPDATE work_items SET status = 'pending', lease_owner = NULL, lease_expires = NULL, "
                f"attempts = {'MAX(attempts - 1, 0)' if refund_attempt else 'attempts'}, updated_at = ? "
                "WHERE id = ? AND status = 'leased'",
                (time.time(), int(item_id)),
            )

    def release_owner(self, owner: str) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE work_items SET status = 'pending', lease_owner = NULL, lease_expires = NULL, "
                "attempts = MAX(attempts - 1, 0), updated_at = ? WHERE lease_owner = ? AND status = 'leased'",
                (time.time(), owner),
            )
            return cur.rowcount

    def release_stale(self, keep_owner: Optional[str] = None) -> int:
        """At startup: leases held by a previous process can be reclaimed now."""
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE work_items SET status = 'pending', lease_owner = NULL, lease_expires = NULL, "
                "updated_at = ? WHERE status = 'leased' AND (lease_owner IS NULL OR lease_owner != ?)",
                (time.time(), keep_owner or ""),
            )
            return cur.rowcount

    def retry_item(self, item_id: int) -> bool:
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE work_items SET status = 'pending', attempts = 0, last_error = NULL, updated_at = ? "
                "WHERE id = ? AND status IN ('failed', 'done')",
                (time.time(), int(item_id)),
            )
            return cur.rowcount == 1

    def reset_plan(self, plan: str) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE work_items SET status = 'pending', attempts = 0, last_error = NULL, result = NULL, "
                "lease_owner = NULL, lease_expires = NULL, updated_at = ? WHERE plan = ?",
                (time.time(), plan),
            )
            return cur.rowcount

    def items(self, plan: Optional[str] = None, status: Optional[str] = None,
              limit: int = 200, offset: int = 0) -> List[dict]:
        clauses, params = [], []
        if plan is not None:
            clauses.append("plan = ?")
            params.append(plan)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn().execute(
            f"SELECT * FROM work_items {where} ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
            params + [int(limit), int(offset)],
        ).fetchall()
        return [self._row(row) for row in rows]

    def counts(self, plan: Optional[str] = None) -> dict:
        params: list = []
        where = ""
        if plan is not None:
            where = "WHERE plan = ?"
            params.append(plan)
        rows = self._conn().execute(
            f"SELECT status, COUNT(*) AS n FROM work_items {where} GROUP BY status", params,
        ).fetchall()
        out = {"pending": 0, "leased": 0, "done": 0, "failed": 0}
        for row in rows:
            out[row["status"]] = int(row["n"])
        out["total"] = sum(out[k] for k in ("pending", "leased", "done", "failed"))
        return out

    def get_item(self, item_id: int) -> Optional[dict]:
        row = self._conn().execute("SELECT * FROM work_items WHERE id = ?", (int(item_id),)).fetchone()
        return self._row(row) if row else None

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        item = dict(row)
        for key in ("payload", "result"):
            raw = item.get(key)
            if isinstance(raw, str):
                try:
                    item[key] = json.loads(raw)
                except ValueError:
                    item[key] = {}
        return item

    @contextmanager
    def keep_lease(self, item: dict, owner: str, lease_seconds: float = 3600.0,
                   every: float = 60.0) -> Iterator[None]:
        """Extend a lease in the background while a long search runs."""
        stop = threading.Event()

        def beat():
            while not stop.wait(every):
                try:
                    self.heartbeat(item["id"], owner, lease_seconds)
                except Exception:
                    pass

        thread = threading.Thread(target=beat, name=f"lease-{item['id']}", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()

    # -- runs --------------------------------------------------------------

    def save_run(self, run: dict) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO runs (id, payload, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload = excluded.payload, updated_at = excluded.updated_at",
                (str(run["id"]), json.dumps(run, default=str), time.time()),
            )

    def latest_run(self) -> Optional[dict]:
        row = self._conn().execute("SELECT payload FROM runs ORDER BY updated_at DESC LIMIT 1").fetchone()
        return json.loads(row["payload"]) if row else None

    # -- labels ------------------------------------------------------------

    def set_label(self, candidate_id: str, label: str, note: str = "", payload: Optional[dict] = None) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO labels (candidate_id, label, note, payload, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(candidate_id) DO UPDATE SET label = excluded.label, note = excluded.note, "
                "payload = excluded.payload, updated_at = excluded.updated_at",
                (candidate_id, label, note, json.dumps(payload or {}, default=str), time.time()),
            )

    def labels(self) -> dict:
        rows = self._conn().execute("SELECT candidate_id, label, note, updated_at FROM labels").fetchall()
        return {row["candidate_id"]: dict(row) for row in rows}
