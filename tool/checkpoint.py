"""
Checkpoint the desk data directory to durable storage and restore it on boot.

A checkpoint is one ``state.tar.gz`` holding a consistent copy of
``state.db`` (taken with the SQLite backup API while the worker keeps
writing), the candidate evidence, exports, per-sector hunter logs and the
ranked queue. Anything that can be rebuilt (light-curve caches) is left out.

Where it goes is picked from the environment:

* ``STATE_BUCKET=s3://bucket/prefix`` - any S3-compatible store (AWS S3,
  Cloudflare R2, Backblaze B2, MinIO). ``STATE_ENDPOINT_URL`` points boto3 at
  R2/B2; credentials come from the usual ``AWS_*`` variables.
* ``STATE_BUCKET=file:///some/dir`` - a plain directory (tests, NFS, volumes).
* ``STATE_GIT_BRANCH=hunt-state`` (the default inside GitHub Actions) -
  a single-commit orphan branch in the current repository, force-pushed so
  the repository does not grow with every checkpoint.

``python -m tool.checkpoint pull|push --data var/desk`` does the same from a
shell or a CI step.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Optional

ARCHIVE_NAME = "state.tar.gz"
INCLUDE = ("candidates", "exports", "hunt_logs", "queue.json", "continuous.log", "dashboard")
GIT_SIZE_LIMIT = 95 * 1024 * 1024


def build_archive(root: Path, store=None) -> bytes:
    """Return a gzip tarball of ``root``; ``state.db`` is copied consistently."""
    root = Path(root)
    buffer = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp:
        db_copy = Path(tmp) / "state.db"
        db = root / "state.db"
        if store is not None:
            store.backup_to(db_copy)
        elif db.is_file():
            from tool.state import StateStore

            source = StateStore(db)
            try:
                source.backup_to(db_copy)
            finally:
                source.close()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            if db_copy.is_file():
                tar.add(db_copy, arcname="state.db")
            for name in INCLUDE:
                path = root / name
                if path.exists():
                    tar.add(path, arcname=name)
            meta = json.dumps({"created_at": time.time(), "host": os.uname().nodename}).encode()
            info = tarfile.TarInfo("checkpoint.json")
            info.size = len(meta)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(meta))
    return buffer.getvalue()


def _safe_members(tar: tarfile.TarFile, dest: Path):
    dest = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if target != dest and dest not in target.parents:
            raise ValueError(f"Refusing to extract {member.name!r} outside {dest}")
        if member.issym() or member.islnk() or member.isdev():
            continue
        yield member


def restore_archive(data: bytes, root: Path) -> list:
    """Unpack a checkpoint into ``root``; existing files with the same name are replaced."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for stale in ("state.db-wal", "state.db-shm"):
        (root / stale).unlink(missing_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        members = list(_safe_members(tar, root))
        tar.extractall(root, members=members)
    return [m.name for m in members]


class DirectoryBackend:
    def __init__(self, directory: str):
        self.directory = Path(directory)

    def describe(self) -> str:
        return f"file://{self.directory}"

    def get(self) -> Optional[bytes]:
        path = self.directory / ARCHIVE_NAME
        return path.read_bytes() if path.is_file() else None

    def put(self, data: bytes) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.directory / f".{ARCHIVE_NAME}.tmp"
        tmp.write_bytes(data)
        os.replace(tmp, self.directory / ARCHIVE_NAME)


class S3Backend:
    def __init__(self, url: str, endpoint_url: Optional[str] = None):
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on the image
            raise RuntimeError("STATE_BUCKET is an s3:// URL but boto3 is not installed.") from exc
        rest = url[len("s3://"):]
        self.bucket, _, prefix = rest.partition("/")
        self.key = f"{prefix.strip('/')}/{ARCHIVE_NAME}".lstrip("/")
        self.client = boto3.client("s3", endpoint_url=endpoint_url or None)

    def describe(self) -> str:
        return f"s3://{self.bucket}/{self.key}"

    def get(self) -> Optional[bytes]:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.key)
        except self.client.exceptions.NoSuchKey:
            return None
        except Exception as exc:
            if "NoSuchKey" in str(exc) or "404" in str(exc):
                return None
            raise
        return response["Body"].read()

    def put(self, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self.key, Body=data)


class GitBranchBackend:
    """Store the archive as the only file of an orphan branch, force-pushed."""

    def __init__(self, branch: str = "hunt-state", remote: str = "origin", repo: str | Path = "."):
        self.branch = branch
        self.remote = remote
        self.repo = str(repo)

    def describe(self) -> str:
        return f"git {self.remote}/{self.branch}"

    def _git(self, *args, input: bytes | None = None, check: bool = True) -> subprocess.CompletedProcess:
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": os.environ.get("GIT_AUTHOR_NAME", "hunt-worker"),
            "GIT_AUTHOR_EMAIL": os.environ.get("GIT_AUTHOR_EMAIL", "hunt-worker@users.noreply.github.com"),
            "GIT_COMMITTER_NAME": os.environ.get("GIT_COMMITTER_NAME", "hunt-worker"),
            "GIT_COMMITTER_EMAIL": os.environ.get("GIT_COMMITTER_EMAIL", "hunt-worker@users.noreply.github.com"),
        }
        return subprocess.run(["git", "-C", self.repo, *args], input=input, capture_output=True,
                              check=check, env=env)

    def get(self) -> Optional[bytes]:
        ref = f"refs/remotes/{self.remote}/{self.branch}"
        fetched = self._git("fetch", "--depth=1", self.remote, f"+refs/heads/{self.branch}:{ref}", check=False)
        if fetched.returncode != 0:
            return None
        shown = self._git("show", f"{ref}:{ARCHIVE_NAME}", check=False)
        return shown.stdout if shown.returncode == 0 else None

    def put(self, data: bytes) -> None:
        if len(data) > GIT_SIZE_LIMIT:
            raise RuntimeError(
                f"Checkpoint is {len(data) / 1e6:.0f} MB, over the git file limit; set STATE_BUCKET instead."
            )
        blob = self._git("hash-object", "-w", "--stdin", input=data).stdout.decode().strip()
        tree = self._git("mktree", input=f"100644 blob {blob}\t{ARCHIVE_NAME}\n".encode()).stdout.decode().strip()
        message = f"checkpoint {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}"
        commit = self._git("commit-tree", tree, "-m", message).stdout.decode().strip()
        self._git("push", "--force", self.remote, f"{commit}:refs/heads/{self.branch}")


def backend_from_env(env: Optional[dict] = None):
    """Pick a backend from ``STATE_BUCKET`` / ``STATE_GIT_BRANCH``; ``None`` disables checkpoints."""
    env = os.environ if env is None else env
    bucket = (env.get("STATE_BUCKET") or "").strip()
    if bucket.startswith("s3://"):
        return S3Backend(bucket, env.get("STATE_ENDPOINT_URL"))
    if bucket.startswith("file://"):
        return DirectoryBackend(bucket[len("file://"):])
    if bucket:
        return DirectoryBackend(bucket)
    branch = (env.get("STATE_GIT_BRANCH") or "").strip()
    if branch.lower() in {"0", "off", "none"}:
        return None
    in_actions = env.get("GITHUB_ACTIONS") == "true"
    if branch or (in_actions and shutil.which("git") and Path(env.get("STATE_GIT_REPO", ".")).joinpath(".git").exists()):
        return GitBranchBackend(branch or "hunt-state", env.get("STATE_GIT_REMOTE", "origin"),
                                env.get("STATE_GIT_REPO", "."))
    return None


def pull(root: Path, backend=None) -> bool:
    """Restore the latest checkpoint into ``root``. Returns False when there is none yet."""
    backend = backend if backend is not None else backend_from_env()
    if backend is None:
        return False
    data = backend.get()
    if not data:
        return False
    restore_archive(data, Path(root))
    return True


def push(root: Path, backend=None, store=None) -> Optional[int]:
    """Upload a checkpoint of ``root``; returns its size in bytes, or None when disabled."""
    backend = backend if backend is not None else backend_from_env()
    if backend is None:
        return None
    data = build_archive(Path(root), store=store)
    backend.put(data)
    return len(data)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Checkpoint the desk data directory")
    parser.add_argument("action", choices=["pull", "push"])
    parser.add_argument("--data", default=os.environ.get("DESK_DATA", "var/desk"))
    args = parser.parse_args(argv)
    backend = backend_from_env()
    if backend is None:
        print("No checkpoint backend configured (set STATE_BUCKET or STATE_GIT_BRANCH).")
        return 0
    if args.action == "pull":
        found = pull(Path(args.data), backend)
        print(f"{'Restored' if found else 'No checkpoint yet at'} {backend.describe()}")
    else:
        size = push(Path(args.data), backend)
        print(f"Pushed {size or 0} bytes to {backend.describe()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
