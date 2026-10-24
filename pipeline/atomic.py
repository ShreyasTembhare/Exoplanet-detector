"""Crash-safe file writes: write a temp file in the same directory, fsync, rename."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def atomic_write_text(path: PathLike, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_json(path: PathLike, payload: Any, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(payload, indent=indent, default=str))


def read_json(path: PathLike, default: Any = None) -> Any:
    """Read JSON, returning ``default`` when the file is missing or torn."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default
