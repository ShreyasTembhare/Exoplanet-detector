"""In-memory candidate index keyed by file mtime.

The inbox used to re-read and re-parse every candidate JSON on every 1.2 s
poll. This keeps parsed payloads and only re-reads files whose mtime or size
changed, so listing stays cheap as the inbox grows.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class CandidateIndex:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._lock = threading.Lock()
        self._entries: Dict[str, Tuple[float, int, dict]] = {}

    def refresh(self) -> List[dict]:
        with self._lock:
            seen = set()
            if self.directory.is_dir():
                for path in self.directory.glob("TIC_*.json"):
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    key = path.stem
                    seen.add(key)
                    cached = self._entries.get(key)
                    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
                        continue
                    try:
                        meta = json.loads(path.read_text())
                    except (OSError, ValueError):
                        # A torn write is skipped until the writer finishes.
                        continue
                    if not isinstance(meta, dict):
                        continue
                    meta["_json_path"] = str(path)
                    meta["_id"] = key
                    png = path.with_suffix(".png")
                    if png.exists():
                        meta["_plot_path"] = str(png)
                    self._entries[key] = (stat.st_mtime, stat.st_size, meta)
            for key in list(self._entries):
                if key not in seen:
                    del self._entries[key]
            return [dict(entry[2]) for entry in self._entries.values()]

    def get(self, candidate_id: str) -> Optional[dict]:
        self.refresh()
        with self._lock:
            entry = self._entries.get(candidate_id)
            return dict(entry[2]) if entry else None
