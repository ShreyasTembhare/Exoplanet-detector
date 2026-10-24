"""In-memory candidate index keyed by file mtime.

The inbox used to re-read and re-parse every candidate JSON on every 1.2 s
poll. This keeps parsed payloads and only re-reads files whose mtime or size
changed, so listing stays cheap as the inbox grows. Plot arrays (light
curve, periodogram, folds) are dropped from the index; :meth:`get` reads the
full file for one candidate.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HEAVY_KEYS = ("light_curve", "periodogram", "phase_curve", "odd_even_curve", "secondary_curve")


class CandidateIndex:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._lock = threading.Lock()
        self._entries: Dict[str, Tuple[float, int, dict]] = {}

    @staticmethod
    def _decorate(meta: dict, path: Path) -> dict:
        meta["_json_path"] = str(path)
        meta["_id"] = path.stem
        png = path.with_suffix(".png")
        if png.exists():
            meta["_plot_path"] = str(png)
        return meta

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
                    if (meta.get("phase_curve") or {}).get("phase"):
                        meta["_has_phase_curve"] = True
                    for heavy in HEAVY_KEYS:
                        meta.pop(heavy, None)
                    self._entries[key] = (stat.st_mtime, stat.st_size, self._decorate(meta, path))
            for key in list(self._entries):
                if key not in seen:
                    del self._entries[key]
            return [dict(entry[2]) for entry in self._entries.values()]

    def get(self, candidate_id: str) -> Optional[dict]:
        self.refresh()
        with self._lock:
            entry = self._entries.get(candidate_id)
        if not entry:
            return None
        path = Path(entry[2]["_json_path"])
        try:
            meta = json.loads(path.read_text())
        except (OSError, ValueError):
            return dict(entry[2])
        if not isinstance(meta, dict):
            return dict(entry[2])
        return self._decorate(meta, path)
