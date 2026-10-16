"""
Fill radius, n_sectors, CROWDSAP, and Tmag for bare TIC lists before ranking.

MAST sector queries often return only TIC ids (and maybe Tmag). Ranking then
falls back to defaults and becomes pull-order. This module keeps a small local
JSON cache and, when a catalog client is available, looks up the missing fields.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_CACHE = Path("cache/tic_enrich.json")

# Fields ranking needs. Missing values stay NaN / 1 so rank_targets can still run.
_ENRICH_KEYS = ("radius", "n_sectors", "crowdsap", "tmag")


def _finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number


def _needs_enrichment(record) -> bool:
    if not _finite(getattr(record, "radius", float("nan"))):
        return True
    if not _finite(getattr(record, "crowdsap", float("nan"))):
        return True
    if not _finite(getattr(record, "tmag", float("nan"))):
        return True
    n = getattr(record, "n_sectors", 1)
    try:
        if int(n) <= 1 and "n_sectors" not in (getattr(record, "extra", None) or {}):
            # Default n_sectors=1 is indistinguishable from "unknown"; still try.
            return True
    except (TypeError, ValueError):
        return True
    return False


def _tic_key(record) -> str:
    text = str(getattr(record, "tic_id", "") or "").strip().upper()
    if text.startswith("TIC"):
        text = text[3:].strip()
    return text


def load_cache(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_cache(path: Path, cache: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True))
    tmp.replace(path)


def apply_cached(record, entry: dict) -> None:
    """Copy known fields from a cache entry onto the record when missing."""
    if not entry:
        return
    if not _finite(getattr(record, "radius", float("nan"))) and _finite(entry.get("radius")):
        record.radius = float(entry["radius"])
    if not _finite(getattr(record, "crowdsap", float("nan"))) and _finite(entry.get("crowdsap")):
        record.crowdsap = float(entry["crowdsap"])
    if not _finite(getattr(record, "tmag", float("nan"))) and _finite(entry.get("tmag")):
        record.tmag = float(entry["tmag"])
    try:
        current_n = int(getattr(record, "n_sectors", 1) or 1)
    except (TypeError, ValueError):
        current_n = 1
    if current_n <= 1 and entry.get("n_sectors") not in (None, ""):
        try:
            record.n_sectors = max(int(entry["n_sectors"]), 1)
        except (TypeError, ValueError):
            pass


def fetch_tic_row(tic: str) -> Optional[dict]:
    """Look up one TIC via MAST Catalogs when installed. Returns None offline."""
    try:
        from astroquery.mast import Catalogs
    except ImportError:
        return None
    try:
        table = Catalogs.query_object(f"TIC {tic}", catalog="TIC")
    except Exception as exc:
        logger.info("TIC %s catalog lookup failed: %s", tic, exc)
        return None
    if table is None or len(table) == 0:
        return None
    row = table[0]
    out = {}
    for src, dst in (
        ("rad", "radius"),
        ("Radius", "radius"),
        ("Tmag", "tmag"),
        ("tmag", "tmag"),
        ("contratio", "crowdsap"),
    ):
        if src in row.colnames and dst not in out:
            try:
                value = float(row[src])
            except (TypeError, ValueError):
                continue
            if value == value:
                if dst == "crowdsap":
                    # contratio is contamination; CROWDSAP ≈ 1 - contamination.
                    out[dst] = float(max(0.0, min(1.0, 1.0 - value))) if value <= 1.5 else float("nan")
                    if out[dst] != out[dst]:
                        out.pop(dst, None)
                else:
                    out[dst] = value
    if "nsectors" in row.colnames:
        try:
            out["n_sectors"] = int(row["nsectors"])
        except (TypeError, ValueError):
            pass
    return out or None


def enrich_records(
    records: Sequence,
    *,
    cache_path: Optional[Path] = None,
    fetcher: Optional[Callable[[str], Optional[dict]]] = None,
    persist: bool = True,
) -> List:
    """Fill missing ranking fields from cache, then optional catalog fetcher.

    ``fetcher`` defaults to :func:`fetch_tic_row`. Pass a stub in tests.
    Records are mutated in place and also returned.
    """
    path = Path(cache_path) if cache_path is not None else DEFAULT_CACHE
    cache = load_cache(path)
    lookup = fetcher if fetcher is not None else fetch_tic_row
    dirty = False
    for record in records:
        tic = _tic_key(record)
        if not tic:
            continue
        if tic in cache:
            apply_cached(record, cache[tic])
        if not _needs_enrichment(record):
            continue
        entry = lookup(tic)
        if not entry:
            continue
        cache[tic] = {**cache.get(tic, {}), **{k: entry[k] for k in entry if k in _ENRICH_KEYS or k == "n_sectors"}}
        apply_cached(record, cache[tic])
        dirty = True
    if dirty and persist:
        save_cache(path, cache)
    return list(records)


def enrich_rows(
    rows: Iterable[dict],
    *,
    cache_path: Optional[Path] = None,
    fetcher: Optional[Callable[[str], Optional[dict]]] = None,
) -> List[dict]:
    """Enrich dict rows (desk TIC tables) the same way as TargetRecord lists."""
    from pipeline.target_rank import records_from_rows

    rows = list(rows)
    records = records_from_rows(rows)
    enrich_records(records, cache_path=cache_path, fetcher=fetcher)
    by_tic = {_tic_key(rec): rec for rec in records}
    out = []
    for row in rows:
        tic = str(row.get("tic_id") or row.get("tic") or row.get("tid") or "").strip()
        key = tic.upper()[3:].strip() if tic.upper().startswith("TIC") else tic
        rec = by_tic.get(key) or by_tic.get(tic)
        merged = dict(row)
        if rec is not None:
            if "radius" not in merged or str(merged.get("radius") or "").strip() == "":
                if _finite(rec.radius):
                    merged["radius"] = rec.radius
            if "n_sectors" not in merged or str(merged.get("n_sectors") or "").strip() == "":
                merged["n_sectors"] = int(rec.n_sectors) if rec.n_sectors else 1
            if "crowdsap" not in merged or str(merged.get("crowdsap") or "").strip() == "":
                if _finite(rec.crowdsap):
                    merged["crowdsap"] = rec.crowdsap
            if "tmag" not in merged or str(merged.get("tmag") or "").strip() == "":
                if _finite(rec.tmag):
                    merged["tmag"] = rec.tmag
        out.append(merged)
    return out
