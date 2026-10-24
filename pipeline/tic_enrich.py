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


BATCH_SIZE = 500


def _row_entry(row) -> dict:
    out = {}
    for src, dst in (("rad", "radius"), ("Tmag", "tmag"), ("contratio", "crowdsap")):
        if src not in row.colnames:
            continue
        try:
            value = float(row[src])
        except (TypeError, ValueError):
            continue
        if value != value:
            continue
        if dst == "crowdsap":
            if value > 1.5:
                continue
            value = float(max(0.0, min(1.0, 1.0 - value)))
        out[dst] = value
    return out


def fetch_tic_rows(tics: Sequence[str], batch_size: int = BATCH_SIZE) -> Optional[dict]:
    """Look up many TICs with one MAST query per ``batch_size`` ids.

    Returns ``{tic: entry}`` (TICs with no row are absent) or ``None`` when
    astroquery is unavailable or every batch failed.
    """
    try:
        from astroquery.mast import Catalogs
    except ImportError:
        return None
    found: dict = {}
    ok = False
    tics = [str(t) for t in tics if str(t).strip()]
    for start in range(0, len(tics), batch_size):
        chunk = tics[start:start + batch_size]
        try:
            table = Catalogs.query_criteria(catalog="TIC", ID=[int(t) for t in chunk if t.isdigit()])
        except Exception as exc:
            logger.info("TIC batch lookup failed (%d ids): %s", len(chunk), exc)
            continue
        ok = True
        for row in table or []:
            entry = _row_entry(row)
            if entry:
                found[str(row["ID"]).strip()] = entry
    return found if ok else None


def enrich_records(
    records: Sequence,
    *,
    cache_path: Optional[Path] = None,
    fetcher: Optional[Callable[[str], Optional[dict]]] = None,
    batch_fetcher: Optional[Callable[[Sequence[str]], Optional[dict]]] = None,
    persist: bool = True,
) -> List:
    """Fill missing ranking fields from cache, then the TIC catalog.

    By default all missing TICs are looked up in batches with
    :func:`fetch_tic_rows`. ``fetcher`` (one TIC at a time) is kept for tests
    and custom catalogs. A TIC the catalog has nothing for is remembered so it
    is not queried again on every run. Records are mutated in place and also
    returned.
    """
    path = Path(cache_path) if cache_path is not None else DEFAULT_CACHE
    cache = load_cache(path)
    dirty = False
    pending = []
    for record in records:
        tic = _tic_key(record)
        if not tic:
            continue
        if tic in cache:
            apply_cached(record, cache[tic])
        if _needs_enrichment(record) and not cache.get(tic, {}).get("_missing"):
            pending.append((tic, record))

    if pending and fetcher is None:
        batch = (batch_fetcher or fetch_tic_rows)(sorted({tic for tic, _ in pending}))
        if batch is not None:
            lookup = batch.get
        else:
            lookup = None
    else:
        lookup = fetcher

    if lookup is not None:
        for tic, record in pending:
            entry = lookup(tic)
            if not entry:
                if fetcher is None:
                    cache[tic] = {**cache.get(tic, {}), "_missing": True}
                    dirty = True
                continue
            keep = {k: entry[k] for k in entry if k in _ENRICH_KEYS}
            cache[tic] = {**cache.get(tic, {}), **keep}
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
