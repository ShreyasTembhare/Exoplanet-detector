"""
Run expensive statistical validation (TRICERATOPS) on the top of the list only.

A TRICERATOPS fit takes minutes, so running it on every SDE-pass peak made the
hunt crawl. After a sector (or a hunt batch) the best ``n`` new candidates by
score that have not been validated yet are fitted; FPP/NFPP then feed the
graded demotion and the score is recomputed.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Callable, Optional

from pipeline.atomic import atomic_write_json, read_json

logger = logging.getLogger(__name__)


def _finite(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def pick_shortlist(candidate_dir: str, n: int = 5, min_score: float = 0.0) -> list:
    """Paths of the top-``n`` unvalidated, not-already-known candidates by score."""
    rows = []
    for path in glob.glob(os.path.join(candidate_dir, "*.json")):
        meta = read_json(path)
        if not isinstance(meta, dict) or "tic_id" not in meta:
            continue
        vetting = meta.get("vetting") or {}
        if (vetting.get("triceratops") or {}).get("checked"):
            continue
        if meta.get("discovery_status") == "KNOWN" or vetting.get("discovery_status") == "KNOWN":
            continue
        if vetting.get("kind") == "monotransit" or _finite(meta.get("best_period")) is None:
            continue
        score = _finite(meta.get("score"))
        if score is None or score < min_score:
            continue
        rows.append((score, path))
    rows.sort(reverse=True)
    return [path for _, path in rows[: max(0, int(n))]]


def validate_shortlist(
    candidate_dir: str,
    n: int = 5,
    min_score: float = 0.0,
    validator: Optional[Callable[..., dict]] = None,
) -> list:
    """Fit the shortlist and rewrite each candidate JSON. Returns updated paths."""
    from pipeline.hard_flags import demotion
    from strategy_profiles import DEFAULT_PROFILE, PROFILES, compute_candidate_score

    if validator is None:
        from pipeline.triceratops_validate import triceratops_available, validate_fpp

        if not triceratops_available():
            return []
        validator = validate_fpp
    updated = []
    for path in pick_shortlist(candidate_dir, n=n, min_score=min_score):
        meta = read_json(path)
        if not isinstance(meta, dict):
            continue
        vetting = dict(meta.get("vetting") or {})
        depth = _finite(vetting.get("depth"))
        result = validator(
            meta["tic_id"], float(meta["best_period"]),
            float(meta.get("epoch") or vetting.get("transit_center_btjd") or float("nan")),
            depth_ppm=depth * 1e6 if depth is not None else float("nan"),
            duration_hours=_finite(vetting.get("duration_hours")) or float("nan"),
        )
        result = {**result, "checked": True}
        vetting["triceratops"] = result
        if result.get("available"):
            vetting["fpp"] = result.get("fpp")
            vetting["nfpp"] = result.get("nfpp")
        vetting["demotion"] = demotion(vetting)
        meta["vetting"] = vetting
        meta["demotion"] = vetting["demotion"]
        profile = meta.get("strategy_profile")
        meta["score"] = compute_candidate_score(
            meta.get("prob_planet"), meta.get("bls_max_power") or 0.0, meta.get("centroid_offset"),
            profile if profile in PROFILES else DEFAULT_PROFILE, vetting=vetting, sde=meta.get("sde"),
        )
        atomic_write_json(path, meta)
        updated.append(path)
        logger.info("Validated %s: FPP=%s NFPP=%s", meta["tic_id"], result.get("fpp"), result.get("nfpp"))
    return updated
