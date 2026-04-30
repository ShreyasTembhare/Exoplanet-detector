"""
TFOP-style discovery packet exporter.

The TESS Follow-up Observing Program (TFOP) consumes candidate alerts in a
fairly opinionated JSON shape: TIC ID, period, t0 (epoch), depth, duration,
SDE, vetting metrics, plot path. This module serializes our internal
``candidates/`` JSONs to that shape so they can be uploaded directly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

TFOP_SCHEMA_VERSION = "1.0"


@dataclass
class TFOPCandidate:
    tic_id: str
    period_days: float
    epoch_btjd: float
    depth_ppm: float
    duration_hours: float
    sde: float
    n_transits: int
    odd_even_ratio: float
    secondary_significance: float
    v_shape_score: float
    centroid_offset_arcsec: float
    prob_planet: float
    discovery_status: str
    plot_path: Optional[str] = None
    submitter: str = "tess-exoplanet-detector"
    notes: str = ""

    def packet_id(self) -> str:
        """Stable id: TIC + period rounded to 4 digits."""
        h = hashlib.md5(f"{self.tic_id}_{self.period_days:.4f}".encode()).hexdigest()
        return h[:12]


def _safe_float(v, default: float = float("nan")) -> float:
    try:
        f = float(v)
        if f != f:  # NaN
            return default
        return f
    except Exception:
        return default


def candidate_json_to_tfop(candidate: dict) -> TFOPCandidate:
    """Translate a ``candidates/TIC_*.json`` payload into a TFOPCandidate."""
    vetting = candidate.get("vetting") or {}
    centroid_px = _safe_float(candidate.get("centroid_offset"))
    plate_scale = 21.0  # arcsec / pixel for TESS
    return TFOPCandidate(
        tic_id=str(candidate.get("tic_id", "")),
        period_days=_safe_float(candidate.get("best_period")),
        epoch_btjd=_safe_float(vetting.get("transit_center_btjd"), 0.0),
        depth_ppm=_safe_float(vetting.get("depth")) * 1e6
            if vetting.get("depth") is not None else float("nan"),
        duration_hours=_safe_float(vetting.get("duration_hours")),
        sde=_safe_float(candidate.get("sde")),
        n_transits=int(_safe_float(vetting.get("n_transits"), 0)),
        odd_even_ratio=_safe_float(vetting.get("odd_even_ratio")),
        secondary_significance=_safe_float(vetting.get("secondary_significance")),
        v_shape_score=_safe_float(vetting.get("v_shape_score")),
        centroid_offset_arcsec=centroid_px * plate_scale
            if centroid_px == centroid_px else float("nan"),
        prob_planet=_safe_float(candidate.get("prob_planet")),
        discovery_status=str(candidate.get("discovery_status", "NEW_CANDIDATE")),
        plot_path=candidate.get("plot"),
    )


def export_packet(
    candidates: Iterable[dict],
    out_path: str,
    submitter: str = "tess-exoplanet-detector",
    profile: str = "balanced",
) -> str:
    """Write a TFOP-compatible JSON packet to ``out_path``."""
    items: List[dict] = []
    for c in candidates:
        if c.get("discovery_status") == "KNOWN":
            continue
        cand = candidate_json_to_tfop(c)
        cand.submitter = submitter
        items.append({
            "packet_id": cand.packet_id(),
            "tic_id": cand.tic_id,
            "period_days": cand.period_days,
            "epoch_btjd": cand.epoch_btjd,
            "depth_ppm": cand.depth_ppm,
            "duration_hours": cand.duration_hours,
            "sde": cand.sde,
            "n_transits": cand.n_transits,
            "odd_even_ratio": cand.odd_even_ratio,
            "secondary_significance": cand.secondary_significance,
            "v_shape_score": cand.v_shape_score,
            "centroid_offset_arcsec": cand.centroid_offset_arcsec,
            "prob_planet": cand.prob_planet,
            "discovery_status": cand.discovery_status,
            "plot_path": cand.plot_path,
        })

    packet = {
        "schema_version": TFOP_SCHEMA_VERSION,
        "submitter": submitter,
        "strategy_profile": profile,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_candidates": len(items),
        "candidates": items,
    }

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(packet, indent=2, default=str))
    return str(out)


def export_from_candidate_dir(candidate_dir: str = "candidates",
                              out_path: str = "exports/discovery_packet.json",
                              **kwargs) -> str:
    candidates = []
    for jf in sorted(Path(candidate_dir).glob("TIC_*.json")):
        try:
            candidates.append(json.loads(jf.read_text()))
        except Exception:
            continue
    return export_packet(candidates, out_path, **kwargs)


__all__ = [
    "TFOPCandidate",
    "candidate_json_to_tfop",
    "export_packet",
    "export_from_candidate_dir",
    "TFOP_SCHEMA_VERSION",
]
