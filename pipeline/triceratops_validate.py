"""
TRICERATOPS FPP / nearby-FPP wrapper for the shortlist only.

Giacalone et al. 2021 cuts used in the literature are FPP < 0.015 and
NFPP < 1e-3. The import is optional: when ``triceratops`` is not installed
the wrapper returns ``available=False`` and does not raise, so unit tests
and a hunt without the package both succeed.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

FPP_MAX = 0.015
NFPP_MAX = 1e-3


def triceratops_available() -> bool:
    try:
        import triceratops  # noqa: F401
        return True
    except Exception:
        return False


def validate_fpp(
    tic_id: str,
    period_days: float,
    epoch_btjd: float,
    *,
    depth_ppm: float = float("nan"),
    duration_hours: float = float("nan"),
    sectors: Optional[list] = None,
) -> dict:
    """Return FPP and NFPP for one shortlisted candidate.

    This does not run on every star. Callers should pass only SDE-pass
    peaks a human might follow up. A missing package or a failed fit
    yields ``available=False`` with NaN probabilities.
    """
    result = {
        "available": False,
        "fpp": float("nan"),
        "nfpp": float("nan"),
        "likely_planet": None,
        "error": "",
        "tic_id": str(tic_id),
        "period_days": float(period_days) if period_days == period_days else float("nan"),
        "epoch_btjd": float(epoch_btjd) if epoch_btjd == epoch_btjd else float("nan"),
    }
    if not triceratops_available():
        result["error"] = "triceratops is not installed"
        return result
    try:  # pragma: no cover - optional package
        import triceratops.triceratops as tr

        target = tr.Target(ID=int(str(tic_id).replace("TIC", "").strip()), sectors=sectors or "all")
        target.calc_depths(tdepth=depth_ppm, tdepth_err=max(depth_ppm * 0.1, 1.0))
        target.calc_probs(time=epoch_btjd)
        fpp = float(target.FPP)
        nfpp = float(getattr(target, "NFPP", float("nan")))
        result.update({
            "available": True,
            "fpp": fpp,
            "nfpp": nfpp,
            "likely_planet": bool(fpp < FPP_MAX and nfpp < NFPP_MAX),
            "error": "",
        })
        return result
    except Exception as exc:
        logger.warning("TRICERATOPS failed for %s: %s", tic_id, exc)
        result["error"] = str(exc)
        return result


def likely_planet(fpp: float, nfpp: float) -> bool:
    """Literature cut: FPP < 0.015 and NFPP < 1e-3. Non-finite values are not a pass."""
    try:
        fpp_f = float(fpp)
        nfpp_f = float(nfpp)
    except (TypeError, ValueError):
        return False
    if fpp_f != fpp_f or nfpp_f != nfpp_f:
        return False
    return fpp_f < FPP_MAX and nfpp_f < NFPP_MAX
