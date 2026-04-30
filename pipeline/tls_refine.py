"""
Phase 5: Transit Least Squares (TLS) second-stage refiner.

BLS uses a box-shaped transit model and is extremely fast, but it
under-weights small or grazing transits. TLS (Hippke & Heller 2019) uses
the actual physical transit shape (limb-darkened) and is significantly
better at recovering ``R_p < 2 R_earth`` planets. The cost is ~10x slower,
so we only run it on the BLS top-k peaks that already passed the SDE gate.

API:
  * :func:`refine_peaks` runs TLS in a narrow window around each BLS peak
    and returns refined :class:`TLSRefinement` objects.
  * Failures (TLS not installed, fit fails) degrade gracefully -- the BLS
    peak's parameters are returned with ``available=False``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TLSRefinement:
    period: float
    epoch: float
    depth: float
    duration_hours: float
    sde: float
    snr: float
    odd_even_mismatch: float = float("nan")
    transit_count: int = 0
    available: bool = False


def _tls_available() -> bool:
    try:
        import transitleastsquares  # noqa: F401
        return True
    except Exception:
        return False


def refine_peak(
    time: np.ndarray, flux: np.ndarray,
    period_seed: float, epoch_seed: float,
    half_window_frac: float = 0.05,
) -> TLSRefinement:
    """Refine one peak. Returns a :class:`TLSRefinement`."""
    if not _tls_available():
        return TLSRefinement(
            period=period_seed, epoch=epoch_seed, depth=float("nan"),
            duration_hours=float("nan"), sde=float("nan"), snr=float("nan"),
        )
    try:  # pragma: no cover - external library
        from transitleastsquares import transitleastsquares

        model = transitleastsquares(time, flux)
        period_min = period_seed * (1 - half_window_frac)
        period_max = period_seed * (1 + half_window_frac)
        result = model.power(period_min=period_min, period_max=period_max)
        return TLSRefinement(
            period=float(result.period),
            epoch=float(result.T0),
            depth=float(1.0 - result.depth),
            duration_hours=float(result.duration * 24.0),
            sde=float(result.SDE),
            snr=float(result.snr),
            odd_even_mismatch=float(getattr(result, "odd_even_mismatch", float("nan"))),
            transit_count=int(getattr(result, "transit_count", 0)),
            available=True,
        )
    except Exception as exc:
        logger.warning("TLS refine failed: %s", exc)
        return TLSRefinement(
            period=period_seed, epoch=epoch_seed, depth=float("nan"),
            duration_hours=float("nan"), sde=float("nan"), snr=float("nan"),
        )


def refine_peaks(
    time: np.ndarray, flux: np.ndarray,
    bls_peaks: list,
    half_window_frac: float = 0.05,
    max_peaks: int = 5,
) -> List[TLSRefinement]:
    """Refine the top-N BLS peaks."""
    out = []
    for pk in bls_peaks[:max_peaks]:
        period = getattr(pk, "period", None) or pk.get("period")
        epoch = getattr(pk, "epoch", None) or pk.get("epoch")
        if period is None or epoch is None:
            continue
        out.append(refine_peak(time, flux, float(period), float(epoch), half_window_frac))
    return out


def best_refinement(refinements: List[TLSRefinement]) -> Optional[TLSRefinement]:
    avail = [r for r in refinements if r.available]
    if not avail:
        return None
    return max(avail, key=lambda r: r.sde)
