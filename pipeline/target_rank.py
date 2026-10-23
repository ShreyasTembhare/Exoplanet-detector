"""
Rank a sector TIC queue before any light curve is downloaded.

The sort key is expected SNR for a roughly 1–4 Rearth planet in the period
window the available sectors can cover:

    transit probability × depth / CDPP × sqrt(n_transits)

Small stars, long continuous-viewing-zone baselines, and quiet uncrowded
stars rise to the top because those terms are in the score. ``limit`` is
the best N, not the first N a catalog happened to return.

The default hunt objective is :func:`expected_yield` instead: occurrence
prior × transit probability × detection probability summed over a period
and radius grid, counting only the gain over sectors already searched.

Known CP/KP hosts are not searched blindly for the same signal. They go to
a second-pass queue whose known ephemeris is masked. FP/FA dispositions are
deprioritized so a repeat of that signal is not the first thing downloaded.

Right ascension and declination are ignored. A sky map of already published
planets is where surveys stared, not where planets clump.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence

import numpy as np

# Earth radius / solar radius.
R_EARTH_OVER_R_SUN = 0.0091577
# a/R* at 1 AU around a Sun-density star, in the usual (P/365.25)^(2/3) units.
A_OVER_RSTAR_1AU = 215.032
SECTOR_DAYS = 27.4

# Cheap TIC cuts applied before download. They match the post-download
# prefilter so a faint or crowded star never enters the queue.
DEFAULT_MAX_TMAG = 16.0
DEFAULT_MIN_CROWDSAP = 0.5

CONFIRMED_DISPOSITIONS = {"CP", "KP"}
FALSE_DISPOSITIONS = {"FP", "FA"}


def _finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and np.isfinite(number)


def cdpp_ppm_from_tmag(tmag: float) -> float:
    """Rough TESS CDPP when the catalog did not provide one.

    About 30 ppm at Tmag 8 and a factor of ten every five magnitudes.
    """
    if not _finite(tmag):
        return 200.0
    return float(30.0 * 10 ** ((float(tmag) - 8.0) / 5.0))


@dataclass
class TargetRecord:
    """Catalog row used to rank a target before its light curve is fetched."""

    tic_id: str
    radius: float = float("nan")  # R_sun
    n_sectors: int = 1
    crowdsap: float = float("nan")
    tmag: float = float("nan")
    cdpp_ppm: float = float("nan")
    rho: float = float("nan")
    disposition: str = ""
    known_period: float = float("nan")
    known_epoch: float = float("nan")
    ra: float = float("nan")
    dec: float = float("nan")
    score: float = float("nan")
    queue: str = "primary"
    source_index: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def tic_norm(self) -> str:
        text = str(self.tic_id).strip().upper()
        if text.startswith("TIC"):
            text = text[3:].strip()
        return text


@dataclass
class RankedQueue:
    """Best-N primary targets plus a masked second pass for known hosts."""

    primary: List[TargetRecord]
    second_pass: List[TargetRecord]

    def search_order(self) -> List[TargetRecord]:
        """Primary best-N first, then known hosts for an additional planet."""
        return list(self.primary) + list(self.second_pass)


def passes_tic_cuts(
    record: TargetRecord,
    *,
    max_tmag: float = DEFAULT_MAX_TMAG,
    min_crowdsap: float = DEFAULT_MIN_CROWDSAP,
) -> bool:
    """Drop stars the cheap catalog cuts already reject. Missing columns pass."""
    if _finite(record.tmag) and float(record.tmag) > max_tmag:
        return False
    if _finite(record.crowdsap) and float(record.crowdsap) < min_crowdsap:
        return False
    if _finite(record.radius) and float(record.radius) <= 0:
        return False
    return True


def expected_snr(
    record: TargetRecord,
    *,
    period_min: float = 0.5,
    period_max: float = 15.0,
    rp_rearth: float = 2.5,
) -> float:
    """Transit probability × depth / CDPP × sqrt(n_transits).

    ``rp_rearth`` defaults to 2.5, the middle of a 1–4 Rearth search.
    The representative period is the geometric mean of the window the
    baseline can actually cover with two or more transits.
    """
    radius = float(record.radius) if _finite(record.radius) and float(record.radius) > 0 else 1.0
    n_sectors = int(record.n_sectors) if record.n_sectors and int(record.n_sectors) > 0 else 1
    baseline = n_sectors * SECTOR_DAYS
    p_lo = max(float(period_min), 1e-3)
    p_hi_cover = baseline / 2.0  # at least two transits
    p_hi = min(float(period_max), p_hi_cover) if p_hi_cover > 0 else float(period_max)
    if p_hi <= p_lo:
        p_rep = p_lo
        n_transits = max(baseline / p_rep, 1.0)
    else:
        p_rep = float(np.sqrt(p_lo * p_hi))
        n_transits = max(baseline / p_rep, 2.0)

    depth = (float(rp_rearth) * R_EARTH_OVER_R_SUN / radius) ** 2
    rho = float(record.rho) if _finite(record.rho) and float(record.rho) > 0 else 1.0
    a_over_r = (p_rep / 365.25) ** (2.0 / 3.0) * (rho ** (1.0 / 3.0)) * A_OVER_RSTAR_1AU
    transit_prob = 1.0 / a_over_r if a_over_r > 0 else 0.0
    transit_prob = float(np.clip(transit_prob, 0.0, 1.0))

    cdpp = float(record.cdpp_ppm) if _finite(record.cdpp_ppm) and float(record.cdpp_ppm) > 0 else cdpp_ppm_from_tmag(record.tmag)
    cdpp = max(cdpp, 1.0)
    snr = transit_prob * (depth * 1e6) / cdpp * np.sqrt(n_transits)
    if _finite(record.crowdsap):
        snr *= float(np.clip(float(record.crowdsap), 0.05, 1.0))
    return float(snr)


# Approximate occurrence per (ln P, ln Rp) cell, after Kepler FGK results
# (Hsu et al. 2019; Fulton & Petigura 2018): small planets dominate and the
# rate rises as P^~1 below ~10 d then flattens. M dwarfs host roughly 2x as
# many small planets (Dressing & Charbonneau 2015). Only relative values
# matter for ranking, so the normalisation is loose.
OCCURRENCE_RADIUS_BINS = (  # (Rp_lo, Rp_hi, rate per unit ln P per unit ln R at P >= 10 d)
    (0.8, 1.4, 0.18),
    (1.4, 2.0, 0.22),
    (2.0, 2.8, 0.20),
    (2.8, 4.0, 0.10),
    (4.0, 8.0, 0.02),
    (8.0, 16.0, 0.01),
)
DETECTION_SNR = 7.1
DETECTION_WIDTH = 1.0


def occurrence_rate(period: float, rp_rearth: float, stellar_radius: float = 1.0) -> float:
    rate = 0.0
    for lo, hi, value in OCCURRENCE_RADIUS_BINS:
        if lo <= rp_rearth < hi:
            rate = value
            break
    rate *= min((float(period) / 10.0), 1.0) ** 1.0
    if stellar_radius < 0.6 and rp_rearth < 4.0:
        rate *= 2.0
    return float(rate)


def default_objective() -> str:
    """``HUNT_RANK_OBJECTIVE`` (``yield`` or ``snr``); yield by default."""
    value = os.environ.get("HUNT_RANK_OBJECTIVE", "yield").strip().lower()
    return value if value in {"yield", "snr"} else "yield"


def detection_probability(snr: float) -> float:
    """Logistic recovery curve centred on the usual 7.1-sigma TESS threshold."""
    return float(1.0 / (1.0 + np.exp(-(float(snr) - DETECTION_SNR) / DETECTION_WIDTH)))


def _stellar_terms(record: TargetRecord):
    radius = float(record.radius) if _finite(record.radius) and float(record.radius) > 0 else 1.0
    rho = float(record.rho) if _finite(record.rho) and float(record.rho) > 0 else 1.0 / radius ** 2
    cdpp = float(record.cdpp_ppm) if _finite(record.cdpp_ppm) and float(record.cdpp_ppm) > 0 else cdpp_ppm_from_tmag(record.tmag)
    dilution = float(np.clip(float(record.crowdsap), 0.05, 1.0)) if _finite(record.crowdsap) else 1.0
    return radius, rho, max(cdpp, 1.0), dilution


def _detect_grid(record: TargetRecord, periods: np.ndarray, radii: np.ndarray, n_sectors: float) -> np.ndarray:
    """Detection probability on a (period, radius) grid for ``n_sectors`` of data."""
    if n_sectors <= 0:
        return np.zeros((len(periods), len(radii)))
    radius, rho, cdpp, dilution = _stellar_terms(record)
    baseline = n_sectors * SECTOR_DAYS
    n_tr = baseline / periods
    # Central-transit duration in hours for this stellar density.
    dur_h = 13.0 * (periods / 365.25) ** (1.0 / 3.0) * rho ** (-1.0 / 3.0)
    depth_ppm = (radii[None, :] * R_EARTH_OVER_R_SUN / radius) ** 2 * 1e6 * dilution
    snr = depth_ppm / cdpp * np.sqrt(np.maximum(n_tr * dur_h, 0.0))[:, None]
    p = 1.0 / (1.0 + np.exp(-(snr - DETECTION_SNR) / DETECTION_WIDTH))
    # A periodic search needs two transits; the window function cuts long P.
    return np.where((n_tr >= 2.0)[:, None], p, 0.0)


def expected_yield(
    record: TargetRecord,
    *,
    period_min: float = 0.5,
    period_max: float = 15.0,
    prior_sectors: Optional[float] = None,
    n_period: int = 24,
) -> float:
    """Expected number of *new* detectable planets this target adds.

    Sums occurrence × transit probability × detection probability over a
    log (P, Rp) grid. When the star was already searched with
    ``prior_sectors`` of data (default ``record.extra['searched_sectors']``),
    only the gain in detection probability counts, so stars whose added
    baseline unlocks longer periods or smaller planets rise and fully
    exhausted ones fall.
    """
    radius, rho, _, _ = _stellar_terms(record)
    n_sectors = float(record.n_sectors) if record.n_sectors and int(record.n_sectors) > 0 else 1.0
    if prior_sectors is None:
        prior_sectors = float((record.extra or {}).get("searched_sectors") or 0.0)
    log_p = np.linspace(np.log(max(period_min, 1e-3)), np.log(max(period_max, period_min * 1.01)), n_period)
    periods = np.exp(log_p)
    dlnp = log_p[1] - log_p[0] if n_period > 1 else 1.0
    radii, dlnr = [], []
    for lo, hi, _ in OCCURRENCE_RADIUS_BINS:
        radii.append(np.sqrt(lo * hi))
        dlnr.append(np.log(hi / lo))
    radii = np.asarray(radii)
    dlnr = np.asarray(dlnr)

    a_over_r = (periods / 365.25) ** (2.0 / 3.0) * rho ** (1.0 / 3.0) * A_OVER_RSTAR_1AU
    p_transit = np.clip(1.0 / a_over_r, 0.0, 1.0)
    occ = np.array([[occurrence_rate(p, r, radius) for r in radii] for p in periods])
    gain = _detect_grid(record, periods, radii, n_sectors) - _detect_grid(
        record, periods, radii, min(prior_sectors, n_sectors)
    )
    cell = occ * p_transit[:, None] * np.clip(gain, 0.0, 1.0) * dlnp * dlnr[None, :]
    return float(np.sum(cell))


def rank_targets(
    records: Sequence[TargetRecord],
    *,
    limit: int,
    period_min: float = 0.5,
    period_max: float = 15.0,
    rp_rearth: float = 2.5,
    max_tmag: float = DEFAULT_MAX_TMAG,
    min_crowdsap: float = DEFAULT_MIN_CROWDSAP,
    second_pass_limit: Optional[int] = None,
    objective: str = "snr",
) -> RankedQueue:
    """Rank before download. ``limit`` keeps the best N primary targets.

    ``objective="snr"`` sorts by :func:`expected_snr`; ``"yield"`` sorts by
    :func:`expected_yield`, the marginal number of detectable planets.
    Sky position is not an input. Confirmed hosts are removed from the
    primary queue and returned in ``second_pass`` for a masked re-search.
    """
    primary: List[TargetRecord] = []
    second: List[TargetRecord] = []
    for index, record in enumerate(records):
        record.source_index = index
        if not passes_tic_cuts(record, max_tmag=max_tmag, min_crowdsap=min_crowdsap):
            continue
        if objective == "yield":
            record.score = expected_yield(record, period_min=period_min, period_max=period_max)
        else:
            record.score = expected_snr(
                record, period_min=period_min, period_max=period_max, rp_rearth=rp_rearth,
            )
        disposition = (record.disposition or "").strip().upper()
        record.disposition = disposition
        if disposition in CONFIRMED_DISPOSITIONS:
            record.queue = "second_pass"
            second.append(record)
        else:
            record.queue = "primary"
            primary.append(record)

    def sort_key(rec: TargetRecord):
        # Higher SNR first. FP/FA sort after every other star so a repeat of
        # a known false signal is not downloaded ahead of an unvetted target.
        # Ties keep catalog order. RA/Dec are not in this key.
        score = rec.score if _finite(rec.score) else -np.inf
        false_positive = 1 if (rec.disposition or "") in FALSE_DISPOSITIONS else 0
        return (false_positive, -score, rec.source_index)

    primary.sort(key=sort_key)
    second.sort(key=sort_key)
    cap = max(int(limit), 0)
    second_cap = cap if second_pass_limit is None else max(int(second_pass_limit), 0)
    return RankedQueue(primary=primary[:cap], second_pass=second[:second_cap])


def records_from_rows(rows: Iterable[dict]) -> List[TargetRecord]:
    """Build records from dicts. Unknown keys are ignored. RA/Dec are stored only."""
    out: List[TargetRecord] = []
    for row in rows:
        tic = row.get("tic_id", row.get("tic", row.get("tid", "")))
        disposition = row.get("disposition", row.get("tfopwg_disp", "")) or ""
        n_sectors = row.get("n_sectors", row.get("sectors", 1))
        try:
            n_sectors_i = int(n_sectors) if n_sectors not in (None, "") else 1
        except (TypeError, ValueError):
            # "1,2,13" style sector lists: count the tokens.
            n_sectors_i = len([p for p in str(n_sectors).replace(";", ",").split(",") if p.strip()]) or 1
        out.append(
            TargetRecord(
                tic_id=str(tic),
                radius=_as_float(row.get("radius", row.get("rad", row.get("st_rad")))),
                n_sectors=n_sectors_i,
                crowdsap=_as_float(row.get("crowdsap")),
                tmag=_as_float(row.get("tmag", row.get("Tmag"))),
                cdpp_ppm=_as_float(row.get("cdpp_ppm", row.get("cdpp"))),
                rho=_as_float(row.get("rho", row.get("rho_star"))),
                disposition=str(disposition).strip().upper(),
                known_period=_as_float(row.get("known_period", row.get("period", row.get("pl_orbper")))),
                known_epoch=_as_float(row.get("known_epoch", row.get("epoch", row.get("pl_tranmid")))),
                ra=_as_float(row.get("ra")),
                dec=_as_float(row.get("dec")),
            )
        )
    return out


def _as_float(value) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
