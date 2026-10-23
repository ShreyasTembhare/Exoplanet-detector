"""
Niche strategy profiles for exoplanet detection.

Each profile tunes BLS search parameters, thresholds, and scoring
heuristics to optimise for a specific class of planet discovery.
"""

from typing import Dict, Optional

PROFILES: Dict[str, dict] = {
    "balanced": {
        "label": "Balanced (Default)",
        "description": "General-purpose search; covers ultra-short to ~3-week periods.",
        "period_min": 0.5,
        "period_max": 15.0,
        "nperiods": 10000,
        "sde_threshold": 7.0,
        "bls_threshold": 0.0,  # legacy raw-power gate retained at 0 (off)
        "probability_threshold": 0.85,
        "scoring_weights": {"prob_planet": 0.5, "bls_power": 0.25, "centroid": 0.15, "niche_bonus": 0.10},
    },
    "ultra_short_period": {
        "label": "Ultra-Short Period (<1 day)",
        "description": (
            "Targets hot, close-in planets with periods under 1 day. "
            "Uses a tight period grid and stricter periodic consistency checks."
        ),
        "period_min": 0.2,
        "period_max": 1.0,
        "nperiods": 15000,
        "sde_threshold": 8.0,
        "bls_threshold": 0.0,
        "probability_threshold": 0.80,
        "scoring_weights": {"prob_planet": 0.45, "bls_power": 0.30, "centroid": 0.15, "niche_bonus": 0.10},
    },
    "single_transit_long_period": {
        "label": "Single-Transit / Long Period",
        "description": (
            "Looks for single or few-transit events from long-period planets. "
            "Uses a wide period range with lower BLS gate to catch weak signals."
        ),
        "period_min": 10.0,
        "period_max": 100.0,
        "nperiods": 8000,
        "sde_threshold": 6.0,
        "bls_threshold": 0.0,
        "probability_threshold": 0.75,
        "scoring_weights": {"prob_planet": 0.40, "bls_power": 0.20, "centroid": 0.20, "niche_bonus": 0.20},
    },
    "low_snr_m_dwarf": {
        "label": "Low-SNR M-Dwarf Multi-Sector",
        "description": (
            "Optimised for faint M-dwarf hosts where transit depth is larger but "
            "SNR is low. Uses a lower SDE gate and benefits from multi-sector stacking."
        ),
        "period_min": 0.5,
        "period_max": 20.0,
        "nperiods": 12000,
        "sde_threshold": 6.0,
        "bls_threshold": 0.0,
        "probability_threshold": 0.70,
        "scoring_weights": {"prob_planet": 0.40, "bls_power": 0.25, "centroid": 0.20, "niche_bonus": 0.15},
    },
}

DEFAULT_PROFILE = "balanced"


def get_profile(name: str) -> dict:
    """Return a strategy profile dict by name, falling back to balanced."""
    return PROFILES.get(name, PROFILES[DEFAULT_PROFILE])


def profile_names() -> list:
    return list(PROFILES.keys())


def apply_profile_to_hunt_kwargs(profile_name: str, overrides: Optional[dict] = None) -> dict:
    """
    Build keyword-argument dict for run_hunt() from a strategy profile,
    with optional caller overrides taking precedence.
    """
    p = get_profile(profile_name)
    kwargs = {
        "period_min": p["period_min"],
        "period_max": p["period_max"],
        "nperiods": p["nperiods"],
        "bls_threshold": p.get("bls_threshold", 0.0),
        "sde_threshold": p.get("sde_threshold", 7.0),
        "threshold": p["probability_threshold"],
    }
    if overrides:
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return kwargs


def _finite(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and abs(number) != float("inf")


def detection_strength(sde: Optional[float] = None, bls_power: Optional[float] = None) -> float:
    """Map detection significance onto [0, 1].

    BLS power here is ``depth^2 * n_in * n_out / n`` and is not on a fixed
    scale, so SDE is preferred: SDE 7 (the gate) maps to 0 and SDE 20+ to 1.
    """
    if _finite(sde):
        return max(0.0, min(1.0, (float(sde) - 7.0) / 13.0))
    if _finite(bls_power) and float(bls_power) > 0:
        return max(0.0, min(1.0, float(bls_power) / (float(bls_power) + 1e-4)))
    return 0.0


def compute_candidate_score(prob_planet: Optional[float], bls_power: float,
                            centroid_offset: float, profile_name: str = "balanced",
                            vetting: Optional[dict] = None,
                            sde: Optional[float] = None) -> float:
    """
    Unified candidate score combining model probability, detection strength
    (SDE when given), centroid offset, and a niche-specific bonus.

    Without a classifier (``prob_planet`` is None) the probability weight is
    moved onto detection strength so SDE-only runs still rank sensibly.

    If ``vetting`` is provided, the score also penalises eclipsing-binary
    indicators (low odd/even ratio, strong secondary eclipse, low V/U)
    and contamination (centroid offset, Gaia neighbours).

    Returns a value in [0, 1] where higher is better.
    """
    p = get_profile(profile_name)
    w = dict(p["scoring_weights"])

    strength = detection_strength(sde, bls_power)
    if _finite(prob_planet):
        prob_component = max(0.0, min(1.0, float(prob_planet)))
    else:
        prob_component = 0.0
        w["bls_power"] += w["prob_planet"]
        w["prob_planet"] = 0.0
    centroid_ok = (
        max(0.0, 1.0 - float(centroid_offset) / 5.0)
        if _finite(centroid_offset)
        else 0.5
    )

    if profile_name == "ultra_short_period":
        niche_bonus = 0.8 if strength > 0.3 else 0.3
    elif profile_name == "single_transit_long_period":
        niche_bonus = 0.9
    elif profile_name == "low_snr_m_dwarf":
        niche_bonus = 0.7 if (prob_component > 0.6 or strength > 0.4) else 0.3
    else:
        niche_bonus = 0.5

    score = (
        w["prob_planet"] * prob_component
        + w["bls_power"] * strength
        + w["centroid"] * centroid_ok
        + w["niche_bonus"] * niche_bonus
    )

    # Graded demotion from pipeline.hard_flags already folds in odd/even,
    # secondary, shape, centroid, density and dilution evidence.
    demotion = (vetting or {}).get("demotion") or {}
    if isinstance(demotion, dict) and _finite(demotion.get("factor")):
        return round(min(score * float(demotion["factor"]), 1.0), 4)

    # Vetting penalties (multiplicative, in [0.3, 1.0]).
    if vetting:
        oer = vetting.get("odd_even_ratio")
        if oer is not None and isinstance(oer, (int, float)) and oer == oer and oer > 0:
            if oer < 0.5:
                score *= 0.5
            elif oer < 0.8:
                score *= 0.85
        sec_sig = vetting.get("secondary_significance")
        if sec_sig is not None and isinstance(sec_sig, (int, float)) and sec_sig == sec_sig:
            if sec_sig > 5.0:
                score *= 0.4  # likely EB
            elif sec_sig > 3.0:
                score *= 0.7
        v_shape = vetting.get("v_shape_score")
        if v_shape is not None and isinstance(v_shape, (int, float)) and v_shape == v_shape:
            if v_shape < 0.5:
                score *= 0.7
        gaia = vetting.get("gaia_neighbors_within_30arcsec", -1)
        if isinstance(gaia, int) and gaia > 5:
            score *= 0.85
        dilution = vetting.get("gaia_flux_ratio")
        if _finite(dilution) and float(dilution) > 0.5:
            score *= 0.7

    return round(min(score, 1.0), 4)
