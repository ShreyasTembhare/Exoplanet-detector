"""
Niche strategy profiles for exoplanet detection.

Each profile tunes BLS search parameters, thresholds, and scoring
heuristics to optimise for a specific class of planet discovery.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

PROFILES: Dict[str, dict] = {
    "balanced": {
        "label": "Balanced (Default)",
        "description": "General-purpose search across standard period ranges.",
        "period_min": 1.0,
        "period_max": 15.0,
        "nperiods": 10000,
        "bls_threshold": 0.001,
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
        "bls_threshold": 0.002,
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
        "bls_threshold": 0.0005,
        "probability_threshold": 0.75,
        "scoring_weights": {"prob_planet": 0.40, "bls_power": 0.20, "centroid": 0.20, "niche_bonus": 0.20},
    },
    "low_snr_m_dwarf": {
        "label": "Low-SNR M-Dwarf Multi-Sector",
        "description": (
            "Optimised for faint M-dwarf hosts where transit depth is larger but "
            "SNR is low. Uses a lower BLS gate and benefits from multi-sector stacking."
        ),
        "period_min": 0.5,
        "period_max": 20.0,
        "nperiods": 12000,
        "bls_threshold": 0.0003,
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
        "bls_threshold": p["bls_threshold"],
        "threshold": p["probability_threshold"],
    }
    if overrides:
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return kwargs


def compute_candidate_score(prob_planet: float, bls_power: float,
                            centroid_offset: float, profile_name: str = "balanced") -> float:
    """
    Unified candidate score combining model probability, BLS power,
    centroid offset, and a niche-specific bonus.

    Returns a value in [0, 1] where higher is better.
    """
    p = get_profile(profile_name)
    w = p["scoring_weights"]

    prob_component = max(0.0, min(1.0, prob_planet))

    bls_norm = min(bls_power / 0.01, 1.0)

    centroid_ok = max(0.0, 1.0 - centroid_offset / 5.0) if (centroid_offset == centroid_offset and centroid_offset is not None) else 0.5

    niche_bonus = 0.0
    if profile_name == "ultra_short_period":
        niche_bonus = 0.8 if bls_power > 0.003 else 0.3
    elif profile_name == "single_transit_long_period":
        niche_bonus = 0.9
    elif profile_name == "low_snr_m_dwarf":
        niche_bonus = 0.7 if prob_planet > 0.6 else 0.3
    else:
        niche_bonus = 0.5

    score = (
        w["prob_planet"] * prob_component
        + w["bls_power"] * bls_norm
        + w["centroid"] * centroid_ok
        + w["niche_bonus"] * niche_bonus
    )
    return round(min(score, 1.0), 4)
