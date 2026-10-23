"""Graded demotion, Gaia contamination, known-orbit context, top-N validation."""

from __future__ import annotations

import json
import math

from pipeline.gaia_contamination import contamination_from_sources
from pipeline.hard_flags import demotion
from pipeline.known_planets import build_known_table
from pipeline.planet_search import known_context
from pipeline.shortlist import pick_shortlist, validate_shortlist
from strategy_profiles import compute_candidate_score


def test_demotion_is_graded():
    clean = demotion({"odd_even_ratio": 0.95, "secondary_significance": 1.0, "n_transits": 4})
    assert clean == {"factor": 1.0, "reasons": []}
    mild = demotion({"odd_even_ratio": 0.6})["factor"]
    harsh = demotion({"odd_even_ratio": 0.2})["factor"]
    assert 0.2 <= harsh < mild < 1.0
    both = demotion({"odd_even_ratio": 0.4, "secondary_significance": 10.0})
    assert {r["code"] for r in both["reasons"]} == {"odd_even", "secondary"}
    assert math.isclose(both["factor"], round(0.4 / 0.65 * 0.5, 4), rel_tol=1e-3)


def test_monotransit_is_not_demoted_for_one_transit():
    assert demotion({"n_transits": 1, "kind": "monotransit"})["factor"] == 1.0
    assert demotion({"n_transits": 1})["factor"] == 0.5


def test_contamination_penalty_skipped_when_blend_impossible():
    assert demotion({"gaia_flux_ratio": 1.5})["factor"] < 1.0
    assert demotion({"gaia_flux_ratio": 1.5, "gaia_max_neighbor_depth_needed": 3.0})["factor"] == 1.0


def test_contamination_from_sources():
    isolated = contamination_from_sources([0.3], [10.0])
    assert isolated["gaia_flux_ratio"] == 0.0
    assert isolated["gaia_neighbors_within_30arcsec"] == 0
    blended = contamination_from_sources([0.2, 5.0, 40.0], [12.0, 11.0, 9.0], depth=0.01)
    assert blended["gaia_neighbors_within_30arcsec"] == 1
    assert blended["gaia_flux_ratio"] > 1.0
    assert blended["gaia_brightest_neighbor_dmag"] == -3.0
    assert 0 < blended["gaia_max_neighbor_depth_needed"] < 1.0
    faint = contamination_from_sources([0.2, 10.0], [9.0, 17.0], depth=0.01)
    assert faint["gaia_max_neighbor_depth_needed"] > 1.0
    assert math.isnan(contamination_from_sources([], [])["gaia_flux_ratio"])


def test_known_context_marks_new_planet_on_known_host():
    table = build_known_table([{"tid": 42, "toi": "100.01", "pl_orbper": 3.0, "pl_tranmid": 2459000.5}])
    assert known_context("42", 3.0, 1000.5, table)["discovery_status"] == "KNOWN"
    assert known_context("42", 7.7, 1000.0, table)["discovery_status"] == "NEW_ON_KNOWN_HOST"
    assert known_context("7", 3.0, 1000.0, table)["discovery_status"] == "NEW_CANDIDATE"
    assert known_context("7", 3.0, 1000.0, {}) == {}


def test_score_uses_demotion_factor():
    base = compute_candidate_score(None, 0.01, 0.0, "balanced", vetting={}, sde=15.0)
    demoted = compute_candidate_score(None, 0.01, 0.0, "balanced",
                                      vetting={"demotion": {"factor": 0.5, "reasons": []}}, sde=15.0)
    assert math.isclose(demoted, base * 0.5, abs_tol=1e-3)


def _candidate(directory, name, score, **extra):
    meta = {"tic_id": name, "best_period": 3.0, "epoch": 1000.0, "score": score, "sde": 12.0,
            "bls_max_power": 0.01, "vetting": {"depth": 0.001, "duration_hours": 2.0}, **extra}
    path = directory / f"{name}.json"
    path.write_text(json.dumps(meta))
    return path


def test_shortlist_validates_top_n_once(tmp_path):
    _candidate(tmp_path, "a", 0.9)
    _candidate(tmp_path, "b", 0.5)
    _candidate(tmp_path, "c", 0.95, discovery_status="KNOWN")
    _candidate(tmp_path, "d", 0.99, vetting={"kind": "monotransit"})
    assert [p.split("/")[-1] for p in pick_shortlist(str(tmp_path), n=1)] == ["a.json"]

    calls = []

    def validator(tic, period, epoch, **kwargs):
        calls.append(tic)
        return {"available": True, "fpp": 0.9, "nfpp": 0.1}

    validate_shortlist(str(tmp_path), n=1, validator=validator)
    assert calls == ["a"]
    meta = json.loads((tmp_path / "a.json").read_text())
    assert meta["vetting"]["fpp"] == 0.9
    assert meta["demotion"]["factor"] < 1.0
    validate_shortlist(str(tmp_path), n=1, validator=validator)
    assert calls == ["a", "b"]
