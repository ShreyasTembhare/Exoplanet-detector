"""Catalog-ephemeris labels, shortlist vetting, and pre-download ranking."""

from __future__ import annotations

import numpy as np
import pytest


def test_catalog_fold_uses_ephemeris_not_bls(monkeypatch):
    import pipeline.bls_gpu as bls
    from pipeline.catalog_labels import fold_at_catalog_ephemeris

    def boom(*args, **kwargs):
        raise AssertionError("BLS must not label a catalog row")

    monkeypatch.setattr(bls, "compute_bls_periodogram", boom)

    time = np.linspace(0.0, 30.0, 6000)
    flux = np.ones_like(time)
    # Deeper impostor at a different period. Catalog ephemeris is the shallow one.
    impostor_p, impostor_e = 5.5, 0.3
    phase = (time - impostor_e) % impostor_p
    flux[(phase < 0.08) | (phase > impostor_p - 0.08)] -= 0.02
    catalog_p, catalog_e, catalog_dur = 2.2, 1.15, 0.08
    phase = (time - catalog_e) % catalog_p
    flux[(phase < catalog_dur / 2) | (phase > catalog_p - catalog_dur / 2)] -= 0.004

    global_view, local_view = fold_at_catalog_ephemeris(
        time, flux, catalog_p, catalog_e, catalog_dur,
    )
    assert global_view[0] < np.median(global_view) - 0.001
    assert np.median(local_view) - np.min(local_view) < 0.012

    # build_from_catalog_records is asserted in the temp-dir test. This test
    # only has to prove the fold itself never calls BLS.
    assert global_view.shape[0] == 2048
    assert local_view.shape[0] == 256


def test_catalog_records_skip_pc_and_do_not_inject(tmp_path):
    from pipeline.catalog_labels import build_from_catalog_records

    time = np.linspace(0.0, 20.0, 2000)
    flux = np.ones_like(time)
    period, epoch, duration = 3.0, 1.4, 0.1
    phase = (time - epoch) % period
    flux[(phase < duration / 2) | (phase > period - duration / 2)] -= 0.006
    manifest = build_from_catalog_records(
        [
            {"tic": "10", "label": 1, "period": period, "epoch": epoch,
             "duration_days": duration, "sector": 4},
            {"tic": "11", "disposition": "FA", "period": 6.0, "epoch": 2.0,
             "duration_hours": 3.0, "sector": 4},
            {"tic": "12", "disposition": "PC", "period": 2.0, "epoch": 0.5,
             "duration_days": 0.08, "sector": 4},
        ],
        {"10": (time, flux), "11": (time, np.ones_like(time)), "12": (time, flux)},
        str(tmp_path),
    )
    assert manifest is not None
    import pandas as pd
    frame = pd.read_csv(manifest)
    assert set(frame["label"]) == {0, 1}
    assert "12" not in set(frame["tic"].astype(str))
    assert set(frame["folded_at"]) == {"catalog"}
    saved = np.load(frame.loc[frame["label"] == 1, "path"].iloc[0])
    assert saved["epoch"] == pytest.approx(epoch)
    assert saved["period"] == pytest.approx(period)


def test_hard_flags_and_difference_image_and_triceratops():
    from pipeline.difference_image import difference_image_score
    from pipeline.hard_flags import hard_flag_reasons
    from pipeline.triceratops_validate import likely_planet, triceratops_available, validate_fpp
    from pipeline.vetting import compute_vetting_metrics

    assert "odd_even" in hard_flag_reasons({"odd_even_ratio": 0.2, "n_transits": 6})
    assert "secondary" in hard_flag_reasons({"secondary_significance": 8.0, "n_transits": 4})
    assert "centroid" in hard_flag_reasons({"centroid_offset_px": 1.4, "n_transits": 4})
    assert "n_transits" in hard_flag_reasons({"n_transits": 1})
    assert hard_flag_reasons({
        "odd_even_ratio": 0.97, "secondary_significance": 1.0,
        "centroid_offset_px": 0.2, "n_transits": 5,
    }) == []

    time = np.linspace(1700.0, 1727.0, 500)
    flux = np.ones_like(time)
    metrics = compute_vetting_metrics(time, flux, 3.0, 1701.5, bls_sde=8.0)
    assert metrics.transit_center_btjd == pytest.approx(1701.5)

    n = 50
    tpf_time = np.linspace(0.0, 10.0, n)
    cube = np.ones((n, 5, 5))
    period, epoch = 2.5, 0.2
    phase = (tpf_time - epoch) % period
    cube[(phase < 0.12) | (phase > period - 0.12), 2, 2] -= 0.4
    score = difference_image_score(tpf_time, cube, period, epoch, half_width=0.06)
    assert score["available"] is True
    assert score["diff_image_significance"] > 1.0

    fpp = validate_fpp("999", 3.5, 1700.0, depth_ppm=500.0, duration_hours=2.0)
    if not triceratops_available():
        assert fpp["available"] is False
        assert fpp["fpp"] != fpp["fpp"]
    assert likely_planet(0.01, 1e-4) is True
    assert likely_planet(0.2, 1e-4) is False
    assert likely_planet(float("nan"), 1e-6) is False


def test_batman_injection_on_quiet_star():
    pytest.importorskip("batman")
    from pipeline.injection_recovery import completeness_on_quiet_star, completeness_summary

    time = np.linspace(0.0, 27.0, 1800)
    flux = np.ones_like(time)  # quiet residual, not an FP/FA curve
    rows = completeness_on_quiet_star(
        time, flux, periods=[3.0], radii_rearth=[8.0], tmag=10.5,
        rstar_rsun=1.0, sde_threshold=5.0, nperiods=40, backend="numpy",
    )
    assert rows[0]["tmag"] == pytest.approx(10.5)
    assert rows[0]["cdpp_ppm"] == rows[0]["cdpp_ppm"]
    assert rows[0]["recovered"] is True
    summary = completeness_summary(rows)
    assert summary["by_period"][3.0] == pytest.approx(1.0)
    assert summary["by_radius"][8.0] == pytest.approx(1.0)


def test_rank_expected_snr_not_sky_or_catalog_order():
    from pipeline.target_rank import TargetRecord, expected_snr, rank_targets

    small = TargetRecord("small", radius=0.3, n_sectors=5, crowdsap=0.95, tmag=11, cdpp_ppm=80)
    large = TargetRecord("large", radius=1.4, n_sectors=5, crowdsap=0.95, tmag=11, cdpp_ppm=80)
    assert expected_snr(small) > expected_snr(large)

    one = TargetRecord("one", radius=0.5, n_sectors=1, crowdsap=0.95, tmag=11, cdpp_ppm=100)
    cvz = TargetRecord("cvz", radius=0.5, n_sectors=13, crowdsap=0.95, tmag=11, cdpp_ppm=100)
    assert expected_snr(cvz) > expected_snr(one)

    crowded = TargetRecord("crowd", radius=0.4, n_sectors=4, crowdsap=0.55, tmag=12, cdpp_ppm=120)
    quiet = TargetRecord("quiet", radius=0.4, n_sectors=4, crowdsap=0.98, tmag=12, cdpp_ppm=120)
    assert expected_snr(quiet) > expected_snr(crowded)

    north = TargetRecord("north", radius=0.4, n_sectors=3, crowdsap=0.9, tmag=12, cdpp_ppm=100, ra=10, dec=80)
    south = TargetRecord("south", radius=0.4, n_sectors=3, crowdsap=0.9, tmag=12, cdpp_ppm=100, ra=190, dec=-70)
    assert expected_snr(north) == pytest.approx(expected_snr(south))

    faint = TargetRecord("faint", radius=0.2, n_sectors=10, tmag=17.5, crowdsap=0.99)
    packed = TargetRecord("packed", radius=0.2, n_sectors=10, tmag=10, crowdsap=0.2)
    good = TargetRecord("good", radius=0.5, n_sectors=2, tmag=12, crowdsap=0.9, cdpp_ppm=150)
    bad_first = TargetRecord("bad", radius=2.0, n_sectors=1, tmag=15, crowdsap=0.6, cdpp_ppm=900)
    host = TargetRecord(
        "host", radius=0.25, n_sectors=12, tmag=9, crowdsap=0.99, cdpp_ppm=40,
        disposition="CP", known_period=5.5, known_epoch=1710.0,
    )
    false_pos = TargetRecord(
        "fp", radius=0.2, n_sectors=13, tmag=9, crowdsap=0.99, cdpp_ppm=30, disposition="FP",
    )
    queue = rank_targets(
        [bad_first, faint, packed, false_pos, host, good],
        limit=2,
    )
    assert [rec.tic_id for rec in queue.primary][0] == "good"
    assert "faint" not in [rec.tic_id for rec in queue.primary]
    assert "packed" not in [rec.tic_id for rec in queue.primary]
    assert "host" not in [rec.tic_id for rec in queue.primary]
    assert queue.second_pass[0].tic_id == "host"
    assert queue.second_pass[0].known_period == pytest.approx(5.5)
    assert queue.primary[0].tic_id != "bad"
    # limit is best N of the primary queue, not "first two rows".
    assert len(queue.primary) <= 2
    assert queue.primary[0].tic_id == "good"


def test_tic_file_rank_limit_is_best_n(tmp_path, monkeypatch):
    from hunter import _get_ranked_queue

    path = tmp_path / "tics.csv"
    path.write_text(
        "tic,radius,n_sectors,crowdsap,tmag,cdpp_ppm,disposition,period\n"
        "1,1.8,1,0.7,15,800,,\n"
        "2,0.28,13,0.99,10,40,,\n"
        "3,0.3,8,0.97,11,50,CP,4.2\n"
    )
    monkeypatch.setattr("hunter._attach_toi_dispositions", lambda records: records)
    queue = _get_ranked_queue(sector=1, limit=1, tic_list_path=str(path))
    assert [rec.tic_id for rec in queue.primary] == ["2"]
    assert [rec.tic_id for rec in queue.second_pass] == ["3"]
