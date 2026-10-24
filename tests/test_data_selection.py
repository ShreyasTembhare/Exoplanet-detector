"""Light-curve product selection and batched TIC enrichment (no network)."""

from __future__ import annotations

from astropy.table import Table

from pipeline.phase1_preprocess import select_products
from pipeline.target_rank import TargetRecord
from pipeline.tic_enrich import enrich_records, load_cache


def _table(rows):
    return Table(rows=rows, names=("author", "exptime", "sequence_number"))


def test_widest_author_wins_and_no_sector_cap():
    rows = [("SPOC", 120.0, 14), ("SPOC", 120.0, 15)]
    rows += [("QLP", 1800.0, s) for s in range(14, 26)]
    chosen, author = select_products(_table(rows))
    assert author == "QLP"
    assert len(chosen) == 12


def test_tie_prefers_spoc_and_shortest_cadence_per_sector():
    rows = [
        ("QLP", 600.0, 40), ("SPOC", 120.0, 40), ("SPOC", 20.0, 40),
        ("QLP", 600.0, 41), ("SPOC", 120.0, 41),
    ]
    chosen, author = select_products(_table(rows))
    assert author == "SPOC"
    assert chosen == [2, 4]


def test_single_sector_and_cap():
    rows = [("TESS-SPOC", 600.0, 30), ("QLP", 200.0, 30), ("QLP", 200.0, 31), ("QLP", 200.0, 32)]
    chosen, author = select_products(_table(rows), sector=30)
    assert len(chosen) == 1 and author == "TESS-SPOC"
    capped, _ = select_products(_table(rows), max_sectors=2)
    assert capped == [1, 2]


def test_batch_enrichment_queries_once_and_remembers_misses(tmp_path):
    cache = tmp_path / "tic.json"
    calls = []

    def batch(tics):
        calls.append(list(tics))
        return {"1": {"radius": 0.5, "tmag": 10.0, "crowdsap": 0.9}}

    records = [TargetRecord("1"), TargetRecord("2")]
    enrich_records(records, cache_path=cache, batch_fetcher=batch)
    assert calls == [["1", "2"]]
    assert records[0].radius == 0.5
    disk = load_cache(cache)
    assert disk["2"]["_missing"] is True

    enrich_records([TargetRecord("2")], cache_path=cache, batch_fetcher=batch)
    assert len(calls) == 1


def test_offline_batch_does_not_poison_cache(tmp_path):
    cache = tmp_path / "tic.json"
    enrich_records([TargetRecord("9")], cache_path=cache, batch_fetcher=lambda tics: None)
    assert load_cache(cache) == {}
