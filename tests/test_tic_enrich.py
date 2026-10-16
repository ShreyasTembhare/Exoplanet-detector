"""TIC enrichment fills ranking fields before download order is chosen."""

from __future__ import annotations

from pipeline.target_rank import TargetRecord, rank_targets
from pipeline.tic_enrich import enrich_records, enrich_rows, load_cache, save_cache


def test_enrich_records_from_cache_and_fetcher(tmp_path):
    cache = tmp_path / "tic_enrich.json"
    save_cache(cache, {
        "100": {"radius": 0.4, "n_sectors": 8, "crowdsap": 0.96, "tmag": 11.2},
    })
    bare = [
        TargetRecord("100"),
        TargetRecord("200"),
    ]
    calls = []

    def fetcher(tic):
        calls.append(tic)
        if tic == "200":
            return {"radius": 1.1, "n_sectors": 2, "crowdsap": 0.8, "tmag": 13.0}
        return None

    enrich_records(bare, cache_path=cache, fetcher=fetcher)
    assert bare[0].radius == 0.4
    assert bare[0].n_sectors == 8
    assert bare[0].crowdsap == 0.96
    assert bare[0].tmag == 11.2
    assert bare[1].radius == 1.1
    assert bare[1].n_sectors == 2
    assert "200" in calls
    disk = load_cache(cache)
    assert disk["200"]["radius"] == 1.1

    # Ranking no longer collapses to pull order once fields exist.
    queue = rank_targets(bare, limit=2)
    assert [rec.tic_id for rec in queue.primary][0] == "100"


def test_enrich_rows_for_desk_table(tmp_path):
    cache = tmp_path / "tic_enrich.json"

    def fetcher(tic):
        return {"radius": 0.3, "n_sectors": 12, "crowdsap": 0.99, "tmag": 10.0}

    rows = enrich_rows([{"tic_id": "55"}], cache_path=cache, fetcher=fetcher)
    assert rows[0]["radius"] == 0.3
    assert rows[0]["n_sectors"] == 12
    assert rows[0]["crowdsap"] == 0.99
    assert rows[0]["tmag"] == 10.0
