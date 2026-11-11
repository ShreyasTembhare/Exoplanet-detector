"""MAST sector discovery without calling the archive."""

from tool.mast_setup import cached_sector_max, discover_sector_max, read_seen, remember_seen


def test_discover_sector_max_binary_search():
    published = set(range(1, 18))

    def count_fn(sector):
        return 3 if sector in published else 0

    assert discover_sector_max(count_fn) == 17


def test_cached_sector_max_reuses_a_fresh_file(tmp_path):
    calls = {"n": 0}

    def count_fn(sector):
        calls["n"] += 1
        return 1 if sector <= 4 else 0

    assert cached_sector_max(tmp_path, count_fn=count_fn) == 4
    first = calls["n"]
    assert cached_sector_max(tmp_path, count_fn=count_fn) == 4
    assert calls["n"] == first
    assert cached_sector_max(tmp_path, count_fn=count_fn, force=True) == 4
    assert calls["n"] > first


def test_remember_seen_appends_once(tmp_path):
    remember_seen(tmp_path, 2, ["10", "11"])
    remember_seen(tmp_path, 2, ["11", "12"])
    assert read_seen(tmp_path, 2) == {"10", "11", "12"}
