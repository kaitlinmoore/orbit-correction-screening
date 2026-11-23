"""Tests for the pure parts of ``scripts/download_spacetrack.py`` (no network)."""

import importlib.util

import pytest

from .conftest import REPO

_spec = importlib.util.spec_from_file_location(
    "download_spacetrack", REPO / "scripts" / "download_spacetrack.py"
)
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


def test_parse_ids():
    assert D.parse_ids("26998, 22236\n# comment\n13777 2828 # trailing\n26998") == [
        2828, 13777, 22236, 26998,
    ]
    with pytest.raises(ValueError):
        D.parse_ids("0")


def test_batches():
    assert D.batches([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    with pytest.raises(ValueError):
        D.batches([1], 0)


def test_query_url():
    url = D.build_query_url([26998, 22236], "2024-02-14", "2024-03-01")
    assert url == (
        "https://www.space-track.org/basicspacedata/query/class/gp_history/"
        "NORAD_CAT_ID/26998,22236/EPOCH/2024-02-14--2024-03-01/"
        "orderby/NORAD_CAT_ID%20asc,EPOCH%20asc/format/json"
    )
    with pytest.raises(ValueError):
        D.build_query_url([1], "2024-03-01", "2024-02-14")
    with pytest.raises(ValueError):
        D.build_query_url([1], "2024-13-01", "2024-12-14")


def test_batch_filename_is_stable():
    a = D.batch_filename([7219, 26207], "2005-01-01", "2005-01-18")
    assert a == D.batch_filename([7219, 26207], "2005-01-01", "2005-01-18")
    assert a != D.batch_filename([7219, 26208], "2005-01-01", "2005-01-18")
    assert a.endswith(".json.gz")


def test_rate_limiter_spacing_and_hourly_cap():
    clock = {"t": 0.0}
    slept = []

    def sleep(s):
        slept.append(s)
        clock["t"] += s

    lim = D.RateLimiter(min_interval_s=3.0, max_per_hour=5, clock=lambda: clock["t"], sleep=sleep)
    for _ in range(5):
        lim.wait()
    assert slept == [3.0] * 4
    lim.wait()  # sixth request in the hour waits until the first one is an hour old
    assert slept[-1] == pytest.approx(3600.0 - 12.0)
    starts = list(lim.sent)
    assert all(b - a >= 3.0 for a, b in zip(starts, starts[1:], strict=False))


def test_dry_run_needs_no_credentials(capsys, monkeypatch):
    monkeypatch.delenv("SPACETRACK_USER", raising=False)
    monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)
    rc = D.main(["--ids", "26998,22236,13777", "--start", "2024-02-14", "--end", "2024-03-01",
                 "--batch-size", "2", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "2 requests for 3 objects (dry run; nothing sent)" in out
