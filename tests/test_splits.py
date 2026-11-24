"""Tests for ``orbit_correction.splits``. Synthetic IDs and epochs only; no real data."""

import dataclasses

import numpy as np
import pytest

from orbit_correction import splits as Sp
from orbit_correction.frames import utc_to_tt_seconds
from orbit_correction.types import SPLIT_LABELS

from .conftest import REPO

DAY = 86400.0
#: Data end for tests that are not about it: after every synthetic window.
LAST_OBS = "2026-11-22"


def label(ids, starts, cfg, last_observed_date_utc=LAST_OBS):
    return Sp.assign_split_labels(ids, starts, cfg, last_observed_date_utc=last_observed_date_utc)


@pytest.fixture
def cfg():
    return Sp.load_split_config(REPO / "config" / "splits.toml")


@pytest.fixture
def rows(cfg):
    """2000 synthetic objects (IDs 1..2000 plus the forced ones), 40 starts each, 2019-2026."""
    rng = np.random.default_rng(7)
    ids = np.concatenate([np.arange(1, 2001), np.array(cfg.forced_heldout_norad_ids)])
    t0 = utc_to_tt_seconds("2019-01-01T00:00:00")
    t1 = utc_to_tt_seconds("2026-06-01T00:00:00")
    norad = np.repeat(ids, 40)
    start = rng.uniform(t0, t1, size=norad.size)
    return norad, start


def test_config_is_approved_and_complete(cfg):
    assert cfg.status == "approved 2026-11-23"
    assert (cfg.seed, cfg.heldout_object_fraction, cfg.dev_object_fraction) == (
        20261123, 0.2, 0.15
    )
    assert cfg.time_cutoff_utc == "2025-01-01T00:00:00" and cfg.max_horizon_days == 7.0
    assert (cfg.event_exclusion_days_before, cfg.event_exclusion_days_after) == (7.0, 1.0)
    assert cfg.truth_data_end_margin_days == 41.0
    assert {26998, 22236, 13777, 2828, 19826, 36123, 7219, 26207} <= set(
        cfg.forced_heldout_norad_ids
    )
    assert len(cfg.validation_events) == 5


def test_labels_are_contract_labels(rows, cfg):
    labels = label(*rows, cfg)
    assert set(labels.tolist()) <= set(SPLIT_LABELS)
    assert {"train", "dev", "test", "reserved_object", "reserved_time", "excluded"} == set(
        labels.tolist()
    )


def test_no_object_on_both_sides(rows, cfg):
    norad, start = rows
    labels = label(norad, start, cfg)
    training_side = set(norad[np.isin(labels, ["train", "dev"])].tolist())
    held_side = set(norad[np.isin(labels, ["test", "reserved_object"])].tolist())
    assert training_side and held_side
    assert not training_side & held_side
    # and train and dev objects are disjoint too
    assert not set(norad[labels == "train"].tolist()) & set(norad[labels == "dev"].tolist())


@pytest.mark.parametrize("seed", [0, 1, 20261123, 987654321])
def test_forced_objects_always_held_out(rows, cfg, seed):
    c = dataclasses.replace(cfg, seed=seed, heldout_object_fraction=0.0)
    norad, start = rows
    labels = label(norad, start, c)
    forced = np.isin(norad, c.forced_heldout_norad_ids)
    assert set(labels[forced].tolist()) <= {"test", "reserved_object"}
    roles = Sp.object_roles(c.forced_heldout_norad_ids, c)
    assert set(roles.values()) == {"heldout"}


def test_no_training_epoch_at_or_after_cutoff(rows, cfg):
    norad, start = rows
    labels = label(norad, start, cfg)
    trainable = np.isin(labels, ["train", "dev"])
    cutoff = cfg.time_cutoff_tt_s
    assert np.all(start[trainable] < cutoff)
    # the whole prediction window ends by the cutoff
    assert np.all(start[trainable] + cfg.max_horizon_days * DAY <= cutoff)
    # test rows are all at or after the cutoff
    assert np.all(start[labels == "test"] >= cutoff)


def test_cutoff_boundary_exact():
    c = Sp.SplitConfig(
        status="test", seed=1, heldout_object_fraction=0.0, dev_object_fraction=0.0,
        time_cutoff_utc="2025-01-01T00:00:00", max_horizon_days=7.0,
        forced_heldout_norad_ids=(5,), event_exclusion_days_before=0.0,
        event_exclusion_days_after=0.0, truth_data_end_margin_days=0.0, validation_events=(),
    )
    cut = c.time_cutoff_tt_s
    starts = np.array([cut - 7 * DAY, cut - 7 * DAY + 1e-3, cut - 1e-3, cut, cut, cut - 1.0])
    ids = np.array([1, 1, 1, 1, 5, 5])
    labels = label(ids, starts, c).tolist()
    assert labels == [
        "train", "excluded", "excluded", "reserved_time", "test", "reserved_object"
    ]


def test_event_window_excluded(cfg):
    timed = next(e for e in cfg.validation_events if e.name.startswith("TIMED"))
    tca = timed.tca_tt_s
    # pick a training object
    roles = Sp.object_roles(range(1, 200), cfg)
    obj = next(i for i, r in roles.items() if r == "train")
    starts = np.array([tca - 20 * DAY, tca - 3 * DAY, tca + 0.5 * DAY, tca + 2 * DAY])
    labels = label(np.full(4, obj), starts, cfg).tolist()
    # windows: [-20,-13] clear; [-3,+4] overlaps; [+0.5,+7.5] overlaps (after = 1 d); [+2,+9] clear
    assert labels == ["train", "excluded", "excluded", "train"]


def test_deterministic_and_stable(rows, cfg):
    norad, start = rows
    a = label(norad, start, cfg)
    b = label(norad, start, cfg)
    np.testing.assert_array_equal(a, b)
    # an object's role does not depend on which other objects are present
    full = Sp.object_roles(range(1, 2001), cfg)
    part = Sp.object_roles(range(1000, 1100), cfg)
    assert all(full[i] == part[i] for i in part)
    other = Sp.object_roles(range(1, 2001), dataclasses.replace(cfg, seed=cfg.seed + 1))
    assert other != full


def test_fractions_close_to_config(cfg):
    roles = Sp.object_roles(range(100_000, 120_000), cfg)
    vals = np.array(list(roles.values()))
    held = np.mean(vals == "heldout")
    dev_of_rest = np.mean(vals[vals != "heldout"] == "dev")
    assert held == pytest.approx(cfg.heldout_object_fraction, abs=0.01)
    assert dev_of_rest == pytest.approx(cfg.dev_object_fraction, abs=0.01)


def test_object_draw_known_value():
    """Pins the hash so a refactor cannot silently reshuffle the split."""
    assert Sp.object_draw(26998, 20261123) == pytest.approx(
        int.from_bytes(
            __import__("hashlib").sha256(b"20261123:heldout:26998").digest()[:8], "big"
        ) / 2.0**64
    )
    assert 0.0 <= Sp.object_draw(1, 0) < 1.0


def test_config_validation(cfg):
    with pytest.raises(ValueError, match="not in forced"):
        dataclasses.replace(cfg, forced_heldout_norad_ids=(26998,))
    with pytest.raises(ValueError, match="heldout_object_fraction"):
        dataclasses.replace(cfg, heldout_object_fraction=1.0)


def test_summary_counts(rows, cfg):
    norad, start = rows
    labels = label(norad, start, cfg)
    summ = Sp.split_summary(labels, norad)
    assert sum(v["rows"] for v in summ.values()) == norad.size


def test_windows_ending_after_data_end_are_excluded_in_every_split():
    """Any window ending after the last fully observed space-weather day is excluded.

    Margin 0 here, to test the data-end rule on its own; the 41-day truth margin
    is tested below.
    """
    c = Sp.SplitConfig(
        status="test", seed=1, heldout_object_fraction=0.0, dev_object_fraction=0.0,
        time_cutoff_utc="2025-01-01T00:00:00", max_horizon_days=7.0,
        forced_heldout_norad_ids=(5,), event_exclusion_days_before=0.0,
        event_exclusion_days_after=0.0, truth_data_end_margin_days=0.0, validation_events=(),
    )
    last_obs = "2026-11-22"
    end = Sp.data_end_tt_s(last_obs)
    assert end == pytest.approx(utc_to_tt_seconds("2026-11-23T00:00:00"))
    # object 1 trains (reserved_time after the cutoff); object 5 is held out (test)
    starts = np.array([end - 7 * DAY, end - 7 * DAY + 1e-3, end - 1 * DAY, end + DAY])
    for obj, before in ((1, "reserved_time"), (5, "test")):
        got = label(np.full(4, obj), starts, c, last_obs).tolist()
        assert got == [before, "excluded", "excluded", "excluded"], obj
    # before the cutoff too: training and reserved_object windows past the data end
    early_end = Sp.data_end_tt_s("2024-06-30")
    s2 = np.array([early_end - 8 * DAY, early_end - 3 * DAY])
    assert label(np.full(2, 1), s2, c, "2024-06-30").tolist() == ["train", "excluded"]
    assert label(np.full(2, 5), s2, c, "2024-06-30").tolist() == ["reserved_object", "excluded"]


def test_truth_margin_41_days_before_data_end(cfg):
    """With the approved margin, windows end >= 41 days before the end of the last observed day."""
    last_obs = "2026-11-22"
    c = dataclasses.replace(cfg, heldout_object_fraction=0.0, dev_object_fraction=0.0)
    assert c.truth_data_end_margin_days == 41.0
    latest = Sp.latest_window_end_tt_s(last_obs, c)
    # end of 2026-11-22 is 2026-11-23T00:00 UTC; 41 days earlier is 2026-10-13T00:00 UTC
    assert latest == pytest.approx(utc_to_tt_seconds("2026-10-13T00:00:00"))
    assert latest == pytest.approx(Sp.data_end_tt_s(last_obs) - 41 * DAY)  # no leap second inside
    roles = Sp.object_roles(range(1, 200), c)
    obj = next(i for i, r in roles.items() if r == "train")
    forced = c.forced_heldout_norad_ids[0]
    starts = np.array([latest - 7 * DAY, latest - 7 * DAY + 1e-3, latest - 3 * DAY])
    # after the 2025 cutoff, so a training object is reserved_time and a held-out one test
    assert label(np.full(3, obj), starts, c, last_obs).tolist() == [
        "reserved_time", "excluded", "excluded"
    ]
    got = label(np.full(3, forced), starts, c, last_obs).tolist()
    assert got == ["test", "excluded", "excluded"]
    # the same windows were allowed under the bare data-end rule
    c0 = dataclasses.replace(c, truth_data_end_margin_days=0.0)
    assert label(np.full(3, obj), starts, c0, last_obs).tolist() == ["reserved_time"] * 3
    # a leap second inside the margin is counted in UTC days, not 86400 s blocks
    leap = Sp.latest_window_end_tt_s("2017-01-30", c)
    assert leap == pytest.approx(utc_to_tt_seconds("2016-12-21T00:00:00"))
    with pytest.raises(ValueError, match="margin"):
        dataclasses.replace(c, truth_data_end_margin_days=-1.0)


def test_data_end_is_required(rows, cfg):
    with pytest.raises(TypeError):
        Sp.assign_split_labels(*rows, cfg)
