"""Tests for ``orbit_correction.splits``. Synthetic IDs and epochs only; no real data."""

import dataclasses

import numpy as np
import pytest

from orbit_correction import splits as Sp
from orbit_correction.frames import utc_to_tt_seconds
from orbit_correction.types import SPLIT_LABELS

from .conftest import REPO

DAY = 86400.0


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


def test_config_is_pending_and_complete(cfg):
    assert cfg.status == "pending approval"
    assert {26998, 22236, 13777, 2828, 19826, 36123, 7219, 26207} <= set(
        cfg.forced_heldout_norad_ids
    )
    assert len(cfg.validation_events) == 5


def test_labels_are_contract_labels(rows, cfg):
    labels = Sp.assign_split_labels(*rows, cfg)
    assert set(labels.tolist()) <= set(SPLIT_LABELS)
    assert {"train", "dev", "test", "reserved_object", "reserved_time", "excluded"} == set(
        labels.tolist()
    )


def test_no_object_on_both_sides(rows, cfg):
    norad, start = rows
    labels = Sp.assign_split_labels(norad, start, cfg)
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
    labels = Sp.assign_split_labels(norad, start, c)
    forced = np.isin(norad, c.forced_heldout_norad_ids)
    assert set(labels[forced].tolist()) <= {"test", "reserved_object"}
    roles = Sp.object_roles(c.forced_heldout_norad_ids, c)
    assert set(roles.values()) == {"heldout"}


def test_no_training_epoch_at_or_after_cutoff(rows, cfg):
    norad, start = rows
    labels = Sp.assign_split_labels(norad, start, cfg)
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
        event_exclusion_days_after=0.0, validation_events=(),
    )
    cut = c.time_cutoff_tt_s
    starts = np.array([cut - 7 * DAY, cut - 7 * DAY + 1e-3, cut - 1e-3, cut, cut, cut - 1.0])
    ids = np.array([1, 1, 1, 1, 5, 5])
    labels = Sp.assign_split_labels(ids, starts, c).tolist()
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
    labels = Sp.assign_split_labels(np.full(4, obj), starts, cfg).tolist()
    # windows: [-20,-13] clear; [-3,+4] overlaps; [+0.5,+7.5] overlaps (after = 1 d); [+2,+9] clear
    assert labels == ["train", "excluded", "excluded", "train"]


def test_deterministic_and_stable(rows, cfg):
    norad, start = rows
    a = Sp.assign_split_labels(norad, start, cfg)
    b = Sp.assign_split_labels(norad, start, cfg)
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
    labels = Sp.assign_split_labels(norad, start, cfg)
    summ = Sp.split_summary(labels, norad)
    assert sum(v["rows"] for v in summ.values()) == norad.size
