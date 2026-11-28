"""Tests for ``orbit_correction.generation``: candidates, TLE reading, pool, split,
sampling, drop statuses, chunked runs, resume and the manifest.

Fixtures: ``satcat_sample.csv`` and ``celestrak_gp_sample.json`` (CelesTrak extracts),
``spacetrack_gp_history_synthetic.json`` (synthetic, Space-Track layout),
``sw_celestrak_2024_extract.txt`` (CelesTrak space weather, observed through 2024).
TLE lines are written at test time from the CelesTrak element sets with the ``sgp4``
package's exporter. No Space-Track data is used.

Most run tests use a stand-in truth runner (truth = SGP4 plus a known residual), so
chunking, resume and the log are tested without Orekit. Two ``datagen`` tests use the
real truth propagator.
"""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import json
import math

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from orbit_correction import catalog
from orbit_correction import cheap_prop as C
from orbit_correction import frames as F
from orbit_correction import generation as G
from orbit_correction import residuals as R
from orbit_correction.spaceweather import parse_spaceweather
from orbit_correction.splits import load_split_config
from orbit_correction.truth_prop import TruthResult
from orbit_correction.types import Trajectory, validate_residual_table

from .conftest import FIXTURES, REPO

HAVE_OREKIT = all(importlib.util.find_spec(m) for m in ("orekit_jpype", "jdk4py", "orekitdata"))
DAY = 86400.0


@pytest.fixture(scope="module")
def filters():
    return catalog.load_filter_config(REPO / "config" / "catalog_filters.toml")


@pytest.fixture(scope="module")
def elsets():
    return catalog.load_celestrak_omm(FIXTURES / "celestrak_gp_sample.json")


@pytest.fixture(scope="module")
def sw():
    return parse_spaceweather(FIXTURES / "sw_celestrak_2024_extract.txt")


@pytest.fixture(scope="module")
def rcfg():
    return R.load_residual_config()


@pytest.fixture(scope="module")
def gcfg():
    return G.load_generation_config()


# ----------------------------------------------------------------- config


def test_generation_config_is_approved_and_has_the_decided_stages(gcfg):
    assert gcfg.status == "approved 2026-11-25"
    assert gcfg.fit_rms_flag_m == 1000.0
    assert gcfg.stage_windows == {"pilot": 5000, "first_batch": 50000, "ceiling": 300000}
    assert gcfg.roles == ("train", "dev")
    assert gcfg.min_start_spacing_days == 7.0


def test_generation_config_refuses_test_roles_and_short_spacing(gcfg):
    with pytest.raises(ValueError, match="roles"):
        dataclasses.replace(gcfg, roles=("train", "test"))
    with pytest.raises(ValueError, match="at least 7"):
        dataclasses.replace(gcfg, min_start_spacing_days=6.9)


# ----------------------------------------------------------------- candidates


SATCAT_HEADER = ["OBJECT_NAME", "OBJECT_ID", "NORAD_CAT_ID", "OBJECT_TYPE", "OPS_STATUS_CODE",
                 "OWNER", "LAUNCH_DATE", "LAUNCH_SITE", "DECAY_DATE", "PERIOD", "INCLINATION",
                 "APOGEE", "PERIGEE", "RCS", "DATA_STATUS_CODE", "ORBIT_CENTER", "ORBIT_TYPE"]


def _satcat(tmp_path, rows):
    path = tmp_path / "satcat.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, SATCAT_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SATCAT_HEADER})
    return path


def _obj(nid, name="OBJ", otype="DEB", status="", launch="2010-01-01", decay="", apogee="800",
         perigee="700"):
    return dict(NORAD_CAT_ID=nid, OBJECT_NAME=name, OBJECT_TYPE=otype, OPS_STATUS_CODE=status,
                LAUNCH_DATE=launch, DECAY_DATE=decay, APOGEE=apogee, PERIGEE=perigee)


def test_candidate_rule_leaves_out_only_what_cannot_pass(tmp_path, filters):
    rows = [
        _obj(1),                                         # kept
        _obj(2, otype="UNK"),                            # object_type
        _obj(3, otype="PAY", status="+"),                # operational_payload
        _obj(4, otype="PAY", status="-"),                # dead payload: kept
        _obj(5, launch="2027-01-01"),                    # launched_after_span
        _obj(6, decay="2014-11-15"),                     # decayed more than 30 d before
        _obj(7, decay="2014-12-15"),                     # inside the 30-day margin: kept
        _obj(8, apogee="2101"),                          # apogee_above_leo
        _obj(9, apogee="2099"),                          # inside the 100 km margin: kept
        _obj(10, apogee="", perigee=""),                 # blank: kept, counted
        _obj(11, perigee="150", decay="2020-05-01"),     # low perigee now: kept (no perigee test)
        _obj(12, name="STARLINK-1007", otype="PAY", status="D"),   # name rule (payload)
        _obj(13, name="IRIDIUM 33 DEB"),                 # constellation debris: kept
        _obj(14, name="ORBCOMM FM5 R/B", otype="R/B"),   # constellation rocket body: kept
        _obj(15, otype="PAY", status="+", name="TIMED"), # operational, but forced
    ]
    res = G.candidate_ids_from_satcat(_satcat(tmp_path, rows), filters, "2015-01-01",
                                      "2026-11-24", forced_ids=[15])
    assert res.ids.tolist() == [1, 4, 7, 9, 10, 11, 13, 14, 15]
    assert res.dropped == {"object_type": 1, "operational_payload": 2, "launched_after_span": 1,
                           "decayed_before_span": 1, "apogee_above_leo": 1, "name_rule": 1}
    assert res.forced_added == (15,) and res.unknown_kept == {"launch_date": 0, "apogee": 1}
    assert res.n_satcat == len(rows)


def test_candidate_name_rule_only_when_the_filter_reads_satcat_names(tmp_path, filters):
    path = _satcat(tmp_path, [_obj(12, name="STARLINK-1007", otype="PAY", status="D")])
    assert G.candidate_ids_from_satcat(path, filters, "2015-01-01", "2026-01-01").ids.size == 0
    by_elset = dataclasses.replace(filters, name_rule_column="object_name")
    # the filter would read each element set's own name: SATCAT names can't decide it here
    assert G.candidate_ids_from_satcat(path, by_elset, "2015-01-01", "2026-01-01").ids.tolist() \
        == [12]


def test_candidates_are_a_superset_of_what_the_filters_keep(tmp_path, filters, elsets):
    """Every fixture object the filters keep is a candidate (and the forced ones are in)."""
    joined = catalog.attach_satcat(elsets, catalog.load_satcat(FIXTURES / "satcat_sample.csv"))
    kept = set(catalog.apply_filters(joined, filters).kept.column("norad_id").to_pylist())
    forced = load_split_config(REPO / "config" / "splits.toml").forced_heldout_norad_ids
    res = G.candidate_ids_from_satcat(FIXTURES / "satcat_sample.csv", filters, "2015-01-01",
                                      "2026-11-24", forced)
    assert kept <= set(res.ids.tolist())
    assert set(forced) <= set(res.ids.tolist())


def test_download_sample_is_seeded_and_disjoint_from_the_rest():
    ids = list(range(1000, 3000))
    a = G.download_sample_ids(ids, 50, seed=7)
    assert a.tolist() == G.download_sample_ids(reversed(ids), 50, seed=7).tolist()
    assert a.size == 50 and len(set(a.tolist())) == 50
    assert a.tolist() != G.download_sample_ids(ids, 50, seed=8).tolist()


# ----------------------------------------------------------------- TLE reading


def _tle_lines(row, elnum=999, revnum=12345):
    """TLE lines for a catalog row, written by the sgp4 package's exporter."""
    from sgp4.exporter import export_tle

    sat = C.satrec_from_row(row)
    sat.classification = "U"
    desig = row["intl_designator"] or "00000A"
    sat.intldesg = desig[2:4] + desig[5:] if "-" in desig else desig
    sat.elnum, sat.revnum = elnum, revnum
    return export_tle(sat)


def test_tle_checksum_and_alpha5():
    l1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    l2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    assert G.tle_checksum(l1) == 7 and G.tle_checksum(l2) == 7
    assert G.tle_norad_id(l1) == 25544
    assert G.tle_norad_id("1 A0001U") == 100001 and G.tle_norad_id("1 Z9999U") == 339999
    assert G.tle_norad_id("1 B1425U") == 111425


def test_tle_epoch_to_iso():
    assert G.tle_epoch_to_iso("08264.51782528") == "2008-09-20T12:25:40.104192"
    assert G.tle_epoch_to_iso("57001.00000000") == "1957-01-01T00:00:00.000000"
    assert G.tle_epoch_to_iso("56366.50000000") == "2056-12-31T12:00:00.000000"
    # a leap-second day counts 86,400 s, so the last fraction stays inside 23:59:59
    assert G.tle_epoch_to_iso("16366.99999999") == "2016-12-31T23:59:59.999136"
    # rounding to the microsecond: 0.00000001 d = 864 us exactly
    assert G.tle_epoch_to_iso("24001.00000001") == "2024-01-01T00:00:00.000864"


def test_tle_record_matches_the_omm_record_and_the_library_reader(elsets):
    """TLE -> record -> row gives the OMM element set, and SGP4 states agree to a millimetre."""
    from sgp4.api import WGS72, Satrec

    for row in elsets.to_pylist():
        l1, l2 = _tle_lines(row)
        rec = G.tle_to_record(l1, l2, name="X")
        got = catalog.records_to_table([rec], "spacetrack_bulk_tle").to_pylist()[0]
        assert got["norad_id"] == row["norad_id"]
        assert got["intl_designator"] == row["intl_designator"]
        assert got["object_name"] == "X" and got["element_set_no"] == 999
        assert got["rev_at_epoch"] == 12345 and got["ephemeris_type"] == 0
        # TLE fields carry fewer digits than CelesTrak's OMM, so compare at TLE precision
        assert abs(got["epoch_tt_s"] - row["epoch_tt_s"]) < 1e-3
        assert got["mean_motion_rev_per_day"] == pytest.approx(row["mean_motion_rev_per_day"],
                                                                abs=1e-8)
        assert got["eccentricity"] == pytest.approx(row["eccentricity"], abs=1e-7)
        for c in ("inclination_deg", "raan_deg", "arg_perigee_deg", "mean_anomaly_deg"):
            assert got[c] == pytest.approx(row[c], abs=1e-4), c
        assert got["bstar_per_earth_radii"] == pytest.approx(row["bstar_per_earth_radii"],
                                                              rel=1e-4, abs=1e-9)
        # the library's own reader on the same lines: same mean elements, and SGP4
        # states over a week agree to a millimetre (the epoch differs < 1 microsecond)
        lib = Satrec.twoline2rv(l1, l2, WGS72)
        ours = C.satrec_from_row(got)
        for a in ("no_kozai", "ecco", "inclo", "nodeo", "argpo", "mo", "bstar"):
            assert getattr(ours, a) == getattr(lib, a), a
        assert abs((ours.jdsatepoch + ours.jdsatepochF) - (lib.jdsatepoch + lib.jdsatepochF)) \
            * DAY < 1e-6
        for tmin in (0.0, 1440.0, 7 * 1440.0):
            e1, r1, _ = ours.sgp4_tsince(tmin)
            e2, r2, _ = lib.sgp4_tsince(tmin)
            if e1 == 0 and e2 == 0:
                assert np.linalg.norm(np.subtract(r1, r2)) < 1e-6  # km


def test_tle_reader_two_and_three_line_sets_bad_and_stray_lines(elsets):
    rows = elsets.to_pylist()
    a1, a2 = _tle_lines(rows[0])
    b1, b2 = _tle_lines(rows[1])
    c1, c2 = _tle_lines(rows[2])
    d1, d2 = _tle_lines(rows[3])
    bad2 = c2[:68] + str((int(c2[68]) + 1) % 10)  # checksum broken
    text = [
        "0 FIRST OBJECT", a1, a2,      # three-line set
        b1, b2,                        # two-line set
        "", c1, bad2,                  # bad checksum: counted, not yielded
        d1,                            # line 1 without its line 2
        "0 TITLE", d1, d2,             # then a full three-line set
        a2,                            # a lone line 2
    ]
    stats = G.TleReadStats()
    recs = list(G.iter_tle_records(text, stats))
    assert [r["NORAD_CAT_ID"] for r in recs] == [rows[0]["norad_id"], rows[1]["norad_id"],
                                                 rows[3]["norad_id"]]
    assert [r["OBJECT_NAME"] for r in recs] == ["FIRST OBJECT", None, "TITLE"]
    assert stats.tles == 4 and stats.bad == 1 and stats.stray_lines == 2
    assert "checksum" in stats.examples[0] and stats.lines == len(text)
    # unwanted IDs are skipped after reading only the number
    stats = G.TleReadStats()
    recs = list(G.iter_tle_records(text, stats, want_ids={rows[1]["norad_id"]}))
    assert [r["NORAD_CAT_ID"] for r in recs] == [rows[1]["norad_id"]]
    assert stats.skipped_id == 3 and stats.bad == 0


def test_bulk_file_quirks_backslash_and_missing_checksum(elsets):
    """Line 1s ending in a backslash are read; 68-character lines are counted, not read."""
    rows = elsets.to_pylist()
    a1, a2 = _tle_lines(rows[0])
    b1, b2 = _tle_lines(rows[1])
    nochk1, nochk2 = b1[:68], b2[:68]
    text = [a1 + "\\\n", a2 + "\n", nochk1, nochk2]
    stats = G.TleReadStats()
    recs = list(G.iter_tle_records(text, stats))
    assert [r["NORAD_CAT_ID"] for r in recs] == [rows[0]["norad_id"]]
    assert stats.trailing_backslash == 1 and stats.no_checksum == 1 and stats.bad == 0
    assert stats.epoch_days == {a1[18:23]: 1, b1[18:23]: 1} or a1[18:23] == b1[18:23]
    with pytest.raises(G.TleNoChecksumError):
        G.tle_to_record(nochk1, nochk2)
    assert G.tle_to_record(a1 + "\\", a2)["NORAD_CAT_ID"] == rows[0]["norad_id"]
    # accepted on layout checks when allowed (Space-Track's 2015-2020 files), flagged
    stats = G.TleReadStats()
    recs = list(G.iter_tle_records(text, stats, allow_no_checksum=True))
    assert [r["_checksum"] for r in recs] == [True, False] and stats.no_checksum == 1
    assert recs[1] == {**G.tle_to_record(b1, b2), "_checksum": False}
    # a 68-character line whose columns are shifted is bad, not accepted
    shifted = nochk1[:30] + " " + nochk1[30:67]
    stats = G.TleReadStats()
    assert list(G.iter_tle_records([shifted, nochk2], stats, allow_no_checksum=True)) == []
    assert stats.bad == 1 and "layout" in stats.examples[0]


def test_tle_with_mismatched_numbers_is_refused(elsets):
    rows = elsets.to_pylist()
    a1, _ = _tle_lines(rows[0])
    _, b2 = _tle_lines(rows[1])
    with pytest.raises(G.TleFormatError, match="numbers differ"):
        G.tle_to_record(a1, b2)


# ----------------------------------------------------------------- pool and split


def _history():
    return catalog.load_spacetrack_gp_history(FIXTURES / "spacetrack_gp_history_synthetic.json")


def _history_satcat(tmp_path):
    return catalog.load_satcat(_satcat(tmp_path, [
        _obj(90001, name="SYNTH A", otype="DEB"), _obj(90002, name="SYNTH B", otype="DEB")]))


def test_exact_duplicates_are_removed_and_same_epoch_changes_counted():
    h = _history()
    doubled = pa.concat_tables([h, h.slice(0, 3)])
    out, st = G.drop_exact_duplicates(doubled)
    assert out.equals(h) and st == G.DedupeStats(15, 3, 0)
    # a reissue: same object and epoch, different B*
    rec = h.slice(0, 1).to_pylist()[0]
    rec["bstar_per_earth_radii"] = 9e-4
    changed = pa.Table.from_pylist([rec], schema=h.schema)
    out, st = G.drop_exact_duplicates(pa.concat_tables([h, changed]))
    assert out.num_rows == 13 and st.exact_duplicates == 0 and st.same_epoch_different == 2


def _reissue(rec, *, dt_s=0.0, number=None, source=None, **changes):
    """A copy of an element-set record shifted by ``dt_s`` with some fields changed."""
    out = dict(rec)
    out["epoch_tt_s"] = rec["epoch_tt_s"] + dt_s
    if number is not None:
        out["element_set_no"] = number
    if source is not None:
        out["source"] = source
    out.update(changes)
    return out


def test_reissues_collapse_to_the_highest_element_set_number():
    h = _history()
    rows = h.to_pylist()
    base = rows[0]  # object 90001's first set
    a = _reissue(base, number=10)
    b = _reissue(base, dt_s=0.0, number=12, bstar_per_earth_radii=9e-4)   # same epoch, kept
    c = _reissue(base, dt_s=30.0, number=11, mean_motion_rev_per_day=15.01)  # 30 s later
    t = pa.Table.from_pylist([a, b, c, *rows[1:]], schema=h.schema)
    out, st = G.collapse_reissues(t)
    assert out.num_rows == t.num_rows - 2
    first = out.filter(pc.and_(
        pc.equal(out.column("epoch_tt_s"), base["epoch_tt_s"]),
        pc.equal(out.column("norad_id"), base["norad_id"])))
    assert first.num_rows == 1 and first.column("element_set_no")[0].as_py() == 12
    assert st.groups == 1 and st.rows_removed == 2 and st.near_simultaneous_groups == 1
    assert st.same_epoch_groups == 0 and st.max_group_span_s == 30.0 and st.tie_groups == 0
    # sets more than 60 s apart are separate element sets, untouched
    far = _reissue(base, dt_s=61.0, number=1)
    out, st = G.collapse_reissues(pa.Table.from_pylist([a, far], schema=h.schema))
    assert out.num_rows == 2 and st.groups == 0 and st.rows_removed == 0
    # chaining: 50 s steps join into one group spanning 100 s
    chain = [_reissue(base, dt_s=50.0 * k, number=k + 1) for k in range(3)]
    out, st = G.collapse_reissues(pa.Table.from_pylist(chain, schema=h.schema))
    assert out.num_rows == 1 and out.column("element_set_no")[0].as_py() == 3
    assert st.max_group_span_s == 100.0
    # other objects never join a group
    other = _reissue(base, number=99, norad_id=90002)
    out, st = G.collapse_reissues(pa.Table.from_pylist([a, other], schema=h.schema))
    assert out.num_rows == 2 and st.groups == 0


def test_reissue_ties_go_to_the_set_read_last():
    h = _history()
    base = h.to_pylist()[0]
    a = _reissue(base, number=7, source="first")
    b = _reissue(base, number=7, source="second", bstar_per_earth_radii=9e-4)
    out, st = G.collapse_reissues(pa.Table.from_pylist([a, b], schema=h.schema))
    assert out.column("source").to_pylist() == ["second"]
    assert st.tie_groups == 1 and st.same_epoch_groups == 1 and st.removed_by_source == {"first": 1}
    # a missing number counts lowest, and all-missing is a tie settled by read order
    out, _ = G.collapse_reissues(pa.Table.from_pylist(
        [_reissue(base, number=3, source="numbered"), {**b, "element_set_no": None}],
        schema=h.schema))
    assert out.column("source").to_pylist() == ["numbered"]
    out, st = G.collapse_reissues(pa.Table.from_pylist(
        [{**a, "element_set_no": None}, {**b, "element_set_no": None}], schema=h.schema))
    assert out.column("source").to_pylist() == ["second"] and st.tie_groups == 1
    # the last-read rule follows input position, not epoch: the later-read set wins
    # even when its epoch is earlier
    early_read_late_epoch = _reissue(base, dt_s=20.0, number=5, source="read_first")
    late_read_early_epoch = _reissue(base, number=5, source="read_second")
    out, _ = G.collapse_reissues(pa.Table.from_pylist(
        [early_read_late_epoch, late_read_early_epoch], schema=h.schema))
    assert out.column("source").to_pylist() == ["read_second"]


def test_collapsing_reissues_removes_a_noise_sma_rise(tmp_path, filters):
    """A same-epoch reissue with a 2 km higher SMA would trip the SMA-rise rule alone."""
    h = _history()
    keep_obj = h.filter(pc.equal(h.column("norad_id"), 90001))
    rows = keep_obj.to_pylist()
    last = rows[-1]
    a_km = last["semimajor_axis_km"]
    noisy = _reissue(last, dt_s=5.0, number=(last["element_set_no"] or 0) - 1,
                     semimajor_axis_km=a_km + 2.0)
    t = pa.Table.from_pylist([*rows[:-1], noisy, last], schema=h.schema)
    sat = _history_satcat(tmp_path)
    res = G.build_pool(t, sat, filters, "2000-01-01", "2026-01-01")
    assert res.reissues.rows_removed == 1
    assert res.objects_dropped_at_rule["no_sma_increase"] == 0
    assert set(res.table.column("norad_id").to_pylist()) == {90001}


def test_pool_keeps_an_object_with_one_spike_and_counts_it(tmp_path, filters):
    """A single set 2 km above both neighbours is left out of the history rules only."""
    h = _history()
    rows = h.filter(pc.equal(h.column("norad_id"), 90001)).to_pylist()
    rows[2] = {**rows[2], "semimajor_axis_km": rows[2]["semimajor_axis_km"] + 2.0}
    t = pa.Table.from_pylist(rows, schema=h.schema)
    sat = _history_satcat(tmp_path)
    res = G.build_pool(t, sat, filters, "2000-01-01", "2026-01-01")
    assert res.table.num_rows == len(rows)  # the spike set stays in the pool
    rise = res.history_rules["no_sma_increase"]
    assert rise == {"objects_flagged": 0, "trips": 0, "spike_sets_left_out": 1,
                    "objects_flagged_without_spike_rule": 1, "trips_without_spike_rule": 1}
    assert set(res.history_rules) == set(catalog.HISTORY_RULES)
    off = dataclasses.replace(filters, exclude_history_spikes=False)
    assert G.build_pool(t, sat, off, "2000-01-01", "2026-01-01").table.num_rows == 0


def test_pool_applies_filters_and_reports_each_rule(tmp_path, filters):
    h = _history()
    res = G.build_pool(pa.concat_tables([h, h]), _history_satcat(tmp_path), filters,
                       "2000-01-01", "2026-01-01")
    assert res.n_loaded == 24 and res.n_in_span == 24 and res.dedupe.exact_duplicates == 12
    # 90002 raises its orbit by ~37 km: the SMA-rise rule drops the whole object
    assert res.dropped_at_rule["no_sma_increase"] == 6
    assert res.objects_dropped_at_rule["no_sma_increase"] == 1
    assert sum(res.objects_dropped_at_rule.values()) == 1
    assert set(res.table.column("norad_id").to_pylist()) == {90001}
    assert res.n_objects_in == 2 and res.n_objects_kept == 1
    t = res.table.column("epoch_tt_s").to_numpy()
    assert np.all(np.diff(t) > 0)  # sorted by (object, epoch)
    narrow = G.build_pool(h, _history_satcat(tmp_path), filters, "1990-01-01", "1990-02-01")
    assert narrow.n_in_span == 0 and narrow.table.num_rows == 0


def test_label_pool_counts_by_label():
    cfg = load_split_config(REPO / "config" / "splits.toml")
    t = pa.table({"norad_id": [90001, 90001, 24946],
                  "epoch_tt_s": [F.utc_to_tt_seconds(x) for x in
                                 ("2020-03-01T00:00:00", "2026-06-01T00:00:00",
                                  "2020-03-01T00:00:00")]})
    labeled, summary, moved = G.label_pool(t, cfg, last_observed_date_utc="2026-11-22")
    lab = labeled.column("split").to_pylist()
    assert lab[2] == "reserved_object"  # Iridium 33 is forced held out
    assert lab[1] in ("reserved_time", "test")
    assert sum(v["rows"] for v in summary.values()) == 3 and moved == {}


def test_windows_must_end_within_the_downloaded_history():
    """History ends 2026-01-01: a 2025-12-28 start (ends 2026-01-04) is excluded, any role."""
    cfg = load_split_config(REPO / "config" / "splits.toml")
    starts = ("2025-06-01T00:00:00", "2025-12-24T23:59:59", "2025-12-28T00:00:00",
              "2025-12-28T00:00:00")
    t = pa.table({"norad_id": [90001, 90001, 90001, 24946],
                  "epoch_tt_s": [F.utc_to_tt_seconds(x) for x in starts]})
    plain, _, _ = G.label_pool(t, cfg, "2026-11-22")
    labeled, summary, moved = G.label_pool(t, cfg, "2026-11-22",
                                           history_end_utc="2026-01-01T00:00:00")
    before, after = plain.column("split").to_pylist(), labeled.column("split").to_pylist()
    assert after[:2] == before[:2] and before[2] != "excluded"
    assert after[2] == after[3] == "excluded"
    # 24946 is forced held out: its 2025 start is a test row before the rule
    assert before[3] == "test"
    want = {"test": 2} if before[2] == "test" else {before[2]: 1, "test": 1}
    assert moved == want


# ----------------------------------------------------------------- sampling


def _labeled_pool(n_obj=40, n_epochs=60, seed=1):
    rng = np.random.default_rng(seed)
    ids, t, lab = [], [], []
    base = F.utc_to_tt_seconds("2020-01-01T00:00:00")
    for o in range(n_obj):
        times = np.sort(base + rng.uniform(0, 365 * DAY, n_epochs))
        role = ["train", "dev", "reserved_object", "test"][o % 4]
        for x in times:
            ids.append(10000 + o)
            t.append(float(x))
            lab.append(role if rng.random() > 0.1 else "excluded")
    return pa.table({"norad_id": ids, "epoch_tt_s": t, "split": lab})


def test_sampling_spacing_roles_and_rounds(gcfg):
    pool = _labeled_pool()
    cfg = dataclasses.replace(gcfg, max_starts_per_object=5)
    w = G.sample_windows(pool, cfg)
    assert set(w.column("split").to_pylist()) <= {"train", "dev"}
    ids, t = w.column("norad_id").to_numpy(), w.column("epoch_tt_s").to_numpy()
    for o in np.unique(ids):
        ts = np.sort(t[ids == o])
        assert ts.size <= 5
        assert np.all(np.diff(ts) >= 7 * DAY)
    # every object gets its first start before any gets a second
    rounds = w.column("start_round").to_numpy()
    assert np.all(np.diff(rounds) >= 0)
    n_obj = np.unique(ids).size
    assert np.unique(ids[:n_obj]).size == n_obj
    assert w.column("window_index").to_pylist() == list(range(w.num_rows))


def test_sampling_is_seeded_and_stages_are_prefixes(gcfg):
    pool = _labeled_pool()
    small = G.sample_windows(pool, gcfg, 30)
    big = G.sample_windows(pool, gcfg, 120)
    assert small.equals(big.slice(0, 30))
    assert small.equals(G.sample_windows(pool, gcfg, 30))
    other = G.sample_windows(pool, dataclasses.replace(gcfg, seed=gcfg.seed + 1), 30)
    assert not small.equals(other)
    with pytest.raises(ValueError, match="asked for"):
        G.sample_windows(pool, gcfg, 10**6)


def test_dailyap_subsample_is_seeded_subset(gcfg):
    w = G.sample_windows(_labeled_pool(), gcfg, 100)
    a = G.dailyap_subsample(w, 10, gcfg.seed)
    assert a.num_rows == 10 and set(a.column("window_index").to_pylist()) <= set(range(100))
    assert np.all(np.diff(a.column("window_index").to_numpy()) > 0)
    assert a.equals(G.dailyap_subsample(w, 10, gcfg.seed))


def test_dailyap_draw_follows_window_identity_not_position(gcfg):
    w = G.sample_windows(_labeled_pool(), gcfg, 100)
    a = G.dailyap_subsample(w, 10, gcfg.seed)
    chosen = set(G.dailyap_window_keys(a))
    # reorder the sample and renumber window_index: the same windows are chosen
    perm = np.random.default_rng(7).permutation(w.num_rows)
    shuffled = w.take(pa.array(perm)).set_column(
        w.column_names.index("window_index"), "window_index",
        pa.array(np.arange(w.num_rows), type=pa.int64()))
    assert set(G.dailyap_window_keys(G.dailyap_subsample(shuffled, 10, gcfg.seed))) == chosen
    # drop half of the other windows: every chosen window still in the sample stays chosen
    keep = [k in chosen or i % 2 == 0 for i, k in enumerate(G.dailyap_window_keys(w))]
    part = w.filter(pa.array(keep))
    assert chosen <= set(G.dailyap_window_keys(G.dailyap_subsample(part, 10, gcfg.seed)))
    sub = w.slice(0, 40)  # a smaller stage is a prefix of a larger one
    in_sub = chosen & set(G.dailyap_window_keys(sub))
    assert in_sub <= set(G.dailyap_window_keys(G.dailyap_subsample(sub, 10, gcfg.seed)))
    # the key is the object and the start epoch
    row = w.slice(0, 1).to_pylist()[0]
    assert G.dailyap_window_keys(w)[0] == f"{row['norad_id']}:{float(row['epoch_tt_s'])!r}"
    other = G.dailyap_subsample(w, 10, gcfg.seed + 1)
    assert set(G.dailyap_window_keys(other)) != chosen


# ----------------------------------------------------------------- drop statuses


def _one(elsets, norad, epoch_utc="2024-06-15T00:00:00", **changes):
    i = elsets.column("norad_id").to_pylist().index(norad)
    rec = elsets.slice(i, 1).to_pylist()[0]
    rec.update(epoch_utc=epoch_utc, epoch_tt_s=float(F.elset_epoch_to_tt_seconds(epoch_utc)))
    rec.update(changes)
    return pa.Table.from_pylist([rec], schema=elsets.schema)


def _truth_with(sgp4, rtn, status="ok"):
    tr = sgp4.trajectory
    return TruthResult(norad_id=tr.norad_id, status=status, trajectory=Trajectory(
        norad_id=tr.norad_id, t_tt_s=tr.t_tt_s.copy(),
        r_gcrf_km=R.states_from_rtn(tr.r_gcrf_km, tr.v_gcrf_km_s, rtn),
        v_gcrf_km_s=tr.v_gcrf_km_s.copy(), source="truth"))


def test_rtn_domain_is_a_counted_drop_status(elsets, sw, rcfg):
    grid = R.horizon_grid(rcfg)
    el = _one(elsets, 25544)
    sg = C.propagate_sgp4_from_epoch(el, grid)[0]
    ok = _truth_with(sg, np.zeros((grid.size, 3)))
    tr = ok.trajectory
    flipped = dataclasses.replace(ok, trajectory=dataclasses.replace(tr, r_gcrf_km=-tr.r_gcrf_km))
    row = el.to_pylist()[0]
    res = G.window_result(row, sg, flipped, sw, "train", grid, rcfg)
    assert res.status == "rtn_domain" and res.causes == ("rtn_domain",) and res.table is None
    assert res.altitude_band == "400-600" and "RTNDomainError" in res.message
    assert R.count_windows([res]) == {("rtn_domain", "400-600", ">=150"): 1}
    assert "rtn_domain" in R.DROP_STATUSES and R.DROP_STATUSES[-1] == "rtn_domain"
    assert G.window_result(row, sg, ok, sw, "train", grid, rcfg).status == "ok"
    # a continuity break mid-window: du jumps by 2 rad between two horizons
    rc = np.linalg.norm(tr.r_gcrf_km, axis=1)
    rtn = np.zeros((grid.size, 3))
    rtn[100:, 1] = 2.0 * rc[100:]
    res = G.window_result(row, sg, _truth_with(sg, rtn), sw, "train", grid, rcfg)
    assert res.status == "rtn_domain" and "continuity break" in res.message


def test_large_fit_misfit_is_a_flag_not_a_drop():
    from types import SimpleNamespace as NS

    def truth(rms_m):
        return NS(start_fit=NS(rms_m=rms_m))

    assert G._flags(truth(999.0), 1000.0) == []
    assert G._flags(truth(1000.5), 1000.0) == ["fit_rms_above_threshold"]
    assert G._flags(None, 1000.0) == [] and G._flags(NS(start_fit=None), 1000.0) == []
    assert G._flags(truth(None), 1000.0) == []


def test_start_below_the_floor_gets_its_own_status(elsets, sw, rcfg):
    # ISS elements at 16.6 rev/day: about 111 km, below the 120 km floor at the epoch
    el = _one(elsets, 25544, mean_motion_rev_per_day=16.6)
    start = C.sgp4_epoch_states(el)[0]
    assert start.trajectory is None and start.status.below_floor
    res = G.pre_truth_result(el.to_pylist()[0], start, sw, "train", rcfg)
    assert res.status == "start_below_floor" and res.truth_status == "not_run"
    assert set(G.PRE_TRUTH_STATUSES) <= set(G.generation_statuses())
    assert set(R.WINDOW_STATUSES) <= set(G.generation_statuses())


# ----------------------------------------------------------------- chunked runs


def _windows(elsets, gcfg):
    """Eight windows on fixture objects at 2024 epochs, labeled train/dev, sample order."""
    plan = [(25544, "2024-03-01T00:00:00"), (13777, "2024-03-05T06:00:00"),
            (19826, "2024-04-01T00:00:00"), (22236, "2024-05-01T12:00:00"),
            (25544, "2024-06-01T00:00:00"), (36123, "2024-06-20T00:00:00"),
            (2828, "2024-07-01T00:00:00"), (25544, "2024-08-01T00:00:00")]
    rows = [_one(elsets, n, e).to_pylist()[0] for n, e in plan]
    t = pa.Table.from_pylist(rows, schema=elsets.schema)
    t = t.append_column("split", pa.array(["train", "dev"] * 4))
    t = t.append_column("window_index", pa.array(np.arange(8), type=pa.int64()))
    t = t.append_column("start_round", pa.array([1] * 8, type=pa.int64()))
    return t


class FakeTruth:
    """Stand-in truth runner: SGP4 plus a known residual; some windows fail on purpose."""

    def __init__(self, windows, dt, fail=None):
        self.rows = {(r["norad_id"], r["epoch_tt_s"]): r for r in windows.to_pylist()}
        self.schema = windows.select(C.REQUIRED_COLUMNS).schema
        self.dt = dt
        self.fail = fail or {}
        self.calls = 0
        self.jobs = 0

    def __call__(self, jobs):
        self.calls += 1
        self.jobs += len(jobs)
        out = []
        for job in jobs:
            key = (job.initial.norad_id, float(job.initial.t_tt_s[0]))
            row = self.rows[key]
            assert np.array_equal(job.t_out_tt_s, row["epoch_tt_s"] + self.dt)
            # fitted start (the default): SGP4 samples over [epoch - 0.25 d, epoch]
            fs = job.fit_samples
            assert fs is not None and fs.norad_id == key[0]
            assert fs.t_tt_s[-1] == key[1] and fs.t_tt_s[0] == key[1] - 0.25 * DAY
            el = pa.Table.from_pylist([{c: row[c] for c in C.REQUIRED_COLUMNS}],
                                      schema=self.schema)
            sg = C.propagate_sgp4_from_epoch(el, self.dt)[0]
            kind = self.fail.get(key[1])
            if kind == "error":
                out.append(TruthResult(norad_id=key[0], status="error", message="boom"))
                continue
            if kind == "fit_failed":
                out.append(TruthResult(norad_id=key[0], status="fit_failed",
                                       message="not_converged"))
                continue
            rtn = np.column_stack([np.full(self.dt.size, 0.01), self.dt / DAY,
                                   np.zeros(self.dt.size)])
            res = _truth_with(sg, rtn)
            if kind == "far":
                tr = res.trajectory
                res = dataclasses.replace(
                    res, trajectory=dataclasses.replace(tr, r_gcrf_km=-tr.r_gcrf_km))
            out.append(dataclasses.replace(res, wall_s=0.001, n_evaluations=1, start_scale=1.0))
        return out


@pytest.fixture(scope="module")
def tcfg():
    from orbit_correction import truth_prop as T

    return T.load_truth_config()  # no JVM: reading the config only


def _run(tmp_path, elsets, sw, rcfg, gcfg, runner=None, chunk=3, windows=None, restart=None,
         truth_cfg=None):
    from orbit_correction import truth_prop as T

    w = windows if windows is not None else _windows(elsets, gcfg)
    dt = R.horizon_grid(rcfg)
    dt = dt[dt <= DAY]
    runner = runner or FakeTruth(w, dt)
    strata = G.window_strata(w, sw, rcfg, gcfg)
    settings = G.RunSettings(run_id="test", label="baseline", chunk_windows=chunk, n_workers=1)
    man = G.run_generation(w, tmp_path / "run", settings, dt_s=dt, residual_cfg=rcfg,
                           space_weather=sw, strata=strata, truth_runner=runner,
                           truth_cfg=truth_cfg or T.load_truth_config(),
                           restart_runner=restart, manifest_extra={"inputs": {"x": 1}})
    return w, dt, runner, man


def test_run_writes_chunks_log_and_manifest(tmp_path, elsets, sw, rcfg, gcfg):
    w = _windows(elsets, gcfg)
    dt = R.horizon_grid(rcfg)
    dt = dt[dt <= DAY]
    fail = {w.column("epoch_tt_s")[2].as_py(): "error", w.column("epoch_tt_s")[5].as_py(): "far",
            w.column("epoch_tt_s")[7].as_py(): "fit_failed"}
    runner = FakeTruth(w, dt, fail)
    _, _, _, man = _run(tmp_path, elsets, sw, rcfg, gcfg, runner=runner, windows=w)
    out = tmp_path / "run"
    assert sorted(p.name for p in out.glob("chunk_*.parquet")) == [
        "chunk_00000.parquet", "chunk_00001.parquet", "chunk_00002.parquet"]
    log = G.read_logs(out)
    assert [r["window_index"] for r in log] == list(range(8))  # every window, once
    st = {r["window_index"]: r["status"] for r in log}
    assert st[2] == "truth_error" and st[5] == "rtn_domain" and st[7] == "truth_fit_failed"
    assert all(st[i] == "ok" for i in (0, 1, 3, 4, 6))
    assert all(r["label"] == "baseline" and r["chunk_file"].startswith("chunk_") for r in log)
    # rows: only ok windows, contract-valid, labeled in the file metadata
    tables = [pq.read_table(p) for p in sorted(out.glob("chunk_*.parquet"))]
    for t in tables:
        validate_residual_table(t)
        assert t.schema.metadata[b"label"] == b"baseline"
    assert sum(t.num_rows for t in tables) == 5 * dt.size == man["n_rows"]
    assert man["status"] == {"ok": 5, "rtn_domain": 1, "truth_error": 1, "truth_fit_failed": 1}
    assert man["plan"]["start_rule"] == "fit_to_sgp4_before_epoch"
    assert man["plan"]["rtn_conversion"] == G.RTN_CONVERSION
    assert man["plan"]["fit_rms_flag_m"] == 1000.0 and man["flags"] == {}
    # the unwrapped angle's extent is logged for kept windows, None for dropped ones
    for r in log:
        if r["status"] == "ok":
            assert 0 < r["du_max_abs_rad"] < 1e-3 and 0 <= r["du_max_step_rad"] < 1e-4
        else:
            assert r["du_max_abs_rad"] is None and r["du_max_step_rad"] is None
        assert r["flags"] == []
    assert man["worker_pool_crashes"] == 0
    assert sum(man["counts_by_status_band_bin"].values()) == 8
    assert man["n_chunks"] == 3 and man["chunks_written_this_call"] == [0, 1, 2]
    assert man["inputs"] == {"x": 1} and "commit" in man["code"]
    assert json.loads((out / "run_manifest.json").read_text()) == man


def test_resume_skips_verified_chunks_and_redoes_damaged_ones(tmp_path, elsets, sw, rcfg, gcfg):
    w, dt, runner, man = _run(tmp_path, elsets, sw, rcfg, gcfg)
    first = runner.jobs
    out = tmp_path / "run"
    # a second call does nothing new
    _, _, runner2, man2 = _run(tmp_path, elsets, sw, rcfg, gcfg, runner=FakeTruth(w, dt))
    assert runner2.calls == 0 and man2["chunks_skipped_this_call"] == [0, 1, 2]
    assert man2["counts_by_status_band_bin"] == man["counts_by_status_band_bin"]
    # a damaged Parquet file and a missing marker: only those chunks are redone
    with open(out / "chunk_00001.parquet", "ab") as f:
        f.write(b"x")
    (out / "chunk_00002.done.json").unlink()
    runner3 = FakeTruth(w, dt)
    _, _, _, man3 = _run(tmp_path, elsets, sw, rcfg, gcfg, runner=runner3)
    assert man3["chunks_written_this_call"] == [1, 2] and runner3.jobs == first - 3
    assert len(G.read_logs(out)) == 8
    # a different chunk size is a different run
    with pytest.raises(ValueError, match="different run"):
        _run(tmp_path, elsets, sw, rcfg, gcfg, chunk=4)


def test_chunk_marker_names_its_windows(tmp_path, elsets, sw, rcfg, gcfg):
    w, *_ = _run(tmp_path, elsets, sw, rcfg, gcfg)
    out = tmp_path / "run"
    digest = G._ids_digest(w.slice(0, 3))
    assert G.chunk_is_complete(out, 0, digest)
    assert not G.chunk_is_complete(out, 0, G._ids_digest(w.slice(1, 3)))
    assert G.chunk_bounds(8, 3) == [(0, 3), (3, 6), (6, 8)]


def test_below_floor_start_makes_no_truth_job(tmp_path, elsets, sw, rcfg, gcfg):
    w = _windows(elsets, gcfg)
    i = w.column_names.index("mean_motion_rev_per_day")
    mm = w.column("mean_motion_rev_per_day").to_pylist()
    mm[4] = 16.6
    w = w.set_column(i, "mean_motion_rev_per_day", pa.array(mm))
    _, _, runner, man = _run(tmp_path, elsets, sw, rcfg, gcfg, windows=w)
    assert runner.jobs == 7 and man["status"]["start_below_floor"] == 1
    rec = [r for r in G.read_logs(tmp_path / "run") if r["window_index"] == 4][0]
    assert rec["truth_status"] == "not_run" and rec["truth_wall_s"] is None


def test_missing_fit_samples_make_no_truth_job(tmp_path, elsets, sw, rcfg, gcfg):
    """Start above 120 km, but the 0.25 days before it dip below: no job, no other start."""
    w = _windows(elsets, gcfg)
    rows = w.to_pylist()
    # eccentric orbit, epoch at apogee (about 750 km), perigee about 70 km
    rows[3].update(eccentricity=0.05, mean_anomaly_deg=180.0, mean_motion_rev_per_day=15.5)
    from orbit_correction import truth_prop as T

    fit = T.fit_samples_from_elsets(pa.Table.from_pylist(rows[3:4], schema=w.schema),
                                    T.load_truth_config())[0]
    assert not fit.status.ok
    w = pa.Table.from_pylist(rows, schema=w.schema)
    one = w.slice(3, 1)
    assert C.sgp4_epoch_states(one)[0].trajectory is not None
    _, _, runner, man = _run(tmp_path, elsets, sw, rcfg, gcfg, windows=w)
    assert runner.jobs == 7 and man["status"]["fit_samples_unavailable"] == 1
    rec = [r for r in G.read_logs(tmp_path / "run") if r["window_index"] == 3][0]
    assert rec["truth_status"] == "not_run" and rec["status"] == "fit_samples_unavailable"


def test_epoch_state_rule_passes_no_fit_samples(tmp_path, elsets, sw, rcfg, gcfg, tcfg):
    w = _windows(elsets, gcfg)
    dt = R.horizon_grid(rcfg)
    dt = dt[dt <= DAY]
    seen = []

    def runner(jobs):
        seen.extend(j.fit_samples for j in jobs)
        fake = FakeTruth(w, dt)
        out = []
        for j in jobs:
            row = fake.rows[(j.initial.norad_id, float(j.initial.t_tt_s[0]))]
            el = pa.Table.from_pylist([{c: row[c] for c in C.REQUIRED_COLUMNS}],
                                      schema=fake.schema)
            sg = C.propagate_sgp4_from_epoch(el, dt)[0]
            out.append(_truth_with(sg, np.zeros((dt.size, 3))))
        return out

    old_rule = dataclasses.replace(tcfg, start_rule="preserve_sgp4_mean_motion")
    _, _, _, man = _run(tmp_path, elsets, sw, rcfg, gcfg, runner=runner, truth_cfg=old_rule)
    assert seen and all(f is None for f in seen) and man["status"] == {"ok": 8}


class CrashingRunner(FakeTruth):
    """Raises BrokenProcessPool on the first ``n_crash`` calls for chunk 1's windows."""

    def __init__(self, windows, dt, n_crash):
        super().__init__(windows, dt)
        self.n_crash = n_crash
        self.restarts = 0
        self.crash_epochs = {r["epoch_tt_s"] for r in windows.to_pylist()[3:6]}

    def restart(self):
        self.restarts += 1

    def __call__(self, jobs):
        from concurrent.futures.process import BrokenProcessPool

        if self.n_crash and float(jobs[0].initial.t_tt_s[0]) in self.crash_epochs:
            self.n_crash -= 1
            raise BrokenProcessPool("a worker died")
        return super().__call__(jobs)


def test_broken_pool_retries_the_chunk_and_counts_the_crash(tmp_path, elsets, sw, rcfg, gcfg):
    w = _windows(elsets, gcfg)
    dt = R.horizon_grid(rcfg)
    dt = dt[dt <= DAY]
    runner = CrashingRunner(w, dt, n_crash=1)
    _, _, _, man = _run(tmp_path, elsets, sw, rcfg, gcfg, runner=runner, restart=runner.restart)
    assert runner.restarts == 1 and man["worker_pool_crashes"] == 1
    assert man["worker_pool_crash_records"][0]["chunk"] == 1
    assert man["status"]["ok"] == 8 and len(G.read_logs(tmp_path / "run")) == 8
    # a chunk that keeps breaking stops the run after three attempts, all counted
    stuck = CrashingRunner(w, dt, n_crash=10)
    with pytest.raises(RuntimeError, match="3 attempts"):
        _run(tmp_path / "b", elsets, sw, rcfg, gcfg, runner=stuck, restart=stuck.restart)
    crashes = (tmp_path / "b" / "run" / "crashes.jsonl").read_text().splitlines()
    assert len(crashes) == 3 and stuck.restarts == 2


# ----------------------------------------------------------------- real truth (datagen)


@pytest.fixture(scope="module")
def sw_copy(tmp_path_factory):
    from orbit_correction import truth_prop as T

    d = tmp_path_factory.mktemp("sw")
    return T.write_space_weather_copy(
        FIXTURES / "sw_celestrak_2024_extract.txt", d / "sw_orekit_2024-02-01_2024-11-15.txt",
        F.utc_to_tt_seconds("2024-02-01T00:00:00"), F.utc_to_tt_seconds("2024-11-15T00:00:00"))


@pytest.mark.datagen
@pytest.mark.skipif(not HAVE_OREKIT, reason="needs `uv sync --group datagen` (Orekit)")
def test_window_past_the_single_state_limit_is_kept_unwrapped(elsets, sw, rcfg, sw_copy):
    """The independent review's case: 350 km perigee, B* 0.01, 2024-06-15, real truth.

    Under the single-state limit (before 2026-11-25) this window was ``rtn_domain``:
    |du| passes pi/2 at about 6.7 days. With du unwrapped along the window it is kept,
    its targets go past a quarter orbit, and they invert to the truth positions.
    """
    from orbit_correction import truth_prop as T

    MU72, RE72 = 398600.8, 6378.135
    e, hp = 0.0003, 350.0
    a = (RE72 + hp) / (1 - e)
    n = math.sqrt(MU72 / a**3) * DAY / (2 * math.pi)
    el = _one(elsets, 44714, eccentricity=e, bstar_per_earth_radii=0.01,
              mean_motion_rev_per_day=n, perigee_alt_km=hp)
    grid = R.horizon_grid(rcfg)
    sg = C.propagate_sgp4_from_epoch(el, grid)[0]
    start = C.sgp4_epoch_states(el)[0].trajectory
    tcfg = T.load_truth_config()
    fit = T.fit_samples_from_elsets(el, tcfg)[0].trajectory
    tr = T.propagate_truth(start, 0.01, el["epoch_tt_s"][0].as_py() + grid, tcfg, sw_copy,
                           fit_samples=fit)
    assert sg.status.ok and tr.status == "ok"
    res = G.window_result(el.to_pylist()[0], sg, tr, sw, "train", grid, rcfg)
    assert res.status == "ok" and res.altitude_band == "250-400"
    cheap = sg.trajectory
    rc = np.linalg.norm(cheap.r_gcrf_km, axis=1)
    du = res.table.column("target_t_km").to_numpy() / rc
    assert np.abs(du).max() > math.pi / 2  # past the single-state limit
    assert np.abs(np.diff(du)).max() < 0.5  # far from a continuity break
    rtn = np.column_stack([res.table.column(c).to_numpy()
                           for c in ("target_r_km", "target_t_km", "target_n_km")])
    back = R.states_from_rtn(cheap.r_gcrf_km, cheap.v_gcrf_km_s, rtn)
    assert np.max(np.linalg.norm(back - tr.trajectory.r_gcrf_km, axis=1)) < 1e-6
    with pytest.raises(R.RTNDomainError, match="along-track"):
        R.rtn_from_states(cheap.r_gcrf_km, cheap.v_gcrf_km_s, tr.trajectory.r_gcrf_km)


@pytest.mark.datagen
@pytest.mark.skipif(not HAVE_OREKIT, reason="needs `uv sync --group datagen` (Orekit)")
def test_run_with_spawned_truth_workers(tmp_path, elsets, sw, rcfg, gcfg, sw_copy):
    """Three windows through the worker pool: rows, log, and the parent never starts a JVM."""
    from orbit_correction import truth_prop as T

    # windows with positive B* (IRAS's negative B* is one the catalog filters drop)
    w = _windows(elsets, gcfg).take([0, 2, 4])
    dt = R.horizon_grid(rcfg)
    dt = dt[dt <= DAY / 4]
    tcfg = T.load_truth_config()
    before = T.jvm_started()  # another test in this session may have started one
    with G.TruthWorkerPool(tcfg, sw_copy, n_workers=2) as pool:
        strata = G.window_strata(w, sw, rcfg, gcfg)
        settings = G.RunSettings(run_id="real", label="baseline", chunk_windows=2, n_workers=2)
        man = G.run_generation(w, tmp_path / "real", settings, dt_s=dt, residual_cfg=rcfg,
                               space_weather=sw, strata=strata, truth_runner=pool.run,
                               truth_cfg=tcfg, restart_runner=pool.restart)
    assert man["status"] == {"ok": 3} and man["n_rows"] == 3 * dt.size
    assert T.jvm_started() == before  # the workers' JVMs are theirs, not this process's
    log = G.read_logs(tmp_path / "real")
    # the default fitted start: every job fitted and converged
    assert all(r["truth_wall_s"] > 0 and r["fit_status"] == "converged" for r in log)
    assert man["fit_status"] == {"converged": 3} and man["worker_pool_crashes"] == 0
