"""Tests for ``orbit_correction.catalog``.

Fixtures: ``celestrak_gp_sample.json``/``.csv`` and ``satcat_sample.csv`` are cut
from CelesTrak downloads (validation-case objects plus one object per filter
case). ``spacetrack_gp_history_synthetic.json`` is made up, in Space-Track's
layout; no Space-Track data is used.
"""

import gzip
import json

import numpy as np
import pyarrow as pa
import pytest

from orbit_correction import catalog as C
from orbit_correction.frames import utc_to_tt_seconds

from .conftest import REPO

VALIDATION_IDS = {24946, 22675, 26998, 22236, 13777, 2828, 19826, 36123, 7219, 26207}


@pytest.fixture
def gp(fixtures):
    return C.load_celestrak_omm(fixtures / "celestrak_gp_sample.json")


@pytest.fixture
def gp_sat(gp, fixtures):
    return C.attach_satcat(gp, C.load_satcat(fixtures / "satcat_sample.csv"))


@pytest.fixture
def history(fixtures):
    return C.load_spacetrack_gp_history(fixtures / "spacetrack_gp_history_synthetic.json")


# ---------------------------------------------------------------- loading


def test_celestrak_json_and_csv_agree(gp, fixtures):
    csv_table = C.load_celestrak_omm(fixtures / "celestrak_gp_sample.csv")
    assert csv_table.schema == C.ELEMENT_SET_SCHEMA
    assert csv_table.equals(gp)
    assert gp.num_rows == 17
    assert VALIDATION_IDS <= set(gp.column("norad_id").to_pylist())


def test_epoch_converted_to_contract_time(gp):
    row = gp.to_pylist()[0]
    assert row["epoch_tt_s"] == pytest.approx(utc_to_tt_seconds(row["epoch_utc"]), abs=1e-6)
    assert set(gp.column("source").to_pylist()) == {"celestrak_omm"}


def test_derived_altitudes_match_satcat(gp, fixtures):
    """Kepler altitudes from mean motion agree with CelesTrak's SATCAT values (to ~10 km)."""
    import csv

    sat = {int(r["NORAD_CAT_ID"]): r for r in csv.DictReader(open(fixtures / "satcat_sample.csv"))}
    for row in gp.to_pylist():
        s = sat[row["norad_id"]]
        assert row["perigee_alt_km"] == pytest.approx(float(s["PERIGEE"]), abs=15)
        assert row["apogee_alt_km"] == pytest.approx(float(s["APOGEE"]), abs=15)


def test_spacetrack_layout_with_string_values(history, fixtures, tmp_path):
    assert history.num_rows == 12
    assert history.schema == C.ELEMENT_SET_SCHEMA
    assert set(history.column("source").to_pylist()) == {"spacetrack_gp_history"}
    first = history.to_pylist()[0]
    assert first["norad_id"] == 90001 and first["bstar_per_earth_radii"] == pytest.approx(1.23e-4)
    # gzipped download and several files at once
    gz = tmp_path / "part.json.gz"
    with gzip.open(gz, "wt") as f:
        f.write((fixtures / "spacetrack_gp_history_synthetic.json").read_text())
    both = C.load_spacetrack_gp_history(
        [fixtures / "spacetrack_gp_history_synthetic.json", gz]
    )
    assert both.num_rows == 24


@pytest.mark.parametrize("key, value", [("MEAN_MOTION", ""), ("EPOCH", None), ("BSTAR", "abc")])
def test_unreadable_record_raises(tmp_path, fixtures, key, value):
    recs = json.loads((fixtures / "spacetrack_gp_history_synthetic.json").read_text())
    recs[3][key] = value
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(recs))
    with pytest.raises(C.CatalogFormatError, match=key):
        C.load_spacetrack_gp_history(p)


def test_attach_satcat_keeps_order_and_nulls_unknown(gp, fixtures):
    sat = C.load_satcat(fixtures / "satcat_sample.csv")
    unknown = pa.table(
        {"norad_id": [999999]}, schema=pa.schema([pa.field("norad_id", pa.int64())])
    )
    joined = C.attach_satcat(gp, sat)
    assert joined.column("norad_id").equals(gp.column("norad_id"))
    rows = {r["norad_id"]: r for r in joined.to_pylist()}
    assert rows[7219]["object_type"] == "R/B"
    assert rows[26998]["ops_status"] == "+"
    assert rows[22236]["ops_status"] is None  # blank SATCAT status stays null
    lone = C.attach_satcat(unknown, sat)
    assert lone.column("object_type").to_pylist() == [None]


# ---------------------------------------------------------------- filters


def test_config_file_matches_code_defaults():
    cfg = C.load_filter_config(REPO / "config" / "catalog_filters.toml")
    default = C.CatalogFilterConfig()
    for name in ("perigee_min_km", "apogee_max_km", "bstar_min_exclusive", "bstar_max",
                 "ephemeris_types", "object_types", "exclude_ops_status",
                 "sma_increase_max_km", "sma_drop_drag_multiple", "sma_drop_floor_km",
                 "inclination_jump_max_deg", "name_rule_column", "name_rule_object_types"):
        assert getattr(cfg, name) == getattr(default, name), name
    assert cfg.status == "approved 2026-11-23"
    assert {"STARLINK", "FLOCK", "LEMUR"} <= set(cfg.exclude_name_patterns)


def test_approved_values():
    cfg = C.load_filter_config(REPO / "config" / "catalog_filters.toml")
    assert (cfg.perigee_min_km, cfg.apogee_max_km) == (250.0, 2000.0)
    assert (cfg.bstar_min_exclusive, cfg.bstar_max) == (0.0, 0.01)
    assert cfg.object_types == ("PAY", "R/B", "DEB") and cfg.ephemeris_types == (0,)


def _with(t, **cols):
    for name, values in cols.items():
        t = t.set_column(t.column_names.index(name), name, pa.array(values))
    return t


def test_name_rule_catches_flock_and_lemur(gp_sat):
    cfg = C.load_filter_config(REPO / "config" / "catalog_filters.toml")
    names = ["FLOCK 4P-1", "LEMUR-2-JOHANLORAN", "Lemur-2-x", "SYNTH FLOCKISH"]
    t = _with(gp_sat.slice(0, 4), satcat_name=names, object_type=["PAY"] * 4)
    rule = next(r for r in C.DEFAULT_RULES if r.name == "not_maneuvering_constellation")
    assert rule.keep(t, cfg).tolist() == [False, False, False, True]


def test_name_rule_reads_satcat_names_of_payloads_only(gp_sat):
    """Decided 2026-11-24: SATCAT names, payloads only; constellation debris is kept."""
    cfg = C.load_filter_config(REPO / "config" / "catalog_filters.toml")
    assert cfg.name_rule_column == "satcat_name" and cfg.name_rule_object_types == ("PAY",)
    rule = next(r for r in C.DEFAULT_RULES if r.name == "not_maneuvering_constellation")
    t = _with(
        gp_sat.slice(0, 5),
        satcat_name=["IRIDIUM 33 DEB", "STARLINK-1008", "ORBCOMM FM5 R/B", "OBJECT A",
                     "STARLINK-30000"],
        object_name=["IRIDIUM 33 DEB", "STARLINK-1008", "ORBCOMM FM5 R/B", "OBJECT A",
                     "TBA - TO BE ASSIGNED"],
        object_type=["DEB", "PAY", "R/B", "PAY", "PAY"],
    )
    # debris and rocket bodies kept; payloads go by their SATCAT name, so an early
    # element set still named "TBA" goes when the object is now STARLINK-30000
    assert rule.keep(t, cfg).tolist() == [True, False, True, True, False]
    # the element set's own name and every type: the earlier behaviour
    old = C.CatalogFilterConfig(
        exclude_name_patterns=cfg.exclude_name_patterns, name_rule_column="object_name",
        name_rule_object_types=("PAY", "R/B", "DEB"),
    )
    assert rule.keep(t, old).tolist() == [False, False, False, True, True]
    with pytest.raises(ValueError, match="name_rule_column"):
        rule.keep(t, C.CatalogFilterConfig(exclude_name_patterns=("X",), name_rule_column="x"))


def test_default_filters_on_fixture(gp_sat):
    cfg = C.load_filter_config(REPO / "config" / "catalog_filters.toml")
    res = C.apply_filters(gp_sat, cfg)
    kept = set(res.kept.column("norad_id").to_pylist())
    assert kept == {2828, 7219, 19826, 22236, 22675, 26207, 36123}
    assert res.n_input == 17
    assert sum(res.dropped_at_rule.values()) == 17 - len(kept)


def _rejected(table, rule_name, **cfg_kwargs):
    rule = next(r for r in C.DEFAULT_RULES if r.name == rule_name)
    mask = rule.keep(table, C.CatalogFilterConfig(**cfg_kwargs))
    return set(np.asarray(table.column("norad_id").to_pylist())[~mask].tolist())


def test_each_rule_rejects_the_intended_objects(gp_sat):
    pats = ("STARLINK", "ONEWEB", "IRIDIUM")
    assert _rejected(gp_sat, "leo_perigee") == {44753}  # STARLINK-1048 at ~227 km
    assert _rejected(gp_sat, "leo_apogee") == {19548, 26113}  # TDRS 3 (GEO), IMAGE (HEO)
    # negative or zero B*: IRAS (a validation object!), CALSPHERE 2, TDRS 3, IMAGE
    assert _rejected(gp_sat, "bstar_positive") == {13777, 902, 19548, 26113}
    assert _rejected(gp_sat, "bstar_not_extreme") == {44058}  # ONEWEB-0010, B* 0.012
    # operational payloads, including TIMED, which has no propulsion
    assert {25544, 26998, 44714} <= _rejected(gp_sat, "not_operational_payload")
    assert 22236 not in _rejected(gp_sat, "not_operational_payload")
    assert _rejected(gp_sat, "not_maneuvering_constellation", exclude_name_patterns=pats) == {
        44714, 44753, 44058, 24946,
    }
    assert _rejected(gp_sat, "object_type", object_types=("PAY", "R/B")) == {26207}
    assert _rejected(gp_sat, "sgp4_ephemeris_type") == set()


def test_ephemeris_type_rule_rejects_sgp4xp_and_missing(gp_sat):
    et = gp_sat.column("ephemeris_type").to_pylist()
    et[0], et[1] = 4, None
    i = gp_sat.column_names.index("ephemeris_type")
    edited = gp_sat.set_column(i, "ephemeris_type", pa.array(et, pa.int64()))
    ids = gp_sat.column("norad_id").to_pylist()
    assert _rejected(edited, "sgp4_ephemeris_type") == {ids[0], ids[1]}


def test_rules_needing_satcat_raise_without_it(gp):
    with pytest.raises(KeyError, match="object_type"):
        C.apply_filters(gp, C.CatalogFilterConfig())


def test_maneuver_detection_on_history(history):
    assert C.maneuver_flagged_ids(history, 1.0) == {90002}
    assert C.maneuver_flagged_ids(history, 10.0) == set()
    rule = [r for r in C.DEFAULT_RULES if r.name == "no_sma_increase"]
    res = C.apply_filters(history, C.CatalogFilterConfig(), rules=rule)
    assert set(res.kept.column("norad_id").to_pylist()) == {90001}
    assert res.dropped_at_rule == {"no_sma_increase": 6}


def test_maneuver_detection_orders_by_epoch(history):
    shuffled = history.take(pa.array(np.random.default_rng(1).permutation(history.num_rows)))
    assert C.maneuver_flagged_ids(shuffled, 1.0) == {90002}


def test_unknown_config_key_rejected(tmp_path):
    p = tmp_path / "f.toml"
    p.write_text("[catalog_filters]\nperigee_min_km = 200.0\nperige_max = 3\n")
    with pytest.raises(ValueError, match="perige_max"):
        C.load_filter_config(p)


# ---------------------------------------------------------------- element-history maneuver rules


def _synthetic_history(
    nid=91001, days=20, a0_km=6778.0, decay_km_per_day=0.3, events=(), noise_km=0.0, seed=0
):
    """One object's daily element sets under drag, with optional maneuvers.

    The mean motion follows ``a(t) = a0 - decay * t`` and MEAN_MOTION_DOT is the
    value consistent with that decay, as an element-set fit would report it.
    ``events`` are ``(day, kind, size)``: ``("sma", -3.0)`` lowers the orbit by
    3 km from that day on, ``("inc", 0.1)`` changes the inclination by 0.1 deg,
    ``("rate", 3.0)`` multiplies the decay rate from that day on (a storm).
    """
    rng = np.random.default_rng(seed)
    recs = []
    a, inc, rate = a0_km, 51.6, decay_km_per_day
    for d in range(days):
        for day, kind, size in events:
            if day == d:
                if kind == "sma":
                    a += size
                elif kind == "inc":
                    inc += size
                elif kind == "rate":
                    rate *= size
        a_fit = a + noise_km * rng.standard_normal()
        n = np.sqrt(C.MU_WGS72_KM3_S2 / a_fit**3) * 86400 / (2 * np.pi)
        # da/dt = -(2/3)(a/n) dn/dt  ->  MEAN_MOTION_DOT = dn/dt / 2
        ndot_half = rate * 1.5 * n / a_fit / 2
        recs.append({
            "NORAD_CAT_ID": nid, "OBJECT_NAME": "SYNTH", "OBJECT_ID": "2000-000Z",
            "EPOCH": f"2024-03-{d + 1:02d}T00:00:00", "MEAN_MOTION": n, "ECCENTRICITY": 0.001,
            "INCLINATION": inc, "RA_OF_ASC_NODE": 10.0, "ARG_OF_PERICENTER": 20.0,
            "MEAN_ANOMALY": 30.0, "BSTAR": 1e-4, "MEAN_MOTION_DOT": ndot_half,
            "MEAN_MOTION_DDOT": 0.0, "EPHEMERIS_TYPE": 0,
        })
        a -= rate
    return C.records_to_table(recs, "synthetic")


def _history_flags(table):
    cfg = C.load_filter_config(REPO / "config" / "catalog_filters.toml")
    rules = [r for r in C.DEFAULT_RULES if r.name in
             ("no_sma_increase", "no_sma_unexplained_drop", "no_inclination_jump")]
    return C.apply_filters(table, cfg, rules=rules).rejected_by_rule


def test_drag_decay_rate_inverts_mean_motion_dot():
    t = _synthetic_history(decay_km_per_day=0.3)
    rate = C.drag_decay_rate_km_per_day(
        C._col(t, "semimajor_axis_km"), C._col(t, "mean_motion_rev_per_day"),
        C._col(t, "mean_motion_dot"),
    )
    np.testing.assert_allclose(rate, 0.3, rtol=1e-6)
    assert C.drag_decay_rate_km_per_day(
        np.array([6778.0]), np.array([15.5]), np.array([np.nan])
    ).tolist() == [0.0]


@pytest.mark.parametrize("decay", [0.0, 0.3, 2.0])
def test_pure_drag_decay_is_not_flagged(decay):
    flags = _history_flags(_synthetic_history(decay_km_per_day=decay, noise_km=0.1))
    assert flags == {"no_sma_increase": 0, "no_sma_unexplained_drop": 0,
                     "no_inclination_jump": 0}


def test_storm_speeding_up_decay_is_not_flagged():
    t = _synthetic_history(decay_km_per_day=0.5, events=[(10, "rate", 3.0)])
    assert _history_flags(t)["no_sma_unexplained_drop"] == 0


def test_lowering_burn_is_flagged():
    t = _synthetic_history(decay_km_per_day=0.3, events=[(10, "sma", -3.0)])
    flags = _history_flags(t)
    assert flags["no_sma_unexplained_drop"] == t.num_rows  # the whole object goes
    assert flags["no_sma_increase"] == 0 and flags["no_inclination_jump"] == 0
    assert C.sma_drop_flagged_ids(t, 3.0, 1.0) == {91001}
    # below the floor plus the drag allowance it is missed, as documented
    small = _synthetic_history(decay_km_per_day=0.3, events=[(10, "sma", -0.5)])
    assert C.sma_drop_flagged_ids(small, 3.0, 1.0) == set()


def test_plane_change_is_flagged():
    t = _synthetic_history(events=[(7, "inc", 0.1)])
    flags = _history_flags(t)
    assert flags["no_inclination_jump"] == t.num_rows
    assert flags["no_sma_increase"] == 0 and flags["no_sma_unexplained_drop"] == 0
    assert C.inclination_jump_flagged_ids(t, 0.05) == {91001}
    assert C.inclination_jump_flagged_ids(t, 0.2) == set()


def test_history_rules_keep_objects_apart_and_ignore_order():
    clean = _synthetic_history(nid=91001)
    burn = _synthetic_history(nid=91002, events=[(5, "sma", -4.0)])
    both = pa.concat_tables([clean, burn])
    shuffled = both.take(pa.array(np.random.default_rng(3).permutation(both.num_rows)))
    assert C.sma_drop_flagged_ids(shuffled, 3.0, 1.0) == {91002}
    # the jump between the last set of one object and the first of the next is not a pair
    lowered = _synthetic_history(nid=91003, a0_km=6700.0)
    assert C.sma_drop_flagged_ids(pa.concat_tables([clean, lowered]), 3.0, 1.0) == set()
