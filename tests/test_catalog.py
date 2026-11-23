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
                 "sma_increase_max_km", "status"):
        assert getattr(cfg, name) == getattr(default, name), name
    assert cfg.status == "pending approval"
    assert "STARLINK" in cfg.exclude_name_patterns


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
