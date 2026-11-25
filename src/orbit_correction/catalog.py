"""Element-set and SATCAT loading, and the named filters that define the v1 catalog.

Why this module exists
----------------------
Everything downstream starts from SGP4 mean element sets: CelesTrak's current
GP data (OMM JSON or CSV) for development and tests, and Space-Track's
``gp_history`` for training and test epochs. This module reads both into one
columnar table (:data:`ELEMENT_SET_SCHEMA`), joins CelesTrak SATCAT metadata,
and applies the filters that restrict the study to its locked regime: LEO,
non-maneuvering objects, usable drag terms.

Filters are instruments
-----------------------
Each filter is a named rule (:class:`FilterRule`) driven by
:class:`CatalogFilterConfig`. The default values live in
``config/catalog_filters.toml`` and are logged in ``docs/changelog.md``;
changing one is an instrument change. :func:`apply_filters` reports how many
rows each rule rejects, so the catalog's construction is auditable (dropped
rows are a metric, never silent).

Conventions
-----------
* Element sets are SGP4 mean elements (TLE/OMM conventions): mean motion in
  rev/day (Kozai), angles in degrees, B* in 1/Earth radii.
* Epochs are UTC in the source; the table keeps the ISO string and adds the
  contract time ``epoch_tt_s`` (TT seconds since J2000.0).
* Perigee and apogee altitudes are derived with Kepler's third law from the
  mean motion, WGS-72 ``mu = 398600.8 km^3/s^2`` and equatorial radius
  6378.135 km (the SGP4 constants; Space-Track computes its PERIAPSIS and
  APOAPSIS the same way). They are good enough for regime filters, not for
  physics.

Space-Track data is never committed: tests use synthetic records in its layout.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from orbit_correction.frames import elset_epoch_to_tt_seconds

# WGS-72 constants used by SGP4.
MU_WGS72_KM3_S2 = 398600.8
RE_WGS72_KM = 6378.135

#: The table every loader returns. Derived columns come last.
ELEMENT_SET_SCHEMA = pa.schema(
    [
        pa.field("norad_id", pa.int64(), nullable=False),
        pa.field("object_name", pa.string()),
        pa.field("intl_designator", pa.string()),
        pa.field("epoch_utc", pa.string(), nullable=False),
        pa.field("epoch_tt_s", pa.float64(), nullable=False),
        pa.field("mean_motion_rev_per_day", pa.float64(), nullable=False),
        pa.field("eccentricity", pa.float64(), nullable=False),
        pa.field("inclination_deg", pa.float64(), nullable=False),
        pa.field("raan_deg", pa.float64(), nullable=False),
        pa.field("arg_perigee_deg", pa.float64(), nullable=False),
        pa.field("mean_anomaly_deg", pa.float64(), nullable=False),
        pa.field("bstar_per_earth_radii", pa.float64(), nullable=False),
        pa.field("mean_motion_dot", pa.float64()),
        pa.field("mean_motion_ddot", pa.float64()),
        pa.field("ephemeris_type", pa.int64()),
        pa.field("classification", pa.string()),
        pa.field("element_set_no", pa.int64()),
        pa.field("rev_at_epoch", pa.int64()),
        pa.field("source", pa.string(), nullable=False),
        pa.field("semimajor_axis_km", pa.float64(), nullable=False),
        pa.field("perigee_alt_km", pa.float64(), nullable=False),
        pa.field("apogee_alt_km", pa.float64(), nullable=False),
    ]
)

# OMM keyword -> (table column, converter). Shared by CelesTrak and Space-Track.
_OMM_FIELDS: dict[str, tuple[str, Callable[[Any], Any]]] = {}


def _opt_float(x: Any) -> float | None:
    if x is None or (isinstance(x, str) and not x.strip()):
        return None
    return float(x)


def _opt_int(x: Any) -> int | None:
    if x is None or (isinstance(x, str) and not x.strip()):
        return None
    return int(float(x))


def _opt_str(x: Any) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s or None


def _req(conv: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def f(x: Any) -> Any:
        v = conv(x)
        if v is None:
            raise ValueError("required field is empty")
        return v

    return f


_OMM_FIELDS.update(
    {
        "NORAD_CAT_ID": ("norad_id", _req(_opt_int)),
        "OBJECT_NAME": ("object_name", _opt_str),
        "OBJECT_ID": ("intl_designator", _opt_str),
        "EPOCH": ("epoch_utc", _req(_opt_str)),
        "MEAN_MOTION": ("mean_motion_rev_per_day", _req(_opt_float)),
        "ECCENTRICITY": ("eccentricity", _req(_opt_float)),
        "INCLINATION": ("inclination_deg", _req(_opt_float)),
        "RA_OF_ASC_NODE": ("raan_deg", _req(_opt_float)),
        "ARG_OF_PERICENTER": ("arg_perigee_deg", _req(_opt_float)),
        "MEAN_ANOMALY": ("mean_anomaly_deg", _req(_opt_float)),
        "BSTAR": ("bstar_per_earth_radii", _req(_opt_float)),
        "MEAN_MOTION_DOT": ("mean_motion_dot", _opt_float),
        "MEAN_MOTION_DDOT": ("mean_motion_ddot", _opt_float),
        "EPHEMERIS_TYPE": ("ephemeris_type", _opt_int),
        "CLASSIFICATION_TYPE": ("classification", _opt_str),
        "ELEMENT_SET_NO": ("element_set_no", _opt_int),
        "REV_AT_EPOCH": ("rev_at_epoch", _opt_int),
    }
)


class CatalogFormatError(ValueError):
    """A record could not be read as an OMM element set."""


def records_to_table(records: Iterable[Mapping[str, Any]], source: str) -> pa.Table:
    """Convert OMM-keyword records to :data:`ELEMENT_SET_SCHEMA`.

    The one conversion every loader goes through (CelesTrak OMM, Space-Track
    ``gp_history``, single records built by scripts and tests), so an element set
    gives the same row whichever way it arrives. ``source`` fills the ``source``
    column. Records are mappings from OMM keywords (``NORAD_CAT_ID``, ``EPOCH``,
    ``MEAN_MOTION``...) to values, as strings or numbers; extra keywords are ignored.

    Any record missing a required element raises (with its position): an
    element set that cannot be read is an error to look at, not a row to drop.
    """
    cols: dict[str, list[Any]] = {col: [] for col, _ in _OMM_FIELDS.values()}
    for n, rec in enumerate(records):
        for key, (col, conv) in _OMM_FIELDS.items():
            try:
                cols[col].append(conv(rec.get(key)))
            except (TypeError, ValueError) as exc:
                raise CatalogFormatError(
                    f"record {n} (NORAD {rec.get('NORAD_CAT_ID')!r}): bad {key}={rec.get(key)!r}"
                ) from exc
    count = len(cols["norad_id"])
    n_rad_s = np.asarray(cols["mean_motion_rev_per_day"], dtype=np.float64) * 2 * np.pi / 86400.0
    ecc = np.asarray(cols["eccentricity"], dtype=np.float64)
    with np.errstate(divide="ignore"):
        a = np.cbrt(MU_WGS72_KM3_S2 / n_rad_s**2)
    epochs = cols["epoch_utc"]
    epoch_tt = (
        np.atleast_1d(elset_epoch_to_tt_seconds(np.asarray(epochs, dtype=str)))
        if count
        else np.zeros(0)
    )
    arrays = dict(cols)
    arrays["epoch_tt_s"] = epoch_tt
    arrays["source"] = [source] * count
    arrays["semimajor_axis_km"] = a
    arrays["perigee_alt_km"] = a * (1 - ecc) - RE_WGS72_KM
    arrays["apogee_alt_km"] = a * (1 + ecc) - RE_WGS72_KM
    return pa.table(
        {f.name: pa.array(arrays[f.name], type=f.type) for f in ELEMENT_SET_SCHEMA},
        schema=ELEMENT_SET_SCHEMA,
    )


#: Former private name, kept as an alias for callers not yet moved
#: (``truth_prop._omm_record_elset``). New code calls :func:`records_to_table`.
_records_to_table = records_to_table


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise CatalogFormatError(f"{path}: expected a JSON array of OMM records")
    return data


def load_celestrak_omm(path: str | Path) -> pa.Table:
    """Load CelesTrak GP data in OMM JSON (``.json``) or OMM CSV (``.csv``) form.

    Source URL pattern: ``https://celestrak.org/NORAD/elements/gp.php?...&FORMAT=json``
    (or ``csv``). Returns a table with :data:`ELEMENT_SET_SCHEMA`.
    """
    path = Path(path)
    if path.suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            records: list[dict[str, Any]] = list(csv.DictReader(f))
    else:
        records = _read_json_records(path)
    return records_to_table(records, "celestrak_omm")


def load_spacetrack_gp_history(paths: str | Path | Iterable[str | Path]) -> pa.Table:
    """Load Space-Track ``gp_history`` downloads (JSON, optionally ``.json.gz``).

    Space-Track returns OMM keywords with string values; extra keywords
    (``TLE_LINE1``, ``GP_ID``...) are ignored. Several files are concatenated.
    Returns a table with :data:`ELEMENT_SET_SCHEMA`.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    tables = [
        records_to_table(_read_json_records(Path(p)), "spacetrack_gp_history") for p in paths
    ]
    if not tables:
        return ELEMENT_SET_SCHEMA.empty_table()
    return pa.concat_tables(tables)


# --------------------------------------------------------------------------------------
# SATCAT
# --------------------------------------------------------------------------------------

SATCAT_SCHEMA = pa.schema(
    [
        pa.field("norad_id", pa.int64(), nullable=False),
        pa.field("satcat_name", pa.string()),
        pa.field("object_type", pa.string()),
        pa.field("ops_status", pa.string()),
        pa.field("owner", pa.string()),
        pa.field("launch_date", pa.string()),
        pa.field("decay_date", pa.string()),
        pa.field("rcs_m2", pa.float64()),
    ]
)


def load_satcat(path: str | Path) -> pa.Table:
    """Load CelesTrak's SATCAT CSV (``https://celestrak.org/pub/satcat.csv``).

    ``object_type`` is PAY, R/B, DEB or UNK. ``ops_status`` is CelesTrak's
    operational status code (``+`` operational, ``-`` non-operational, ``P``
    partially operational, ``B`` backup, ``S`` spare, ``X`` extended mission,
    ``D`` decayed, ``?`` unknown); an empty code is kept as null. Note that the
    status is the object's status *now*, not at a historical epoch.
    """
    cols: dict[str, list[Any]] = {f.name: [] for f in SATCAT_SCHEMA}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cols["norad_id"].append(int(row["NORAD_CAT_ID"]))
            cols["satcat_name"].append(_opt_str(row.get("OBJECT_NAME")))
            cols["object_type"].append(_opt_str(row.get("OBJECT_TYPE")))
            cols["ops_status"].append(_opt_str(row.get("OPS_STATUS_CODE")))
            cols["owner"].append(_opt_str(row.get("OWNER")))
            cols["launch_date"].append(_opt_str(row.get("LAUNCH_DATE")))
            cols["decay_date"].append(_opt_str(row.get("DECAY_DATE")))
            cols["rcs_m2"].append(_opt_float(row.get("RCS")))
    return pa.table(cols, schema=SATCAT_SCHEMA)


def attach_satcat(elsets: pa.Table, satcat: pa.Table) -> pa.Table:
    """Left-join SATCAT columns onto element sets by NORAD ID, keeping row order.

    Element sets with no SATCAT entry get nulls (and then fail the object-type
    rule rather than passing silently).
    """
    keep = [c for c in SATCAT_SCHEMA.names if c != "norad_id"]
    idx = {int(k): i for i, k in enumerate(satcat.column("norad_id").to_pylist())}
    rows = [idx.get(int(k)) for k in elsets.column("norad_id").to_pylist()]
    take = pa.array(rows, type=pa.int64())
    out = elsets
    for name in keep:
        out = out.append_column(satcat.schema.field(name), satcat.column(name).take(take))
    return out


# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogFilterConfig:
    """Parameters of the catalog filters. Values are instruments (see changelog).

    Attributes:
        perigee_min_km: LEO rule, lowest allowed perigee altitude.
        apogee_max_km: LEO rule, highest allowed apogee altitude.
        bstar_min_exclusive: B* must be strictly greater (drops negative and,
            at 0, zero B*: the truth propagator derives area-to-mass from B*).
        bstar_max: B* must be at most this (1/ER); larger values are usually
            poor fits or objects about to decay.
        ephemeris_types: allowed OMM EPHEMERIS_TYPE values (0 = SGP4; SGP4-XP
            element sets are not SGP4 mean elements and must not be fed to SGP4).
        object_types: allowed SATCAT object types.
        exclude_ops_status: payloads with one of these current status codes are
            treated as able to maneuver and removed.
        exclude_name_patterns: regular expressions (case-insensitive, matched at
            the start of the name) for constellations known to maneuver.
        name_rule_column: the name the pattern rule reads: ``"satcat_name"`` (the
            object's current CelesTrak SATCAT name, from :func:`attach_satcat`;
            decided 2026-11-24) or ``"object_name"`` (the element set's own name,
            the earlier behaviour). Bulk TLE histories carry no names, and an
            element set's name can differ from the object's later name, so the
            SATCAT name gives one name per object across its whole history.
        name_rule_object_types: object types the name rule applies to (decided
            2026-11-24: payloads only). Debris and rocket bodies named after a
            constellation (``IRIDIUM 33 DEB``) cannot maneuver and are kept.
        sma_increase_max_km: across an object's element history, a rise in mean
            semimajor axis larger than this between consecutive element sets
            flags a maneuver (drag only lowers it); the whole object is dropped.
        sma_drop_drag_multiple, sma_drop_floor_km: a fall in mean semimajor axis
            between consecutive element sets is flagged as a maneuver when it
            exceeds ``sma_drop_drag_multiple`` times the drag decay implied by the
            element sets' mean-motion derivative over the gap, plus
            ``sma_drop_floor_km`` (see :func:`sma_drop_flagged_ids`).
        inclination_jump_max_deg: a change in mean inclination larger than this
            between consecutive element sets flags a plane change.
        status: ``"pending approval"`` until the values are approved, then
            ``"approved <date>"``.
    """

    perigee_min_km: float = 250.0
    apogee_max_km: float = 2000.0
    bstar_min_exclusive: float = 0.0
    bstar_max: float = 0.01
    ephemeris_types: tuple[int, ...] = (0,)
    object_types: tuple[str, ...] = ("PAY", "R/B", "DEB")
    exclude_ops_status: tuple[str, ...] = ("+", "P", "B", "S", "X")
    exclude_name_patterns: tuple[str, ...] = ()
    name_rule_column: str = "satcat_name"
    name_rule_object_types: tuple[str, ...] = ("PAY",)
    sma_increase_max_km: float = 1.0
    sma_drop_drag_multiple: float = 3.0
    sma_drop_floor_km: float = 1.0
    inclination_jump_max_deg: float = 0.05
    status: str = "pending approval"


def load_filter_config(path: str | Path) -> CatalogFilterConfig:
    """Read ``[catalog_filters]`` from a TOML file into :class:`CatalogFilterConfig`."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)["catalog_filters"]
    known = {f.name for f in fields(CatalogFilterConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown catalog filter keys: {sorted(unknown)}")
    kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}
    return CatalogFilterConfig(**kwargs)


@dataclass(frozen=True)
class FilterRule:
    """A named catalog filter.

    Attributes:
        name: stable identifier used in reports and the changelog.
        description: what it keeps, in words.
        keep: function (table, config) -> boolean numpy mask of rows to keep.
        needs: columns the rule reads; missing columns raise rather than skip.
    """

    name: str
    description: str
    keep: Callable[[pa.Table, CatalogFilterConfig], np.ndarray]
    needs: tuple[str, ...]


def _col(table: pa.Table, name: str) -> np.ndarray:
    return table.column(name).to_numpy(zero_copy_only=False)


def name_rule_matches(
    names: Iterable[str | None],
    object_types: Iterable[str | None],
    cfg: CatalogFilterConfig,
) -> np.ndarray:
    """True where the constellation name rule removes a row (or SATCAT entry).

    A row is removed when its object type is in ``cfg.name_rule_object_types`` and
    its name starts with one of ``cfg.exclude_name_patterns`` (case-insensitive).
    Shared by the catalog filter and the download's candidate rule, so both apply
    exactly the same test.
    """
    names, types = list(names), list(object_types)
    if not cfg.exclude_name_patterns:
        return np.zeros(len(names), dtype=bool)
    pattern = re.compile("|".join(f"(?:{p})" for p in cfg.exclude_name_patterns), re.IGNORECASE)
    applies = set(cfg.name_rule_object_types)
    return np.array(
        [bool(t in applies and n and pattern.match(n)) for n, t in zip(names, types, strict=True)],
        dtype=bool,
    )


def _keep_constellation(table: pa.Table, cfg: CatalogFilterConfig) -> np.ndarray:
    if cfg.name_rule_column not in ("satcat_name", "object_name"):
        raise ValueError(
            f"name_rule_column must be 'satcat_name' or 'object_name', got {cfg.name_rule_column!r}"
        )
    names = table.column(cfg.name_rule_column).to_pylist()
    return ~name_rule_matches(names, table.column("object_type").to_pylist(), cfg)


def _keep_not_operational_payload(table: pa.Table, cfg: CatalogFilterConfig) -> np.ndarray:
    otype = table.column("object_type").to_pylist()
    status = table.column("ops_status").to_pylist()
    pairs = zip(otype, status, strict=True)
    excluded = cfg.exclude_ops_status
    return np.array([not (t == "PAY" and s in excluded) for t, s in pairs], dtype=bool)


def _consecutive_pairs(table: pa.Table, *columns: str) -> tuple[np.ndarray, ...]:
    """Per-object consecutive element-set pairs, ordered by epoch.

    Returns ``(ids, same, dt_days, *pairs)``: ``ids`` of the later set of each
    pair, a mask ``same`` that is True where both sets belong to one object, the
    gap in days, and for each requested column an ``(earlier, later)`` tuple.
    Nulls in a requested column come back as NaN.
    """
    ids = _col(table, "norad_id")
    t = _col(table, "epoch_tt_s")
    order = np.lexsort((t, ids))
    ids, t = ids[order], t[order]
    same = ids[1:] == ids[:-1]
    out: list[object] = [ids[1:], same, (t[1:] - t[:-1]) / 86400.0]
    for name in columns:
        v = table.column(name).to_numpy(zero_copy_only=False).astype(np.float64)[order]
        out.append((v[:-1], v[1:]))
    return tuple(out)


def maneuver_flagged_ids(table: pa.Table, sma_increase_max_km: float) -> set[int]:
    """NORAD IDs whose mean semimajor axis rises by more than the threshold.

    Element sets are ordered by epoch per object; a rise between consecutive
    sets larger than ``sma_increase_max_km`` is taken as a maneuver (drag only
    lowers the semimajor axis). Orbit-lowering burns are caught by
    :func:`sma_drop_flagged_ids` and plane changes by
    :func:`inclination_jump_flagged_ids`. Fit noise can trip any of the three
    at small thresholds; small maneuvers slip under all of them.
    """
    if table.num_rows == 0:
        return set()
    ids, same, _, (a0, a1) = _consecutive_pairs(table, "semimajor_axis_km")
    jump = (a1 - a0) > sma_increase_max_km
    return {int(x) for x in ids[same & jump]}


def drag_decay_rate_km_per_day(
    semimajor_axis_km: np.ndarray,
    mean_motion_rev_per_day: np.ndarray,
    mean_motion_dot: np.ndarray,
) -> np.ndarray:
    """Semimajor-axis decay rate implied by an element set's mean-motion derivative.

    The element set's ``MEAN_MOTION_DOT`` field is half the rate of change of
    mean motion (rev/day^2). From Kepler's third law, ``a ~ n^(-2/3)``, so
    ``da/dt = -(2/3) (a/n) dn/dt``. Returned as a positive decay rate in km/day
    (``(2/3) (a/n) * 2 * MEAN_MOTION_DOT``), clipped at 0; a missing derivative
    gives 0 (no drag allowance).
    """
    a = np.asarray(semimajor_axis_km, dtype=np.float64)
    n = np.asarray(mean_motion_rev_per_day, dtype=np.float64)
    ndot_half = np.nan_to_num(np.asarray(mean_motion_dot, dtype=np.float64), nan=0.0)
    return np.clip((2.0 / 3.0) * (a / n) * 2.0 * ndot_half, 0.0, None)


def sma_drop_flagged_ids(
    table: pa.Table, drag_multiple: float, floor_km: float
) -> set[int]:
    """NORAD IDs whose mean semimajor axis falls faster than drag can explain.

    For each pair of consecutive element sets of one object, the drag
    allowance is the larger of the two sets' implied decay rates
    (:func:`drag_decay_rate_km_per_day`) times the gap. Taking the larger rate
    lets a storm that speeds decay up between the sets pass, since the later
    set's mean-motion derivative already reflects it. A drop larger than
    ``drag_multiple * allowance + floor_km`` is taken as an orbit-lowering
    maneuver. The floor absorbs fit noise between element sets; the multiple
    absorbs drag that changes faster than the derivative suggests.

    Limits: a lowering burn smaller than the floor plus the drag allowance is
    missed, and so is any maneuver hidden inside a long gap between sets.
    """
    if table.num_rows == 0:
        return set()
    ids, same, dt, (a0, a1), (n0, n1), (nd0, nd1) = _consecutive_pairs(
        table, "semimajor_axis_km", "mean_motion_rev_per_day", "mean_motion_dot"
    )
    rate = np.maximum(
        drag_decay_rate_km_per_day(a0, n0, nd0), drag_decay_rate_km_per_day(a1, n1, nd1)
    )
    drop = a0 - a1
    flagged = drop > drag_multiple * rate * dt + floor_km
    return {int(x) for x in ids[same & flagged]}


def inclination_jump_flagged_ids(table: pa.Table, max_deg: float) -> set[int]:
    """NORAD IDs whose mean inclination changes by more than ``max_deg`` between sets.

    Drag barely changes inclination, and natural changes in SGP4 mean
    inclination over days to weeks are small, so a step is taken as a plane
    change. Over very long gaps slow natural drift adds up; the threshold is
    applied per pair regardless of the gap.
    """
    if table.num_rows == 0:
        return set()
    ids, same, _, (i0, i1) = _consecutive_pairs(table, "inclination_deg")
    jump = np.abs(i1 - i0) > max_deg
    return {int(x) for x in ids[same & jump]}


def _keep_not_flagged(flagged: set[int], table: pa.Table) -> np.ndarray:
    return ~np.isin(_col(table, "norad_id"), list(flagged))


def _keep_no_sma_increase(table: pa.Table, cfg: CatalogFilterConfig) -> np.ndarray:
    return _keep_not_flagged(maneuver_flagged_ids(table, cfg.sma_increase_max_km), table)


def _keep_no_sma_drop(table: pa.Table, cfg: CatalogFilterConfig) -> np.ndarray:
    flagged = sma_drop_flagged_ids(table, cfg.sma_drop_drag_multiple, cfg.sma_drop_floor_km)
    return _keep_not_flagged(flagged, table)


def _keep_no_inclination_jump(table: pa.Table, cfg: CatalogFilterConfig) -> np.ndarray:
    flagged = inclination_jump_flagged_ids(table, cfg.inclination_jump_max_deg)
    return _keep_not_flagged(flagged, table)


#: The default rules, applied in this order.
DEFAULT_RULES: tuple[FilterRule, ...] = (
    FilterRule(
        "sgp4_ephemeris_type", "OMM EPHEMERIS_TYPE in ephemeris_types (0 = SGP4)",
        # A missing EPHEMERIS_TYPE is NaN here and fails the rule.
        lambda t, c: np.isin(_col(t, "ephemeris_type").astype(float), c.ephemeris_types),
        ("ephemeris_type",),
    ),
    FilterRule(
        "leo_perigee", "perigee altitude >= perigee_min_km",
        lambda t, c: _col(t, "perigee_alt_km") >= c.perigee_min_km, ("perigee_alt_km",),
    ),
    FilterRule(
        "leo_apogee", "apogee altitude <= apogee_max_km",
        lambda t, c: _col(t, "apogee_alt_km") <= c.apogee_max_km, ("apogee_alt_km",),
    ),
    FilterRule(
        "bstar_positive", "B* > bstar_min_exclusive",
        lambda t, c: _col(t, "bstar_per_earth_radii") > c.bstar_min_exclusive,
        ("bstar_per_earth_radii",),
    ),
    FilterRule(
        "bstar_not_extreme", "B* <= bstar_max",
        lambda t, c: _col(t, "bstar_per_earth_radii") <= c.bstar_max, ("bstar_per_earth_radii",),
    ),
    FilterRule(
        "object_type", "SATCAT object type in object_types",
        lambda t, c: np.array([x in c.object_types for x in t.column("object_type").to_pylist()],
                              dtype=bool),
        ("object_type",),
    ),
    FilterRule(
        "not_operational_payload",
        "not a payload whose current SATCAT status is in exclude_ops_status",
        _keep_not_operational_payload, ("object_type", "ops_status"),
    ),
    FilterRule(
        "not_maneuvering_constellation",
        "not an object of a type in name_rule_object_types whose name (name_rule_column) "
        "starts with one of exclude_name_patterns",
        # needs both name columns so a config switch can't hit a missing column late
        _keep_constellation, ("object_name", "satcat_name", "object_type"),
    ),
    FilterRule(
        "no_sma_increase",
        "object's element history has no mean-SMA rise > sma_increase_max_km",
        _keep_no_sma_increase, ("norad_id", "epoch_tt_s", "semimajor_axis_km"),
    ),
    FilterRule(
        "no_sma_unexplained_drop",
        "object's element history has no mean-SMA drop > sma_drop_drag_multiple x the "
        "drag decay implied by mean_motion_dot over the gap + sma_drop_floor_km",
        _keep_no_sma_drop,
        ("norad_id", "epoch_tt_s", "semimajor_axis_km", "mean_motion_rev_per_day",
         "mean_motion_dot"),
    ),
    FilterRule(
        "no_inclination_jump",
        "object's element history has no mean-inclination change > inclination_jump_max_deg",
        _keep_no_inclination_jump, ("norad_id", "epoch_tt_s", "inclination_deg"),
    ),
)


@dataclass(frozen=True)
class FilterResult:
    """Output of :func:`apply_filters`.

    Attributes:
        kept: rows that pass every rule, in input order.
        rejected_by_rule: rule name -> rows that rule rejects on its own (rules
            overlap, so these need not sum to the total dropped).
        dropped_at_rule: rule name -> rows removed at that step when the rules
            are applied in order (these do sum to the total dropped).
        n_input: input rows.
    """

    kept: pa.Table
    rejected_by_rule: dict[str, int]
    dropped_at_rule: dict[str, int]
    n_input: int


def apply_filters(
    table: pa.Table,
    config: CatalogFilterConfig,
    rules: Iterable[FilterRule] = DEFAULT_RULES,
) -> FilterResult:
    """Apply named rules to an element-set table and report what each removed.

    Raises:
        KeyError: if a rule needs a column the table lacks (for example SATCAT
            columns before :func:`attach_satcat`).
    """
    rules = tuple(rules)
    for rule in rules:
        missing = [c for c in rule.needs if c not in table.column_names]
        if missing:
            raise KeyError(f"rule {rule.name!r} needs columns {missing}")
    keep_all = np.ones(table.num_rows, dtype=bool)
    rejected: dict[str, int] = {}
    dropped: dict[str, int] = {}
    for rule in rules:
        mask = np.asarray(rule.keep(table, config), dtype=bool)
        rejected[rule.name] = int((~mask).sum())
        dropped[rule.name] = int((keep_all & ~mask).sum())
        keep_all &= mask
    kept = table.filter(pa.array(keep_all))
    return FilterResult(kept, rejected, dropped, table.num_rows)


def unique_ids(table: pa.Table) -> np.ndarray:
    """Sorted unique NORAD IDs in a table."""
    return np.unique(pc.unique(table.column("norad_id")).to_numpy())
