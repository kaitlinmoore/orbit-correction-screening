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
        exclude_history_spikes: the spike rule (approved 2026-11-27): each
            history rule leaves out of its comparisons a single element set that
            departs from both neighbours by more than that rule's threshold, in
            opposite directions (see :func:`history_rule`). Thresholds unchanged.
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
    exclude_history_spikes: bool = True
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


#: The element-history rules, in :data:`DEFAULT_RULES` order. Each flags an object
#: from pairs of consecutive element sets; the spike rule applies to all three.
HISTORY_RULES: tuple[str, ...] = (
    "no_sma_increase", "no_sma_unexplained_drop", "no_inclination_jump",
)

_HISTORY_COLUMNS: dict[str, tuple[str, ...]] = {
    "no_sma_increase": ("semimajor_axis_km",),
    "no_sma_unexplained_drop": ("semimajor_axis_km", "mean_motion_rev_per_day",
                                "mean_motion_dot"),
    "no_inclination_jump": ("inclination_deg",),
}


@dataclass(frozen=True)
class HistoryRuleResult:
    """One element-history rule evaluated on a table (see :func:`history_rule`).

    Attributes:
        rule: the rule's name (one of :data:`HISTORY_RULES`).
        flagged_ids: NORAD IDs with at least one trip; the rule drops these objects.
        trip_ids: the NORAD ID of each tripped pair, one entry per trip.
        spike_ids: the NORAD ID of each element set the spike rule left out of the
            rule's comparisons, one entry per set (empty without the spike rule).
    """

    rule: str
    flagged_ids: set[int]
    trip_ids: np.ndarray
    spike_ids: np.ndarray


def _pair_change_and_threshold(
    rule: str, t: np.ndarray, v: dict[str, np.ndarray], params: Mapping[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """The quantity's change (later minus earlier) and the rule's threshold, per pair.

    Pairs are consecutive entries of the sorted arrays (object boundaries are masked
    by the caller). The threshold is a positive magnitude: a fixed value for the
    SMA-rise and inclination rules, the drag allowance plus the floor for the
    SMA-drop rule.
    """
    if rule == "no_sma_increase":
        a = v["semimajor_axis_km"]
        return a[1:] - a[:-1], np.full(a.size - 1, params["sma_increase_max_km"])
    if rule == "no_sma_unexplained_drop":
        a, n, nd = (v[c] for c in _HISTORY_COLUMNS[rule])
        rate = np.maximum(drag_decay_rate_km_per_day(a[:-1], n[:-1], nd[:-1]),
                          drag_decay_rate_km_per_day(a[1:], n[1:], nd[1:]))
        dt_days = (t[1:] - t[:-1]) / 86400.0
        allowance = params["sma_drop_drag_multiple"] * rate * dt_days + params["sma_drop_floor_km"]
        return a[1:] - a[:-1], allowance
    if rule == "no_inclination_jump":
        i = v["inclination_deg"]
        return i[1:] - i[:-1], np.full(i.size - 1, params["inclination_jump_max_deg"])
    raise ValueError(f"not an element-history rule: {rule!r}")


def _trips(rule: str, change: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Pairs the rule flags: a rise, a drop or either, beyond the pair's threshold."""
    if rule == "no_sma_increase":
        return change > threshold
    if rule == "no_sma_unexplained_drop":
        return -change > threshold
    return np.abs(change) > threshold


def isolated_spikes(same: np.ndarray, change: np.ndarray, threshold: np.ndarray) -> np.ndarray:
    """Element sets that go out past the threshold and come straight back (spikes).

    Set ``j`` (an interior set of one object's sorted history) is a spike when its
    change from the previous set and its change to the next set both exceed the
    pair's threshold, in opposite directions: above both neighbours, or below both,
    by more than the threshold. A spike is *isolated* when neither neighbour is a
    spike, so two or more bad sets in a row, or an oscillation, are never left out.

    Args:
        same: per pair, True where both sets belong to one object.
        change: per pair, later minus earlier.
        threshold: per pair, the rule's positive threshold.

    Returns:
        A boolean mask over the sets (``len(same) + 1``).
    """
    n_sets = same.size + 1
    up = change > threshold
    down = change < -threshold
    spike = np.zeros(n_sets, dtype=bool)
    if n_sets >= 3:
        out_up, out_down, back_up, back_down = up[:-1], down[:-1], up[1:], down[1:]
        both = same[:-1] & same[1:]
        spike[1:-1] = both & ((out_up & back_down) | (out_down & back_up))
    isolated = spike.copy()
    isolated[1:] &= ~spike[:-1]
    isolated[:-1] &= ~spike[1:]
    return isolated


def history_rule(
    table: pa.Table, rule: str, params: Mapping[str, float] | CatalogFilterConfig,
    exclude_spikes: bool = False,
) -> HistoryRuleResult:
    """Evaluate one element-history rule on every object of ``table``.

    Element sets are ordered by epoch per object and compared pairwise:

    * ``no_sma_increase``: a mean semimajor-axis rise above ``sma_increase_max_km``
      (drag only lowers it);
    * ``no_sma_unexplained_drop``: a fall larger than ``sma_drop_drag_multiple`` x the
      drag decay implied by the sets' ``MEAN_MOTION_DOT`` over the gap +
      ``sma_drop_floor_km``;
    * ``no_inclination_jump``: a mean-inclination change above
      ``inclination_jump_max_deg``.

    **Spike rule** (``exclude_spikes``; approved 2026-11-27): an isolated element set
    that departs from both of its neighbours by more than this rule's threshold, in
    opposite directions (out and back, see :func:`isolated_spikes`), is left out of
    this rule's comparisons, and its two neighbours are compared with each other
    instead (over the combined gap). The set stays in the pool, and each rule finds
    its own spikes. A lone out-and-back set is a bad element set, not a maneuver; a
    maneuver is a step that later sets keep. Limits: a maneuver undone at the very
    next set looks like a spike and is missed, and a bad first or last set of a
    history has only one neighbour and still trips the rule.

    Args:
        table: element sets with ``norad_id``, ``epoch_tt_s`` and the rule's columns.
        rule: one of :data:`HISTORY_RULES`.
        params: a :class:`CatalogFilterConfig` or a mapping with the rule's thresholds.
        exclude_spikes: apply the spike rule.
    """
    if isinstance(params, CatalogFilterConfig):
        params = {f.name: getattr(params, f.name) for f in fields(params)}
    empty = np.zeros(0, dtype=np.int64)
    if table.num_rows == 0:
        return HistoryRuleResult(rule, set(), empty, empty)
    ids = _col(table, "norad_id")
    t = _col(table, "epoch_tt_s").astype(np.float64)
    order = np.lexsort((t, ids))
    ids, t = ids[order], t[order]
    v = {c: table.column(c).to_numpy(zero_copy_only=False).astype(np.float64)[order]
         for c in _HISTORY_COLUMNS[rule]}
    spike_ids = empty
    if exclude_spikes:
        change, thr = _pair_change_and_threshold(rule, t, v, params)
        spikes = isolated_spikes(ids[1:] == ids[:-1], change, thr)
        spike_ids = ids[spikes]
        ids, t = ids[~spikes], t[~spikes]
        v = {c: x[~spikes] for c, x in v.items()}
    if ids.size < 2:
        return HistoryRuleResult(rule, set(), empty, spike_ids)
    change, thr = _pair_change_and_threshold(rule, t, v, params)
    trip = (ids[1:] == ids[:-1]) & _trips(rule, change, thr)
    trip_ids = ids[1:][trip]
    return HistoryRuleResult(rule, {int(x) for x in trip_ids}, trip_ids, spike_ids)


def maneuver_flagged_ids(
    table: pa.Table, sma_increase_max_km: float, exclude_spikes: bool = False
) -> set[int]:
    """NORAD IDs whose mean semimajor axis rises by more than the threshold.

    Element sets are ordered by epoch per object; a rise between consecutive
    sets larger than ``sma_increase_max_km`` is taken as a maneuver (drag only
    lowers the semimajor axis). Orbit-lowering burns are caught by
    :func:`sma_drop_flagged_ids` and plane changes by
    :func:`inclination_jump_flagged_ids`. Fit noise can trip any of the three
    at small thresholds; small maneuvers slip under all of them.
    ``exclude_spikes``: the spike rule of :func:`history_rule`.
    """
    params = {"sma_increase_max_km": sma_increase_max_km}
    return history_rule(table, "no_sma_increase", params, exclude_spikes).flagged_ids


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
    table: pa.Table, drag_multiple: float, floor_km: float, exclude_spikes: bool = False
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
    ``exclude_spikes``: the spike rule of :func:`history_rule`, with this
    allowance as the threshold of both legs.

    Limits: a lowering burn smaller than the floor plus the drag allowance is
    missed, and so is any maneuver hidden inside a long gap between sets.
    """
    params = {"sma_drop_drag_multiple": drag_multiple, "sma_drop_floor_km": floor_km}
    return history_rule(table, "no_sma_unexplained_drop", params, exclude_spikes).flagged_ids


def inclination_jump_flagged_ids(
    table: pa.Table, max_deg: float, exclude_spikes: bool = False
) -> set[int]:
    """NORAD IDs whose mean inclination changes by more than ``max_deg`` between sets.

    Drag barely changes inclination, and natural changes in SGP4 mean
    inclination over days to weeks are small, so a step is taken as a plane
    change. Over very long gaps slow natural drift adds up; the threshold is
    applied per pair regardless of the gap.
    ``exclude_spikes``: the spike rule of :func:`history_rule`.
    """
    params = {"inclination_jump_max_deg": max_deg}
    return history_rule(table, "no_inclination_jump", params, exclude_spikes).flagged_ids


def _keep_not_flagged(flagged: set[int], table: pa.Table) -> np.ndarray:
    return ~np.isin(_col(table, "norad_id"), list(flagged))


def _keep_history(rule: str) -> Callable[[pa.Table, CatalogFilterConfig], np.ndarray]:
    def keep(table: pa.Table, cfg: CatalogFilterConfig) -> np.ndarray:
        res = history_rule(table, rule, cfg, exclude_spikes=cfg.exclude_history_spikes)
        return _keep_not_flagged(res.flagged_ids, table)
    return keep


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
        "object's element history has no mean-SMA rise > sma_increase_max_km "
        "(isolated out-and-back sets left out when exclude_history_spikes)",
        _keep_history("no_sma_increase"), ("norad_id", "epoch_tt_s", "semimajor_axis_km"),
    ),
    FilterRule(
        "no_sma_unexplained_drop",
        "object's element history has no mean-SMA drop > sma_drop_drag_multiple x the "
        "drag decay implied by mean_motion_dot over the gap + sma_drop_floor_km "
        "(isolated out-and-back sets left out when exclude_history_spikes)",
        _keep_history("no_sma_unexplained_drop"),
        ("norad_id", "epoch_tt_s", "semimajor_axis_km", "mean_motion_rev_per_day",
         "mean_motion_dot"),
    ),
    FilterRule(
        "no_inclination_jump",
        "object's element history has no mean-inclination change > inclination_jump_max_deg "
        "(isolated out-and-back sets left out when exclude_history_spikes)",
        _keep_history("no_inclination_jump"), ("norad_id", "epoch_tt_s", "inclination_deg"),
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
