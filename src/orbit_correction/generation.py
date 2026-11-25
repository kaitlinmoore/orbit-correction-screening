"""Residual-dataset generation: candidate objects, training pool, windows, chunked runs.

Why this module exists
----------------------
The pieces from earlier steps (catalog filters, splits, SGP4, the truth propagator and
the row builder) each handle one element set or one window. This module turns them
into a pipeline that makes a residual dataset from real historical element sets:

1. **Candidate IDs** (:func:`candidate_ids_from_satcat`): which NORAD IDs to download
   ``gp_history`` for, chosen from SATCAT alone before any element set is seen.
2. **Training pool** (later sections): the downloaded history with SATCAT attached and
   the approved catalog filters applied.
3. **Windows**: start epochs sampled from the pool, labeled by the split.
4. **Chunked runs**: SGP4 in this process, truth in spawned workers, rows written one
   Parquet file per chunk, every window's fate logged.

This module never starts a JVM. Truth runs in worker processes (``truth_prop``'s pool
functions); the process that imports this module stays Java-free.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from orbit_correction.catalog import CatalogFilterConfig, name_rule_matches

# --------------------------------------------------------------------------------------
# Candidate IDs (the download list)
# --------------------------------------------------------------------------------------

#: Reasons a SATCAT row is left out of the candidate list, in the order they are tested.
CANDIDATE_DROP_REASONS = (
    "object_type",
    "operational_payload",
    "launched_after_span",
    "decayed_before_span",
    "apogee_above_leo",
    "name_rule",
)


@dataclass(frozen=True)
class CandidateResult:
    """Output of :func:`candidate_ids_from_satcat`.

    Attributes:
        ids: sorted candidate NORAD IDs, forced IDs included.
        n_satcat: SATCAT rows read.
        dropped: reason -> rows left out for it (first applicable reason only).
        forced_added: forced IDs that the rule alone would have left out.
        unknown_kept: rows kept because a field the rule needs was blank
            (``launch_date``, ``apogee``; a blank ``decay_date`` means still in orbit and
            isn't counted).
    """

    ids: np.ndarray
    n_satcat: int
    dropped: dict[str, int]
    forced_added: tuple[int, ...]
    unknown_kept: dict[str, int] = field(default_factory=dict)


def candidate_ids_from_satcat(
    satcat_csv: str | Path,
    filters: CatalogFilterConfig,
    span_start_utc: str,
    span_end_utc: str,
    forced_ids: Iterable[int] = (),
    *,
    apogee_margin_km: float = 100.0,
    decay_margin_days: int = 30,
    apply_name_rule: bool | None = None,
) -> CandidateResult:
    """NORAD IDs that could pass the catalog filters at some epoch in the span.

    A **superset** of the objects the filters can keep, chosen from CelesTrak's
    SATCAT alone, so the download never misses an object the filters would keep.
    Each test leaves out only rows that cannot have a kept element set in
    ``[span_start_utc, span_end_utc)``:

    * ``object_type``: SATCAT type not in ``filters.object_types``. Exact: the
      filter reads the same SATCAT column.
    * ``operational_payload``: a payload whose current status is in
      ``filters.exclude_ops_status``. Exact for the same reason (the filter uses the
      *current* status too).
    * ``launched_after_span``: launch date on or after the span end (no element set
      can precede launch).
    * ``decayed_before_span``: decay date more than ``decay_margin_days`` before the
      span start (no element set after decay; the margin covers decay dates that
      are a little off).
    * ``apogee_above_leo``: SATCAT apogee above ``filters.apogee_max_km +
      apogee_margin_km``. SATCAT's apogee is the latest one; drag only lowers
      apogee, and at these altitudes the natural long-period changes are tens of km
      at most, so an object above the bound now was above the LEO bound throughout
      the span **unless it raised its orbit after the span**. That is the one
      assumption; a raise inside the span shows in the element history and the
      filters drop the object anyway. The margin also covers SATCAT values from an
      older element set (measured against the current CelesTrak GP data: SATCAT
      apogee 39 km below to 97 km above the loader's value, 1st-99th percentile
      -1.4 to +2.8 km).

    No **perigee** test: perigee only falls under drag, so an object below 250 km
    now (or decayed) may have been above it earlier in the span.

    * ``name_rule``: the constellation name rule
      (:func:`orbit_correction.catalog.name_rule_matches`), applied only when the
      filter reads SATCAT names (``filters.name_rule_column == "satcat_name"``, the
      setting decided 2026-11-24). It is then exact: the same names, types and
      patterns. If the filter reads each element set's own name instead, an
      element set's name need not be the object's current one, so the rule is not
      applied here unless ``apply_name_rule=True`` forces it (no longer a strict
      superset).

    Blank fields keep the row (counted in ``unknown_kept``). Forced IDs (the
    validation-case objects) are always added.

    Args:
        satcat_csv: CelesTrak ``satcat.csv``.
        filters: the approved catalog filters (``config/catalog_filters.toml``).
        span_start_utc, span_end_utc: ``YYYY-MM-DD``; the span is ``[start, end)``.
        forced_ids: always included.
        apogee_margin_km, decay_margin_days: see above.
        apply_name_rule: None (default) applies the name rule when it is exact;
            True or False forces it on or off.
    """
    start = np.datetime64(span_start_utc, "D")
    end = np.datetime64(span_end_utc, "D")
    if not start < end:
        raise ValueError("span_start_utc must be before span_end_utc")
    decay_cut = start - np.timedelta64(int(decay_margin_days), "D")
    apogee_bound = float(filters.apogee_max_km) + float(apogee_margin_km)
    if apply_name_rule is None:
        apply_name_rule = filters.name_rule_column == "satcat_name"

    dropped = dict.fromkeys(CANDIDATE_DROP_REASONS, 0)
    unknown = {"launch_date": 0, "apogee": 0}
    kept: set[int] = set()
    n = 0
    with open(satcat_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n += 1
            nid = int(row["NORAD_CAT_ID"])
            otype = (row.get("OBJECT_TYPE") or "").strip()
            status = (row.get("OPS_STATUS_CODE") or "").strip()
            launch = (row.get("LAUNCH_DATE") or "").strip()
            decay = (row.get("DECAY_DATE") or "").strip()
            apogee = (row.get("APOGEE") or "").strip()
            name = (row.get("OBJECT_NAME") or "").strip()
            reason = None
            if otype not in filters.object_types:
                reason = "object_type"
            elif otype == "PAY" and status in filters.exclude_ops_status:
                reason = "operational_payload"
            elif launch and np.datetime64(launch, "D") >= end:
                reason = "launched_after_span"
            elif decay and np.datetime64(decay, "D") < decay_cut:
                reason = "decayed_before_span"
            elif apogee and float(apogee) > apogee_bound:
                reason = "apogee_above_leo"
            elif apply_name_rule and name_rule_matches([name], [otype], filters)[0]:
                reason = "name_rule"
            if reason is not None:
                dropped[reason] += 1
                continue
            unknown["launch_date"] += not launch
            unknown["apogee"] += not apogee
            kept.add(nid)
    forced = sorted({int(i) for i in forced_ids} - kept)
    ids = np.array(sorted(kept | set(forced)), dtype=np.int64)
    return CandidateResult(ids, n, dropped, tuple(forced), unknown)


def download_sample_ids(ids: Iterable[int], n: int, seed: int) -> np.ndarray:
    """``n`` IDs picked by a seeded hash draw (``splits.object_draw``, salt ``download``).

    Used to size the history download from a small first request whose files are
    then part of the full download (the sample's IDs are left out of the main list,
    so nothing is fetched twice).
    """
    from orbit_correction.splits import object_draw

    ids = sorted({int(i) for i in ids})
    draws = np.array([object_draw(i, seed, "download") for i in ids])
    pick = np.argsort(draws, kind="stable")[: int(n)]
    return np.array(sorted(np.asarray(ids)[pick].tolist()), dtype=np.int64)


# --------------------------------------------------------------------------------------
# Bulk TLE files (Space-Track's yearly zips)
# --------------------------------------------------------------------------------------
#
# Space-Track asks that large history pulls use its yearly zip files of TLEs rather
# than ``gp_history`` queries (decided 2026-11-24: training history from these files).
# A TLE carries the same SGP4 mean elements as the OMM record of the same element set,
# in fixed columns. These functions turn TLE lines into OMM-keyword records, so the
# element sets go through the one conversion every loader uses
# (:func:`orbit_correction.catalog.records_to_table`).
#
# Library behaviour: the column layout and the number arithmetic follow the ``sgp4``
# package's TLE reader (``sgp4.io.twoline2rv``) and its Alpha-5 decoder
# (``sgp4.alpha5.from_alpha5``), so a record read here gives the same ``Satrec`` as
# that reader, apart from the epoch (below). Ours: the record layout, the checks and
# the epoch string.


class TleFormatError(ValueError):
    """Two lines that are not a valid TLE (layout, checksum or matching numbers)."""


class TleNoChecksumError(TleFormatError):
    """A 68-character line (no checksum column) where such lines are not accepted.

    Space-Track's 2015-2020 bulk files carry many such element sets (explicit ``+``
    signs, every field in its standard column, no checksum). They are accepted with
    ``allow_no_checksum=True`` (decided 2026-11-25), checked by layout instead.
    """


#: Column separators of the TLE layout that ``sgp4.io.twoline2rv`` checks (0-based
#: index -> character). A line without a checksum must pass these instead.
_LINE1_LAYOUT = {1: " ", 8: " ", 23: ".", 32: " ", 34: ".", 43: " ", 52: " ", 61: " ", 63: " "}
_LINE2_LAYOUT = {1: " ", 7: " ", 11: ".", 16: " ", 20: ".", 25: " ", 33: " ", 37: ".", 42: " ",
                 46: ".", 51: " "}


def _layout_ok(line: str, layout: dict[int, str]) -> bool:
    return all(line[i] == c for i, c in layout.items())


def _clean(line: str) -> str:
    """A TLE line without its line ending or one trailing backslash.

    Some bulk files end every line 1 with ``\\`` (the 2015 file); the backslash is
    not part of the format (column 70 would be past the checksum).
    """
    line = line.rstrip("\r\n")
    return line[:-1] if line.endswith("\\") else line


def tle_checksum(line: str) -> int:
    """TLE checksum of a line's first 68 characters: digits summed, ``-`` counts 1, mod 10."""
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


def tle_epoch_to_iso(field: str) -> str:
    """TLE epoch ``YYDDD.DDDDDDDD`` (UTC) to an ISO 8601 string with microseconds.

    Two-digit years 57-99 are 1957-1999, 00-56 are 2000-2056 (the TLE convention).
    The day fraction counts a day as 86,400 s, as SGP4 and the TLE format do, so a
    leap-second day never produces ``:60``. The time is rounded to the microsecond
    (the precision of Space-Track's OMM ``EPOCH``); the TLE's 8 fraction digits
    resolve 0.864 ms, so rounding moves the epoch by at most 0.5 microseconds.
    """
    import datetime as dt
    from decimal import ROUND_HALF_EVEN, Decimal

    field = field.strip()
    yy = int(field[:2])
    year = 1900 + yy if yy >= 57 else 2000 + yy
    days = Decimal(field[2:])
    whole = int(days)
    micro = int(
        ((days - whole) * Decimal(86_400_000_000)).quantize(Decimal(1), rounding=ROUND_HALF_EVEN)
    )
    t = dt.datetime(year, 1, 1) + dt.timedelta(days=whole - 1, microseconds=micro)
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f")


def _intl_designator(field: str) -> str | None:
    """TLE ``YYNNNPPP`` (columns 10-17) to OMM ``OBJECT_ID`` ``YYYY-NNNPPP``."""
    f = field.strip()
    if len(f) < 5 or not f[:5].isdigit():
        return f or None
    yy = int(f[:2])
    return f"{1900 + yy if yy >= 57 else 2000 + yy}-{f[2:]}"


def _implied(mantissa_sign: str, digits: str, exponent: str) -> float:
    """TLE implied-decimal field (``-12345-3`` = -0.12345e-3), with twoline2rv's arithmetic."""
    return float(mantissa_sign + "." + digits) * pow(10.0, int(exponent))


def tle_norad_id(line1: str) -> int:
    """NORAD ID from TLE line 1 (columns 3-7, Alpha-5 above 99,999)."""
    from sgp4.alpha5 import from_alpha5

    return int(from_alpha5(line1[2:7].strip().rjust(5, "0")))


def tle_to_record(line1: str, line2: str, name: str | None = None, *,
                  allow_no_checksum: bool = False) -> dict:
    """One TLE as an OMM-keyword record (the keys :func:`catalog.records_to_table` reads).

    Values are numbers computed exactly as ``sgp4.io.twoline2rv`` computes them, so the
    ``Satrec`` built from the record (``cheap_prop.satrec_from_row``) matches the TLE
    reader's, apart from the epoch's sub-microsecond rounding (:func:`tle_epoch_to_iso`).

    Args:
        line1, line2: the two element lines.
        name: the title line of a three-line set without its ``0 `` prefix, if any.
            Recorded as ``OBJECT_NAME``; the catalog's name rule reads SATCAT names,
            not this.
        allow_no_checksum: accept 68-character lines with no checksum column, if
            they pass the library reader's column-separator checks (decided
            2026-11-25 for Space-Track's bulk files). The record then carries
            ``"_checksum": False`` (``True`` for verified lines), so the caller can
            tag its source.

    Raises:
        TleNoChecksumError: a line without a checksum when not allowed.
        TleFormatError: wrong layout, a checksum that doesn't match, or different
            object numbers on the two lines.
    """
    l1, l2 = _clean(line1), _clean(line2)
    if not l1.startswith("1 ") or not l2.startswith("2 "):
        raise TleFormatError(f"not a TLE pair: {l1[:20]!r} / {l2[:20]!r}")
    no_checksum = len(l1) == 68 or len(l2) == 68
    if no_checksum:
        if not allow_no_checksum:
            raise TleNoChecksumError(f"no checksum column: {l1!r} / {l2!r}")
        if len(l1) < 68 or len(l2) < 68 or not (
            _layout_ok(l1, _LINE1_LAYOUT) and _layout_ok(l2, _LINE2_LAYOUT)
        ):
            raise TleFormatError(f"no checksum and the layout doesn't match: {l1!r} / {l2!r}")
    else:
        if len(l1) < 69 or len(l2) < 69:
            raise TleFormatError(f"line too short: {l1!r} / {l2!r}")
        for n, line in ((1, l1), (2, l2)):
            if not line[68].isdigit() or tle_checksum(line) != int(line[68]):
                raise TleFormatError(f"line {n} checksum mismatch: {line!r}")
    if l1[2:7] != l2[2:7]:
        raise TleFormatError(f"object numbers differ: {l1[2:7]!r} / {l2[2:7]!r}")
    try:
        eph = l1[62].strip()
        rec = {
            "NORAD_CAT_ID": tle_norad_id(l1),
            "OBJECT_NAME": (name or "").strip() or None,
            "OBJECT_ID": _intl_designator(l1[9:17]),
            "CLASSIFICATION_TYPE": l1[7].strip() or "U",
            "EPOCH": tle_epoch_to_iso(l1[18:32]),
            "MEAN_MOTION_DOT": float(l1[33:43]),
            "MEAN_MOTION_DDOT": _implied(l1[44], l1[45:50], l1[50:52]),
            "BSTAR": _implied(l1[53], l1[54:59], l1[59:61]),
            "EPHEMERIS_TYPE": int(eph) if eph else 0,
            "ELEMENT_SET_NO": int(l1[64:68]) if l1[64:68].strip() else None,
            "INCLINATION": float(l2[8:16]),
            "RA_OF_ASC_NODE": float(l2[17:25]),
            "ECCENTRICITY": float("0." + l2[26:33].replace(" ", "0")),
            "ARG_OF_PERICENTER": float(l2[34:42]),
            "MEAN_ANOMALY": float(l2[43:51]),
            "MEAN_MOTION": float(l2[52:63]),
            "REV_AT_EPOCH": int(l2[63:68]) if l2[63:68].strip() else None,
            "_checksum": not no_checksum,
        }
    except ValueError as exc:
        raise TleFormatError(f"unreadable field in {l1!r} / {l2!r}: {exc}") from exc
    return rec


@dataclass
class TleReadStats:
    """Counts from :func:`iter_tle_records`; nothing read is dropped without a count.

    Attributes:
        lines: lines read (blank lines included).
        tles: element-line pairs found.
        bad: wanted pairs that failed :func:`tle_to_record` (with up to 20 examples).
        no_checksum: wanted pairs with a 68-character line (no checksum column):
            yielded when ``allow_no_checksum`` (and the layout checks pass), else
            left out; counted either way.
        trailing_backslash: lines that ended in a backslash (removed before parsing).
        stray_lines: lines that belong to no pair (neither a title nor an element line
            in sequence).
        skipped_id: well-formed pairs whose ID was not wanted (not fully parsed).
        epoch_days: ``YYDDD`` of every pair's epoch (wanted or not) -> count, so a
            file's coverage by epoch can be checked.
        examples: messages of the first bad pairs.
    """

    lines: int = 0
    tles: int = 0
    bad: int = 0
    no_checksum: int = 0
    trailing_backslash: int = 0
    stray_lines: int = 0
    skipped_id: int = 0
    epoch_days: dict[str, int] = field(default_factory=dict)
    examples: list[str] = field(default_factory=list)


def iter_tle_records(lines: Iterable[str], stats: TleReadStats, want_ids: set[int] | None = None,
                     *, allow_no_checksum: bool = False):
    """Yield OMM-keyword records from TLE text: two-line or three-line sets, mixed.

    A line starting ``1 `` followed by one starting ``2 `` is an element set; a line
    before it that is neither (a ``0 NAME`` or bare ``NAME`` title) is its name.
    ``want_ids``, if given, skips other objects after reading only the ID (counted in
    ``stats.skipped_id``). Bad pairs are counted in ``stats.bad``, never raised, so
    one damaged line can't stop a year's file; the caller reports the count. Pairs
    without a checksum are counted in ``stats.no_checksum`` and yielded only with
    ``allow_no_checksum`` (see :func:`tle_to_record`).
    """
    name: str | None = None
    pending: str | None = None
    for raw in lines:
        stats.lines += 1
        line = raw.rstrip("\r\n")
        if line.endswith("\\"):
            stats.trailing_backslash += 1
            line = line[:-1]
        if not line.strip():
            continue
        if line.startswith("1 "):
            if pending is not None:  # a line 1 without its line 2
                stats.stray_lines += 1
            pending = line
            continue
        if line.startswith("2 "):
            if pending is None:  # a line 2 without its line 1
                stats.stray_lines += 1
                name = None
                continue
            stats.tles += 1
            l1, pending = pending, None
            nm, name = name, None
            day = l1[18:23]
            stats.epoch_days[day] = stats.epoch_days.get(day, 0) + 1
            try:
                nid = tle_norad_id(l1)
            except ValueError:
                nid = None
            if want_ids is not None and nid is not None and nid not in want_ids:
                stats.skipped_id += 1
                continue
            if len(_clean(l1)) == 68 or len(_clean(line)) == 68:
                stats.no_checksum += 1
            try:
                yield tle_to_record(l1, line, nm, allow_no_checksum=allow_no_checksum)
            except TleNoChecksumError:
                pass  # counted above, left out
            except TleFormatError as exc:
                stats.bad += 1
                if len(stats.examples) < 20:
                    stats.examples.append(str(exc)[:200])
            continue
        # a title line
        if pending is not None:
            stats.stray_lines += 1
            pending = None
        if name is not None:  # two titles in a row
            stats.stray_lines += 1
        name = line[2:] if line.startswith("0 ") else line
    if pending is not None:
        stats.stray_lines += 1


# --------------------------------------------------------------------------------------
# Generation config (config/generation.toml)
# --------------------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "config" / "generation.toml"
_DAY_S = 86400.0


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling, staging and run settings from ``config/generation.toml`` (documented there).

    Attributes:
        status: approval status of the values.
        seed: seed of every draw in sampling (starts, object order, the daily-Ap subsample).
        roles: split labels windows are drawn from (train and dev; never test).
        min_start_spacing_days: an object's starts are at least this far apart.
        max_starts_per_object: cap on starts per object over all stages.
        stage_windows: stage name -> target window count (pilot, first_batch, ceiling).
        chunk_windows: windows per chunk (one Parquet file each; memory bound).
        n_workers: truth worker processes (one JVM each).
        dailyap_windows: size of the daily-Ap sensitivity subsample of a stage.
        eccentricity_edges: reporting bins of the element set's eccentricity.
    """

    status: str
    seed: int
    roles: tuple[str, ...]
    min_start_spacing_days: float
    max_starts_per_object: int
    stage_windows: dict[str, int]
    chunk_windows: int
    n_workers: int
    dailyap_windows: int
    eccentricity_edges: tuple[float, ...]

    def __post_init__(self) -> None:
        bad = set(self.roles) - {"train", "dev"}
        if bad or not self.roles:
            raise ValueError(f"roles must be train and/or dev, got {self.roles}")
        if self.min_start_spacing_days < 7.0:
            raise ValueError("min_start_spacing_days must be at least 7 (decision 3(c))")
        if self.max_starts_per_object < 1 or self.chunk_windows < 1 or self.n_workers < 1:
            raise ValueError("max_starts_per_object, chunk_windows and n_workers must be >= 1")
        e = np.asarray(self.eccentricity_edges, dtype=np.float64)
        if e.size == 0 or np.any(np.diff(e) <= 0):
            raise ValueError("eccentricity_edges must be strictly increasing")


def load_generation_config(path: str | Path = CONFIG_PATH) -> GenerationConfig:
    """Read ``config/generation.toml`` (or a file with the same layout)."""
    import tomllib

    with open(path, "rb") as f:
        g = tomllib.load(f)["generation"]
    s, r, c = g["sampling"], g["run"], g["coverage"]
    return GenerationConfig(
        status=str(g["status"]),
        seed=int(s["seed"]),
        roles=tuple(s["roles"]),
        min_start_spacing_days=float(s["min_start_spacing_days"]),
        max_starts_per_object=int(s["max_starts_per_object"]),
        stage_windows={k: int(v) for k, v in g["stages"].items()},
        chunk_windows=int(r["chunk_windows"]),
        n_workers=int(r["n_workers"]),
        dailyap_windows=int(r["dailyap_windows"]),
        eccentricity_edges=tuple(float(x) for x in c["eccentricity_edges"]),
    )


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes (configs, inputs and outputs are recorded by it)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# Training pool
# --------------------------------------------------------------------------------------

#: Element columns that identify an element set's content (duplicates are rows that
#: agree on all of them). ``epoch_tt_s`` stands for the epoch: it is computed from the
#: epoch string, so equal strings give equal values.
_ELEMENT_KEY = (
    "norad_id", "epoch_tt_s", "mean_motion_rev_per_day", "eccentricity", "inclination_deg",
    "raan_deg", "arg_perigee_deg", "mean_anomaly_deg", "bstar_per_earth_radii",
    "mean_motion_dot", "mean_motion_ddot", "ephemeris_type",
)


def _key_matrix(table, columns) -> np.ndarray:
    """Columns as one float64 matrix (nulls as a fixed NaN), rows comparable bytewise."""
    out = np.empty((table.num_rows, len(columns)), dtype=np.float64)
    for j, c in enumerate(columns):
        col = table.column(c)
        v = col.to_numpy(zero_copy_only=False).astype(np.float64)
        if col.null_count:
            v[np.asarray(col.is_null())] = np.nan
        out[:, j] = v
    out[np.isnan(out)] = np.nan  # one NaN bit pattern
    return out


def _first_of_each(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the first occurrence of each distinct row, and each row's group id."""
    view = np.ascontiguousarray(rows).view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1])))
    _, first, inverse = np.unique(view.ravel(), return_index=True, return_inverse=True)
    return first, inverse.ravel()


@dataclass(frozen=True)
class DedupeStats:
    """What :func:`drop_exact_duplicates` found.

    Attributes:
        n_input: rows in.
        exact_duplicates: rows removed because an earlier row had the same object,
            epoch and elements (the same element set read twice).
        same_epoch_different: rows kept that share object and epoch with another row
            but differ in some element (corrections or reissues; not decided here).
    """

    n_input: int
    exact_duplicates: int
    same_epoch_different: int


def drop_exact_duplicates(elsets):
    """Remove repeated copies of the same element set; count, don't touch, the rest.

    Bulk files and API downloads can carry the same element set more than once.
    Identical copies add nothing and, in the element-history rules, form a pair with a
    zero gap and no change, which no rule flags, so removing them changes no filter
    outcome. Rows that share object and epoch but differ are kept and counted: which
    one to trust is a decision, not a cleaning step.

    Returns:
        ``(table, DedupeStats)``; the table keeps the first copy of each, in input order.
    """
    import pyarrow as pa

    n = elsets.num_rows
    if n == 0:
        return elsets, DedupeStats(0, 0, 0)
    first, _ = _first_of_each(_key_matrix(elsets, _ELEMENT_KEY))
    keep = np.zeros(n, dtype=bool)
    keep[first] = True
    out = elsets.filter(pa.array(keep))
    _, group = _first_of_each(_key_matrix(out, ("norad_id", "epoch_tt_s")))
    sizes = np.bincount(group)
    same_epoch_different = int(sizes[sizes > 1].sum())
    return out, DedupeStats(n, int((~keep).sum()), int(same_epoch_different))


@dataclass(frozen=True)
class PoolResult:
    """The training pool and how it was made.

    Attributes:
        table: kept element sets with SATCAT columns, sorted by (norad_id, epoch).
        n_loaded: element sets read.
        n_in_span: of those, epochs inside ``[span_start, span_end)``.
        dedupe: :class:`DedupeStats`.
        dropped_at_rule, rejected_by_rule: from :func:`catalog.apply_filters`.
        objects_dropped_at_rule: rule name -> objects whose last remaining element
            sets that rule removed (rules applied in order; sums to the objects lost).
        n_objects_in, n_objects_kept: distinct NORAD IDs before and after the filters.
    """

    table: object
    n_loaded: int
    n_in_span: int
    dedupe: DedupeStats
    dropped_at_rule: dict[str, int]
    rejected_by_rule: dict[str, int]
    objects_dropped_at_rule: dict[str, int]
    n_objects_in: int
    n_objects_kept: int


def build_pool(elsets, satcat, filters: CatalogFilterConfig, span_start_utc: str,
               span_end_utc: str) -> PoolResult:
    """Element sets in the span -> duplicates removed -> SATCAT attached -> filters applied.

    The element-history rules see every element set of an object inside the span
    (after duplicate removal, before any other rule), as :func:`catalog.apply_filters`
    evaluates each rule on the whole input.

    Args:
        elsets: element sets (``catalog.ELEMENT_SET_SCHEMA``), any sources.
        satcat: ``catalog.load_satcat`` output.
        filters: the approved catalog filters.
        span_start_utc, span_end_utc: ISO UTC; epochs in ``[start, end)`` are kept.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    from orbit_correction import catalog
    from orbit_correction.frames import utc_to_tt_seconds

    lo, hi = float(utc_to_tt_seconds(span_start_utc)), float(utc_to_tt_seconds(span_end_utc))
    t = elsets.column("epoch_tt_s")
    in_span = elsets.filter(pc.and_(pc.greater_equal(t, lo), pc.less(t, hi)))
    deduped, stats = drop_exact_duplicates(in_span)
    joined = catalog.attach_satcat(deduped, satcat)
    res = catalog.apply_filters(joined, filters)
    kept = res.kept
    # objects lost at each rule, applying the same masks in the same order
    ids = joined.column("norad_id").to_numpy()
    alive = np.ones(joined.num_rows, dtype=bool)
    objects_at: dict[str, int] = {}
    for rule in catalog.DEFAULT_RULES:
        before = set(np.unique(ids[alive]).tolist())
        alive &= np.asarray(rule.keep(joined, filters), dtype=bool)
        objects_at[rule.name] = len(before - set(np.unique(ids[alive]).tolist()))
    if int(alive.sum()) != kept.num_rows:
        raise RuntimeError("object count by rule disagrees with apply_filters")
    order = np.lexsort((kept.column("epoch_tt_s").to_numpy(), kept.column("norad_id").to_numpy()))
    kept = kept.take(pa.array(order))
    return PoolResult(
        table=kept,
        n_loaded=elsets.num_rows,
        n_in_span=in_span.num_rows,
        dedupe=stats,
        dropped_at_rule=res.dropped_at_rule,
        rejected_by_rule=res.rejected_by_rule,
        objects_dropped_at_rule=objects_at,
        n_objects_in=int(len(catalog.unique_ids(joined))) if joined.num_rows else 0,
        n_objects_kept=int(len(catalog.unique_ids(kept))) if kept.num_rows else 0,
    )


# --------------------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------------------


def label_pool(pool, split_cfg, last_observed_date_utc: str, history_end_utc: str | None = None):
    """The pool with a ``split`` column: every element set labeled as a window start.

    Thin wrapper over :func:`orbit_correction.splits.assign_split_labels` (seeded
    object draws, the time cutoff, the event exclusion and the 41-day data-end margin
    measured from ``last_observed_date_utc``), plus one rule of the downloaded data:

    * ``history_end_utc`` (ISO UTC, exclusive end of the element-set history; decided
      2026-11-25): a window of **any** role whose end (start + ``max_horizon_days``)
      is later than this is labeled ``excluded``. Windows must end within the data
      that was downloaded. Today that is 2026-01-01T00:00:00: the 2025 bulk file is
      the last one, so the test period is 2025; 2026 is added as a test-period
      extension when Space-Track publishes it.

    Returns:
        ``(table, summary, history_end_excluded)``: the labeled table,
        :func:`splits.split_summary` (row and object counts per label) and, per
        original label, the rows the history-end rule relabeled. Counts only:
        nothing about residuals or targets exists at this point.
    """
    import pyarrow as pa

    from orbit_correction import splits
    from orbit_correction.frames import utc_to_tt_seconds

    ids = pool.column("norad_id").to_numpy()
    start = pool.column("epoch_tt_s").to_numpy()
    labels = splits.assign_split_labels(
        ids, start, split_cfg, last_observed_date_utc=last_observed_date_utc,
    )
    relabeled: dict[str, int] = {}
    if history_end_utc is not None:
        end = start + split_cfg.max_horizon_days * _DAY_S
        late = (end > float(utc_to_tt_seconds(history_end_utc))) & (labels != "excluded")
        for lab in np.unique(labels[late]):
            relabeled[str(lab)] = int(np.sum(labels[late] == lab))
        labels[late] = "excluded"
    table = pool.append_column("split", pa.array(labels.tolist(), type=pa.string()))
    return table, splits.split_summary(labels, ids), relabeled


# --------------------------------------------------------------------------------------
# Window sampling
# --------------------------------------------------------------------------------------


def _draws(seed: int, salt: str, keys) -> np.ndarray:
    """Seeded hash draws in [0, 1), one per key (independent of the other keys)."""
    import hashlib

    out = np.empty(len(keys))
    for i, k in enumerate(keys):
        d = hashlib.sha256(f"{seed}:{salt}:{k}".encode()).digest()
        out[i] = int.from_bytes(d[:8], "big") / 2.0**64
    return out


def sample_windows(labeled, cfg: GenerationConfig, n_windows: int | None = None):
    """Choose window starts from a labeled pool (v1: starts at element-set epochs).

    Proposed rule (pending approval, ``config/generation.toml``):

    1. **Eligible starts**: element sets whose split label is in ``cfg.roles`` (train,
       dev). Test-role and reserved rows are never drawn.
    2. **Per object**: its eligible epochs are visited in the order of a seeded hash
       draw on ``(object, epoch)``, which is a uniform random order; a start is
       accepted if it is at least ``min_start_spacing_days`` from every start already
       accepted for that object, up to ``max_starts_per_object``. The result is that
       object's start list, in acceptance order.
    3. **Across objects**: round k takes the k-th start of every object that has one,
       objects in the order of a seeded draw on the object. Rounds are concatenated
       and the first ``n_windows`` are the sample. So every object gets one start
       before any gets two (many distinct objects before many starts per object), and
       a smaller stage is the first part of a larger one: the pilot's windows are the
       first windows of the first batch.

    Uniform over an object's *element sets* means periods with more element sets per
    day (better tracking) get proportionally more starts; the coverage report shows
    the result by year.

    Args:
        labeled: :func:`label_pool` output (must have ``split``).
        cfg: the generation config.
        n_windows: how many to return (None: every accepted start).

    Returns:
        The chosen rows (all pool columns), in sample order, with added columns
        ``window_index`` (0, 1, ...: position in the sample order), ``start_round``
        (1 for an object's first start) and ``draw`` (the start's hash draw).
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    mask = pc.is_in(labeled.column("split"), value_set=pa.array(list(cfg.roles)))
    elig = labeled.filter(mask)
    ids = elig.column("norad_id").to_numpy()
    t = elig.column("epoch_tt_s").to_numpy()
    draw = _draws(cfg.seed, "start", [f"{i}:{x!r}" for i, x in zip(ids, t, strict=True)])
    spacing = cfg.min_start_spacing_days * _DAY_S

    order = np.lexsort((draw, ids))  # by object, then draw
    per_obj: dict[int, list[int]] = {}
    for j in order:
        nid = int(ids[j])
        acc = per_obj.setdefault(nid, [])
        if len(acc) >= cfg.max_starts_per_object:
            continue
        if all(abs(t[j] - t[k]) >= spacing for k in acc):
            acc.append(int(j))
    objs = np.array(sorted(per_obj), dtype=np.int64)
    obj_draw = _draws(cfg.seed, "object", [int(o) for o in objs])
    obj_order = objs[np.argsort(obj_draw, kind="stable")]
    chosen: list[int] = []
    rounds: list[int] = []
    for r in range(cfg.max_starts_per_object):
        added = False
        for o in obj_order:
            acc = per_obj[int(o)]
            if r < len(acc):
                chosen.append(acc[r])
                rounds.append(r + 1)
                added = True
        if not added:
            break
    if n_windows is not None:
        if n_windows > len(chosen):
            raise ValueError(f"asked for {n_windows} windows, the pool gives {len(chosen)}")
        chosen, rounds = chosen[:n_windows], rounds[:n_windows]
    out = elig.take(pa.array(chosen, type=pa.int64()))
    out = out.append_column("window_index", pa.array(np.arange(len(chosen)), type=pa.int64()))
    out = out.append_column("start_round", pa.array(rounds, type=pa.int64()))
    return out.append_column("draw", pa.array(draw[chosen], type=pa.float64()))


def dailyap_subsample(windows, n: int, seed: int):
    """``n`` windows of a sample for the daily-Ap sensitivity run (seeded draw, salt ``dailyap``).

    Returned in ``window_index`` order. The same windows, so the two truth settings
    can be compared window by window.
    """
    import pyarrow as pa

    idx = windows.column("window_index").to_numpy()
    d = _draws(seed, "dailyap", [int(i) for i in idx])
    pick = np.sort(np.argsort(d, kind="stable")[: int(n)])
    return windows.take(pa.array(pick, type=pa.int64()))


def window_strata(windows, space_weather, residual_cfg, gen_cfg) -> dict[str, list]:
    """Reporting labels of each window: year, altitude band, solar bin, eccentricity bin.

    Band and solar bin are exactly the row builder's (``residuals.altitude_band``,
    ``residuals.solar_activity_bin`` of ``SpaceWeather.as_of`` at the start); the year
    is the start's UTC year; the eccentricity bin uses ``gen_cfg.eccentricity_edges``.
    Labels only (element-set and space-weather values at the start), no residuals.
    """
    from orbit_correction import residuals as R

    out: dict[str, list] = {"year": [], "altitude_band": [], "solar_activity_bin": [],
                            "eccentricity_bin": []}
    for row in windows.select(["epoch_utc", "epoch_tt_s", "perigee_alt_km",
                               "eccentricity"]).to_pylist():
        out["year"].append(int(row["epoch_utc"][:4]))
        out["altitude_band"].append(R.altitude_band(row["perigee_alt_km"], residual_cfg))
        asof = space_weather.as_of(float(row["epoch_tt_s"]))
        out["solar_activity_bin"].append(
            R.solar_activity_bin(asof.f107_obs_trailing81_sfu, residual_cfg)
        )
        out["eccentricity_bin"].append(
            R.bin_label(float(row["eccentricity"]), gen_cfg.eccentricity_edges)
        )
    return out


def coverage(strata: dict[str, list], roles: Iterable[str]) -> dict[str, dict[str, int]]:
    """Window counts per value of each stratum label, plus per split role."""
    from collections import Counter

    out = {k: dict(sorted(Counter(v).items())) for k, v in strata.items()}
    out["split"] = dict(sorted(Counter(roles).items()))
    return out


# --------------------------------------------------------------------------------------
# One window's fate
# --------------------------------------------------------------------------------------

#: Statuses of windows that never reach the row builder, decided in this module:
#: start_below_floor: the SGP4 state at the element-set epoch is already below the 120 km
#:   floor (``cheap_prop.sgp4_epoch_states`` gives no state), so there is no truth job.
#: start_sgp4_error: SGP4 flags an error code at the epoch itself; no state, no job.
#: fit_samples_unavailable: under the fitted start rule, SGP4 can't give the samples
#:   before the epoch the fit needs (an SGP4 error code or the 120 km floor somewhere in
#:   ``[epoch - fit span, epoch]``). No job is made and no other start is used.
PRE_TRUTH_STATUSES = ("start_below_floor", "start_sgp4_error", "fit_samples_unavailable")


def generation_statuses() -> tuple[str, ...]:
    """Every status a sampled window can end with: the row builder's plus the two above."""
    from orbit_correction.residuals import WINDOW_STATUSES

    return (*WINDOW_STATUSES, *PRE_TRUTH_STATUSES)


def window_result(elset_row, sgp4, truth, space_weather, split, dt_s, residual_cfg, *,
                  include_observed: bool = False):
    """:func:`residuals.build_window_rows`, with a domain refusal counted as ``rtn_domain``.

    The row builder raises :class:`residuals.RTNDomainError` when a residual is outside
    the conversion's domain (the pi/2 along-track limit is reachable at the low,
    high-drag edge). Here that becomes a dropped window with status ``rtn_domain``,
    in the window's stratum, with the error message kept, so it is counted like every
    other drop. Any other exception propagates (it is a bug, not a data outcome).
    """
    from orbit_correction import residuals as R

    try:
        return R.build_window_rows(elset_row, sgp4, truth, space_weather, split, dt_s,
                                   residual_cfg, include_observed=include_observed)
    except R.RTNDomainError as exc:
        asof = space_weather.as_of(float(elset_row["epoch_tt_s"]))
        return R.WindowResult(
            norad_id=int(elset_row["norad_id"]),
            start_epoch_tt_s=float(elset_row["epoch_tt_s"]),
            split=split,
            status="rtn_domain",
            causes=("rtn_domain",),
            altitude_band=R.altitude_band(float(elset_row["perigee_alt_km"]), residual_cfg),
            solar_activity_bin=R.solar_activity_bin(asof.f107_obs_trailing81_sfu, residual_cfg),
            table=None,
            sgp4_code=int(sgp4.status.code),
            truth_status=str(truth.status),
            message=f"RTNDomainError: {exc}",
        )


def pre_truth_result(elset_row, start, space_weather, split, residual_cfg, *,
                     fit_samples=None):
    """WindowResult for a window that gets no truth job.

    Either its start has no SGP4 state (``start``'s status: ``start_below_floor`` or
    ``start_sgp4_error``), or the start exists but the fitted rule's SGP4 samples before
    the epoch don't (``fit_samples`` given with a status that isn't ok:
    ``fit_samples_unavailable``). The start is tested first.
    """
    from orbit_correction import residuals as R

    if start.trajectory is None:
        status = "start_below_floor" if start.status.below_floor else "start_sgp4_error"
        detail = start.status
    elif fit_samples is not None and not fit_samples.status.ok:
        status, detail = "fit_samples_unavailable", fit_samples.status
    else:
        raise ValueError("the window has a start state and usable fit samples")
    asof = space_weather.as_of(float(elset_row["epoch_tt_s"]))
    return R.WindowResult(
        norad_id=int(elset_row["norad_id"]),
        start_epoch_tt_s=float(elset_row["epoch_tt_s"]),
        split=split,
        status=status,
        causes=(status,),
        altitude_band=R.altitude_band(float(elset_row["perigee_alt_km"]), residual_cfg),
        solar_activity_bin=R.solar_activity_bin(asof.f107_obs_trailing81_sfu, residual_cfg),
        table=None,
        sgp4_code=int(detail.code),
        truth_status="not_run",
        message=detail.message,
    )


# --------------------------------------------------------------------------------------
# Truth workers (spawned; one JVM each)
# --------------------------------------------------------------------------------------

_GEN_WORKER: dict[str, object] = {}


def _gen_worker_init(truth_cfg, sw_copy, apc: int) -> None:
    """Initializer of a generation worker: start its JVM once, keep the settings."""
    from orbit_correction import truth_prop

    truth_prop.start_jvm(apc)
    _GEN_WORKER.update(cfg=truth_cfg, sw=sw_copy)


def _gen_worker_run(job):
    """One truth job in a worker; an exception becomes ``status="error"`` (counted)."""
    from orbit_correction import truth_prop

    try:
        return truth_prop.propagate_truth(
            job.initial, job.bstar_per_er, job.t_out_tt_s, _GEN_WORKER["cfg"], _GEN_WORKER["sw"],
            area_to_mass_m2_per_kg=job.area_to_mass_m2_per_kg, fit_samples=job.fit_samples,
        )
    except Exception as exc:  # noqa: BLE001  (reported per window; the run continues)
        return truth_prop.TruthResult(
            norad_id=job.initial.norad_id, status="error",
            message=f"{type(exc).__name__}: {exc}",
        )


class TruthWorkerPool:
    """A pool of spawned truth workers kept for a whole run (warm JVMs across chunks).

    ``truth_prop.propagate_batch`` starts and stops a pool per call, so a chunked run
    would pay every worker's JVM start and slow first propagation (5-28 s measured)
    once per chunk. This pool starts them once. Workers call only public
    ``truth_prop`` functions (``start_jvm``, ``propagate_truth``), and every job's
    window is checked against the space-weather copy before a chunk is submitted, as
    ``propagate_batch`` does. The calling process never starts a JVM.

    Use as a context manager; :meth:`run` returns results in job order.
    """

    def __init__(self, truth_cfg, sw_copy, n_workers: int, active_processor_count=None):
        self.truth_cfg = truth_cfg
        self.sw_copy = sw_copy
        self.n_workers = int(n_workers)
        self.apc = (truth_cfg.jvm_active_processor_count if active_processor_count is None
                    else int(active_processor_count))
        self._pool = None

    def __enter__(self):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        self._pool = ProcessPoolExecutor(
            max_workers=self.n_workers, mp_context=mp.get_context("spawn"),
            initializer=_gen_worker_init, initargs=(self.truth_cfg, self.sw_copy, self.apc),
        )
        return self

    def __exit__(self, *exc) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def restart(self) -> None:
        """Replace a broken pool (a worker died) with a fresh one: new workers, new JVMs."""
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        self.__enter__()

    def run(self, jobs):
        """Run jobs in the workers; results in job order.

        Every job is checked first, as ``truth_prop.propagate_batch`` checks: fit samples
        present under the fitted start rule and absent under any other, and the window,
        from the first fit sample under the fitted rule, inside the space-weather copy.
        One failing job refuses the chunk. A worker that dies raises
        ``concurrent.futures.process.BrokenProcessPool`` (handled by the caller).
        """
        from orbit_correction.truth_prop import FIT_START_RULE, SpaceWeatherRangeError

        fitted = self.truth_cfg.start_rule == FIT_START_RULE
        for i, job in enumerate(jobs):
            if fitted and job.fit_samples is None:
                raise ValueError(f"job {i} (NORAD {job.initial.norad_id}): the fitted start "
                                 "rule needs fit_samples")
            if not fitted and job.fit_samples is not None:
                raise ValueError(f"job {i}: fit_samples given but the start rule is "
                                 f"{self.truth_cfg.start_rule!r}")
            if self.truth_cfg.atmosphere != "none":
                first = float(job.fit_samples.t_tt_s[0]) if fitted else float(job.initial.t_tt_s[0])
                try:
                    self.sw_copy.check_window(first, float(np.asarray(job.t_out_tt_s)[-1]))
                except SpaceWeatherRangeError as exc:
                    raise SpaceWeatherRangeError(
                        f"job {i} (NORAD {job.initial.norad_id}): {exc}") from None
        if self._pool is None:
            raise RuntimeError("use TruthWorkerPool as a context manager")
        return list(self._pool.map(_gen_worker_run, jobs, chunksize=1))


# --------------------------------------------------------------------------------------
# Chunked runs
# --------------------------------------------------------------------------------------

#: Window-table columns carried into the generation log.
_LOG_ELSET_COLUMNS = ("window_index", "norad_id", "epoch_utc", "epoch_tt_s", "split",
                      "start_round")


@dataclass(frozen=True)
class RunSettings:
    """What a run is, recorded in its plan file and checked on resume.

    Attributes:
        run_id: name of the run directory.
        label: ``"baseline"`` or a sensitivity label (``"dailyap"``): it goes into
            every output (file metadata, log, manifest).
        chunk_windows: windows per chunk.
        n_workers: truth workers.
        include_observed: fill the ``obs_*`` columns.
    """

    run_id: str
    label: str
    chunk_windows: int
    n_workers: int
    include_observed: bool = False


def chunk_bounds(n_windows: int, chunk_windows: int) -> list[tuple[int, int]]:
    """``[(start, stop), ...]`` row ranges of consecutive chunks covering ``n_windows``."""
    if chunk_windows < 1:
        raise ValueError("chunk_windows must be >= 1")
    return [(a, min(a + chunk_windows, n_windows)) for a in range(0, n_windows, chunk_windows)]


def _ids_digest(windows) -> str:
    import hashlib

    idx = windows.column("window_index").to_numpy().astype(np.int64)
    nid = windows.column("norad_id").to_numpy().astype(np.int64)
    t = windows.column("epoch_tt_s").to_numpy().astype(np.float64)
    return hashlib.sha256(idx.tobytes() + nid.tobytes() + t.tobytes()).hexdigest()


def _chunk_paths(out_dir: Path, k: int) -> tuple[Path, Path, Path]:
    stem = out_dir / f"chunk_{k:05d}"
    return (stem.with_suffix(".parquet"), stem.with_suffix(".log.jsonl"),
            stem.with_suffix(".done.json"))


def chunk_is_complete(out_dir: str | Path, k: int, windows_digest: str) -> bool:
    """True if chunk ``k`` was written and verified: marker present, hashes and windows match.

    A chunk counts as done only when its marker (written last) names the same windows
    and both files on disk still hash to what the marker recorded. Anything else (no
    marker, a partial file, a file changed since) means the chunk is redone.
    """
    import json

    parquet, log, done = _chunk_paths(Path(out_dir), k)
    if not (done.exists() and parquet.exists() and log.exists()):
        return False
    try:
        m = json.loads(done.read_text())
    except ValueError:
        return False
    return (
        m.get("windows_sha256") == windows_digest
        and m.get("parquet_sha256") == file_sha256(parquet)
        and m.get("log_sha256") == file_sha256(log)
    )


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def run_chunk(windows, k: int, out_dir: Path, settings: RunSettings, dt_s, residual_cfg,
              space_weather, strata, truth_runner, truth_cfg) -> dict:
    """Generate one chunk: SGP4 here, truth through ``truth_runner``, rows, files, marker.

    Steps: SGP4 on the horizon grid, the epoch states and, under the fitted start rule,
    the fit samples before each epoch (``truth_prop.fit_samples_from_elsets``), all in
    this process with no JVM; a truth job for every window with a start state and,
    under the fitted rule, usable fit samples (``start_below_floor``,
    ``start_sgp4_error`` and ``fit_samples_unavailable`` windows get none, and no
    other start is tried); truth results; one :class:`WindowResult`
    per window (``rtn_domain`` caught); ``ok`` rows written as one zstd Parquet file;
    one log record per window; then the marker with both files' hashes. Files are
    written to temporary names and renamed, so a crash leaves no file that looks done.

    Args:
        windows: this chunk's rows of the window table (catalog columns, ``split``,
            ``window_index``...).
        k: chunk number.
        strata: :func:`window_strata` for these rows (the log carries the labels).
        truth_runner: callable ``jobs -> [TruthResult]`` in job order
            (:meth:`TruthWorkerPool.run`, or a stand-in in tests).
        truth_cfg: the truth configuration the runner uses (its start rule decides
            whether fit samples are built and passed).

    Returns:
        The marker's content (counts, hashes, timings).
    """
    import json
    import time

    import pyarrow as pa
    import pyarrow.parquet as pq

    from orbit_correction import cheap_prop
    from orbit_correction.truth_prop import FIT_START_RULE, TruthJob, fit_samples_from_elsets
    from orbit_correction.types import RESIDUAL_SCHEMA

    parquet, log, done = _chunk_paths(out_dir, k)
    dt = np.asarray(dt_s, dtype=np.float64)
    rows = windows.to_pylist()
    t0 = time.perf_counter()
    sgp4 = cheap_prop.propagate_sgp4_from_epoch(windows, dt)
    starts = cheap_prop.sgp4_epoch_states(windows)
    fitted = truth_cfg.start_rule == FIT_START_RULE
    fits = fit_samples_from_elsets(windows, truth_cfg) if fitted else [None] * len(rows)
    t_sgp4 = time.perf_counter() - t0

    jobs, job_of = [], {}
    for i, (row, st, fit) in enumerate(zip(rows, starts, fits, strict=True)):
        if st.trajectory is None or (fit is not None and not fit.status.ok):
            continue
        job_of[i] = len(jobs)
        jobs.append(TruthJob(initial=st.trajectory,
                             bstar_per_er=float(row["bstar_per_earth_radii"]),
                             t_out_tt_s=float(row["epoch_tt_s"]) + dt,
                             fit_samples=None if fit is None else fit.trajectory))
    t1 = time.perf_counter()
    truths = truth_runner(jobs) if jobs else []
    t_truth = time.perf_counter() - t1
    if len(truths) != len(jobs):
        raise RuntimeError(f"truth runner returned {len(truths)} results for {len(jobs)} jobs")

    tables, records, results = [], [], []
    t2 = time.perf_counter()
    for i, row in enumerate(rows):
        tw = time.perf_counter()
        truth = None
        if i in job_of:
            truth = truths[job_of[i]]
            res = window_result(row, sgp4[i], truth, space_weather, row["split"], dt,
                                residual_cfg, include_observed=settings.include_observed)
        else:
            res = pre_truth_result(row, starts[i], space_weather, row["split"], residual_cfg,
                                   fit_samples=fits[i])
        if res.table is not None:
            tables.append(res.table)
        results.append(res)
        rec = {c: row[c] for c in _LOG_ELSET_COLUMNS}
        rec.update(
            run_id=settings.run_id, label=settings.label,
            year=strata["year"][i], altitude_band=res.altitude_band,
            solar_activity_bin=res.solar_activity_bin,
            eccentricity_bin=strata["eccentricity_bin"][i],
            status=res.status, causes=list(res.causes), sgp4_code=res.sgp4_code,
            sgp4_n_valid=int(sgp4[i].status.n_valid), truth_status=res.truth_status,
            message=res.message,
            truth_wall_s=None if truth is None else truth.wall_s,
            truth_n_evaluations=None if truth is None else truth.n_evaluations,
            truth_start_scale=None if truth is None else truth.start_scale,
            **_fit_fields(truth),
            rows_s=time.perf_counter() - tw,
            n_rows=res.n_rows, chunk=k, chunk_file=parquet.name,
        )
        if res.altitude_band != strata["altitude_band"][i] or (
            res.solar_activity_bin != strata["solar_activity_bin"][i]
        ):
            raise RuntimeError(f"window {row['window_index']}: stratum labels disagree")
        records.append(rec)
    t_rows = time.perf_counter() - t2

    table = pa.concat_tables(tables) if tables else RESIDUAL_SCHEMA.empty_table()
    table = table.replace_schema_metadata({
        "run_id": settings.run_id, "label": settings.label, "chunk": str(k),
    })
    tmp = parquet.with_name(parquet.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(parquet)
    _atomic_write_text(log, "".join(json.dumps(r) + "\n" for r in records))
    marker = {
        "chunk": k,
        "run_id": settings.run_id,
        "label": settings.label,
        "window_index_first": int(rows[0]["window_index"]),
        "window_index_last": int(rows[-1]["window_index"]),
        "n_windows": len(rows),
        "n_truth_jobs": len(jobs),
        "n_ok": sum(r.status == "ok" for r in results),
        "n_rows": table.num_rows,
        "parquet_bytes": parquet.stat().st_size,
        "parquet_sha256": file_sha256(parquet),
        "log_sha256": file_sha256(log),
        "windows_sha256": _ids_digest(windows),
        "sgp4_s": t_sgp4,
        "truth_s": t_truth,
        "rows_s": t_rows,
        "wall_s": time.perf_counter() - t0,
    }
    _atomic_write_text(done, json.dumps(marker, indent=1) + "\n")
    return marker


def _fit_fields(truth) -> dict:
    """The fitted start's record for the log (all None without a fit)."""
    fit = None if truth is None else getattr(truth, "start_fit", None)
    if fit is None:
        return {"fit_status": None, "fit_iterations": None, "fit_rms_m": None,
                "fit_max_m": None, "fit_wall_s": None, "fit_n_evaluations": None}
    return {"fit_status": fit.status, "fit_iterations": fit.iterations, "fit_rms_m": fit.rms_m,
            "fit_max_m": fit.max_m, "fit_wall_s": fit.wall_s,
            "fit_n_evaluations": fit.n_evaluations}


def read_logs(out_dir: str | Path) -> list[dict]:
    """Every chunk's log records, in chunk order (the generation log of a run)."""
    import json

    out: list[dict] = []
    for p in sorted(Path(out_dir).glob("chunk_*.log.jsonl")):
        out.extend(json.loads(line) for line in p.read_text().splitlines() if line)
    return out


def status_counts(records: Iterable[dict]) -> dict[str, int]:
    """Windows per (status, altitude band, solar bin) from log records.

    The same counting as :func:`residuals.count_windows`, keyed ``"status|band|bin"``
    so it can be written as JSON.
    """
    from orbit_correction.residuals import WindowResult, count_windows

    results = [
        WindowResult(r["norad_id"], r["epoch_tt_s"], r["split"], r["status"], tuple(r["causes"]),
                     r["altitude_band"], r["solar_activity_bin"], None, r["sgp4_code"],
                     r["truth_status"])
        for r in records
    ]
    return {"|".join(k): v for k, v in count_windows(results).items()}


def code_provenance(repo: str | Path = REPO) -> dict:
    """The code commit a run used, and whether the working tree had changes."""
    import subprocess

    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True,
                              check=False).stdout.strip()

    return {"commit": git("rev-parse", "HEAD") or None,
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def run_generation(windows, out_dir: str | Path, settings: RunSettings, *, dt_s, residual_cfg,
                   space_weather, strata, truth_runner, truth_cfg, restart_runner=None,
                   max_chunk_attempts: int = 3, manifest_extra: dict | None = None,
                   progress=None) -> dict:
    """Run (or resume) a chunked generation over a window table and write its manifest.

    Chunks are fixed by the window order and ``settings.chunk_windows``. A plan file
    (``run_plan.json``) records the run's identity on the first call; a later call with
    a different window table, chunk size or label is refused, so a resumed run can't
    mix two runs. Chunks already complete (:func:`chunk_is_complete`) are skipped;
    the rest are generated. Memory is bounded by one chunk.

    Args:
        windows: the window table (sample order).
        out_dir: the run directory.
        settings: :class:`RunSettings`.
        dt_s: the horizon grid.
        residual_cfg, space_weather: for the row builder.
        strata: :func:`window_strata` of ``windows`` (all rows).
        truth_runner: ``jobs -> [TruthResult]``.
        truth_cfg: the truth configuration the runner uses. Its settings are part of
            the plan, so a resume with other truth settings is refused.
        restart_runner: called after the worker pool breaks (a worker process died:
            ``BrokenProcessPool``), before the chunk is retried; for
            :class:`TruthWorkerPool`, its ``restart``. Nothing of a crashed chunk is
            kept (files are written only after truth returns). Each crash is appended
            to ``crashes.jsonl`` and counted in the manifest; after
            ``max_chunk_attempts`` failed attempts on one chunk the run stops.
        manifest_extra: recorded in the manifest (configs, inputs, space-weather copy).
        progress: optional callable(marker) after each new chunk.

    Returns:
        The manifest written to ``run_manifest.json``.
    """
    import dataclasses
    import datetime as dt_mod
    import hashlib
    import json
    from concurrent.futures.process import BrokenProcessPool

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    digest = _ids_digest(windows)
    truth_json = json.dumps(dataclasses.asdict(truth_cfg), sort_keys=True)
    plan = {
        "run_id": settings.run_id, "label": settings.label,
        "chunk_windows": settings.chunk_windows, "include_observed": settings.include_observed,
        "n_windows": windows.num_rows, "windows_sha256": digest,
        "horizons": int(np.asarray(dt_s).size),
        "truth_settings_sha256": hashlib.sha256(truth_json.encode()).hexdigest(),
        "start_rule": truth_cfg.start_rule, "msis_ap_switch": truth_cfg.msis_ap_switch,
    }
    plan_path = out / "run_plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        diff = {k: (old.get(k), v) for k, v in plan.items() if old.get(k) != v}
        if diff:
            raise ValueError(f"{out} holds a different run; differing plan fields: {diff}")
    else:
        _atomic_write_text(plan_path, json.dumps(plan, indent=1) + "\n")

    bounds = chunk_bounds(windows.num_rows, settings.chunk_windows)
    new, skipped = [], []
    for k, (a, b) in enumerate(bounds):
        part = windows.slice(a, b - a)
        if chunk_is_complete(out, k, _ids_digest(part)):
            skipped.append(k)
            continue
        part_strata = {key: v[a:b] for key, v in strata.items()}
        attempt = 0
        while True:
            attempt += 1
            try:
                marker = run_chunk(part, k, out, settings, dt_s, residual_cfg, space_weather,
                                   part_strata, truth_runner, truth_cfg)
                break
            except BrokenProcessPool as exc:
                crash = {"chunk": k, "attempt": attempt,
                         "utc": dt_mod.datetime.now(dt_mod.UTC).isoformat(timespec="seconds"),
                         "error": f"{type(exc).__name__}: {exc}"}
                with open(out / "crashes.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(crash) + "\n")
                if attempt >= max_chunk_attempts:
                    raise RuntimeError(
                        f"chunk {k}: the worker pool broke on {attempt} attempts; stopping"
                    ) from exc
                if restart_runner is not None:
                    restart_runner()
        new.append(k)
        if progress is not None:
            progress(marker)

    markers = [json.loads(_chunk_paths(out, k)[2].read_text()) for k in range(len(bounds))]
    records = read_logs(out)
    if len(records) != windows.num_rows:
        raise RuntimeError(f"log has {len(records)} records for {windows.num_rows} windows")
    from collections import Counter

    crash_path = out / "crashes.jsonl"
    crashes = ([json.loads(x) for x in crash_path.read_text().splitlines() if x]
               if crash_path.exists() else [])
    manifest = {
        "plan": plan,
        "code": code_provenance(),
        "n_chunks": len(bounds),
        "chunks_written_this_call": new,
        "chunks_skipped_this_call": skipped,
        "n_windows": windows.num_rows,
        "n_rows": sum(m["n_rows"] for m in markers),
        "parquet_bytes": sum(m["parquet_bytes"] for m in markers),
        "status": dict(sorted(Counter(r["status"] for r in records).items())),
        "status_by_split": {
            f"{k[0]}|{k[1]}": v
            for k, v in sorted(Counter((r["split"], r["status"]) for r in records).items())
        },
        "counts_by_status_band_bin": status_counts(records),
        "timing_s": {
            "sgp4": sum(m["sgp4_s"] for m in markers),
            "truth": sum(m["truth_s"] for m in markers),
            "rows": sum(m["rows_s"] for m in markers),
            "chunks_wall": sum(m["wall_s"] for m in markers),
            "truth_job_wall_sum": sum(r["truth_wall_s"] or 0.0 for r in records),
        },
        "n_workers": settings.n_workers,
        "worker_pool_crashes": len(crashes),
        "worker_pool_crash_records": crashes,
        "fit_status": dict(sorted(Counter(str(r.get("fit_status")) for r in records).items())),
        **(manifest_extra or {}),
    }
    _atomic_write_text(out / "run_manifest.json", json.dumps(manifest, indent=1) + "\n")
    return manifest


__all__ = [
    "CANDIDATE_DROP_REASONS",
    "CONFIG_PATH",
    "PRE_TRUTH_STATUSES",
    "DedupeStats",
    "GenerationConfig",
    "PoolResult",
    "RunSettings",
    "TruthWorkerPool",
    "build_pool",
    "chunk_bounds",
    "chunk_is_complete",
    "code_provenance",
    "coverage",
    "dailyap_subsample",
    "drop_exact_duplicates",
    "file_sha256",
    "generation_statuses",
    "label_pool",
    "load_generation_config",
    "pre_truth_result",
    "read_logs",
    "run_chunk",
    "run_generation",
    "sample_windows",
    "status_counts",
    "window_result",
    "window_strata",
    "CandidateResult",
    "candidate_ids_from_satcat",
    "TleFormatError",
    "TleNoChecksumError",
    "TleReadStats",
    "download_sample_ids",
    "iter_tle_records",
    "tle_checksum",
    "tle_epoch_to_iso",
    "tle_norad_id",
    "tle_to_record",
]
