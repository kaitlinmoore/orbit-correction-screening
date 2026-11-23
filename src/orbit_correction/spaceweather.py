"""CelesTrak space-weather file (F10.7, Kp, Ap): parsing, leak-free lookups, Orekit copy.

Why this module exists
----------------------
Space weather enters the project twice, under different rules:

* **Model inputs** may only use what was known at the prediction start ``t0``.
  :meth:`SpaceWeather.as_of` enforces that: it reads only OBSERVED rows, only
  days complete by ``t0``, and offers a *trailing* 81-day F10.7 mean instead of
  the file's centred one (which averages 40 days after the date).
* **The truth propagator** uses the physical values over the prediction
  horizon. :meth:`SpaceWeather.observed` returns those, centred averages
  included, and says which file section each value came from.

It also writes a copy of the file that Orekit 13.x can parse
(:func:`write_orekit_cssi`), because Orekit's CSSI parser fails on the file as
CelesTrak currently publishes it (project research, 2026-11-23):

1. predicted rows leave the F10.7 quality flag ``Q`` (columns 99-100) blank;
2. MONTHLY_PREDICTED rows leave every Kp/Ap/Cp/C9 field (columns 19-88) blank;
3. there is no MONTHLY_FIT section (Orekit throws on lookup without one).

The parser tolerates all three and keeps blanks as missing (NaN): a placeholder
is never returned as a value. The writer fills blanks only where Orekit needs a
number to parse the line and only on dates outside the range the project will
propagate; inside that range it refuses (:class:`PlaceholderInRangeError`).

File format
-----------
Fixed width, ``FORMAT(I4,I3,I3,I5,I3,8I3,I4,8I4,I4,F4.1,I2,I4,F6.1,I2,5F6.1)``,
130 columns, described at https://celestrak.org/SpaceData/SpaceWx-format.php.
Kp values are stored as Kp x 10 (``43`` means 4.3, i.e. 4+), as in the file.
Sections: OBSERVED (daily, measured), DAILY_PREDICTED (daily forecast, about
45 days), MONTHLY_PREDICTED (one row per month, dated the 1st), and optionally
MONTHLY_FIT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# Format
# --------------------------------------------------------------------------------------

#: Line width of a data row.
LINE_WIDTH = 130

SECTIONS = ("OBSERVED", "DAILY_PREDICTED", "MONTHLY_PREDICTED", "MONTHLY_FIT")


@dataclass(frozen=True)
class _Col:
    name: str
    start: int  # 0-based start column
    width: int
    decimals: int | None  # None for integer fields

    def fmt(self, value: float) -> str:
        if self.decimals is None:
            return f"{int(round(value)):>{self.width}d}"
        return f"{value:>{self.width}.{self.decimals}f}"


def _build_columns() -> tuple[_Col, ...]:
    cols: list[_Col] = []
    pos = 0

    def add(name: str, width: int, decimals: int | None = None) -> None:
        nonlocal pos
        cols.append(_Col(name, pos, width, decimals))
        pos += width

    add("year", 4)
    add("month", 3)
    add("day", 3)
    add("bsrn", 5)
    add("nd", 3)
    for k in range(8):
        add(f"kp_x10_{k}", 3)
    add("kp_sum_x10", 4)
    for k in range(8):
        add(f"ap_{k}", 4)
    add("ap_avg", 4)
    add("cp", 4, 1)
    add("c9", 2)
    add("isn", 4)
    add("f107_adj", 6, 1)
    add("q", 2)
    add("f107_adj_ctr81", 6, 1)
    add("f107_adj_lst81", 6, 1)
    add("f107_obs", 6, 1)
    add("f107_obs_ctr81", 6, 1)
    add("f107_obs_lst81", 6, 1)
    assert pos == LINE_WIDTH
    return tuple(cols)


_COLUMNS = _build_columns()
_DATE_COLS = ("year", "month", "day")
#: Scalar (one value per day) fields, in file order. Kp and ap 3-hourly values are
#: stored separately as (N, 8) arrays.
SCALAR_FIELDS = (
    "bsrn", "nd", "kp_sum_x10", "ap_avg", "cp", "c9", "isn", "f107_adj", "q",
    "f107_adj_ctr81", "f107_adj_lst81", "f107_obs", "f107_obs_ctr81", "f107_obs_lst81",
)
#: Fields Orekit needs a number in to parse a line (blank -> parse error). The
#: writer may fill these outside the propagation range, never inside.
_OREKIT_REQUIRED_TO_PARSE = (
    *(f"kp_x10_{k}" for k in range(8)), "kp_sum_x10",
    *(f"ap_{k}" for k in range(8)), "ap_avg", "cp", "c9", "q",
)


class SpaceWeatherFormatError(ValueError):
    """A line or section of the file does not match the published format."""


class PlaceholderInRangeError(ValueError):
    """The writer would have had to put a placeholder into a date the project propagates."""


# --------------------------------------------------------------------------------------
# Parsed table
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpaceWeather:
    """A parsed space-weather file. Missing values are NaN; nothing is filled.

    Attributes:
        dates: (N,) ``datetime64[D]`` UTC dates, one per row, in file order.
            MONTHLY rows carry the 1st of their month.
        section: (N,) section name of each row (see :data:`SECTIONS`).
        kp_x10: (N, 8) Kp x 10 for the eight 3-hour intervals (00-03 UT first).
        ap: (N, 8) ap for the eight 3-hour intervals.
        scalars: name -> (N,) float64 for each of :data:`SCALAR_FIELDS`.
        header: ``DATATYPE``, ``VERSION`` and ``UPDATED`` lines as found.
        had_monthly_fit: whether the source file had a MONTHLY_FIT section.
    """

    dates: np.ndarray
    section: np.ndarray
    kp_x10: np.ndarray
    ap: np.ndarray
    scalars: dict[str, np.ndarray]
    header: dict[str, str] = field(default_factory=dict)
    had_monthly_fit: bool = False

    # ---------------------------------------------------------------- basic access

    def __len__(self) -> int:
        return int(self.dates.shape[0])

    def rows_in(self, section: str) -> np.ndarray:
        """Indices of the rows in one section."""
        return np.flatnonzero(self.section == section)

    def value(self, name: str, i: int) -> float:
        """One field of one row by name (``kp_x10_3``, ``ap_0``, ``f107_obs``...)."""
        if name.startswith("kp_x10_") and name != "kp_x10":
            return float(self.kp_x10[i, int(name[-1])])
        if name.startswith("ap_") and name[3:].isdigit():
            return float(self.ap[i, int(name[3:])])
        return float(self.scalars[name][i])

    def _observed_index(self) -> tuple[np.ndarray, np.ndarray]:
        idx = self.rows_in("OBSERVED")
        return self.dates[idx], idx

    def last_fully_observed_date(self) -> str:
        """Latest UTC date (``YYYY-MM-DD``) with a complete OBSERVED row.

        Complete means every numeric field of the row is present (the 3-hour
        Kp and ap values, daily Ap, F10.7 observed and adjusted, their 81-day
        means, the quality flag...). Later dates exist only as forecasts, so
        ``splits.assign_split_labels`` excludes prediction windows ending
        after this day.

        Note: the centred 81-day means of the last ~40 observed days average
        over forecast days too; this method does not look into that.

        Raises:
            LookupError: if no OBSERVED row is complete.
        """
        numeric = [c.name for c in _COLUMNS[3:]]
        for i in self.rows_in("OBSERVED")[::-1]:
            if all(np.isfinite(self.value(n, int(i))) for n in numeric):
                return str(self.dates[i])
        raise LookupError("no complete OBSERVED row in the file")

    def _row_for_date(self, day: np.datetime64) -> int | None:
        """Row index whose values apply to ``day``: a daily row, else the month's row."""
        daily = (self.section == "OBSERVED") | (self.section == "DAILY_PREDICTED")
        hit = np.flatnonzero(daily & (self.dates == day))
        if hit.size:
            return int(hit[0])
        month_start = day.astype("datetime64[M]").astype("datetime64[D]")
        hit = np.flatnonzero((self.section == "MONTHLY_PREDICTED") & (self.dates == month_start))
        return int(hit[0]) if hit.size else None

    # ---------------------------------------------------------------- lookups

    def as_of(self, t0_tt_s: float, extra_lag_days: int = 0) -> SpaceWeatherAsOf:
        """Space weather known at the prediction start ``t0``. For model inputs.

        Availability rule:
          * Only OBSERVED rows are read. Predicted rows in the file are forecasts
            made when the file was produced, not at ``t0``.
          * Rows dated after ``t0``'s UTC date are never read.
          * A day's daily values (F10.7, daily Ap, Kp sum) count as known once
            the day is over: day D is usable from D+1 00:00 UTC (+ ``extra_lag_days``).
          * A 3-hour ap value counts as known once its interval has ended
            (+ ``extra_lag_days``).
          * The 81-day F10.7 mean is *trailing*: the mean of observed F10.7 over
            the 81 known days ending on the last complete day. It is NaN unless
            all 81 values are present. The file's centred average is never used.

        The model's four space-weather inputs (``types.SPACE_WEATHER_MODEL_INPUTS``,
        approved 2026-11-23, with this rule and ``extra_lag_days = 0``) are
        ``f107_obs_sfu``, ``f107_obs_trailing81_sfu``, ``ap_daily`` and
        ``ap_3h_last`` of the result, stored as the ``asof_*`` residual columns.
        ``f107_adj_sfu`` and ``kp_sum`` are returned for inspection only.

        The file is a snapshot of final, revised values; what an operator knew
        in real time at ``t0`` may have differed. That is a limitation of the
        data, not of this rule.

        Args:
            t0_tt_s: prediction start, TT seconds since J2000.0.
            extra_lag_days: additional publication delay, whole days (default 0).
        """
        t0 = _tt_seconds_to_utc_datetime64(t0_tt_s)
        t0_day = t0.astype("datetime64[D]")
        obs_dates, obs_idx = self._observed_index()
        lag = np.timedelta64(extra_lag_days, "D")
        # Rows dated on or before t0's date: the only rows this method may read.
        readable = obs_dates <= t0_day
        dates = obs_dates[readable]
        idx = obs_idx[readable]

        complete = (dates + np.timedelta64(1, "D") + lag).astype("datetime64[us]") <= t0
        last_complete = int(np.flatnonzero(complete)[-1]) if complete.any() else None

        out = SpaceWeatherAsOf(t0_utc=str(t0))
        if last_complete is not None:
            i = int(idx[last_complete])
            out.last_complete_day = str(dates[last_complete])
            out.f107_obs_sfu = float(self.scalars["f107_obs"][i])
            out.f107_adj_sfu = float(self.scalars["f107_adj"][i])
            out.ap_daily = float(self.scalars["ap_avg"][i])
            out.kp_sum = _div10(self.scalars["kp_sum_x10"][i])
            out.f107_obs_trailing81_sfu = _trailing_mean(
                self.scalars["f107_obs"], idx, dates, last_complete, 81
            )
        # Last 3-hour ap interval that has ended by t0 (may be in t0's own day).
        ends = (
            dates.astype("datetime64[us]")[:, None]
            + np.arange(1, 9)[None, :] * np.timedelta64(3, "h").astype("timedelta64[us]")
            + lag
        )
        done = ends <= t0
        if done.any():
            flat = np.flatnonzero(done.ravel())[-1]
            r, k = divmod(int(flat), 8)
            out.ap_3h_last = float(self.ap[idx[r], k])
            out.ap_3h_last_end_utc = str(
                dates[r].astype("datetime64[us]") + np.timedelta64(3 * (k + 1), "h")
            )
        out.rows_read_max_date = str(dates[-1]) if dates.size else None
        return out

    def observed(self, t_tt_s: float) -> SpaceWeatherObserved:
        """Physical space weather at time ``t``. For the truth propagator.

        Returns the row whose values apply to ``t``'s UTC date (a daily row, or
        the month's row in MONTHLY_PREDICTED), including centred 81-day means,
        the 3-hour ap interval containing ``t`` and the previous day's observed
        F10.7 (what NRLMSISE-00 takes as daily flux). ``section`` says whether
        the values are observed or predicted. Missing values are NaN.

        Raises:
            LookupError: if no row covers ``t``'s date.
        """
        t = _tt_seconds_to_utc_datetime64(t_tt_s)
        day = t.astype("datetime64[D]")
        i = self._row_for_date(day)
        if i is None:
            raise LookupError(f"no space-weather row covers {day}")
        k = int((t - day.astype("datetime64[us]")) // np.timedelta64(3, "h"))
        prev = self._row_for_date(day - np.timedelta64(1, "D"))
        s = self.scalars
        return SpaceWeatherObserved(
            t_utc=str(t),
            date=str(self.dates[i]),
            section=str(self.section[i]),
            f107_obs_sfu=float(s["f107_obs"][i]),
            f107_obs_prev_day_sfu=float(s["f107_obs"][prev]) if prev is not None else math.nan,
            f107_obs_ctr81_sfu=float(s["f107_obs_ctr81"][i]),
            f107_obs_lst81_sfu=float(s["f107_obs_lst81"][i]),
            f107_adj_sfu=float(s["f107_adj"][i]),
            f107_adj_ctr81_sfu=float(s["f107_adj_ctr81"][i]),
            ap_daily=float(s["ap_avg"][i]),
            ap_3h=tuple(float(x) for x in self.ap[i]),
            ap_3h_current=float(self.ap[i, min(k, 7)]),
            kp_3h=tuple(_div10(x) for x in self.kp_x10[i]),
        )


@dataclass
class SpaceWeatherAsOf:
    """Result of :meth:`SpaceWeather.as_of`. Any value may be NaN (missing)."""

    t0_utc: str
    last_complete_day: str | None = None
    f107_obs_sfu: float = math.nan
    f107_adj_sfu: float = math.nan
    f107_obs_trailing81_sfu: float = math.nan
    ap_daily: float = math.nan
    kp_sum: float = math.nan
    ap_3h_last: float = math.nan
    ap_3h_last_end_utc: str | None = None
    #: Latest row date the lookup was allowed to read (never after t0's date).
    rows_read_max_date: str | None = None


@dataclass(frozen=True)
class SpaceWeatherObserved:
    """Result of :meth:`SpaceWeather.observed`. Any value may be NaN (missing)."""

    t_utc: str
    date: str
    section: str
    f107_obs_sfu: float
    f107_obs_prev_day_sfu: float
    f107_obs_ctr81_sfu: float
    f107_obs_lst81_sfu: float
    f107_adj_sfu: float
    f107_adj_ctr81_sfu: float
    ap_daily: float
    ap_3h: tuple[float, ...]
    ap_3h_current: float
    kp_3h: tuple[float, ...]


def _div10(x: float) -> float:
    return float(x) / 10.0 if np.isfinite(x) else math.nan


def _trailing_mean(
    values: np.ndarray, idx: np.ndarray, dates: np.ndarray, end: int, n_days: int
) -> float:
    """Mean of ``values`` over the ``n_days`` consecutive dates ending at ``dates[end]``.

    NaN unless every one of the ``n_days`` dates has a row and a value.
    """
    start = end - n_days + 1
    if start < 0:
        return math.nan
    span = dates[start : end + 1]
    if (span[-1] - span[0]) != np.timedelta64(n_days - 1, "D"):
        return math.nan  # a date is missing inside the window
    vals = values[idx[start : end + 1]]
    if not np.all(np.isfinite(vals)):
        return math.nan
    return float(vals.mean())


def _tt_seconds_to_utc_datetime64(t_tt_s: float) -> np.datetime64:
    """TT seconds -> UTC ``datetime64[us]``. A leap second maps to 23:59:59.999999."""
    from orbit_correction.frames import tt_seconds_to_utc_isot

    iso = str(tt_seconds_to_utc_isot(float(t_tt_s)))
    if iso[17:19] == "60":
        iso = iso[:17] + "59.999999"
    return np.datetime64(iso, "us")


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def _parse_field(text: str, col: _Col, lineno: int) -> float:
    s = text[col.start : col.start + col.width]
    if not s.strip():
        return math.nan
    try:
        return float(s)
    except ValueError as exc:
        raise SpaceWeatherFormatError(
            f"line {lineno}: cannot parse {col.name} from {s!r}"
        ) from exc


def parse_spaceweather(path: str | Path) -> SpaceWeather:
    """Parse a CelesTrak/CSSI space-weather file as published.

    Tolerates blank fields anywhere (they become NaN), CRLF or LF line endings,
    lines with trailing blanks stripped, and a missing MONTHLY_FIT section.
    Rejects lines longer than the format, unparsable non-blank fields, rows
    outside a section, and a ``NUM_..._POINTS`` count that disagrees with the
    rows found.
    """
    text = Path(path).read_text(encoding="ascii")
    header: dict[str, str] = {}
    declared: dict[str, int] = {}
    seen_sections: list[str] = []
    dates: list[np.datetime64] = []
    sections: list[str] = []
    rows: list[list[float]] = []
    current: str | None = None
    counts: dict[str, int] = {}

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\r\n")
        stripped = line.strip()
        if current is None:
            if not stripped or stripped.startswith("#"):
                continue
            word = stripped.split()[0]
            if word in ("DATATYPE", "VERSION", "UPDATED"):
                header[word] = stripped[len(word) :].strip()
            elif word.startswith("NUM_") and word.endswith("_POINTS"):
                declared[word[4:-7]] = int(stripped.split()[1])
            elif word == "MONTHLY_FIT_TYPE":
                header[word] = stripped[len(word) :].strip()
            elif word == "BEGIN":
                current = stripped.split()[1]
                if current not in SECTIONS:
                    raise SpaceWeatherFormatError(f"line {lineno}: unknown section {current!r}")
                seen_sections.append(current)
                counts[current] = 0
            else:
                raise SpaceWeatherFormatError(f"line {lineno}: unexpected {stripped[:40]!r}")
            continue
        if stripped == f"END {current}":
            current = None
            continue
        if not stripped:
            continue
        if stripped[0].isalpha():
            raise SpaceWeatherFormatError(
                f"line {lineno}: expected END {current} before {stripped[:40]!r}"
            )
        if len(line) > LINE_WIDTH:
            raise SpaceWeatherFormatError(
                f"line {lineno}: {len(line)} columns, format has {LINE_WIDTH}"
            )
        line = line.ljust(LINE_WIDTH)
        vals = [_parse_field(line, c, lineno) for c in _COLUMNS]
        y, m, d = vals[0], vals[1], vals[2]
        if not all(np.isfinite([y, m, d])):
            raise SpaceWeatherFormatError(f"line {lineno}: blank date")
        dates.append(np.datetime64(f"{int(y):04d}-{int(m):02d}-{int(d):02d}", "D"))
        sections.append(current)
        rows.append(vals[3:])
        counts[current] += 1

    if current is not None:
        raise SpaceWeatherFormatError(f"section {current} has no END line")
    for sec, n in declared.items():
        if counts.get(sec, 0) != n:
            raise SpaceWeatherFormatError(
                f"NUM_{sec}_POINTS says {n}, found {counts.get(sec, 0)} rows"
            )

    names = [c.name for c in _COLUMNS[3:]]
    data = np.array(rows, dtype=np.float64).reshape(len(rows), len(names))
    col = {n: data[:, j] for j, n in enumerate(names)}
    kp = np.stack([col[f"kp_x10_{k}"] for k in range(8)], axis=1)
    ap = np.stack([col[f"ap_{k}"] for k in range(8)], axis=1)
    scalars = {n: np.ascontiguousarray(col[n]) for n in SCALAR_FIELDS}
    return SpaceWeather(
        dates=np.array(dates, dtype="datetime64[D]"),
        section=np.array(sections, dtype=object),
        kp_x10=kp,
        ap=ap,
        scalars=scalars,
        header=header,
        had_monthly_fit="MONTHLY_FIT" in seen_sections,
    )


# --------------------------------------------------------------------------------------
# Writing an Orekit-compatible copy
# --------------------------------------------------------------------------------------


@dataclass
class WriteReport:
    """What :func:`write_orekit_cssi` changed relative to the parsed values.

    Attributes:
        required_first, required_last: inclusive UTC date range that had to be
            free of placeholders.
        filled: (date, section, field) for every blank the writer filled with 0.
            All lie outside the required range.
        appended_empty_monthly_fit: whether an empty MONTHLY_FIT block was added.
    """

    required_first: str
    required_last: str
    filled: list[tuple[str, str, str]] = field(default_factory=list)
    appended_empty_monthly_fit: bool = False


def _format_row(sw: SpaceWeather, i: int, fill: bool) -> tuple[str, list[str]]:
    """Format row ``i``; with ``fill``, blank parse-required fields become 0."""
    date = sw.dates[i].astype(object)
    parts = [f"{date.year:4d}", f" {date.month:02d}", f" {date.day:02d}"]
    filled: list[str] = []
    for c in _COLUMNS[3:]:
        v = sw.value(c.name, i)
        if np.isfinite(v):
            parts.append(c.fmt(v))
        elif fill and c.name in _OREKIT_REQUIRED_TO_PARSE:
            parts.append(c.fmt(0.0))
            filled.append(c.name)
        else:
            parts.append(" " * c.width)
    line = "".join(parts)
    assert len(line) == LINE_WIDTH
    return line, filled


def write_orekit_cssi(
    sw: SpaceWeather,
    path: str | Path,
    propagate_first_utc: str,
    propagate_last_utc: str,
    margin_before_days: int = 3,
    margin_after_days: int = 1,
) -> WriteReport:
    """Write a copy of ``sw`` in the CSSI fixed-width format Orekit 13.x parses.

    Orekit's parser needs a number in every Kp/Ap/Cp/C9/Q field and a
    MONTHLY_FIT section. Blanks in those fields are filled with 0, but only on
    dates outside the required range; an empty MONTHLY_FIT block is appended if
    the source had none.

    The required range is every UTC date from ``propagate_first_utc`` minus
    ``margin_before_days`` to ``propagate_last_utc`` plus ``margin_after_days``.
    The margins cover what NRLMSISE-00 looks back at (previous-day F10.7, ap
    history up to 57 hours earlier). Every required date must have a row with
    every numeric field present; a blank there raises instead of being filled,
    as does a date with no row at all.

    Args:
        sw: parsed file.
        path: output path (overwritten).
        propagate_first_utc, propagate_last_utc: ISO dates or datetimes (UTC)
            bounding the propagation the copy is for.
        margin_before_days, margin_after_days: whole days added around them.

    Returns:
        A :class:`WriteReport` listing every fill.

    Raises:
        PlaceholderInRangeError: if a required date is missing or has a blank.
    """
    first = np.datetime64(propagate_first_utc, "D") - np.timedelta64(margin_before_days, "D")
    last = np.datetime64(propagate_last_utc, "D") + np.timedelta64(margin_after_days, "D")
    report = WriteReport(required_first=str(first), required_last=str(last))

    # 1. Check the required range before writing anything.
    numeric = [c.name for c in _COLUMNS[3:]]
    day = first
    while day <= last:
        i = sw._row_for_date(day)
        if i is None:
            raise PlaceholderInRangeError(f"no space-weather row for required date {day}")
        blanks = [n for n in numeric if not np.isfinite(sw.value(n, i))]
        if blanks:
            raise PlaceholderInRangeError(
                f"required date {day} ({sw.section[i]} row {sw.dates[i]}) has blank "
                f"{blanks}; refusing to write placeholders into the propagation range"
            )
        day += np.timedelta64(1, "D")

    # 2. Write.
    lines: list[str] = [
        "DATATYPE CssiSpaceWeather",
        f"VERSION {sw.header.get('VERSION', '1.2')}",
        f"UPDATED {sw.header.get('UPDATED', 'unknown')}",
        "# Rewritten by orbit_correction.spaceweather.write_orekit_cssi from the published file.",
        f"# Blank Kp/Ap/Cp/C9/Q fields outside {first}..{last} were filled with 0 so Orekit",
        "# can parse them; no value inside that range is a placeholder.",
        "#",
    ]
    fit_type = sw.header.get("MONTHLY_FIT_TYPE", "CSSI Vallado_2014_10")
    for sec in SECTIONS:
        idx = sw.rows_in(sec)
        if sec == "MONTHLY_FIT" and idx.size == 0:
            lines += ["NUM_MONTHLY_FIT_POINTS 0", f"MONTHLY_FIT_TYPE  {fit_type}",
                      "BEGIN MONTHLY_FIT", "END MONTHLY_FIT", ""]
            report.appended_empty_monthly_fit = True
            continue
        if idx.size == 0:
            continue
        lines.append(f"NUM_{sec}_POINTS {idx.size}")
        if sec == "MONTHLY_FIT":
            lines.append(f"MONTHLY_FIT_TYPE  {fit_type}")
        lines.append(f"BEGIN {sec}")
        for i in idx:
            line, filled = _format_row(sw, int(i), fill=True)
            for name in filled:
                report.filled.append((str(sw.dates[i]), sec, name))
            lines.append(line)
        lines.append(f"END {sec}")
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")
    return report


def format_row(sw: SpaceWeather, i: int) -> str:
    """Format one parsed row back to its 130-column line, blanks kept blank."""
    return _format_row(sw, i, fill=False)[0]
