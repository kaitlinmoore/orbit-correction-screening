"""Tests for ``orbit_correction.spaceweather``.

The fixture ``sw_quirks.txt`` is synthetic and reproduces the three quirks of
CelesTrak's published file: blank Q flags in DAILY_PREDICTED rows, blank
Kp/Ap/Cp/C9 (and Q) in MONTHLY_PREDICTED rows, and no MONTHLY_FIT section.
It has 120 OBSERVED days from 2024-01-01 to 2024-04-29, DAILY_PREDICTED
2024-04-30 to 2024-05-04, and MONTHLY_PREDICTED for June to August 2024.
"""

import copy
import math

import numpy as np
import pytest

from orbit_correction import spaceweather as S
from orbit_correction.frames import utc_to_tt_seconds

from .conftest import REPO


@pytest.fixture
def sw(fixtures):
    return S.parse_spaceweather(fixtures / "sw_quirks.txt")


def _data_lines(path):
    return [
        ln.rstrip("\r\n")
        for ln in open(path, encoding="ascii")
        if ln[:4].isdigit() and len(ln.rstrip("\r\n")) > 20
    ]


# ---------------------------------------------------------------- parsing and the quirks


def test_sections_and_counts(sw):
    assert len(sw) == 128
    assert sw.rows_in("OBSERVED").size == 120
    assert sw.rows_in("DAILY_PREDICTED").size == 5
    assert sw.rows_in("MONTHLY_PREDICTED").size == 3
    assert sw.header["VERSION"] == "1.2"
    assert sw.dates[0] == np.datetime64("2024-01-01")


def test_quirk_blank_q_in_predicted_rows_is_missing(sw):
    q = sw.scalars["q"]
    assert np.all(q[sw.rows_in("OBSERVED")] == 0)
    assert np.all(np.isnan(q[sw.rows_in("DAILY_PREDICTED")]))
    # the rest of a daily predicted row is present
    assert np.all(np.isfinite(sw.ap[sw.rows_in("DAILY_PREDICTED")]))


def test_quirk_blank_geomagnetic_fields_in_monthly_rows_are_missing(sw):
    m = sw.rows_in("MONTHLY_PREDICTED")
    assert np.all(np.isnan(sw.kp_x10[m])) and np.all(np.isnan(sw.ap[m]))
    for name in ("kp_sum_x10", "ap_avg", "cp", "c9", "q"):
        assert np.all(np.isnan(sw.scalars[name][m])), name
    # flux in monthly rows is present
    assert np.all(np.isfinite(sw.scalars["f107_obs"][m]))


def test_quirk_no_monthly_fit_section(sw):
    assert sw.had_monthly_fit is False


def test_no_placeholders_anywhere(sw):
    """Missing values are NaN, never 0 or another stand-in."""
    m = sw.rows_in("MONTHLY_PREDICTED")
    assert not np.any(sw.ap[m] == 0)
    assert S.SpaceWeatherAsOf("x").f107_obs_sfu != 0 and math.isnan(
        S.SpaceWeatherAsOf("x").f107_obs_sfu
    )


def test_crlf_and_stripped_trailing_blanks_parse_identically(sw, fixtures, tmp_path):
    raw = (fixtures / "sw_quirks.txt").read_text()
    crlf = tmp_path / "crlf.txt"
    crlf.write_bytes(raw.replace("\n", "\r\n").encode("ascii"))
    stripped = tmp_path / "stripped.txt"
    stripped.write_text("\n".join(ln.rstrip() for ln in raw.splitlines()) + "\n")
    for p in (crlf, stripped):
        other = S.parse_spaceweather(p)
        np.testing.assert_array_equal(other.dates, sw.dates)
        for k in S.SCALAR_FIELDS:
            np.testing.assert_array_equal(other.scalars[k], sw.scalars[k])
        np.testing.assert_array_equal(other.ap, sw.ap)


def test_format_row_reproduces_lines_byte_for_byte(sw, fixtures):
    """The fixture lines were written independently of the module's formatter."""
    lines = _data_lines(fixtures / "sw_quirks.txt")
    assert len(lines) == len(sw)
    for i, line in enumerate(lines):
        assert S.format_row(sw, i) == line


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda s: s.replace("NUM_OBSERVED_POINTS 120", "NUM_OBSERVED_POINTS 121"), "says 121"),
        (lambda s: s.replace("END OBSERVED", ""), "END|unexpected"),
        (lambda s: s.replace("2024 01 05", "2024 01 05" + "x" * 200, 1), "columns"),
        (lambda s: s.replace(" 150.0", " 15O.0", 1), "cannot parse"),
        (lambda s: s.replace("BEGIN MONTHLY_PREDICTED", "BEGIN SOMETHING_ELSE"), "unknown section"),
    ],
)
def test_parser_rejects_malformed(fixtures, tmp_path, edit, match):
    p = tmp_path / "bad.txt"
    p.write_text(edit((fixtures / "sw_quirks.txt").read_text()))
    with pytest.raises(S.SpaceWeatherFormatError, match=match):
        S.parse_spaceweather(p)


# ---------------------------------------------------------------- as-of lookup (model inputs)


def _poison_after(sw: S.SpaceWeather, day: str) -> S.SpaceWeather:
    """Copy of ``sw`` with every value in rows dated after ``day`` replaced by 9999."""
    out = copy.deepcopy(sw)
    future = out.dates > np.datetime64(day)
    out.kp_x10[future] = 9999.0
    out.ap[future] = 9999.0
    for v in out.scalars.values():
        v[future] = 9999.0
    return out


@pytest.mark.parametrize(
    "t0",
    [
        "2024-03-15T00:00:00",
        "2024-03-15T00:00:00.000001",
        "2024-03-15T07:30:00",
        "2024-03-15T23:59:59.999",
        "2024-04-29T12:00:00",
        "2024-05-02T12:00:00",  # inside the predicted period
        "2024-07-15T00:00:00",  # inside the monthly period
    ],
)
def test_as_of_never_reads_rows_dated_after_t0(sw, t0):
    t = utc_to_tt_seconds(t0)
    a = sw.as_of(t)
    b = _poison_after(sw, t0[:10]).as_of(t)
    assert a == b  # values from rows after t0's date cannot influence the result
    assert a.rows_read_max_date <= t0[:10]
    for v in (a.f107_obs_sfu, a.ap_daily, a.ap_3h_last, a.f107_obs_trailing81_sfu):
        assert v != 9999.0


def test_as_of_uses_last_complete_day(sw):
    obs = sw.rows_in("OBSERVED")
    f = sw.scalars["f107_obs"]
    # any time on 2024-03-15 sees 2024-03-14 as the last complete day
    for t0 in ("2024-03-15T00:00:00", "2024-03-15T23:59:59"):
        a = sw.as_of(utc_to_tt_seconds(t0))
        assert a.last_complete_day == "2024-03-14"
        i = int(obs[sw.dates[obs] == np.datetime64("2024-03-14")][0])
        assert a.f107_obs_sfu == f[i]
        assert a.ap_daily == sw.scalars["ap_avg"][i]
        assert a.kp_sum == sw.scalars["kp_sum_x10"][i] / 10


def test_as_of_three_hour_ap(sw):
    a = sw.as_of(utc_to_tt_seconds("2024-03-15T07:30:00"))
    i = int(np.flatnonzero(sw.dates == np.datetime64("2024-03-15"))[0])
    assert a.ap_3h_last == sw.ap[i, 1]  # 03-06 UT ended; 06-09 UT has not
    assert a.ap_3h_last_end_utc.startswith("2024-03-15T06:00:00")
    a = sw.as_of(utc_to_tt_seconds("2024-03-15T00:00:00"))
    j = int(np.flatnonzero(sw.dates == np.datetime64("2024-03-14"))[0])
    assert a.ap_3h_last == sw.ap[j, 7]  # 21-24 UT of the previous day


def test_as_of_trailing_81_day_mean(sw):
    f = sw.scalars["f107_obs"]
    a = sw.as_of(utc_to_tt_seconds("2024-04-15T12:00:00"))  # last complete: 04-14 (index 104)
    assert a.f107_obs_trailing81_sfu == pytest.approx(f[104 - 80 : 105].mean())
    # fewer than 81 known days: missing, not a shorter average
    early = sw.as_of(utc_to_tt_seconds("2024-03-01T12:00:00"))
    assert math.isnan(early.f107_obs_trailing81_sfu)
    assert np.isfinite(early.f107_obs_sfu)


def test_as_of_ignores_predicted_rows(sw):
    a = sw.as_of(utc_to_tt_seconds("2024-05-03T12:00:00"))
    assert a.last_complete_day == "2024-04-29"  # last OBSERVED day, not a forecast
    assert a.rows_read_max_date == "2024-04-29"


def test_as_of_extra_lag(sw):
    a = sw.as_of(utc_to_tt_seconds("2024-03-15T07:30:00"), extra_lag_days=2)
    assert a.last_complete_day == "2024-03-12"


def test_as_of_before_data_is_all_missing(sw):
    a = sw.as_of(utc_to_tt_seconds("2023-06-01T00:00:00"))
    assert a.last_complete_day is None
    assert math.isnan(a.f107_obs_sfu) and math.isnan(a.ap_3h_last)


# ---------------------------------------------------------------- observed lookup (truth)


def test_observed_includes_centred_average_and_previous_day(sw):
    o = sw.observed(utc_to_tt_seconds("2024-03-15T07:30:00"))
    i = int(np.flatnonzero(sw.dates == np.datetime64("2024-03-15"))[0])
    assert o.section == "OBSERVED"
    assert o.f107_obs_ctr81_sfu == sw.scalars["f107_obs_ctr81"][i]
    assert o.f107_obs_prev_day_sfu == sw.scalars["f107_obs"][i - 1]
    assert o.ap_3h_current == sw.ap[i, 2]
    assert o.kp_3h[0] == sw.kp_x10[i, 0] / 10


def test_observed_monthly_values_stay_missing(sw):
    o = sw.observed(utc_to_tt_seconds("2024-07-15T12:00:00"))
    assert o.section == "MONTHLY_PREDICTED" and o.date == "2024-07-01"
    assert math.isnan(o.ap_daily) and all(math.isnan(x) for x in o.ap_3h)
    assert np.isfinite(o.f107_obs_sfu)


def test_observed_raises_without_a_row(sw):
    with pytest.raises(LookupError):
        sw.observed(utc_to_tt_seconds("2024-05-20T00:00:00"))  # gap before June


# ---------------------------------------------------------------- Orekit-compatible writer


def test_writer_round_trip(sw, tmp_path):
    out = tmp_path / "orekit.txt"
    rep = S.write_orekit_cssi(sw, out, "2024-03-01", "2024-03-08")
    assert (rep.required_first, rep.required_last) == ("2024-02-27", "2024-03-09")
    assert rep.appended_empty_monthly_fit
    back = S.parse_spaceweather(out)
    assert back.had_monthly_fit
    np.testing.assert_array_equal(back.dates, sw.dates)
    np.testing.assert_array_equal(back.section, sw.section)
    filled = {(d, f) for d, _, f in rep.filled}
    for i in range(len(sw)):
        d = str(sw.dates[i])
        for name in [f"kp_x10_{k}" for k in range(8)] + [f"ap_{k}" for k in range(8)] + list(
            S.SCALAR_FIELDS
        ):
            a, b = sw.value(name, i), back.value(name, i)
            if (d, name) in filled:
                assert math.isnan(a) and b == 0.0
                assert not (rep.required_first <= d <= rep.required_last)
            elif math.isnan(a):
                assert math.isnan(b)
            else:
                assert a == b, (d, name)
    # every blank Orekit cannot parse was filled; nothing else was
    assert {f for _, _, f in rep.filled} <= set(S._OREKIT_REQUIRED_TO_PARSE)
    assert sum(1 for _, s, _ in rep.filled if s == "DAILY_PREDICTED") == 5  # Q only


def test_writer_unfilled_rows_are_byte_identical(sw, fixtures, tmp_path):
    out = tmp_path / "orekit.txt"
    S.write_orekit_cssi(sw, out, "2024-03-01", "2024-03-08")
    src = _data_lines(fixtures / "sw_quirks.txt")
    dst = _data_lines(out)
    assert src[:120] == dst[:120]  # OBSERVED rows had nothing to fill


@pytest.mark.parametrize(
    "first, last, match",
    [
        ("2024-04-27", "2024-05-02", "blank \\['q'\\]"),  # reaches DAILY_PREDICTED
        ("2024-06-10", "2024-06-17", "blank"),  # MONTHLY_PREDICTED
        ("2024-05-10", "2024-05-12", "no space-weather row"),  # gap
        ("2023-12-01", "2023-12-05", "no space-weather row"),  # before the file
        ("2024-04-29", "2024-04-29", "blank"),  # margin_after reaches 04-30
    ],
)
def test_writer_refuses_placeholders_in_range(sw, tmp_path, first, last, match):
    out = tmp_path / "orekit.txt"
    with pytest.raises(S.PlaceholderInRangeError, match=match):
        S.write_orekit_cssi(sw, out, first, last)
    assert not out.exists()  # refused before writing anything


# ---------------------------------------------------------------- the published file (optional)

REAL = REPO / "data" / "raw" / "celestrak" / "SW-All.txt"


@pytest.mark.skipif(not REAL.exists(), reason="CelesTrak SW-All.txt not downloaded")
def test_real_file_parses_formats_and_matches_lst81():
    sw = S.parse_spaceweather(REAL)
    lines = _data_lines(REAL)
    assert all(S.format_row(sw, i) == ln for i, ln in enumerate(lines))
    # Triangulation: the trailing 81-day mean this module computes equals
    # CelesTrak's own Lst81 column (rounded to 0.1 sfu) over the last 3000 days.
    obs = sw.rows_in("OBSERVED")
    f = sw.scalars["f107_obs"][obs]
    lst = sw.scalars["f107_obs_lst81"][obs]
    ours = np.convolve(f, np.ones(81) / 81, "valid")  # ours[j] ends at day j + 80
    assert np.nanmax(np.abs(ours - lst[80:])[-3000:]) <= 0.05 + 1e-9
