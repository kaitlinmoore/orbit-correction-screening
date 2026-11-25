"""Tests for ``orbit_correction.frames``: TEME -> GCRF and time scales.

Reference: Vallado, Crawford, Hujsak and Kelso, "Revisiting Spacetrack Report
#3", AIAA 2006-6753 (Rev 3), Appendix C. Two worked examples are used:

1. TLE 00005 propagated 3 days with SGP4, then TEME -> J2000 (IAU-76/FK5,
   "of date"). Tolerance 0.5 m and 0.5 mm/s. Astropy uses IAU 2006/2000A with
   IERS EOP rather than IAU-76/FK5, and GCRF differs from FK5 J2000 by the frame
   bias; together these are expected at the decimetre level. Observed: 0.12 m,
   0.11 mm/s.
2. An ITRF state on 2004-04-06 07:51:28.386009 UTC with its TEME equivalent,
   computed with the paper's EOP (dUT1 = -0.439961 s, xp = -0.140682",
   yp = 0.333309"). Both vectors must map to the same GCRF state through this
   module (TEME path) and astropy (ITRS path). Tolerance 0.5 m and 1 mm/s:
   astropy's IERS dUT1 for that date differs from the paper's by 0.46 ms, about
   0.2 m of Earth rotation at this radius. Observed: 0.36 m, 0.27 mm/s.
"""

import datetime as dt

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianDifferential, CartesianRepresentation
from astropy.time import Time
from astropy.utils import iers
from sgp4.api import WGS72, Satrec, jday

from orbit_correction import frames as F

# ---------------------------------------------------------------- Vallado case 1

L1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
L2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"
R_TEME_3D = np.array([-9060.47373569, 4658.70952502, 813.68673153])
V_TEME_3D = np.array([-2.232832783, -4.110453490, -3.157345433])
R_J2000_3D = np.array([-9059.9415541, 4659.6971990, 813.9569402])
V_J2000_3D = np.array([-2.233347413, -4.110136158, -3.157394560])

# ---------------------------------------------------------------- Vallado case 2

EPOCH_2004 = "2004-04-06T07:51:28.386009"
R_ITRF_2004 = np.array([-1033.47938300, 7901.29527540, 6380.35659580])
V_ITRF_2004 = np.array([-3.225636520, -2.872451450, 5.531924446])
R_TEME_2004 = np.array([5094.18010720, 6127.64470520, 6380.34453270])
V_TEME_2004 = np.array([-4.746131494, 0.785817998, 5.531931288])


def test_vallado_sgp4_teme_reproduced():
    """The sgp4 library reproduces the paper's TEME state to well under a metre."""
    sat = Satrec.twoline2rv(L1, L2, WGS72)
    err, r, v = sat.sgp4_tsince(3 * 1440.0)
    assert err == 0
    np.testing.assert_allclose(r, R_TEME_3D, atol=1e-5)  # 1 cm
    np.testing.assert_allclose(v, V_TEME_3D, atol=1e-8)  # 0.01 mm/s


def test_vallado_teme_to_gcrf_matches_j2000():
    sat = Satrec.twoline2rv(L1, L2, WGS72)
    t = F.satrec_epoch_to_tt_seconds(sat.jdsatepoch, sat.jdsatepochF) + 3 * 86400.0
    r, v = F.teme_to_gcrf(R_TEME_3D, V_TEME_3D, t)
    assert np.linalg.norm(r - R_J2000_3D) < 0.5e-3  # km
    assert np.linalg.norm(v - V_J2000_3D) < 0.5e-6  # km/s


def test_vallado_2004_teme_and_itrf_agree_in_gcrf():
    t = F.utc_to_tt_seconds(EPOCH_2004)
    r_ours, v_ours = F.teme_to_gcrf(R_TEME_2004, V_TEME_2004, t)
    obstime = F.tt_seconds_to_time(t)
    rep = CartesianRepresentation(
        R_ITRF_2004 * u.km, differentials=CartesianDifferential(V_ITRF_2004 * u.km / u.s)
    )
    g = ITRS(rep, obstime=obstime).transform_to(GCRS(obstime=obstime))
    assert np.linalg.norm(r_ours - g.cartesian.xyz.to_value(u.km)) < 0.5e-3
    assert np.linalg.norm(v_ours - g.velocity.d_xyz.to_value(u.km / u.s)) < 1e-6


# ---------------------------------------------------------------- round trips and shapes


def test_round_trip_vectorised():
    rng = np.random.default_rng(0)
    n = 50
    r = rng.normal(size=(n, 3))
    r = 7000.0 * r / np.linalg.norm(r, axis=1, keepdims=True)
    v = rng.normal(size=(n, 3))
    v = 7.5 * v / np.linalg.norm(v, axis=1, keepdims=True)
    t = F.utc_to_tt_seconds("2020-01-29T00:00:00") + 600.0 * np.arange(n)
    rg, vg = F.teme_to_gcrf(r, v, t)
    assert rg.shape == (n, 3) and rg.dtype == np.float64
    r2, v2 = F.gcrf_to_teme(rg, vg, t)
    np.testing.assert_allclose(r2, r, atol=1e-9)  # 1 micron
    np.testing.assert_allclose(v2, v, atol=1e-11)
    # norms are preserved: the transform is a rotation (plus frame rotation rate in v)
    np.testing.assert_allclose(np.linalg.norm(rg, axis=1), 7000.0, rtol=1e-12)


def test_scalar_time_broadcasts_and_single_state_shape():
    t = F.utc_to_tt_seconds("2024-02-28T06:34:00")
    r1, v1 = F.teme_to_gcrf(R_TEME_2004, V_TEME_2004, t)
    assert r1.shape == (3,)
    rn, vn = F.teme_to_gcrf(np.stack([R_TEME_2004] * 2), np.stack([V_TEME_2004] * 2), t)
    np.testing.assert_allclose(rn[1], r1, atol=1e-12)
    with pytest.raises(ValueError):
        F.teme_to_gcrf(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(3))


# ---------------------------------------------------------------- time scales


def test_j2000_epoch_is_zero():
    # J2000.0 = 2000-01-01T12:00:00 TT = 11:58:55.816 UTC (TT - UTC = 64.184 s then)
    assert F.utc_to_tt_seconds("2000-01-01T11:58:55.816") == pytest.approx(0.0, abs=1e-6)
    assert F.time_to_tt_seconds(Time("2000-01-01T12:00:00", scale="tt")) == 0.0


@pytest.mark.parametrize(
    "utc, tt_minus_utc",
    [
        ("1990-01-01T00:00:00", 32.184 + 25),
        ("2004-04-06T07:51:28", 32.184 + 32),
        ("2009-02-10T16:56:00", 32.184 + 34),
        ("2016-12-31T12:00:00", 32.184 + 36),
        ("2017-01-01T12:00:00", 32.184 + 37),
        ("2026-11-23T00:00:00", 32.184 + 37),
    ],
)
def test_tt_minus_utc(utc, tt_minus_utc):
    assert F.tt_minus_utc_s(utc) == pytest.approx(tt_minus_utc, abs=1e-9)
    # and the TT seconds agree with it: TT seconds of the UTC instant equal the
    # same calendar reading taken as TT, plus TT - UTC
    as_tt = F.time_to_tt_seconds(Time(utc, scale="tt"))
    assert F.utc_to_tt_seconds(utc) - as_tt == pytest.approx(tt_minus_utc, abs=1e-6)


def test_leap_second_2016():
    s = F.utc_to_tt_seconds(
        ["2016-12-31T23:59:59", "2016-12-31T23:59:60", "2017-01-01T00:00:00"]
    )
    np.testing.assert_allclose(np.diff(s), [1.0, 1.0], atol=1e-6)
    back = F.tt_seconds_to_utc_isot(s)
    assert list(back) == [
        "2016-12-31T23:59:59.000000",
        "2016-12-31T23:59:60.000000",
        "2017-01-01T00:00:00.000000",
    ]
    assert F.tai_minus_utc_s("2016-12-31T23:59:60") == 36.0
    assert F.tai_minus_utc_s("2017-01-01T00:00:00") == 37.0
    # a 7-day UTC window containing the leap second lasts 7 x 86400 + 1 SI seconds
    span = F.utc_to_tt_seconds("2017-01-04T00:00:00") - F.utc_to_tt_seconds("2016-12-28T00:00:00")
    assert span == pytest.approx(7 * 86400 + 1, abs=1e-6)


def test_tt_seconds_round_trip_precision():
    s = np.array([-1.0e8, 0.0, 1.23456789e8, 8.48664069184e8])
    back = F.time_to_tt_seconds(F.tt_seconds_to_time(s))
    np.testing.assert_allclose(back, s, atol=1e-6)


def test_elset_epoch_matches_satrec_epoch():
    sat = Satrec.twoline2rv(L1, L2, WGS72)
    from_satrec = F.satrec_epoch_to_tt_seconds(sat.jdsatepoch, sat.jdsatepochF)
    # day 179.78495062 of 2000 = 2000-06-27T18:50:19.733568 UTC
    from_iso = F.elset_epoch_to_tt_seconds("2000-06-27T18:50:19.733568")
    assert from_satrec == pytest.approx(from_iso, abs=1e-4)


# ---------------------------------------------------------------- SGP4 epoch on leap-second days

#: TAI - UTC (Delta AT) on the dates used below, typed in from the IERS leap-second list
#: (Bulletin C) so the reference doesn't go through astropy or ERFA.
_DELTA_AT = {"2015-06-30": 35.0, "2016-12-30": 36.0, "2016-12-31": 36.0, "2017-01-01": 37.0}


def _tt_seconds_by_hand(date: str, h: int, m: int, s: float) -> float:
    """TT seconds since J2000.0 from calendar arithmetic and a typed-in Delta AT.

    TT = UTC + Delta AT + 32.184 s. Counting naive (86,400-s) days from
    2000-01-01T12:00:00 and adding the leap seconds inserted since then
    (Delta AT - 32, the value in 2000) gives SI seconds, because the instant is not
    inside a leap second.
    """
    naive = (
        dt.datetime.fromisoformat(date) + dt.timedelta(hours=h, minutes=m, seconds=s)
        - dt.datetime(2000, 1, 1, 12)
    ).total_seconds()
    return naive + (_DELTA_AT[date] - 32.0) + 64.184


@pytest.mark.parametrize("date", ["2016-12-31", "2015-06-30"])
@pytest.mark.parametrize(
    "hms", [(0, 0, 0.0), (0, 0, 0.25), (6, 0, 0.0), (12, 0, 0.0), (23, 0, 0.0), (23, 59, 59.5)]
)
def test_satrec_epoch_on_a_leap_second_day(date, hms):
    """SGP4's 86,400-s Julian date on a leap-second day maps to the right instant.

    Two independent references: astropy from the ISO string (calendar fields, not a
    Julian date) and calendar arithmetic with a typed-in Delta AT. Before the fix this
    function was 0.5 s late at 12:00 and 0.96 s late at 23:00 on 2016-12-31.
    """
    y, mo, d = (int(x) for x in date.split("-"))
    jd, fr = jday(y, mo, d, *hms)
    got = F.satrec_epoch_to_tt_seconds(jd, fr)
    iso = f"{date}T{hms[0]:02d}:{hms[1]:02d}:{hms[2]:09.6f}"
    tol = 1e-6  # s; TT seconds near 2016 resolve ~6e-8 s, the pair's fraction ~1e-11 d
    assert got == pytest.approx(F.utc_to_tt_seconds(iso), abs=tol)
    assert got == pytest.approx(_tt_seconds_by_hand(date, *hms), abs=tol)


def test_satrec_epoch_from_satrec_and_across_midnight():
    """The pair as a TLE reader stores it, and both sides of a leap-second midnight."""
    sat = Satrec.twoline2rv(
        "1 00005U 58002B   16366.50000000  .00000023  00000-0  28098-4 0  4753",
        "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
        WGS72,
    )  # day 366 of 2016 = 2016-12-31, 12:00 UTC
    got = F.satrec_epoch_to_tt_seconds(sat.jdsatepoch, sat.jdsatepochF)
    assert got == pytest.approx(_tt_seconds_by_hand("2016-12-31", 12, 0, 0.0), abs=1e-6)
    # the last second before the leap second and the first after it are 2 SI s apart
    before = F.satrec_epoch_to_tt_seconds(*jday(2016, 12, 31, 23, 59, 59.0))
    after = F.satrec_epoch_to_tt_seconds(*jday(2017, 1, 1, 0, 0, 0.0))
    assert after - before == pytest.approx(2.0, abs=1e-6)
    assert after == pytest.approx(_tt_seconds_by_hand("2017-01-01", 0, 0, 0.0), abs=1e-6)
    # a pair split differently (all in the fraction) gives the same instant
    jd, fr = jday(2016, 12, 31, 18, 0, 0.0)
    assert F.satrec_epoch_to_tt_seconds(jd + fr - 0.25, 0.25) == pytest.approx(
        F.satrec_epoch_to_tt_seconds(jd, fr), abs=1e-6
    )


@pytest.mark.parametrize("date", ["2016-12-31", "2015-06-30", "2016-12-30"])
@pytest.mark.parametrize("before_s", [1e-3, 3e-5, 1e-5, 1e-6])
def test_satrec_epoch_just_before_the_end_of_a_day(date, before_s):
    """Microseconds before midnight stay on their own day, leap second or not.

    The sum ``jd + fraction`` of such a pair rounds to the next midnight (a Julian date
    resolves about 40 microseconds in one float64). Picking the day from that sum put an
    epoch 10 or 1 microseconds before the 2016 leap second a full second late; the day
    must come from the two parts separately. 2016-12-30 is the same check on a day
    without a leap second, where the same slip would cost nothing and so must not
    change the answer either.
    """
    y, mo, d = (int(x) for x in date.split("-"))
    sec = 60.0 - before_s
    jd, fr = jday(y, mo, d, 23, 59, sec)
    if before_s <= 1e-5:
        assert jd + fr == jd + 1.0  # the trap: the float sum is already the next midnight
    got = F.satrec_epoch_to_tt_seconds(jd, fr)
    # tolerance: TT seconds near 2016 resolve 6e-8 s; a one-second slip is the target
    assert got == pytest.approx(_tt_seconds_by_hand(date, 23, 59, sec), abs=2e-7)
    iso = f"{date}T23:59:{sec:09.6f}"
    assert got == pytest.approx(F.utc_to_tt_seconds(iso), abs=2e-7)
    # the same instant with a whole day moved from one part to the other
    assert F.satrec_epoch_to_tt_seconds(jd - 1.0, fr + 1.0) == pytest.approx(got, abs=2e-7)
    assert F.satrec_epoch_to_tt_seconds(jd + 1.0, fr - 1.0) == pytest.approx(got, abs=2e-7)


# ---------------------------------------------------------------- EOP pinning


def test_eop_is_pinned_and_offline():
    prov = F.eop_provenance()
    assert prov["kind"].startswith("astropy-iers-data")
    assert prov["astropy_iers_data"]
    assert len(prov["iers_a_sha256"]) == 64
    assert float(prov["last_measured_mjd"]) <= float(prov["last_mjd"])
    assert iers.conf.auto_download is False


@pytest.mark.filterwarnings("ignore::erfa.ErfaWarning")
def test_eop_coverage_enforced():
    r, v = np.array([7000.0, 0, 0]), np.array([0, 7.5, 0])
    with pytest.raises(F.EOPCoverageError):
        F.teme_to_gcrf(r, v, F.utc_to_tt_seconds("2045-01-01T00:00:00"))
    with pytest.raises(F.EOPCoverageError):
        F.teme_to_gcrf(r, v, F.utc_to_tt_seconds("1965-01-01T00:00:00"))


def test_teme_to_gcrf_insensitive_to_ut1_but_itrs_is_not():
    """UT1 errors cancel in TEME -> ITRS -> GCRS; they do not cancel in Earth-fixed ITRS."""
    from astropy.coordinates import TEME

    t = F.utc_to_tt_seconds("2024-02-28T06:34:00")

    def run():
        # A fresh Time each call: astropy caches UT1-UTC on a Time object once used.
        obstime = F.tt_seconds_to_time(t)
        g = F.teme_to_gcrf(R_TEME_2004, V_TEME_2004, t)[0]
        rep = CartesianRepresentation(R_TEME_2004 * u.km)
        i = TEME(rep, obstime=obstime).transform_to(ITRS(obstime=obstime))
        return g, i.cartesian.xyz.to_value(u.km)

    g0, i0 = run()
    try:
        table = iers.IERS_A.open(iers.IERS_A_FILE).copy()  # IERS_Auto would re-read its file
        table["UT1_UTC"] = table["UT1_UTC"] + 0.1 * u.s
        iers.earth_orientation_table.set(table)
        g1, i1 = run()
    finally:
        F.configure_offline_eop()
    assert np.linalg.norm(g1 - g0) < 1e-6  # < 1 mm in GCRF
    assert np.linalg.norm(i1 - i0) > 0.03  # > 30 m Earth-fixed


def test_use_eop_file_switches_and_restores():
    t = F.utc_to_tt_seconds(EPOCH_2004)
    before = F.teme_to_gcrf(R_TEME_2004, V_TEME_2004, t)[0]
    try:
        F.use_eop_file(iers.IERS_A_FILE)
        prov = F.eop_provenance()
        assert prov["kind"] == "finals2000A file"
        after = F.teme_to_gcrf(R_TEME_2004, V_TEME_2004, t)[0]
        # same IERS data for 2004, so the same answer to well under a millimetre
        np.testing.assert_allclose(after, before, atol=1e-6)
    finally:
        F.configure_offline_eop()
    assert F.eop_provenance()["kind"].startswith("astropy-iers-data")


# ---------------------------------------------------------------- fast TEME -> GCRF
#
# teme_to_gcrf_fast interpolates the exact rotation between nodes 6 h apart on a fixed
# TT lattice. Tolerance: 10 micrometres and 10 micrometres/s against the exact path,
# 100 times under the 1 mm target (observed: about 0.7 micrometres and 0.7
# micrometres/s; the velocity floor is consistent with astropy's own finite-difference
# velocity in the exact path).

FAST_TOL_KM = 1e-8
FAST_TOL_KM_S = 1e-8


def _random_states(n, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(size=(n, 3))
    r = 7000.0 * r / np.linalg.norm(r, axis=1, keepdims=True)
    v = rng.normal(size=(n, 3))
    v = 7.5 * v / np.linalg.norm(v, axis=1, keepdims=True)
    return r, v


@pytest.mark.parametrize(
    "start_utc",
    [
        "2005-06-10T00:00:00",
        "2016-12-28T00:00:00",  # the window contains the 2016-12-31T23:59:60 leap second
        "2024-03-01T00:00:00",
    ],
)
def test_fast_matches_exact_over_7_days_at_60s(start_utc):
    t = F.utc_to_tt_seconds(start_utc) + 60.0 * np.arange(7 * 1440 + 1)
    r, v = _random_states(t.size)
    re, ve = F.teme_to_gcrf(r, v, t)
    rf, vf = F.teme_to_gcrf_fast(r, v, t)
    assert np.linalg.norm(rf - re, axis=1).max() < FAST_TOL_KM
    assert np.linalg.norm(vf - ve, axis=1).max() < FAST_TOL_KM_S


def test_fast_across_the_leap_second_instants():
    labels = [
        "2016-12-31T23:59:58.5", "2016-12-31T23:59:59", "2016-12-31T23:59:59.5",
        "2016-12-31T23:59:60", "2016-12-31T23:59:60.5", "2017-01-01T00:00:00",
        "2017-01-01T00:00:00.5",
    ]
    t = F.utc_to_tt_seconds(labels)
    np.testing.assert_allclose(np.diff(t), 0.5, atol=1e-6)
    r, v = _random_states(t.size, seed=3)
    re, ve = F.teme_to_gcrf(r, v, t)
    rf, vf = F.teme_to_gcrf_fast(r, v, t)
    step = 5  # 2017-01-01T00:00:00, the instant TAI - UTC steps from 36 s to 37 s
    others = np.arange(t.size) != step
    np.testing.assert_allclose(rf[others], re[others], rtol=0, atol=FAST_TOL_KM)
    np.testing.assert_allclose(vf[others], ve[others], rtol=0, atol=FAST_TOL_KM_S)
    # At exactly that instant astropy (8.0.3) takes UT1 one second late
    # (00:00:01.591 instead of 00:00:00.591), so the exact path is off there by tens of
    # micrometres in GCRF (UT1 mostly cancels in this chain). One microsecond either
    # side it is fine. The fast path's nodes never sit on a leap second (they are at
    # whole multiples of 6 h in TT, while UTC midnight is TT 00:01:09.184), so it is
    # smooth through it; check it against the exact path a microsecond either side.
    rm, _ = F.teme_to_gcrf(r[step], v[step], t[step] - 1e-6)
    rp, _ = F.teme_to_gcrf(r[step], v[step], t[step] + 1e-6)
    np.testing.assert_allclose(rf[step], 0.5 * (rm + rp), rtol=0, atol=FAST_TOL_KM)
    assert np.linalg.norm(rf[step] - re[step]) < 1e-7  # 100 micrometres, whichever is right
    # the interpolated rotation is smooth through the leap second: equal TT steps give
    # equal changes of M (second differences at the level of rounding)
    m, _ = F.teme_to_gcrf_rotation_fast(t)
    assert np.abs(np.diff(m, n=2, axis=0)).max() < 1e-14


def test_fast_shapes_and_blocks():
    t0 = F.utc_to_tt_seconds("2024-02-28T06:34:00")
    t = t0 + 60.0 * np.arange(5)
    r, v = _random_states(5, seed=1)
    r1, v1 = F.teme_to_gcrf_fast(R_TEME_2004, V_TEME_2004, t0)
    assert r1.shape == v1.shape == (3,)
    rn, vn = F.teme_to_gcrf_fast(r, v, t)
    assert rn.shape == (5, 3)
    # (K, N, 3): K objects on one grid give the same rows as one object at a time
    block_r = np.stack([r, 2.0 * r])
    block_v = np.stack([v, -v])
    rk, vk = F.teme_to_gcrf_fast(block_r, block_v, t)
    assert rk.shape == (2, 5, 3)
    np.testing.assert_array_equal(rk[0], rn)
    np.testing.assert_array_equal(vk[0], vn)
    np.testing.assert_array_equal(rk[1], F.teme_to_gcrf_fast(2.0 * r, -v, t)[0])
    # a scalar time broadcasts
    rs, _ = F.teme_to_gcrf_fast(r, v, t0)
    np.testing.assert_array_equal(rs[0], F.teme_to_gcrf_fast(r[0], v[0], t0)[0])
    with pytest.raises(ValueError):
        F.teme_to_gcrf_fast(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(3))
    with pytest.raises(ValueError):
        F.teme_to_gcrf_fast(np.zeros((2, 3)), np.zeros((3, 3)), np.zeros(2))


def test_fast_result_depends_only_on_each_states_time():
    """Fixed lattice, local stencil: a state's result is the same in any batch."""
    t = F.utc_to_tt_seconds("2010-05-05T00:00:00") + 977.0 * np.arange(400)
    r, v = _random_states(t.size, seed=2)
    rf, vf = F.teme_to_gcrf_fast(r, v, t)
    F.clear_fast_rotation_cache()
    for idx in (slice(0, 1), slice(199, 201), slice(399, 400), np.array([5, 300, 17])):
        ri, vi = F.teme_to_gcrf_fast(r[idx], v[idx], t[idx])
        np.testing.assert_array_equal(ri, rf[idx])
        np.testing.assert_array_equal(vi, vf[idx])


def test_fast_rotation_is_orthonormal_and_derivative_consistent():
    t = F.utc_to_tt_seconds("2020-07-01T00:00:00") + 3600.0 * np.arange(48)
    m, m_dot = F.teme_to_gcrf_rotation_fast(t)
    eye = np.einsum("nij,nkj->nik", m, m)
    np.testing.assert_allclose(eye, np.broadcast_to(np.eye(3), eye.shape), atol=1e-13)
    # M_dot against a central difference of M
    h = 10.0
    mp, _ = F.teme_to_gcrf_rotation_fast(t + h)
    mm, _ = F.teme_to_gcrf_rotation_fast(t - h)
    np.testing.assert_allclose(m_dot, (mp - mm) / (2 * h), rtol=0, atol=1e-16)


@pytest.mark.filterwarnings("ignore::erfa.ErfaWarning")
def test_fast_eop_coverage_same_as_exact():
    r, v = np.array([7000.0, 0, 0]), np.array([0, 7.5, 0])
    for when in ("2045-01-01T00:00:00", "1965-01-01T00:00:00"):
        t = F.utc_to_tt_seconds(when)
        with pytest.raises(F.EOPCoverageError):
            F.teme_to_gcrf(r, v, t)
        with pytest.raises(F.EOPCoverageError):
            F.teme_to_gcrf_fast(r, v, t)
    # Inside the table but within a stencil of its last day: the exact path answers
    # these, so the fast path must too (it falls back to the exact path there).
    last_mjd = float(F.eop_provenance()["last_mjd"])
    t_end = Time(last_mjd - 0.2, format="mjd", scale="utc")
    t = F.time_to_tt_seconds(t_end) - 600.0 * np.arange(3)[::-1]
    rr, vv = np.tile(r, (3, 1)), np.tile(v, (3, 1))
    re, ve = F.teme_to_gcrf(rr, vv, t)
    rf, vf = F.teme_to_gcrf_fast(rr, vv, t)
    np.testing.assert_array_equal(rf, re)
    np.testing.assert_array_equal(vf, ve)


def test_fast_cache_follows_the_eop_table():
    t = F.utc_to_tt_seconds(EPOCH_2004) + 60.0 * np.arange(10)
    r, v = _random_states(10, seed=4)
    before = F.teme_to_gcrf_fast(r, v, t)[0]
    try:
        F.use_eop_file(iers.IERS_A_FILE)  # a different table object, same data
        after = F.teme_to_gcrf_fast(r, v, t)[0]
        # the cache was rebuilt for the new table (one entry per node actually used)
        assert F._NODE_CACHE_TABLE[0] is iers.earth_orientation_table.get()
        np.testing.assert_allclose(after, before, rtol=0, atol=1e-9)
    finally:
        F.configure_offline_eop()
    again = F.teme_to_gcrf_fast(r, v, t)[0]
    np.testing.assert_allclose(again, before, rtol=0, atol=1e-12)
