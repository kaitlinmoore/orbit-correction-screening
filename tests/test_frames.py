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

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import GCRS, ITRS, CartesianDifferential, CartesianRepresentation
from astropy.time import Time
from astropy.utils import iers
from sgp4.api import WGS72, Satrec

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
