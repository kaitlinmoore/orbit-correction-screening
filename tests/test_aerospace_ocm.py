"""Tests for the OCM reader (``scripts/aerospace_ocm.py``) and the test-set subset rule.

The fixture ``tests/fixtures/ocm_synthetic.ocm`` is synthetic: two TRAJ blocks of
analytic circular two-body orbits (the second after a 0.5 deg plane change), so
every value it holds can be recomputed here independently of the reader.
"""

import importlib
import sys

import numpy as np
import pytest

from orbit_correction.types import Trajectory

from .conftest import FIXTURES, REPO

# The scripts are not a package: put scripts/ on the path and import them by name
# (compare_aerospace_testset imports aerospace_ocm the same way).
sys.path.insert(0, str(REPO / "scripts"))
OCM = importlib.import_module("aerospace_ocm")
C = importlib.import_module("compare_aerospace_testset")

FIX = FIXTURES / "ocm_synthetic.ocm"

# The fixture's orbit (see the COMMENT lines in the file).
MU, R = 398600.4418, 6878.137
N = np.sqrt(MU / R**3)
INC = np.radians(51.6)
P1, Q1 = np.array([1.0, 0.0, 0.0]), np.array([0.0, np.cos(INC), np.sin(INC)])
TM = 180.0  # block 2 starts at 12:03:00, seconds after 12:00:00 UTC
#: TT - UTC in 2025 (37 leap seconds + 32.184 s), and 2025-01-01T12:00:00 as a
#: TT label minus J2000.0 (2000-01-01T12:00:00 TT), counted in whole days.
TT_MINUS_UTC = 69.184
T_NOON_TT = (np.datetime64("2025-01-01T12:00:00") - np.datetime64("2000-01-01T12:00:00")
             ).astype(np.int64) + TT_MINUS_UTC


def circ(p, q, tref, t):
    th = N * (np.asarray(t) - tref)
    r = R * (np.cos(th)[:, None] * p + np.sin(th)[:, None] * q)
    v = R * N * (-np.sin(th)[:, None] * p + np.cos(th)[:, None] * q)
    return r, v


def block2_axes():
    rm, vm = circ(P1, Q1, 0.0, [TM])
    p2 = rm[0] / R
    q = vm[0] / (R * N)
    ang = np.radians(0.5)
    q2 = q * np.cos(ang) + np.cross(p2, q) * np.sin(ang) + p2 * np.dot(p2, q) * (1 - np.cos(ang))
    return p2, q2


@pytest.fixture(scope="module")
def parsed():
    return OCM.read_ocm(FIX)


def test_parse_structure(parsed):
    assert parsed.designator == "70001"
    assert parsed.time_system == "UTC"
    assert len(parsed.trajectories) == 2
    assert [b.epochs.size for b in parsed.trajectories] == [11, 13]
    assert parsed.block_counts == {"PHYS": 1, "COV": 1, "OD": 1}
    assert parsed.od["OD_EPOCH"] == "2024-12-30T01:00:00.000000"
    assert parsed.trajectories[0].frame == "EME2000"
    lo, hi = parsed.trajectories[0].useable
    assert str(lo) == "2025-01-01T11:57:00.000000" and str(hi) == "2025-01-01T12:03:00.000000"


def test_samples_match_the_analytic_orbit(parsed):
    b1, b2 = parsed.trajectories
    t1 = (b1.epochs - np.datetime64("2025-01-01T12:00:00", "us")).astype(np.int64) / 1e6
    r, v = circ(P1, Q1, 0.0, t1)
    np.testing.assert_allclose(b1.states[:, :3], r, atol=1e-9)
    np.testing.assert_allclose(b1.states[:, 3:], v, atol=1e-9)
    p2, q2 = block2_axes()
    t2 = (b2.epochs - np.datetime64("2025-01-01T12:00:00", "us")).astype(np.int64) / 1e6
    r, v = circ(p2, q2, TM, t2)
    np.testing.assert_allclose(b2.states[:, :3], r, atol=1e-9)
    np.testing.assert_allclose(b2.states[:, 3:], v, atol=1e-9)


def test_utc_to_tt_is_label_plus_69_184_s(parsed):
    t = OCM.epochs_to_tt_seconds(parsed.trajectories[0].epochs, "UTC")
    np.testing.assert_allclose(t, T_NOON_TT + np.arange(-300.0, 301.0, 60.0), rtol=0, atol=1e-6)
    tt = OCM.epochs_to_tt_seconds(parsed.trajectories[0].epochs, "TT")
    np.testing.assert_allclose(t - tt, TT_MINUS_UTC, atol=1e-6)
    tai = OCM.epochs_to_tt_seconds(parsed.trajectories[0].epochs, "TAI")
    np.testing.assert_allclose(tai - tt, 32.184, atol=1e-6)


def test_leap_second_inside_a_block_is_refused():
    ep = np.array(["2016-12-31T23:59:00", "2017-01-01T00:01:00"], dtype="datetime64[us]")
    with pytest.raises(OCM.OcmError, match="leap second"):
        OCM.epochs_to_tt_seconds(ep, "UTC")


def test_frame_bias_rotation():
    b = OCM.FRAME_BIAS
    np.testing.assert_allclose(b @ b.T, np.eye(3), atol=1e-15)
    angle_arcsec = np.degrees(np.arccos((np.trace(b) - 1) / 2)) * 3600
    assert angle_arcsec == pytest.approx(0.0230, abs=2e-4)
    rng = np.random.default_rng(1)
    s = rng.normal(size=(5, 6)) * [7000, 7000, 7000, 7, 7, 7]
    g = OCM.to_gcrf(s, "EME2000")
    np.testing.assert_allclose(OCM.from_gcrf(g, "EME2000"), s, atol=1e-9)
    # a rotation: norms and every pairwise distance are unchanged
    np.testing.assert_allclose(np.linalg.norm(g[:, :3], axis=1), np.linalg.norm(s[:, :3], axis=1))
    np.testing.assert_allclose(np.linalg.norm(g[0, :3] - g[1, :3]),
                               np.linalg.norm(s[0, :3] - s[1, :3]), rtol=1e-14)
    # size of the shift at LEO radius: under a metre
    shift = np.linalg.norm(g[:, :3] - s[:, :3], axis=1) / np.linalg.norm(s[:, :3], axis=1)
    assert np.all(shift < 1.2e-7)
    np.testing.assert_array_equal(OCM.to_gcrf(s, "GCRF"), s)
    with pytest.raises(OCM.OcmError):
        OCM.to_gcrf(s, "TEME")


def test_hermite_reproduces_polynomials_and_nodes():
    t = np.array([0.0, 50.0, 110.0, 160.0, 230.0, 300.0])
    coef = np.array([[1.0, -2.0, 0.5], [0.3, 0.1, -0.2], [1e-3, 2e-3, 3e-3], [1e-6, -1e-6, 2e-6]])

    def poly(x):
        x = np.asarray(x)[:, None]
        return coef[0] + coef[1] * x + coef[2] * x**2 + coef[3] * x**3

    def dpoly(x):
        x = np.asarray(x)[:, None]
        return coef[1] + 2 * coef[2] * x + 3 * coef[3] * x**2

    tq = np.array([0.0, 7.5, 50.0, 133.3, 299.0, 300.0])
    for m in (2, 3, 4):
        r, v = OCM.hermite_interpolate(t, poly(t), dpoly(t), tq, m=m)
        np.testing.assert_allclose(r, poly(tq), rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(v, dpoly(tq), rtol=1e-10, atol=1e-12)
    with pytest.raises(ValueError):
        OCM.hermite_interpolate(t, poly(t), dpoly(t), [301.0])


def test_hermite_on_the_orbit_held_back_samples():
    """60 s LEO samples, degree 7: error at mid-points far below a millimetre."""
    t = np.arange(0.0, 1201.0, 60.0)
    r, v = circ(P1, Q1, 0.0, t)
    tq = t[:-1] + 30.0
    rq, vq = OCM.hermite_interpolate(t, r, v, tq, m=4)
    r_ref, v_ref = circ(P1, Q1, 0.0, tq)
    assert np.abs(rq - r_ref).max() < 1e-6  # km
    assert np.abs(vq - v_ref).max() < 1e-9  # km/s


def test_lagrange_reproduces_polynomials_and_orbit():
    t = np.array([0.0, 40.0, 100.0, 150.0, 220.0, 260.0, 330.0, 400.0, 450.0, 520.0])
    rng = np.random.default_rng(3)
    scale = np.array([1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14])[:, None]
    c = rng.normal(size=(8, 3)) * scale

    def poly(x):
        return sum(c[p] * np.asarray(x)[:, None] ** p for p in range(8))

    def dpoly(x):
        return sum(p * c[p] * np.asarray(x)[:, None] ** (p - 1) for p in range(1, 8))

    tq = np.array([0.0, 17.0, 150.0, 333.3, 519.0, 520.0])
    r, v = OCM.lagrange_interpolate(t, poly(t), tq, n_points=8)  # degree 7: exact
    np.testing.assert_allclose(r, poly(tq), rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(v, dpoly(tq), rtol=1e-7, atol=1e-12)

    # 60 s LEO samples, 8 points: mid-point position error far below a millimetre,
    # and the derivative is the orbit's velocity to better than 0.01 mm/s
    t = np.arange(0.0, 1201.0, 60.0)
    rs, _ = circ(P1, Q1, 0.0, t)
    tq = t[:-1] + 30.0
    rq, vq = OCM.lagrange_interpolate(t, rs, tq)
    r_ref, v_ref = circ(P1, Q1, 0.0, tq)
    assert np.abs(rq - r_ref).max() < 1e-6
    assert np.abs(vq - v_ref).max() < 1e-8


def test_inconsistent_file_velocities():
    """Why positions-only is the default: the test set's velocities are not d/dt of
    its positions. With random 5 m/s errors on the sampled velocities, Hermite bends
    the positions by metres between samples; Lagrange on positions does not care."""
    t = np.arange(0.0, 1201.0, 60.0)
    r, v = circ(P1, Q1, 0.0, t)
    dv = np.random.default_rng(7).normal(scale=0.005, size=v.shape)  # km/s
    v_bad = v + dv
    blk = OCM.TtBlock(t, np.hstack([r, v_bad]), (0.0, 1200.0))
    tq = t[2:-3] + 15.0
    r_ref, v_ref = circ(P1, Q1, 0.0, tq)
    rl, vl, how = OCM.resample([blk], tq)
    assert how == "lagrange"
    assert np.abs(rl - r_ref).max() < 1e-6 and np.abs(vl - v_ref).max() < 1e-8
    rh, _, how = OCM.resample([blk], tq, method="hermite")
    assert how == "hermite"
    assert np.linalg.norm(rh - r_ref, axis=1).max() > 1e-3  # more than a metre off
    _, vf, _ = OCM.resample([blk], tq, velocity="file")
    assert np.linalg.norm(vf - v_ref, axis=1).max() > 1e-3  # the file's errors come through
    # on exact samples: positions as written; velocity derivative or file
    rs, vs, how = OCM.resample([blk], t[3:6])
    assert how == "samples"
    np.testing.assert_array_equal(rs, r[3:6])
    assert np.abs(vs - v[3:6]).max() < 1e-8
    _, vs, _ = OCM.resample([blk], t[3:6], velocity="file")
    np.testing.assert_array_equal(vs, v_bad[3:6])


def test_resample_samples_and_block_ownership(parsed):
    blocks = OCM.tt_blocks(parsed)
    # USEABLE spans in TT
    assert blocks[0].useable_tt_s == pytest.approx((T_NOON_TT - 180.0, T_NOON_TT + TM))
    assert OCM.useable_span_tt(blocks) == pytest.approx((T_NOON_TT - 180.0, T_NOON_TT + 600.0))

    # on block 1's own 60 s samples: taken as they are
    t_out = T_NOON_TT + np.arange(-180.0, 121.0, 60.0)
    r, v, how = OCM.resample(blocks[:1], t_out)
    assert how == "samples"
    np.testing.assert_allclose(r, OCM.to_gcrf(parsed.trajectories[0].states, "EME2000")[2:8, :3])

    # across the maneuver: before 12:03 block 1, from 12:03 block 2 (post-maneuver velocity)
    t_out = T_NOON_TT + np.array([150.0, 170.0, TM, 200.0, 400.0])
    r, v, how = OCM.resample(blocks, t_out)
    assert how == "lagrange"  # neither block is sampled exactly at these times
    p2, q2 = block2_axes()
    r1, v1 = circ(P1, Q1, 0.0, [150.0, 170.0])
    r2, v2 = circ(p2, q2, TM, [TM, 200.0, 400.0])
    ref_r = OCM.to_gcrf(np.hstack([np.vstack([r1, r2]), np.vstack([v1, v2])]), "EME2000")
    np.testing.assert_allclose(r, ref_r[:, :3], atol=1e-6)
    np.testing.assert_allclose(v, ref_r[:, 3:], atol=1e-8)
    # the velocity really is block 2's: 0.5 deg away from block 1's at the maneuver
    _, vb1 = circ(P1, Q1, 0.0, [TM])
    turn = np.degrees(np.arccos(np.dot(v[2], OCM.to_gcrf(np.hstack([[np.zeros(3)], vb1]),
                                                       "EME2000")[0, 3:])
                                / np.linalg.norm(v[2]) ** 2))
    assert turn == pytest.approx(0.5, abs=1e-6)

    # outside every USEABLE interval: refused (11:56 is data but not useable)
    with pytest.raises(ValueError, match="USEABLE"):
        OCM.resample(blocks, np.array([T_NOON_TT - 240.0]))


def test_to_trajectory_meets_the_contract(parsed):
    blocks = OCM.tt_blocks(parsed)
    t_out = T_NOON_TT + np.linspace(-180.0, 600.0, 27)
    tr, how = OCM.to_trajectory(blocks, 70001, t_out)
    assert isinstance(tr, Trajectory)
    assert tr.frame == "GCRF" and tr.time_scale == "TT" and tr.source == "aerospace_ivv"
    assert how == "lagrange"  # neither block is sampled exactly at these times
    _, how = OCM.to_trajectory(blocks, 70001, t_out, method="hermite")
    assert how == "hermite"


def test_scan_header_agrees_with_full_parse(parsed):
    scan = OCM.scan_ocm_header(FIX.read_bytes())
    assert scan["OBJECT_DESIGNATOR"] == "70001"
    assert scan["OD_EPOCH"] == "2024-12-30T01:00:00.000000"
    assert scan["n_traj_blocks"] == 2
    assert scan["n_samples"] == 24
    assert scan["traj_ref_frames"] == "EME2000"
    assert scan["useable_start"] == "2025-01-01T11:57:00.000000"
    assert scan["useable_stop"] == "2025-01-01T12:10:00.000000"
    assert scan["block0_n"] == 11
    assert scan["n_cov_blocks"] == 1
    np.testing.assert_allclose(scan["first_state"], parsed.trajectories[0].states[0])
    st = OCM.state_summary(scan["first_state"])
    assert st["a_km"] == pytest.approx(R, abs=1e-6)
    assert st["e"] < 1e-9


def test_unsupported_structure_is_refused():
    text = FIX.read_text()
    with pytest.raises(OCM.OcmError, match="TRAJ_REF_FRAME|not supported"):
        b = OCM.tt_blocks(OCM.parse_ocm(text.replace("= EME2000", "= ITRF", 1)))
        del b
    with pytest.raises(OCM.OcmError, match="TIME_SYSTEM"):
        OCM.parse_ocm(text.replace("TIME_SYSTEM              = UTC", "TIME_SYSTEM = GPS"))
    with pytest.raises(OCM.OcmError, match="TRAJ_UNITS"):
        OCM.parse_ocm(text.replace("[km, km, km, km/s, km/s, km/s]", "[m, m, m, m/s, m/s, m/s]", 1))
    with pytest.raises(OCM.OcmError, match="epoch format"):
        OCM.parse_epoch("2025-001T00:00:00")


def test_subset_rule_is_deterministic():
    # pinned values: the sampler must not change silently
    assert C.hash_u(25544) == 0.3541567695224626
    assert C.hash_u(5) == 0.08667111130990313
    u = [C.hash_u(i) for i in range(1, 2001)]
    assert 0.0 <= min(u) and max(u) < 1.0
    assert abs(np.mean(np.array(u) < 0.10) - 0.10) < 0.03
    assert C.hash_u(25544, seed=1) != C.hash_u(25544)
    assert C.stem_id("43476_candidate07.ocm") == 43476
    assert C.id_type(95001) == "historical_cdm"
    oid = next(i for i in range(5, 5000) if C.hash_u(i) < C.MAIN_FRACTION)
    assert C.select_group(f"{oid}.ocm", {"alt_a_km": 550.0}, set()) == "main"
    assert C.select_group(f"{oid}.ocm", {"alt_a_km": 650.0}, set()) is None
    assert C.select_group(f"{oid}_candidate01.ocm", {"alt_a_km": 550.0}, set()) is None
    assert C.select_group("90010.ocm", {}, set()) == "maneuver"
    assert C.select_group("12345.ocm", {}, {"12345.ocm"}) == "fictitious"
