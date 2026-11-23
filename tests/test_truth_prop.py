"""Tests for ``orbit_correction.truth_prop`` (the Orekit truth propagator).

Two kinds of test live here:

* **Always run** (default ``uv sync``): JVM isolation, configuration, the B* relation.
* **``datagen``** (need ``uv sync --group datagen``; skipped cleanly without it):
  closed-form physics checks, independent-implementation checks and determinism.
  They share one JVM per pytest session (the ``ctx`` fixture), because a JVM
  can't be restarted inside a process.
* **``crosscheck``** (brahe): only with ``ORBIT_CROSSCHECK_TESTS=1`` and the
  ``crosscheck`` group installed.

Synthetic states use epochs where the fixture space-weather file
(``tests/fixtures/sw_celestrak_2024_extract.txt``) is fully observed, or the
J2000 epoch when no space weather is needed.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from orbit_correction import truth_prop as T
from orbit_correction.frames import utc_to_tt_seconds
from orbit_correction.types import Trajectory

from .conftest import FIXTURES, REPO

HAVE_OREKIT = all(importlib.util.find_spec(m) for m in ("orekit_jpype", "jdk4py", "orekitdata"))
HAVE_BRAHE = importlib.util.find_spec("brahe") is not None


def datagen(f):
    """Mark a test as needing the ``datagen`` group (Orekit); skip it otherwise."""
    f = pytest.mark.skipif(not HAVE_OREKIT, reason="needs `uv sync --group datagen` (Orekit)")(f)
    return pytest.mark.datagen(f)


DAY = 86400.0
SW_FIXTURE = FIXTURES / "sw_celestrak_2024_extract.txt"
#: Window of the fixture space-weather copy used by the drag tests.
SW_FIRST_UTC, SW_LAST_UTC = "2024-02-01T00:00:00", "2024-11-15T00:00:00"
MU_KM = 398600.4415  # only for building synthetic states; tests use Orekit's mu in checks


def circular_state(alt_km: float, inc_deg: float, t0: float, raan_deg: float = 0.0, nid=1):
    """A circular-velocity GCRF state at altitude ``alt_km`` above the equatorial radius."""
    r = 6378.137 + alt_km
    v = np.sqrt(MU_KM / r)
    i, o = np.radians(inc_deg), np.radians(raan_deg)
    pos = r * np.array([np.cos(o), np.sin(o), 0.0])
    vel = v * np.array([-np.sin(o) * np.cos(i), np.cos(o) * np.cos(i), np.sin(i)])
    return Trajectory(nid, np.array([t0]), pos[None, :], vel[None, :], "test")


def kepler_state(a_km, e, inc_deg, raan_deg, argp_deg, nu_deg, t0, mu_km=MU_KM, nid=1):
    """GCRF state from classical elements (true anomaly), for synthetic test orbits."""
    i, o, w, nu = np.radians([inc_deg, raan_deg, argp_deg, nu_deg])
    p = a_km * (1 - e**2)
    r_pf = p / (1 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu_km / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    co, so, ci, si, cw, sw = np.cos(o), np.sin(o), np.cos(i), np.sin(i), np.cos(w), np.sin(w)
    rot = np.array(
        [
            [co * cw - so * sw * ci, -co * sw - so * cw * ci, so * si],
            [so * cw + co * sw * ci, -so * sw + co * cw * ci, -co * si],
            [sw * si, cw * si, ci],
        ]
    )
    return Trajectory(nid, np.array([t0]), (rot @ r_pf)[None, :], (rot @ v_pf)[None, :], "test")


def elements(r, v, mu):
    """Osculating a, e, i, RAAN, argp, mean anomaly from (N, 3) states (independent of Orekit)."""
    rn = np.linalg.norm(r, axis=1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1)
    inc = np.arccos(h[:, 2] / hn)
    node = np.cross(np.array([0.0, 0.0, 1.0]), h)
    raan = np.arctan2(node[:, 1], node[:, 0])
    ev = np.cross(v, h) / mu - r / rn[:, None]
    e = np.linalg.norm(ev, axis=1)
    nn = np.linalg.norm(node, axis=1)
    argp = np.arctan2(
        np.einsum("ij,ij->i", np.cross(node, ev), h) / hn, np.einsum("ij,ij->i", node, ev)
    )
    del nn
    nu = np.arctan2(np.einsum("ij,ij->i", np.cross(ev, r), h) / hn, np.einsum("ij,ij->i", ev, r))
    big_e = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
    m = big_e - e * np.sin(big_e)
    a = 1.0 / (2.0 / rn - np.einsum("ij,ij->i", v, v) / mu)
    return a, e, inc, raan, argp, m


# --------------------------------------------------------------------------------------
# Always run: JVM isolation and configuration
# --------------------------------------------------------------------------------------


def test_importing_project_modules_starts_no_jvm():
    """Importing any project module (truth_prop included) imports no jpype and starts no JVM."""
    code = textwrap.dedent(
        """
        import sys
        import orbit_correction
        from orbit_correction import types, frames, spaceweather, catalog, splits, truth_prop
        bad = [m for m in ("jpype", "orekit_jpype", "jdk4py", "_jpype") if m in sys.modules]
        assert not bad, bad
        assert not truth_prop.jvm_started()
        cfg = truth_prop.load_truth_config()
        truth_prop.area_to_mass_from_bstar(1e-4, cfg)
        assert "jpype" not in sys.modules
        print("ok")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_config_loads_with_decision_1_values():
    cfg = T.load_truth_config()
    assert cfg.gravity_field_file == "eigen-6s.gfc" and cfg.zonal_order == 0
    assert cfg.zonal_degree >= 2
    assert cfg.atmosphere == "NRLMSISE-00" and cfg.cd == 2.2
    assert cfg.integrator == "DOP853"
    assert cfg.position_tolerance_m == 1.0 and cfg.max_step_s == 300.0
    assert cfg.output_step_s == 60.0
    assert cfg.iers_conventions == "IERS_2010"
    assert cfg.orekit_data_commit.startswith("2d6edc9")
    # the pinned commit in pyproject.toml is the one the config names
    assert cfg.orekit_data_commit in (REPO / "pyproject.toml").read_text()


def test_config_validation():
    cfg = T.load_truth_config()
    for bad in (
        dict(zonal_order=2),
        dict(zonal_degree=1),
        dict(atmosphere="JB2008"),
        dict(msis_ap_switch=0),
        dict(integrator="RK4"),
        dict(max_step_s=-1.0),
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(cfg, **bad)


def test_area_to_mass_from_bstar():
    cfg = T.load_truth_config()
    # A/m = 2 B* / (Cd rho0): B* = 1e-4 /ER, Cd 2.2, rho0 0.15696615 -> 5.7917e-4 m^2/kg
    assert T.area_to_mass_from_bstar(1e-4, cfg) == pytest.approx(2e-4 / (2.2 * 0.15696615))
    # and back: B* = Cd (A/m) rho0 / 2
    am = T.area_to_mass_from_bstar(3.3e-4, cfg)
    assert cfg.cd * am * cfg.bstar_reference_density_kg_per_m2_er / 2 == pytest.approx(3.3e-4)
    for bad in (0.0, -2.2675e-05, float("nan")):
        with pytest.raises(ValueError, match="area_to_mass"):
            T.area_to_mass_from_bstar(bad, cfg)


def test_output_grid():
    g = T.output_grid(100.0, 7 * DAY, 60.0)
    assert g.size == 10081 and g[0] == 100.0 and g[-1] == 100.0 + 7 * DAY
    assert np.all(np.diff(g) == pytest.approx(60.0))


# --------------------------------------------------------------------------------------
# datagen: shared JVM and inputs
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cfg():
    return T.load_truth_config()


@pytest.fixture(scope="session")
def ctx(cfg):
    """The session's Orekit context: one JVM per pytest session."""
    if not HAVE_OREKIT:
        pytest.skip("needs the datagen group")
    return T._context(cfg)


@pytest.fixture(scope="session")
def sw_copy(tmp_path_factory):
    """Orekit copy of the fixture space-weather file, for SW_FIRST..SW_LAST."""
    path = tmp_path_factory.mktemp("sw") / "sw_orekit.txt"
    rep = T.write_space_weather_copy(
        SW_FIXTURE, path, utc_to_tt_seconds(SW_FIRST_UTC), utc_to_tt_seconds(SW_LAST_UTC)
    )
    assert rep.appended_empty_monthly_fit
    return path


@pytest.fixture(scope="session")
def gp_records():
    return {
        r["NORAD_CAT_ID"]: r
        for r in json.loads((FIXTURES / "celestrak_gp_sample.json").read_text())
    }


def _sample(ctx, traj_or_result):
    tr = traj_or_result.trajectory if hasattr(traj_or_result, "trajectory") else traj_or_result
    return tr.r_gcrf_km * 1000.0, tr.v_gcrf_km_s * 1000.0  # metres, for physics checks


# --------------------------------------------------------------------------------------
# datagen: closed-form physics
# --------------------------------------------------------------------------------------


@datagen
def test_two_body_energy_conserved_7_days(cfg, ctx):
    """Point mass, no drag: specific energy and angular momentum stay constant for 7 days."""
    init = kepler_state(7000.0, 0.05, 51.6, 30.0, 40.0, 0.0, t0=DAY)
    drift = {}
    for tol in (1.0, 1e-3):
        c = dataclasses.replace(cfg, zonal_degree=0, atmosphere="none", position_tolerance_m=tol)
        res = T.propagate_truth(init, None, T.output_grid(DAY, 7 * DAY, 60.0), c, None)
        assert res.status == "ok"
        r, v = _sample(ctx, res)
        energy = 0.5 * np.sum(v**2, axis=1) - ctx.mu / np.linalg.norm(r, axis=1)
        h = np.linalg.norm(np.cross(r, v), axis=1)
        drift[tol] = (np.max(np.abs(energy / energy[0] - 1)), np.max(np.abs(h / h[0] - 1)))
    # At the truth tolerance (1 m) the first run gave 3.2e-8 relative energy drift over
    # 7 days (about 0.2 m of semi-major axis); bounds are set from that.
    assert drift[1.0][0] < 1e-7 and drift[1.0][1] < 1e-7
    # The drift is integration error, not a force: it shrinks with the tolerance
    # (first run: 3.2e-8 at 1 m, 7.5e-9 at 1 mm; at 1 mm the dense-output
    # interpolation and rounding floor dominate, so the drop is not 1000-fold).
    assert drift[1e-3][0] < drift[1.0][0] / 3 and drift[1e-3][1] < drift[1.0][1] / 3


@datagen
def test_j2_secular_rates_match_closed_form(cfg, ctx):
    """J2 only: fitted node, perigee and mean-anomaly rates over 12 days vs first-order theory.

    Epoch 2000-01-02, so the GCRF pole and the Earth's pole agree to nutation
    level (arcseconds) and GCRF elements are the elements J2 acts on.
    """
    c = dataclasses.replace(cfg, zonal_degree=2, atmosphere="none")
    t0 = DAY
    init = kepler_state(7000.0, 0.02, 50.0, 30.0, 40.0, 10.0, t0=t0)
    t = T.output_grid(t0, 12 * DAY, 60.0)
    res = T.propagate_truth(init, None, t, c, None)
    r, v = _sample(ctx, res)
    a, e, inc, raan, argp, m = elements(r, v, ctx.mu)
    prov = ctx.gravity_provider(2)
    j2 = -np.sqrt(5.0) * prov.onDate(ctx.date(t0)).getNormalizedCnm(2, 0)
    re = prov.getAe()
    am, em, im = a.mean(), e.mean(), inc.mean()
    n = np.sqrt(ctx.mu / am**3)
    p = am * (1 - em**2)
    k = n * j2 * (re / p) ** 2
    raan_dot = -1.5 * k * np.cos(im)
    argp_dot = 0.75 * k * (5 * np.cos(im) ** 2 - 1)
    m_dot_j2 = 0.75 * k * np.sqrt(1 - em**2) * (3 * np.cos(im) ** 2 - 1)
    dt = t - t0
    fit = {
        "raan": np.polyfit(dt, np.unwrap(raan), 1)[0],
        "argp": np.polyfit(dt, np.unwrap(argp), 1)[0],
        "m": np.polyfit(dt, np.unwrap(m), 1)[0] - n,
    }
    assert fit["raan"] == pytest.approx(raan_dot, rel=2e-3)
    assert fit["argp"] == pytest.approx(argp_dot, rel=5e-3)
    assert fit["m"] == pytest.approx(m_dot_j2, rel=3e-2)


def _zonal_closed_form(x, y, z, mu, re, j2, j3, j4):
    """Closed-form J2, J3, J4 acceleration in the Earth-fixed frame (Vallado 2013, sec. 8.7)."""
    r2 = x * x + y * y + z * z
    r = np.sqrt(r2)
    zr2 = z * z / r2
    a2 = -1.5 * j2 * mu * re**2 / r**5
    acc = np.array([a2 * x * (1 - 5 * zr2), a2 * y * (1 - 5 * zr2), a2 * z * (3 - 5 * zr2)])
    a3 = -2.5 * j3 * mu * re**3 / r**7
    acc += np.array(
        [
            a3 * x * (3 * z - 7 * z**3 / r2),
            a3 * y * (3 * z - 7 * z**3 / r2),
            a3 * (6 * z * z - 7 * z**4 / r2 - 0.6 * r2),
        ]
    )
    a4 = 1.875 * j4 * mu * re**4 / r**7
    acc += np.array(
        [
            a4 * x * (1 - 14 * zr2 + 21 * zr2**2),
            a4 * y * (1 - 14 * zr2 + 21 * zr2**2),
            a4 * z * (5 - 70 / 3 * zr2 + 21 * zr2**2),
        ]
    )
    return acc


@datagen
def test_zonal_truncation_tesserals_zero_and_j2_j4_closed_form(cfg, ctx):
    """Order-0 field: no tesseral terms, and degree 4 equals closed-form J2-J4."""
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel

    date = ctx.date(utc_to_tt_seconds("2024-06-15T00:00:00"))
    prov = ctx.gravity_provider(cfg.zonal_degree)
    assert prov.getMaxOrder() == 0 and prov.getMaxDegree() == cfg.zonal_degree

    # tesserals exactly zero: the field is symmetric about the pole, so rotating a
    # point about z rotates the acceleration by the same angle, to rounding (the
    # provider has order 0, so no tesseral coefficient is even loaded)
    hf = HolmesFeatherstoneAttractionModel(ctx.itrf, prov)
    rng = np.random.default_rng(3)
    for _ in range(20):
        rad = rng.uniform(6.6e6, 7.8e6)
        lat, lon, dlon = (
            rng.uniform(-1.4, 1.4),
            rng.uniform(-np.pi, np.pi),
            rng.uniform(0, 2 * np.pi),
        )
        p1 = rad * np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        c, s = np.cos(dlon), np.sin(dlon)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        g1 = np.array(list(hf.gradient(date, Vector3D(*p1), ctx.mu)))
        g2 = np.array(list(hf.gradient(date, Vector3D(*(rot @ p1)), ctx.mu)))
        # rounding only: 1.5e-15 relative at degree 36 in the first run; one tesseral
        # term as small as C22 (about 2.4e-6 normalised) would break symmetry by ~1e-3
        np.testing.assert_allclose(g2, rot @ g1, rtol=0, atol=1e-13 * np.linalg.norm(g1))

    # degree 4 against closed form with the same (date-evaluated) coefficients
    p4 = ctx.gravity_provider(4)
    hf4 = HolmesFeatherstoneAttractionModel(ctx.itrf, p4)
    h = p4.onDate(date)
    js = {n: -np.sqrt(2 * n + 1) * h.getNormalizedCnm(n, 0) for n in (2, 3, 4)}
    for _ in range(20):
        p = rng.normal(size=3)
        p *= rng.uniform(6.6e6, 7.8e6) / np.linalg.norm(p)
        got = np.array(list(hf4.gradient(date, Vector3D(*p), ctx.mu)))
        want = _zonal_closed_form(*p, ctx.mu, p4.getAe(), js[2], js[3], js[4])
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-13 * np.linalg.norm(want))


@datagen
def test_drag_power_equals_energy_loss_rate(cfg, ctx, sw_copy):
    """Point mass + NRLMSISE-00: the energy lost equals the work done by drag.

    Two forms, both against v . a_drag from Orekit's DragForce at the integrated
    states:

    * integral: over each of three orbits, E(end) - E(start) = integral of v . a_drag dt
      (first run: 3e-5 relative at 1 mm tolerance, 3e-6 at 0.01 mm);
    * pointwise: dE/dt from a 5-point central difference of the ephemeris equals
      v . a_drag at sample points. The residual is noise from differentiating the
      dense-output interpolant: up to 4e-3 at 1 mm tolerance and 6e-4 at 0.1 mm
      (first runs), the same size with a smooth exponential atmosphere, so the
      pointwise check runs at 0.1 mm. Samples avoid the 3-hour ap boundaries,
      where the density steps.
    """
    from scipy.integrate import simpson

    c = dataclasses.replace(cfg, zonal_degree=0, position_tolerance_m=1e-4)
    t0 = utc_to_tt_seconds("2024-05-10T00:00:00")
    init = circular_state(300.0, 51.6, t0)
    prop, _, _ = T.build_propagator(c, init, 0.01, sw_copy)
    gen = prop.getEphemerisGenerator()
    prop.propagate(ctx.date(t0), ctx.date(t0 + 0.3 * DAY))
    eph = gen.getGeneratedEphemeris()
    drag = [f for f in prop.getAllForceModels() if "DragForce" in str(f.getClass())][0]

    def energy(t):
        pv = eph.getPVCoordinates(ctx.date(float(t)), ctx.gcrf)
        return 0.5 * pv.getVelocity().getNormSq() - ctx.mu / pv.getPosition().getNorm()

    def power(t):
        state = eph.propagate(ctx.date(float(t)))
        acc = drag.acceleration(state, drag.getParameters(state.getDate()))
        return state.getPVCoordinates().getVelocity().dotProduct(acc)

    period = 2 * np.pi * np.sqrt((np.linalg.norm(init.r_gcrf_km[0]) * 1e3) ** 3 / ctx.mu)
    for k in range(3):
        ts = t0 + 300.0 + k * period + np.linspace(0.0, period, 541)
        work = simpson([power(t) for t in ts], x=ts)
        assert work < 0
        assert (energy(ts[-1]) - energy(ts[0])) / work == pytest.approx(1.0, abs=1e-4)

    h = 30.0
    rel = []
    for tk in t0 + np.linspace(
        900.0, 0.1 * DAY - 900.0, 12
    ):  # 00:15-02:09 UTC, inside one ap interval
        dedt = (
            -energy(tk + 2 * h) + 8 * energy(tk + h) - 8 * energy(tk - h) + energy(tk - 2 * h)
        ) / (12 * h)
        rel.append(dedt / power(tk) - 1)
    assert np.max(np.abs(rel)) < 2e-3, rel


@datagen
def test_circular_decay_under_exponential_atmosphere(cfg, ctx):
    """Non-rotating exponential atmosphere: circular decay vs da/dt = -Cd (A/m) rho sqrt(mu a)."""
    from org.orekit.bodies import OneAxisEllipsoid
    from org.orekit.models.earth.atmosphere import SimpleExponentialAtmosphere
    from scipy.integrate import solve_ivp

    re, h0, rho0, hs = 6378137.0, 400e3, 3e-12, 60e3
    body = OneAxisEllipsoid(re, 0.0, ctx.gcrf)  # spherical and inertial: no co-rotation
    atm = SimpleExponentialAtmosphere(body, rho0, h0, hs)
    c = dataclasses.replace(cfg, zonal_degree=0, position_tolerance_m=1e-3)
    t0 = utc_to_tt_seconds("2024-06-15T00:00:00")
    init = circular_state(400.0 + 0.137, 51.6, t0)  # 6778.274 km radius
    a2m = 0.01
    prop, _, _ = T.build_propagator(c, init, a2m, None, atmosphere=atm)
    days = 5.0
    final = prop.propagate(ctx.date(t0 + days * DAY))

    def sma(pv):
        return 1.0 / (2.0 / pv.getPosition().getNorm() - pv.getVelocity().getNormSq() / ctx.mu)

    a_end = sma(final.getPVCoordinates())
    a0 = np.linalg.norm(init.r_gcrf_km[0]) * 1000.0
    v0 = np.linalg.norm(init.v_gcrf_km_s[0]) * 1000.0
    a0 = 1.0 / (2.0 / a0 - v0**2 / ctx.mu)  # osculating a of the initial state
    bc = c.cd * a2m

    def rate(_, a):
        return -bc * rho0 * np.exp(-(a - re - h0) / hs) * np.sqrt(ctx.mu * a)

    sol = solve_ivp(rate, (0, days * DAY), [a0], rtol=1e-12, atol=1e-6)
    want = sol.y[0, -1] - a0
    got = a_end - a0
    assert want < -1000.0  # a real decay of more than 1 km
    # the orbit's eccentricity under drag oscillates slightly, which the
    # circular-orbit formula ignores: a 0.5% tolerance
    assert got == pytest.approx(want, rel=5e-3)


# --------------------------------------------------------------------------------------
# datagen: independent implementations
# --------------------------------------------------------------------------------------


DENSITY_SAMPLES = [
    ("2024-03-01T03:20:00", 350.0, 10.0, 20.0),
    ("2024-05-10T18:45:00", 450.0, -40.0, 120.0),  # during the May 2024 storm
    ("2024-05-11T06:10:00", 500.0, 60.0, -75.0),
    ("2024-07-04T12:00:00", 600.0, 0.0, 180.0),
    ("2024-09-20T22:30:00", 800.0, -70.0, -10.0),
    ("2024-10-10T09:00:00", 250.0, 30.0, 45.0),
]


@datagen
def test_nrlmsise00_matches_pymsis(cfg, ctx, sw_copy):
    """Orekit NRLMSISE-00 vs pymsis (msis00) at the same point, time and space weather."""
    import pymsis
    from org.hipparchus.util import FastMath
    from org.orekit.bodies import GeodeticPoint

    atm = ctx.atmosphere(sw_copy)
    swd = ctx.space_weather(sw_copy)
    ratios = []
    for utc, alt_km, lat, lon in DENSITY_SAMPLES:
        date = ctx.date(utc_to_tt_seconds(utc))
        gp = GeodeticPoint(FastMath.toRadians(lat), FastMath.toRadians(lon), alt_km * 1000.0)
        pos = ctx.earth.transform(gp)  # ITRF, metres
        rho_ok = atm.getDensity(date, pos, ctx.itrf)
        # identical inputs: Orekit's own space-weather values and its UT1 instant
        comp = date.getComponents(ctx.ut1)
        d, tm = comp.getDate(), comp.getTime()
        ut1 = np.datetime64(
            f"{d.getYear():04d}-{d.getMonth():02d}-{d.getDay():02d}"
        ) + np.timedelta64(int(round(tm.getSecondsInLocalDay() * 1e6)), "us")
        aps = list(swd.getAp(date))
        out = pymsis.calculate(
            np.array([ut1]),
            np.array([lon]),
            np.array([lat]),
            np.array([alt_km]),
            f107s=[swd.getDailyFlux(date)],
            f107as=[swd.getAverageFlux(date)],
            aps=[aps],
            version=0,
            geomagnetic_activity=cfg.msis_ap_switch,
        )
        ratios.append(rho_ok / float(np.squeeze(out)[0]) - 1)
    # Independent implementations (Orekit's Java port vs NRL's Fortran via pymsis).
    # First run: |ratio - 1| <= 1.1e-5 at these points with the configured settings
    # (mean local solar time, ap-array mode); tolerance 1e-4 from that. With Orekit's
    # constructor default (apparent solar time) the same points differ by up to 1.2%.
    assert np.max(np.abs(ratios)) < 1e-4, ratios


SW_SAMPLE_UTC = [
    "2024-03-01T03:20:00",
    "2024-05-10T18:45:00",
    "2024-05-11T06:10:00",
    "2024-07-04T12:00:01",
    "2024-09-20T22:30:00",
    "2024-11-14T01:00:00",
]


@datagen
def test_orekit_reads_our_space_weather_copy(cfg, ctx, sw_copy):
    """Orekit's daily F10.7, 81-day mean and Ap array equal spaceweather.observed values."""
    from orbit_correction.spaceweather import parse_spaceweather

    sw = parse_spaceweather(SW_FIXTURE)
    swd = ctx.space_weather(sw_copy)
    for utc in SW_SAMPLE_UTC:
        t = utc_to_tt_seconds(utc)
        date = ctx.date(t)
        ob = sw.observed(t)
        assert ob.section == "OBSERVED"
        assert swd.getDailyFlux(date) == pytest.approx(ob.f107_obs_prev_day_sfu, abs=1e-9)
        # the centred 81-day mean of OBSERVED F10.7 for the day containing t
        assert swd.getAverageFlux(date) == pytest.approx(ob.f107_obs_ctr81_sfu, abs=1e-9)
        ap = list(swd.getAp(date))
        assert ap[0] == ob.ap_daily and ap[1] == ob.ap_3h_current
        # 3, 6 and 9 hours earlier, and the 24-hour means 12-33 h and 36-57 h earlier
        for k, back_h in ((2, 3), (3, 6), (4, 9)):
            assert ap[k] == sw.observed(t - back_h * 3600).ap_3h_current
        for k, start_h in ((5, 12), (6, 36)):
            want = np.mean(
                [sw.observed(t - (start_h + 3 * j) * 3600).ap_3h_current for j in range(8)]
            )
            assert ap[k] == pytest.approx(want, abs=1e-12)


@datagen
def test_orekit_reads_the_copy_not_the_published_file(cfg, ctx, tmp_path):
    """A value changed in our copy is what Orekit returns: it reads the file we wrote."""
    from orbit_correction.spaceweather import parse_spaceweather, write_orekit_cssi

    sw = parse_spaceweather(SW_FIXTURE)
    i = int(np.flatnonzero(sw.dates == np.datetime64("2024-06-14"))[0])
    f107 = sw.scalars["f107_obs"].copy()
    f107[i] = 999.9
    sw2 = dataclasses.replace(sw, scalars={**sw.scalars, "f107_obs": f107})
    path = tmp_path / "marked.txt"
    write_orekit_cssi(sw2, path, "2024-06-01", "2024-06-30")
    swd = ctx.space_weather(path)
    # NRLMSISE-00's daily flux at a time on 2024-06-15 is the previous day's value
    assert swd.getDailyFlux(ctx.date(utc_to_tt_seconds("2024-06-15T12:00:00"))) == 999.9


TEME_EPOCHS_UTC = [
    "2000-01-02T00:00:00",
    "2009-02-10T16:56:00",
    "2016-12-31T23:59:59",
    "2017-01-01T00:00:01",
    "2024-05-10T18:45:00",
    "2026-11-22T12:00:00",
]


@datagen
def test_orekit_teme_to_gcrf_matches_frames_module(ctx):
    """Orekit's TEME -> GCRF agrees with frames.teme_to_gcrf (astropy) at sample epochs."""
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.utils import PVCoordinates

    from orbit_correction.frames import teme_to_gcrf

    r = np.array([6700.0, -1200.0, 1500.0])
    v = np.array([1.1, 6.9, 3.0])
    for utc in TEME_EPOCHS_UTC:
        t = utc_to_tt_seconds(utc)
        tr = ctx.teme.getTransformTo(ctx.gcrf, ctx.date(t))
        pv = tr.transformPVCoordinates(PVCoordinates(Vector3D(*(r * 1e3)), Vector3D(*(v * 1e3))))
        ro = np.array(pv.getPosition().toArray()) / 1e3
        vo = np.array(pv.getVelocity().toArray()) / 1e3
        ra, va = teme_to_gcrf(r, v, t)
        # First run: 0.53 m (2000) growing to 1.10 m (2026), 0.6-1.0 mm/s: a
        # systematic difference between Orekit's and astropy's TEME realisations
        # that grows with time from J2000. The pipeline uses frames.teme_to_gcrf for
        # both the cheap and the truth initial state, so it doesn't enter residuals.
        assert np.linalg.norm(ro - ra) < 2e-3, utc  # km: 2 m
        assert np.linalg.norm(vo - va) < 2e-6, utc  # km/s: 2 mm/s


SGP4_IDS = [
    902,
    44714,
    44753,
    44058,
]  # Calsphere 2, Starlink-1008, Starlink-1048, OneWeb-0010 (train role)


@datagen
def test_orekit_tle_propagator_matches_sgp4_package(ctx, gp_records):
    """Orekit's TLEPropagator and the sgp4 package agree on the same element sets (TEME)."""
    from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
    from sgp4 import exporter, omm
    from sgp4.api import Satrec

    for nid in SGP4_IDS:
        sat0 = Satrec()
        omm.initialize(sat0, {k: str(v) for k, v in gp_records[nid].items()})
        l1, l2 = exporter.export_tle(sat0)
        sat = Satrec.twoline2rv(l1, l2)  # both libraries read the same two lines
        tle = TLE(l1, l2)
        tp = TLEPropagator.selectExtrapolator(tle)
        worst = 0.0
        for dt in np.linspace(0.0, 3 * DAY, 37):
            err, r, _ = sat.sgp4_tsince(dt / 60.0)
            assert err == 0
            pv = tp.getPVCoordinates(tle.getDate().shiftedBy(float(dt)), ctx.teme)
            ro = np.array(pv.getPosition().toArray()) / 1e3
            worst = max(worst, np.linalg.norm(ro - np.array(r)))
        # first run: at most 9 micrometres over 3 days
        assert worst < 1e-5, (nid, worst)  # km: 1 cm


# --------------------------------------------------------------------------------------
# datagen: numerics, re-entry, determinism
# --------------------------------------------------------------------------------------

#: Synthetic starts for real element sets (see truth_prop.initial_state_from_omm).
SAMPLE_EPOCH_UTC = "2024-06-15T00:00:00"


@datagen
@pytest.mark.parametrize("case", ["starlink_1008_340km", "circular_800km"])
def test_integrator_tolerance_1m_vs_1cm(cfg, ctx, sw_copy, gp_records, case):
    """Position tolerance 1 m vs 1 cm: the 7-day difference stays small.

    The per-band values for the zonal-degree threshold come from
    scripts/zonal_convergence.py; this test guards the order of magnitude.
    """
    if case == "starlink_1008_340km":
        init, bstar = T.initial_state_from_omm(gp_records[44714], SAMPLE_EPOCH_UTC)
        a2m = None
    else:
        init, bstar = circular_state(800.0, 98.6, utc_to_tt_seconds(SAMPLE_EPOCH_UTC)), None
        a2m = 0.005
    grid = T.output_grid(init.t_tt_s[0], 7 * DAY, 3600.0)
    a = T.propagate_truth(init, bstar, grid, cfg, sw_copy, area_to_mass_m2_per_kg=a2m)
    b = T.propagate_truth(
        init,
        bstar,
        grid,
        dataclasses.replace(cfg, position_tolerance_m=0.01),
        sw_copy,
        area_to_mass_m2_per_kg=a2m,
    )
    d = np.linalg.norm(a.trajectory.r_gcrf_km - b.trajectory.r_gcrf_km, axis=1) * 1000.0
    assert d[-1] < 100.0, d[-1]
    assert d[24] < 5.0, d[24]  # 1 day


@datagen
def test_reentry_returns_status_not_partial_trajectory(cfg, ctx, sw_copy):
    init = circular_state(200.0, 51.6, utc_to_tt_seconds("2024-05-10T00:00:00"))
    grid = T.output_grid(init.t_tt_s[0], 7 * DAY, 60.0)
    res = T.propagate_truth(init, None, grid, cfg, sw_copy, area_to_mass_m2_per_kg=0.05)
    assert res.status == "reentry" and res.trajectory is None
    assert init.t_tt_s[0] < res.stop_t_tt_s < grid[-1]
    assert "120.0 km" in res.message


@datagen
def test_truth_output_satisfies_contract(cfg, ctx, sw_copy, gp_records):
    init, bstar = T.initial_state_from_omm(gp_records[44714], SAMPLE_EPOCH_UTC)
    grid = T.output_grid(init.t_tt_s[0], 1 * DAY, 60.0)
    res = T.propagate_truth(init, bstar, grid, cfg, sw_copy)
    tr = res.trajectory
    assert isinstance(tr, Trajectory) and tr.source == "truth" and tr.norad_id == 44714
    np.testing.assert_array_equal(tr.t_tt_s, grid)
    # starts where it was told to (dense output at t0 reproduces the initial state)
    np.testing.assert_allclose(tr.r_gcrf_km[0], init.r_gcrf_km[0], atol=1e-9)
    assert res.area_to_mass_m2_per_kg == pytest.approx(T.area_to_mass_from_bstar(bstar, cfg))


@datagen
def test_rejects_bad_grids(cfg, ctx, sw_copy):
    init = circular_state(500.0, 51.6, utc_to_tt_seconds(SAMPLE_EPOCH_UTC))
    t0 = init.t_tt_s[0]
    with pytest.raises(ValueError, match="precede"):
        T.propagate_truth(init, 1e-4, np.array([t0 - 1.0, t0 + 60.0]), cfg, sw_copy)
    with pytest.raises(ValueError, match="increasing"):
        T.propagate_truth(init, 1e-4, np.array([t0 + 60.0, t0 + 60.0]), cfg, sw_copy)
    with pytest.raises(ValueError, match="area_to_mass"):
        T.propagate_truth(init, -1e-4, np.array([t0, t0 + 60.0]), cfg, sw_copy)


_POOL_SCRIPT = """
import sys, json, numpy as np
from orbit_correction import truth_prop as T
from orbit_correction.types import Trajectory

def main():
    sw, out, recs = sys.argv[1], sys.argv[2], json.loads(open(sys.argv[3]).read())
    cfg = T.load_truth_config()
    jobs = []
    for rec in recs:
        init, bstar = T.initial_state_from_omm(rec, "2024-06-15T00:00:00")
        jobs.append(T.TruthJob(init, bstar, T.output_grid(init.t_tt_s[0], 2 * 86400.0, 60.0)))
    two = T.propagate_batch(jobs, cfg, sw, n_workers=2)
    one = T.propagate_batch(jobs[:1], cfg, sw, n_workers=1)
    assert not T.jvm_started(), "parent started a JVM"
    assert "jpype" not in sys.modules, "parent imported jpype"
    np.savez(out, **{f"two_{i}": r.trajectory.r_gcrf_km for i, r in enumerate(two)},
             one_0=one[0].trajectory.r_gcrf_km, v_one_0=one[0].trajectory.v_gcrf_km_s,
             v_two_0=two[0].trajectory.v_gcrf_km_s)

if __name__ == "__main__":
    main()
"""


@datagen
def test_determinism_pool_vs_single_worker_vs_in_process(cfg, ctx, sw_copy, gp_records, tmp_path):
    """Same object: bit-identical in a 2-worker pool (twice), a 1-worker pool and in-process.

    The pool runs from a fresh parent process that must never import jpype.
    """
    recs = [gp_records[44714], gp_records[44058], gp_records[44714]]
    (tmp_path / "recs.json").write_text(json.dumps(recs))
    (tmp_path / "pool.py").write_text(_POOL_SCRIPT)
    out = tmp_path / "out.npz"
    proc = subprocess.run(
        [
            sys.executable,
            str(tmp_path / "pool.py"),
            str(sw_copy),
            str(out),
            str(tmp_path / "recs.json"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    got = np.load(out)
    np.testing.assert_array_equal(got["two_0"], got["two_2"])
    np.testing.assert_array_equal(got["two_0"], got["one_0"])
    np.testing.assert_array_equal(got["v_two_0"], got["v_one_0"])
    assert not np.array_equal(got["two_0"], got["two_1"])
    init, bstar = T.initial_state_from_omm(gp_records[44714], SAMPLE_EPOCH_UTC)
    here = T.propagate_truth(
        init, bstar, T.output_grid(init.t_tt_s[0], 2 * DAY, 60.0), cfg, sw_copy
    )
    np.testing.assert_array_equal(here.trajectory.r_gcrf_km, got["one_0"])


# --------------------------------------------------------------------------------------
# crosscheck (brahe): only on request
# --------------------------------------------------------------------------------------


@pytest.mark.crosscheck
@pytest.mark.skipif(
    os.environ.get("ORBIT_CROSSCHECK_TESTS") != "1" or not (HAVE_OREKIT and HAVE_BRAHE),
    reason="brahe cross-check runs only with ORBIT_CROSSCHECK_TESTS=1 and the datagen and "
    "crosscheck groups",
)
def test_brahe_crosscheck_within_tolerance(tmp_path):
    """Runs scripts/crosscheck_brahe.py on the committed sample and checks its tolerance."""
    out = tmp_path / "crosscheck.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "crosscheck_brahe.py"),
            "--quick",
            "--json",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=3600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    res = json.loads(out.read_text())
    assert res["within_tolerance"], res["summary"]


def test_paths_exist():
    assert SW_FIXTURE.is_file()
    assert Path(T.CONFIG_PATH).is_file()
