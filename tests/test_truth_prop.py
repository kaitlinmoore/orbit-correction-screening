"""Tests for ``orbit_correction.truth_prop`` (the Orekit truth propagator).

Two kinds of test live here:

* **Always run** (default ``uv sync``): JVM isolation, configuration, the B* relation,
  the start rule's algebra and the space-weather range and margin guards (they act
  before any JVM is needed).
* **``datagen``** (need ``uv sync --group datagen``; skipped cleanly without it):
  closed-form physics checks, independent-implementation checks and determinism.
  They share one JVM per pytest session (the ``ctx`` fixture), because a JVM
  can't be restarted inside a process.
* **``crosscheck``** (brahe): only with ``ORBIT_CROSSCHECK_TESTS=1`` and the
  ``crosscheck`` group installed.

The configured start rule is the fitted one (``fit_to_sgp4_before_epoch``), which needs
SGP4 samples before the epoch. Tests that start truth from a single state (physics,
guards, determinism) name the epoch-state rule explicitly (``epoch_cfg``, the ``cfg``
fixture); nothing here relies on a fallback to it.

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
EPOCH = T.EPOCH_START_RULE


def epoch_cfg(**changes):
    """The configured truth model under the epoch-state rule, named explicitly.

    For tests that start truth from one state with no SGP4 fit samples; under the
    configured (fitted) rule such a call is refused.
    """
    return dataclasses.replace(T.load_truth_config(), start_rule=EPOCH, **changes)


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
    """Importing every project module (found by walking the package) starts no JVM."""
    code = textwrap.dedent(
        """
        import importlib, pkgutil, sys
        import orbit_correction
        names = [m.name for m in pkgutil.walk_packages(orbit_correction.__path__,
                                                        "orbit_correction.")]
        assert "orbit_correction.truth_prop" in names and len(names) >= 7, names
        for name in names:
            importlib.import_module(name)
        bad = [m for m in ("jpype", "orekit_jpype", "jdk4py", "_jpype") if m in sys.modules]
        assert not bad, bad
        from orbit_correction import truth_prop
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


def test_start_jvm_refuses_a_process_that_imported_torch():
    """The JVM never shares a process with PyTorch (checked before any Java import)."""
    code = textwrap.dedent(
        """
        import sys, types
        sys.modules["torch"] = types.ModuleType("torch")  # stands in for `import torch`
        from orbit_correction import truth_prop
        try:
            truth_prop.start_jvm()
        except RuntimeError as exc:
            assert "torch" in str(exc)
        else:
            raise AssertionError("start_jvm ran with torch imported")
        assert "jpype" not in sys.modules
        print("ok")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_config_loads_with_decided_values():
    """Decision 1 plus the truth settings decided on 2026-11-23 (start rule: 2026-11-25)."""
    cfg = T.load_truth_config()
    assert cfg.gravity_field_file == "eigen-6s.gfc" and cfg.zonal_order == 0
    assert cfg.zonal_degree == 36
    assert cfg.atmosphere == "NRLMSISE-00" and cfg.cd == 2.2
    assert cfg.msis_local_solar_time == "MEAN" and cfg.msis_ap_switch == -1
    assert cfg.integrator == "DOP853"
    assert cfg.position_tolerance_m == 1.0 and cfg.max_step_s == 180.0
    assert cfg.start_rule == T.FIT_START_RULE  # changed 2026-11-25 (fitted-start tests below)
    assert cfg.reentry_altitude_km == 120.0
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
        dict(start_rule="fitted"),
        dict(sgp4_mu_km3_s2=0.0),
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


def test_sgp4_mu_is_the_sgp4_packages_wgs72():
    from sgp4.earth_gravity import wgs72

    from orbit_correction import cheap_prop

    assert T.load_truth_config().sgp4_mu_km3_s2 == wgs72.mu
    assert cheap_prop.GRAVITY_MODEL_NAME == "WGS-72"


MU_TRUTH_M3 = 3.986004415e14  # EIGEN-6S GM, as Orekit reads it from the bundle


def test_start_rule_preserves_two_body_period_and_shape():
    """k = (mu_truth / mu_WGS72)^(1/3): same period, eccentricity vector and plane."""
    cfg = epoch_cfg()
    mu72 = cfg.sgp4_mu_km3_s2
    mu_t = MU_TRUTH_M3 / 1e9
    k = T.start_rule_scale(MU_TRUTH_M3, cfg)
    assert k == pytest.approx((mu_t / mu72) ** (1 / 3), rel=1e-15)
    assert -3.1e-7 < k - 1 < -2.9e-7  # the GMs differ by 9.0e-7 relative

    init = kepler_state(6900.0, 0.012, 97.4, 30.0, 40.0, 75.0, t0=DAY, mu_km=mu72)
    out = T.apply_start_rule(init, MU_TRUTH_M3, cfg)
    r0, v0 = init.r_gcrf_km, init.v_gcrf_km_s
    r1, v1 = out.r_gcrf_km, out.v_gcrf_km_s
    a0, e0, i0, raan0, w0, _ = elements(r0, v0, mu72)
    a1, e1, i1, raan1, w1, _ = elements(r1, v1, mu_t)
    period0 = 2 * np.pi * np.sqrt(a0[0] ** 3 / mu72)
    period1 = 2 * np.pi * np.sqrt(a1[0] ** 3 / mu_t)
    assert period1 == pytest.approx(period0, rel=1e-13)
    # without the rule the same state is slower under the truth GM: about 1.15 km/day
    a_direct = elements(r0, v0, mu_t)[0][0]
    n_direct = np.sqrt(mu_t / a_direct**3)
    n0 = 2 * np.pi / period0
    drift_km_per_day = (n0 - n_direct) * DAY * np.linalg.norm(r0)
    assert 1.0 < drift_km_per_day < 1.3
    np.testing.assert_allclose([e1[0], i1[0], raan1[0], w1[0]], [e0[0], i0[0], raan0[0], w0[0]],
                               rtol=0, atol=1e-12)
    # the scaled state sits (k - 1) |r|, about 2 m, from the SGP4 state
    assert 1.9e-3 < np.linalg.norm(r1 - r0) < 2.2e-3
    assert out.t_tt_s[0] == init.t_tt_s[0] and out.norad_id == init.norad_id

    direct = dataclasses.replace(cfg, start_rule="sgp4_state_direct")
    assert T.start_rule_scale(MU_TRUTH_M3, direct) == 1.0
    assert T.apply_start_rule(init, MU_TRUTH_M3, direct) is init


# --------------------------------------------------------------------------------------
# Always run: space-weather range and data-end margin guards (before any JVM)
# --------------------------------------------------------------------------------------


def _june_copy(tmp_path):
    """A copy checked for 2024-06-15 .. 2024-06-22 (the fixture is observed to 2024-12-31)."""
    t0 = utc_to_tt_seconds("2024-06-15T00:00:00")
    return t0, T.write_space_weather_copy(SW_FIXTURE, tmp_path / "sw_june.txt", t0, t0 + 7 * DAY)


def test_space_weather_copy_records_its_checked_range(tmp_path):
    t0, copy = _june_copy(tmp_path)
    assert isinstance(copy, T.SpaceWeatherCopy)
    assert (copy.window_first_utc, copy.window_last_utc) == ("2024-06-15", "2024-06-22")
    assert copy.last_fully_observed_utc == "2024-12-31" and copy.margin_days == 41.0
    assert copy.first_allowed_tt_s == t0
    assert copy.last_allowed_tt_s == pytest.approx(utc_to_tt_seconds("2024-06-23T00:00:00"))
    assert Path(copy.path).is_file() and len(copy.sha256) == 64


def test_window_outside_the_copys_range_is_refused(tmp_path):
    """A copy checked for June refuses an August window and one starting before it."""
    t0, copy = _june_copy(tmp_path)
    init = circular_state(500.0, 51.6, t0)
    cfg = epoch_cfg()
    late = circular_state(500.0, 51.6, utc_to_tt_seconds("2024-08-10T00:00:00"))
    with pytest.raises(T.SpaceWeatherRangeError, match="outside"):
        T.propagate_truth(late, 1e-4, T.output_grid(late.t_tt_s[0], DAY, 60.0), cfg, copy)
    early = circular_state(500.0, 51.6, t0 - 60.0)
    with pytest.raises(T.SpaceWeatherRangeError, match="outside"):
        T.propagate_truth(early, 1e-4, T.output_grid(t0 - 60.0, DAY, 60.0), cfg, copy)
    # running past the checked range's end is refused too
    with pytest.raises(T.SpaceWeatherRangeError):
        T.propagate_truth(init, 1e-4, T.output_grid(t0, 8 * DAY, 60.0), cfg, copy)
    # and the pool refuses the whole batch before starting a worker
    jobs = [T.TruthJob(init, 1e-4, T.output_grid(t0, DAY, 60.0)),
            T.TruthJob(late, 1e-4, T.output_grid(late.t_tt_s[0], DAY, 60.0))]
    with pytest.raises(T.SpaceWeatherRangeError, match="job 1"):
        T.propagate_batch(jobs, cfg, copy, n_workers=1)


def test_bare_path_is_refused_when_drag_is_on(tmp_path):
    t0, copy = _june_copy(tmp_path)
    init = circular_state(500.0, 51.6, t0)
    cfg = epoch_cfg()
    with pytest.raises(TypeError, match="SpaceWeatherCopy"):
        T.propagate_truth(init, 1e-4, T.output_grid(t0, DAY, 60.0), cfg, copy.path)
    with pytest.raises(TypeError, match="SpaceWeatherCopy"):
        T.propagate_batch([T.TruthJob(init, 1e-4, T.output_grid(t0, DAY, 60.0))], cfg,
                          Path(copy.path), n_workers=1)


def test_data_end_margin_enforced_where_copies_are_written_and_windows_run(tmp_path):
    """Fixture last observed day 2024-12-31: windows may end by 2024-11-21T00:00 UTC.

    The rule is the splits one (``splits.latest_window_end_tt_s``, 41 days), applied
    here independently of split labels.
    """
    limit = utc_to_tt_seconds("2024-11-21T00:00:00")
    # writing a copy for a window ending inside the margin is refused
    with pytest.raises(T.SpaceWeatherRangeError, match="41 days"):
        T.write_space_weather_copy(SW_FIXTURE, tmp_path / "a.txt", limit - 7 * DAY, limit + 60.0)
    # a copy for a window ending exactly at the limit is written, and its checked range
    # (to the end of 2024-11-21) is cut back to the limit
    copy = T.write_space_weather_copy(SW_FIXTURE, tmp_path / "b.txt", limit - 7 * DAY, limit)
    assert copy.window_last_utc == "2024-11-21"
    assert copy.last_allowed_tt_s == pytest.approx(limit)
    copy.check_window(limit - 7 * DAY, limit)
    # so a window inside the copy's days but past the margin is refused at propagation
    init = circular_state(500.0, 51.6, limit - 7 * DAY)
    grid = T.output_grid(limit - 7 * DAY, 7 * DAY + 60.0, 60.0)
    with pytest.raises(T.SpaceWeatherRangeError):
        T.propagate_truth(init, 1e-4, grid, epoch_cfg(), copy)


def test_output_grid():
    g = T.output_grid(100.0, 7 * DAY, 60.0)
    assert g.size == 10081 and g[0] == 100.0 and g[-1] == 100.0 + 7 * DAY
    assert np.all(np.diff(g) == pytest.approx(60.0))


# --------------------------------------------------------------------------------------
# always run: the initial-state wrapper (SGP4 only, no JVM)
# --------------------------------------------------------------------------------------


def test_initial_state_wrapper_uses_the_replacement_epoch():
    """``initial_state_from_omm`` with a replaced epoch, checked without Orekit.

    The same mean elements started at another date: the state's time is the new
    epoch, and its position is the element set's TEME state at its epoch (which
    doesn't depend on the date for a near-Earth set) rotated to GCRF **at the new
    date**. The reference goes through the ``sgp4`` package's TLE-free initialiser
    and the exact rotation, not through ``sgp4_epoch_states``. Rotating at the
    record's own 2026 epoch instead would move the position by kilometres
    (precession over 2.4 years), so ignoring the replacement can't pass.
    """
    from orbit_correction import catalog, cheap_prop, frames

    rec = json.loads((FIXTURES / "celestrak_gp_sample.json").read_text())
    rec = next(r for r in rec if r["NORAD_CAT_ID"] == 44714)
    new_epoch = "2024-06-15T00:00:00"
    assert rec["EPOCH"][:4] == "2026"

    init, bstar = T.initial_state_from_omm(rec, new_epoch)
    assert bstar == rec["BSTAR"]
    assert isinstance(init, Trajectory) and len(init) == 1 and init.source == "sgp4"
    assert init.norad_id == 44714
    t_new = float(utc_to_tt_seconds(new_epoch))
    assert init.t_tt_s[0] == t_new

    row = catalog.load_celestrak_omm(FIXTURES / "celestrak_gp_sample.json")
    row = next(r for r in row.to_pylist() if r["norad_id"] == 44714)
    err, r_teme, v_teme = cheap_prop.satrec_from_row(row).sgp4_tsince(0.0)
    assert err == 0
    r_new, v_new = frames.teme_to_gcrf(np.array(r_teme), np.array(v_teme), t_new)
    np.testing.assert_allclose(init.r_gcrf_km[0], r_new, rtol=0, atol=1e-9)  # 1 micrometre
    np.testing.assert_allclose(init.v_gcrf_km_s[0], v_new, rtol=0, atol=1e-12)

    # without a replacement the record's own epoch is used, and the state is elsewhere
    own, _ = T.initial_state_from_omm(rec)
    assert own.t_tt_s[0] == row["epoch_tt_s"] != t_new
    r_own, _ = frames.teme_to_gcrf(np.array(r_teme), np.array(v_teme), row["epoch_tt_s"])
    np.testing.assert_allclose(own.r_gcrf_km[0], r_own, rtol=0, atol=1e-9)
    assert np.linalg.norm(own.r_gcrf_km[0] - init.r_gcrf_km[0]) > 1.0  # km


def test_initial_state_wrapper_refuses_a_start_below_the_floor():
    """The wrapper passes the 120 km floor through: no state, a clear error.

    ISS elements with the mean motion raised to 16.6 rev/day sit at about 111 km at
    the epoch. SGP4 itself returns a state there, so only the floor stops it; truth
    must not be started from it.
    """
    rec = json.loads((FIXTURES / "celestrak_gp_sample.json").read_text())
    rec = next(r for r in rec if r["NORAD_CAT_ID"] == 25544)
    low = dict(rec, MEAN_MOTION=16.6)
    for epoch in (None, "2024-06-15T00:00:00"):
        with pytest.raises(ValueError, match="no SGP4 state at the epoch.*120 km floor"):
            T.initial_state_from_omm(low, epoch)
    # the unchanged record is fine, so the refusal is the altitude and nothing else
    init, _ = T.initial_state_from_omm(rec, "2024-06-15T00:00:00")
    assert len(init) == 1


# --------------------------------------------------------------------------------------
# Always run: the fitted start rule's configuration, samples and guards (no JVM)
# --------------------------------------------------------------------------------------

FIT = T.FIT_START_RULE


def _fit_cfg(**changes):
    return dataclasses.replace(T.load_truth_config(), start_rule=FIT, **changes)


def _gravity_elset(gp_record, epoch_utc, *, hp_km=800.0, e=0.001, ma_deg=90.0, bstar=1e-7):
    """A fixture element set with its perigee altitude, eccentricity, mean anomaly and B*
    replaced (a synthetic case, as in ``initial_state_from_omm``)."""
    import math

    rec = dict(gp_record)
    a = (6378.135 + hp_km) / (1.0 - e)  # WGS-72 radius and GM, as SGP4's mean elements
    rec.update(
        ECCENTRICITY=e,
        MEAN_ANOMALY=ma_deg,
        BSTAR=bstar,
        MEAN_MOTION=math.sqrt(398600.8 / a**3) * DAY / (2 * math.pi),
    )
    return T._omm_record_elset(rec, epoch_utc)


def _gp_record(norad):
    for r in json.loads((FIXTURES / "celestrak_gp_sample.json").read_text()):
        if int(r["NORAD_CAT_ID"]) == norad:
            return r
    raise KeyError(norad)


def test_default_is_the_fitted_rule_with_a_quarter_day_span(tmp_path):
    """Decided 2026-11-25: the fitted rule, 0.25-day span, 60 s samples."""
    cfg = T.load_truth_config()
    assert FIT == "fit_to_sgp4_before_epoch" and FIT in T.START_RULES
    assert cfg.start_rule == FIT
    assert (cfg.fit_span_s, cfg.fit_step_s, cfg.fit_max_iterations) == (0.25 * DAY, 60.0, 8)
    assert (cfg.fit_tolerance_m, cfg.fit_tolerance_m_s) == (1e-3, 1e-6)
    # the epoch-state rules stay selectable, by name
    assert T.EPOCH_START_RULE == "preserve_sgp4_mean_motion"
    for rule in (T.EPOCH_START_RULE, "sgp4_state_direct"):
        assert dataclasses.replace(cfg, start_rule=rule).start_rule == rule
    # a config file written before the fitted rule (epoch rule, no fit_* keys) still
    # loads, with its own rule
    text = (REPO / "config" / "truth.toml").read_text()
    assert 'rule = "fit_to_sgp4_before_epoch"' in text
    old = "\n".join(line for line in text.splitlines() if not line.startswith("fit_"))
    old = old.replace('rule = "fit_to_sgp4_before_epoch"', 'rule = "preserve_sgp4_mean_motion"')
    (tmp_path / "old.toml").write_text(old)
    assert T.load_truth_config(tmp_path / "old.toml") == epoch_cfg()
    for bad in (
        dict(fit_span_s=0.0),
        dict(fit_step_s=-60.0),
        dict(fit_step_s=DAY),  # fewer than three samples
        dict(fit_max_iterations=0),
        dict(fit_tolerance_m=0.0),
        dict(fit_tolerance_m_s=-1e-6),
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(cfg, **bad)


def test_no_path_reaches_the_epoch_rule_by_default_or_by_fallback(tmp_path):
    """The epoch-state rule is reached only by naming it.

    A config built without a rule gets the fitted one; a config file without a rule
    is refused; and under the configured rule a job without SGP4 fit samples is
    refused by every entry point (before any space-weather check or JVM), never
    started from the epoch state.
    """
    cfg = T.load_truth_config()
    fields = dataclasses.asdict(cfg)
    for name in ("start_rule", "fit_span_s", "fit_step_s"):
        del fields[name]
    assert T.TruthConfig(**fields) == cfg  # the dataclass defaults are the configured rule
    text = (REPO / "config" / "truth.toml").read_text()
    no_rule = "\n".join(ln for ln in text.splitlines() if not ln.startswith("rule ="))
    (tmp_path / "no_rule.toml").write_text(no_rule)
    with pytest.raises(KeyError, match="rule"):
        T.load_truth_config(tmp_path / "no_rule.toml")

    t0 = utc_to_tt_seconds("2024-06-15T00:00:00")
    init = circular_state(500.0, 51.6, t0)
    grid = T.output_grid(t0, DAY, 60.0)
    # drag on and no space-weather copy: reaching the copy check would raise TypeError,
    # so the ValueError shows the samples are checked first
    with pytest.raises(ValueError, match="needs fit_samples"):
        T.propagate_truth(init, 1e-4, grid, cfg, None)
    with pytest.raises(ValueError, match="job 0 .*needs fit_samples"):
        T.propagate_batch([T.TruthJob(init, 1e-4, grid)], cfg, None, 1)
    with pytest.raises(ValueError, match="fit_start_to_sgp4"):
        T.apply_start_rule(init, MU_TRUTH_M3, cfg)


def test_fit_sample_offsets_end_exactly_at_the_epoch():
    off = T.fit_sample_offsets_s(_fit_cfg())
    assert off.size == 361 and off[0] == -0.25 * DAY and off[-1] == 0.0
    assert np.all(off <= 0.0) and np.all(np.diff(off) == 60.0)
    off = T.fit_sample_offsets_s(_fit_cfg(fit_span_s=1000.0, fit_step_s=300.0))
    np.testing.assert_array_equal(off, [-900.0, -600.0, -300.0, 0.0])


def test_fit_samples_ask_sgp4_for_nothing_after_the_epoch(monkeypatch):
    """Every time handed to SGP4 for the fit samples is at or before the epoch."""
    from orbit_correction import cheap_prop

    asked = []
    real = cheap_prop._sgp4_teme

    def spy(elsets, grid_s, from_epoch, chunk_objects):
        asked.append((np.array(grid_s), from_epoch))
        return real(elsets, grid_s, from_epoch, chunk_objects)

    monkeypatch.setattr(cheap_prop, "_sgp4_teme", spy)
    el = _gravity_elset(_gp_record(25544), "2024-06-15T00:00:00")
    res = T.fit_samples_from_elsets(el, _fit_cfg())[0]
    epoch = el.column("epoch_tt_s")[0].as_py()
    assert res.status.ok and len(res.trajectory) == 361
    assert len(asked) == 1 and asked[0][1] is True and asked[0][0].max() == 0.0
    assert res.trajectory.t_tt_s.max() == epoch
    assert res.trajectory.t_tt_s.min() == epoch - 0.25 * DAY


def test_fit_samples_after_the_epoch_are_refused():
    """Refused before any JVM is needed: late samples, missing samples, samples under
    another rule, too few, another object's, and the start rule applied without samples."""
    t0 = utc_to_tt_seconds("2024-06-15T00:00:00")
    init = circular_state(800.0, 51.6, t0, nid=7)
    grid = T.output_grid(t0, DAY, 60.0)
    no_drag = _fit_cfg(atmosphere="none")

    def samples(times, nid=7):
        n = len(times)
        return Trajectory(nid, np.asarray(times, float), np.tile(init.r_gcrf_km, (n, 1)),
                          np.tile(init.v_gcrf_km_s, (n, 1)), "sgp4")

    ok = samples(t0 + np.array([-120.0, -60.0, 0.0]))
    late = samples(t0 + np.array([-120.0, -60.0, 0.0, 1e-3]))
    with pytest.raises(ValueError, match="after the epoch"):
        T.propagate_truth(init, None, grid, no_drag, None, fit_samples=late)
    with pytest.raises(ValueError, match="needs fit_samples"):
        T.propagate_truth(init, None, grid, no_drag, None)
    with pytest.raises(ValueError, match="at least 3"):
        T.propagate_truth(init, None, grid, no_drag, None, fit_samples=samples([t0 - 60.0, t0]))
    with pytest.raises(ValueError, match="NORAD"):
        T.propagate_truth(init, None, grid, no_drag, None,
                          fit_samples=samples(t0 + np.array([-120.0, -60.0, 0.0]), nid=8))
    with pytest.raises(ValueError, match="would be ignored"):
        T.propagate_truth(init, None, grid, dataclasses.replace(no_drag, start_rule=
                          "preserve_sgp4_mean_motion"), None, fit_samples=ok)
    with pytest.raises(ValueError, match="fit_start_to_sgp4"):
        T.apply_start_rule(init, MU_TRUTH_M3, no_drag)
    # the pool refuses the batch before starting a worker
    with pytest.raises(ValueError, match="job 1 .*after the epoch"):
        T.propagate_batch([T.TruthJob(init, None, grid, fit_samples=ok),
                           T.TruthJob(init, None, grid, fit_samples=late)], no_drag, None, 1)
    with pytest.raises(ValueError, match="job 0 .*needs fit_samples"):
        T.propagate_batch([T.TruthJob(init, None, grid)], no_drag, None, 1)


def test_space_weather_guard_covers_the_fit_span(tmp_path):
    """A copy checked from the epoch's day refuses a fitted window whose span starts the
    day before; a copy checked from the first sample's day accepts it."""
    t0, june = _june_copy(tmp_path)  # checked from 2024-06-15T00:00
    init = circular_state(500.0, 51.6, t0 + 3600.0)
    grid = T.output_grid(t0 + 3600.0, DAY, 60.0)
    off = T.fit_sample_offsets_s(_fit_cfg())  # 0.25 d: the first sample is 5 h before t0
    fit_s = Trajectory(1, t0 + 3600.0 + off, np.tile(init.r_gcrf_km, (off.size, 1)),
                       np.tile(init.v_gcrf_km_s, (off.size, 1)), "sgp4")
    cfg = _fit_cfg()
    with pytest.raises(T.SpaceWeatherRangeError, match="outside"):
        T.propagate_truth(init, 1e-4, grid, cfg, june, fit_samples=fit_s)
    with pytest.raises(T.SpaceWeatherRangeError, match="job 0"):
        T.propagate_batch([T.TruthJob(init, 1e-4, grid, fit_samples=fit_s)], cfg, june, 1)
    wide = T.write_space_weather_copy(SW_FIXTURE, tmp_path / "sw_wide.txt",
                                      float(fit_s.t_tt_s[0]), float(grid[-1]))
    wide.check_window(float(fit_s.t_tt_s[0]), float(grid[-1]))


# --------------------------------------------------------------------------------------
# datagen: shared JVM and inputs
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cfg():
    """The configured truth model under the epoch-state rule (named, not defaulted).

    The physics, determinism and guard tests start from one state; the fitted rule's
    tests replace ``start_rule`` and pass SGP4 samples.
    """
    return epoch_cfg()


@pytest.fixture(scope="session")
def ctx(cfg):
    """The session's Orekit context: one JVM per pytest session."""
    if not HAVE_OREKIT:
        pytest.skip("needs the datagen group")
    return T._context(cfg)


@pytest.fixture(scope="session")
def sw_copy(tmp_path_factory):
    """Orekit copy of the fixture space-weather file, for SW_FIRST..SW_LAST."""
    path = tmp_path_factory.mktemp("sw") / "sw_orekit_2024-02-01_2024-11-15.txt"
    copy = T.write_space_weather_copy(
        SW_FIXTURE, path, utc_to_tt_seconds(SW_FIRST_UTC), utc_to_tt_seconds(SW_LAST_UTC)
    )
    assert isinstance(copy, T.SpaceWeatherCopy)
    return copy


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
    for key, tol, step in (("truth", cfg.position_tolerance_m, cfg.max_step_s),
                           ("converged", 1e-3, 30.0)):
        c = dataclasses.replace(
            cfg, zonal_degree=0, atmosphere="none", position_tolerance_m=tol, max_step_s=step
        )
        res = T.propagate_truth(init, None, T.output_grid(DAY, 7 * DAY, 60.0), c, None)
        assert res.status == "ok"
        r, v = _sample(ctx, res)
        energy = 0.5 * np.sum(v**2, axis=1) - ctx.mu / np.linalg.norm(r, axis=1)
        h = np.linalg.norm(np.cross(r, v), axis=1)
        drift[key] = (np.max(np.abs(energy / energy[0] - 1)), np.max(np.abs(h / h[0] - 1)))
    # Truth setting (1 m, 180 s): 3.5e-10 relative energy drift and 1.4e-10 in angular
    # momentum over 7 days (measured 2026-11-23). The 180 s cap sets it: 1 mm at 180 s gives
    # the same, 1 m at 300 s gives 3.2e-8 and 1 m with no cap 2e-5.
    assert drift["truth"][0] < 1e-9 and drift["truth"][1] < 1e-9, drift
    # The drift is integration error, not a force: a converged run (1 mm, 30 s) has
    # 5.6e-14 and 2.8e-14.
    assert drift["converged"][0] < drift["truth"][0] / 100, drift
    assert drift["converged"][1] < drift["truth"][1] / 100, drift


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
    # First run: -1497.098 m against -1497.089 m analytic (6e-6 relative). The
    # orbit's eccentricity under drag oscillates slightly, which the circular-orbit
    # formula ignores; tolerance 1e-4.
    assert got == pytest.approx(want, rel=1e-4)


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
    # constructor default (apparent solar time) these six points differ by up to 1.2%;
    # along real orbits the difference reaches 6.8% (800 km, November 2024).
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
    """Library check: Orekit's TLEPropagator and the sgp4 package agree (TEME).

    The project never uses Orekit's TLEPropagator (SGP4 comes from cheap_prop.py);
    this confirms the two SGP4 implementations available here agree, so a later
    switch between them would not change the cheap side of the residual.
    """
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


#: Bounds on |truth setting - converged run| (m) per test case at 1 and 7 days, at the
#: configured 180 s maximum step. Measured 2026-11-23: Starlink-1008 (340 km, drag)
#: 0.124 m at 1 d and 1.57 m at 7 d; the circular 800 km orbit 0.037 m and 0.083 m. The
#: bounds are about 2.5x those. At the earlier 300 s step the same cases gave 16.6 m and
#: 16.1 m at 7 days, so a return to it fails here.
INTEGRATION_BOUND_7D_M = {"starlink_1008_340km": 4.0, "circular_800km": 0.25}
INTEGRATION_BOUND_1D_M = {"starlink_1008_340km": 0.3, "circular_800km": 0.1}


@datagen
@pytest.mark.parametrize("case", ["starlink_1008_340km", "circular_800km"])
def test_integration_error_of_the_truth_setting(cfg, ctx, sw_copy, gp_records, case):
    """Truth setting (1 m, 180 s) against a converged run (1 mm, 30 s), same force model.

    At the 180 s cap DOP853 runs at the step limit, so this is the error the step
    sets, which is part of the truth definition. Measured values are in the module
    constants' comment; per-band values for the sample objects are in the changelog.
    """
    if case == "starlink_1008_340km":
        init, bstar = T.initial_state_from_omm(gp_records[44714], SAMPLE_EPOCH_UTC)
        a2m = None
    else:
        init, bstar = circular_state(800.0, 98.6, utc_to_tt_seconds(SAMPLE_EPOCH_UTC)), None
        a2m = 0.005
    assert cfg.max_step_s == 180.0 and cfg.position_tolerance_m == 1.0
    grid = T.output_grid(init.t_tt_s[0], 7 * DAY, 3600.0)
    a = T.propagate_truth(init, bstar, grid, cfg, sw_copy, area_to_mass_m2_per_kg=a2m)
    conv = dataclasses.replace(cfg, position_tolerance_m=1e-3, max_step_s=30.0)
    b = T.propagate_truth(init, bstar, grid, conv, sw_copy, area_to_mass_m2_per_kg=a2m)
    d = np.linalg.norm(a.trajectory.r_gcrf_km - b.trajectory.r_gcrf_km, axis=1) * 1000.0
    assert d[-1] < INTEGRATION_BOUND_7D_M[case], d[-1]
    assert d[24] < INTEGRATION_BOUND_1D_M[case], d[24]  # 1 day


@datagen
def test_configured_gravity_reaches_the_force_model(cfg, ctx, sw_copy):
    """The built propagator carries EIGEN-6S to the configured degree, order 0, Earth-fixed.

    Guards against a silent fallback (for example J2 only, an 8-26 km change at 7 days)
    that every physics test would still pass. Checked on the force model itself and by
    its effect: degree 36 and degree 2 differ by kilometres after one day.
    """
    init = circular_state(500.0, 51.6, utc_to_tt_seconds(SAMPLE_EPOCH_UTC))
    prop, _, _ = T.build_propagator(cfg, init, 0.005, sw_copy)
    grav = T.describe_gravity(prop)
    assert len(grav) == 1, grav
    assert grav[0]["degree"] == cfg.zonal_degree == 36 and grav[0]["order"] == 0
    assert grav[0]["body_frame"] == ctx.itrf.getName() and "ITRF" in grav[0]["body_frame"]
    assert grav[0]["mu"] == ctx.mu
    no_drag = dataclasses.replace(cfg, atmosphere="none")
    grid = T.output_grid(init.t_tt_s[0], DAY, 3600.0)
    full = T.propagate_truth(init, None, grid, no_drag, None).trajectory
    j2 = T.propagate_truth(init, None, grid, dataclasses.replace(no_drag, zonal_degree=2), None)
    d_km = np.linalg.norm(full.r_gcrf_km[-1] - j2.trajectory.r_gcrf_km[-1])
    assert d_km > 0.5, d_km  # measured 1.29 km


@datagen
def test_rescaled_start_keeps_the_sgp4_period_in_a_two_body_run(cfg, ctx):
    """Two-body (point mass) under the truth GM: after one WGS-72 period the scaled state is
    back where it started; the unscaled state is ~78 m behind."""
    mu72 = cfg.sgp4_mu_km3_s2
    init = kepler_state(6900.0, 0.012, 97.4, 30.0, 40.0, 75.0, t0=DAY, mu_km=mu72)
    a72 = elements(init.r_gcrf_km, init.v_gcrf_km_s, mu72)[0][0]
    period72 = 2 * np.pi * np.sqrt(a72**3 / mu72)
    two_body = dataclasses.replace(
        cfg, zonal_degree=0, atmosphere="none", position_tolerance_m=1e-4, max_step_s=30.0
    )
    grid = np.array([DAY, DAY + period72])
    gap = {}
    for rule in ("preserve_sgp4_mean_motion", "sgp4_state_direct"):
        c = dataclasses.replace(two_body, start_rule=rule)
        tr = T.propagate_truth(init, None, grid, c, None).trajectory
        gap[rule] = np.linalg.norm(tr.r_gcrf_km[1] - tr.r_gcrf_km[0]) * 1000.0
    # the scaled state's period under mu_truth equals the original's under mu_WGS72
    assert gap["preserve_sgp4_mean_motion"] < 0.01, gap
    # without the rule the same r, v has mean motion lower by 2 dmu/mu = 1.8e-6 under the
    # truth GM (a grows by dmu/mu as well), so it is 2 pi * 1.8e-6 * |r| ~ 78 m behind
    # after one revolution (measured 78.6 m), ~1.15 km per day
    assert 70.0 < gap["sgp4_state_direct"] < 90.0, gap


@datagen
def test_reentry_returns_status_and_the_trajectory_before_it(cfg, ctx, sw_copy):
    init = circular_state(200.0, 51.6, utc_to_tt_seconds("2024-05-10T00:00:00"))
    grid = T.output_grid(init.t_tt_s[0], 7 * DAY, 60.0)
    res = T.propagate_truth(init, None, grid, cfg, sw_copy, area_to_mass_m2_per_kg=0.05)
    assert res.status == "reentry" and "120.0 km" in res.message
    assert init.t_tt_s[0] < res.stop_t_tt_s < grid[-1]
    tr = res.trajectory
    assert isinstance(tr, Trajectory) and tr.source == "truth"
    # every grid point up to the stop time, and none after it
    np.testing.assert_array_equal(tr.t_tt_s, grid[grid <= res.stop_t_tt_s])
    assert len(tr) > 1
    # the last kept point is still above 120 km, within one 60 s step of the crossing
    last = Trajectory(1, tr.t_tt_s[-1:], tr.r_gcrf_km[-1:], tr.v_gcrf_km_s[-1:], "truth")
    assert 120.0 <= T.initial_geodetic_altitude_km(cfg, last) < 130.0
    assert res.stop_t_tt_s - tr.t_tt_s[-1] < 60.0


@datagen
@pytest.mark.parametrize("alt_km", [100.0, 110.0, 119.0])
def test_start_below_reentry_altitude_returns_reentry_at_once(cfg, ctx, sw_copy, alt_km):
    """A start below 120 km is never integrated (it used to run on or raise)."""
    init = circular_state(alt_km, 0.0, utc_to_tt_seconds("2024-05-10T00:00:00"))
    grid = T.output_grid(init.t_tt_s[0], DAY, 60.0)
    res = T.propagate_truth(init, None, grid, cfg, sw_copy, area_to_mass_m2_per_kg=0.01)
    assert res.status == "reentry" and res.trajectory is None
    assert res.stop_t_tt_s == init.t_tt_s[0] and res.n_evaluations == 0
    assert "initial geodetic altitude" in res.message


@datagen
def test_truth_output_satisfies_contract(cfg, ctx, sw_copy, gp_records):
    init, bstar = T.initial_state_from_omm(gp_records[44714], SAMPLE_EPOCH_UTC)
    grid = T.output_grid(init.t_tt_s[0], 1 * DAY, 60.0)
    res = T.propagate_truth(init, bstar, grid, cfg, sw_copy)
    tr = res.trajectory
    assert isinstance(tr, Trajectory) and tr.source == "truth" and tr.norad_id == 44714
    np.testing.assert_array_equal(tr.t_tt_s, grid)
    # starts at the start-rule state (dense output at t0 reproduces it), about 2 m from
    # the SGP4 state: the non-zero residual at t0
    start = T.apply_start_rule(init, ctx.mu, cfg)
    np.testing.assert_allclose(tr.r_gcrf_km[0], start.r_gcrf_km[0], atol=1e-9)
    assert res.start_scale == T.start_rule_scale(ctx.mu, cfg)
    assert 1.5e-3 < np.linalg.norm(tr.r_gcrf_km[0] - init.r_gcrf_km[0]) < 2.5e-3
    assert res.area_to_mass_m2_per_kg == pytest.approx(T.area_to_mass_from_bstar(bstar, cfg))


@datagen
def test_copy_rewritten_at_the_same_path_is_read_again(cfg, ctx, tmp_path):
    """Orekit objects are cached by content hash, not path alone."""
    from orbit_correction.spaceweather import parse_spaceweather

    sw = parse_spaceweather(SW_FIXTURE)
    t0 = utc_to_tt_seconds("2024-06-15T00:00:00")
    path = tmp_path / "same.txt"
    a = T.write_space_weather_copy(sw, path, t0, t0 + DAY)
    date = ctx.date(utc_to_tt_seconds("2024-06-15T12:00:00"))
    first = ctx.space_weather(a).getDailyFlux(date)
    i = int(np.flatnonzero(sw.dates == np.datetime64("2024-06-14"))[0])
    f107 = sw.scalars["f107_obs"].copy()
    f107[i] = 999.9
    b = T.write_space_weather_copy(
        dataclasses.replace(sw, scalars={**sw.scalars, "f107_obs": f107}), path, t0, t0 + DAY
    )
    assert a.sha256 != b.sha256 and first != 999.9
    assert ctx.space_weather(b).getDailyFlux(date) == 999.9


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
import dataclasses, pickle, sys, json, numpy as np
from orbit_correction import truth_prop as T

def main():
    sw = pickle.loads(open(sys.argv[1], "rb").read())
    out, recs = sys.argv[2], json.loads(open(sys.argv[3]).read())
    cfg = dataclasses.replace(T.load_truth_config(), start_rule=T.EPOCH_START_RULE)
    jobs = []
    for rec, epoch in recs:
        init, bstar = T.initial_state_from_omm(rec, epoch)
        jobs.append(T.TruthJob(init, bstar, T.output_grid(init.t_tt_s[0], 2 * 86400.0, 60.0)))
    # 2 workers: the last job runs in a JVM that has already propagated other epochs
    two = T.propagate_batch(jobs, cfg, sw, n_workers=2)
    one = T.propagate_batch(jobs[-1:], cfg, sw, n_workers=1)  # a fresh worker
    assert all(r.status == "ok" for r in two + one), [r.message for r in two + one]
    assert not T.jvm_started(), "parent started a JVM"
    assert "jpype" not in sys.modules, "parent imported jpype"
    np.savez(out, **{f"two_{i}": r.trajectory.r_gcrf_km for i, r in enumerate(two)},
             one=one[0].trajectory.r_gcrf_km, v_one=one[0].trajectory.v_gcrf_km_s,
             v_two_last=two[-1].trajectory.v_gcrf_km_s)

if __name__ == "__main__":
    main()
"""


@datagen
def test_reproducible_to_a_millimetre_across_workers_and_job_order(
    cfg, ctx, sw_copy, gp_records, tmp_path
):
    """Same job: the same trajectory to about a millimetre, whatever ran before it.

    Not bit-identical: a job run after another at a nearby, different epoch can differ
    from a fresh-worker run by ~1e-4 m over 2 days (Orekit's interpolated frame
    transforms take their sample grid from the first date a JVM requests). The jobs
    here run at different epochs, so the last one follows nearby epochs in its worker.
    The pool runs from a fresh parent process that must never import jpype.
    """
    import pickle

    recs = [
        (gp_records[44714], "2024-06-15T00:00:00"),
        (gp_records[44058], "2024-06-16T05:00:00"),
        (gp_records[44714], "2024-06-14T09:30:00"),
        (gp_records[44058], "2024-06-15T21:00:00"),
        (gp_records[44714], SAMPLE_EPOCH_UTC),
    ]
    (tmp_path / "recs.json").write_text(json.dumps(recs))
    (tmp_path / "sw.pkl").write_bytes(pickle.dumps(sw_copy))
    (tmp_path / "pool.py").write_text(_POOL_SCRIPT)
    out = tmp_path / "out.npz"
    proc = subprocess.run(
        [
            sys.executable,
            str(tmp_path / "pool.py"),
            str(tmp_path / "sw.pkl"),
            str(out),
            str(tmp_path / "recs.json"),
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    got = np.load(out)
    tol_km = 1e-6  # 1 mm
    np.testing.assert_allclose(got["two_4"], got["two_0"], rtol=0, atol=tol_km)
    np.testing.assert_allclose(got["two_4"], got["one"], rtol=0, atol=tol_km)
    np.testing.assert_allclose(got["v_two_last"], got["v_one"], rtol=0, atol=1e-9)
    init, bstar = T.initial_state_from_omm(gp_records[44714], SAMPLE_EPOCH_UTC)
    here = T.propagate_truth(
        init, bstar, T.output_grid(init.t_tt_s[0], 2 * DAY, 60.0), cfg, sw_copy
    )
    np.testing.assert_allclose(here.trajectory.r_gcrf_km, got["one"], rtol=0, atol=tol_km)
    # different jobs really are different (the comparison above isn't vacuous)
    assert np.max(np.abs(got["two_0"] - got["two_1"])) > 1.0


# --------------------------------------------------------------------------------------
# datagen: the fitted start rule
# --------------------------------------------------------------------------------------


def _fit_inputs(el):
    from orbit_correction.cheap_prop import sgp4_epoch_states

    row = el.to_pylist()[0]
    init = sgp4_epoch_states(el)[0].trajectory
    return row, init


@datagen
def test_fitted_start_reads_no_sgp4_state_after_the_epoch(ctx, sw_copy, monkeypatch):
    """The whole fitted path, SGP4 samples to truth trajectory, under a spy on SGP4: every
    instant SGP4 is evaluated at is at or before the epoch, and the fitted start is the
    same when every SGP4 state after the epoch is replaced by garbage."""
    from orbit_correction import cheap_prop

    el = _gravity_elset(_gp_record(25544), SAMPLE_EPOCH_UTC, e=0.02, ma_deg=90.0)
    row, init = _fit_inputs(el)
    epoch = row["epoch_tt_s"]
    asked = []
    real = cheap_prop._sgp4_teme

    def poisoned(elsets, grid_s, from_epoch, chunk_objects):
        batch = real(elsets, grid_s, from_epoch, chunk_objects)
        asked.append(batch.t_tt_s.copy())
        after = batch.t_tt_s > batch.epoch_tt_s[:, None]
        batch.r_teme_km[after] = 1.0e6  # garbage anything after the epoch would read
        return batch

    monkeypatch.setattr(cheap_prop, "_sgp4_teme", poisoned)
    cfg = _fit_cfg()
    samples = T.fit_samples_from_elsets(el, cfg)[0].trajectory
    grid = T.output_grid(epoch, DAY, 600.0)
    res = T.propagate_truth(init, row["bstar_per_earth_radii"], grid, cfg, sw_copy,
                            fit_samples=samples)
    assert asked and max(float(a.max()) for a in asked) <= epoch
    assert res.status == "ok" and res.start_fit.converged
    assert res.start_fit.last_sample_tt_s <= epoch
    assert res.start_scale is None and res.trajectory.t_tt_s[0] == epoch
    # nothing above read the poisoned states, so the start is the unpoisoned fit's
    monkeypatch.setattr(cheap_prop, "_sgp4_teme", real)
    clean = T.fit_samples_from_elsets(el, cfg)[0].trajectory
    again, _ = T.fit_start_to_sgp4(init, clean, row["bstar_per_earth_radii"], cfg, sw_copy)
    np.testing.assert_allclose(res.trajectory.r_gcrf_km[0], again.r_gcrf_km[0], rtol=0,
                               atol=1e-9)


@datagen
def test_fit_recovers_a_known_state(cfg, ctx):
    """Synthetic ground truth: samples made by the truth model itself from a known state.
    Started 200 m and 0.2 m/s away, the fit returns that state to a millimetre."""
    no_drag = dataclasses.replace(cfg, start_rule=FIT, atmosphere="none")
    t0 = utc_to_tt_seconds(SAMPLE_EPOCH_UTC)
    true = kepler_state(7178.0, 0.03, 63.0, 40.0, 25.0, 140.0, t0, nid=5)
    prop, _, ctx2 = T.build_propagator(no_drag, true, None, None)
    gen = prop.getEphemerisGenerator()
    times = t0 + T.fit_sample_offsets_s(no_drag)
    prop.propagate(ctx2.date(t0), ctx2.date(float(times[0])))
    eph = gen.getGeneratedEphemeris()
    r = np.array([[p.getX(), p.getY(), p.getZ()] for p in
                  (eph.getPVCoordinates(ctx2.date(float(t)), ctx2.gcrf).getPosition()
                   for t in times)]) / 1000.0
    samples = Trajectory(5, times, r, np.zeros_like(r), "sgp4")
    guess = Trajectory(5, true.t_tt_s.copy(), true.r_gcrf_km + [0.2, -0.1, 0.1],
                       true.v_gcrf_km_s + [1e-4, 2e-4, -1e-4], "sgp4")
    fitted, rec = T.fit_start_to_sgp4(guess, samples, None, no_drag, None)
    assert rec.converged and rec.iterations <= 4, rec
    assert rec.rms_initial_m > 1e3 and rec.rms_m < 1e-3, rec
    dr = np.linalg.norm(fitted.r_gcrf_km[0] - true.r_gcrf_km[0]) * 1e3
    dv = np.linalg.norm(fitted.v_gcrf_km_s[0] - true.v_gcrf_km_s[0]) * 1e3
    assert dr < 1e-3 and dv < 1e-6, (dr, dv)


@datagen
def test_a_fit_that_does_not_converge_is_a_status(ctx, sw_copy):
    """One pass is not enough: status fit_failed with the fit's record, no trajectory,
    no fallback start."""
    el = _gravity_elset(_gp_record(25544), SAMPLE_EPOCH_UTC)
    row, init = _fit_inputs(el)
    cfg = _fit_cfg(fit_max_iterations=1)
    samples = T.fit_samples_from_elsets(el, cfg)[0].trajectory
    res = T.propagate_truth(init, row["bstar_per_earth_radii"],
                            T.output_grid(row["epoch_tt_s"], DAY, 600.0), cfg, sw_copy,
                            fit_samples=samples)
    assert res.status == "fit_failed" and res.trajectory is None
    f = res.start_fit
    assert f.status == "not_converged" and f.iterations == 1 and not f.converged
    assert f.n_samples == 361 and f.n_evaluations > 0 and "not_converged" in res.message
    assert f.status in T.FIT_STATUSES


@datagen
def test_fitted_start_removes_the_phase_drift(ctx, sw_copy):
    """800 km, e = 0.02, almost no drag, mean anomaly 90 and 270 degrees at the epoch.
    Started at the SGP4 state the truth orbit drifts against SGP4 by about +74 and
    -72 km in 7 days (opposite signs: the artifact depends on phase). The fitted start
    leaves under 2 km."""
    from orbit_correction.cheap_prop import propagate_sgp4
    from orbit_correction.residuals import rtn_from_states

    t7 = {}
    for ma in (90.0, 270.0):
        el = _gravity_elset(_gp_record(25544), SAMPLE_EPOCH_UTC, e=0.02, ma_deg=ma)
        row, init = _fit_inputs(el)
        grid = T.output_grid(row["epoch_tt_s"], 7 * DAY, 3600.0)
        sg = propagate_sgp4(el, grid)[0].trajectory
        samples = T.fit_samples_from_elsets(el, _fit_cfg())[0].trajectory
        for rule in (EPOCH, FIT):
            cfg = dataclasses.replace(T.load_truth_config(), start_rule=rule)
            kw = {"fit_samples": samples} if rule == FIT else {}
            tr = T.propagate_truth(init, row["bstar_per_earth_radii"], grid, cfg, sw_copy, **kw)
            rtn = rtn_from_states(sg.r_gcrf_km, sg.v_gcrf_km_s, tr.trajectory.r_gcrf_km)
            t7[(ma, rule)] = rtn[-1, 1]
    assert t7[(90.0, EPOCH)] > 50.0, t7
    assert t7[(270.0, EPOCH)] < -50.0, t7
    assert abs(t7[(90.0, FIT)]) < 2.0 and abs(t7[(270.0, FIT)]) < 2.0, t7


_FIT_POOL_SCRIPT = """
import dataclasses, pickle, sys, json
from orbit_correction import truth_prop as T
from orbit_correction.cheap_prop import sgp4_epoch_states

def main():
    sw = pickle.loads(open(sys.argv[1], "rb").read())
    rec, epoch = json.loads(open(sys.argv[2]).read())
    cfg = dataclasses.replace(T.load_truth_config(), start_rule=T.FIT_START_RULE)
    el = T._omm_record_elset(rec, epoch)
    init = sgp4_epoch_states(el)[0].trajectory
    samples = T.fit_samples_from_elsets(el, cfg)[0]
    assert samples.status.ok
    bstar = el.column("bstar_per_earth_radii")[0].as_py()
    job = T.TruthJob(init, bstar, T.output_grid(init.t_tt_s[0], 86400.0, 600.0),
                     fit_samples=samples.trajectory)
    res = T.propagate_batch([job], cfg, sw, n_workers=1)[0]
    assert res.status == "ok", res.message
    assert res.start_fit is not None and res.start_fit.converged, res.start_fit
    assert not T.jvm_started() and "jpype" not in sys.modules
    print("OK", res.start_fit.iterations, res.start_fit.rms_m)

if __name__ == "__main__":
    main()
"""


@datagen
def test_fitted_start_runs_through_the_worker_pool(sw_copy, gp_records, tmp_path):
    """The generation hook: SGP4 fit samples made in a JVM-free parent, carried by
    TruthJob into a spawned worker; the result carries the fit's record back."""
    import pickle

    (tmp_path / "rec.json").write_text(json.dumps([gp_records[44714], SAMPLE_EPOCH_UTC]))
    (tmp_path / "sw.pkl").write_bytes(pickle.dumps(sw_copy))
    (tmp_path / "pool.py").write_text(_FIT_POOL_SCRIPT)
    proc = subprocess.run(
        [sys.executable, str(tmp_path / "pool.py"), str(tmp_path / "sw.pkl"),
         str(tmp_path / "rec.json")],
        capture_output=True, text=True, cwd=REPO, timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.startswith("OK")


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
