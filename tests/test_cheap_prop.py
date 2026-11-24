"""Tests for ``orbit_correction.cheap_prop``: the SGP4 wrapper.

References:

* Vallado, Crawford, Hujsak and Kelso, "Revisiting Spacetrack Report #3",
  AIAA 2006-6753: the verification element sets (``SGP4-VER.TLE``) and the C++
  reference output (``tcppver.out``), both shipped inside the ``sgp4`` package and
  used by its own test suite. Tolerance as in that suite: 2e-7 km and 2e-7 km/s on
  each component, and the same SGP4 error codes for the seven cases that fail.
* The leap second of 2016-12-31T23:59:60 UTC (TAI - UTC went from 36 s to 37 s).
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
from importlib.resources import files

import numpy as np
import pyarrow as pa
import pytest
from astropy.time import Time
from sgp4.api import WGS72, Satrec, jday

from orbit_correction import catalog
from orbit_correction import cheap_prop as C
from orbit_correction import frames as F
from orbit_correction.types import Trajectory

RE_WGS72_KM = 6378.135
WEEK_60S = 60.0 * np.arange(7 * 1440 + 1)


# ----------------------------------------------------------------- helpers


@pytest.fixture(scope="module")
def sample(request) -> pa.Table:
    """The CelesTrak fixture catalog (17 element sets, LEO to GEO)."""
    path = request.config.rootpath / "tests" / "fixtures" / "celestrak_gp_sample.json"
    return catalog.load_celestrak_omm(path)


def _with(table: pa.Table, row: int, **changes) -> pa.Table:
    """One-row table copied from ``table[row]`` with some fields changed.

    A changed ``epoch_utc`` also updates ``epoch_tt_s``, as the catalog loader does.
    """
    rec = table.slice(row, 1).to_pylist()[0]
    rec.update(changes)
    if "epoch_utc" in changes:
        rec["epoch_tt_s"] = float(F.elset_epoch_to_tt_seconds(rec["epoch_utc"]))
    return pa.Table.from_pylist([rec], schema=table.schema)


def _row_index(table: pa.Table, norad: int) -> int:
    return table.column("norad_id").to_pylist().index(norad)


# ----------------------------------------------------------------- Vallado verification


def _tle_float(field: str) -> float:
    """TLE implied-decimal exponent field, e.g. ' 28098-4' -> 0.28098e-4."""
    f = field.strip()
    if not f:
        return 0.0
    sign = -1.0 if f[0] == "-" else 1.0
    f = f.lstrip("+-")
    return sign * float("0." + f[:-2]) * 10.0 ** int(f[-2:])


def _tle_to_omm(line1: str, line2: str) -> dict:
    """OMM keyword record from a TLE, read from the text (not via the library)."""
    yy = int(line1[18:20])
    year = 2000 + yy if yy < 57 else 1900 + yy
    doy = float(line1[20:32])
    epoch = dt.datetime(year, 1, 1) + dt.timedelta(days=doy - 1.0)
    return {
        "OBJECT_NAME": f"VER {line1[2:7].strip()}",
        "OBJECT_ID": "",
        "EPOCH": epoch.strftime("%Y-%m-%dT%H:%M:%S.%f"),
        "MEAN_MOTION": float(line2[52:63]),
        "ECCENTRICITY": float("0." + line2[26:33].strip()),
        "INCLINATION": float(line2[8:16]),
        "RA_OF_ASC_NODE": float(line2[17:25]),
        "ARG_OF_PERICENTER": float(line2[34:42]),
        "MEAN_ANOMALY": float(line2[43:51]),
        "EPHEMERIS_TYPE": 0,
        "CLASSIFICATION_TYPE": line1[7],
        "NORAD_CAT_ID": int(line1[2:7]),
        "ELEMENT_SET_NO": int(line1[64:68]),
        "REV_AT_EPOCH": int(line2[63:68]),
        "BSTAR": _tle_float(line1[53:61]),
        "MEAN_MOTION_DOT": float(line1[33:43]),
        "MEAN_MOTION_DDOT": _tle_float(line1[44:52]),
    }


def _verification_cases():
    """(line1, line2, minutes grid) per case, and (NORAD ID, reference states) per case."""
    tle = files("sgp4").joinpath("SGP4-VER.TLE").read_text().splitlines()
    cases = []
    for i, line1 in enumerate(tle):
        if not line1.startswith("1 "):
            continue
        line2 = tle[i + 1]
        tstart, tend, tstep = (float(x) for x in line2[69:].split())
        # The same grid the sgp4 package's generator walks (after the state at 0).
        grid = []
        t = tstart
        while t <= tend:
            if not (t == tstart == 0.0):
                grid.append(t)
            t += tstep
        if t - tend < tstep - 1e-6:
            grid.append(tend)
        cases.append((line1, line2[:69], np.array(grid)))

    # Reference blocks in file order (one NORAD ID, 20413, appears twice).
    ref: list[tuple[int, list[np.ndarray]]] = []
    for line in files("sgp4").joinpath("tcppver.out").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == "xx":
            ref.append((int(parts[0]), []))
        elif parts:
            ref[-1][1].append(np.array([float(x) for x in parts[:7]]))
    assert len(ref) == len(cases)
    return cases, [(k, np.array(v)) for k, v in ref]


def test_vallado_verification_cases(tmp_path):
    """Through the catalog loader and this wrapper, every case matches tcppver.out."""
    cases, ref = _verification_cases()
    records = [_tle_to_omm(l1, l2) for l1, l2, _ in cases]
    path = tmp_path / "ver.json"
    path.write_text(json.dumps(records))
    elsets = catalog.load_celestrak_omm(path)
    assert elsets.num_rows == len(cases) == 33

    errors = []
    n_compared = 0
    for i, (_, _, grid_min) in enumerate(cases):
        one = elsets.slice(i, 1)
        norad = int(one.column("norad_id")[0].as_py())
        at0 = C.sgp4_teme_from_epoch(one, [0.0])
        if at0.error[0, 0]:
            errors.append(int(at0.error[0, 0]))  # fails at epoch: no reference states
            continue
        batch = C.sgp4_teme_from_epoch(one, 60.0 * grid_min)
        got = np.concatenate(
            [
                np.column_stack([[0.0], at0.r_teme_km[0], at0.v_teme_km_s[0]]),
                np.column_stack([grid_min, batch.r_teme_km[0], batch.v_teme_km_s[0]]),
            ]
        )
        bad = np.flatnonzero(batch.error[0])
        n_ok = 1 + (bad[0] if bad.size else len(grid_min))
        if bad.size:
            errors.append(int(batch.error[0, bad[0]]))
        ref_norad, want = ref[i]
        assert ref_norad == norad
        assert want.shape[0] == n_ok, f"NORAD {norad}: {n_ok} valid states, ref {len(want)}"
        np.testing.assert_allclose(got[:n_ok, 0], want[:, 0], atol=1e-6)
        np.testing.assert_allclose(got[:n_ok, 1:], want[:, 1:], rtol=0, atol=2e-7)
        n_compared += n_ok
    # the sgp4 package's own expected error list for these cases, in file order
    assert errors == [1, 1, 6, 6, 4, 3, 6]
    assert n_compared > 600


def test_satrec_from_row_matches_twoline2rv():
    l1 = "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753"
    l2 = "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667"
    ref = Satrec.twoline2rv(l1, l2, WGS72)
    rec = _tle_to_omm(l1, l2)
    row = {
        "norad_id": rec["NORAD_CAT_ID"], "epoch_utc": rec["EPOCH"],
        "mean_motion_rev_per_day": rec["MEAN_MOTION"], "eccentricity": rec["ECCENTRICITY"],
        "inclination_deg": rec["INCLINATION"], "raan_deg": rec["RA_OF_ASC_NODE"],
        "arg_perigee_deg": rec["ARG_OF_PERICENTER"], "mean_anomaly_deg": rec["MEAN_ANOMALY"],
        "bstar_per_earth_radii": rec["BSTAR"], "mean_motion_dot": rec["MEAN_MOTION_DOT"],
    }
    sat = C.satrec_from_row(row)
    assert sat.operationmode == C.OPERATION_MODE == "i"
    for attr in ("no_kozai", "ecco", "inclo", "nodeo", "argpo", "mo", "bstar"):
        assert getattr(sat, attr) == pytest.approx(getattr(ref, attr), rel=1e-14, abs=1e-15)
    assert sat.jdsatepoch + sat.jdsatepochF == pytest.approx(
        ref.jdsatepoch + ref.jdsatepochF, abs=1e-10  # epoch rounded to the microsecond
    )
    e1, r1, _ = sat.sgp4_tsince(4320.0)
    e2, r2, _ = ref.sgp4_tsince(4320.0)
    assert e1 == e2 == 0
    np.testing.assert_allclose(r1, r2, atol=1e-8)


# ----------------------------------------------------------------- OMM fields


def test_satrec_from_row_epoch_forms_and_guards(sample):
    row = sample.slice(0, 1).to_pylist()[0]
    a = C.satrec_from_row({**row, "epoch_utc": "2024-01-01T00:00:00Z"})
    b = C.satrec_from_row({**row, "epoch_utc": "2024-01-01T00:00:00.000000"})
    assert (a.jdsatepoch, a.jdsatepochF) == (b.jdsatepoch, b.jdsatepochF) == (2460310.5, 0.0)
    with pytest.raises(ValueError, match="leap second"):
        C.satrec_from_row({**row, "epoch_utc": "2016-12-31T23:59:60.500000"})
    with pytest.raises(ValueError, match="offset"):
        C.satrec_from_row({**row, "epoch_utc": "2024-01-01T00:00:00+01:00"})
    with pytest.raises(ValueError, match="EPHEMERIS_TYPE 4"):
        C.satrec_from_row({**row, "ephemeris_type": 4})
    # null mean-motion derivatives are read as 0 (SGP4 doesn't use them)
    bare = {**row, "mean_motion_dot": None, "mean_motion_ddot": None}
    sat = C.satrec_from_row(bare)
    assert sat.error == 0 and sat.ndot == 0.0 and sat.nddot == 0.0
    np.testing.assert_array_equal(sat.sgp4_tsince(600.0)[1],
                                  C.satrec_from_row(row).sgp4_tsince(600.0)[1])


def test_vallado_satrecs_identical_to_twoline2rv():
    """Our reader of the element-set fields gives the library TLE reader's propagation."""
    cases, _ = _verification_cases()
    for line1, line2, grid in cases:
        ref = Satrec.twoline2rv(line1, line2, WGS72)
        rec = _tle_to_omm(line1, line2)
        sat = C.satrec_from_row(
            {
                "norad_id": rec["NORAD_CAT_ID"], "epoch_utc": rec["EPOCH"],
                "mean_motion_rev_per_day": rec["MEAN_MOTION"],
                "eccentricity": rec["ECCENTRICITY"], "inclination_deg": rec["INCLINATION"],
                "raan_deg": rec["RA_OF_ASC_NODE"], "arg_perigee_deg": rec["ARG_OF_PERICENTER"],
                "mean_anomaly_deg": rec["MEAN_ANOMALY"],
                "bstar_per_earth_radii": rec["BSTAR"],
                "mean_motion_dot": rec["MEAN_MOTION_DOT"],
                "mean_motion_ddot": rec["MEAN_MOTION_DDOT"],
            }
        )
        for t in np.concatenate([[0.0], grid]):
            e1, r1, v1 = sat.sgp4_tsince(t)
            e2, r2, v2 = ref.sgp4_tsince(t)
            assert e1 == e2
            if e1 == 0:
                assert r1 == r2 and v1 == v2


def test_missing_columns_raise(sample):
    with pytest.raises(ValueError, match="lacks columns"):
        C.propagate_sgp4(sample.drop_columns(["epoch_tt_s"]), [0.0])
    with pytest.raises(TypeError):
        C.propagate_sgp4(sample.to_pylist(), [0.0])


@pytest.mark.parametrize("grid", [[], [1.0, 1.0], [2.0, 1.0], [0.0, np.nan], [[0.0, 1.0]]])
def test_bad_grids_raise(sample, grid):
    with pytest.raises(ValueError):
        C.propagate_sgp4_from_epoch(sample.slice(0, 1), grid)


# ----------------------------------------------------------------- time handling


def test_tsince_is_physical_time_since_epoch(sample):
    """The wrapper's TEME state equals sgp4_tsince at (t_TT - epoch_TT) / 60 minutes."""
    one = sample.slice(_row_index(sample, 25544), 1)  # ISS
    epoch = float(one.column("epoch_tt_s")[0].as_py())
    t = epoch + np.array([-86400.0, -1.0, 0.0, 0.5, 3600.0, 5.5 * 86400.0])
    batch = C.sgp4_teme(one, t)
    sat = C.satrec_from_row(one.to_pylist()[0])
    for j, tj in enumerate(t):
        e, r, v = sat.sgp4_tsince((tj - epoch) / 60.0)
        assert e == 0
        np.testing.assert_allclose(batch.r_teme_km[0, j], r, rtol=0, atol=1e-9)
        np.testing.assert_allclose(batch.v_teme_km_s[0, j], v, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(batch.t_tt_s[0], t)


def test_leap_second_window_is_continuous_in_physical_time(sample):
    """A window across 2016-12-31T23:59:60 UTC, from an epoch on the leap-second day.

    The UTC labels run 23:59:55 ... 23:59:59, 23:59:60, 00:00:00 ... 00:00:05: twelve
    instants one SI second apart. The wrapper must place each at its physical time since
    epoch, so consecutive positions are one second of motion apart everywhere, including
    across the leap second; a UTC-date clock would put the instants after it 1 s
    (about 7.7 km) too early.
    """
    iss = _with(sample, _row_index(sample, 25544), epoch_utc="2016-12-31T12:00:00.000000")
    labels = [f"2016-12-31T23:59:{s:02d}" for s in range(55, 61)] + [
        f"2017-01-01T00:00:{s:02d}" for s in range(0, 6)
    ]
    t = np.asarray(F.utc_to_tt_seconds(labels))
    np.testing.assert_allclose(np.diff(t), 1.0, atol=1e-6)

    res = C.propagate_sgp4(iss, t)[0]
    assert res.status.ok
    epoch_tt = float(iss.column("epoch_tt_s")[0].as_py())
    # 12 h of UTC labels to 23:59:60 is 43,200 SI seconds; the leap second is one more
    assert t[5] - epoch_tt == pytest.approx(43200.0, abs=1e-6)
    assert t[6] - epoch_tt == pytest.approx(43201.0, abs=1e-6)

    # consecutive GCRF positions: one second of motion apart, everywhere
    r, v = res.trajectory.r_gcrf_km, res.trajectory.v_gcrf_km_s
    step = np.linalg.norm(np.diff(r, axis=0), axis=1)
    speed = np.linalg.norm(v, axis=1)
    np.testing.assert_allclose(step, 0.5 * (speed[1:] + speed[:-1]), rtol=1e-5)

    # equal to propagating in physical time directly
    sat = C.satrec_from_row(iss.to_pylist()[0])
    phys = np.array([sat.sgp4_tsince((tj - epoch_tt) / 60.0)[1] for tj in t])
    teme = C.sgp4_teme(iss, t).r_teme_km[0]
    np.testing.assert_allclose(teme, phys, rtol=0, atol=1e-9)

    # The trap, two ways. (1) A calendar clock (sgp4.api.jday on the UTC labels, the
    # library's usual interface) counts 86,400 s per day: 23:59:60 and 00:00:00 land on
    # the same minute, and every instant after the leap second is 1 s (~7.6 km) early.
    def at_jd(jd_fr):
        return np.array(sat.sgp4_array(*map(np.array, zip(*jd_fr, strict=True)))[1])

    cal = [jday(*map(int, (x[:4], x[5:7], x[8:10], x[11:13], x[14:16])), float(x[17:]))
           for x in labels]
    miss = np.linalg.norm(at_jd(cal) - phys, axis=1)
    assert np.all(miss[:6] < 1e-6)  # through 23:59:60: right
    assert np.all((miss[6:] > 7.0) & (miss[6:] < 8.0))  # after it: one second of motion
    # (2) Astropy's UTC Julian dates stretch a leap-second day to 86,401 s, while SGP4's
    # epoch fraction assumes 86,400 s: mixing the two is wrong by up to 1 s all day,
    # before the leap second as well as after it.
    utc = Time(labels, scale="utc")
    miss_astropy = np.linalg.norm(at_jd(list(zip(utc.jd1, utc.jd2, strict=True))) - phys, axis=1)
    assert np.all(miss_astropy > 3.0)


# ----------------------------------------------------------------- contract and frames


def test_trajectory_contract_and_exact_rotation(sample):
    res = C.propagate_sgp4_from_epoch(sample, WEEK_60S[:181], rotation="exact")
    batch = C.sgp4_teme_from_epoch(sample, WEEK_60S[:181])
    for i, r in enumerate(res):
        tr = r.trajectory
        assert isinstance(tr, Trajectory)
        assert (tr.source, tr.frame, tr.time_scale) == ("sgp4", "GCRF", "TT")
        assert tr.norad_id == r.norad_id
        assert r.status.ok and len(tr) == 181
        np.testing.assert_array_equal(tr.t_tt_s, r.epoch_tt_s + WEEK_60S[:181])
        rg, vg = F.teme_to_gcrf(batch.r_teme_km[i], batch.v_teme_km_s[i], batch.t_tt_s[i])
        np.testing.assert_allclose(tr.r_gcrf_km, rg, rtol=0, atol=1e-12)
        np.testing.assert_allclose(tr.v_gcrf_km_s, vg, rtol=0, atol=1e-15)


def test_fast_rotation_matches_exact_over_a_week(sample):
    """Full 7-day, 60 s windows for two LEO objects: fast within 10 um of exact."""
    two = sample.take([_row_index(sample, 25544), _row_index(sample, 26998)])  # ISS, TIMED
    fast = C.propagate_sgp4_from_epoch(two, WEEK_60S)
    exact = C.propagate_sgp4_from_epoch(two, WEEK_60S, rotation="exact")
    for a, b in zip(fast, exact, strict=True):
        dr = np.linalg.norm(a.trajectory.r_gcrf_km - b.trajectory.r_gcrf_km, axis=1)
        dv = np.linalg.norm(a.trajectory.v_gcrf_km_s - b.trajectory.v_gcrf_km_s, axis=1)
        assert dr.max() < 1e-8  # km (10 micrometres; observed ~0.7)
        assert dv.max() < 1e-8  # km/s (10 micrometres/s; observed ~0.7)


def test_epoch_states(sample):
    states = C.sgp4_epoch_states(sample)
    for row, s in zip(sample.to_pylist(), states, strict=True):
        assert s.status.ok and len(s.trajectory) == 1
        assert s.trajectory.t_tt_s[0] == row["epoch_tt_s"]
        e, r, v = C.satrec_from_row(row).sgp4_tsince(0.0)
        rg, vg = F.teme_to_gcrf(np.array(r), np.array(v), row["epoch_tt_s"])
        np.testing.assert_allclose(s.trajectory.r_gcrf_km[0], rg, rtol=0, atol=1e-12)
        np.testing.assert_allclose(s.trajectory.v_gcrf_km_s[0], vg, rtol=0, atol=1e-15)


def test_rotation_argument_checked(sample):
    with pytest.raises(ValueError, match="rotation"):
        C.propagate_sgp4_from_epoch(sample.slice(0, 1), [0.0], rotation="approx")


# ----------------------------------------------------------------- batching


def test_batched_and_single_results_identical(sample):
    grid_rel = WEEK_60S[::30]
    t0 = float(np.max(sample.column("epoch_tt_s").to_numpy()))
    grid_abs = t0 + grid_rel
    for run in (
        lambda tab, **kw: C.propagate_sgp4_from_epoch(tab, grid_rel, **kw),
        lambda tab, **kw: C.propagate_sgp4(tab, grid_abs, **kw),
    ):
        together = run(sample)
        chunked = run(sample, chunk_objects=3)
        for i in range(sample.num_rows):
            alone = run(sample.slice(i, 1))[0]
            for other in (together[i], chunked[i]):
                assert other.status == alone.status
                np.testing.assert_array_equal(other.trajectory.t_tt_s, alone.trajectory.t_tt_s)
                np.testing.assert_array_equal(
                    other.trajectory.r_gcrf_km, alone.trajectory.r_gcrf_km
                )
                np.testing.assert_array_equal(
                    other.trajectory.v_gcrf_km_s, alone.trajectory.v_gcrf_km_s
                )


# ----------------------------------------------------------------- errors and decay


def test_decay_inside_window_truncates(sample):
    """A constructed high-drag ISS element set decays (SGP4 code 6) inside 7 days.

    With the altitude floor off, so SGP4's own decay test is what cuts the trajectory.
    """
    i = _row_index(sample, 25544)
    decaying = _with(sample, i, bstar_per_earth_radii=0.1)
    both = pa.concat_tables([sample.slice(i, 1), decaying])
    ok, dec = C.propagate_sgp4_from_epoch(both, WEEK_60S, floor_km=None)
    assert ok.status.ok and len(ok.trajectory) == len(WEEK_60S)

    s = dec.status
    assert s.code == 6 and "decayed" in s.message
    assert s.truncated and not s.ok
    assert 0 < s.n_valid < s.n_requested == len(WEEK_60S)
    assert len(dec.trajectory) == s.n_valid
    assert s.first_error_t_tt_s == pytest.approx(dec.epoch_tt_s + WEEK_60S[s.n_valid])
    # every returned state is above one Earth radius; the raw batch confirms the rule
    assert np.all(np.linalg.norm(dec.trajectory.r_gcrf_km, axis=1) > RE_WGS72_KM)
    raw = C.sgp4_teme_from_epoch(decaying, WEEK_60S)
    assert np.all(raw.error[0, : s.n_valid] == 0) and raw.error[0, s.n_valid] == 6

    counts = C.status_counts([ok, dec])
    assert counts == {"ok": 1, "truncated": 1, "empty": 0, "code_6": 1}
    assert C.results_ok([ok, dec]) == [ok.trajectory]


def test_invalid_elements_give_no_trajectory(sample):
    bad = _with(sample, 0, eccentricity=-0.01)  # SGP4 accepts down to -0.001
    res = C.propagate_sgp4_from_epoch(bad, WEEK_60S[:10])[0]
    assert res.trajectory is None
    assert res.status.code == 1 and res.status.n_valid == 0
    assert res.status.first_error_t_tt_s == res.epoch_tt_s
    assert C.status_counts([res]) == {"ok": 0, "truncated": 0, "empty": 1, "code_1": 1}


# ----------------------------------------------------------------- 120 km altitude floor


def test_floor_cuts_a_decaying_trajectory_before_sgp4_does(sample):
    """Default floor: the same decaying set stops at 120 km, long before SGP4's code 6."""
    i = _row_index(sample, 25544)
    decaying = _with(sample, i, bstar_per_earth_radii=0.1)
    res = C.propagate_sgp4_from_epoch(decaying, WEEK_60S)[0]
    no_floor = C.propagate_sgp4_from_epoch(decaying, WEEK_60S, floor_km=None)[0]
    s = res.status
    assert s.code == C.ALTITUDE_FLOOR_CODE and s.below_floor and s.truncated and not s.ok
    assert "120 km floor" in s.message
    assert 0 < s.n_valid < no_floor.status.n_valid
    assert s.first_error_t_tt_s == pytest.approx(res.epoch_tt_s + WEEK_60S[s.n_valid])
    # the kept states are those of the unfloored run, and the cut is at the crossing
    np.testing.assert_array_equal(
        res.trajectory.r_gcrf_km, no_floor.trajectory.r_gcrf_km[: s.n_valid]
    )
    raw = C.sgp4_teme_from_epoch(decaying, WEEK_60S)
    alt = C.geodetic_altitude_km(raw.r_teme_km[0, : s.n_valid + 1])
    assert np.all(alt[:-1] >= C.ALTITUDE_FLOOR_KM) and alt[-1] < C.ALTITUDE_FLOOR_KM
    assert raw.error[0, s.n_valid] == 0  # SGP4 itself still calls this state valid
    assert C.status_counts([res]) == {"ok": 0, "truncated": 1, "empty": 0, "altitude_floor": 1}
    assert not C.Sgp4Status(6, "x", 2, 1, 0.0).below_floor


def test_floor_matches_the_truth_reentry_altitude():
    """One floor for both propagators: cheap_prop's constant equals config/truth.toml's."""
    from orbit_correction.truth_prop import load_truth_config

    assert C.ALTITUDE_FLOOR_KM == load_truth_config().reentry_altitude_km


def _batch(r_teme: np.ndarray, error: np.ndarray, t0: float = 7.8e8) -> C.TemeBatch:
    """A hand-made one-object TemeBatch (positions and error codes chosen by the test)."""
    n = r_teme.shape[0]
    return C.TemeBatch(
        norad_id=np.array([1]),
        epoch_tt_s=np.array([t0]),
        t_tt_s=t0 + 60.0 * np.arange(n)[None, :],
        r_teme_km=r_teme[None],
        v_teme_km_s=np.tile([0.0, 7.5, 0.0], (1, n, 1)),
        error=error[None].astype(np.uint8),
    )


def test_floor_against_sgp4_flags_on_a_hand_made_batch():
    """Which cut wins, and flagged (possibly NaN) states never reach the altitude test."""
    a = 6378.137  # equatorial points: altitude = radius - a
    r = np.array([[a + h, 0.0, 0.0] for h in (400.0, 300.0, 110.0, 400.0)])
    # floor first
    res = C._to_results(_batch(r, np.array([0, 0, 0, 6])), "exact")[0]
    assert res.status.below_floor and res.status.n_valid == 2
    # SGP4 flag on the same state as the floor crossing: SGP4's code wins
    res = C._to_results(_batch(r, np.array([0, 0, 6, 0])), "exact")[0]
    assert res.status.code == 6 and res.status.n_valid == 2
    # flagged state with NaN position before any crossing: SGP4's code, no error raised
    r_nan = r.copy()
    r_nan[1] = np.nan
    res = C._to_results(_batch(r_nan, np.array([0, 1, 0, 0])), "exact")[0]
    assert res.status.code == 1 and res.status.n_valid == 1
    # start below the floor: no trajectory
    res = C._to_results(_batch(r[2:], np.array([0, 0])), "exact")[0]
    assert res.trajectory is None and res.status.below_floor and res.status.n_valid == 0
    # floor off: only SGP4's flags count
    res = C._to_results(_batch(r, np.array([0, 0, 0, 0])), "exact", floor_km=None)[0]
    assert res.status.ok


def test_geodetic_altitude_known_points_and_guards():
    a, b = 6378.137, 6356.752314245  # WGS-84 semi-axes, km
    r = np.array([[a + 120.0, 0, 0], [0, 0, b + 120.0], [0, -(a + 500.0), 0]])
    np.testing.assert_allclose(C.geodetic_altitude_km(r), [120.0, 120.0, 500.0], atol=1e-9)
    with pytest.raises(ValueError, match="NaN"):
        C.geodetic_altitude_km(np.array([[np.nan, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="shape"):
        C.geodetic_altitude_km(np.zeros((2, 2)))


def _itrs_altitude_km(r_teme_km: np.ndarray, t_tt_s: np.ndarray) -> np.ndarray:
    """Altitude after astropy's full TEME -> ITRS rotation (which includes polar motion)."""
    import erfa
    from astropy import units as u
    from astropy.coordinates import ITRS, TEME, CartesianRepresentation

    obstime = F.tt_seconds_to_time(t_tt_s)
    teme = TEME(CartesianRepresentation(r_teme_km.T * u.km), obstime=obstime)
    itrs = teme.transform_to(ITRS(obstime=obstime)).cartesian.xyz.to_value(u.m).T
    return erfa.gc2gd(1, itrs)[2] / 1000.0


def test_ignoring_polar_motion_moves_altitude_by_centimetres(sample):
    """TEME-based altitude against astropy's ITRS-based altitude: under 0.1 m.

    TEME -> ITRS in astropy is a rotation about z (GMST) followed by polar motion, so
    the difference isolates polar motion; the bound the handoff asks for is 1 m.
    """
    leo = sample.filter(pa.compute.less(sample.column("perigee_alt_km"), 2000.0))
    res = C.sgp4_teme_from_epoch(leo, 60.0 * np.arange(0, 1440, 7))  # a day, 7-min steps
    ok = res.error == 0
    r, t = res.r_teme_km[ok], res.t_tt_s[ok]
    diff_m = np.abs(C.geodetic_altitude_km(r) - _itrs_altitude_km(r, t)) * 1000.0
    assert r.shape[0] > 1000
    assert diff_m.max() < 0.1, diff_m.max()


HAVE_OREKIT = all(
    importlib.util.find_spec(m) for m in ("orekit_jpype", "jdk4py", "orekitdata")
)


@pytest.mark.datagen
@pytest.mark.skipif(not HAVE_OREKIT, reason="needs `uv sync --group datagen` (Orekit)")
def test_floor_altitude_matches_orekit(sample):
    """The JVM-free altitude against Orekit's (the truth propagator's) within 10 m.

    Orekit converts GCRF to its WGS-84 body frame with IERS 2010 and the bundle's EOP;
    here the same state's TEME position goes through ERFA. Sample: every LEO object of
    the fixture over one day, plus a decaying set's states near the floor.
    """
    from orbit_correction import truth_prop as T

    cfg = T.load_truth_config()
    leo = sample.filter(pa.compute.less(sample.column("perigee_alt_km"), 2000.0))
    i = _row_index(sample, 25544)
    decaying = _with(sample, i, bstar_per_earth_radii=0.1)
    no_floor = C.propagate_sgp4_from_epoch(decaying, WEEK_60S, floor_km=None)[0]
    cut = C.propagate_sgp4_from_epoch(decaying, WEEK_60S)[0].status.n_valid
    grids = [(leo, 60.0 * np.arange(0, 1440, 97)), (decaying, WEEK_60S[cut - 3 : cut + 3])]
    worst_m, lowest_km, n = 0.0, np.inf, 0
    for elsets, grid in grids:
        raw = C.sgp4_teme_from_epoch(elsets, grid)
        for k in range(elsets.num_rows):
            for j in np.flatnonzero(raw.error[k] == 0):
                r, v = F.teme_to_gcrf(raw.r_teme_km[k, j], raw.v_teme_km_s[k, j], raw.t_tt_s[k, j])
                one = Trajectory(
                    norad_id=int(raw.norad_id[k]),
                    t_tt_s=np.array([raw.t_tt_s[k, j]]),
                    r_gcrf_km=r[None],
                    v_gcrf_km_s=v[None],
                    source="sgp4",
                )
                ours = float(C.geodetic_altitude_km(raw.r_teme_km[k, j]))
                theirs = T.initial_geodetic_altitude_km(cfg, one)
                worst_m = max(worst_m, abs(ours - theirs) * 1000.0)
                lowest_km = min(lowest_km, ours)
                n += 1
    assert n > 200 and lowest_km < C.ALTITUDE_FLOOR_KM < lowest_km + 5.0
    assert no_floor.status.n_valid > cut
    assert worst_m < 10.0, worst_m
