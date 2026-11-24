"""Tests for the residual row builder and its configuration (``orbit_correction.residuals``).

Most tests build the truth trajectory synthetically: truth = the inverse conversion
applied to the SGP4 states with a known residual, so the targets the builder must
return are known exactly. SGP4 runs on fixture element sets moved to 2024 epochs,
where the fixture space-weather file is fully observed. One ``datagen`` test runs the
real truth propagator end to end.

Fixtures: ``celestrak_gp_sample.json`` (element sets), ``sw_celestrak_2024_extract.txt``
(observed space weather 2023-10-01 to 2024-12-31, plus the published forecast
sections from 2026-11).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math

import numpy as np
import pyarrow as pa
import pytest

from orbit_correction import catalog
from orbit_correction import cheap_prop as C
from orbit_correction import frames as F
from orbit_correction import residuals as R
from orbit_correction.spaceweather import parse_spaceweather
from orbit_correction.truth_prop import TruthResult
from orbit_correction.types import (
    RESIDUAL_SCHEMA,
    TARGET_COLUMNS,
    ContractError,
    Trajectory,
    validate_residual_table,
)

from .conftest import FIXTURES

HAVE_OREKIT = all(importlib.util.find_spec(m) for m in ("orekit_jpype", "jdk4py", "orekitdata"))
EPOCH_UTC = "2024-06-15T00:00:00"
DAY = 86400.0


# ----------------------------------------------------------------- helpers


@pytest.fixture(scope="module")
def elsets() -> pa.Table:
    return catalog.load_celestrak_omm(FIXTURES / "celestrak_gp_sample.json")


@pytest.fixture(scope="module")
def sw():
    return parse_spaceweather(FIXTURES / "sw_celestrak_2024_extract.txt")


@pytest.fixture(scope="module")
def cfg() -> R.ResidualConfig:
    return R.load_residual_config()


@pytest.fixture(scope="module")
def grid(cfg) -> np.ndarray:
    return R.horizon_grid(cfg)


def one(elsets: pa.Table, norad: int, epoch_utc: str = EPOCH_UTC, **changes) -> pa.Table:
    """One element set moved to ``epoch_utc`` (epoch_tt_s recomputed), with changes."""
    i = elsets.column("norad_id").to_pylist().index(norad)
    rec = elsets.slice(i, 1).to_pylist()[0]
    rec.update(epoch_utc=epoch_utc, epoch_tt_s=float(F.elset_epoch_to_tt_seconds(epoch_utc)))
    rec.update(changes)
    return pa.Table.from_pylist([rec], schema=elsets.schema)


def known_rtn(dt_s: np.ndarray) -> np.ndarray:
    """A residual history of plausible shape: slow radial drift, quadratic along-track."""
    d = dt_s / DAY
    return np.column_stack(
        [0.002 + 0.05 * d, -3.0 * d**2 + 0.1 * np.sin(2 * np.pi * d), 0.01 * np.cos(d)]
    )


def synthetic_truth(sgp4: C.Sgp4Result, rtn: np.ndarray, *, n: int | None = None,
                    status: str = "ok") -> TruthResult:
    """Truth = inverse conversion of the SGP4 states with residual ``rtn`` (first n)."""
    tr = sgp4.trajectory
    n = len(tr) if n is None else n
    traj = None
    if n:
        traj = Trajectory(
            norad_id=tr.norad_id,
            t_tt_s=tr.t_tt_s[:n].copy(),
            r_gcrf_km=R.states_from_rtn(tr.r_gcrf_km[:n], tr.v_gcrf_km_s[:n], rtn[:n]),
            v_gcrf_km_s=tr.v_gcrf_km_s[:n].copy(),
            source="truth",
        )
    stop = None if status == "ok" else float(tr.t_tt_s[max(n - 1, 0)])
    return TruthResult(norad_id=tr.norad_id, status=status, trajectory=traj, stop_t_tt_s=stop)


def window(elsets, norad, grid, epoch_utc=EPOCH_UTC, **changes):
    el = one(elsets, norad, epoch_utc, **changes)
    sgp4 = C.propagate_sgp4_from_epoch(el, grid)[0]
    return el, sgp4


# ----------------------------------------------------------------- configuration


def test_config_is_pending_and_grid_has_the_proposed_shape(cfg, grid):
    assert "pending" in cfg.status
    assert cfg.altitude_edges_km == (250.0, 400.0, 600.0, 800.0, 1200.0, 2000.0)
    assert cfg.solar_edges_sfu == (90.0, 150.0)
    assert grid.size == 199 and grid[0] == 60.0 and grid[-1] == 7 * DAY
    assert np.all(np.diff(grid) > 0) and np.all(grid == np.round(grid))
    # log part dense at the start, uniform part every 6,048 s at the end
    assert np.all(np.diff(grid[:10]) < 15.0)
    assert np.all(np.isin(6048.0 * np.arange(1, 101), grid))


def test_bin_labels():
    edges = (250.0, 400.0, 2000.0)
    assert R.bin_label(249.99, edges) == "<250"
    assert R.bin_label(250.0, edges) == "250-400"  # half-open bins
    assert R.bin_label(399.99, edges) == "250-400"
    assert R.bin_label(2000.0, edges) == ">=2000"
    assert R.bin_label(90.5, (90.0, 150.0)) == "90-150"
    with pytest.raises(ValueError):
        R.bin_label(math.nan, edges)


def test_solar_bin_missing_is_labeled_not_guessed(cfg):
    assert R.solar_activity_bin(math.nan, cfg) == "unknown"
    assert R.solar_activity_bin(89.9, cfg) == "<90"
    assert R.solar_activity_bin(150.0, cfg) == ">=150"
    assert R.altitude_band(413.0, cfg) == "400-600"


def test_bad_config_values_raise(cfg):
    with pytest.raises(ValueError, match="increasing"):
        dataclasses.replace(cfg, altitude_edges_km=(400.0, 250.0))
    with pytest.raises(ValueError, match="log_first_s"):
        dataclasses.replace(cfg, log_first_s=0.0)
    with pytest.raises(ValueError, match="log_points"):
        dataclasses.replace(cfg, log_points=1)
    small = dataclasses.replace(cfg, log_points=0, uniform_points=7)
    np.testing.assert_array_equal(R.horizon_grid(small), DAY * np.arange(1, 8))


# ----------------------------------------------------------------- rows for an ok window


def test_ok_window_rows_targets_and_features(elsets, sw, cfg, grid):
    el, sgp4 = window(elsets, 25544, grid)
    rtn = known_rtn(grid)
    res = R.build_window_rows(el, sgp4, synthetic_truth(sgp4, rtn), sw, "train", grid, cfg)

    assert res.status == "ok" and res.causes == () and res.n_rows == grid.size
    t = res.table
    validate_residual_table(t)
    assert t.schema.equals(RESIDUAL_SCHEMA, check_metadata=True)
    # targets are the residual the truth was built from
    got = np.column_stack([t.column(c).to_numpy() for c in TARGET_COLUMNS])
    assert np.abs(got - rtn).max() < 1e-9
    # time columns
    rec = el.to_pylist()[0]
    np.testing.assert_array_equal(t.column("dt_s").to_numpy(), grid)
    assert set(t.column("start_epoch_tt_s").to_pylist()) == {rec["epoch_tt_s"]}
    assert set(t.column("elset_epoch_tt_s").to_pylist()) == {rec["epoch_tt_s"]}
    # element-set features copied unchanged
    for col in ("mean_motion_rev_per_day", "eccentricity", "inclination_deg", "raan_deg",
                "arg_perigee_deg", "mean_anomaly_deg", "bstar_per_earth_radii",
                "mean_motion_dot"):
        assert set(t.column(col).to_pylist()) == {rec[col]}, col
    assert set(t.column("norad_id").to_pylist()) == {25544}
    assert set(t.column("split").to_pylist()) == {"train"}
    # the four start-of-prediction space-weather values, from as_of at the epoch
    asof = sw.as_of(rec["epoch_tt_s"])
    assert asof.last_complete_day == "2024-06-14"
    for col, want in (
        ("asof_f107_obs_sfu", asof.f107_obs_sfu),
        ("asof_f107_obs_trailing81_sfu", asof.f107_obs_trailing81_sfu),
        ("asof_ap_daily", asof.ap_daily),
        ("asof_ap_3h_last", asof.ap_3h_last),
    ):
        assert math.isfinite(want) and set(t.column(col).to_pylist()) == {want}, col
    # obs_* are optional and null unless asked for
    for col in ("obs_f107_obs_sfu", "obs_f107_obs_ctr81_sfu", "obs_ap_daily"):
        assert t.column(col).null_count == grid.size
    # stratum: ISS perigee ~413 km; June 2024 trailing mean well above 150 sfu
    assert res.altitude_band == "400-600" and asof.f107_obs_trailing81_sfu > 150
    assert res.solar_activity_bin == ">=150"
    assert set(t.column("altitude_band").to_pylist()) == {"400-600"}
    assert set(t.column("solar_activity_bin").to_pylist()) == {">=150"}


def test_observed_columns_follow_the_day_of_each_row(elsets, sw, cfg, grid):
    el, sgp4 = window(elsets, 25544, grid)
    res = R.build_window_rows(
        el, sgp4, synthetic_truth(sgp4, known_rtn(grid)), sw, "dev", grid, cfg,
        include_observed=True,
    )
    t = res.table
    epoch = el.column("epoch_tt_s")[0].as_py()
    for j in (0, 50, 120, grid.size - 1):
        obs = sw.observed(epoch + grid[j])
        assert obs.section == "OBSERVED"
        assert t.column("obs_f107_obs_sfu")[j].as_py() == obs.f107_obs_sfu
        assert t.column("obs_f107_obs_ctr81_sfu")[j].as_py() == obs.f107_obs_ctr81_sfu
        assert t.column("obs_ap_daily")[j].as_py() == obs.ap_daily
    # 8 UTC days touched (2024-06-15 .. 2024-06-22), so the values change over the window
    assert len(set(t.column("obs_ap_daily").to_pylist())) > 1


def test_observed_values_beyond_the_observed_rows_stay_null(elsets, sw, cfg, grid):
    """Days with no row (2025 in the fixture) and forecast rows give null, never a value."""
    el, sgp4 = window(elsets, 25544, grid, "2024-12-29T00:00:00")
    res = R.build_window_rows(
        el, sgp4, synthetic_truth(sgp4, known_rtn(grid)), sw, "excluded", grid, cfg,
        include_observed=True,
    )
    days = np.array([s[:10] for s in F.tt_seconds_to_utc_isot(el["epoch_tt_s"][0].as_py()
                                                              + grid)])
    f107 = np.array(res.table.column("obs_f107_obs_sfu").to_pylist(), dtype=object)
    assert all(v is not None for v in f107[days <= "2024-12-31"])
    assert all(v is None for v in f107[days >= "2025-01-01"])
    # a forecast row (DAILY_PREDICTED, 2026-12 in this file) is not used either
    t_fc = F.utc_to_tt_seconds("2026-12-01T12:00:00")
    assert sw.observed(t_fc).section != "OBSERVED"
    vals = R._observed_columns(sw, np.array([t_fc]))
    assert all(np.isnan(v[0]) for v in vals)


def test_missing_values_stay_null(elsets, sw, cfg, grid):
    """Null mean_motion_dot stays null; a start without 81 days of history is 'unknown'."""
    el, sgp4 = window(elsets, 25544, grid, "2023-10-20T06:00:00", mean_motion_dot=None)
    res = R.build_window_rows(el, sgp4, synthetic_truth(sgp4, known_rtn(grid)), sw, "train",
                              grid, cfg)
    t = res.table
    assert t.column("mean_motion_dot").null_count == grid.size
    assert t.column("asof_f107_obs_trailing81_sfu").null_count == grid.size
    assert t.column("asof_f107_obs_sfu").null_count == 0
    assert res.solar_activity_bin == "unknown"
    assert set(t.column("solar_activity_bin").to_pylist()) == {"unknown"}


def test_rows_per_window_and_size(elsets, sw, cfg, grid):
    """One row per horizon; the in-memory size is what H-013's volume estimate needs."""
    el, sgp4 = window(elsets, 44714, grid)
    res = R.build_window_rows(el, sgp4, synthetic_truth(sgp4, known_rtn(grid)), sw, "train",
                              grid, cfg, include_observed=True)
    assert res.n_rows == grid.size == 199
    assert res.table.nbytes < 50_000  # measured ~40 kB in memory (25 columns x 199 rows)


# ----------------------------------------------------------------- the drop rule


def test_truth_reentry_drops_the_window(elsets, sw, cfg, grid):
    el, sgp4 = window(elsets, 25544, grid)
    for n in (120, 0):  # trajectory up to the stop, or none (start below 120 km)
        truth = synthetic_truth(sgp4, known_rtn(grid), n=n, status="reentry")
        res = R.build_window_rows(el, sgp4, truth, sw, "train", grid, cfg)
        assert res.status == "truth_reentry" and res.causes == ("truth_reentry",)
        assert res.table is None and res.n_rows == 0
        assert res.altitude_band == "400-600" and res.solar_activity_bin == ">=150"
        assert res.truth_status == "reentry" and res.sgp4_code == 0


def test_sgp4_error_drops_the_window(elsets, sw, cfg, grid):
    """SGP4's own decay code (floor off) inside the window: no rows."""
    el = one(elsets, 25544, bstar_per_earth_radii=0.1)
    sgp4 = C.propagate_sgp4_from_epoch(el, grid, floor_km=None)[0]
    assert sgp4.status.code == 6 and sgp4.status.truncated
    truth = synthetic_truth(sgp4, known_rtn(grid), n=len(sgp4.trajectory), status="reentry")
    res = R.build_window_rows(el, sgp4, truth, sw, "train", grid, cfg)
    assert res.status == "sgp4_error" and res.causes == ("sgp4_error", "truth_reentry")
    assert res.table is None and res.sgp4_code == 6 and "decayed" in res.message


def test_sgp4_floor_drops_the_window(elsets, sw, cfg, grid):
    """SGP4 below 120 km with truth still above it (truth ok over the whole window)."""
    el = one(elsets, 25544, bstar_per_earth_radii=0.1)
    sgp4 = C.propagate_sgp4_from_epoch(el, grid)[0]
    assert sgp4.status.below_floor and sgp4.status.truncated
    full = C.propagate_sgp4_from_epoch(el, grid, floor_km=None)[0]
    m = len(full.trajectory)
    # truth "ok" needs every grid point; reuse the healthy ISS states for the tail
    healthy = C.propagate_sgp4_from_epoch(one(elsets, 25544), grid)[0].trajectory
    truth = TruthResult(
        norad_id=25544,
        status="ok",
        trajectory=Trajectory(
            norad_id=25544,
            t_tt_s=healthy.t_tt_s.copy(),
            r_gcrf_km=np.concatenate([full.trajectory.r_gcrf_km, healthy.r_gcrf_km[m:]]),
            v_gcrf_km_s=np.concatenate([full.trajectory.v_gcrf_km_s, healthy.v_gcrf_km_s[m:]]),
            source="truth",
        ),
    )
    res = R.build_window_rows(el, sgp4, truth, sw, "test", grid, cfg)
    assert res.status == "sgp4_floor" and res.causes == ("sgp4_floor",)
    assert res.table is None and res.sgp4_code == C.ALTITUDE_FLOOR_CODE


def test_truth_error_drops_and_counts_by_stratum(elsets, sw, cfg, grid):
    el, sgp4 = window(elsets, 25544, grid)
    err = TruthResult(norad_id=25544, status="error", message="RuntimeError: boom")
    dropped = R.build_window_rows(el, sgp4, err, sw, "train", grid, cfg)
    assert dropped.status == "truth_error" and "boom" in dropped.message
    ok = R.build_window_rows(el, sgp4, synthetic_truth(sgp4, known_rtn(grid)), sw, "train",
                             grid, cfg)
    reentry = R.build_window_rows(
        el, sgp4, synthetic_truth(sgp4, known_rtn(grid), n=5, status="reentry"), sw, "train",
        grid, cfg,
    )
    el2, sgp4b = window(elsets, 13777, grid)  # IRAS, ~880 km
    ok2 = R.build_window_rows(el2, sgp4b, synthetic_truth(sgp4b, known_rtn(grid)), sw,
                              "train", grid, cfg)
    counts = R.count_windows([dropped, ok, reentry, ok2, ok])
    assert counts == {
        ("ok", "400-600", ">=150"): 2,
        ("ok", "800-1200", ">=150"): 1,
        ("truth_error", "400-600", ">=150"): 1,
        ("truth_reentry", "400-600", ">=150"): 1,
    }
    assert set(R.WINDOW_STATUSES) == {"ok", *R.DROP_STATUSES}


# ----------------------------------------------------------------- refused inputs


def test_mismatched_grid_is_refused(elsets, sw, cfg, grid):
    el, sgp4 = window(elsets, 25544, grid)
    truth = synthetic_truth(sgp4, known_rtn(grid))
    tr = truth.trajectory
    shifted = dataclasses.replace(
        truth, trajectory=dataclasses.replace(tr, t_tt_s=tr.t_tt_s + 1e-6)
    )
    with pytest.raises(R.GridMismatchError, match="truth times differ"):
        R.build_window_rows(el, sgp4, shifted, sw, "train", grid, cfg)
    # one time off by one float step is still refused: equality is exact
    t_bad = tr.t_tt_s.copy()
    t_bad[100] = np.nextafter(t_bad[100], np.inf)
    nudged = dataclasses.replace(truth, trajectory=dataclasses.replace(tr, t_tt_s=t_bad))
    with pytest.raises(R.GridMismatchError):
        R.build_window_rows(el, sgp4, nudged, sw, "train", grid, cfg)
    # truth "ok" but shorter than the window
    short = synthetic_truth(sgp4, known_rtn(grid), n=50)
    with pytest.raises(R.GridMismatchError, match="doesn't cover"):
        R.build_window_rows(el, sgp4, short, sw, "train", grid, cfg)
    # SGP4 run on another grid (same length, different values; and a different length)
    other = C.propagate_sgp4_from_epoch(el, grid + 1.0)[0]
    with pytest.raises(R.GridMismatchError, match="SGP4 times differ"):
        R.build_window_rows(el, other, truth, sw, "train", grid, cfg)
    with pytest.raises(R.GridMismatchError, match="SGP4 was run on"):
        R.build_window_rows(el, sgp4, truth, sw, "train", grid[:-1], cfg)
    # a dropped window is checked too: a re-entry trajectory off the grid is refused
    reentry = synthetic_truth(sgp4, known_rtn(grid), n=20, status="reentry")
    rt = reentry.trajectory
    off = dataclasses.replace(reentry, trajectory=dataclasses.replace(rt, t_tt_s=rt.t_tt_s + 30))
    with pytest.raises(R.GridMismatchError):
        R.build_window_rows(el, sgp4, off, sw, "train", grid, cfg)


def test_mismatched_objects_and_labels_are_refused(elsets, sw, cfg, grid):
    el, sgp4 = window(elsets, 25544, grid)
    truth = synthetic_truth(sgp4, known_rtn(grid))
    with pytest.raises(ValueError, match="split"):
        R.build_window_rows(el, sgp4, truth, sw, "training", grid, cfg)
    with pytest.raises(ValueError, match="NORAD"):
        R.build_window_rows(one(elsets, 13777), sgp4, truth, sw, "train", grid, cfg)
    with pytest.raises(ValueError, match="epoch"):
        later = one(elsets, 25544, "2024-06-15T00:00:01")
        R.build_window_rows(later, sgp4, truth, sw, "train", grid, cfg)
    with pytest.raises(ValueError, match="lacks"):
        rec = el.to_pylist()[0]
        del rec["perigee_alt_km"]
        R.build_window_rows(rec, sgp4, truth, sw, "train", grid, cfg)
    with pytest.raises(ValueError, match="increasing"):
        R.build_window_rows(el, sgp4, truth, sw, "train", grid[::-1], cfg)


def test_residual_outside_the_conversion_domain_raises(elsets, sw, cfg, grid):
    """Truth half an orbit away is an error to look at, not a row and not a silent drop."""
    el, sgp4 = window(elsets, 25544, grid)
    truth = synthetic_truth(sgp4, known_rtn(grid))
    tr = truth.trajectory
    flipped = dataclasses.replace(
        truth, trajectory=dataclasses.replace(tr, r_gcrf_km=-tr.r_gcrf_km)
    )
    with pytest.raises(R.RTNDomainError, match="along-track"):
        R.build_window_rows(el, sgp4, flipped, sw, "train", grid, cfg)


def test_table_rejects_tampering(elsets, sw, cfg, grid):
    """Sanity check that the produced table is held to the contract, not just accepted."""
    el, sgp4 = window(elsets, 25544, grid)
    t = R.build_window_rows(el, sgp4, synthetic_truth(sgp4, known_rtn(grid)), sw, "train",
                            grid, cfg).table
    with pytest.raises(ContractError):
        validate_residual_table(t.drop_columns(["target_n_km"]))


# ----------------------------------------------------------------- end to end with Orekit


@pytest.mark.datagen
@pytest.mark.skipif(not HAVE_OREKIT, reason="needs `uv sync --group datagen` (Orekit)")
def test_end_to_end_with_the_truth_propagator(elsets, sw, cfg, tmp_path):
    """SGP4 and real truth on the same grid -> rows that pass the contract.

    Starlink-1008 (~355 km) moved to 2024-06-15, the first day of the horizon grid.
    At 60 s the residual is the start-rule offset (about 2 m) plus a minute of
    divergence; over a day the along-track error grows to kilometres while R and N
    stay small, which is the shape the curvilinear frame is meant to show.
    """
    from orbit_correction import truth_prop as T

    tcfg = T.load_truth_config()
    copy = T.write_space_weather_copy(
        FIXTURES / "sw_celestrak_2024_extract.txt",
        tmp_path / "sw_orekit_2024-02-01_2024-11-15.txt",
        F.utc_to_tt_seconds("2024-02-01T00:00:00"),
        F.utc_to_tt_seconds("2024-11-15T00:00:00"),
    )
    grid = R.horizon_grid(cfg)
    grid = grid[grid <= DAY]
    el = one(elsets, 44714)
    sgp4 = C.propagate_sgp4_from_epoch(el, grid)[0]
    start = C.sgp4_epoch_states(el)[0].trajectory
    truth = T.propagate_truth(
        start, el["bstar_per_earth_radii"][0].as_py(), sgp4.trajectory.t_tt_s, tcfg, copy
    )
    assert truth.status == "ok"
    res = R.build_window_rows(el, sgp4, truth, sw, "train", grid, cfg, include_observed=True)
    assert res.status == "ok" and res.n_rows == grid.size
    validate_residual_table(res.table)
    rtn = np.column_stack([res.table.column(c).to_numpy() for c in TARGET_COLUMNS])
    assert np.linalg.norm(rtn[0]) < 0.05  # 60 s: start offset ~2 m plus a minute
    assert abs(rtn[-1, 1]) > 1.0  # a day: along-track error in kilometres
    assert np.abs(rtn[:, 0]).max() < 2.0 and np.abs(rtn[:, 2]).max() < 1.0
    assert res.altitude_band == "250-400"
