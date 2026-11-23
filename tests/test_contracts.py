"""Contract tests for ``orbit_correction.types``.

These pin the three contracts between components. If one fails after a change
to ``types.py``, the change is a contract change and needs approval.
"""

import dataclasses

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from orbit_correction.types import (
    FEATURE_COLUMNS,
    FRAME,
    OBSERVED_FEATURE_COLUMNS,
    RESIDUAL_COLUMNS,
    RESIDUAL_SCHEMA,
    SPACE_WEATHER_MODEL_INPUTS,
    SPLIT_LABELS,
    TARGET_COLUMNS,
    TIME_SCALE,
    ContractError,
    CorrectionModel,
    Trajectory,
    check_correction_output,
    check_feature_columns,
    validate_residual_table,
)

# ------------------------------------------------------------------ trajectory


def _traj_arrays(n: int = 4):
    t = 8.0e8 + 60.0 * np.arange(n, dtype=np.float64)
    ang = np.linspace(0, 0.1, n)
    r = np.stack([7000 * np.cos(ang), 7000 * np.sin(ang), np.zeros(n)], axis=1)
    v = np.stack([-7.5 * np.sin(ang), 7.5 * np.cos(ang), np.zeros(n)], axis=1)
    return t, r, v


def test_trajectory_valid_construction_and_read_only():
    t, r, v = _traj_arrays()
    tr = Trajectory(25544, t, r, v, "sgp4")
    assert len(tr) == 4
    assert tr.frame == FRAME == "GCRF"
    assert tr.time_scale == TIME_SCALE == "TT"
    assert not tr.r_gcrf_km.flags.writeable
    # stored arrays are copies: mutating the input does not change the trajectory
    r[0, 0] = 1.0
    assert tr.r_gcrf_km[0, 0] == pytest.approx(7000.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        tr.source = "truth"  # type: ignore[misc]


def test_single_state_is_a_trajectory():
    t, r, v = _traj_arrays(1)
    assert len(Trajectory(1, t, r, v, "truth")) == 1


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda t, r, v: (t, r[:, :2].copy(), v), "shape"),
        (lambda t, r, v: (t, r, v[:-1].copy()), "length mismatch"),
        (lambda t, r, v: (t.reshape(-1, 1).copy(), r, v), "shape"),
        (lambda t, r, v: (t.astype(np.float32), r, v), "float64"),
        (lambda t, r, v: (t, r.astype(np.float32), v), "float64"),
        (lambda t, r, v: (t, r.tolist(), v), "numpy array"),
        (lambda t, r, v: (t[::-1].copy(), r, v), "strictly increasing"),
        (lambda t, r, v: (t, np.where(r == r[0, 0], np.nan, r), v), "NaN"),
        (lambda t, r, v: (t, r * 1000.0, v), "km"),  # metres
        (lambda t, r, v: (t, r / 6378.0, v), "km"),  # Earth radii
        (lambda t, r, v: (t, r, v * 1000.0), "km/s"),  # m/s
        (lambda t, r, v: (t[:0], r[:0], v[:0]), "at least one"),
    ],
)
def test_trajectory_rejects_bad_arrays(mutate, match):
    t, r, v = mutate(*_traj_arrays())
    with pytest.raises(ContractError, match=match):
        Trajectory(25544, t, r, v, "sgp4")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"frame": "TEME"}, "frame"),
        ({"time_scale": "UTC"}, "time_scale"),
        ({"position_unit": "m"}, "position_unit"),
        ({"velocity_unit": "m/s"}, "velocity_unit"),
        ({"source": "SGP4"}, "source"),
        ({"source": ""}, "source"),
        ({"norad_id": True}, "norad_id"),
        ({"norad_id": 0}, "positive"),
        ({"norad_id": 1.0}, "norad_id"),
    ],
)
def test_trajectory_rejects_bad_metadata(kwargs, match):
    t, r, v = _traj_arrays()
    base = {"norad_id": 25544, "source": "sgp4"}
    base.update(kwargs)
    with pytest.raises(ContractError, match=match):
        Trajectory(t_tt_s=t, r_gcrf_km=r, v_gcrf_km_s=v, **base)


# ------------------------------------------------------------------ residual schema


def _residual_table(n: int = 3) -> pa.Table:
    """A small valid residual table built from the column list."""
    cols = {}
    for c in RESIDUAL_COLUMNS:
        if c.name == "norad_id":
            cols[c.name] = pa.array([26998] * n, pa.int64())
        elif c.name == "split":
            cols[c.name] = pa.array(["train", "dev", "test"][:n], pa.string())
        elif c.name == "dt_s":
            cols[c.name] = pa.array(np.arange(n) * 3600.0)
        elif c.name in ("altitude_band", "solar_activity_bin"):
            cols[c.name] = pa.array(["500-600km"] * n, pa.string())
        elif c.nullable:
            cols[c.name] = pa.array([None] + [1.5] * (n - 1), pa.float64())
        else:
            cols[c.name] = pa.array(np.full(n, 0.25))
    return pa.table(cols, schema=RESIDUAL_SCHEMA)


def test_residual_table_validates():
    validate_residual_table(_residual_table())


def test_residual_parquet_round_trip_keeps_schema_and_units(tmp_path):
    path = tmp_path / "residuals.parquet"
    pq.write_table(_residual_table(), path)
    back = pq.read_table(path)
    validate_residual_table(back)
    assert back.schema.field("target_t_km").metadata[b"unit"] == b"km"
    assert back.schema.metadata[b"time_scale"] == b"TT"


def test_residual_schema_partitions_columns():
    names = [c.name for c in RESIDUAL_COLUMNS]
    assert len(names) == len(set(names))
    assert TARGET_COLUMNS == ("target_r_km", "target_t_km", "target_n_km")
    assert "bstar_per_earth_radii" in FEATURE_COLUMNS
    # observed-horizon space weather is never mixed into the headline feature list
    assert not set(FEATURE_COLUMNS) & set(OBSERVED_FEATURE_COLUMNS)
    assert all(c.startswith("obs_") for c in OBSERVED_FEATURE_COLUMNS)
    assert all(c.startswith("asof_") for c in FEATURE_COLUMNS if "f107" in c or "ap_" in c)


def test_space_weather_model_inputs_are_the_approved_four():
    """The approved start-of-prediction inputs, exactly; observed-horizon columns are optional."""
    assert SPACE_WEATHER_MODEL_INPUTS == (
        "asof_f107_obs_sfu",
        "asof_f107_obs_trailing81_sfu",
        "asof_ap_daily",
        "asof_ap_3h_last",
    )
    space_weather_features = [c for c in FEATURE_COLUMNS if "f107" in c or "ap_" in c]
    assert tuple(space_weather_features) == SPACE_WEATHER_MODEL_INPUTS
    by_name = {c.name: c for c in RESIDUAL_COLUMNS}
    assert all(by_name[c].role == "feature" for c in SPACE_WEATHER_MODEL_INPUTS)
    assert all(by_name[c].nullable for c in OBSERVED_FEATURE_COLUMNS)
    assert all(by_name[c].role == "feature_observed" for c in OBSERVED_FEATURE_COLUMNS)


def _replace(table: pa.Table, name: str, arr: pa.Array) -> pa.Table:
    i = table.column_names.index(name)
    return table.set_column(i, pa.field(name, arr.type), arr)


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda t: t.drop_columns(["target_n_km"]), "columns differ"),
        (lambda t: t.append_column("extra", pa.array([1, 2, 3])), "columns differ"),
        (lambda t: t.select(list(reversed(t.column_names))), "columns differ"),
        (lambda t: _replace(t, "norad_id", pa.array([1, 2, 3], pa.int32())), "type"),
        (lambda t: _replace(t, "target_r_km", pa.array([1.0, None, 2.0])), "nulls"),
        (lambda t: _replace(t, "split", pa.array(["train", "holdout", "test"])), "split"),
        (lambda t: _replace(t, "dt_s", pa.array([0.0, -1.0, 2.0])), "dt_s"),
        (lambda t: _replace(t, "target_t_km", pa.array([0.0, np.inf, 2.0])), "NaN or inf"),
    ],
)
def test_residual_table_rejects(edit, match):
    with pytest.raises(ContractError, match=match):
        validate_residual_table(edit(_residual_table()))


def test_split_labels_fixed():
    assert SPLIT_LABELS == (
        "train", "dev", "test", "reserved_object", "reserved_time", "excluded"
    )


# ------------------------------------------------------------------ correction interface


class _ZeroModel:
    """A dummy correction: predicts no residual."""

    feature_columns = ("dt_s", "bstar_per_earth_radii", "asof_f107_obs_sfu")

    def predict_rtn_km(self, features: pa.Table) -> np.ndarray:
        return np.zeros((features.num_rows, 3), dtype=np.float64)


class _NotAModel:
    feature_columns = ("dt_s",)


def test_dummy_satisfies_correction_protocol():
    model = _ZeroModel()
    assert isinstance(model, CorrectionModel)
    assert not isinstance(_NotAModel(), CorrectionModel)
    check_feature_columns(model)
    table = _residual_table()
    out = check_correction_output(table, model.predict_rtn_km(table.select(model.feature_columns)))
    assert out.shape == (3, 3)


def test_correction_output_checks():
    table = _residual_table()
    with pytest.raises(ContractError, match="rows"):
        check_correction_output(table, np.zeros((2, 3)))
    with pytest.raises(ContractError, match="shape"):
        check_correction_output(table, np.zeros((3, 2)))
    with pytest.raises(ContractError, match="float64"):
        check_correction_output(table, np.zeros((3, 3), dtype=np.float32))


def test_model_may_not_read_targets():
    class Leaky(_ZeroModel):
        feature_columns = ("dt_s", "target_t_km")

    with pytest.raises(ContractError, match="target_t_km"):
        check_feature_columns(Leaky())
