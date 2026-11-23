"""Contracts between the project's components.

Three contracts live here. Every component that produces or consumes one of them
builds against this file, not against another component's work in progress, and
``tests/test_contracts.py`` checks them. Changing a contract needs approval
(see CLAUDE.md), because every producer and consumer has to move with it.

1. :class:`Trajectory`: a sampled orbit (one state or many). Fixes the frame
   (GCRF), the time scale and epoch (TT seconds since J2000.0) and the units
   (km, km/s), so SGP4 output, truth output and corrected output can be
   subtracted without hidden conversions.
2. The residual dataset schema (:data:`RESIDUAL_SCHEMA`, :data:`RESIDUAL_COLUMNS`,
   :func:`validate_residual_table`): one row per (object, start epoch, time
   since start), holding the model's features and its curvilinear RTN targets.
3. :class:`CorrectionModel`: the interface a trained correction exposes to
   screening: features in, curvilinear (R, T, N) corrections in km out.

Curvilinear RTN residual (definition; the code is in a later step)
-------------------------------------------------------------------
The residual is **cheap minus truth**, expressed in a frame built from the
**cheap (SGP4) state alone**. That is deliberate: when a correction is applied
during screening, the cheap state is the only state available, so the frame
must be reconstructible from it.

Inputs, all GCRF: the cheap position and velocity ``r_c``, ``v_c`` (km, km/s)
and the truth position ``r_t`` (km), at the same instant.

Axes from the cheap state:

* ``u_r = r_c / |r_c|`` (radial unit vector).
* ``n = (r_c x v_c) / |r_c x v_c|`` (orbit normal, along the cheap orbit's
  angular momentum).
* ``u_t = n x u_r`` (in-plane, perpendicular to ``u_r``, pointing in the
  direction of motion).

Angles of the truth position in that frame:

* ``phi_t = arcsin((r_t . n) / |r_t|)``: out-of-plane angle of the truth position
  from the cheap orbit plane, positive on the ``+n`` side.
* ``r_p = r_t - (r_t . n) n``: truth position projected onto the cheap plane.
* ``du = atan2((r_p x r_c) . n, r_p . r_c)``: in-plane angle **from** ``r_p``
  **to** ``r_c``, measured about ``n``, in (-pi, pi]. Positive when the cheap
  position is *ahead* of the truth position in the direction of motion.

Components (all km, all cheap minus truth):

* **R (radial)** ``= |r_c| - |r_t|``. Positive when the cheap position is higher.
* **T (along-track)** ``= |r_c| * du``. Positive when the cheap position is ahead
  of truth along the direction of motion. Arc length at the cheap radius, so a
  large along-track error does not leak into R the way it does in rectilinear
  RTN.
* **N (cross-track)** ``= |r_c| * (0 - phi_t) = -|r_c| * phi_t``. The cheap
  position has out-of-plane angle 0 by construction, so cheap minus truth is
  ``-phi_t``. N is positive when the cheap position lies on the ``+n`` side of
  the truth position, i.e. when truth is on the ``-n`` side of the cheap plane.

Inverse (how a correction is applied). Given ``r_c``, ``v_c`` and (R, T, N):

1. ``rho = |r_c| - R`` (truth radius), ``du = T / |r_c|``, ``phi = -N / |r_c|``.
2. In-plane direction of truth: rotate ``u_r`` by ``-du`` about ``n``:
   ``d = cos(du) u_r - sin(du) u_t``.
3. ``r_t = rho * (cos(phi) d + sin(phi) n)``.

The inverse is exact (it is a spherical coordinate system attached to the cheap
state) provided ``|du| < pi`` and the cheap angular momentum is non-zero. A model
that predicts (R, T, N) from features is turned into a corrected position with
this inverse, using nothing but the cheap state. Velocity residuals are not part
of the target. Total miss distance in screening stays Cartesian.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

# --------------------------------------------------------------------------------------
# Conventions shared by every contract
# --------------------------------------------------------------------------------------

#: Frame of every position and velocity that crosses a component boundary.
FRAME = "GCRF"
#: Time scale of every time that crosses a component boundary. TT has no leap
#: seconds, so differences of TT seconds are true elapsed (SI) seconds.
TIME_SCALE = "TT"
#: Zero point of the time axis: J2000.0 = 2000-01-01T12:00:00 TT = JD 2451545.0 TT.
TIME_EPOCH = "J2000.0"
J2000_JD_TT = 2451545.0
POSITION_UNIT = "km"
VELOCITY_UNIT = "km/s"

# Plausibility bounds used to catch unit mistakes (metres instead of km, m/s
# instead of km/s, Earth radii instead of km). They are deliberately loose: a
# geocentric object anywhere from the surface to well beyond GEO passes.
_R_MIN_KM = 6000.0
_R_MAX_KM = 100_000.0
_V_MAX_KM_S = 15.0

_SOURCE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class ContractError(ValueError):
    """Raised when data does not satisfy one of the contracts in this module."""


def _check_float64_array(name: str, arr: object, shape_tail: tuple[int, ...]) -> np.ndarray:
    """Check dtype and shape strictly (no silent casting) and return a read-only copy."""
    if not isinstance(arr, np.ndarray):
        raise ContractError(f"{name} must be a numpy array, got {type(arr).__name__}")
    if arr.dtype != np.float64:
        raise ContractError(f"{name} must have dtype float64, got {arr.dtype}")
    if arr.ndim != 1 + len(shape_tail) or arr.shape[1:] != shape_tail:
        want = "(N" + "".join(f", {d}" for d in shape_tail) + ")"
        raise ContractError(f"{name} must have shape {want}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ContractError(f"{name} contains NaN or inf")
    out = np.array(arr, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out


# --------------------------------------------------------------------------------------
# Contract 1: trajectory / state
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Trajectory:
    """Positions and velocities of one object at N instants, in GCRF, km and km/s.

    A single state is a trajectory with N = 1. Construction validates everything
    and stores read-only copies of the arrays, so a ``Trajectory`` that exists is
    known to satisfy the contract.

    Attributes:
        norad_id: NORAD catalog number (positive int).
        t_tt_s: shape (N,), float64. TT seconds since J2000.0, strictly increasing.
            Convert to and from UTC with :mod:`orbit_correction.frames`.
        r_gcrf_km: shape (N, 3), float64. Geocentric GCRF position, km.
        v_gcrf_km_s: shape (N, 3), float64. GCRF velocity, km/s.
        source: what produced it, a lowercase identifier such as ``"sgp4"``,
            ``"truth"``, ``"corrected"`` or ``"crosscheck"``.
        frame, time_scale, position_unit, velocity_unit: declared conventions.
            They exist so that a producer states what it is handing over; any
            value other than the project convention is rejected.

    Raises:
        ContractError: on a wrong type, dtype, shape, non-finite value, declared
            frame/time scale/unit, non-increasing times, or magnitudes that
            point to a unit mistake.
    """

    norad_id: int
    t_tt_s: np.ndarray
    r_gcrf_km: np.ndarray
    v_gcrf_km_s: np.ndarray
    source: str
    frame: str = FRAME
    time_scale: str = TIME_SCALE
    position_unit: str = POSITION_UNIT
    velocity_unit: str = VELOCITY_UNIT

    def __post_init__(self) -> None:
        if isinstance(self.norad_id, bool) or not isinstance(self.norad_id, (int, np.integer)):
            raise ContractError(f"norad_id must be an int, got {type(self.norad_id).__name__}")
        if self.norad_id <= 0:
            raise ContractError(f"norad_id must be positive, got {self.norad_id}")
        object.__setattr__(self, "norad_id", int(self.norad_id))

        for attr, want in (
            ("frame", FRAME),
            ("time_scale", TIME_SCALE),
            ("position_unit", POSITION_UNIT),
            ("velocity_unit", VELOCITY_UNIT),
        ):
            got = getattr(self, attr)
            if got != want:
                raise ContractError(f"{attr} must be {want!r}, got {got!r}")

        if not isinstance(self.source, str) or not _SOURCE_RE.match(self.source):
            raise ContractError(
                f"source must be a lowercase identifier like 'sgp4', got {self.source!r}"
            )

        t = _check_float64_array("t_tt_s", self.t_tt_s, ())
        r = _check_float64_array("r_gcrf_km", self.r_gcrf_km, (3,))
        v = _check_float64_array("v_gcrf_km_s", self.v_gcrf_km_s, (3,))
        n = t.shape[0]
        if n == 0:
            raise ContractError("a trajectory needs at least one state")
        if r.shape[0] != n or v.shape[0] != n:
            raise ContractError(
                f"length mismatch: t has {n}, r has {r.shape[0]}, v has {v.shape[0]}"
            )
        if n > 1 and not np.all(np.diff(t) > 0):
            raise ContractError("t_tt_s must be strictly increasing")

        rmag = np.linalg.norm(r, axis=1)
        if np.any(rmag < _R_MIN_KM) or np.any(rmag > _R_MAX_KM):
            raise ContractError(
                f"|r| outside [{_R_MIN_KM}, {_R_MAX_KM}] km (range {rmag.min():.6g}"
                f"-{rmag.max():.6g}); positions must be geocentric and in km"
            )
        vmag = np.linalg.norm(v, axis=1)
        if np.any(vmag > _V_MAX_KM_S):
            raise ContractError(
                f"|v| above {_V_MAX_KM_S} km/s (max {vmag.max():.6g}); velocities must be in km/s"
            )

        object.__setattr__(self, "t_tt_s", t)
        object.__setattr__(self, "r_gcrf_km", r)
        object.__setattr__(self, "v_gcrf_km_s", v)

    def __len__(self) -> int:
        return int(self.t_tt_s.shape[0])


# --------------------------------------------------------------------------------------
# Contract 2: residual dataset schema
# --------------------------------------------------------------------------------------

#: Split labels a residual row may carry. ``splits.py`` assigns them.
#: train / dev: training objects, windows entirely before the time cutoff.
#: test: held-out objects at or after the cutoff (the confirmatory set).
#: reserved_object: held-out objects before the cutoff (never trained on).
#: reserved_time: training objects at or after the cutoff (never trained on).
#: excluded: rows removed by a rule (window crossing the cutoff, event window).
SPLIT_LABELS = ("train", "dev", "test", "reserved_object", "reserved_time", "excluded")


@dataclass(frozen=True)
class ResidualColumn:
    """One column of the residual dataset.

    Attributes:
        name: column name.
        dtype: pyarrow type.
        unit: physical unit, or ``""`` for labels and identifiers.
        role: ``"id"``, ``"time"``, ``"feature"``, ``"feature_observed"``
            (space weather observed over the horizon, for the labeled comparison
            only, never a headline model input), ``"label"`` or ``"target"``.
        nullable: whether nulls are allowed. Missing values stay missing; they are
            never replaced by placeholders.
        doc: one-line description.
    """

    name: str
    dtype: pa.DataType
    unit: str
    role: str
    nullable: bool
    doc: str


_F64 = pa.float64()

#: Every column of a residual file, in order. Columns marked nullable may hold
#: nulls; all others must be present and non-null in every row.
RESIDUAL_COLUMNS: tuple[ResidualColumn, ...] = (
    # identifiers and split
    ResidualColumn("norad_id", pa.int64(), "", "id", False, "NORAD catalog number"),
    ResidualColumn("split", pa.string(), "", "id", False, f"one of {SPLIT_LABELS}"),
    # time
    ResidualColumn(
        "start_epoch_tt_s", _F64, "s", "time", False,
        "prediction start, TT seconds since J2000.0 (the element-set epoch in v1)",
    ),
    ResidualColumn(
        "dt_s", _F64, "s", "time", False, "time since start, SI seconds, >= 0"
    ),
    # features: SGP4 mean elements from the element set (TLE/OMM conventions)
    ResidualColumn(
        "elset_epoch_tt_s", _F64, "s", "feature", False,
        "element-set epoch, TT seconds since J2000.0",
    ),
    ResidualColumn("mean_motion_rev_per_day", _F64, "rev/day", "feature", False,
                   "SGP4 mean motion (Kozai, as in the element set)"),
    ResidualColumn("eccentricity", _F64, "", "feature", False, "SGP4 mean eccentricity"),
    ResidualColumn("inclination_deg", _F64, "deg", "feature", False, "SGP4 mean inclination"),
    ResidualColumn("raan_deg", _F64, "deg", "feature", False, "right ascension of ascending node"),
    ResidualColumn("arg_perigee_deg", _F64, "deg", "feature", False, "argument of perigee"),
    ResidualColumn("mean_anomaly_deg", _F64, "deg", "feature", False, "mean anomaly at epoch"),
    ResidualColumn("bstar_per_earth_radii", _F64, "1/ER", "feature", False, "SGP4 drag term B*"),
    ResidualColumn("mean_motion_dot", _F64, "rev/day^2", "feature", True,
                   "first derivative of mean motion / 2, as in the element set"),
    # features: space weather available at the start (as-of lookup; no centred averages)
    ResidualColumn("asof_f107_obs_sfu", _F64, "sfu", "feature", True,
                   "observed F10.7 of the last day complete at the start"),
    ResidualColumn("asof_f107_obs_trailing81_sfu", _F64, "sfu", "feature", True,
                   "trailing 81-day mean of observed F10.7, days complete at the start"),
    ResidualColumn("asof_ap_daily", _F64, "", "feature", True,
                   "daily Ap of the last day complete at the start"),
    ResidualColumn("asof_ap_3h_last", _F64, "", "feature", True,
                   "last 3-hour ap interval complete at the start"),
    # space weather observed over the horizon (labeled comparison only)
    ResidualColumn("obs_f107_obs_sfu", _F64, "sfu", "feature_observed", True,
                   "observed F10.7 of the day containing start + dt"),
    ResidualColumn("obs_f107_obs_ctr81_sfu", _F64, "sfu", "feature_observed", True,
                   "centred 81-day F10.7 mean for the day containing start + dt"),
    ResidualColumn("obs_ap_daily", _F64, "", "feature_observed", True,
                   "daily Ap of the day containing start + dt"),
    # stratification labels (bin definitions are set where the labels are computed)
    ResidualColumn("altitude_band", pa.string(), "", "label", False,
                   "altitude band label, for stratified reporting"),
    ResidualColumn("solar_activity_bin", pa.string(), "", "label", False,
                   "solar-activity bin label, for stratified reporting"),
    # targets: curvilinear RTN, cheap minus truth (definition in the module docstring)
    ResidualColumn("target_r_km", _F64, "km", "target", False, "radial, |r_c| - |r_t|"),
    ResidualColumn("target_t_km", _F64, "km", "target", False,
                   "along-track arc at the cheap radius, + when cheap is ahead"),
    ResidualColumn("target_n_km", _F64, "km", "target", False,
                   "cross-track arc at the cheap radius, + when cheap is on the +h side"),
)

#: The same columns as a pyarrow schema; units and roles are in field metadata,
#: so they travel with Parquet files written against it.
RESIDUAL_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field(
            c.name,
            c.dtype,
            nullable=c.nullable,
            metadata={"unit": c.unit, "role": c.role, "doc": c.doc},
        )
        for c in RESIDUAL_COLUMNS
    ],
    metadata={
        "contract": "orbit_correction.types.RESIDUAL_SCHEMA",
        "frame_of_targets": "curvilinear RTN of the cheap (SGP4) state",
        "time_scale": TIME_SCALE,
        "time_epoch": TIME_EPOCH,
    },
)

#: Column names by role, for convenience.
FEATURE_COLUMNS = tuple(c.name for c in RESIDUAL_COLUMNS if c.role == "feature")
OBSERVED_FEATURE_COLUMNS = tuple(c.name for c in RESIDUAL_COLUMNS if c.role == "feature_observed")
TARGET_COLUMNS = ("target_r_km", "target_t_km", "target_n_km")


def validate_residual_table(table: pa.Table) -> None:
    """Check a residual table against :data:`RESIDUAL_SCHEMA`.

    Checks: exactly the contract's columns, in order, with the contract's types;
    no nulls in non-nullable columns; split labels from :data:`SPLIT_LABELS`;
    ``dt_s >= 0``; finite times and targets. Extra or missing columns fail, so a
    schema change has to go through this file.

    Raises:
        ContractError: describing the first problem found.
    """
    if not isinstance(table, pa.Table):
        raise ContractError(f"expected a pyarrow Table, got {type(table).__name__}")
    want = [c.name for c in RESIDUAL_COLUMNS]
    if table.column_names != want:
        missing = [n for n in want if n not in table.column_names]
        extra = [n for n in table.column_names if n not in want]
        raise ContractError(
            f"columns differ from the contract (missing {missing}, extra {extra}, "
            "or out of order)"
        )
    for c in RESIDUAL_COLUMNS:
        col = table.column(c.name)
        if col.type != c.dtype:
            raise ContractError(f"column {c.name!r} has type {col.type}, contract says {c.dtype}")
        if not c.nullable and col.null_count:
            raise ContractError(f"column {c.name!r} has {col.null_count} nulls; not nullable")

    bad = pc.invert(pc.is_in(table.column("split"), value_set=pa.array(SPLIT_LABELS)))
    if pc.any(bad).as_py():
        raise ContractError(f"split labels must be in {SPLIT_LABELS}")
    for name in ("start_epoch_tt_s", "dt_s", "elset_epoch_tt_s", *TARGET_COLUMNS):
        arr = table.column(name).to_numpy(zero_copy_only=False)
        if not np.all(np.isfinite(arr)):
            raise ContractError(f"column {name!r} contains NaN or inf")
    if np.any(table.column("dt_s").to_numpy(zero_copy_only=False) < 0):
        raise ContractError("dt_s must be >= 0")


# --------------------------------------------------------------------------------------
# Contract 3: correction interface
# --------------------------------------------------------------------------------------


@runtime_checkable
class CorrectionModel(Protocol):
    """What a trained correction exposes to the rest of the project.

    A correction model takes rows of residual-schema features and returns the
    predicted curvilinear (R, T, N) residual, cheap minus truth, in km. The
    corrected position is then recovered from the cheap state alone with the
    inverse in this module's docstring, so screening never needs truth to use a
    model.

    Attributes:
        feature_columns: the residual-schema columns the model reads. Each must
            be one of :data:`FEATURE_COLUMNS`, :data:`OBSERVED_FEATURE_COLUMNS`
            or ``"dt_s"``. A model that reads observed-horizon space weather is the
            labeled comparison variant, never the headline.
    """

    feature_columns: tuple[str, ...]

    def predict_rtn_km(self, features: pa.Table) -> np.ndarray:
        """Predict residuals for each row.

        Args:
            features: a table with at least ``feature_columns``, named and typed as
                in :data:`RESIDUAL_SCHEMA`. Target columns may be absent.

        Returns:
            float64 array of shape (features.num_rows, 3): R, T, N in km.
        """
        ...


def check_correction_output(features: pa.Table, rtn_km: np.ndarray) -> np.ndarray:
    """Validate a :meth:`CorrectionModel.predict_rtn_km` result against its input.

    Returns the array unchanged if it is float64, shape (num_rows, 3) and finite.

    Raises:
        ContractError: otherwise.
    """
    arr = _check_float64_array("rtn_km", rtn_km, (3,))
    if arr.shape[0] != features.num_rows:
        raise ContractError(
            f"rtn_km has {arr.shape[0]} rows, features have {features.num_rows}"
        )
    return rtn_km


def check_feature_columns(model: CorrectionModel) -> None:
    """Check that a model only declares columns the residual schema offers as inputs."""
    allowed = set(FEATURE_COLUMNS) | set(OBSERVED_FEATURE_COLUMNS) | {"dt_s"}
    unknown = [c for c in model.feature_columns if c not in allowed]
    if unknown:
        raise ContractError(f"feature_columns not allowed as model inputs: {unknown}")
