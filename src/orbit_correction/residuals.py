"""Residual targets: curvilinear RTN conversion, its inverse, and the row builder.

Why this module exists
----------------------
The model learns SGP4's error against the truth propagator. This module turns one
*window* (an element set, its SGP4 trajectory and the truth trajectory on the same
time grid) into rows of the residual dataset (:data:`orbit_correction.types.RESIDUAL_SCHEMA`):
features known at the prediction start, stratification labels, and the
**curvilinear RTN residual, cheap minus truth**, as targets. The inverse turns a
model's predicted (R, T, N) back into a corrected position using the cheap state
alone, which is what screening does with a trained model.

What is ours and what is library behaviour
------------------------------------------
Everything here is project code on numpy and pyarrow. The definition of the frame
and of the components is the contract in :mod:`orbit_correction.types` (module
docstring), which this module implements and does not change. Space-weather values
come from :class:`orbit_correction.spaceweather.SpaceWeather`; SGP4 and truth
trajectories from :mod:`~orbit_correction.cheap_prop` and
:mod:`~orbit_correction.truth_prop`.

The conversion (summary; the authoritative text is in ``types.py``)
------------------------------------------------------------------
Axes from the cheap state: ``u_r = r_c/|r_c|``, ``n = (r_c x v_c)/|r_c x v_c|``,
``u_t = n x u_r``. With the truth position's components ``x = r_t . u_r``,
``y = r_t . u_t``, ``z = r_t . n``:

* ``phi_t = atan2(z, sqrt(x^2 + y^2))``. This is the contract's
  ``arcsin((r_t . n)/|r_t|)``, written with ``atan2`` because it keeps full
  precision at every angle.
* ``du = atan2(-y, x)``. This is the contract's ``atan2((r_p x r_c) . n, r_p . r_c)``:
  with ``r_p = x u_r + y u_t`` and ``r_c = |r_c| u_r``, ``(r_p x r_c) . n = -y |r_c|``
  (because ``u_t x u_r = -n``) and ``r_p . r_c = x |r_c|``; the common factor
  ``|r_c| > 0`` does not change ``atan2``.
* ``R = |r_c| - |r_t|``, ``T = |r_c| du``, ``N = -|r_c| phi_t``.

``tests/test_rtn.py`` also checks this against a literal, row-by-row transcription
of the contract's cross-product formulas.

Where the contract leaves a choice (raised for a decision, not changed here)
---------------------------------------------------------------------------
* **Angle wrapping.** ``du`` is defined modulo a full turn, in (-pi, pi]. A truth
  position more than half an orbit behind (or ahead of) the cheap one would wrap to
  the other sign, and one instant can't tell the two apart. Near ``|du| = pi`` the
  target also jumps from about ``+pi |r_c|`` to ``-pi |r_c|`` for a tiny change in
  the state, which a model can't learn. This module refuses ``|du|`` above
  :data:`MAX_ABS_DU_RAD` (pi/2, about 10,500 km of along-track error at LEO radii;
  the largest 7-day error measured so far is about 1,000 km at 300 km altitude), in
  both directions, with :class:`RTNDomainError`.
* **Out-of-plane angle.** ``du`` is undefined when the truth position sits on the
  cheap orbit's normal (``phi_t = +-pi/2``) and badly conditioned near it. Refused
  above :data:`MAX_ABS_PHI_RAD` (pi/4); real cross-track errors are a few km
  (about 1e-3 rad).
* **Degenerate angular momentum.** ``n`` needs ``r_c x v_c != 0``. Refused when
  ``|r_c x v_c| < H_REL_MIN |r_c| |v_c|`` (velocity within about 2e-4 degrees of
  radial), and for zero or non-finite inputs.

Row builder and the drop rule
-----------------------------
:func:`build_window_rows` produces one row per horizon of a window, or no rows and a
status (:class:`WindowResult`). The drop rule (approved shared 120 km rule): a window
gives no rows if truth returned ``reentry`` (or ``error``), or SGP4 truncated or
failed anywhere in it, either from one of SGP4's own error codes or at the 120 km
altitude floor (:data:`orbit_correction.cheap_prop.ALTITUDE_FLOOR_CODE`). Dropped
windows keep their altitude band and solar-activity bin so the caller can count
them by stratum (:func:`count_windows`); they are never dropped silently.

The SGP4 and truth trajectories must sit on exactly the same instants: the row
builder refuses (raises :class:`GridMismatchError`) any trajectory whose
``t_tt_s`` differs, by any amount, from ``elset epoch + dt_s``.

This module never starts a JVM.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa

from orbit_correction import frames
from orbit_correction.types import (
    RESIDUAL_COLUMNS,
    RESIDUAL_SCHEMA,
    SPLIT_LABELS,
    validate_residual_table,
)

if TYPE_CHECKING:
    from orbit_correction.cheap_prop import Sgp4Result
    from orbit_correction.spaceweather import SpaceWeather
    from orbit_correction.truth_prop import TruthResult

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "config" / "residuals.toml"

# --------------------------------------------------------------------------------------
# Curvilinear RTN: forward and inverse
# --------------------------------------------------------------------------------------

#: Largest along-track angle |du| accepted, rad (see "Where the contract leaves a
#: choice" in the module docstring). Pending a project decision.
MAX_ABS_DU_RAD = math.pi / 2
#: Largest out-of-plane angle |phi| accepted, rad. Pending a project decision.
MAX_ABS_PHI_RAD = math.pi / 4
#: Smallest accepted |r_c x v_c| / (|r_c| |v_c|) (sine of the angle between r and v).
H_REL_MIN = 1e-6


class RTNDomainError(ValueError):
    """Inputs outside the domain where the curvilinear RTN conversion is defined."""


def _as_rows(name: str, a: object) -> tuple[np.ndarray, bool]:
    """(N, 3) float64 view of a (3,) or (N, 3) input, and whether it was a single row."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1 and arr.shape == (3,):
        return arr[None, :], True
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr, False
    raise ValueError(f"{name} must have shape (3,) or (N, 3), got {arr.shape}")


def _check_inputs(**arrays: np.ndarray) -> int:
    n = None
    for name, arr in arrays.items():
        if not np.all(np.isfinite(arr)):
            raise RTNDomainError(f"{name} contains NaN or inf")
        if n is None:
            n = arr.shape[0]
        elif arr.shape[0] != n:
            raise ValueError(f"row counts differ: {name} has {arr.shape[0]}, expected {n}")
    return int(n)


def cheap_frame(
    r_cheap_km: np.ndarray, v_cheap_km_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The curvilinear RTN axes of cheap states: ``(|r_c|, u_r, u_t, n)``.

    Args:
        r_cheap_km, v_cheap_km_s: shape (N, 3), GCRF.

    Returns:
        ``|r_c|`` shape (N,), and the unit vectors ``u_r``, ``u_t``, ``n``, each (N, 3).

    Raises:
        RTNDomainError: zero position, or angular momentum degenerate (see
            :data:`H_REL_MIN`).
    """
    rc = np.linalg.norm(r_cheap_km, axis=1)
    vc = np.linalg.norm(v_cheap_km_s, axis=1)
    if np.any(rc == 0.0):
        raise RTNDomainError("cheap position is zero")
    h = np.cross(r_cheap_km, v_cheap_km_s)
    hn = np.linalg.norm(h, axis=1)
    bad = ~(hn >= H_REL_MIN * rc * vc) | (hn == 0.0)
    if np.any(bad):
        raise RTNDomainError(
            f"cheap angular momentum degenerate in {int(bad.sum())} row(s) "
            f"(|r x v| / (|r| |v|) < {H_REL_MIN:g}); the orbit normal is undefined"
        )
    u_r = r_cheap_km / rc[:, None]
    n = h / hn[:, None]
    u_t = np.cross(n, u_r)
    return rc, u_r, u_t, n


def rtn_from_states(
    r_cheap_km: np.ndarray, v_cheap_km_s: np.ndarray, r_truth_km: np.ndarray
) -> np.ndarray:
    """Curvilinear RTN residual, cheap minus truth, in the frame of the cheap state.

    Implements the definition in :mod:`orbit_correction.types` (see this module's
    docstring for the equivalent component form used).

    Args:
        r_cheap_km: cheap (SGP4) positions, GCRF km, shape (3,) or (N, 3).
        v_cheap_km_s: cheap velocities, GCRF km/s, same shape.
        r_truth_km: truth positions at the same instants, GCRF km, same shape.

    Returns:
        (R, T, N) in km, shape (3,) or (N, 3) like the inputs. R > 0: cheap higher;
        T > 0: cheap ahead along the direction of motion; N > 0: cheap on the ``+n``
        side of truth.

    Raises:
        RTNDomainError: non-finite input, zero truth position, degenerate cheap
            angular momentum, ``|du| > MAX_ABS_DU_RAD`` or ``|phi_t| > MAX_ABS_PHI_RAD``.
        ValueError: on shapes that don't match.
    """
    rc_in, single = _as_rows("r_cheap_km", r_cheap_km)
    vc_in, _ = _as_rows("v_cheap_km_s", v_cheap_km_s)
    rt_in, _ = _as_rows("r_truth_km", r_truth_km)
    _check_inputs(r_cheap_km=rc_in, v_cheap_km_s=vc_in, r_truth_km=rt_in)
    rc, u_r, u_t, n = cheap_frame(rc_in, vc_in)

    x = np.einsum("ij,ij->i", rt_in, u_r)
    y = np.einsum("ij,ij->i", rt_in, u_t)
    z = np.einsum("ij,ij->i", rt_in, n)
    rho_p = np.hypot(x, y)  # |r_p|, the truth position projected onto the cheap plane
    rt = np.hypot(rho_p, z)
    if np.any(rt == 0.0):
        raise RTNDomainError("truth position is zero")
    phi = np.arctan2(z, rho_p)
    du = np.arctan2(-y, x)
    _check_angles(du, phi)

    out = np.column_stack([rc - rt, rc * du, -rc * phi])
    return out[0] if single else out


def states_from_rtn(
    r_cheap_km: np.ndarray, v_cheap_km_s: np.ndarray, rtn_km: np.ndarray
) -> np.ndarray:
    """Inverse: the truth (or corrected) position from the cheap state and (R, T, N).

    Steps exactly as in :mod:`orbit_correction.types`: ``rho = |r_c| - R``,
    ``du = T/|r_c|``, ``phi = -N/|r_c|``; ``d = cos(du) u_r - sin(du) u_t``;
    ``r = rho (cos(phi) d + sin(phi) n)``. Uses nothing but the cheap state, so it is
    how a model's predicted residual becomes a corrected position in screening.

    Args:
        r_cheap_km, v_cheap_km_s: cheap states, GCRF, shape (3,) or (N, 3).
        rtn_km: (R, T, N) in km, same shape.

    Returns:
        Positions in GCRF km, same shape as the inputs.

    Raises:
        RTNDomainError: non-finite input, degenerate cheap angular momentum,
            ``R >= |r_c|`` (no positive radius), ``|du| > MAX_ABS_DU_RAD`` or
            ``|phi| > MAX_ABS_PHI_RAD``.
    """
    rc_in, single = _as_rows("r_cheap_km", r_cheap_km)
    vc_in, _ = _as_rows("v_cheap_km_s", v_cheap_km_s)
    rtn, _ = _as_rows("rtn_km", rtn_km)
    _check_inputs(r_cheap_km=rc_in, v_cheap_km_s=vc_in, rtn_km=rtn)
    rc, u_r, u_t, n = cheap_frame(rc_in, vc_in)

    rho = rc - rtn[:, 0]
    if np.any(rho <= 0.0):
        raise RTNDomainError("R >= |r_c|: the implied truth radius is not positive")
    du = rtn[:, 1] / rc
    phi = -rtn[:, 2] / rc
    _check_angles(du, phi)

    d = np.cos(du)[:, None] * u_r - np.sin(du)[:, None] * u_t
    out = rho[:, None] * (np.cos(phi)[:, None] * d + np.sin(phi)[:, None] * n)
    return out[0] if single else out


def _check_angles(du: np.ndarray, phi: np.ndarray) -> None:
    big_du = np.abs(du) > MAX_ABS_DU_RAD
    if np.any(big_du):
        raise RTNDomainError(
            f"|du| up to {np.abs(du).max():.4g} rad in {int(big_du.sum())} row(s), above "
            f"{MAX_ABS_DU_RAD:.4g}: along-track separation too large for an unambiguous angle"
        )
    big_phi = np.abs(phi) > MAX_ABS_PHI_RAD
    if np.any(big_phi):
        raise RTNDomainError(
            f"|phi| up to {np.abs(phi).max():.4g} rad in {int(big_phi.sum())} row(s), above "
            f"{MAX_ABS_PHI_RAD:.4g}: truth too far out of the cheap orbit plane"
        )


def rectilinear_rtn(
    r_cheap_km: np.ndarray, v_cheap_km_s: np.ndarray, r_truth_km: np.ndarray
) -> np.ndarray:
    """Rectilinear RTN, cheap minus truth: ``r_c - r_t`` projected on ``u_r, u_t, n``.

    Not a project target. Kept for comparison (tests and the practice notebook): it
    agrees with the curvilinear residual to second order in the separation, and a
    large along-track separation ``s`` leaks about ``s^2 / (2 |r|)`` into its radial
    component.
    """
    rc_in, single = _as_rows("r_cheap_km", r_cheap_km)
    vc_in, _ = _as_rows("v_cheap_km_s", v_cheap_km_s)
    rt_in, _ = _as_rows("r_truth_km", r_truth_km)
    _check_inputs(r_cheap_km=rc_in, v_cheap_km_s=vc_in, r_truth_km=rt_in)
    _, u_r, u_t, n = cheap_frame(rc_in, vc_in)
    d = rc_in - rt_in
    out = np.column_stack(
        [np.einsum("ij,ij->i", d, a) for a in (u_r, u_t, n)]
    )
    return out[0] if single else out


# --------------------------------------------------------------------------------------
# Configuration: stratification bins and the horizon grid
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualConfig:
    """Bins and horizon grid from ``config/residuals.toml`` (documented there).

    Attributes:
        status: the config's approval status.
        altitude_edges_km: increasing perigee-altitude edges.
        solar_edges_sfu: increasing edges of the trailing 81-day F10.7 mean.
        solar_missing_label: label of a window whose trailing mean is missing.
        max_horizon_s, log_first_s, log_points, uniform_points, round_to_s: grid.
    """

    status: str
    altitude_edges_km: tuple[float, ...]
    solar_edges_sfu: tuple[float, ...]
    solar_missing_label: str
    max_horizon_s: float
    log_first_s: float
    log_points: int
    uniform_points: int
    round_to_s: float

    def __post_init__(self) -> None:
        for name in ("altitude_edges_km", "solar_edges_sfu"):
            e = np.asarray(getattr(self, name), dtype=np.float64)
            if e.size == 0 or not np.all(np.isfinite(e)) or np.any(np.diff(e) <= 0):
                raise ValueError(f"{name} must be a non-empty, strictly increasing list")
        if not (0 < self.log_first_s < self.max_horizon_s):
            raise ValueError("need 0 < log_first_s < max_horizon_s")
        if self.log_points < 0 or self.uniform_points < 0 or self.log_points == 1:
            raise ValueError("log_points must be 0 or >= 2; uniform_points must be >= 0")
        if self.log_points + self.uniform_points == 0:
            raise ValueError("the grid needs at least one point")
        if not self.round_to_s > 0:
            raise ValueError("round_to_s must be positive")
        if not self.solar_missing_label:
            raise ValueError("solar_missing_label must not be empty")


def load_residual_config(path: str | Path = CONFIG_PATH) -> ResidualConfig:
    """Read ``config/residuals.toml`` (or a file with the same layout)."""
    with open(path, "rb") as f:
        t = tomllib.load(f)["residuals"]
    alt, sol, grid = t["altitude_band"], t["solar_activity_bin"], t["horizon_grid"]
    if alt.get("quantity") != "perigee_alt_km":
        raise ValueError("altitude_band.quantity must be 'perigee_alt_km'")
    if sol.get("quantity") != "asof_f107_obs_trailing81_sfu":
        raise ValueError("solar_activity_bin.quantity must be 'asof_f107_obs_trailing81_sfu'")
    return ResidualConfig(
        status=str(t["status"]),
        altitude_edges_km=tuple(float(x) for x in alt["edges_km"]),
        solar_edges_sfu=tuple(float(x) for x in sol["edges_sfu"]),
        solar_missing_label=str(sol["missing_label"]),
        max_horizon_s=float(grid["max_horizon_s"]),
        log_first_s=float(grid["log_first_s"]),
        log_points=int(grid["log_points"]),
        uniform_points=int(grid["uniform_points"]),
        round_to_s=float(grid["round_to_s"]),
    )


def horizon_grid(cfg: ResidualConfig) -> np.ndarray:
    """Times since the start at which a window is sampled (``dt_s``), SI seconds.

    Union of ``log_points`` log-spaced values from ``log_first_s`` to
    ``max_horizon_s`` and ``uniform_points`` evenly spaced values ending at
    ``max_horizon_s``, rounded to multiples of ``round_to_s``, sorted, duplicates
    removed. Whole-second values are exact in float64, so ``epoch + dt`` is the same
    number wherever it is computed.
    """
    parts = []
    if cfg.log_points:
        parts.append(np.geomspace(cfg.log_first_s, cfg.max_horizon_s, cfg.log_points))
    if cfg.uniform_points:
        step = cfg.max_horizon_s / cfg.uniform_points
        parts.append(step * np.arange(1, cfg.uniform_points + 1))
    g = np.concatenate(parts)
    g = np.round(g / cfg.round_to_s) * cfg.round_to_s
    g = np.unique(g[(g > 0) & (g <= cfg.max_horizon_s)])
    return g.astype(np.float64)


def bin_label(value: float, edges: Sequence[float]) -> str:
    """Label of the bin holding ``value``: ``"<e0"``, ``"e0-e1"``, ..., ``">=e_last"``.

    Bins are half-open, ``[e_i, e_i+1)``. Edges print without a trailing ``.0``.
    Raises ValueError for NaN (the caller decides what a missing value is labeled).
    """
    if not math.isfinite(value):
        raise ValueError(f"cannot bin {value!r}")
    fmt = [f"{e:g}" for e in edges]
    i = int(np.searchsorted(np.asarray(edges, dtype=np.float64), value, side="right"))
    if i == 0:
        return f"<{fmt[0]}"
    if i == len(edges):
        return f">={fmt[-1]}"
    return f"{fmt[i - 1]}-{fmt[i]}"


def altitude_band(perigee_alt_km: float, cfg: ResidualConfig) -> str:
    """Altitude-band label of a window from its element set's perigee altitude."""
    return bin_label(float(perigee_alt_km), cfg.altitude_edges_km)


def solar_activity_bin(f107_trailing81_sfu: float, cfg: ResidualConfig) -> str:
    """Solar-activity label from the trailing 81-day F10.7 mean known at the start."""
    v = float(f107_trailing81_sfu)
    if not math.isfinite(v):
        return cfg.solar_missing_label
    return bin_label(v, cfg.solar_edges_sfu)


# --------------------------------------------------------------------------------------
# Row builder
# --------------------------------------------------------------------------------------

#: Statuses of a window that gives no rows, in the order used to pick
#: :attr:`WindowResult.status` when several apply (all are in ``causes``).
#: truth_error: the truth job raised (``propagate_batch``'s ``"error"``).
#: sgp4_error: SGP4 flagged an error code (1-6) inside the window.
#: truth_reentry: truth fell below 120 km (or started below it).
#: sgp4_floor: SGP4 fell below the 120 km floor (``cheap_prop``).
DROP_STATUSES = ("truth_error", "sgp4_error", "truth_reentry", "sgp4_floor")
WINDOW_STATUSES = ("ok", *DROP_STATUSES)

#: Element-set columns the row builder reads.
ELSET_COLUMNS = (
    "norad_id",
    "epoch_tt_s",
    "mean_motion_rev_per_day",
    "eccentricity",
    "inclination_deg",
    "raan_deg",
    "arg_perigee_deg",
    "mean_anomaly_deg",
    "bstar_per_earth_radii",
    "mean_motion_dot",
    "perigee_alt_km",
)


class GridMismatchError(ValueError):
    """The SGP4 and truth trajectories are not on exactly the window's time grid."""


@dataclass(frozen=True)
class WindowResult:
    """What :func:`build_window_rows` made of one window.

    Attributes:
        norad_id, start_epoch_tt_s, split: the window.
        status: ``"ok"`` or the first applicable entry of :data:`DROP_STATUSES`.
        causes: every applicable drop status (empty for ``ok``).
        altitude_band, solar_activity_bin: the window's stratum (set for dropped
            windows too, so drops can be counted by stratum).
        table: the rows (passes :func:`~orbit_correction.types.validate_residual_table`),
            or None when dropped.
        sgp4_code: the SGP4 status code (0, 1-6, or the floor code).
        truth_status: the truth result's status (``ok``, ``reentry``, ``error``).
        message: detail for a dropped window.
    """

    norad_id: int
    start_epoch_tt_s: float
    split: str
    status: str
    causes: tuple[str, ...]
    altitude_band: str
    solar_activity_bin: str
    table: pa.Table | None
    sgp4_code: int
    truth_status: str
    message: str = ""

    @property
    def n_rows(self) -> int:
        return 0 if self.table is None else self.table.num_rows


def _elset_record(elset: Mapping[str, Any] | pa.Table) -> dict[str, Any]:
    if isinstance(elset, pa.Table):
        if elset.num_rows != 1:
            raise ValueError(f"elset table must have one row, got {elset.num_rows}")
        rec = elset.to_pylist()[0]
    else:
        rec = dict(elset)
    missing = [c for c in ELSET_COLUMNS if c not in rec]
    if missing:
        raise ValueError(f"element set lacks {missing}")
    return rec


def _check_dt(dt_s: object) -> np.ndarray:
    g = np.asarray(dt_s, dtype=np.float64)
    if g.ndim != 1 or g.size == 0:
        raise ValueError(f"dt_s must be a non-empty 1-D array, got shape {g.shape}")
    if not np.all(np.isfinite(g)) or np.any(g < 0):
        raise ValueError("dt_s must be finite and >= 0")
    if g.size > 1 and not np.all(np.diff(g) > 0):
        raise ValueError("dt_s must be strictly increasing")
    return g


def _check_times(name: str, got: np.ndarray, want: np.ndarray) -> None:
    if got.shape != want.shape or not np.array_equal(got, want):
        if got.shape == want.shape:
            detail = f"max |difference| {np.max(np.abs(got - want)):.3g} s"
        else:
            detail = f"{got.shape[0]} times, expected {want.shape[0]}"
        raise GridMismatchError(
            f"{name} times differ from elset epoch + dt_s ({detail}); "
            "both propagators must be sampled at exactly the same instants"
        )


def _observed_columns(
    space_weather: SpaceWeather, t_tt_s: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``obs_*`` values for each time: the day containing it, OBSERVED rows only.

    One :meth:`SpaceWeather.observed` lookup per UTC day in the window. A day with
    no row, or whose row is a forecast (not OBSERVED), gives NaN (stored as null).
    """
    days = np.array([s[:10] for s in np.atleast_1d(frames.tt_seconds_to_utc_isot(t_tt_s))])
    f107 = np.full(t_tt_s.shape, np.nan)
    ctr81 = np.full(t_tt_s.shape, np.nan)
    ap = np.full(t_tt_s.shape, np.nan)
    for day in np.unique(days):
        sel = days == day
        try:
            obs = space_weather.observed(float(t_tt_s[sel][0]))
        except LookupError:
            continue
        if obs.section != "OBSERVED":
            continue
        f107[sel], ctr81[sel], ap[sel] = obs.f107_obs_sfu, obs.f107_obs_ctr81_sfu, obs.ap_daily
    return f107, ctr81, ap


def build_window_rows(
    elset: Mapping[str, Any] | pa.Table,
    sgp4: Sgp4Result,
    truth: TruthResult,
    space_weather: SpaceWeather,
    split: str,
    dt_s: Sequence[float] | np.ndarray,
    cfg: ResidualConfig,
    *,
    include_observed: bool = False,
) -> WindowResult:
    """Residual rows for one window, or no rows and the reason.

    A window starts at the element-set epoch (v1) and is sampled at
    ``epoch_tt_s + dt_s``. Each row holds the element set's mean elements and B*,
    the four start-of-prediction space-weather values (``SpaceWeather.as_of`` at the
    epoch), optionally the ``obs_*`` values over the horizon, the window's altitude
    band and solar-activity bin, and the curvilinear RTN residual (cheap minus truth)
    at that horizon.

    Args:
        elset: the element set (a catalog-schema row as a mapping, or a one-row
            table); the columns in :data:`ELSET_COLUMNS` are read.
        sgp4: its SGP4 result from :func:`~orbit_correction.cheap_prop.propagate_sgp4_from_epoch`
            on ``dt_s`` (the altitude floor on).
        truth: the truth result sampled at the same instants.
        space_weather: the parsed space-weather file.
        split: the window's split label (one of ``SPLIT_LABELS``).
        dt_s: the horizon grid, SI seconds since the start (e.g. :func:`horizon_grid`).
        cfg: bins (:func:`load_residual_config`).
        include_observed: fill the ``obs_*`` columns (default: null, as the schema
            allows; they serve only the labeled comparison).

    Returns:
        :class:`WindowResult`. ``table`` has ``len(dt_s)`` rows for an ``ok``
        window and is None for a dropped one.

    Raises:
        GridMismatchError: a trajectory's times aren't exactly ``epoch + dt_s``
            (for a shortened trajectory, the matching leading part), or SGP4 was
            run on a grid of a different length.
        ValueError: unknown split label, or the element set, SGP4 and truth results
            belong to different objects or epochs.
        RTNDomainError: from :func:`rtn_from_states` (a residual outside the
            conversion's domain); left to the caller rather than dropped.
    """
    rec = _elset_record(elset)
    if split not in SPLIT_LABELS:
        raise ValueError(f"split must be one of {SPLIT_LABELS}, got {split!r}")
    dt = _check_dt(dt_s)
    norad = int(rec["norad_id"])
    epoch = float(rec["epoch_tt_s"])
    if sgp4.norad_id != norad or truth.norad_id != norad:
        raise ValueError(
            f"NORAD IDs differ: elset {norad}, sgp4 {sgp4.norad_id}, truth {truth.norad_id}"
        )
    if sgp4.epoch_tt_s != epoch:
        raise ValueError(f"SGP4 epoch {sgp4.epoch_tt_s} is not the element-set epoch {epoch}")

    # Exact grid: the same arithmetic as cheap_prop (base + grid), so equality is exact.
    t = epoch + dt
    if sgp4.status.n_requested != t.size:
        raise GridMismatchError(
            f"SGP4 was run on {sgp4.status.n_requested} times, the window has {t.size}"
        )
    if sgp4.trajectory is not None:
        tr = sgp4.trajectory.t_tt_s
        _check_times("SGP4", tr, t[: tr.size])
    if truth.trajectory is not None:
        tr = truth.trajectory.t_tt_s
        if tr.size > t.size:
            raise GridMismatchError(f"truth has {tr.size} times, the window has {t.size}")
        _check_times("truth", tr, t[: tr.size])

    asof = space_weather.as_of(epoch)
    band = altitude_band(float(rec["perigee_alt_km"]), cfg)
    sbin = solar_activity_bin(asof.f107_obs_trailing81_sfu, cfg)

    causes = []
    if truth.status == "error":
        causes.append("truth_error")
    elif truth.status not in ("ok", "reentry"):
        raise ValueError(f"unknown truth status {truth.status!r}")
    if not sgp4.status.ok:
        causes.append("sgp4_floor" if sgp4.status.below_floor else "sgp4_error")
    if truth.status == "reentry":
        causes.append("truth_reentry")
    if truth.status == "ok" and (truth.trajectory is None or len(truth.trajectory) != t.size):
        raise GridMismatchError("truth status is ok but its trajectory doesn't cover the window")
    common = dict(
        norad_id=norad,
        start_epoch_tt_s=epoch,
        split=split,
        altitude_band=band,
        solar_activity_bin=sbin,
        sgp4_code=int(sgp4.status.code),
        truth_status=str(truth.status),
    )
    if causes:
        causes.sort(key=DROP_STATUSES.index)
        msg = "; ".join(
            m for m in (sgp4.status.message if not sgp4.status.ok else "", truth.message) if m
        )
        return WindowResult(
            status=causes[0], causes=tuple(causes), table=None, message=msg, **common
        )

    cheap, true = sgp4.trajectory, truth.trajectory
    rtn = rtn_from_states(cheap.r_gcrf_km, cheap.v_gcrf_km_s, true.r_gcrf_km)

    n = t.size
    if include_observed:
        obs_f107, obs_ctr81, obs_ap = _observed_columns(space_weather, t)
    else:
        obs_f107 = obs_ctr81 = obs_ap = np.full(n, np.nan)
    mmdot = rec.get("mean_motion_dot")
    values: dict[str, Any] = {
        "norad_id": np.full(n, norad, dtype=np.int64),
        "split": [split] * n,
        "start_epoch_tt_s": np.full(n, epoch),
        "dt_s": dt,
        "elset_epoch_tt_s": np.full(n, epoch),
        "mean_motion_rev_per_day": np.full(n, float(rec["mean_motion_rev_per_day"])),
        "eccentricity": np.full(n, float(rec["eccentricity"])),
        "inclination_deg": np.full(n, float(rec["inclination_deg"])),
        "raan_deg": np.full(n, float(rec["raan_deg"])),
        "arg_perigee_deg": np.full(n, float(rec["arg_perigee_deg"])),
        "mean_anomaly_deg": np.full(n, float(rec["mean_anomaly_deg"])),
        "bstar_per_earth_radii": np.full(n, float(rec["bstar_per_earth_radii"])),
        "mean_motion_dot": np.full(n, np.nan if mmdot is None else float(mmdot)),
        "asof_f107_obs_sfu": np.full(n, asof.f107_obs_sfu),
        "asof_f107_obs_trailing81_sfu": np.full(n, asof.f107_obs_trailing81_sfu),
        "asof_ap_daily": np.full(n, asof.ap_daily),
        "asof_ap_3h_last": np.full(n, asof.ap_3h_last),
        "obs_f107_obs_sfu": obs_f107,
        "obs_f107_obs_ctr81_sfu": obs_ctr81,
        "obs_ap_daily": obs_ap,
        "altitude_band": [band] * n,
        "solar_activity_bin": [sbin] * n,
        "target_r_km": rtn[:, 0],
        "target_t_km": rtn[:, 1],
        "target_n_km": rtn[:, 2],
    }
    arrays = []
    for col in RESIDUAL_COLUMNS:
        v = values[col.name]
        if isinstance(v, np.ndarray) and v.dtype.kind == "f" and col.nullable:
            # missing stays missing: NaN becomes null, never a placeholder
            arrays.append(pa.array(v, type=col.dtype, mask=np.isnan(v)))
        else:
            arrays.append(pa.array(v, type=col.dtype))
    table = pa.Table.from_arrays(arrays, schema=RESIDUAL_SCHEMA)
    validate_residual_table(table)
    return WindowResult(status="ok", causes=(), table=table, **common)


def count_windows(results: Iterable[WindowResult]) -> dict[tuple[str, str, str], int]:
    """Windows per (status, altitude band, solar-activity bin), ``ok`` included.

    The drop rule's count: dropped windows are a reported metric, never silently
    lost. Filter ``results`` by split first if a per-split count is wanted.
    """
    out: dict[tuple[str, str, str], int] = {}
    for r in results:
        key = (r.status, r.altitude_band, r.solar_activity_bin)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


__all__ = [
    "CONFIG_PATH",
    "DROP_STATUSES",
    "ELSET_COLUMNS",
    "H_REL_MIN",
    "MAX_ABS_DU_RAD",
    "MAX_ABS_PHI_RAD",
    "WINDOW_STATUSES",
    "GridMismatchError",
    "RTNDomainError",
    "ResidualConfig",
    "WindowResult",
    "altitude_band",
    "bin_label",
    "build_window_rows",
    "cheap_frame",
    "count_windows",
    "horizon_grid",
    "load_residual_config",
    "rectilinear_rtn",
    "rtn_from_states",
    "solar_activity_bin",
    "states_from_rtn",
]
