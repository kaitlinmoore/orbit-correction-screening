"""SGP4 wrapper: the cheap propagator, from element sets to GCRF trajectories.

Why this module exists
----------------------
The study compares SGP4 (cheap, analytic, from public mean element sets) against a
numerical truth propagator. This module is the SGP4 side: it turns element sets as
loaded by :mod:`orbit_correction.catalog` into :class:`~orbit_correction.types.Trajectory`
objects (``source="sgp4"``) in GCRF on a TT time grid, for many objects at once.

What is library behaviour and what is ours
------------------------------------------
* The SGP4/SDP4 theory and its implementation are the ``sgp4`` package's (Brandon
  Rhodes' Python wrapper of Vallado's C++ code): ``Satrec.sgp4init`` (initialisation
  from mean elements) and :class:`sgp4.api.SatrecArray` (propagation of many objects
  in one C loop).
* The TEME -> GCRF rotation is :mod:`orbit_correction.frames` (astropy underneath).
* Ours: reading the element-set fields and converting their units for ``sgp4init``,
  the time handling (below), the batching, the per-object error status and
  truncation, and the project conventions.

Choices
-------
* **WGS-72 constants** (``sgp4.api.WGS72``). Public element sets are fitted with SGP4
  using WGS-72, so propagating them with other constants adds an error the element set
  never had; Vallado et al. (2006) recommend WGS-72 for this reason. (Measured on a
  LEO verification case: WGS-84 instead moves the position by up to 0.26 km over
  7 days.)
* **Operation mode ``'i'`` ("improved")**, the ``sgp4`` package's default and the mode
  of Vallado's verification output (``tcppver.out``). Mode ``'a'`` keeps the old AFSPC
  code's handling of a few deep-space details. Measured over 7 days for a LEO and a
  Molniya verification case, the two modes give identical positions.
* **Initialisation from the element-set fields.** The fields are converted to SGP4's
  internal units exactly as the library's TLE reader (``Satrec.twoline2rv``) converts
  them, then passed to ``Satrec.sgp4init``. The library's OMM reader
  (``sgp4.omm.initialize``) is not used: it converts mean motion and the epoch with
  different (equivalent) arithmetic, so its mean motion can differ in the last bit,
  and on two deep-space verification cases that reaches 4 mm, outside the package's
  own verification tolerance. With the TLE reader's arithmetic, every verification
  case propagates identically to ``twoline2rv`` (``tests/test_cheap_prop.py``). In
  this project's LEO regime the two readers agree to far below a millimetre.
* **Time since epoch from TT differences.** SGP4 propagates in minutes since the
  element-set epoch. The ``sgp4`` package computes them from UTC Julian dates, which
  count every UTC day as 86,400 s: across a leap second (for example
  2016-12-31T23:59:60) a UTC-date clock runs one second short, about 7.6 km of
  along-track motion in LEO. Here ``tsince = (t_TT - epoch_TT) / 60``, with the epoch
  converted from its UTC string by astropy (``catalog``'s ``epoch_tt_s`` column), so
  elapsed time is physical (SI) time. Mechanically, each private ``Satrec``'s epoch
  fields (``jdsatepoch``, ``jdsatepochF``) are reset to the epoch on our own TT clock
  before the batch is built; SGP4 reads those fields only to form ``tsince``. The
  epoch-dependent quantities set at initialisation (Greenwich sidereal time for the
  deep-space terms) are left as the library computed them.
* **Errors.** SGP4 flags a state with a nonzero code (1-4: elements or perturbed
  elements out of range; 6: the satellite has decayed, radius under one Earth radius).
  The package still returns numbers for some of these states (decayed positions keep
  shrinking). A trajectory is **truncated before the first flagged time**: it keeps
  the states before the first error and nothing after, because once SGP4 has failed
  for an object its later output isn't trustworthy even if the flag clears. If the
  first state already fails, there is no trajectory. Each object gets an
  :class:`Sgp4Status` saying what happened.
* **120 km altitude floor.** SGP4's decay flag (code 6) fires only below one Earth
  radius, so a decaying object keeps returning "valid" states far below any physical
  altitude. A trajectory is also truncated before its first state whose geodetic
  altitude (WGS-84) is below :data:`ALTITUDE_FLOOR_KM`, the same 120 km threshold at
  which the truth propagator stops (``config/truth.toml``, ``[truth.reentry]``):
  one floor for both propagators (approved 2026-11-23). Such a status carries
  :data:`ALTITUDE_FLOOR_CODE`, which is not one of SGP4's codes. Residual generation
  drops any window where either propagator stopped early (``residuals.py``).
  The altitude is computed from the TEME position (:func:`geodetic_altitude_km`), with
  no JVM and no Earth-rotation angle: the ellipsoid is symmetric about its axis, so
  only the pole matters, and TEME's pole differs from the Earth-fixed (ITRS) pole only
  by polar motion, which moves the altitude by centimetres (``tests/test_cheap_prop.py``).

Time grids
----------
* :func:`propagate_sgp4`: one absolute TT grid shared by every object (screening).
* :func:`propagate_sgp4_from_epoch`: offsets from each object's own element-set epoch
  (data generation, where every window starts at its element set's epoch). The
  returned times are ``epoch_tt_s + dt_s``; the truth propagator should be sampled at
  exactly those values.
* :func:`sgp4_epoch_states`: the single SGP4 state at each element-set epoch in GCRF,
  which the truth propagator starts from.

This module never starts a JVM.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import erfa
import numpy as np
import pyarrow as pa
from sgp4.api import SGP4_ERRORS, WGS72, Satrec, SatrecArray, jday

from orbit_correction import frames
from orbit_correction.types import Trajectory

#: Gravity constants for SGP4 (see module docstring).
GRAVITY_MODEL = WGS72
GRAVITY_MODEL_NAME = "WGS-72"
#: SGP4 operation mode: "i" (improved), as used by the sgp4 package and Vallado's tests.
OPERATION_MODE = "i"
#: ``source`` of every trajectory this module produces.
SOURCE = "sgp4"
#: Geodetic altitude (km, WGS-84) below which a trajectory is truncated. The same value
#: as the truth propagator's re-entry altitude (``config/truth.toml``); a test checks it.
ALTITUDE_FLOOR_KM = 120.0
#: :attr:`Sgp4Status.code` of a trajectory cut at the altitude floor. SGP4's own codes
#: are 1-6, so a negative code can't be mistaken for one of them.
ALTITUDE_FLOOR_CODE = -1

#: Objects propagated per SatrecArray call. Bounds memory: one chunk of a 7-day, 60 s
#: grid holds 256 x 10,081 x 6 float64 values, about 124 MB.
DEFAULT_CHUNK_OBJECTS = 256

_DAY_S = 86400.0

#: Columns of :data:`orbit_correction.catalog.ELEMENT_SET_SCHEMA` this module reads.
REQUIRED_COLUMNS = (
    "norad_id",
    "epoch_utc",
    "epoch_tt_s",
    "mean_motion_rev_per_day",
    "eccentricity",
    "inclination_deg",
    "raan_deg",
    "arg_perigee_deg",
    "mean_anomaly_deg",
    "bstar_per_earth_radii",
)

Rotation = Literal["fast", "exact"]


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sgp4Status:
    """What happened when one element set was propagated.

    Attributes:
        code: 0 if every requested state is valid; otherwise the SGP4 error code at
            the first failing time (1-4 invalid or out-of-range elements, 6 decayed),
            or :data:`ALTITUDE_FLOOR_CODE` (-1) if the trajectory was cut first at
            the altitude floor.
        message: ``"ok"``, the ``sgp4`` package's message for ``code``, or a
            description of the floor crossing.
        n_requested: number of times asked for.
        n_valid: number of states returned (the states before the first error or
            the first state below the floor).
        first_error_t_tt_s: TT seconds since J2000.0 of the first failing time (or
            first time below the floor), or None if there was none.
    """

    code: int
    message: str
    n_requested: int
    n_valid: int
    first_error_t_tt_s: float | None

    @property
    def ok(self) -> bool:
        """True when every requested state was returned."""
        return self.code == 0

    @property
    def truncated(self) -> bool:
        """True when some, but not all, requested states were returned."""
        return 0 < self.n_valid < self.n_requested

    @property
    def below_floor(self) -> bool:
        """True when the trajectory was cut at the altitude floor (not an SGP4 error)."""
        return self.code == ALTITUDE_FLOOR_CODE


@dataclass(frozen=True)
class Sgp4Result:
    """One element set's SGP4 output.

    Attributes:
        norad_id: NORAD catalog number of the element set.
        epoch_tt_s: element-set epoch, TT seconds since J2000.0.
        trajectory: the valid states in GCRF (``source="sgp4"``), or None when the
            first requested state already failed.
        status: see :class:`Sgp4Status`.
    """

    norad_id: int
    epoch_tt_s: float
    trajectory: Trajectory | None
    status: Sgp4Status


@dataclass(frozen=True)
class TemeBatch:
    """Raw SGP4 output for K element sets at N times each, before any frame change.

    Attributes:
        norad_id: shape (K,), int64.
        epoch_tt_s: shape (K,), element-set epochs, TT seconds since J2000.0.
        t_tt_s: shape (K, N), the instant of each state, TT seconds since J2000.0.
        r_teme_km, v_teme_km_s: shape (K, N, 3), TEME. Values at flagged states are
            whatever the library returned (possibly NaN) and must not be used.
        error: shape (K, N), uint8 SGP4 error code per state (0 = valid).
    """

    norad_id: np.ndarray
    epoch_tt_s: np.ndarray
    t_tt_s: np.ndarray
    r_teme_km: np.ndarray
    v_teme_km_s: np.ndarray
    error: np.ndarray


# --------------------------------------------------------------------------------------
# Element sets -> Satrec
# --------------------------------------------------------------------------------------


#: Unit conversions of the library's TLE reader (``sgp4.io.twoline2rv``), so an element
#: set read here and the same element set read from a TLE give the same ``Satrec``.
_XPDOTP = 1440.0 / (2.0 * math.pi)  # rev/day -> rad/min divisor
_DEG2RAD = math.pi / 180.0
_JD_1949_12_31 = 2433281.5  # zero of sgp4init's epoch argument (days)


def _epoch_jd(epoch_utc: str) -> tuple[float, float]:
    """OMM ``EPOCH`` (ISO 8601 UTC) to SGP4's Julian-date pair ``(jd, fraction)``.

    As in SGP4 and the TLE format, the pair counts UTC calendar days of 86,400 s; it is
    used only for SGP4's epoch bookkeeping, never for elapsed time (see the module
    docstring). A trailing ``Z`` is accepted. An epoch inside a leap second (``:60``)
    has no place on that count and raises.
    """
    s = epoch_utc.strip().removesuffix("Z")
    if s[17:19] == "60":
        raise ValueError(f"element-set epoch {epoch_utc!r} falls inside a leap second")
    e = dt.datetime.fromisoformat(s)
    if e.tzinfo is not None:
        raise ValueError(f"element-set epoch {epoch_utc!r} must be UTC without an offset")
    return jday(e.year, e.month, e.day, e.hour, e.minute, e.second + e.microsecond * 1e-6)


def _opt(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    return 0.0 if value is None else float(value)


def satrec_from_row(row: Mapping[str, Any]) -> Satrec:
    """Build an ``sgp4`` ``Satrec`` (WGS-72, mode ``'i'``) from one element-set row.

    Args:
        row: one row of a table with :data:`~orbit_correction.catalog.ELEMENT_SET_SCHEMA`
            columns (for example from ``table.to_pylist()``).

    The mean-motion derivatives are recorded on the ``Satrec`` but SGP4 doesn't use
    them in propagation; a null one is taken as 0.

    Raises:
        ValueError: if ``ephemeris_type`` is set and not 0. Only type 0 element sets are
            SGP4 mean elements; SGP4-XP (type 4) sets must not be fed to SGP4. Also for
            an epoch inside a leap second.
    """
    eph = row.get("ephemeris_type")
    if eph is not None and int(eph) != 0:
        raise ValueError(
            f"NORAD {row.get('norad_id')}: EPHEMERIS_TYPE {eph} is not an SGP4 element set"
        )
    jd, fr = _epoch_jd(str(row["epoch_utc"]))
    sat = Satrec()
    sat.sgp4init(
        GRAVITY_MODEL,
        OPERATION_MODE,
        int(row["norad_id"]),
        jd + fr - _JD_1949_12_31,
        float(row["bstar_per_earth_radii"]),
        _opt(row, "mean_motion_dot") / (_XPDOTP * 1440.0),
        _opt(row, "mean_motion_ddot") / (_XPDOTP * 1440.0 * 1440.0),
        float(row["eccentricity"]),
        float(row["arg_perigee_deg"]) * _DEG2RAD,
        float(row["inclination_deg"]) * _DEG2RAD,
        float(row["mean_anomaly_deg"]) * _DEG2RAD,
        float(row["mean_motion_rev_per_day"]) / _XPDOTP,
        float(row["raan_deg"]) * _DEG2RAD,
    )
    return sat


def _check_elsets(elsets: pa.Table) -> None:
    if not isinstance(elsets, pa.Table):
        raise TypeError(f"elsets must be a pyarrow Table, got {type(elsets).__name__}")
    missing = [c for c in REQUIRED_COLUMNS if c not in elsets.column_names]
    if missing:
        raise ValueError(f"element-set table lacks columns {missing}")


def satrecs_from_elsets(elsets: pa.Table) -> list[Satrec]:
    """One ``Satrec`` per row of an element-set table (catalog schema), in row order."""
    _check_elsets(elsets)
    return [satrec_from_row(row) for row in elsets.to_pylist()]


# --------------------------------------------------------------------------------------
# SGP4 in TEME
# --------------------------------------------------------------------------------------


def _split_days(x_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Seconds on our clock -> (whole days, fraction of a day), the pair SGP4 takes."""
    days = np.floor(x_s / _DAY_S)
    return days, (x_s - days * _DAY_S) / _DAY_S


def _check_grid(name: str, grid: object) -> np.ndarray:
    g = np.asarray(grid, dtype=np.float64)
    if g.ndim != 1 or g.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array, got shape {g.shape}")
    if not np.all(np.isfinite(g)):
        raise ValueError(f"{name} contains NaN or inf")
    if g.shape[0] > 1 and not np.all(np.diff(g) > 0):
        raise ValueError(f"{name} must be strictly increasing")
    return g


def _sgp4_teme(
    elsets: pa.Table, grid_s: np.ndarray, from_epoch: bool, chunk_objects: int
) -> TemeBatch:
    """Core: propagate every element set at ``base_i + grid_s`` (TT seconds).

    ``base_i`` is 0 (absolute grid) or the object's epoch (offsets from epoch). Each
    ``Satrec``'s epoch is moved onto a clock whose zero is ``base_i``, so SGP4's own
    ``tsince = (jd - jdsatepoch) * 1440 + (fr - jdsatepochF) * 1440`` comes out as
    ``(base_i + grid_s - epoch_TT) / 60`` minutes: physical elapsed time.
    """
    _check_elsets(elsets)
    k = elsets.num_rows
    n = grid_s.shape[0]
    norad = np.asarray(elsets.column("norad_id").to_numpy(zero_copy_only=False), dtype=np.int64)
    epoch_tt = np.asarray(
        elsets.column("epoch_tt_s").to_numpy(zero_copy_only=False), dtype=np.float64
    )
    base = epoch_tt.copy() if from_epoch else np.zeros(k)
    jd, fr = _split_days(grid_s)

    r = np.empty((k, n, 3))
    v = np.empty((k, n, 3))
    err = np.empty((k, n), dtype=np.uint8)
    rows = elsets.to_pylist()
    for start in range(0, k, chunk_objects):
        stop = min(start + chunk_objects, k)
        sats = []
        for i in range(start, stop):
            sat = satrec_from_row(rows[i])  # private copy: its epoch fields get reset
            ep_days, ep_frac = _split_days(np.array([epoch_tt[i] - base[i]]))
            sat.jdsatepoch = float(ep_days[0])
            sat.jdsatepochF = float(ep_frac[0])
            sats.append(sat)
        e, rr, vv = SatrecArray(sats).sgp4(jd, fr)
        r[start:stop], v[start:stop], err[start:stop] = rr, vv, e
    t_tt = base[:, None] + grid_s[None, :]
    return TemeBatch(norad, epoch_tt, t_tt, r, v, err)


def sgp4_teme(elsets: pa.Table, times_tt_s: Sequence[float] | np.ndarray,
              *, chunk_objects: int = DEFAULT_CHUNK_OBJECTS) -> TemeBatch:
    """Raw SGP4 TEME states of every element set on one shared absolute TT grid."""
    grid = _check_grid("times_tt_s", times_tt_s)
    return _sgp4_teme(elsets, grid, from_epoch=False, chunk_objects=chunk_objects)


def sgp4_teme_from_epoch(elsets: pa.Table, dt_s: Sequence[float] | np.ndarray,
                         *, chunk_objects: int = DEFAULT_CHUNK_OBJECTS) -> TemeBatch:
    """Raw SGP4 TEME states at ``dt_s`` SI seconds after each element set's own epoch."""
    grid = _check_grid("dt_s", dt_s)
    return _sgp4_teme(elsets, grid, from_epoch=True, chunk_objects=chunk_objects)


# --------------------------------------------------------------------------------------
# GCRF trajectories
# --------------------------------------------------------------------------------------


def _first_true(mask: np.ndarray) -> np.ndarray:
    """Index of the first True in each row, or the row length if none."""
    return np.where(mask.any(axis=1), mask.argmax(axis=1), mask.shape[1])


def _first_errors(error: np.ndarray) -> np.ndarray:
    """Index of the first nonzero code in each row, or the row length if none."""
    return _first_true(error != 0)


def geodetic_altitude_km(r_teme_km: np.ndarray) -> np.ndarray:
    """Geodetic altitude above the WGS-84 ellipsoid of TEME positions, km.

    Uses ERFA's ``gc2gd`` (WGS-84) on the TEME position as if it were Earth-fixed.
    That is exact up to polar motion: geodetic altitude depends only on the distance
    from the ellipsoid's axis and along it, so the Earth-rotation angle drops out, and
    TEME's z axis (the true pole of date) differs from the ITRS pole only by polar
    motion (~0.5 arcsec), which moves the altitude by centimetres. No JVM, no EOP.

    Args:
        r_teme_km: shape (..., 3), finite TEME positions, km.

    Returns:
        shape (...), altitude in km.
    """
    r = np.asarray(r_teme_km, dtype=np.float64)
    if r.shape[-1:] != (3,):
        raise ValueError(f"r_teme_km must have shape (..., 3), got {r.shape}")
    if not np.all(np.isfinite(r)):
        raise ValueError("r_teme_km contains NaN or inf")
    _, _, height_m = erfa.gc2gd(1, r * 1000.0)  # 1 = WGS-84
    return np.asarray(height_m, dtype=np.float64) / 1000.0


def _first_below_floor(batch: TemeBatch, n_err: np.ndarray, floor_km: float) -> np.ndarray:
    """Index of each row's first state below the floor among its unflagged states."""
    k, n = batch.error.shape
    unflagged = np.arange(n)[None, :] < n_err[:, None]
    below = np.zeros((k, n), dtype=bool)
    if unflagged.any():
        below[unflagged] = geodetic_altitude_km(batch.r_teme_km[unflagged]) < floor_km
    return _first_true(below)


def _to_results(
    batch: TemeBatch, rotation: Rotation, floor_km: float | None = ALTITUDE_FLOOR_KM
) -> list[Sgp4Result]:
    if rotation not in ("fast", "exact"):
        raise ValueError(f"rotation must be 'fast' or 'exact', got {rotation!r}")
    k, n = batch.error.shape
    n_err = _first_errors(batch.error)
    if floor_km is None:
        n_floor = np.full(k, n)
    else:
        n_floor = _first_below_floor(batch, n_err, float(floor_km))
    # Cut at whichever comes first; on the same state, SGP4's own flag wins (a flagged
    # state's position isn't trustworthy enough to place it against the floor).
    n_valid = np.minimum(n_err, n_floor)

    # Rotate every valid state in one call; the fast path's cost is then dominated by
    # the node rotations it hasn't cached yet.
    keep = np.arange(n)[None, :] < n_valid[:, None]
    r_g = np.full(batch.r_teme_km.shape, np.nan)
    v_g = np.full(batch.v_teme_km_s.shape, np.nan)
    if keep.any():
        convert = frames.teme_to_gcrf_fast if rotation == "fast" else frames.teme_to_gcrf
        rr, vv = convert(batch.r_teme_km[keep], batch.v_teme_km_s[keep], batch.t_tt_s[keep])
        r_g[keep], v_g[keep] = rr, vv

    out = []
    for i in range(k):
        m = int(n_valid[i])
        if m == n:
            status = Sgp4Status(0, "ok", n, n, None)
        elif n_floor[i] < n_err[i]:
            status = Sgp4Status(
                ALTITUDE_FLOOR_CODE,
                f"geodetic altitude below the {floor_km:g} km floor",
                n,
                m,
                float(batch.t_tt_s[i, m]),
            )
        else:
            code = int(batch.error[i, m])
            status = Sgp4Status(
                code, SGP4_ERRORS.get(code, f"SGP4 error {code}"), n, m, float(batch.t_tt_s[i, m])
            )
        traj = None
        if m > 0:
            traj = Trajectory(
                norad_id=int(batch.norad_id[i]),
                t_tt_s=np.ascontiguousarray(batch.t_tt_s[i, :m]),
                r_gcrf_km=np.ascontiguousarray(r_g[i, :m]),
                v_gcrf_km_s=np.ascontiguousarray(v_g[i, :m]),
                source=SOURCE,
            )
        out.append(Sgp4Result(int(batch.norad_id[i]), float(batch.epoch_tt_s[i]), traj, status))
    return out


def propagate_sgp4(
    elsets: pa.Table,
    times_tt_s: Sequence[float] | np.ndarray,
    *,
    rotation: Rotation = "fast",
    chunk_objects: int = DEFAULT_CHUNK_OBJECTS,
    floor_km: float | None = ALTITUDE_FLOOR_KM,
) -> list[Sgp4Result]:
    """Propagate every element set with SGP4 onto one shared TT grid, in GCRF.

    Args:
        elsets: element sets with the catalog schema (``catalog.ELEMENT_SET_SCHEMA``;
            the columns in :data:`REQUIRED_COLUMNS` are read).
        times_tt_s: shape (N,), strictly increasing, TT seconds since J2000.0. The
            grid may start before or after each element set's epoch.
        rotation: ``"fast"`` (:func:`frames.teme_to_gcrf_fast`, under a micrometre from
            the exact path) or ``"exact"`` (:func:`frames.teme_to_gcrf`).
        chunk_objects: element sets per SGP4 batch (memory bound only; results do not
            depend on it).
        floor_km: geodetic altitude floor (default :data:`ALTITUDE_FLOOR_KM`); None
            turns it off and keeps only SGP4's own error test.

    Returns:
        One :class:`Sgp4Result` per row, in row order. Trajectories are truncated
        before the first SGP4 error or the first state below the floor, whichever
        comes first (see the module docstring).
    """
    batch = sgp4_teme(elsets, times_tt_s, chunk_objects=chunk_objects)
    return _to_results(batch, rotation, floor_km)


def propagate_sgp4_from_epoch(
    elsets: pa.Table,
    dt_s: Sequence[float] | np.ndarray,
    *,
    rotation: Rotation = "fast",
    chunk_objects: int = DEFAULT_CHUNK_OBJECTS,
    floor_km: float | None = ALTITUDE_FLOOR_KM,
) -> list[Sgp4Result]:
    """Propagate each element set to ``dt_s`` SI seconds after its own epoch, in GCRF.

    Same as :func:`propagate_sgp4` except for the grid: object i's trajectory times
    are ``epoch_tt_s[i] + dt_s``. Use for windows that start at the element-set epoch.
    """
    batch = sgp4_teme_from_epoch(elsets, dt_s, chunk_objects=chunk_objects)
    return _to_results(batch, rotation, floor_km)


def sgp4_epoch_states(elsets: pa.Table) -> list[Sgp4Result]:
    """The SGP4 state at each element-set epoch, in GCRF (exact rotation), as N = 1.

    This is the state the truth propagator starts from. Converting the SGP4 mean
    elements to an osculating state this way is the project's deliberate, logged
    initialisation (see ``docs/methodology.md``). A state already below the altitude
    floor gives no trajectory (status code :data:`ALTITUDE_FLOOR_CODE`).
    """
    return propagate_sgp4_from_epoch(elsets, np.zeros(1), rotation="exact")


def results_ok(results: Sequence[Sgp4Result]) -> list[Trajectory]:
    """The trajectories of the results whose every requested state is valid."""
    return [r.trajectory for r in results if r.status.ok and r.trajectory is not None]


def status_counts(results: Sequence[Sgp4Result]) -> dict[str, int]:
    """Counts of ok, truncated and empty results, plus counts per cause.

    Causes are keyed ``code_<n>`` for SGP4's error codes and ``altitude_floor`` for
    trajectories cut at the floor. Failed propagations are a metric to report, never
    something to drop silently.
    """
    counts: dict[str, int] = {"ok": 0, "truncated": 0, "empty": 0}
    for r in results:
        s = r.status
        if s.ok:
            counts["ok"] += 1
            continue
        counts["truncated" if s.n_valid else "empty"] += 1
        key = "altitude_floor" if s.below_floor else f"code_{s.code}"
        counts[key] = counts.get(key, 0) + 1
    return counts


__all__ = [
    "ALTITUDE_FLOOR_CODE",
    "ALTITUDE_FLOOR_KM",
    "DEFAULT_CHUNK_OBJECTS",
    "GRAVITY_MODEL",
    "GRAVITY_MODEL_NAME",
    "OPERATION_MODE",
    "REQUIRED_COLUMNS",
    "SOURCE",
    "Sgp4Result",
    "Sgp4Status",
    "TemeBatch",
    "geodetic_altitude_km",
    "propagate_sgp4",
    "propagate_sgp4_from_epoch",
    "results_ok",
    "satrec_from_row",
    "satrecs_from_elsets",
    "sgp4_epoch_states",
    "sgp4_teme",
    "sgp4_teme_from_epoch",
    "status_counts",
]
