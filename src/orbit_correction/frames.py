"""TEME -> GCRF conversion and the time-scale helpers the contracts need.

Why this module exists
----------------------
SGP4 returns positions and velocities in TEME ("true equator, mean equinox"), a
frame tied to the date of each state. Everything that crosses a component
boundary in this project is in GCRF with times in TT seconds since J2000.0
(see :mod:`orbit_correction.types`). This module is the one place where SGP4's
output and element-set epochs (UTC) are turned into those conventions.

What is library behaviour and what is ours
------------------------------------------
* The frame rotations are astropy's (``TEME`` -> ``ITRS`` -> ``CIRS`` ->
  ``GCRS``, IAU 2006/2000A precession-nutation with IERS Earth orientation).
  Velocities go through astropy's finite-difference handling of the
  time-dependent rotations. Astropy's ``GCRS`` with the default geocentric
  observer has GCRF axes, so it is used as GCRF here.
* Time-scale arithmetic (UTC, TAI, TT, UT1, leap seconds) is astropy/ERFA's.
* What this module adds: the project conventions (TT seconds since J2000.0,
  km, km/s), vectorised wrappers, and pinning of Earth orientation data so a
  conversion gives the same answer offline and next year.

Earth orientation data (EOP) and reproducibility
-------------------------------------------------
TEME -> GCRF needs UT1-UTC and polar motion. By default astropy downloads a
fresh IERS-A table when its copy is older than a week, which would make results
depend on the day they were computed. On import this module:

* turns off astropy's IERS auto-download (``iers.conf.auto_download = False``),
  so astropy uses the IERS-A (``finals2000A.all``) and IERS-B tables bundled in
  the ``astropy-iers-data`` package, whose version is pinned in ``uv.lock``;
* checks every converted time against the table's coverage itself and raises
  :class:`EOPCoverageError` outside it. (Astropy's own
  ``iers_degraded_accuracy = "error"`` setting is also set, but in testing it did
  not stop a 2040 conversion from silently falling back to mean polar motion.)

Inside the table, the last stretch is IERS *predictions* (flag ``P``), not
measurements. Historical epochs are fully measured; conversions at recent or
current epochs use predicted UT1-UTC, which is pinned (so reproducible) but
less accurate. :func:`eop_provenance` reports where measurements end.

How much the EOP choice matters here: very little. UT1-UTC and polar motion
enter TEME -> ITRS and leave again in ITRS -> GCRS, so they cancel: 100 ms of
UT1 error moves the GCRF result by about 5 micrometres, while the Earth-fixed
(ITRS) position moves about 50 m (``tests/test_frames.py`` and
``practice/teme_gcrf.ipynb``). Astropy does not apply the celestial pole
offsets dX/dY in this chain. EOP matters for Earth-fixed quantities, for
example the truth propagator's gravity field and atmosphere, which Orekit
handles with its own EOP data. Pinning still matters here for bit-for-bit
reproducibility and for the coverage check.

:func:`eop_provenance` reports which tables are in use. :func:`use_eop_file`
switches to another ``finals2000A``-format file (for example the one in the
pinned orekit-data bundle, so the truth propagator and this module share EOP);
whether to do that is an open project decision.

This module never starts a JVM.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import erfa
import numpy as np
from astropy import units as u
from astropy.coordinates import GCRS, TEME, CartesianDifferential, CartesianRepresentation
from astropy.time import Time
from astropy.utils import iers

from orbit_correction.types import J2000_JD_TT

# --------------------------------------------------------------------------------------
# EOP configuration (runs on import)
# --------------------------------------------------------------------------------------

_EOP_SOURCE: dict[str, str] = {}


def configure_offline_eop() -> None:
    """Make astropy use its bundled, version-pinned EOP tables and never download.

    Called on import. Safe to call again (for example after a test changed the
    configuration).
    """
    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "error"
    table = iers.IERS_Auto.open()
    iers.earth_orientation_table.set(table)
    _EOP_SOURCE.clear()
    _EOP_SOURCE.update({"kind": "astropy-iers-data bundled IERS-A + IERS-B"})


def use_eop_file(path: str | Path) -> None:
    """Use a specific ``finals2000A``-format IERS-A file as the EOP table.

    Args:
        path: a ``finals2000A.all`` (or ``finals2000A.data``) file, for example
            the one in the pinned orekit-data bundle.

    Times outside the file's coverage raise :class:`EOPCoverageError`.
    """
    path = Path(path)
    iers.conf.auto_download = False
    iers.conf.iers_degraded_accuracy = "error"
    table = iers.IERS_A.open(str(path))
    iers.earth_orientation_table.set(table)
    _EOP_SOURCE.clear()
    _EOP_SOURCE.update(
        {"kind": "finals2000A file", "path": str(path), "sha256": _sha256(path)}
    )


class EOPCoverageError(ValueError):
    """A time lies outside the Earth orientation table in use."""


def _check_eop_coverage(t: Time) -> None:
    """Raise :class:`EOPCoverageError` if any time is outside the EOP table."""
    table = iers.earth_orientation_table.get()
    mjd = np.asarray(table["MJD"].to_value(u.d))
    lo, hi = float(mjd.min()), float(mjd.max())
    t_mjd = np.atleast_1d(t.utc.mjd)
    if np.any(t_mjd < lo) or np.any(t_mjd > hi):
        raise EOPCoverageError(
            f"time(s) MJD {t_mjd.min():.3f}-{t_mjd.max():.3f} outside the EOP table "
            f"(MJD {lo:.1f}-{hi:.1f}); see frames.eop_provenance()"
        )


def eop_provenance() -> dict[str, str]:
    """Describe the EOP data in use: package versions, file, hash and coverage."""
    import astropy
    import astropy_iers_data

    out = dict(_EOP_SOURCE)
    out["astropy"] = astropy.__version__
    out["astropy_iers_data"] = astropy_iers_data.__version__
    if out.get("kind", "").startswith("astropy-iers-data"):
        out["iers_a_file"] = str(iers.IERS_A_FILE)
        out["iers_a_sha256"] = _sha256(Path(iers.IERS_A_FILE))
    table = iers.earth_orientation_table.get()
    mjd = np.asarray(table["MJD"].to_value(u.d))
    out["first_mjd"] = f"{mjd.min():.1f}"
    out["last_mjd"] = f"{mjd.max():.1f}"
    if "UT1Flag" in table.colnames:
        measured = mjd[np.asarray(table["UT1Flag"]) != "P"]
        out["last_measured_mjd"] = f"{measured.max():.1f}"
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


configure_offline_eop()

# --------------------------------------------------------------------------------------
# Time scales
# --------------------------------------------------------------------------------------

#: Seconds per day, for JD <-> seconds.
_DAY_S = 86400.0


def time_to_tt_seconds(t: Time) -> np.ndarray | float:
    """Convert an astropy ``Time`` (any scale) to TT seconds since J2000.0.

    Uses astropy's two-double Julian date so no precision is lost before the
    subtraction (float64 TT seconds near 2026 resolve about 0.1 microsecond).
    """
    tt = t.tt
    s = ((tt.jd1 - J2000_JD_TT) + tt.jd2) * _DAY_S
    return float(s) if np.ndim(s) == 0 else np.asarray(s, dtype=np.float64)


def tt_seconds_to_time(t_tt_s: float | np.ndarray) -> Time:
    """Convert TT seconds since J2000.0 to an astropy ``Time`` in the TT scale."""
    s = np.asarray(t_tt_s, dtype=np.float64)
    whole_days = np.floor(s / _DAY_S)
    frac = (s - whole_days * _DAY_S) / _DAY_S
    return Time(J2000_JD_TT + whole_days, frac, format="jd", scale="tt")


def utc_to_tt_seconds(utc: str | list[str] | np.ndarray | Time) -> np.ndarray | float:
    """Convert UTC instants to TT seconds since J2000.0.

    Args:
        utc: ISO 8601 string(s) in UTC (``2016-12-31T23:59:60`` is a valid leap
            second), or an astropy ``Time``.
    """
    t = utc if isinstance(utc, Time) else Time(utc, format="isot", scale="utc")
    return time_to_tt_seconds(t)


def tt_seconds_to_utc_isot(t_tt_s: float | np.ndarray) -> str | np.ndarray:
    """Convert TT seconds since J2000.0 to UTC ISO strings (microsecond precision)."""
    t = tt_seconds_to_time(t_tt_s).utc
    t.precision = 6
    return t.isot


def tai_minus_utc_s(utc: str | list[str] | Time) -> np.ndarray | float:
    """Leap-second offset TAI - UTC (Delta AT) at the given UTC instant(s), seconds.

    Read from ERFA's leap-second table for the UTC calendar date. During a leap
    second (``23:59:60``) the offset is still the old one; it steps at 00:00:00.
    """
    t = utc if isinstance(utc, Time) else Time(utc, format="isot", scale="utc")
    ymd = np.atleast_1d(t.utc.ymdhms)
    frac = (ymd["hour"] * 3600.0 + ymd["minute"] * 60.0 + ymd["second"]) / _DAY_S
    dat = erfa.dat(ymd["year"], ymd["month"], ymd["day"], np.clip(frac, 0.0, 0.999999))
    dat = np.asarray(dat, dtype=np.float64)
    return float(dat[0]) if t.isscalar else dat


def tt_minus_utc_s(utc: str | Time) -> np.ndarray | float:
    """TT - UTC = 32.184 s + Delta AT at the given UTC instant(s), seconds."""
    return tai_minus_utc_s(utc) + 32.184


def elset_epoch_to_tt_seconds(epoch_utc: str | list[str] | np.ndarray) -> np.ndarray | float:
    """Element-set epoch (OMM ``EPOCH``, ISO 8601 UTC) to TT seconds since J2000.0.

    Element-set epochs are UTC by definition (OMM ``TIME_SYSTEM`` is UTC).
    """
    return utc_to_tt_seconds(epoch_utc)


def satrec_epoch_to_tt_seconds(jdsatepoch: float, jdsatepoch_f: float) -> float:
    """Epoch of an ``sgp4.api.Satrec`` (its UTC Julian date pair) to TT seconds.

    SGP4's Julian date (``sgp4.api.jday``, the TLE reader) counts every UTC day as
    86,400 s: the fraction of the day is ``seconds since 00:00 UTC / 86400`` even on a
    day that ends with a leap second. Astropy reads a UTC Julian date differently: on
    a leap-second day it stretches the fraction over 86,401 s. Handing SGP4's pair to
    ``Time(..., format="jd", scale="utc")`` therefore puts an epoch on 2016-12-31 up to
    a second late (0.5 s at 12:00). Here the pair is split into the UTC midnight that
    starts the day (an unambiguous instant) and the seconds after it, which are SI
    seconds because the leap second, when there is one, comes at the end of the day.

    An SGP4 epoch can't fall inside a leap second (the 86,400-second count has no place
    for it), so every value of the pair maps to exactly one instant.
    """
    total = float(jdsatepoch) + float(jdsatepoch_f)
    jd_midnight = math.floor(total - 0.5) + 0.5  # the UTC midnight starting the day
    # (jd - midnight) is a small whole number of days when jd is a midnight, as SGP4's
    # pair has it, so adding the fraction last keeps its precision.
    seconds = ((float(jdsatepoch) - jd_midnight) + float(jdsatepoch_f)) * _DAY_S
    midnight = Time(jd_midnight, format="jd", scale="utc")
    return float(time_to_tt_seconds(midnight)) + seconds


# --------------------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------------------


def _as_states(r: np.ndarray, v: np.ndarray, t_tt_s: float | np.ndarray):
    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    single = r.ndim == 1
    r2 = np.atleast_2d(r)
    v2 = np.atleast_2d(v)
    if r2.shape[-1] != 3 or v2.shape != r2.shape:
        raise ValueError(f"r and v must both have shape (3,) or (N, 3); got {r.shape}, {v.shape}")
    t = np.asarray(t_tt_s, dtype=np.float64)
    if t.ndim > 1 or (t.ndim == 1 and t.shape[0] != r2.shape[0]):
        raise ValueError(f"t_tt_s must be a scalar or have shape ({r2.shape[0]},), got {t.shape}")
    return r2, v2, tt_seconds_to_time(t), single


def _transform(r, v, t_tt_s, src_cls, dst_cls) -> tuple[np.ndarray, np.ndarray]:
    r2, v2, obstime, single = _as_states(r, v, t_tt_s)
    _check_eop_coverage(obstime)
    rep = CartesianRepresentation(
        r2.T * u.km, differentials=CartesianDifferential(v2.T * (u.km / u.s))
    )
    out = src_cls(rep, obstime=obstime).transform_to(dst_cls(obstime=obstime))
    r_out = out.cartesian.xyz.to_value(u.km).T
    v_out = out.velocity.d_xyz.to_value(u.km / u.s).T
    r_out = np.ascontiguousarray(r_out, dtype=np.float64)
    v_out = np.ascontiguousarray(v_out, dtype=np.float64)
    if single:
        return r_out[0], v_out[0]
    return r_out, v_out


def teme_to_gcrf(
    r_teme_km: np.ndarray, v_teme_km_s: np.ndarray, t_tt_s: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate SGP4 TEME states into GCRF.

    Args:
        r_teme_km: shape (3,) or (N, 3), TEME position in km (SGP4 output).
        v_teme_km_s: same shape, TEME velocity in km/s.
        t_tt_s: the instant of each state, TT seconds since J2000.0; a scalar or
            shape (N,).

    Returns:
        (r_gcrf_km, v_gcrf_km_s) with the input's shape.

    Accuracy: against Vallado et al. (2006) Appendix C, 0.12 m and 0.11 mm/s
    (see ``tests/test_frames.py``), set mainly by the EOP and precession-nutation
    model differences between that paper and astropy.
    """
    return _transform(r_teme_km, v_teme_km_s, t_tt_s, TEME, GCRS)


def gcrf_to_teme(
    r_gcrf_km: np.ndarray, v_gcrf_km_s: np.ndarray, t_tt_s: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`teme_to_gcrf` (same arguments and shapes, GCRF in, TEME out)."""
    return _transform(r_gcrf_km, v_gcrf_km_s, t_tt_s, GCRS, TEME)


# --------------------------------------------------------------------------------------
# Fast TEME -> GCRF (interpolated rotation)
# --------------------------------------------------------------------------------------
#
# Why: the exact path above costs about 0.3-0.7 ms per state (astropy builds the full
# precession-nutation and Earth-rotation chain for every instant), which would dominate
# catalog-scale SGP4 generation. TEME -> GCRF is a pure rotation, r_gcrf = M(t) r_teme,
# and M(t) varies slowly: Earth rotation enters TEME -> ITRS and leaves again in
# ITRS -> GCRS, so what is left is precession, nutation and the small GMST/ERA
# difference, with no term faster than a few days. So:
#
# * M is computed with the exact path (astropy) at nodes on a fixed lattice in TT,
#   t_k = k * FAST_NODE_STEP_S (k an integer, TT seconds since J2000.0), by rotating
#   the three TEME basis vectors;
# * between nodes, each element of M is interpolated with the degree-5 Lagrange
#   polynomial through the FAST_STENCIL nearest nodes (three on each side of the
#   6-hour interval holding the time). Each interval's polynomial is turned into
#   monomial coefficients once per call and evaluated per state with Horner's rule;
# * velocity is v_gcrf = M v_teme + dM/dt r_teme, with dM/dt the derivative of the
#   same polynomial. The exact path gets its dM/dt term from astropy's finite
#   differences; the two agree to about 1 micrometre/s.
#
# The lattice is fixed (it doesn't depend on the times asked for) and the interpolant is
# local (it uses only nearby nodes), so a state's result depends only on its own time:
# batched and single-object calls agree bit for bit, and node rotations can be cached
# and shared across calls. Measured against the exact path over 7 days at 60 s
# (``tests/test_frames.py``): under 1 micrometre in position and 1 micrometre/s in
# velocity at 2005, 2016 (across the leap second) and 2024 epochs.
#
# Behaviour kept identical to the exact path: times are TT, so leap seconds need no
# special handling (nodes are computed by the exact path, which handles UTC
# internally); a time outside the EOP table raises EOPCoverageError just as
# teme_to_gcrf does; a state whose stencil would reach a node outside the table (within
# about 18 hours of either end of the table) is converted with the exact path instead.
# Changing the EOP table (use_eop_file, configure_offline_eop) empties the node cache.

#: Spacing of the interpolation nodes, seconds (TT). 6 hours.
FAST_NODE_STEP_S = 21600.0
#: Number of nodes in each interpolation stencil (degree-5 Lagrange polynomial).
FAST_STENCIL = 6
# States per evaluation chunk (bounds the temporary arrays to tens of MB).
_FAST_CHUNK = 1 << 18

# Node cache: lattice index k -> 3x3 rotation M(k * FAST_NODE_STEP_S), flattened to 9
# values (row-major). Valid only for the EOP table object it was computed with (kept in
# _NODE_CACHE_TABLE, compared by identity, so the cache is dropped as soon as a
# different table is set). Mutating a table in place is not detected.
_NODE_CACHE: dict[int, np.ndarray] = {}
_NODE_CACHE_TABLE: list[object] = []


def _lagrange_monomials(p: int) -> np.ndarray:
    """Monomial coefficients of the Lagrange basis on centred nodes.

    Nodes sit at ``y_j = j - (p - 1) / 2`` (for p = 6: -2.5, -1.5, ..., 2.5, in units of
    the node step, measured from the middle of the interval being interpolated).
    Returns ``L`` with shape (p, p): basis polynomial j is ``sum_d L[d, j] * y**d``.
    Centring keeps the coefficients well conditioned on the interval ``|y| <= 0.5``.
    """
    y = np.arange(p) - (p - 1) / 2.0
    out = np.empty((p, p))
    for j in range(p):
        others = np.delete(y, j)
        out[:, j] = np.polynomial.polynomial.polyfromroots(others) / np.prod(y[j] - others)
    return out


_LAGRANGE = _lagrange_monomials(FAST_STENCIL)


def clear_fast_rotation_cache() -> None:
    """Forget all cached node rotations used by :func:`teme_to_gcrf_fast`."""
    _NODE_CACHE.clear()
    _NODE_CACHE_TABLE.clear()


def _eop_tt_bounds() -> tuple[float, float]:
    """First and last instant of the EOP table in use, as TT seconds since J2000.0."""
    table = iers.earth_orientation_table.get()
    mjd = np.asarray(table["MJD"].to_value(u.d))
    lo, hi = time_to_tt_seconds(Time([mjd.min(), mjd.max()], format="mjd", scale="utc"))
    return float(lo), float(hi)


def _node_rotations(k: np.ndarray) -> np.ndarray:
    """Exact TEME -> GCRF rotations at lattice nodes ``k``, shape (len(k), 9), row-major.

    Uses (and fills) the node cache. Column j of each matrix is the GCRF image of the
    TEME basis vector e_j, computed with :func:`teme_to_gcrf`.
    """
    table = iers.earth_orientation_table.get()
    if not _NODE_CACHE_TABLE or _NODE_CACHE_TABLE[0] is not table:
        _NODE_CACHE.clear()
        _NODE_CACHE_TABLE[:] = [table]
    missing = [int(x) for x in k if int(x) not in _NODE_CACHE]
    if missing:
        n = len(missing)
        basis = np.tile(np.eye(3), (n, 1))  # rows e_x, e_y, e_z for each node
        t_nodes = np.repeat(np.asarray(missing, dtype=np.float64) * FAST_NODE_STEP_S, 3)
        r, _ = teme_to_gcrf(basis, np.zeros_like(basis), t_nodes)
        mats = r.reshape(n, 3, 3).transpose(0, 2, 1)  # image of e_j becomes column j
        for kk, m in zip(missing, mats, strict=True):
            _NODE_CACHE[kk] = np.ascontiguousarray(m).reshape(9)
    return np.stack([_NODE_CACHE[int(x)] for x in k])


class _IntervalPolynomials:
    """Monomial coefficients of the interpolating polynomial for each needed interval.

    Interval ``k0`` is ``[k0 h, (k0 + 1) h)``; its stencil is nodes ``k0 - 2 ... k0 + 3``.
    ``coef[i, d]`` (9 values) multiplies ``y**d``, with ``y = t / h - k0 - 0.5``.
    """

    def __init__(self, k0: np.ndarray):
        p, half = FAST_STENCIL, FAST_STENCIL // 2
        self.k_lo = int(k0.min())
        span = int(k0.max()) - self.k_lo + 1
        present = np.zeros(span, dtype=bool)
        present[k0 - self.k_lo] = True
        intervals = np.flatnonzero(present) + self.k_lo
        self.slot = np.full(span, -1, dtype=np.int64)
        self.slot[intervals - self.k_lo] = np.arange(intervals.size)

        stencils = intervals[:, None] + np.arange(1 - half, half + 1)  # (n_int, p)
        nodes_k, inverse = np.unique(stencils, return_inverse=True)
        nodes = _node_rotations(nodes_k)[inverse.reshape(stencils.shape)]  # (n_int, p, 9)
        coef = np.zeros((intervals.size, p, 9))
        for d in range(p):  # explicit sums: identical arithmetic in every call
            for j in range(p):
                coef[:, d] += _LAGRANGE[d, j] * nodes[:, j]
        self.coef = coef

    def evaluate(self, t: np.ndarray, k0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(M, dM/dt) at times ``t`` in intervals ``k0``, each shape (len(t), 9)."""
        h, p = FAST_NODE_STEP_S, FAST_STENCIL
        idx = self.slot[k0 - self.k_lo]
        y = (t / h - k0 - 0.5)[:, None]
        m = self.coef[idx, p - 1].copy()
        dm = (p - 1) * self.coef[idx, p - 1]
        for d in range(p - 2, -1, -1):
            m = m * y + self.coef[idx, d]
            if d >= 1:
                dm = dm * y + d * self.coef[idx, d]
        return m, dm / h


def teme_to_gcrf_rotation_fast(t_tt_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interpolated TEME -> GCRF rotation and its time derivative at each time.

    Args:
        t_tt_s: shape (N,), TT seconds since J2000.0. Every stencil node must lie
            inside the EOP table; :func:`teme_to_gcrf_fast` handles the table's edges.

    Returns:
        (M, M_dot): shapes (N, 3, 3), with ``r_gcrf = M @ r_teme`` and
        ``v_gcrf = M @ v_teme + M_dot @ r_teme`` (M_dot in 1/s).
    """
    t = np.asarray(t_tt_s, dtype=np.float64)
    k0 = np.floor(t / FAST_NODE_STEP_S).astype(np.int64)
    m, m_dot = _IntervalPolynomials(k0).evaluate(t, k0)
    return m.reshape(-1, 3, 3), m_dot.reshape(-1, 3, 3)


def _apply_rotation(m: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``M[n] @ x[k, n]`` for every k and n, with M given row-major as shape (N, 9).

    Written out rather than with einsum or matmul so the arithmetic for one state is
    the same whatever the array sizes (BLAS and einsum may reorder the sums).
    """
    out = np.empty(x.shape)
    for i in range(3):
        out[..., i] = (
            m[:, 3 * i] * x[..., 0] + m[:, 3 * i + 1] * x[..., 1] + m[:, 3 * i + 2] * x[..., 2]
        )
    return out


def teme_to_gcrf_fast(
    r_teme_km: np.ndarray, v_teme_km_s: np.ndarray, t_tt_s: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate SGP4 TEME states into GCRF with an interpolated rotation.

    Same conventions and results as :func:`teme_to_gcrf` to well under a millimetre
    (method and measured accuracy in the comment block above), at a small fraction of
    the cost. Accepts the same shapes, plus a block of states for many objects that
    share one time grid.

    Args:
        r_teme_km: shape (3,), (N, 3), or (K, N, 3) for K objects on the same N times.
        v_teme_km_s: same shape, km/s.
        t_tt_s: TT seconds since J2000.0; a scalar, or shape (N,).

    Returns:
        (r_gcrf_km, v_gcrf_km_s) with the input's shape.

    Raises:
        EOPCoverageError: if any time is outside the EOP table (as the exact path).
        ValueError: on inconsistent shapes or non-finite times.
    """
    r = np.asarray(r_teme_km, dtype=np.float64)
    v = np.asarray(v_teme_km_s, dtype=np.float64)
    if r.shape != v.shape or r.ndim not in (1, 2, 3) or r.shape[-1] != 3:
        raise ValueError(
            f"r and v must both have shape (3,), (N, 3) or (K, N, 3); got {r.shape}, {v.shape}"
        )
    single = r.ndim == 1
    r3 = r.reshape(1, 1, 3) if single else (r[None] if r.ndim == 2 else r)
    v3 = v.reshape(1, 1, 3) if single else (v[None] if v.ndim == 2 else v)
    n = r3.shape[1]
    t = np.asarray(t_tt_s, dtype=np.float64)
    if t.ndim == 0:
        t = np.full(n, float(t))
    if t.ndim != 1 or t.shape[0] != n:
        raise ValueError(f"t_tt_s must be a scalar or have shape ({n},), got {t.shape}")
    if n == 0:
        return r.copy(), v.copy()
    if not np.all(np.isfinite(t)):
        raise ValueError("t_tt_s contains NaN or inf")

    # Same coverage check, and the same error, as the exact path. Coverage is an
    # interval, so checking the earliest and latest time is the same check.
    _check_eop_coverage(tt_seconds_to_time(np.array([t.min(), t.max()])))

    # States whose stencil reaches outside the EOP table go through the exact path.
    h, half = FAST_NODE_STEP_S, FAST_STENCIL // 2
    k0 = np.floor(t / h).astype(np.int64)
    lo, hi = _eop_tt_bounds()
    interp = ((k0 + 1 - half) * h >= lo) & ((k0 + half) * h <= hi)

    r_out = np.empty_like(r3)
    v_out = np.empty_like(v3)
    where = np.flatnonzero(interp)
    if where.size:
        poly = _IntervalPolynomials(k0[where])
        for start in range(0, where.size, _FAST_CHUNK):
            sel = where[start : start + _FAST_CHUNK]
            m, m_dot = poly.evaluate(t[sel], k0[sel])
            ri, vi = r3[:, sel], v3[:, sel]
            r_out[:, sel] = _apply_rotation(m, ri)
            v_out[:, sel] = _apply_rotation(m, vi) + _apply_rotation(m_dot, ri)
    if where.size < n:
        sel = np.flatnonzero(~interp)
        k_obj = r3.shape[0]
        tt = np.tile(t[sel], k_obj)
        re, ve = teme_to_gcrf(r3[:, sel].reshape(-1, 3), v3[:, sel].reshape(-1, 3), tt)
        r_out[:, sel] = re.reshape(k_obj, -1, 3)
        v_out[:, sel] = ve.reshape(k_obj, -1, 3)

    if single:
        return r_out[0, 0], v_out[0, 0]
    if r.ndim == 2:
        return r_out[0], v_out[0]
    return r_out, v_out
