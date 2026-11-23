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
    """Epoch of an ``sgp4.api.Satrec`` (its UTC Julian date pair) to TT seconds."""
    return float(time_to_tt_seconds(Time(jdsatepoch, jdsatepoch_f, format="jd", scale="utc")))


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
