"""Reader for CCSDS Orbit Comprehensive Message (OCM) ephemerides in KVN text form.

Why this exists
---------------
The public conjunction-assessment verification set from The Aerospace
Corporation and the Department of Commerce (Auman, Murphy and George, March
2026, CC0-1.0) ships its ephemerides as OCM files. To screen them with this
project's engine they have to become :class:`orbit_correction.types.Trajectory`
objects: GCRF, km and km/s, TT seconds since J2000.0, on one uniform time grid.
This module does that and nothing else; the comparison against the set's answer
key lives in ``scripts/compare_aerospace_testset.py``.

It is a script-side module, not part of the package, because nothing in the
pipeline proper reads OCM. It may be reused for other OCM ephemerides (for
example operator ephemerides of validation objects); see "Scope" below for what
it refuses.

What an OCM file looks like (as used by the test set)
------------------------------------------------------
Plain-text keyword = value lines grouped into blocks::

    CCSDS_OCM_VERS = 502.0-B-3          <- header
    META_START ... META_STOP            <- OBJECT_DESIGNATOR, TIME_SYSTEM, ...
    TRAJ_START                          <- one or more trajectory blocks
      TRAJ_REF_FRAME = EME2000, CENTER_NAME = EARTH, TRAJ_TYPE = CARTPV,
      USEABLE_START_TIME / USEABLE_STOP_TIME, TRAJ_UNITS = [km, ..., km/s]
      2025-01-02T08:59:57.387000  x y z vx vy vz      <- one line per sample
    TRAJ_STOP
    PHYS_START ... PHYS_STOP, COV_START ... COV_STOP, PERT, OD_START ... OD_STOP

The samples between a block's first epoch and USEABLE_START_TIME (and between
USEABLE_STOP_TIME and its last epoch) exist to support interpolation near the
ends and must not be used as states in their own right.

Conventions and conversions (what this module decides)
------------------------------------------------------
* **Time.** Sample epochs are ISO calendar strings in the file's TIME_SYSTEM.
  UTC is converted to TT as follows: the first epoch goes through astropy/ERFA
  (:func:`orbit_correction.frames.utc_to_tt_seconds`), and every later epoch is
  the first plus its elapsed time, which is exact only if no leap second falls
  inside the block. That is checked (TT - UTC at the first and last epoch must
  agree) and a block that spans a leap second is refused rather than handled.
  TT and TAI files are also accepted (TAI -> TT adds 32.184 s).
* **Frame.** EME2000 (mean equator and equinox of J2000.0) is rotated to GCRF
  with the constant IAU 2006 frame-bias matrix (ERFA ``bp06``), r_GCRF =
  B^T r_EME2000. The rotation is about 0.0230 arcsec (1.1e-7 rad): about 0.8 m
  at LEO radius, 4.7 m at GEO, 0.8 mm/s in velocity. It cannot change any range
  between two objects in the same frame, so for screening it is a convention,
  not an accuracy question. GCRF files pass through unchanged. Every other
  frame is refused.
* **Resampling.** :func:`resample` evaluates a block at arbitrary TT times
  either by picking samples that fall exactly on the requested times (when the
  file is already on that grid) or, by default, with a piecewise Lagrange
  interpolant through the positions of the 8 samples around each requested time
  (degree 7). A Hermite interpolant that also fits the file's velocities
  (:func:`hermite_interpolate`) is available but is not the default; see below.
* **Velocities.** In the test set the sampled velocities are not the time
  derivatives of the sampled positions. In 550 TLE-derived files they differ
  from the derivative of a degree-7 fit through the positions by 16 mm/s
  (median; up to 0.37 m/s), and in the maneuvering ephemerides by 4.6 m/s
  (median; up to 9.5 m/s), while the positions alone are smooth to millimetres
  (held-back samples, ``compare_aerospace_testset.py interp``). SGP4 itself
  shows the same kind of inconsistency (5-55 mm/s median on eight sample
  objects). A Hermite fit through such velocities moves the positions by
  0.09 m (median; up to 2 m) in TLE-derived files and about 20 m in maneuvering
  ones between samples, so by default :func:`resample`
  interpolates positions only and returns the derivative of that polynomial as
  the velocity (``velocity="derivative"``), consistent with the positions by
  construction. ``velocity="file"`` returns the file's own velocities instead.

Scope (refused rather than guessed)
-----------------------------------
Relative epochs (seconds from EPOCH_TZERO), day-of-year epochs, TRAJ_TYPE other
than CARTPV/CARTPVA, units other than km and km/s, centres other than EARTH,
frames other than EME2000/GCRF, and time systems other than UTC/TT/TAI each
raise :class:`OcmError`. Covariance, PHYS, PERT and maneuver blocks are counted
but not parsed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import erfa
import numpy as np

from orbit_correction.frames import tt_minus_utc_s, utc_to_tt_seconds
from orbit_correction.types import Trajectory

#: Earth's gravitational parameter (km^3/s^2), used only for the semimajor axis
#: reported by :func:`state_summary` (an index field, not a dynamics input).
MU_EARTH_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.137

#: EME2000 -> GCRF: r_GCRF = FRAME_BIAS.T @ r_EME2000 (ERFA bp06's rb is the
#: GCRS -> mean J2000 rotation, which is constant).
FRAME_BIAS = np.asarray(erfa.bp06(2451545.0, 0.0)[0], dtype=np.float64)

_SUPPORTED_FRAMES = {"EME2000", "GCRF"}
_SUPPORTED_TIME_SYSTEMS = {"UTC", "TT", "TAI"}
_SUPPORTED_TRAJ_TYPES = {"CARTPV": 6, "CARTPVA": 9}
_EXPECTED_UNITS = ("km", "km", "km", "km/s", "km/s", "km/s")
_KV_RE = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$")
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?$")


class OcmError(ValueError):
    """The file is not an OCM this reader handles (see the module docstring)."""


@dataclass(frozen=True)
class OcmTrajBlock:
    """One TRAJ block: its keywords and its samples, as written in the file.

    Attributes:
        meta: the block's keyword lines (``TRAJ_REF_FRAME``, ``USEABLE_START_TIME``...).
        epochs: shape (N,), ``datetime64[us]`` labels in the file's time system.
        states: shape (N, 6), position (km) and velocity (km/s) in ``meta["TRAJ_REF_FRAME"]``.
    """

    meta: dict[str, str]
    epochs: np.ndarray
    states: np.ndarray

    @property
    def frame(self) -> str:
        return self.meta.get("TRAJ_REF_FRAME", "")

    @property
    def useable(self) -> tuple[np.datetime64, np.datetime64]:
        """USEABLE_START/STOP_TIME, or the first and last epochs when absent."""
        lo = self.meta.get("USEABLE_START_TIME")
        hi = self.meta.get("USEABLE_STOP_TIME")
        return (
            parse_epoch(lo) if lo else self.epochs[0],
            parse_epoch(hi) if hi else self.epochs[-1],
        )


@dataclass(frozen=True)
class OcmFile:
    """A parsed OCM: header, metadata, trajectory blocks and the OD block.

    Attributes:
        header: keyword lines before META_START (``CCSDS_OCM_VERS``...).
        meta: the META block's keywords.
        trajectories: TRAJ blocks in file order.
        od: the OD block's keywords (``OD_EPOCH``...), empty if absent.
        block_counts: how many of each other block type the file holds.
    """

    header: dict[str, str]
    meta: dict[str, str]
    trajectories: list[OcmTrajBlock]
    od: dict[str, str] = field(default_factory=dict)
    block_counts: dict[str, int] = field(default_factory=dict)

    @property
    def designator(self) -> str:
        return self.meta.get("OBJECT_DESIGNATOR", "")

    @property
    def time_system(self) -> str:
        return self.meta.get("TIME_SYSTEM", "")


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def parse_epoch(s: str) -> np.datetime64:
    """Parse an ISO calendar epoch (``YYYY-MM-DDThh:mm:ss[.ffffff]``) to datetime64[us].

    The label is kept in whatever time system the file declares; conversion to
    TT happens in :func:`epochs_to_tt_seconds`. Leap-second labels (``:60``)
    cannot be represented and raise.
    """
    s = s.strip()
    if not _ISO_RE.match(s):
        raise OcmError(f"unsupported epoch format {s!r} (only ISO calendar epochs are read)")
    if s[17:19] == "60":
        raise OcmError(f"leap-second epoch {s!r} is not supported")
    return np.datetime64(s.rstrip("Z"), "us")


def _parse_units(s: str) -> tuple[str, ...]:
    return tuple(u.strip() for u in s.strip().strip("[]").split(","))


def parse_ocm(text: str) -> OcmFile:
    """Parse OCM KVN text.

    Raises:
        OcmError: on structure this reader does not handle (see module docstring).
    """
    header: dict[str, str] = {}
    meta: dict[str, str] = {}
    od: dict[str, str] = {}
    trajs: list[OcmTrajBlock] = []
    counts: dict[str, int] = {}
    section: str | None = None
    tmeta: dict[str, str] = {}
    rows: list[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("COMMENT"):
            continue
        if line.endswith("_START") and "=" not in line:
            name = line[: -len("_START")]
            if section is not None:
                raise OcmError(f"{line} inside {section} block")
            section = name
            counts[name] = counts.get(name, 0) + 1
            tmeta, rows = {}, []
            continue
        if line.endswith("_STOP") and "=" not in line:
            name = line[: -len("_STOP")]
            if name != section:
                raise OcmError(f"{line} does not close {section}")
            if name == "TRAJ":
                trajs.append(_build_traj(tmeta, rows))
            section = None
            continue
        if section is None:
            m = _KV_RE.match(line)
            if m:
                header[m.group(1)] = m.group(2)
            continue
        if section == "META":
            m = _KV_RE.match(line)
            if m:
                meta[m.group(1)] = m.group(2)
        elif section == "OD":
            m = _KV_RE.match(line)
            if m:
                od[m.group(1)] = m.group(2)
        elif section == "TRAJ":
            m = _KV_RE.match(line)
            if m and not rows:
                tmeta[m.group(1)] = m.group(2)
            else:
                rows.append(line)
        # other blocks (PHYS, COV, MAN, PERT, USER) are counted, not parsed
    if section is not None:
        raise OcmError(f"unterminated {section} block")
    if not trajs:
        raise OcmError("no TRAJ block")
    ts = meta.get("TIME_SYSTEM", "")
    if ts not in _SUPPORTED_TIME_SYSTEMS:
        raise OcmError(f"TIME_SYSTEM {ts!r} not supported")
    counts.pop("META", None)
    return OcmFile(header, meta, trajs, od, {k: v for k, v in counts.items() if k != "TRAJ"})


def _build_traj(tmeta: dict[str, str], rows: list[str]) -> OcmTrajBlock:
    ttype = tmeta.get("TRAJ_TYPE", "")
    if ttype not in _SUPPORTED_TRAJ_TYPES:
        raise OcmError(f"TRAJ_TYPE {ttype!r} not supported")
    if tmeta.get("CENTER_NAME", "EARTH") != "EARTH":
        raise OcmError(f"CENTER_NAME {tmeta.get('CENTER_NAME')!r} not supported")
    units = tmeta.get("TRAJ_UNITS")
    if units is not None and _parse_units(units)[:6] != _EXPECTED_UNITS:
        raise OcmError(f"TRAJ_UNITS {units!r} not supported (need km and km/s)")
    if not rows:
        raise OcmError("TRAJ block without samples")
    ncol = _SUPPORTED_TRAJ_TYPES[ttype]
    epochs = np.empty(len(rows), dtype="datetime64[us]")
    states = np.empty((len(rows), 6), dtype=np.float64)
    for k, row in enumerate(rows):
        parts = row.split()
        if len(parts) != ncol + 1:
            raise OcmError(f"sample line has {len(parts) - 1} values, {ttype} needs {ncol}")
        epochs[k] = parse_epoch(parts[0])
        states[k] = [float(x) for x in parts[1:7]]
    if np.any(np.diff(epochs) <= np.timedelta64(0, "us")):
        raise OcmError("TRAJ epochs are not strictly increasing")
    return OcmTrajBlock(dict(tmeta), epochs, states)


def scan_ocm_header(data: bytes) -> dict:
    """Index fields of an OCM without parsing its samples (fast, for 22 GB of files).

    Returns the META keywords, OD_EPOCH, and per-file trajectory facts: number of
    TRAJ blocks, total samples, frames, time span, USEABLE times of the first and
    last block, the first three epochs and the last epoch of the first block (a
    cheap uniform-spacing check), and the first state. Blocks are located with
    byte searches and samples are counted as lines, so the cost is close to one
    pass over the bytes.
    """
    out: dict = {}
    m0 = data.find(b"META_START")
    m1 = data.find(b"META_STOP")
    if m0 < 0 or m1 < 0:
        raise OcmError("no META block")
    for line in data[m0:m1].decode("ascii", "replace").splitlines()[1:]:
        m = _KV_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    o0 = data.find(b"OD_START")
    if o0 >= 0:
        o1 = data.find(b"OD_STOP", o0)
        for line in data[o0:o1].decode("ascii", "replace").splitlines()[1:]:
            m = _KV_RE.match(line)
            if m and m.group(1) == "OD_EPOCH":
                out["OD_EPOCH"] = m.group(2)
    blocks = []
    pos = 0
    while True:
        a = data.find(b"TRAJ_START", pos)
        if a < 0:
            break
        b = data.find(b"TRAJ_STOP", a)
        if b < 0:
            raise OcmError("unterminated TRAJ block")
        body = data[a:b]
        # keyword lines come first; the first line starting with a digit is a sample
        kv: dict[str, str] = {}
        first_sample = -1
        off = 0
        for line in body.split(b"\n")[1:]:
            s = line.strip()
            if s[:1].isdigit():
                first_sample = body.find(line, off)
                break
            m = _KV_RE.match(s.decode("ascii", "replace"))
            if m:
                kv[m.group(1)] = m.group(2)
            off += len(line) + 1
        if first_sample < 0:
            raise OcmError("TRAJ block without samples")
        samples = body[first_sample:].strip().split(b"\n")
        head = [r.split() for r in samples[:3]]
        last = samples[-1].split()
        blocks.append(
            {
                "kv": kv,
                "n": len(samples),
                "first_epochs": [h[0].decode() for h in head],
                "last_epoch": last[0].decode(),
                "first_state": [float(x) for x in head[0][1:7]],
            }
        )
        pos = b
    if not blocks:
        raise OcmError("no TRAJ block")
    out["n_traj_blocks"] = len(blocks)
    out["n_samples"] = sum(b["n"] for b in blocks)
    out["traj_ref_frames"] = ",".join(sorted({b["kv"].get("TRAJ_REF_FRAME", "") for b in blocks}))
    out["traj_types"] = ",".join(sorted({b["kv"].get("TRAJ_TYPE", "") for b in blocks}))
    out["center_names"] = ",".join(sorted({b["kv"].get("CENTER_NAME", "") for b in blocks}))
    out["first_epoch"] = blocks[0]["first_epochs"][0]
    out["last_epoch"] = blocks[-1]["last_epoch"]
    out["useable_start"] = blocks[0]["kv"].get("USEABLE_START_TIME", "")
    out["useable_stop"] = blocks[-1]["kv"].get("USEABLE_STOP_TIME", "")
    out["block0_first_epochs"] = "|".join(blocks[0]["first_epochs"])
    out["block0_n"] = blocks[0]["n"]
    out["block0_last_epoch"] = blocks[0]["last_epoch"]
    out["first_state"] = blocks[0]["first_state"]
    out["n_cov_blocks"] = data.count(b"COV_START")
    out["n_man_blocks"] = data.count(b"MAN_START")
    return out


def state_summary(state: Sequence[float]) -> dict[str, float]:
    """Two-body semimajor axis, eccentricity and their altitudes from one state.

    For the index and the subset rule only (a regime label, not dynamics).
    Returns NaN for unbound or degenerate states.
    """
    r = np.asarray(state[:3], dtype=np.float64)
    v = np.asarray(state[3:6], dtype=np.float64)
    rn, vn = float(np.linalg.norm(r)), float(np.linalg.norm(v))
    energy = vn * vn / 2.0 - MU_EARTH_KM3_S2 / rn
    if rn <= 0 or energy >= 0:
        return {"r_km": rn, "a_km": float("nan"), "e": float("nan"),
                "alt_a_km": float("nan"), "perigee_alt_km": float("nan"),
                "apogee_alt_km": float("nan")}
    a = -MU_EARTH_KM3_S2 / (2.0 * energy)
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / MU_EARTH_KM3_S2 - r / rn
    e = float(np.linalg.norm(e_vec))
    return {
        "r_km": rn,
        "a_km": a,
        "e": e,
        "alt_a_km": a - R_EARTH_KM,
        "perigee_alt_km": a * (1 - e) - R_EARTH_KM,
        "apogee_alt_km": a * (1 + e) - R_EARTH_KM,
    }


# --------------------------------------------------------------------------------------
# Time and frame
# --------------------------------------------------------------------------------------


def _iso(t: np.datetime64) -> str:
    return str(np.datetime_as_string(t, unit="us"))


def epochs_to_tt_seconds(epochs: np.ndarray, time_system: str) -> np.ndarray:
    """Epoch labels (datetime64[us]) in ``time_system`` -> TT seconds since J2000.0.

    UTC: the first epoch is converted by ERFA (leap-second table included);
    later epochs add their elapsed time, after checking that TT - UTC is the
    same at both ends of the span, i.e. no leap second falls inside it.

    Raises:
        OcmError: if a leap second falls inside the span, or the time system is
            not UTC, TT or TAI.
    """
    epochs = np.asarray(epochs, dtype="datetime64[us]")
    elapsed = (epochs - epochs[0]).astype(np.int64) / 1e6
    first = _iso(epochs[0])
    if time_system == "UTC":
        off0 = float(tt_minus_utc_s(first))
        off1 = float(tt_minus_utc_s(_iso(epochs[-1])))
        if off0 != off1:
            raise OcmError(
                f"a leap second falls between {first} and {_iso(epochs[-1])} (TT - UTC "
                f"{off0} -> {off1} s); this reader does not handle it"
            )
        t0 = float(utc_to_tt_seconds(first))
    elif time_system in ("TT", "TAI"):
        # TT seconds since J2000.0 straight from the label (J2000.0 = 2000-01-01T12:00 TT)
        t0 = float((epochs[0] - np.datetime64("2000-01-01T12:00:00", "us")).astype(np.int64) / 1e6)
        if time_system == "TAI":
            t0 += 32.184
    else:
        raise OcmError(f"TIME_SYSTEM {time_system!r} not supported")
    return t0 + elapsed


def to_gcrf(states: np.ndarray, frame: str) -> np.ndarray:
    """Rotate (N, 6) states from ``frame`` (EME2000 or GCRF) to GCRF."""
    if frame == "GCRF":
        return np.array(states, dtype=np.float64)
    if frame == "EME2000":
        r = states[:, :3] @ FRAME_BIAS  # (B^T r)^T = r^T B
        v = states[:, 3:6] @ FRAME_BIAS
        return np.concatenate([r, v], axis=1)
    raise OcmError(f"TRAJ_REF_FRAME {frame!r} not supported (EME2000 or GCRF)")


def from_gcrf(states: np.ndarray, frame: str) -> np.ndarray:
    """Inverse of :func:`to_gcrf` (GCRF -> EME2000 or GCRF)."""
    if frame == "GCRF":
        return np.array(states, dtype=np.float64)
    if frame == "EME2000":
        return np.concatenate([states[:, :3] @ FRAME_BIAS.T, states[:, 3:6] @ FRAME_BIAS.T], axis=1)
    raise OcmError(f"frame {frame!r} not supported (EME2000 or GCRF)")


# --------------------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TtBlock:
    """A TRAJ block converted to TT seconds and GCRF (what :func:`resample` reads).

    ``useable_tt_s`` is the block's USEABLE interval in TT seconds.
    """

    t_tt_s: np.ndarray
    states_gcrf: np.ndarray
    useable_tt_s: tuple[float, float]


def tt_blocks(ocm: OcmFile) -> list[TtBlock]:
    """Every TRAJ block of ``ocm`` in TT seconds and GCRF."""
    out = []
    for b in ocm.trajectories:
        t = epochs_to_tt_seconds(b.epochs, ocm.time_system)
        lo, hi = b.useable
        u = epochs_to_tt_seconds(np.array([b.epochs[0], lo, hi], dtype="datetime64[us]"),
                                 ocm.time_system)
        out.append(TtBlock(t, to_gcrf(b.states, b.frame), (float(u[1]), float(u[2]))))
    return out


def useable_span_tt(blocks: Sequence[TtBlock]) -> tuple[float, float]:
    """Earliest useable start and latest useable stop over the blocks (TT seconds)."""
    return min(b.useable_tt_s[0] for b in blocks), max(b.useable_tt_s[1] for b in blocks)


def hermite_interpolate(
    t: np.ndarray, r: np.ndarray, v: np.ndarray, t_out: np.ndarray, m: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise Hermite interpolation of positions and velocities.

    For each output time, take the ``m`` consecutive samples centred on it
    (shifted inward at the ends) and evaluate the degree 2m - 1 polynomial that
    matches position and velocity at all of them (Newton divided differences with
    doubled nodes), together with its derivative. Vectorised over output times.

    Args:
        t: shape (N,), sample times (seconds), strictly increasing, N >= m.
        r, v: shape (N, 3), positions and velocities at ``t``.
        t_out: output times; must lie within [t[0], t[-1]].
        m: samples per local fit (2 <= m <= N).

    Returns:
        positions and velocities at ``t_out``, each (len(t_out), 3).
    """
    t = np.asarray(t, dtype=np.float64)
    t_out = np.atleast_1d(np.asarray(t_out, dtype=np.float64))
    n = t.size
    if m < 2 or n < m:
        raise ValueError(f"need 2 <= m <= number of samples (m={m}, n={n})")
    if np.any(t_out < t[0]) or np.any(t_out > t[-1]):
        raise ValueError("t_out outside the sample span (no extrapolation)")
    # index of the first of the m samples used for each output time
    k = np.searchsorted(t, t_out, side="right") - 1  # t[k] <= t_out < t[k+1]
    start = np.clip(k - (m // 2 - 1), 0, n - m)
    idx = start[:, None] + np.arange(m)[None, :]  # (P, m)
    tc = t[idx]  # (P, m)
    t_ref = tc[:, (m - 1) // 2 : (m - 1) // 2 + 1]  # centre for conditioning
    z = np.repeat(tc - t_ref, 2, axis=1)  # (P, 2m) doubled nodes
    x = (t_out[:, None] - t_ref)[:, 0]  # (P,)
    rr = np.repeat(r[idx], 2, axis=1)  # (P, 2m, 3)
    vv = np.repeat(v[idx], 2, axis=1)
    return _newton_eval(z, rr, vv, x)


def lagrange_interpolate(
    t: np.ndarray, r: np.ndarray, t_out: np.ndarray, n_points: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise Lagrange interpolation of positions alone, with its derivative.

    For each output time, take the ``n_points`` consecutive samples centred on
    the interval that contains it (shifted inward at the ends) and evaluate the
    degree ``n_points - 1`` polynomial through their positions, and its time
    derivative, in Newton form. Vectorised over output times. The velocity it
    returns is the derivative of the interpolated positions, so the two are
    consistent by construction (see "Velocities" in the module docstring).

    Args:
        t: shape (N,), sample times (seconds), strictly increasing, N >= n_points.
        r: shape (N, 3), positions at ``t``.
        t_out: output times; must lie within [t[0], t[-1]].
        n_points: samples per local fit (2 <= n_points <= N).

    Returns:
        positions and their time derivatives at ``t_out``, each (len(t_out), 3).
    """
    t = np.asarray(t, dtype=np.float64)
    t_out = np.atleast_1d(np.asarray(t_out, dtype=np.float64))
    n = t.size
    if n_points < 2 or n < n_points:
        raise ValueError(f"need 2 <= n_points <= number of samples ({n_points}, {n})")
    if np.any(t_out < t[0]) or np.any(t_out > t[-1]):
        raise ValueError("t_out outside the sample span (no extrapolation)")
    k = np.searchsorted(t, t_out, side="right") - 1
    start = np.clip(k - (n_points // 2 - 1), 0, n - n_points)
    idx = start[:, None] + np.arange(n_points)[None, :]
    tc = t[idx]
    t_ref = tc[:, (n_points - 1) // 2 : (n_points - 1) // 2 + 1]
    z = tc - t_ref
    x = (t_out[:, None] - t_ref)[:, 0]
    return _newton_eval(z, r[idx], None, x)


def _newton_eval(z, f, df, x):
    """Newton divided differences on nodes ``z`` (P, q) and evaluation at ``x`` (P,).

    ``f`` holds the values (P, q, 3). For Hermite data the nodes come in equal
    pairs and ``df`` holds the derivatives at them (used where two nodes
    coincide); for Lagrange data ``df`` is None. Returns the polynomial and its
    derivative at ``x``, each (P, 3).
    """
    q = z.shape[1]
    col = f.copy()
    coef = [col[:, 0]]
    dz = z[:, 1:] - z[:, :-1]
    if df is None:
        col = (col[:, 1:] - col[:, :-1]) / dz[:, :, None]
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            col = np.where((dz == 0)[:, :, None], df[:, 1:],
                           (col[:, 1:] - col[:, :-1]) / dz[:, :, None])
    coef.append(col[:, 0])
    for j in range(2, q):
        dzj = z[:, j:] - z[:, :-j]
        col = (col[:, 1:] - col[:, :-1]) / dzj[:, :, None]
        coef.append(col[:, 0])
    # Horner for p and p' in Newton form: p = c0 + (x - z0)(c1 + (x - z1)(c2 + ...))
    p = coef[q - 1]
    dp = np.zeros_like(p)
    for j in range(q - 2, -1, -1):
        w = (x - z[:, j])[:, None]
        dp = p + w * dp
        p = coef[j] + w * p
    return p, dp


def resample(
    blocks: Sequence[TtBlock], t_out: np.ndarray, *, method: str = "lagrange",
    n_points: int = 8, m: int = 4, velocity: str = "derivative", exact_tol_s: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, str]:
    """GCRF states of an object at TT times ``t_out``.

    Each output time is served by the block whose USEABLE interval contains it
    (the later block wins at a shared boundary, so a post-maneuver block is used
    from its first instant). Within a block:

    * if every output time coincides with a sample (within ``exact_tol_s``), the
      sample positions are used as they are;
    * otherwise ``method="lagrange"`` (default): :func:`lagrange_interpolate`
      with ``n_points`` samples (degree 7 by default) on positions;
      ``method="hermite"``: :func:`hermite_interpolate` with ``m`` samples, which
      also fits the file's velocities.

    Velocities: ``velocity="derivative"`` (default) takes them from the Lagrange
    polynomial through the positions, so they are consistent with the positions;
    ``velocity="file"`` uses the file's own velocities (sampled, or Lagrange-
    interpolated). With ``method="hermite"`` the velocity is the Hermite
    polynomial's derivative, which passes through the file's velocities.

    Returns:
        ``(r, v, how)`` with r and v of shape (len(t_out), 3) and ``how`` one of
        ``"samples"``, ``"lagrange"``, ``"hermite"`` or ``"mixed"``.

    Raises:
        ValueError: if some output time is outside every block's USEABLE interval,
            or an option is unknown.
    """
    if method not in ("lagrange", "hermite") or velocity not in ("derivative", "file"):
        raise ValueError(f"unknown method {method!r} or velocity {velocity!r}")
    t_out = np.asarray(t_out, dtype=np.float64)
    r_out = np.full((t_out.size, 3), np.nan)
    v_out = np.full((t_out.size, 3), np.nan)
    owner = np.full(t_out.size, -1)
    for bi, b in enumerate(blocks):
        lo, hi = b.useable_tt_s
        owner[(t_out >= lo - exact_tol_s) & (t_out <= hi + exact_tol_s)] = bi
    if np.any(owner < 0):
        bad = t_out[owner < 0]
        raise ValueError(f"{bad.size} output times outside every USEABLE interval "
                         f"(first {bad[0]!r})")
    hows = set()
    for bi, b in enumerate(blocks):
        sel = owner == bi
        if not np.any(sel):
            continue
        tb, rb, vb = b.t_tt_s, b.states_gcrf[:, :3], b.states_gcrf[:, 3:6]
        tq = np.clip(t_out[sel], tb[0], tb[-1])
        npt = min(n_points, tb.size)
        j = np.clip(np.searchsorted(tb, tq), 0, tb.size - 1)
        jl = np.clip(j - 1, 0, tb.size - 1)
        near = np.where(np.abs(tb[j] - tq) <= np.abs(tb[jl] - tq), j, jl)
        on_samples = bool(np.all(np.abs(tb[near] - tq) <= exact_tol_s))
        if method == "hermite" and not on_samples:
            r, v = hermite_interpolate(tb, rb, vb, tq, m=min(m, tb.size))
            hows.add("hermite")
        else:
            if on_samples:
                r = rb[near]
                hows.add("samples")
            if velocity == "derivative" or not on_samples:
                rl, dl = lagrange_interpolate(tb, rb, tq, n_points=npt)
                if not on_samples:
                    r = rl
                    hows.add("lagrange")
            if velocity == "derivative" and method == "lagrange":
                v = dl
            elif on_samples:
                v = vb[near]
            else:
                v, _ = lagrange_interpolate(tb, vb, tq, n_points=npt)
        r_out[sel], v_out[sel] = r, v
    how = hows.pop() if len(hows) == 1 else "mixed"
    return r_out, v_out, how


def to_trajectory(
    blocks: Sequence[TtBlock], norad_id: int, t_out: np.ndarray, *,
    source: str = "aerospace_ivv", **resample_kw,
) -> tuple[Trajectory, str]:
    """A :class:`Trajectory` on ``t_out`` (TT seconds), plus how it was resampled.

    ``resample_kw`` goes to :func:`resample` (method, n_points, m, velocity).
    """
    r, v, how = resample(blocks, t_out, **resample_kw)
    return Trajectory(norad_id, np.asarray(t_out, dtype=np.float64), r, v, source), how


def read_ocm(path) -> OcmFile:
    """Read and parse an OCM file (ASCII KVN)."""
    with open(path, encoding="ascii") as fh:
        return parse_ocm(fh.read())
