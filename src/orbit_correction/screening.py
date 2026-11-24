"""All-pairs conjunction screening, classical geometric filters and screening metrics.

What this module is for
-----------------------
The screening experiment (planning package, sections 4 and 5) runs the same
all-pairs close-approach search on several sets of trajectories: truth
propagation (the reference answer), uncorrected SGP4 and corrected SGP4. It
compares them with the three classical filters of Hoots, Crawford and Roehrich
(1984). This module provides the pieces that comparison needs. It does not run
the comparison itself:

1. **Engine** (:func:`screen`): all close-approach events between every pair of
   :class:`~orbit_correction.types.Trajectory` objects sampled on one common TT
   grid. An event is a *local minimum of range* between a pair whose miss
   distance is within the screening threshold at its TCA. This is the definition
   the Aerospace / Department of Commerce screening test set uses: a pair can
   have several events in one window, and a pair whose range is inside the
   threshold at the window edge but still decreasing has no event there.
2. **Screening volume** (:class:`ScreeningVolume`, :class:`SphericalThreshold`):
   the threshold as a function of time from the screening epoch, so a padded,
   horizon-dependent threshold d_truth + pad(tau) can be passed for cheap
   screens. RTN-shaped volumes can be added later as another
   :class:`ScreeningVolume` without changing the event table (see that class).
3. **Classical filters** (:func:`apogee_perigee_filter`, :func:`orbit_path_filter`,
   :func:`time_filter`): pair pruning from orbital elements, each with a distance
   pad, each reporting which pairs it prunes (:class:`FilterResult`).
4. **Metrics** (:func:`screening_metrics`, :func:`filter_metrics`): recall
   against a reference event table, per event and per pair, under a configurable
   matching rule (same pair, TCA within a tolerance), false alarms, the fraction
   of pairs pruned, wall-clock cost and recall by TCA horizon (1, 3 and 7 days).

How the engine works (and the bound it relies on)
--------------------------------------------------
Positions between samples come from **cubic Hermite interpolation** of each
object's position and velocity at the two neighbouring samples. Hermite
interpolation is linear in the data, so the relative position of a pair is the
cubic Hermite interpolant of the relative states. Everything below (candidate
search, TCA, miss distance) works on that interpolant. At 60 s sampling in LEO
the interpolant is within about 0.4 m of a Keplerian orbit; the tests measure
this.

*Candidate search.* Each interval [t_k, t_k + h] is split into ``substeps``
sub-intervals of length h/m. At every sub-sample time u the positions of all
objects go into a k-d tree (:class:`scipy.spatial.cKDTree`), and all pairs
closer than a search radius R(u) are listed. The search radius is

    R(u) = D_local(u) + V(u) * h / (2 m) + 1 m,

where D_local(u) is the largest threshold at u and its two neighbouring
sub-samples, and V(u) bounds the relative speed of any pair on the intervals
touching u.

*The speed bound* comes from the interpolant itself, not from the physics. On
an interval, the Hermite cubic of one object is a Bezier curve with control
points r_k, r_k + h v_k / 3, r_{k+1} - h v_{k+1} / 3, r_{k+1}. Its velocity is a
quadratic Bezier curve with control points v_k, 3 (r_{k+1} - r_k) / h - v_k -
v_{k+1} and v_{k+1}, and a Bezier curve stays inside the convex hull of its
control points. So on that interval the object's interpolated speed is at most
B = max(|v_k|, |3 (r_{k+1} - r_k) / h - v_k - v_{k+1}|, |v_{k+1}|). For a pair,
|v_rel| <= B_1 + B_2, so V on interval k is the sum of the two largest B over
all objects. This holds for any sequence of states, including ones whose
velocities do not match their positions (for example corrected positions kept
with SGP4 velocities). On synthetic LEO catalogs with consistent states V is
15.0-16.1 km/s (the previous bound, 2 x (largest sampled speed + a speed-change
margin), gave 15.5-16.6).

*Why nothing is skipped.* A minimum at time t* lies within h/(2m) of a
sub-sample time u, in an interval touching u, and over that time the pair can
close by at most V(u) * h/(2m). So any interval containing a range minimum
under the threshold puts the pair in the candidate list at the nearest
sub-sample, at any relative speed. Nothing is assumed about the encounter
geometry. The threshold must not peak between sub-samples: D_local(u) covers
the threshold at u and its neighbours, not in between. Any threshold that is
monotone between sub-samples satisfies this (a constant, a smooth ramp, or
steps). A threshold with a narrow bump between sub-samples can lose the event
under the bump without warning; sample such a threshold more finely
(``substeps``) or widen the bump.

*Refinement.* Each candidate (pair, interval) first gets a cheap, rigorous lower
bound on its range over the interval: the distance from the origin to the chord
of the Bezier form of the cubic, minus the largest deviation of the cubic from
that chord (a convex-hull bound). Intervals whose bound exceeds the threshold
are dropped. For the rest, the range-rate polynomial f(s) = rho(s) . rho'(s)
(degree 5 in the interval fraction s) is sampled on 16 sub-intervals, and every
sign change from <= 0 to > 0 (a range minimum) is refined by bisection to
1e-15 of the interval. TCA is then t_k + s h, the miss distance is |rho(s)| and
the relative speed is |rho'(s)| / h. Endpoints shared by two intervals use
bit-identical values, so a minimum that falls exactly on a sample is counted
once.

Known limit of the refinement: a range minimum separated from a neighbouring
range maximum by less than h/16 (3.75 s at 60 s sampling) can be missed. That
only happens when the relative motion turns around within seconds, which needs
near-zero relative speed (co-moving objects whose range is nearly constant
anyway). It does not affect fly-by encounters at any speed.

*Complexity and memory.* Per sub-sample, building and querying the tree costs
O(n log n + P), where P is the number of pairs inside R(u). For objects spread
through a LEO shell, P grows as n^2 R^3. Total time is O(N m (n log n + P)) for
N samples. More sub-steps shrink R (and P as R^3) but add tree builds; the
benchmark in ``scripts/benchmark_screening.py`` sets the default. Memory: the
trajectories themselves (n N 56 bytes: position, velocity and each trajectory's
own time array; 0.565 MB = 0.538 MiB per object over 7 days at 60 s, 1.13 GB for
2,000 objects), plus one time chunk of stacked states (``chunk_steps`` x n x 48
bytes) and the candidate arrays for that chunk. The engine never builds an
n^2 x N array.

The classical filters (Hoots, Crawford and Roehrich 1984, as described by
Woodburn, Coppola and Stoner 2009 and Alfano and Finkleman 2013)
-------------------------------------------------------------------------------
All three work from **orbital elements** (:class:`KeplerElements`). For element
sets these are the SGP4 mean elements, propagated with J2 secular rates, as the
classical method assumes. For trajectories with no element set, elements can be
taken from states (:func:`elements_from_states`, osculating), and the
apogee/perigee filter can use the sampled radial extent of each trajectory
instead (:func:`radial_extent`, Woodburn's "APS" variant). Each filter keeps a
pair if its proxy distance is within ``threshold + pad``. ``threshold`` is a
scalar: for a horizon-dependent threshold pass its maximum over the window.

* **Apogee/perigee** (pairwise comparison, cost O(n^2) comparisons after O(n)
  element work): prune if the radial shells [perigee, apogee] of the two orbits
  are further apart than threshold + pad. Exact for unperturbed orbits; misses
  come from orbit evolution over the window (drag decay, J2 short-period terms
  in osculating elements).
* **Orbit path** (geometric pre-filter): prune if the minimum distance between
  the two orbit ellipses exceeds threshold + pad, at every sample time given
  (Woodburn samples start, middle and end). The path distance starts at the two
  relative-node points and is refined by damped Newton iteration (Hoots'
  method). Pairs are always kept when they are near-coplanar (the node line is
  undefined) or when the mutual inclination is small relative to the
  eccentricities (sin I < 1.5 (e1 + e2), where Newton from the nodes can settle
  on a local minimum and overstate the distance).
* **Time filter**: prune if, over the window, the times when object 1 lies
  within threshold + pad of object 2's orbit plane never overlap the times when
  object 2 lies within threshold + pad of object 1's plane, near the same
  relative node. Windows are computed analytically from the elements
  (two-body plus optional J2 secular rates). The geometry is recomputed from
  the same elements every ``segment_s`` and held fixed within each segment, and
  a time pad widens every window.

What the filters lose with exact elements. With pure two-body motion, none of
the three loses an encounter at zero pad. With J2 secular drift, the elements
still describe the motion exactly, but over 7 days the orbit-path filter
(sampled at the start, middle and end) and the time filter (3,600 s segments)
both lose a few encounters at zero pad, because the geometry changes between
samples or within a segment. An independent review measured this on several
synthetic catalogs: for example 1 of 33 events at 5 km on this module's
generator for each filter, and 16 of 67 (orbit path) and 7 of 67 (time filter)
at 5 km on a 60 km thick shell with e <= 0.02. Some orbit-path losses persist
at a 30 km pad; the time-filter losses went away with 2-10 km of distance pad
or a 10 s time pad. The committed test of exact elements covers 1 day only. Losses from what the
elements leave out (drag, short-period terms) come on top of these.

Hoots et al. (1984) is paywalled; the implementations follow the later
descriptions above, and where they differ the docstrings say so.

Conventions: GCRF, km, km/s, TT seconds since J2000.0 (``types.py``). Elements
are referred to whatever frame their states or element set use. A common
rotation of every object (TEME vs GCRF, for instance) changes no distance, so
the filters give the same answer in either frame provided all objects share it.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pyarrow as pa
from scipy.spatial import cKDTree

from orbit_correction.types import Trajectory

#: Earth's gravitational parameter (km^3/s^2), EGM-96 / WGS-84 value. Used for
#: element conversions and the speed bound. SGP4 element sets use WGS-72
#: (398600.8); the 1e-6 relative difference does not matter for filtering.
MU_EARTH_KM3_S2 = 398600.4418
#: Earth's equatorial radius (km) and J2, for J2 secular rates.
R_EARTH_KM = 6378.137
J2_EARTH = 1.08262668e-3
DAY_S = 86400.0
#: TCA horizons (s) for nested recall: 1, 3 and 7 days (planning package section 5).
DEFAULT_HORIZONS_S = (1.0 * DAY_S, 3.0 * DAY_S, 7.0 * DAY_S)
#: Orbit-path filter: pass a pair through when sin(mutual inclination) is below this
#: multiple of e1 + e2 (see :func:`orbit_path_distance`).
ECCENTRIC_COPLANAR_FACTOR = 1.5

_TWO_PI = 2.0 * math.pi
#: Margin added to the search radius and the interval bound for floating-point rounding (km).
#: The events are those of the interpolant, so interpolation error needs no margin here.
_INTERP_MARGIN_KM = 1e-3
#: Number of sub-intervals on which the range-rate polynomial is sampled.
_ROOT_GRID = 16
_BISECTION_STEPS = 50


# --------------------------------------------------------------------------------------
# Screening volume (threshold as a function of time from the screening epoch)
# --------------------------------------------------------------------------------------


@runtime_checkable
class ScreeningVolume(Protocol):
    """What the engine needs from a screening threshold or volume.

    Two methods, so that a non-spherical volume can be added later without
    touching the engine or the event table:

    * :meth:`bound_km` gives the radius of a sphere that contains the volume at
      time ``tau_s`` from the screening epoch. The engine uses it for the
      candidate search and the interval pre-filter, so it must never be smaller
      than the volume.
    * :meth:`accepts` decides, at TCA, whether the event is inside the volume.
      It receives both objects' states, so an RTN volume can build the frame of
      either object. Event identity stays the scalar range minimum, as in the
      Aerospace test set, so the event table is unchanged.

    For a future RTN volume: the engine orders each pair by trajectory index,
    not by "primary". A per-object RTN volume is asymmetric (B can be inside A's
    volume while A is outside B's), so such a volume would test both
    orientations in :meth:`accepts` and might add a flag column naming which
    one triggered; the existing columns would not change.
    """

    def bound_km(self, tau_s: np.ndarray) -> np.ndarray:
        """Radius (km) of a sphere containing the volume at each ``tau_s``."""
        ...

    def accepts(
        self,
        tau_s: np.ndarray,
        r1_km: np.ndarray,
        v1_km_s: np.ndarray,
        r2_km: np.ndarray,
        v2_km_s: np.ndarray,
    ) -> np.ndarray:
        """Boolean per event: is object 2 inside the volume at TCA?"""
        ...


@dataclass(frozen=True)
class SphericalThreshold:
    """Scalar miss-distance threshold, constant or a function of horizon.

    Args:
        radius_km: a constant (km), or a callable mapping ``tau_s`` (array of
            seconds from the screening epoch, which may be negative before it)
            to the threshold in km, element-wise. A truth screen passes d_truth;
            a cheap screen passes d_truth + pad(tau).

    The engine evaluates the callable at sub-sample times for the search, and at
    TCA for acceptance. It assumes the callable does not peak between sub-samples
    (see the module docstring and :func:`screen`).
    """

    radius_km: float | Callable[[np.ndarray], np.ndarray]

    def bound_km(self, tau_s: np.ndarray) -> np.ndarray:
        tau = np.asarray(tau_s, dtype=np.float64)
        if callable(self.radius_km):
            d = np.broadcast_to(np.asarray(self.radius_km(tau), dtype=np.float64), tau.shape)
        else:
            d = np.full(tau.shape, float(self.radius_km))
        if not np.all(np.isfinite(d)) or np.any(d < 0):
            raise ValueError("threshold must be finite and non-negative")
        return np.array(d, dtype=np.float64)

    def accepts(self, tau_s, r1_km, v1_km_s, r2_km, v2_km_s) -> np.ndarray:
        miss = np.linalg.norm(np.asarray(r2_km) - np.asarray(r1_km), axis=-1)
        return miss <= self.bound_km(tau_s)


def as_volume(threshold: float | Callable | ScreeningVolume) -> ScreeningVolume:
    """Accept a number, a callable of ``tau_s`` or a :class:`ScreeningVolume`."""
    if isinstance(threshold, ScreeningVolume):
        return threshold
    if isinstance(threshold, (int, float, np.integer, np.floating)) or callable(threshold):
        return SphericalThreshold(threshold)
    raise TypeError(
        f"threshold must be a number, a callable or a ScreeningVolume, got {threshold!r}"
    )


# --------------------------------------------------------------------------------------
# Event table
# --------------------------------------------------------------------------------------

#: One row per close-approach event (a local range minimum within the threshold).
#: ``norad_id_1 < norad_id_2`` always, so (A, B) and (B, A) give the same row.
EVENT_SCHEMA = pa.schema(
    [
        pa.field("norad_id_1", pa.int64(), nullable=False),
        pa.field("norad_id_2", pa.int64(), nullable=False),
        pa.field("tca_tt_s", pa.float64(), nullable=False),  # TT seconds since J2000.0
        pa.field("tau_s", pa.float64(), nullable=False),  # TCA minus the screening epoch
        pa.field("miss_km", pa.float64(), nullable=False),
        pa.field("rel_speed_km_s", pa.float64(), nullable=False),
        pa.field("threshold_km", pa.float64(), nullable=False),  # bound_km at TCA
    ]
)


def make_event_table(
    norad_id_1, norad_id_2, tca_tt_s, tau_s, miss_km, rel_speed_km_s, threshold_km
) -> pa.Table:
    """Build an event table (pair IDs are reordered so that id_1 < id_2, rows sorted)."""
    a = np.asarray(norad_id_1, dtype=np.int64)
    b = np.asarray(norad_id_2, dtype=np.int64)
    id1, id2 = np.minimum(a, b), np.maximum(a, b)
    if np.any(id1 == id2):
        raise ValueError("an event needs two different objects")
    cols = [
        id1,
        id2,
        np.asarray(tca_tt_s, dtype=np.float64),
        np.asarray(tau_s, dtype=np.float64),
        np.asarray(miss_km, dtype=np.float64),
        np.asarray(rel_speed_km_s, dtype=np.float64),
        np.asarray(threshold_km, dtype=np.float64),
    ]
    order = np.lexsort((cols[1], cols[0], cols[2]))  # by TCA, then pair
    return pa.Table.from_arrays([pa.array(c[order]) for c in cols], schema=EVENT_SCHEMA)


def empty_event_table() -> pa.Table:
    return EVENT_SCHEMA.empty_table()


# --------------------------------------------------------------------------------------
# Pair indexing and pair masks
# --------------------------------------------------------------------------------------


def pair_count(n: int) -> int:
    """Number of unordered pairs of n objects."""
    return n * (n - 1) // 2


def pair_index(i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    """Linear index of pair (i, j), i < j, in ``np.triu_indices(n, 1)`` order."""
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


def pair_from_index(k: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`pair_index`."""
    k = np.asarray(k, dtype=np.int64)
    # Row i is the largest i with start(i) = i(2n - i - 1)/2 <= k.
    disc = (2 * n - 1) ** 2 - 8 * k
    i = np.floor(((2 * n - 1) - np.sqrt(disc.astype(np.float64))) / 2).astype(np.int64)
    start = i * (2 * n - i - 1) // 2
    # Fix float rounding at row boundaries.
    i = np.where(start > k, i - 1, i)
    start = i * (2 * n - i - 1) // 2
    nxt = (i + 1) * (2 * n - i - 2) // 2
    i = np.where(k >= nxt, i + 1, i)
    start = i * (2 * n - i - 1) // 2
    j = k - start + i + 1
    return i, j


@dataclass(frozen=True)
class PairMask:
    """Which pairs of an ordered set of objects are kept.

    Attributes:
        norad_ids: shape (n,). Their order defines pair indexing, and must match
            the order of the trajectories passed to :func:`screen`.
        keep: shape (n(n-1)/2,), bool, in ``np.triu_indices(n, 1)`` order.
    """

    norad_ids: np.ndarray
    keep: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.norad_ids, dtype=np.int64)
        keep = np.asarray(self.keep, dtype=bool)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
            raise ValueError("norad_ids must be 1-D and unique")
        if keep.shape != (pair_count(len(ids)),):
            raise ValueError(f"keep must have shape ({pair_count(len(ids))},), got {keep.shape}")
        object.__setattr__(self, "norad_ids", ids)
        object.__setattr__(self, "keep", keep)

    @classmethod
    def all_pairs(cls, norad_ids) -> PairMask:
        ids = np.asarray(norad_ids, dtype=np.int64)
        return cls(ids, np.ones(pair_count(len(ids)), dtype=bool))

    @property
    def n_pairs(self) -> int:
        return int(self.keep.size)

    @property
    def n_kept(self) -> int:
        return int(np.count_nonzero(self.keep))

    @property
    def pruned_fraction(self) -> float:
        return 1.0 - self.n_kept / self.n_pairs if self.n_pairs else 0.0

    def kept_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """NORAD IDs of the kept pairs, each ordered id_1 < id_2."""
        i, j = pair_from_index(np.flatnonzero(self.keep), len(self.norad_ids))
        a, b = self.norad_ids[i], self.norad_ids[j]
        return np.minimum(a, b), np.maximum(a, b)

    def pruned_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        """NORAD IDs of the pruned pairs, each ordered id_1 < id_2."""
        i, j = pair_from_index(np.flatnonzero(~self.keep), len(self.norad_ids))
        a, b = self.norad_ids[i], self.norad_ids[j]
        return np.minimum(a, b), np.maximum(a, b)

    def is_kept(self, id_a, id_b) -> np.ndarray:
        """Whether each pair (id_a, id_b) is kept. Unknown IDs raise KeyError."""
        pos = {int(x): k for k, x in enumerate(self.norad_ids)}
        ia = np.array([pos[int(x)] for x in np.atleast_1d(id_a)], dtype=np.int64)
        ib = np.array([pos[int(x)] for x in np.atleast_1d(id_b)], dtype=np.int64)
        lo, hi = np.minimum(ia, ib), np.maximum(ia, ib)
        if np.any(lo == hi):
            raise ValueError("a pair needs two different objects")
        return self.keep[pair_index(lo, hi, len(self.norad_ids))]

    def __and__(self, other: PairMask) -> PairMask:
        if not np.array_equal(self.norad_ids, other.norad_ids):
            raise ValueError("masks refer to different object orders")
        return PairMask(self.norad_ids, self.keep & other.keep)


# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScreeningResult:
    """Output of :func:`screen`.

    Attributes:
        events: event table (:data:`EVENT_SCHEMA`).
        stats: counts and costs: ``wall_s`` (whole call), ``search_s`` (k-d tree
            phase), ``refine_s``, ``n_objects``, ``n_samples``, ``step_s``,
            ``substeps``, ``rel_speed_bound_km_s`` (largest per-interval V) and
            ``rel_speed_bound_km_s_min``, ``search_radius_km_min`` /
            ``_max``, ``n_candidate_intervals`` (unique pair-intervals listed by
            the search), ``n_refined_intervals`` (left after the convex-hull
            bound), ``n_minima`` (range minima found) and ``n_events``.
    """

    events: pa.Table
    stats: dict = field(default_factory=dict)


def _common_grid(trajectories: Sequence[Trajectory]) -> tuple[np.ndarray, np.ndarray, float]:
    if len(trajectories) < 2:
        raise ValueError("screening needs at least two trajectories")
    for tr in trajectories:
        if not isinstance(tr, Trajectory):
            raise TypeError(f"expected Trajectory, got {type(tr).__name__}")
    t = trajectories[0].t_tt_s
    if t.size < 2:
        raise ValueError("trajectories need at least two samples")
    for tr in trajectories[1:]:
        if tr.t_tt_s.shape != t.shape or not np.array_equal(tr.t_tt_s, t):
            raise ValueError("all trajectories must share one time grid (identical t_tt_s)")
    h = (t[-1] - t[0]) / (t.size - 1)
    if np.max(np.abs(np.diff(t) - h)) > 1e-6:
        raise ValueError("the time grid must be uniformly spaced")
    ids = np.array([tr.norad_id for tr in trajectories], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("NORAD IDs must be unique within one screening run")
    return ids, t, float(h)


def _hermite(r0, v0, r1, v1, h, s):
    """Cubic Hermite position at fraction ``s`` of an interval of length ``h``."""
    s2, s3 = s * s, s * s * s
    return (
        (2 * s3 - 3 * s2 + 1) * r0
        + (s3 - 2 * s2 + s) * h * v0
        + (-2 * s3 + 3 * s2) * r1
        + (s3 - s2) * h * v1
    )


def _hermite_coeffs(r0, v0, r1, v1, h):
    """Monomial coefficients c0..c3 of the Hermite cubic in the interval fraction s."""
    c0 = r0
    c1 = h * v0
    c2 = 3.0 * (r1 - r0) - h * (2.0 * v0 + v1)
    c3 = 2.0 * (r0 - r1) + h * (v0 + v1)
    return c0, c1, c2, c3


def screen(
    trajectories: Sequence[Trajectory],
    threshold: float | Callable | ScreeningVolume,
    *,
    epoch_tt_s: float | None = None,
    window_tt_s: tuple[float, float] | None = None,
    keep: PairMask | None = None,
    substeps: int = 2,
    chunk_steps: int = 360,
) -> ScreeningResult:
    """Find every close-approach event between all pairs of trajectories.

    Args:
        trajectories: :class:`Trajectory` objects with identical, uniformly spaced
            ``t_tt_s`` grids (60 s in practice) and unique NORAD IDs.
        threshold: km. A number, a callable of ``tau_s`` (seconds from
            ``epoch_tt_s``) or a :class:`ScreeningVolume`. The search samples it
            only at sub-sample times (every h / ``substeps``), so it must not
            peak between them: an event under a narrow bump in the threshold
            between two sub-samples can be missed, with no warning. Constants,
            ramps, steps and other thresholds that are monotone between
            sub-samples are safe. Acceptance uses the threshold at TCA.
        epoch_tt_s: screening epoch from which ``tau_s`` is counted. Default: the
            first grid time.
        window_tt_s: keep only events with TCA in [start, end] (inclusive).
            Default: the whole grid.
        keep: optional :class:`PairMask` (for example from the classical filters)
            restricting which pairs are screened. Its ``norad_ids`` must be in the
            same order as ``trajectories``.
        substeps: sub-samples per interval for the candidate search (m in the
            module docstring). More sub-steps shrink the search radius.
        chunk_steps: intervals processed at once (memory control only; results do
            not depend on it).

    Returns:
        :class:`ScreeningResult` with the event table and run statistics.
    """
    t_start = time.perf_counter()
    ids, t, h = _common_grid(trajectories)
    volume = as_volume(threshold)
    n = len(trajectories)
    if substeps < 1 or chunk_steps < 1:
        raise ValueError("substeps and chunk_steps must be >= 1")
    if keep is not None and not np.array_equal(keep.norad_ids, ids):
        raise ValueError("keep.norad_ids must list the trajectories' NORAD IDs in the same order")
    epoch = float(t[0]) if epoch_tt_s is None else float(epoch_tt_s)
    win_lo, win_hi = (float(t[0]), float(t[-1])) if window_tt_s is None else map(float, window_tt_s)

    m = int(substeps)
    sub_h = h / m

    n_int = t.size - 1
    out: list[tuple[np.ndarray, ...]] = []
    n_cand = n_refined = n_minima = 0
    r_min_seen, r_max_seen = math.inf, 0.0
    v_min_seen, v_max_seen = math.inf, 0.0
    search_s = refine_s = 0.0

    for k0 in range(0, n_int, chunk_steps):
        k1 = min(k0 + chunk_steps, n_int)
        L = k1 - k0
        R = np.stack([tr.r_gcrf_km[k0 : k1 + 1] for tr in trajectories], axis=1)  # (L+1, n, 3)
        V = np.stack([tr.v_gcrf_km_s[k0 : k1 + 1] for tr in trajectories], axis=1)

        # Relative-speed bound per interval (module docstring): each object's
        # interpolated speed is at most the largest control point of its velocity
        # curve; a pair's relative speed is at most the sum of the two largest.
        v_ctrl = np.linalg.norm(3.0 * (R[1:] - R[:-1]) / h - V[:-1] - V[1:], axis=2)  # (L, n)
        v_end = np.linalg.norm(V, axis=2)  # (L+1, n)
        b_obj = np.maximum(np.maximum(v_end[:-1], v_end[1:]), v_ctrl)
        top2 = -np.partition(-b_obj, 1, axis=1)[:, :2]
        v_int = top2.sum(axis=1)  # (L,)
        v_min_seen = min(v_min_seen, float(v_int.min()))
        v_max_seen = max(v_max_seen, float(v_int.max()))

        # Threshold at every sub-sample, the local maximum around each, and per interval.
        n_sub = L * m + 1
        jj = np.arange(n_sub)
        u = t[k0 + jj // m] + (jj % m) * sub_h
        b_sub = volume.bound_km(u - epoch)
        b_loc = b_sub.copy()
        b_loc[1:] = np.maximum(b_loc[1:], b_sub[:-1])
        b_loc[:-1] = np.maximum(b_loc[:-1], b_sub[1:])
        # Speed bound over the intervals each sub-sample registers to (see below):
        # interval (j - 1) // m and interval j // m, where they exist in this chunk.
        v_loc = np.maximum(v_int[np.minimum(jj // m, L - 1)],
                           v_int[np.maximum((jj - 1) // m, 0)])
        radius = b_loc + v_loc * sub_h / 2.0 + _INTERP_MARGIN_KM
        r_min_seen = min(r_min_seen, float(radius.min()))
        r_max_seen = max(r_max_seen, float(radius.max()))
        b_int = np.maximum(b_sub[:-1].reshape(L, m).max(axis=1), b_sub[m::m])

        # Candidate search: one k-d tree per sub-sample.
        ts = time.perf_counter()
        keys: list[np.ndarray] = []
        for j in range(n_sub):
            li, jm = divmod(j, m)
            if jm == 0:
                pos = R[li]
            else:
                pos = _hermite(R[li], V[li], R[li + 1], V[li + 1], h, jm / m)
            tree = cKDTree(pos, leafsize=16, compact_nodes=False, balanced_tree=False)
            pr = tree.query_pairs(float(radius[j]), output_type="ndarray")
            if pr.size == 0:
                continue
            pi, pj = pr[:, 0].astype(np.int64), pr[:, 1].astype(np.int64)
            lo, hi = np.minimum(pi, pj), np.maximum(pi, pj)
            if keep is not None:
                sel = keep.keep[pair_index(lo, hi, n)]
                lo, hi = lo[sel], hi[sel]
                if lo.size == 0:
                    continue
            base = (lo * n + hi) * L
            ks = set()
            if j > 0:
                ks.add((j - 1) // m)
            if j < n_sub - 1:
                ks.add(j // m)
            for kk in ks:
                keys.append(base + kk)
        search_s += time.perf_counter() - ts
        if not keys:
            continue

        tr0 = time.perf_counter()
        uk = np.unique(np.concatenate(keys))
        n_cand += uk.size
        kk = uk % L
        pair = uk // L
        ii, jx = pair // n, pair % n
        res = _refine_intervals(R, V, h, ii, jx, kk, b_int)
        n_refined += res["n_refined"]
        if res["s"].size:
            s, i_e, j_e, k_e = res["s"], res["i"], res["j"], res["k"]
            n_minima += s.size
            tca = t[k0 + k_e] + s * h
            tau = tca - epoch
            r1 = _hermite(R[k_e, i_e], V[k_e, i_e], R[k_e + 1, i_e], V[k_e + 1, i_e], h, s[:, None])
            r2 = _hermite(R[k_e, j_e], V[k_e, j_e], R[k_e + 1, j_e], V[k_e + 1, j_e], h, s[:, None])
            v1 = _hermite_rate(R[k_e, i_e], V[k_e, i_e], R[k_e + 1, i_e], V[k_e + 1, i_e], h, s)
            v2 = _hermite_rate(R[k_e, j_e], V[k_e, j_e], R[k_e + 1, j_e], V[k_e + 1, j_e], h, s)
            ok = volume.accepts(tau, r1, v1, r2, v2) & (tca >= win_lo) & (tca <= win_hi)
            if np.any(ok):
                out.append(
                    (
                        ids[i_e[ok]],
                        ids[j_e[ok]],
                        tca[ok],
                        tau[ok],
                        res["miss"][ok],
                        res["vrel"][ok],
                        volume.bound_km(tau[ok]),
                    )
                )
        refine_s += time.perf_counter() - tr0

    if out:
        cols = [np.concatenate(c) for c in zip(*out, strict=True)]
        events = _dedupe(make_event_table(*cols))
    else:
        events = empty_event_table()

    stats = {
        "wall_s": time.perf_counter() - t_start,
        "search_s": search_s,
        "refine_s": refine_s,
        "n_objects": n,
        "n_pairs_screened": keep.n_kept if keep is not None else pair_count(n),
        "n_samples": int(t.size),
        "step_s": h,
        "substeps": m,
        "rel_speed_bound_km_s": v_max_seen if math.isfinite(v_min_seen) else float("nan"),
        "rel_speed_bound_km_s_min": v_min_seen if math.isfinite(v_min_seen) else float("nan"),
        "search_radius_km_min": r_min_seen if math.isfinite(r_min_seen) else float("nan"),
        "search_radius_km_max": r_max_seen,
        "n_candidate_intervals": int(n_cand),
        "n_refined_intervals": int(n_refined),
        "n_minima": int(n_minima),
        "n_events": events.num_rows,
    }
    return ScreeningResult(events, stats)


def _hermite_rate(r0, v0, r1, v1, h, s):
    """Velocity of the Hermite cubic (km/s) at fractions ``s`` (one per row)."""
    s = np.asarray(s)[:, None]
    s2 = s * s
    return (
        (6 * s2 - 6 * s) * r0 / h
        + (3 * s2 - 4 * s + 1) * v0
        + (-6 * s2 + 6 * s) * r1 / h
        + (3 * s2 - 2 * s) * v1
    )


def _refine_intervals(R, V, h, i, j, k, b_int) -> dict:
    """Range minima inside candidate intervals (see the module docstring).

    ``R``, ``V``: chunk states (L+1, n, 3). ``i``, ``j``, ``k``: object indices and
    chunk-local interval index per candidate. ``b_int``: threshold bound per interval.
    Returns the fraction ``s``, miss distance, relative speed and the indices of
    every minimum found, plus how many intervals passed the convex-hull bound.
    """
    dr0 = R[k, j] - R[k, i]
    dv0 = V[k, j] - V[k, i]
    dr1 = R[k + 1, j] - R[k + 1, i]
    dv1 = V[k + 1, j] - V[k + 1, i]

    # Convex-hull lower bound on |rho| over the interval (Bezier form of the cubic).
    p1 = dr0 + h * dv0 / 3.0
    p2 = dr1 - h * dv1 / 3.0
    chord = dr1 - dr0
    cc = np.einsum("ij,ij->i", chord, chord)
    sc = np.where(cc > 0, -np.einsum("ij,ij->i", dr0, chord) / np.where(cc > 0, cc, 1.0), 0.0)
    sc = np.clip(sc, 0.0, 1.0)
    min_chord = np.linalg.norm(dr0 + sc[:, None] * chord, axis=1)
    dev = np.maximum(
        np.linalg.norm(p1 - (2.0 * dr0 + dr1) / 3.0, axis=1),
        np.linalg.norm(p2 - (dr0 + 2.0 * dr1) / 3.0, axis=1),
    )
    sel = (min_chord - dev) <= b_int[k] + _INTERP_MARGIN_KM
    i, j, k = i[sel], j[sel], k[sel]
    dr0, dv0, dr1, dv1 = dr0[sel], dv0[sel], dr1[sel], dv1[sel]
    n_refined = int(i.size)
    empty = np.empty(0)
    if n_refined == 0:
        return {"n_refined": 0, "s": empty, "miss": empty, "vrel": empty, "i": i, "j": j, "k": k}

    c0, c1, c2, c3 = _hermite_coeffs(dr0, dv0, dr1, dv1, h)

    def dot(a, b):
        return np.einsum("ij,ij->i", a, b)

    # f(s) = rho . drho/ds, degree 5; coefficients low to high.
    f = np.stack(
        [
            dot(c0, c1),
            dot(c1, c1) + 2 * dot(c0, c2),
            3 * dot(c0, c3) + 3 * dot(c1, c2),
            4 * dot(c1, c3) + 2 * dot(c2, c2),
            5 * dot(c2, c3),
            3 * dot(c3, c3),
        ],
        axis=1,
    )

    def feval(coef, s):
        out = coef[..., 5]
        for p in range(4, -1, -1):
            out = out * s + coef[..., p]
        return out

    grid = np.linspace(0.0, 1.0, _ROOT_GRID + 1)
    F = feval(f[:, None, :], grid[None, :])
    # Endpoints straight from the data, so a sample shared by two intervals gives
    # bit-identical values and a minimum exactly on it is counted once.
    F[:, 0] = dot(dr0, h * dv0)
    F[:, -1] = dot(dr1, h * dv1)
    row, g = np.nonzero((F[:, :-1] <= 0.0) & (F[:, 1:] > 0.0))
    if row.size == 0:
        return {"n_refined": n_refined, "s": empty, "miss": empty, "vrel": empty,
                "i": i[:0], "j": j[:0], "k": k[:0]}
    lo, hi = grid[g].copy(), grid[g + 1].copy()
    fr = f[row]
    for _ in range(_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        neg = feval(fr, mid) <= 0.0
        lo = np.where(neg, mid, lo)
        hi = np.where(neg, hi, mid)
    s = 0.5 * (lo + hi)
    sc_ = s[:, None]
    rho = c0[row] + c1[row] * sc_ + c2[row] * sc_**2 + c3[row] * sc_**3
    drho = (c1[row] + 2 * c2[row] * sc_ + 3 * c3[row] * sc_**2) / h
    return {
        "n_refined": n_refined,
        "s": s,
        "miss": np.linalg.norm(rho, axis=1),
        "vrel": np.linalg.norm(drho, axis=1),
        "i": i[row],
        "j": j[row],
        "k": k[row],
    }


def _dedupe(events: pa.Table, tol_s: float = 1e-6) -> pa.Table:
    """Drop repeated rows (same pair, TCA within ``tol_s``). A safety net only."""
    if events.num_rows < 2:
        return events
    a = events.column("norad_id_1").to_numpy()
    b = events.column("norad_id_2").to_numpy()
    tca = events.column("tca_tt_s").to_numpy()
    order = np.lexsort((tca, b, a))
    same = (a[order][1:] == a[order][:-1]) & (b[order][1:] == b[order][:-1])
    close = np.abs(np.diff(tca[order])) <= tol_s
    drop = np.zeros(events.num_rows, dtype=bool)
    drop[order[1:][same & close]] = True
    if not drop.any():
        return events
    return events.filter(pa.array(~drop))


# --------------------------------------------------------------------------------------
# Orbital elements (inputs to the classical filters, and synthetic trajectories)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class KeplerElements:
    """Keplerian elements of n objects, each at its own epoch.

    Attributes (all shape (n,)):
        norad_id: int64.
        epoch_tt_s: TT seconds since J2000.0 at which the elements hold.
        a_km, e, i_rad, raan_rad, argp_rad, mean_anomaly_rad: semimajor axis,
            eccentricity (0 <= e < 1), inclination, right ascension of the
            ascending node, argument of perigee, mean anomaly.

    Mean elements from an element set (with J2 secular rates, as the classical
    filters assume) or osculating elements from states
    (:func:`elements_from_states`) both fit here; which one was used is the
    caller's choice and should be logged.
    """

    norad_id: np.ndarray
    epoch_tt_s: np.ndarray
    a_km: np.ndarray
    e: np.ndarray
    i_rad: np.ndarray
    raan_rad: np.ndarray
    argp_rad: np.ndarray
    mean_anomaly_rad: np.ndarray

    def __post_init__(self) -> None:
        ids = np.atleast_1d(np.asarray(self.norad_id, dtype=np.int64))
        object.__setattr__(self, "norad_id", ids)
        n = ids.size
        for name in ("epoch_tt_s", "a_km", "e", "i_rad", "raan_rad", "argp_rad",
                     "mean_anomaly_rad"):
            arr = np.broadcast_to(np.asarray(getattr(self, name), dtype=np.float64), (n,)).copy()
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN or inf")
            object.__setattr__(self, name, arr)
        if np.any(self.a_km <= 0):
            raise ValueError("a_km must be positive")
        if np.any((self.e < 0) | (self.e >= 1)):
            raise ValueError("e must be in [0, 1) (elliptical orbits only)")

    def __len__(self) -> int:
        return int(self.norad_id.size)

    def take(self, idx) -> KeplerElements:
        return KeplerElements(*(getattr(self, f)[idx] for f in _ELEMENT_FIELDS))

    @property
    def perigee_km(self) -> np.ndarray:
        return self.a_km * (1.0 - self.e)

    @property
    def apogee_km(self) -> np.ndarray:
        return self.a_km * (1.0 + self.e)


_ELEMENT_FIELDS = ("norad_id", "epoch_tt_s", "a_km", "e", "i_rad", "raan_rad", "argp_rad",
                   "mean_anomaly_rad")


def mean_motion_to_a_km(n_rad_s: np.ndarray, mu: float = MU_EARTH_KM3_S2) -> np.ndarray:
    """Semimajor axis (km) from a mean motion in rad/s (Kepler's third law)."""
    n = np.asarray(n_rad_s, dtype=np.float64)
    return np.cbrt(mu / n**2)


def solve_kepler(mean_anomaly: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Eccentric anomaly from mean anomaly (Newton, vectorized; e < 1)."""
    M = np.mod(np.asarray(mean_anomaly, dtype=np.float64), _TWO_PI)
    e = np.asarray(e, dtype=np.float64)
    E = np.where(e < 0.8, M, np.pi)
    for _ in range(50):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < 1e-14):
            break
    return E


def true_to_mean(nu: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Mean anomaly from true anomaly (radians, unwrapped branch kept)."""
    nu = np.asarray(nu, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    E = 2.0 * np.arctan2(np.sqrt(1.0 - e) * np.sin(nu / 2.0), np.sqrt(1.0 + e) * np.cos(nu / 2.0))
    return E - e * np.sin(E)


def _perifocal_axes(raan, inc, argp):
    """Unit vectors P (to perigee), Q (90 deg ahead in the plane) and W (orbit normal)."""
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)
    P = np.stack([cO * cw - sO * sw * ci, sO * cw + cO * sw * ci, sw * si], axis=-1)
    Q = np.stack([-cO * sw - sO * cw * ci, -sO * sw + cO * cw * ci, cw * si], axis=-1)
    W = np.stack([sO * si, -cO * si, ci], axis=-1)
    return P, Q, W


def j2_secular_rates(el: KeplerElements) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First-order J2 secular rates (rad/s): dRAAN/dt, dARGP/dt, dM/dt (incl. mean motion).

    Standard first-order results (e.g. Vallado, *Fundamentals of Astrodynamics*,
    section 9.6), applied to mean elements.
    """
    n0 = np.sqrt(MU_EARTH_KM3_S2 / el.a_km**3)
    p = el.a_km * (1.0 - el.e**2)
    k = 1.5 * J2_EARTH * (R_EARTH_KM / p) ** 2 * n0
    ci = np.cos(el.i_rad)
    raan_dot = -k * ci
    argp_dot = 0.5 * k * (5.0 * ci**2 - 1.0)
    m_dot = n0 + 0.5 * k * np.sqrt(1.0 - el.e**2) * (3.0 * ci**2 - 1.0)
    return raan_dot, argp_dot, m_dot


def propagate_elements(el: KeplerElements, t_tt_s: float, *, j2: bool = True) -> KeplerElements:
    """Elements at time ``t_tt_s``: two-body, plus J2 secular rates if ``j2``."""
    dt = float(t_tt_s) - el.epoch_tt_s
    if j2:
        raan_dot, argp_dot, m_dot = j2_secular_rates(el)
    else:
        raan_dot = argp_dot = np.zeros(len(el))
        m_dot = np.sqrt(MU_EARTH_KM3_S2 / el.a_km**3)
    return KeplerElements(
        el.norad_id,
        np.full(len(el), float(t_tt_s)),
        el.a_km,
        el.e,
        el.i_rad,
        np.mod(el.raan_rad + raan_dot * dt, _TWO_PI),
        np.mod(el.argp_rad + argp_dot * dt, _TWO_PI),
        np.mod(el.mean_anomaly_rad + m_dot * dt, _TWO_PI),
    )


def states_from_elements(
    el: KeplerElements, t_tt_s: np.ndarray, *, j2: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Positions and velocities, shape (n, T, 3), on the grid ``t_tt_s``.

    Two-body motion, with J2 secular drift of RAAN, argument of perigee and mean
    anomaly if ``j2``. The velocity is the exact time derivative of the returned
    position, drift included, so the trajectories are smooth and self-consistent
    (Hermite interpolation of them is accurate). Not a physical force model: for
    synthetic tests, benchmarks and practice only.
    """
    t = np.asarray(t_tt_s, dtype=np.float64)
    dt = t[None, :] - el.epoch_tt_s[:, None]
    n0 = np.sqrt(MU_EARTH_KM3_S2 / el.a_km**3)
    if j2:
        raan_dot, argp_dot, m_dot = j2_secular_rates(el)
    else:
        raan_dot = argp_dot = np.zeros(len(el))
        m_dot = n0
    raan = el.raan_rad[:, None] + raan_dot[:, None] * dt
    argp = el.argp_rad[:, None] + argp_dot[:, None] * dt
    M = el.mean_anomaly_rad[:, None] + m_dot[:, None] * dt
    a, e = el.a_km[:, None], el.e[:, None]
    E = solve_kepler(M, np.broadcast_to(e, M.shape))
    b = a * np.sqrt(1.0 - e**2)
    cE, sE = np.cos(E), np.sin(E)
    inc = np.broadcast_to(el.i_rad[:, None], M.shape)
    P, Q, W = _perifocal_axes(raan, inc, argp)
    x, y = a * (cE - e), b * sE
    r = x[..., None] * P + y[..., None] * Q
    # dE/dM = 1 / (1 - e cos E); position derivative with respect to M.
    dEdM = 1.0 / (1.0 - e * cE)
    dx, dy = -a * sE * dEdM, b * cE * dEdM
    v = (dx[..., None] * P + dy[..., None] * Q) * m_dot[:, None, None]
    if j2:
        z = np.array([0.0, 0.0, 1.0])
        v = v + argp_dot[:, None, None] * np.cross(W, r) + raan_dot[:, None, None] * np.cross(z, r)
    return r, v


def elements_from_states(
    norad_id, epoch_tt_s, r_km: np.ndarray, v_km_s: np.ndarray, mu: float = MU_EARTH_KM3_S2
) -> KeplerElements:
    """Osculating Keplerian elements from Cartesian states, shape (n, 3) each.

    Circular orbits get argp = 0 (the anomaly is then measured from the node);
    equatorial orbits get RAAN = 0 (measured from +x). Hyperbolic states raise.
    """
    r = np.atleast_2d(np.asarray(r_km, dtype=np.float64))
    v = np.atleast_2d(np.asarray(v_km_s, dtype=np.float64))
    rn = np.linalg.norm(r, axis=1)
    hvec = np.cross(r, v)
    hn = np.linalg.norm(hvec, axis=1)
    W = hvec / hn[:, None]
    inc = np.arccos(np.clip(W[:, 2], -1.0, 1.0))
    node = np.stack([-hvec[:, 1], hvec[:, 0], np.zeros(len(r))], axis=1)
    nn = np.linalg.norm(node, axis=1)
    equatorial = nn < 1e-10 * hn
    node_hat = np.where(equatorial[:, None], np.array([1.0, 0.0, 0.0]),
                        node / np.where(nn > 0, nn, 1.0)[:, None])
    raan = np.where(equatorial, 0.0, np.mod(np.arctan2(node_hat[:, 1], node_hat[:, 0]), _TWO_PI))
    v2 = np.einsum("ij,ij->i", v, v)
    rv = np.einsum("ij,ij->i", r, v)
    evec = ((v2 - mu / rn)[:, None] * r - rv[:, None] * v) / mu
    e = np.linalg.norm(evec, axis=1)
    energy = v2 / 2.0 - mu / rn
    if np.any(energy >= 0):
        raise ValueError("hyperbolic or parabolic state")
    a = -mu / (2.0 * energy)
    circular = e < 1e-11
    p_hat = np.where(circular[:, None], node_hat, evec / np.where(e > 0, e, 1.0)[:, None])

    def angle(from_vec, to_vec):
        return np.arctan2(np.einsum("ij,ij->i", np.cross(from_vec, to_vec), W),
                          np.einsum("ij,ij->i", from_vec, to_vec))

    argp = np.where(circular, 0.0, np.mod(angle(node_hat, p_hat), _TWO_PI))
    nu = angle(p_hat, r / rn[:, None])
    e_used = np.where(circular, 0.0, e)
    M = np.mod(true_to_mean(nu, e_used), _TWO_PI)
    return KeplerElements(norad_id, epoch_tt_s, a, e_used, inc, raan, argp, M)


def osculating_elements(trajectories: Sequence[Trajectory], t_tt_s: float) -> KeplerElements:
    """Osculating elements of each trajectory at one of its sample times."""
    out_r, out_v = [], []
    for tr in trajectories:
        k = int(np.argmin(np.abs(tr.t_tt_s - t_tt_s)))
        if abs(tr.t_tt_s[k] - t_tt_s) > 1e-6:
            raise ValueError(f"{t_tt_s} is not a sample time of object {tr.norad_id}")
        out_r.append(tr.r_gcrf_km[k])
        out_v.append(tr.v_gcrf_km_s[k])
    ids = [tr.norad_id for tr in trajectories]
    return elements_from_states(ids, np.full(len(ids), float(t_tt_s)), np.array(out_r),
                                np.array(out_v))


def synthetic_leo_elements(
    n: int, epoch_tt_s: float, *, seed: int = 0, first_norad_id: int = 1
) -> KeplerElements:
    """Random LEO mean elements for tests, benchmarks and practice notebooks.

    Not a model of the real catalog. It only aims to put objects in the places
    that make screening busy: perigee altitudes concentrated between 500 and
    1,000 km (60% drawn from N(750, 150) km, 40% uniform over 300-1,500 km), small
    eccentricities (exponential, mean 0.003, apogee altitude capped at 2,000 km),
    and inclinations clustered near sun-synchronous (35% at 97.5 +/- 1.5 deg),
    near-polar (15% at 86-90 deg) and 50-75 deg (25%), the rest uniform over
    0-100 deg. Node, argument of perigee and mean anomaly are uniform.
    """
    rng = np.random.default_rng(seed)
    hp = np.where(rng.random(n) < 0.6, rng.normal(750.0, 150.0, n), rng.uniform(300.0, 1500.0, n))
    hp = np.clip(hp, 300.0, 1500.0)
    rp = R_EARTH_KM + hp
    e = np.minimum(rng.exponential(0.003, n), 0.05)
    e = np.minimum(e, ((R_EARTH_KM + 2000.0) - rp) / ((R_EARTH_KM + 2000.0) + rp))
    a = rp / (1.0 - e)
    u = rng.random(n)
    inc = np.select(
        [u < 0.35, u < 0.50, u < 0.75],
        [rng.normal(97.5, 1.5, n), rng.uniform(86.0, 90.0, n), rng.uniform(50.0, 75.0, n)],
        rng.uniform(0.0, 100.0, n),
    )
    return KeplerElements(
        np.arange(first_norad_id, first_norad_id + n),
        np.full(n, float(epoch_tt_s)),
        a,
        e,
        np.radians(inc),
        rng.uniform(0.0, _TWO_PI, n),
        rng.uniform(0.0, _TWO_PI, n),
        rng.uniform(0.0, _TWO_PI, n),
    )


def trajectories_from_elements(
    el: KeplerElements, t_tt_s: np.ndarray, *, j2: bool = True, source: str = "synthetic",
    block: int = 100,
) -> list[Trajectory]:
    """:class:`Trajectory` objects from :func:`states_from_elements` (synthetic use only)."""
    t = np.asarray(t_tt_s, dtype=np.float64)
    out: list[Trajectory] = []
    for s in range(0, len(el), block):
        sub = el.take(np.arange(s, min(s + block, len(el))))
        r, v = states_from_elements(sub, t, j2=j2)
        out.extend(Trajectory(int(sub.norad_id[k]), t, r[k], v[k], source) for k in range(len(sub)))
    return out


def radial_extent(trajectories: Sequence[Trajectory]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NORAD IDs and sampled min / max |r| (km) of each trajectory.

    Input to the sampled apogee/perigee filter (Woodburn's "APS" variant). This
    works from states, needs no orbit theory and covers drag decay over the
    window. At 60 s sampling the sampled extremes are within about 0.05 km of the
    true ones in LEO.
    """
    ids = np.array([tr.norad_id for tr in trajectories], dtype=np.int64)
    rn = [np.linalg.norm(tr.r_gcrf_km, axis=1) for tr in trajectories]
    return ids, np.array([x.min() for x in rn]), np.array([x.max() for x in rn])


# --------------------------------------------------------------------------------------
# Classical filters
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FilterResult:
    """What one filter kept and pruned.

    Attributes:
        name: ``"apogee_perigee"``, ``"orbit_path"`` or ``"time"``.
        mask: pairs kept after this filter. Pairs pruned by an earlier filter
            (``prior``) stay pruned.
        n_evaluated: pairs this filter looked at (those the prior kept).
        n_pruned_here: pairs this filter pruned itself.
        params: threshold, pads and other settings, for logging.
        wall_s: wall-clock seconds.
    """

    name: str
    mask: PairMask
    n_evaluated: int
    n_pruned_here: int
    params: dict
    wall_s: float

    @property
    def pruned_fraction(self) -> float:
        """Fraction of all pairs pruned so far (this filter and any prior)."""
        return self.mask.pruned_fraction

    def pruned_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mask.pruned_pairs()


def _pairs_to_evaluate(n: int, prior: PairMask | None, ids: np.ndarray) -> np.ndarray:
    if prior is None:
        return np.arange(pair_count(n), dtype=np.int64)
    if not np.array_equal(prior.norad_ids, ids):
        raise ValueError("prior mask refers to a different object order")
    return np.flatnonzero(prior.keep)


def _blocks(lin: np.ndarray, block: int = 2_000_000):
    for s in range(0, lin.size, block):
        yield lin[s : s + block]


def apsides_from_elements(*elements: KeplerElements) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NORAD IDs, smallest perigee and largest apogee (km) over one or more element samples.

    Pass the elements at the start (and, for Woodburn's "APP" variant, also at
    the end) of the window. All samples must list the same objects in the same
    order.
    """
    if not elements:
        raise ValueError("need at least one set of elements")
    ids = elements[0].norad_id
    for el in elements[1:]:
        if not np.array_equal(el.norad_id, ids):
            raise ValueError("element samples must list the same objects in the same order")
    q = np.min([el.perigee_km for el in elements], axis=0)
    Q = np.max([el.apogee_km for el in elements], axis=0)
    return ids, q, Q


def apogee_perigee_filter(
    norad_ids,
    perigee_km,
    apogee_km,
    *,
    threshold_km: float,
    pad_km: float,
    prior: PairMask | None = None,
) -> FilterResult:
    """Apogee/perigee filter: prune pairs whose radial shells don't come close enough.

    A pair is kept if ``max(q1, q2) - min(Q1, Q2) <= threshold_km + pad_km``,
    where q and Q are the perigee and apogee radii (or the sampled min and max
    radius, :func:`radial_extent`). Works from elements or sampled states; the
    caller chooses (see :func:`apsides_from_elements`).

    With exact, unperturbed orbits and the threshold included, this filter
    cannot miss an encounter. Misses come from radial change the inputs don't
    describe: drag decay over the window, or J2 short-period terms when q and Q
    come from osculating elements at one epoch. The pad has to cover these.
    """
    t0 = time.perf_counter()
    ids = np.asarray(norad_ids, dtype=np.int64)
    q = np.asarray(perigee_km, dtype=np.float64)
    Q = np.asarray(apogee_km, dtype=np.float64)
    n = ids.size
    lin = _pairs_to_evaluate(n, prior, ids)
    keep = np.zeros(pair_count(n), dtype=bool)
    d = float(threshold_km) + float(pad_km)
    for blk in _blocks(lin):
        i, j = pair_from_index(blk, n)
        gap = np.maximum(q[i], q[j]) - np.minimum(Q[i], Q[j])
        keep[blk] = gap <= d
    return FilterResult(
        "apogee_perigee",
        PairMask(ids, keep),
        int(lin.size),
        int(lin.size - np.count_nonzero(keep)),
        {"threshold_km": float(threshold_km), "pad_km": float(pad_km)},
        time.perf_counter() - t0,
    )


def orbit_path_distance(
    el1: KeplerElements,
    el2: KeplerElements,
    *,
    coplanar_tol_rad: float = 1e-3,
    eccentric_coplanar_factor: float = ECCENTRIC_COPLANAR_FACTOR,
    iterations: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Minimum distance (km) between orbit paths, element-wise over two aligned element sets.

    Hoots' method, as described by Woodburn et al. (2009): start at the two
    relative-node points (exact for circular orbits) and refine each with a
    damped Newton iteration on |X1(E1) - X2(E2)|^2 in the eccentric anomalies.
    The smaller of the two results is returned. Converges to a local minimum
    next to a node.

    When the planes are close compared with the eccentricities, the closest
    point of the paths need not be near a node. Near-circular orbits a fraction
    of a degree apart are separated mostly by the in-plane radial difference,
    which varies by up to about a (e1 + e2) around the orbit, while the
    out-of-plane separation is only about a sin I. Newton from the nodes then
    settles on a local minimum and can overstate the distance by hundreds of km
    (Woodburn et al. note the same weakness). Such pairs are not trusted: where
    ``sin I < eccentric_coplanar_factor * (e1 + e2)`` the pair is passed through
    with distance 0. The default factor, 1.5, comes from a sweep against a
    global search (grid plus BFGS) over about 5,700 LEO pairs with q 6,900-7,300
    km, e up to 0.05 and I from 0.06 to 10 deg: Newton from the nodes overstated
    the distance by more than 1 m only below sin I / (e1 + e2) = 0.83, and in none
    of about 4,200 pairs at a ratio of 1 or above. The pass-through costs about
    0.01% of pairs on :func:`synthetic_leo_elements` and 14-21% of the pairs
    inside a tight cluster of similar planes. Setting the factor to 0 restores
    the plain Hoots behaviour.

    Returns:
        (distance_km, passed): passed is True where the pair is kept without a
        distance, because the mutual inclination is below ``coplanar_tol_rad``
        or below the eccentricity rule above; the distance there is 0.
    """
    P1, Q1, W1 = _perifocal_axes(el1.raan_rad, el1.i_rad, el1.argp_rad)
    P2, Q2, W2 = _perifocal_axes(el2.raan_rad, el2.i_rad, el2.argp_rad)
    K = np.cross(W1, W2)
    sin_i = np.linalg.norm(K, axis=1)
    coplanar = sin_i < np.sin(coplanar_tol_rad)
    hhat = K / np.where(coplanar, 1.0, sin_i)[:, None]
    a1, e1, b1 = el1.a_km, el1.e, el1.a_km * np.sqrt(1 - el1.e**2)
    a2, e2, b2 = el2.a_km, el2.e, el2.a_km * np.sqrt(1 - el2.e**2)
    passed = coplanar | (sin_i < float(eccentric_coplanar_factor) * (e1 + e2))

    def X(a, e, b, P, Q, E):
        return (a * (np.cos(E) - e))[:, None] * P + (b * np.sin(E))[:, None] * Q

    def Xp(a, b, P, Q, E):
        return (-a * np.sin(E))[:, None] * P + (b * np.cos(E))[:, None] * Q

    def fval(E1, E2):
        d = X(a1, e1, b1, P1, Q1, E1) - X(a2, e2, b2, P2, Q2, E2)
        return np.einsum("ij,ij->i", d, d)

    def ecc_anom(nu, e):
        return 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))

    best = np.full(len(el1), np.inf)
    for sign in (1.0, -1.0):
        h = sign * hhat
        E1 = ecc_anom(np.arctan2(np.einsum("ij,ij->i", h, Q1), np.einsum("ij,ij->i", h, P1)), e1)
        E2 = ecc_anom(np.arctan2(np.einsum("ij,ij->i", h, Q2), np.einsum("ij,ij->i", h, P2)), e2)
        f = fval(E1, E2)
        for _ in range(iterations):
            x1 = X(a1, e1, b1, P1, Q1, E1)
            x2 = X(a2, e2, b2, P2, Q2, E2)
            d = x1 - x2
            x1p, x2p = Xp(a1, b1, P1, Q1, E1), Xp(a2, b2, P2, Q2, E2)
            x1pp = -(x1 + (a1 * e1)[:, None] * P1)
            x2pp = -(x2 + (a2 * e2)[:, None] * P2)
            g1 = 2 * np.einsum("ij,ij->i", d, x1p)
            g2 = -2 * np.einsum("ij,ij->i", d, x2p)
            h11 = 2 * (np.einsum("ij,ij->i", x1p, x1p) + np.einsum("ij,ij->i", d, x1pp))
            h22 = 2 * (np.einsum("ij,ij->i", x2p, x2p) - np.einsum("ij,ij->i", d, x2pp))
            h12 = -2 * np.einsum("ij,ij->i", x1p, x2p)
            det = h11 * h22 - h12**2
            pd = (det > 0) & (h11 > 0)
            safe_det = np.where(pd, det, 1.0)
            # Newton step where the Hessian is positive definite, scaled gradient elsewhere.
            gd1 = -g1 / (np.abs(h11) + np.abs(h12) + 1.0)
            gd2 = -g2 / (np.abs(h22) + np.abs(h12) + 1.0)
            s1 = np.where(pd, -(h22 * g1 - h12 * g2) / safe_det, gd1)
            s2 = np.where(pd, -(h11 * g2 - h12 * g1) / safe_det, gd2)
            norm = np.hypot(s1, s2)
            scale = np.minimum(1.0, 0.5 / np.where(norm > 0, norm, 1.0))
            s1, s2 = s1 * scale, s2 * scale
            # Backtracking: accept the first step factor that lowers f.
            accepted = np.zeros(len(el1), dtype=bool)
            newE1, newE2, newf = E1.copy(), E2.copy(), f.copy()
            for fac in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                c1, c2 = E1 + fac * s1, E2 + fac * s2
                fc = fval(c1, c2)
                take = (~accepted) & (fc < f)
                newE1 = np.where(take, c1, newE1)
                newE2 = np.where(take, c2, newE2)
                newf = np.where(take, fc, newf)
                accepted |= take
            if not accepted.any():
                break
            E1, E2, f = newE1, newE2, newf
        best = np.minimum(best, f)
    dist = np.sqrt(np.maximum(best, 0.0))
    return np.where(passed, 0.0, dist), passed


def orbit_path_filter(
    elements: KeplerElements | Sequence[KeplerElements],
    sample_times_tt_s: Sequence[float] | None = None,
    *,
    threshold_km: float,
    pad_km: float,
    j2: bool = True,
    prior: PairMask | None = None,
    coplanar_tol_rad: float = 1e-3,
    eccentric_coplanar_factor: float = ECCENTRIC_COPLANAR_FACTOR,
) -> FilterResult:
    """Orbit-path (geometric) filter: prune pairs whose orbit paths stay far apart.

    Two ways to give the samples (Woodburn samples the start, middle and end of
    the window):

    * one :class:`KeplerElements` plus ``sample_times_tt_s``: the elements are
      propagated to each time (two-body plus J2 secular rates if ``j2``). This is
      the classical, element-set form (mean elements);
    * a sequence of :class:`KeplerElements`, one per sample time, for example
      :func:`osculating_elements` of trajectories at the start, middle and end
      (``sample_times_tt_s`` is then not used).

    The path-to-path distance is computed at every sample
    (:func:`orbit_path_distance`), and a pair is kept if it is within
    ``threshold_km + pad_km`` at any sample, or if :func:`orbit_path_distance`
    passes it through (near-coplanar, or a small mutual inclination compared
    with the eccentricities). Works from elements.

    Known misses (Woodburn et al. 2009; Alfano and Finkleman 2013): the path
    distance changes between samples (precession, drag), and a pad alone does not
    remove every miss. With exact elements and J2 secular drift, start, middle
    and end sampling over 7 days still loses a few encounters at zero pad, and
    some at a 30 km pad (module docstring). Woodburn recommends not using this
    filter; it is here as the classical baseline.
    """
    t0 = time.perf_counter()
    if isinstance(elements, KeplerElements):
        if not sample_times_tt_s:
            raise ValueError("give sample_times_tt_s with a single set of elements")
        samples = [propagate_elements(elements, tt, j2=j2) for tt in sample_times_tt_s]
        times = [float(x) for x in sample_times_tt_s]
    else:
        samples = list(elements)
        if not samples:
            raise ValueError("need at least one element sample")
        times = [float(el.epoch_tt_s[0]) for el in samples]
    ids = samples[0].norad_id
    for el in samples[1:]:
        if not np.array_equal(el.norad_id, ids):
            raise ValueError("element samples must list the same objects in the same order")
    n = ids.size
    lin = _pairs_to_evaluate(n, prior, ids)
    keep = np.zeros(pair_count(n), dtype=bool)
    d = float(threshold_km) + float(pad_km)
    for blk in _blocks(lin, 500_000):
        i, j = pair_from_index(blk, n)
        k = np.zeros(blk.size, dtype=bool)
        for el in samples:
            todo = ~k
            if not todo.any():
                break
            dist, _ = orbit_path_distance(el.take(i[todo]), el.take(j[todo]),
                                          coplanar_tol_rad=coplanar_tol_rad,
                                          eccentric_coplanar_factor=eccentric_coplanar_factor)
            k[np.flatnonzero(todo)] = dist <= d
        keep[blk] = k
    return FilterResult(
        "orbit_path",
        PairMask(ids, keep),
        int(lin.size),
        int(lin.size - np.count_nonzero(keep)),
        {"threshold_km": float(threshold_km), "pad_km": float(pad_km), "j2": j2,
         "sample_times_tt_s": times,
         "coplanar_tol_rad": coplanar_tol_rad,
         "eccentric_coplanar_factor": float(eccentric_coplanar_factor)},
        time.perf_counter() - t0,
    )


def time_filter(
    elements: KeplerElements,
    window_tt_s: tuple[float, float],
    *,
    threshold_km: float,
    pad_km: float,
    time_pad_s: float = 0.0,
    j2: bool = True,
    segment_s: float = 3600.0,
    prior: PairMask | None = None,
    coplanar_tol_rad: float = 1e-3,
) -> FilterResult:
    """Time filter: prune pairs that are never near the same relative node at the same time.

    For a pair with mutual inclination I, an object at angle theta from the
    relative node line, at radius r, is ``r |sin theta| sin I`` from the other
    orbit's plane. Being within ``d = threshold_km + pad_km`` of the other object
    requires both objects to be within d of the other's plane, near the same node
    direction, at the same time. With r bounded below by the perigee radius q,
    the angle window around each node is ``|sin theta| <= d / (q sin I)``. For
    fixed planes this is a superset of the true condition (Woodburn et al. 2009,
    figure 8), so with two-body motion and exact elements the proxy never prunes
    a real encounter.

    Each window is turned into a time interval with Kepler's equation from the
    elements (two-body, plus J2 secular rates if ``j2``), repeated every
    anomalistic period, and widened by ``time_pad_s`` on each side. The geometry
    (planes, node line, phase) is recomputed from the same elements at the middle
    of every segment of ``segment_s`` and held fixed within the segment. This
    resembles the rectification Woodburn describes, but it only re-evaluates the
    elements; it does not check them against the trajectory. Hoots' original
    propagates one relative nodal period from the middle of the window. A pair
    is kept if any of its intervals overlap at either node, or if the orbits are
    near-coplanar, or if one object is always within d of the other's plane.

    Works from elements: mean elements for element-set objects (osculating
    semimajor axes have J2 short-period swings of about 10 km in LEO, which
    corrupt the mean motion and the predicted timing).

    Known misses:

    * Geometry frozen within a segment. With J2 secular drift the planes and
      perigees move during each segment, so even exact elements lose a few
      encounters at zero pad over 7 days with 3,600 s segments (module
      docstring). Shorter segments reduce the losses without removing them;
      2-10 km of distance pad or a 10 s time pad removed them in the review's
      catalogs.
    * Timing error the elements don't model, chiefly drag, which shortens the
      period and accumulates along-track drift through the window. The time pad
      (or a large distance pad) has to cover it.
    """
    t0 = time.perf_counter()
    ids = elements.norad_id
    n = ids.size
    lin = _pairs_to_evaluate(n, prior, ids)
    keep = np.zeros(pair_count(n), dtype=bool)
    d = float(threshold_km) + float(pad_km)
    ws, we = float(window_tt_s[0]), float(window_tt_s[1])
    if we <= ws or segment_s <= 0:
        raise ValueError("need window end > start and segment_s > 0")
    n_seg = max(1, math.ceil((we - ws) / segment_s))
    edges = np.linspace(ws, we, n_seg + 1)
    sin_tol = math.sin(coplanar_tol_rad)

    # Per-object geometry and timing at each segment midpoint (O(n) per segment).
    seg = []
    for sa, sb in zip(edges[:-1], edges[1:], strict=True):
        tm = 0.5 * (sa + sb)
        el = propagate_elements(elements, tm, j2=j2)
        if j2:
            _, _, m_dot = j2_secular_rates(elements)
        else:
            m_dot = np.sqrt(MU_EARTH_KM3_S2 / elements.a_km**3)
        P, Q, W = _perifocal_axes(el.raan_rad, el.i_rad, el.argp_rad)
        seg.append((sa, sb, tm, el, P, Q, W, m_dot))

    period_min = float(np.min(_TWO_PI / seg[0][7]))
    period_max = float(np.max(_TWO_PI / seg[0][7]))

    for blk in _blocks(lin, 200_000):
        i, j = pair_from_index(blk, n)
        k = np.zeros(blk.size, dtype=bool)
        for sa, sb, tm, el, P, Q, W, m_dot in seg:
            todo = np.flatnonzero(~k)
            if todo.size == 0:
                break
            ia, ib = i[todo], j[todo]
            K = np.cross(W[ia], W[ib])
            sin_i = np.linalg.norm(K, axis=1)
            coplanar = sin_i < sin_tol
            hhat = K / np.where(coplanar, 1.0, sin_i)[:, None]
            n_occ = int(math.ceil((sb - sa + 2 * time_pad_s + 2 * period_max) / period_min)) + 2
            trains = []
            for idx in (ia, ib):
                trains.append(_node_trains(
                    hhat, sin_i, d, el.e[idx], el.perigee_km[idx], P[idx], Q[idx],
                    el.mean_anomaly_rad[idx], m_dot[idx], tm, sa, sb, time_pad_s, n_occ))
            hit = coplanar.copy()
            for node in (0, 1):
                s1, e1 = trains[0][node]
                s2, e2 = trains[1][node]
                lo = np.maximum(np.maximum(s1[:, :, None], s2[:, None, :]), sa)
                hi = np.minimum(np.minimum(e1[:, :, None], e2[:, None, :]), sb)
                hit |= np.any(lo <= hi, axis=(1, 2))
            k[todo] = hit
        keep[blk] = k
    return FilterResult(
        "time",
        PairMask(ids, keep),
        int(lin.size),
        int(lin.size - np.count_nonzero(keep)),
        {"threshold_km": float(threshold_km), "pad_km": float(pad_km),
         "time_pad_s": float(time_pad_s), "j2": j2, "segment_s": float(segment_s),
         "window_tt_s": (ws, we), "coplanar_tol_rad": coplanar_tol_rad},
        time.perf_counter() - t0,
    )


def _node_trains(hhat, sin_i, d, e, q, P, Q, M_tm, m_dot, tm, sa, sb, time_pad, n_occ):
    """Time intervals near the +hhat node (index 0) and -hhat node (index 1) for one side.

    Returns ``[(starts, ends), (starts, ends)]``, each of shape (pairs, n_occ).
    An object that is always within d of the other plane gets the whole segment.
    """
    nu_h = np.arctan2(np.einsum("ij,ij->i", hhat, Q), np.einsum("ij,ij->i", hhat, P))
    s = d / (q * np.where(sin_i > 0, sin_i, 1e-300))
    always = s >= 0.999
    delta = np.arcsin(np.clip(s, 0.0, 0.999))
    period = _TWO_PI / m_dot
    occ = np.arange(n_occ)[None, :]
    out = []
    for offset in (0.0, np.pi):
        nu_c = nu_h + offset
        m_lo = true_to_mean(nu_c - delta, e)
        m_hi = true_to_mean(nu_c + delta, e)
        width = np.mod(m_hi - m_lo, _TWO_PI) / m_dot
        phase = np.mod(m_lo - M_tm, _TWO_PI) / m_dot  # time from tm to the next window start
        first = tm + phase - period * np.ceil((tm - sa + time_pad + width + phase) / period)
        starts = first[:, None] + period[:, None] * occ - time_pad
        ends = starts + width[:, None] + 2.0 * time_pad
        starts = np.where(always[:, None], np.where(occ == 0, sa, np.inf), starts)
        ends = np.where(always[:, None], np.where(occ == 0, sb, -np.inf), ends)
        out.append((starts, ends))
    return out


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EventMatch:
    """One-to-one matching between reference and candidate events.

    Attributes:
        ref_to_cand: shape (n_ref,), index of the matched candidate row or -1.
        cand_to_ref: shape (n_cand,), index of the matched reference row or -1.
    """

    ref_to_cand: np.ndarray
    cand_to_ref: np.ndarray


def _event_arrays(table: pa.Table):
    return (
        table.column("norad_id_1").to_numpy().astype(np.int64),
        table.column("norad_id_2").to_numpy().astype(np.int64),
        table.column("tca_tt_s").to_numpy().astype(np.float64),
    )


def match_events(
    reference: pa.Table, candidate: pa.Table, *, tca_tol_s: float | None
) -> EventMatch:
    """Match events one to one: same pair, |TCA difference| <= ``tca_tol_s``.

    Within a pair, the closest-in-TCA combinations are matched first (greedy), so
    one candidate never counts for two reference events. ``tca_tol_s=None``
    matches within a pair regardless of TCA (still one to one, nearest first).
    """
    ra, rb, rt = _event_arrays(reference)
    ca, cb, ct = _event_arrays(candidate)
    ref_to_cand = np.full(ra.size, -1, dtype=np.int64)
    cand_to_ref = np.full(ca.size, -1, dtype=np.int64)
    by_pair: dict[tuple[int, int], list[int]] = {}
    for idx, key in enumerate(zip(ca.tolist(), cb.tolist(), strict=True)):
        by_pair.setdefault(key, []).append(idx)
    ref_by_pair: dict[tuple[int, int], list[int]] = {}
    for idx, key in enumerate(zip(ra.tolist(), rb.tolist(), strict=True)):
        ref_by_pair.setdefault(key, []).append(idx)
    tol = math.inf if tca_tol_s is None else float(tca_tol_s)
    for key, r_idx in ref_by_pair.items():
        c_idx = by_pair.get(key)
        if not c_idx:
            continue
        r_arr, c_arr = np.array(r_idx), np.array(c_idx)
        dt = np.abs(rt[r_arr][:, None] - ct[c_arr][None, :])
        rr, cc = np.nonzero(dt <= tol)
        order = np.argsort(dt[rr, cc], kind="stable")
        for o in order:
            r, c = r_arr[rr[o]], c_arr[cc[o]]
            if ref_to_cand[r] < 0 and cand_to_ref[c] < 0:
                ref_to_cand[r] = c
                cand_to_ref[c] = r
    return EventMatch(ref_to_cand, cand_to_ref)


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


@dataclass(frozen=True)
class ScreeningScore:
    """Screening metrics of a candidate event table against a reference.

    Event level: a reference event is found if a candidate event matches it
    (:func:`match_events`). Pair level: a reference pair is found if at least one
    of its reference events is found. False alarms: candidate events with no
    match, and candidate pairs none of whose events matched. Recall by horizon
    uses reference events with ``tau_s <= horizon`` (and pairs with at least one
    such event, found if one of those events is found). Which of these is the
    headline is decided in the screening comparison, not here.
    """

    tca_tol_s: float | None
    n_reference_events: int
    n_candidate_events: int
    n_matched_events: int
    event_recall: float
    n_reference_pairs: int
    n_found_pairs: int
    pair_recall: float
    false_alarm_events: int
    false_alarm_pairs: int
    recall_by_horizon: dict
    missed_event_rows: np.ndarray
    wall_s: float | None = None

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "missed_event_rows"}
        d["missed_event_rows"] = self.missed_event_rows.tolist()
        return d


def screening_metrics(
    reference: pa.Table,
    candidate: pa.Table,
    *,
    tca_tol_s: float | None,
    horizons_s: Sequence[float] = DEFAULT_HORIZONS_S,
    wall_s: float | None = None,
) -> ScreeningScore:
    """Recall (per event and per pair), false alarms and recall by TCA horizon.

    Args:
        reference: event table of the reference screen (truth).
        candidate: event table of the screen under test.
        tca_tol_s: matching tolerance on TCA (None: pair only).
        horizons_s: horizons for nested recall, compared with the reference
            events' ``tau_s``.
        wall_s: the candidate's compute cost, carried into the result.
    """
    match = match_events(reference, candidate, tca_tol_s=tca_tol_s)
    ra, rb, _ = _event_arrays(reference)
    ca, cb, _ = _event_arrays(candidate)
    tau = reference.column("tau_s").to_numpy()
    found = match.ref_to_cand >= 0

    def pair_level(sel: np.ndarray) -> tuple[int, int]:
        keys = list(zip(ra[sel].tolist(), rb[sel].tolist(), strict=True))
        pairs: dict[tuple[int, int], bool] = {}
        for key, f in zip(keys, found[sel].tolist(), strict=True):
            pairs[key] = pairs.get(key, False) or f
        return len(pairs), sum(pairs.values())

    n_ref_pairs, n_found_pairs = pair_level(np.ones(ra.size, dtype=bool))
    cand_pairs: dict[tuple[int, int], bool] = {}
    for key, f in zip(zip(ca.tolist(), cb.tolist(), strict=True),
                      (match.cand_to_ref >= 0).tolist(), strict=True):
        cand_pairs[key] = cand_pairs.get(key, False) or f
    by_h = {}
    for hz in horizons_s:
        sel = tau <= hz
        np_, nf = pair_level(sel)
        by_h[float(hz)] = {
            "n_events": int(sel.sum()),
            "event_recall": _ratio(int(found[sel].sum()), int(sel.sum())),
            "n_pairs": np_,
            "pair_recall": _ratio(nf, np_),
        }
    return ScreeningScore(
        tca_tol_s=tca_tol_s,
        n_reference_events=int(ra.size),
        n_candidate_events=int(ca.size),
        n_matched_events=int(found.sum()),
        event_recall=_ratio(int(found.sum()), int(ra.size)),
        n_reference_pairs=n_ref_pairs,
        n_found_pairs=n_found_pairs,
        pair_recall=_ratio(n_found_pairs, n_ref_pairs),
        false_alarm_events=int(np.count_nonzero(match.cand_to_ref < 0)),
        false_alarm_pairs=int(sum(1 for v in cand_pairs.values() if not v)),
        recall_by_horizon=by_h,
        missed_event_rows=np.flatnonzero(~found),
        wall_s=wall_s,
    )


def filter_metrics(
    reference: pa.Table,
    result: FilterResult | PairMask,
    *,
    horizons_s: Sequence[float] = DEFAULT_HORIZONS_S,
) -> dict:
    """How well a pair filter keeps the reference events, and how much it prunes.

    A reference event is kept if its pair is kept. Returns event and pair recall
    with the reference counts behind them (``n_events``, ``n_pairs``; overall and
    by horizon), the fraction of all pairs pruned (``pruned_fraction``, out of
    ``n_pairs_total``, with ``n_pairs_kept`` kept), the reference pairs the
    filter lost, and its wall-clock cost if known.
    """
    mask = result.mask if isinstance(result, FilterResult) else result
    ra, rb, _ = _event_arrays(reference)
    tau = reference.column("tau_s").to_numpy()
    kept = mask.is_kept(ra, rb) if ra.size else np.zeros(0, dtype=bool)

    def level(sel):
        pairs: dict[tuple[int, int], bool] = {}
        for a, b, kk in zip(ra[sel].tolist(), rb[sel].tolist(), kept[sel].tolist(), strict=True):
            pairs[(a, b)] = kk
        return {
            "n_events": int(sel.sum()),
            "event_recall": _ratio(int(kept[sel].sum()), int(sel.sum())),
            "n_pairs": len(pairs),
            "pair_recall": _ratio(sum(pairs.values()), len(pairs)),
        }

    overall = level(np.ones(ra.size, dtype=bool))
    lost = sorted({(a, b) for a, b, kk in zip(ra.tolist(), rb.tolist(), kept.tolist(),
                                                strict=True) if not kk})
    return {
        **overall,
        "recall_by_horizon": {float(h): level(tau <= h) for h in horizons_s},
        "pruned_fraction": mask.pruned_fraction,
        "n_pairs_total": mask.n_pairs,
        "n_pairs_kept": mask.n_kept,
        "lost_pairs": lost,
        "wall_s": result.wall_s if isinstance(result, FilterResult) else None,
    }
