"""Tests for ``orbit_correction.screening``: synthetic encounters with known answers.

Every trajectory here is built in the test from closed-form motion (straight
lines, circular or Keplerian orbits, a circular orbit with a constant radial
decay standing in for drag), so TCA and miss distance are known analytically or
from a dense, independent minimisation of the exact closed-form range.
No data files are needed.

Accuracy reached at 60 s sampling (checked below):
* straight-line encounters: exact to rounding (Hermite interpolation is exact
  for linear motion), including a 15 km/s encounter half-way between samples;
* circular and eccentric LEO orbits (e <= 0.05), random encounter geometry:
  TCA within 1 ms and miss distance within 1 m of the exact minimum, against a
  dense minimisation of the closed-form range (the asserted bounds). The observed
  maxima over 40 encounters are 6.9e-6 s and 2.6 cm
  (test_accuracy_random_encounters prints them with -s);
* a circular node crossing with an analytic answer: TCA within 1e-5 s and miss
  distance within 1 cm (observed 4e-7 s and 0.3 mm).
"""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pytest
from scipy.optimize import minimize_scalar

from orbit_correction import screening as S
from orbit_correction.types import Trajectory

MU = S.MU_EARTH_KM3_S2
DAY = S.DAY_S
T0 = 7.8e8  # TT seconds since J2000.0 (mid-2024); any epoch works for synthetic orbits
STEP = 60.0


def grid(duration_s: float, step: float = STEP) -> np.ndarray:
    return T0 + np.arange(0.0, duration_s + 0.5 * step, step)


# --------------------------------------------------------------------------------------
# Synthetic trajectory builders
# --------------------------------------------------------------------------------------


def line_traj(nid, p_tca, v, tca, t) -> Trajectory:
    """Straight-line motion through ``p_tca`` at time ``tca``."""
    p_tca, v = np.asarray(p_tca, float), np.asarray(v, float)
    r = p_tca + (t - tca)[:, None] * v
    return Trajectory(nid, t, r, np.broadcast_to(v, r.shape).copy(), "synthetic")


def circular_elements(nid, r, inc_deg, raan_deg, u_at_epoch_deg, epoch) -> S.KeplerElements:
    """Circular orbit; argument of latitude ``u_at_epoch_deg`` at ``epoch`` (argp = 0)."""
    return S.KeplerElements([nid], [epoch], [r], [0.0], [math.radians(inc_deg)],
                            [math.radians(raan_deg)], [0.0], [math.radians(u_at_epoch_deg)])


def kepler_traj(el: S.KeplerElements, t) -> Trajectory:
    r, v = S.states_from_elements(el, t)
    return Trajectory(int(el.norad_id[0]), t, r[0], v[0], "synthetic")


def decaying_circular(nid, r0, k_km_s, inc_deg, raan_deg, t_node, t):
    """Circular orbit whose radius falls linearly, r(t) = r0 - k (t - t[0]).

    The angular rate is the circular rate at the current radius, so the phase is
    the integral of sqrt(mu / r^3). The phase is set so that the object is at its
    ascending node (u = 0) at ``t_node``. A stand-in for drag decay: the radius
    and the timing drift away from what elements at t[0] predict.
    """

    def phase(tt):
        if k_km_s == 0:
            return math.sqrt(MU / r0**3) * (tt - t[0])
        rt = r0 - k_km_s * (tt - t[0])
        return 2.0 * math.sqrt(MU) / k_km_s * (rt**-0.5 - r0**-0.5)

    u0 = -phase(t_node)
    raan, inc = math.radians(raan_deg), math.radians(inc_deg)
    P = np.array([math.cos(raan), math.sin(raan), 0.0])
    Q = np.array([-math.cos(inc) * math.sin(raan), math.cos(inc) * math.cos(raan), math.sin(inc)])
    rt = r0 - k_km_s * (t - t[0])
    u = u0 + np.array([phase(x) for x in t])
    udot = np.sqrt(MU / rt**3)
    cu, su = np.cos(u)[:, None], np.sin(u)[:, None]
    r = rt[:, None] * (cu * P + su * Q)
    v = -k_km_s * (cu * P + su * Q) + (rt * udot)[:, None] * (-su * P + cu * Q)
    return Trajectory(nid, t, r, v, "synthetic")


def exact_minima(pos_a, pos_b, t_lo, t_hi, threshold_km, dense_step=1.0):
    """Independent reference: local range minima under the threshold, from closed forms.

    ``pos_a`` / ``pos_b`` map an array of times to positions (T, 3). Samples the
    range every ``dense_step`` seconds, then refines every discrete local minimum
    with a bounded scalar minimisation.
    """
    tt = np.arange(t_lo, t_hi + dense_step / 2, dense_step)
    d = np.linalg.norm(pos_a(tt) - pos_b(tt), axis=1)
    out = []
    for k in np.flatnonzero((d[1:-1] <= d[:-2]) & (d[1:-1] < d[2:])) + 1:
        # Minimise over the offset from tt[k]: scipy's bounded Brent scales its
        # tolerance with |x|, which would be ~10 s at t ~ 8e8 s.
        tk = tt[k]
        res = minimize_scalar(
            lambda x, tk=tk: float(np.linalg.norm(pos_a(np.array([tk + x]))
                                                  - pos_b(np.array([tk + x])))),
            bounds=(-dense_step, dense_step), method="bounded", options={"xatol": 1e-9},
        )
        if res.fun <= threshold_km:
            out.append((tk + res.x, res.fun))
    return out


def kepler_pos(el):
    return lambda tt: S.states_from_elements(el, tt)[0][0]


# --------------------------------------------------------------------------------------
# Engine: straight lines
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("substeps", [1, 2, 4])
@pytest.mark.parametrize("tca_offset", [1234.5, 1200.0, 1229.99, 1201.0])
def test_straight_line_15_km_s_between_samples(substeps, tca_offset):
    """A 15 km/s fly-by anywhere in an interval: found, with exact TCA and miss distance.

    At the nearest sample (up to 30 s away) the two objects are up to 450 km
    apart, so a sample-only search with a 5 km threshold would miss it.
    """
    t = grid(2400.0)
    tca = T0 + tca_offset
    a = line_traj(1, [7000.0, 0, 0], [0, 7.5, 0], tca, t)
    b = line_traj(2, [7000.0, 0, 2.0], [0, -7.5, 0], tca, t)
    ev = S.screen([a, b], 5.0, substeps=substeps).events.to_pylist()
    assert len(ev) == 1
    assert ev[0]["tca_tt_s"] == pytest.approx(tca, abs=1e-6)
    assert ev[0]["miss_km"] == pytest.approx(2.0, abs=1e-9)
    assert ev[0]["rel_speed_km_s"] == pytest.approx(15.0, abs=1e-9)
    assert ev[0]["tau_s"] == pytest.approx(tca_offset, abs=1e-6)


def test_minimum_exactly_on_a_sample_is_counted_once():
    t = grid(2400.0)
    tca = T0 + 1200.0  # a sample time, shared by two intervals
    a = line_traj(1, [7000.0, 0, 0], [0, 7.5, 0], tca, t)
    b = line_traj(2, [7000.0, 0, 1.0], [0, -7.5, 0], tca, t)
    for chunk in (1, 7, 100):  # also on a chunk boundary
        ev = S.screen([a, b], 5.0, chunk_steps=chunk).events
        assert ev.num_rows == 1


def test_threshold_boundary_is_inclusive_and_sharp():
    t = grid(2400.0)
    tca = T0 + 1234.5
    for miss, expect in ((4.999, 1), (5.0, 1), (5.001, 0)):
        a = line_traj(1, [7000.0, 0, 0], [0, 7.5, 0], tca, t)
        b = line_traj(2, [7000.0, 0, miss], [0, -7.5, 0], tca, t)
        assert S.screen([a, b], 5.0).events.num_rows == expect


# --------------------------------------------------------------------------------------
# Engine: circular orbits crossing at a known node
# --------------------------------------------------------------------------------------


def node_pair(r_a=7000.0, r_b=7002.0, inc_a=45.0, inc_b=60.0, t_node=T0 + 3017.3, lag_deg=0.0):
    """Two circular orbits with the same node, both at the ascending node at ``t_node``.

    ``lag_deg`` puts object B that far behind the node instead. With no lag, the
    range is an even function of time about ``t_node`` (both arguments of
    latitude are odd in time), so TCA is exactly ``t_node`` and the miss distance
    exactly ``r_b - r_a``.
    """
    ea = circular_elements(10, r_a, inc_a, 0.0, 0.0, t_node)
    eb = circular_elements(11, r_b, inc_b, 0.0, -lag_deg, t_node)
    return ea, eb


def test_circular_node_crossing_known_tca_and_miss():
    t = grid(6000.0)
    t_node = T0 + 3017.3
    ea, eb = node_pair(t_node=t_node)
    ev = S.screen([kepler_traj(ea, t), kepler_traj(eb, t)], 5.0).events.to_pylist()
    assert len(ev) == 1
    assert ev[0]["tca_tt_s"] == pytest.approx(t_node, abs=1e-5)
    assert ev[0]["miss_km"] == pytest.approx(2.0, abs=1e-5)
    # relative speed: two 7000 km circular speeds 15 deg apart (plus a tiny radial difference)
    v = math.sqrt(MU / 7000.0)
    assert ev[0]["rel_speed_km_s"] == pytest.approx(2 * v * math.sin(math.radians(7.5)), rel=1e-3)


def test_no_approach_under_threshold():
    """Same orbits, B a quarter orbit behind: the closest approach is thousands of km."""
    t = grid(6000.0)
    ea, eb = node_pair(lag_deg=90.0)
    res = S.screen([kepler_traj(ea, t), kepler_traj(eb, t)], 5.0)
    assert res.events.num_rows == 0
    assert res.events.schema.equals(S.EVENT_SCHEMA)


def test_several_minima_in_window_match_dense_reference():
    """A pair that meets near each node passage, drifting apart slowly: every minimum found."""
    t = grid(DAY)
    ea, eb = node_pair(t_node=T0 + 600.0)
    ref = exact_minima(kepler_pos(ea), kepler_pos(eb), t[0], t[-1], 40.0)
    ev = S.screen([kepler_traj(ea, t), kepler_traj(eb, t)], 40.0).events
    assert len(ref) >= 4  # several minima under 40 km in the day
    assert ev.num_rows == len(ref)
    tca = ev.column("tca_tt_s").to_numpy()
    miss = ev.column("miss_km").to_numpy()
    for (rt, rm), et, em in zip(ref, tca, miss, strict=True):
        assert et == pytest.approx(rt, abs=1e-3)
        assert em == pytest.approx(rm, abs=1e-4)


def test_symmetry_order_does_not_matter():
    t = grid(DAY)
    ea, eb = node_pair(t_node=T0 + 600.0)
    ec = circular_elements(12, 7001.0, 52.0, 0.0, 0.0, T0 + 600.0)
    ta, tb, tc = (kepler_traj(e, t) for e in (ea, eb, ec))
    base = S.screen([ta, tb, tc], 40.0).events
    assert base.num_rows > 0
    for order in ([tb, ta, tc], [tc, tb, ta], [tb, tc, ta]):
        other = S.screen(order, 40.0).events
        assert other.column("norad_id_1").to_pylist() == base.column("norad_id_1").to_pylist()
        assert other.column("norad_id_2").to_pylist() == base.column("norad_id_2").to_pylist()
        np.testing.assert_allclose(other.column("tca_tt_s").to_numpy(),
                                   base.column("tca_tt_s").to_numpy(), atol=1e-6, rtol=0)
        np.testing.assert_allclose(other.column("miss_km").to_numpy(),
                                   base.column("miss_km").to_numpy(), atol=1e-9, rtol=0)
    assert np.all(base.column("norad_id_1").to_numpy() < base.column("norad_id_2").to_numpy())


def test_horizon_dependent_threshold():
    """Each minimum is kept exactly when its miss distance is within d(tau) at its TCA."""
    t = grid(DAY)
    ea, eb = node_pair(t_node=T0 + 600.0)
    ref = exact_minima(kepler_pos(ea), kepler_pos(eb), t[0], t[-1], 1e9)
    ref_tau = np.array([x[0] for x in ref]) - T0
    ref_miss = np.array([x[1] for x in ref])

    # The misses grow by ~9.3 km per half orbit. A ramp from 3 km at 200 km/day grows
    # slower: early minima pass, later ones fail, and a constant at the ramp's start or
    # end would keep fewer or more.
    def d(tau):
        return 3.0 + 200.0 * np.asarray(tau) / DAY

    expected = ref_miss <= d(ref_tau)
    assert expected.any() and (~expected).any()
    assert np.count_nonzero(ref_miss <= d(0.0)) < np.count_nonzero(expected) < np.count_nonzero(
        ref_miss <= d(DAY))

    trajs = [kepler_traj(ea, t), kepler_traj(eb, t)]
    ev = S.screen(trajs, d).events
    np.testing.assert_allclose(ev.column("tau_s").to_numpy(), ref_tau[expected], atol=1e-3)
    np.testing.assert_allclose(ev.column("threshold_km").to_numpy(),
                               d(ev.column("tau_s").to_numpy()), rtol=1e-12)
    # The same through the volume object, and with the epoch moved: tau shifts with it.
    ev2 = S.screen(trajs, S.SphericalThreshold(lambda tau: d(tau + 3600.0)),
                   epoch_tt_s=T0 + 3600.0).events
    np.testing.assert_allclose(ev2.column("tau_s").to_numpy() + 3600.0,
                               ev.column("tau_s").to_numpy(), atol=1e-6)


def test_step_threshold_between_minima():
    t = grid(DAY)
    ea, eb = node_pair(t_node=T0 + 600.0)
    ref = exact_minima(kepler_pos(ea), kepler_pos(eb), t[0], t[-1], 1e9)
    tau = np.array([x[0] for x in ref]) - T0
    cut = 0.5 * (tau[2] + tau[3])

    def step(x):
        return np.where(np.asarray(x) < cut, 1e4, 1.0)  # 1 km after the cut: nothing passes

    ev = S.screen([kepler_traj(ea, t), kepler_traj(eb, t)], step).events
    np.testing.assert_allclose(ev.column("tau_s").to_numpy(), tau[:3], atol=1e-3)


def test_accuracy_random_encounters():
    """Accuracy at 60 s sampling over random LEO encounter geometries (e <= 0.05).

    Each pair is built to pass close to a common point at a random time, which
    lies anywhere inside a sample interval. The reference minimum comes from a
    dense, bounded minimisation of the exact closed-form range.
    """
    rng = np.random.default_rng(3)
    t = grid(4 * 3600.0)
    dtca, dmiss, n_found = [], [], 0
    for trial in range(40):
        t_meet = T0 + rng.uniform(3600.0, 3 * 3600.0)
        r_meet = rng.uniform(6700.0, 8000.0)
        inc_a, inc_b = rng.uniform(0, 110, 2)
        raan = rng.uniform(0, 360)
        ea = circular_elements(100 + 2 * trial, r_meet, inc_a, raan, 0.0, t_meet)
        # B: eccentric, perigee placed so it passes through the same node point
        e = rng.uniform(0.0, 0.05)
        nu = rng.uniform(0, 2 * math.pi)
        a = r_meet * (1 + e * math.cos(nu)) / (1 - e**2) + rng.uniform(-3, 3)
        eb = S.KeplerElements([101 + 2 * trial], [t_meet], [a], [e], [math.radians(inc_b)],
                              [math.radians(raan)], [-nu], [float(S.true_to_mean(nu, e))])
        ref = exact_minima(kepler_pos(ea), kepler_pos(eb), t[0], t[-1], 10.0, dense_step=0.5)
        ev = S.screen([kepler_traj(ea, t), kepler_traj(eb, t)], 10.0).events
        assert ev.num_rows == len(ref)
        for (rt, rm), et, em in zip(ref, ev.column("tca_tt_s").to_numpy(),
                                    ev.column("miss_km").to_numpy(), strict=True):
            dtca.append(abs(et - rt))
            dmiss.append(abs(em - rm))
            n_found += 1
    assert n_found >= 30
    print(f"\n{n_found} encounters: max |dTCA| = {max(dtca):.2e} s, "
          f"max |dmiss| = {max(dmiss) * 1e3:.3f} m")
    assert max(dtca) < 1e-3
    assert max(dmiss) < 1e-3


def test_window_and_keep_mask():
    t = grid(DAY)
    ea, eb = node_pair(t_node=T0 + 600.0)
    ec = circular_elements(12, 7001.0, 52.0, 0.0, 0.0, T0 + 600.0)
    trajs = [kepler_traj(e, t) for e in (ea, eb, ec)]
    full = S.screen(trajs, 40.0).events
    assert set(zip(full.column("norad_id_1").to_pylist(), full.column("norad_id_2").to_pylist(),
                   strict=True)) == {(10, 11), (10, 12), (11, 12)}

    mask = S.PairMask.all_pairs([10, 11, 12])
    keep = mask.keep.copy()
    keep[S.pair_index(0, 1, 3)] = False  # drop (10, 11)
    part = S.screen(trajs, 40.0, keep=S.PairMask([10, 11, 12], keep)).events
    pairs = set(zip(part.column("norad_id_1").to_pylist(), part.column("norad_id_2").to_pylist(),
                    strict=True))
    assert pairs == {(10, 12), (11, 12)}
    with pytest.raises(ValueError):
        S.screen(trajs, 40.0, keep=S.PairMask([11, 10, 12], keep))

    lo, hi = T0 + 3 * 3600.0, T0 + 9 * 3600.0
    win = S.screen(trajs, 40.0, window_tt_s=(lo, hi)).events
    tca = full.column("tca_tt_s").to_numpy()
    assert win.num_rows == np.count_nonzero((tca >= lo) & (tca <= hi))


def test_input_validation():
    t = grid(600.0)
    a = line_traj(1, [7000.0, 0, 0], [0, 7.5, 0], T0, t)
    b = line_traj(2, [7000.0, 0, 1], [0, -7.5, 0], T0, t)
    with pytest.raises(ValueError):
        S.screen([a], 5.0)
    with pytest.raises(ValueError):  # duplicate NORAD ID
        S.screen([a, line_traj(1, [7000.0, 0, 1], [0, -7.5, 0], T0, t)], 5.0)
    with pytest.raises(ValueError):  # different grids
        S.screen([a, line_traj(2, [7000.0, 0, 1], [0, -7.5, 0], T0, t + 1.0)], 5.0)
    t_bad = t.copy()
    t_bad[3] += 10.0
    with pytest.raises(ValueError):  # non-uniform grid
        S.screen([line_traj(1, [7000.0, 0, 0], [0, 7.5, 0], T0, t_bad),
                  line_traj(2, [7000.0, 0, 1], [0, -7.5, 0], T0, t_bad)], 5.0)
    with pytest.raises(ValueError):
        S.screen([a, b], -1.0)
    with pytest.raises(TypeError):
        S.screen([a, b], "5 km")


# --------------------------------------------------------------------------------------
# Elements and synthetic trajectories
# --------------------------------------------------------------------------------------


def test_pair_index_round_trip():
    for n in (2, 3, 10, 2001):
        i, j = np.triu_indices(n, 1)
        k = S.pair_index(i, j, n)
        np.testing.assert_array_equal(k, np.arange(S.pair_count(n)))
        i2, j2 = S.pair_from_index(k, n)
        np.testing.assert_array_equal(i2, i)
        np.testing.assert_array_equal(j2, j)


def test_elements_state_round_trip():
    el = S.synthetic_leo_elements(200, T0, seed=5)
    r, v = S.states_from_elements(el, np.array([T0]))
    back = S.elements_from_states(el.norad_id, el.epoch_tt_s, r[:, 0], v[:, 0])
    np.testing.assert_allclose(back.a_km, el.a_km, rtol=1e-12)
    np.testing.assert_allclose(back.e, el.e, atol=1e-12)
    np.testing.assert_allclose(back.i_rad, el.i_rad, atol=1e-12)
    r2, v2 = S.states_from_elements(back, np.array([T0]))
    np.testing.assert_allclose(r2, r, atol=1e-7)
    np.testing.assert_allclose(v2, v, atol=1e-10)


def test_synthetic_velocity_is_the_derivative_of_position():
    """With J2 drift on, the velocity still matches the numerical derivative of position."""
    el = S.synthetic_leo_elements(20, T0, seed=1)
    t = T0 + 1000.0 + np.array([-2.0, -1.0, 0.0, 1.0, 2.0])  # exact in float64
    r, v = S.states_from_elements(el, t, j2=True)
    num = (-r[:, 4] + 8 * r[:, 3] - 8 * r[:, 1] + r[:, 0]) / 12.0  # 4th-order central
    np.testing.assert_allclose(num, v[:, 2], atol=1e-9)


def test_synthetic_catalog_is_leo():
    el = S.synthetic_leo_elements(5000, T0, seed=0)
    assert np.all(el.perigee_km >= S.R_EARTH_KM + 300.0 - 1e-9)
    assert np.all(el.apogee_km <= S.R_EARTH_KM + 2000.0 + 1e-9)


# --------------------------------------------------------------------------------------
# Classical filters
# --------------------------------------------------------------------------------------


def drag_scenario(k_km_per_day: float, r0_b: float, t_meet_days: float = 5.0):
    """Object A circular at 6778 km; B circular from ``r0_b``, decaying; both at the node at
    ``t_meet_days`` over a 7-day window at 60 s. Returns trajectories and the truth events."""
    t = grid(7 * DAY)
    t_meet = T0 + t_meet_days * DAY
    a = decaying_circular(1, 6778.0, 0.0, 51.6, 0.0, t_meet, t)
    b = decaying_circular(2, r0_b, k_km_per_day / DAY, 97.0, 0.0, t_meet, t)
    truth = S.screen([a, b], 5.0).events
    return t, [a, b], truth


@pytest.fixture(scope="module")
def drag_crossing():
    """B starts 8 km above A and decays 2 km/day, so the radial shells only overlap
    from day 4. The truth encounter is at day 5, about 2 km."""
    return drag_scenario(2.0, 6786.0)


def test_drag_crossing_truth_event(drag_crossing):
    """The designed encounter (day 5, ~2 km) is there; a second, nearby node passage
    also comes within 5 km. Both belong to the one pair."""
    _, _, truth = drag_crossing
    rows = truth.to_pylist()
    assert {(r["norad_id_1"], r["norad_id_2"]) for r in rows} == {(1, 2)}
    designed = [r for r in rows if abs(r["tau_s"] - 5 * DAY) < 5.0]
    assert len(designed) == 1
    assert designed[0]["miss_km"] == pytest.approx(2.0, abs=0.05)


def test_apogee_perigee_filter_matches_brute_force():
    rng = np.random.default_rng(2)
    n = 60
    q = rng.uniform(6700, 7500, n)
    Q = q + rng.exponential(30, n)
    res = S.apogee_perigee_filter(np.arange(1, n + 1), q, Q, threshold_km=5.0, pad_km=3.0)
    i, j = np.triu_indices(n, 1)
    expect = np.maximum(q[i], q[j]) - np.minimum(Q[i], Q[j]) <= 8.0
    np.testing.assert_array_equal(res.mask.keep, expect)
    assert res.n_pruned_here == np.count_nonzero(~expect)
    assert res.pruned_fraction == pytest.approx(np.mean(~expect))


def test_apogee_perigee_filter_zero_pad_miss_and_adequate_pad(drag_crossing):
    """Documented miss: elements at the window start don't know about the decay.

    At the start, B's shell is 8 km above A's, more than the 5 km threshold, so a
    zero pad prunes the pair, yet the two meet at 2 km on day 5. A 3 km pad (or
    Woodburn's two-sample variant, start and end of the window) keeps it.
    """
    t, trajs, truth = drag_crossing
    el0 = S.osculating_elements(trajs, t[0])
    ids, q, Q = S.apsides_from_elements(el0)
    zero = S.apogee_perigee_filter(ids, q, Q, threshold_km=5.0, pad_km=0.0)
    assert S.filter_metrics(truth, zero)["event_recall"] == 0.0
    padded = S.apogee_perigee_filter(ids, q, Q, threshold_km=5.0, pad_km=3.5)
    assert S.filter_metrics(truth, padded)["event_recall"] == 1.0
    ids2, q2, Q2 = S.apsides_from_elements(el0, S.osculating_elements(trajs, t[-1]))
    two = S.apogee_perigee_filter(ids2, q2, Q2, threshold_km=5.0, pad_km=0.0)
    assert S.filter_metrics(truth, two)["event_recall"] == 1.0
    ids3, rmin, rmax = S.radial_extent(trajs)  # sampled extent (states) also keeps it
    assert S.apogee_perigee_filter(ids3, rmin, rmax, threshold_km=5.0, pad_km=0.0).mask.keep[0]


def test_orbit_path_distance_circular_and_coplanar():
    ea = circular_elements(1, 7000.0, 45.0, 10.0, 33.0, T0)
    eb = circular_elements(2, 7004.0, 80.0, 70.0, 200.0, T0)
    dist, cop = S.orbit_path_distance(ea, eb)
    assert dist[0] == pytest.approx(4.0, abs=1e-9)
    assert not cop[0]
    ec = circular_elements(3, 7100.0, 45.0, 10.0, 0.0, T0)
    dist, cop = S.orbit_path_distance(ea, ec)
    assert cop[0] and dist[0] == 0.0


def test_orbit_path_distance_matches_brute_force():
    """Near-circular LEO pairs (e <= 0.03): Newton from the relative nodes reaches the
    global minimum distance between the paths, found independently by a grid search over
    both eccentric anomalies and BFGS from the 12 best grid points."""
    from scipy.optimize import minimize

    rng = np.random.default_rng(11)
    n = 30
    el = S.synthetic_leo_elements(2 * n, T0, seed=11)
    a_idx, b_idx = np.arange(0, 2 * n, 2), np.arange(1, 2 * n, 2)
    a_km = el.a_km.copy()
    a_km[b_idx] = a_km[a_idx] + rng.uniform(-20, 20, n)  # similar radii: small distances
    el = S.KeplerElements(el.norad_id, el.epoch_tt_s, a_km, np.minimum(el.e * 5, 0.03),
                          el.i_rad, el.raan_rad, el.argp_rad, el.mean_anomaly_rad)
    ea, eb = el.take(a_idx), el.take(b_idx)
    dist, cop = S.orbit_path_distance(ea, eb)
    E = np.linspace(0, 2 * np.pi, 361)

    def path(e_, k):
        P, Q, _ = S._perifocal_axes(e_.raan_rad[k], e_.i_rad[k], e_.argp_rad[k])
        a, ecc = e_.a_km[k], e_.e[k]
        b = a * math.sqrt(1 - ecc**2)
        return lambda x: (a * (np.cos(x) - ecc))[..., None] * P + (b * np.sin(x))[..., None] * Q

    checked = 0
    for k in range(n):
        if cop[k]:
            continue
        x1, x2 = path(ea, k), path(eb, k)
        dd = np.linalg.norm(x1(E)[:, None, :] - x2(E)[None, :, :], axis=2)
        best = math.inf
        for flat in np.argsort(dd, axis=None)[:12]:
            g1, g2 = np.unravel_index(flat, dd.shape)
            res = minimize(lambda z, x1=x1, x2=x2: float(np.sum((x1(z[0]) - x2(z[1])) ** 2)),
                           [E[g1], E[g2]], method="BFGS", options={"gtol": 1e-10})
            best = min(best, math.sqrt(res.fun))
        assert dist[k] == pytest.approx(best, abs=1e-6)
        checked += 1
    assert checked >= 25


def test_orbit_path_filter_zero_pad_miss_and_adequate_pad(drag_crossing):
    """Documented miss: the path distance from window-start elements is 8 km, above the
    5 km threshold, but decay brings the paths together. A 3.5 km pad keeps the pair, and
    so does Woodburn's start/middle/end sampling of osculating elements (mid: ~1 km)."""
    t, trajs, truth = drag_crossing
    el0 = S.osculating_elements(trajs, t[0])
    zero = S.orbit_path_filter(el0, [t[0], t[t.size // 2], t[-1]], threshold_km=5.0,
                               pad_km=0.0, j2=False)
    assert S.filter_metrics(truth, zero)["event_recall"] == 0.0
    padded = S.orbit_path_filter(el0, [t[0]], threshold_km=5.0, pad_km=3.5, j2=False)
    assert S.filter_metrics(truth, padded)["event_recall"] == 1.0
    sampled = [S.osculating_elements(trajs, tt) for tt in (t[0], t[t.size // 2], t[-1])]
    woodburn = S.orbit_path_filter(sampled, threshold_km=5.0, pad_km=0.0)
    assert S.filter_metrics(truth, woodburn)["event_recall"] == 1.0


@pytest.fixture(scope="module")
def drag_timing():
    """A and B start at the same radius; B decays 0.4 km/day. Shells and paths always
    overlap, but B runs ahead of what its start elements predict (about 96 s by day 5).
    The truth encounter is at day 5, about 2 km."""
    return drag_scenario(0.4, 6778.0)


def test_drag_timing_truth_event(drag_timing):
    _, _, truth = drag_timing
    assert truth.num_rows == 1
    assert truth.column("miss_km").to_numpy()[0] == pytest.approx(2.0, abs=0.05)


def test_time_filter_zero_pad_miss_and_adequate_pads(drag_timing):
    """Documented miss: windows from start elements put B at the node ~96 s late by
    day 5, while each node window is only ~2 s wide at d = 5 km. A 120 s time pad, or
    a 1000 km distance pad (window +/- 185 s), keeps the pair."""
    t, trajs, truth = drag_timing
    el0 = S.osculating_elements(trajs, t[0])
    window = (t[0], t[-1])
    zero = S.time_filter(el0, window, threshold_km=5.0, pad_km=0.0, j2=False)
    assert S.filter_metrics(truth, zero)["event_recall"] == 0.0
    tpad = S.time_filter(el0, window, threshold_km=5.0, pad_km=0.0, time_pad_s=120.0, j2=False)
    assert S.filter_metrics(truth, tpad)["event_recall"] == 1.0
    dpad = S.time_filter(el0, window, threshold_km=5.0, pad_km=1000.0, j2=False)
    assert S.filter_metrics(truth, dpad)["event_recall"] == 1.0


def test_time_filter_keeps_exact_node_crossing():
    """With elements that describe the motion exactly, a zero-pad time filter keeps the
    pair that meets at the node, and prunes the same pair with B a quarter orbit behind."""
    t = grid(DAY)
    ea, eb = node_pair(t_node=T0 + 5000.0)
    truth = S.screen([kepler_traj(ea, t), kepler_traj(eb, t)], 5.0).events
    assert truth.num_rows >= 1
    el = S.KeplerElements(*(np.concatenate([getattr(ea, f), getattr(eb, f)])
                            for f in S._ELEMENT_FIELDS))
    res = S.time_filter(el, (t[0], t[-1]), threshold_km=5.0, pad_km=0.0, j2=False)
    assert res.mask.keep[0]
    _, eb_far = node_pair(t_node=T0 + 5000.0, lag_deg=90.0)
    el_far = S.KeplerElements(*(np.concatenate([getattr(ea, f), getattr(eb_far, f)])
                                for f in S._ELEMENT_FIELDS))
    assert not S.time_filter(el_far, (t[0], t[-1]), threshold_km=5.0, pad_km=0.0,
                             j2=False).mask.keep[0]


def test_filters_lose_nothing_when_elements_are_exact():
    """On a synthetic catalog whose motion is exactly two-body plus J2 secular drift, the
    elements describe the orbits perfectly. A/P and orbit-path filters at zero pad and the
    time filter with a 1 km pad then keep every truth pair (sanity check of the geometry)."""
    t = grid(DAY)
    el = S.synthetic_leo_elements(300, T0, seed=8)
    trajs = S.trajectories_from_elements(el, t, j2=True)
    truth = S.screen(trajs, 20.0).events
    assert truth.num_rows >= 3
    ids, q, Q = S.apsides_from_elements(el)
    ap = S.apogee_perigee_filter(ids, q, Q, threshold_km=20.0, pad_km=0.0)
    op = S.orbit_path_filter(el, [t[0], t[t.size // 2], t[-1]], threshold_km=20.0,
                             pad_km=0.0, prior=ap.mask)
    tf = S.time_filter(el, (t[0], t[-1]), threshold_km=20.0, pad_km=1.0, prior=op.mask)
    for res in (ap, op, tf):
        assert S.filter_metrics(truth, res)["event_recall"] == 1.0, res.name
    assert ap.pruned_fraction < op.pruned_fraction < tf.pruned_fraction
    # The engine restricted to the filtered pairs finds the same events.
    kept = S.screen(trajs, 20.0, keep=tf.mask).events
    assert S.screening_metrics(truth, kept, tca_tol_s=1e-3).event_recall == 1.0


# --------------------------------------------------------------------------------------
# Metrics on hand-built tables
# --------------------------------------------------------------------------------------


def table(rows):
    """rows: (id_a, id_b, tau_days, miss_km); epoch T0."""
    a, b, tau, miss = (np.array(c) for c in zip(*rows, strict=True)) if rows else ([],) * 4
    tau_s = np.asarray(tau, float) * DAY
    return S.make_event_table(a, b, T0 + tau_s, tau_s, miss, np.full(len(tau_s), 10.0),
                              np.full(len(tau_s), 5.0))


@pytest.fixture
def ref_cand():
    ref = table([
        (1, 2, 0.5, 1.0),   # r0: matched by c0 (3 s off)
        (2, 1, 2.5, 2.0),   # r1: same pair, second event: matched by c1 (8 s off)
        (1, 3, 5.0, 3.0),   # r2: candidate 100 s off: matched only with a wide tolerance
        (2, 3, 6.9, 4.0),   # r3: no candidate for this pair
    ])
    cand = table([
        (1, 2, 0.5 + 3 / DAY, 1.1),
        (1, 2, 2.5 + 8 / DAY, 2.1),
        (1, 2, 0.5 + 5 / DAY, 1.2),   # duplicate near r0: a false alarm, not a second match
        (1, 3, 5.0 + 100 / DAY, 3.0),
        (2, 4, 1.0, 4.5),             # pair with no reference event
    ])
    return ref, cand


def test_make_event_table_orders_pairs_and_rows():
    tb = table([(5, 2, 2.0, 1.0), (3, 9, 1.0, 1.0)])
    assert tb.column("norad_id_1").to_pylist() == [3, 2]
    assert tb.column("norad_id_2").to_pylist() == [9, 5]
    assert tb.schema.equals(S.EVENT_SCHEMA)
    with pytest.raises(ValueError):
        table([(1, 1, 0.0, 1.0)])


def test_screening_metrics_tight_tolerance(ref_cand):
    ref, cand = ref_cand
    sc = S.screening_metrics(ref, cand, tca_tol_s=10.0)
    assert sc.n_reference_events == 4 and sc.n_candidate_events == 5
    assert sc.n_matched_events == 2
    assert sc.event_recall == pytest.approx(0.5)
    # pairs: (1,2) found; (1,3) not (100 s); (2,3) not
    assert sc.n_reference_pairs == 3 and sc.n_found_pairs == 1
    assert sc.pair_recall == pytest.approx(1 / 3)
    # unmatched candidates: the duplicate, the 100 s-off one, (2,4)
    assert sc.false_alarm_events == 3
    # candidate pairs with no matched event: (1,3) and (2,4)
    assert sc.false_alarm_pairs == 2
    by = sc.recall_by_horizon
    assert by[DAY]["n_events"] == 1 and by[DAY]["event_recall"] == 1.0
    assert by[3 * DAY]["n_events"] == 2 and by[3 * DAY]["event_recall"] == 1.0
    assert by[3 * DAY]["n_pairs"] == 1 and by[3 * DAY]["pair_recall"] == 1.0
    assert by[7 * DAY]["event_recall"] == pytest.approx(0.5)
    assert sorted(sc.missed_event_rows.tolist()) == [2, 3]


def test_screening_metrics_loose_and_pair_only_matching(ref_cand):
    ref, cand = ref_cand
    wide = S.screening_metrics(ref, cand, tca_tol_s=200.0)
    assert wide.n_matched_events == 3 and wide.pair_recall == pytest.approx(2 / 3)
    any_t = S.screening_metrics(ref, cand, tca_tol_s=None)
    assert any_t.n_matched_events == 3
    # one-to-one: three (1,2) candidates can match at most the two (1,2) reference events
    assert any_t.false_alarm_events == 2


def test_screening_metrics_empty_tables():
    empty = S.empty_event_table()
    sc = S.screening_metrics(empty, empty, tca_tol_s=1.0)
    assert math.isnan(sc.event_recall) and math.isnan(sc.pair_recall)
    assert sc.false_alarm_events == 0
    ref = table([(1, 2, 0.5, 1.0)])
    sc = S.screening_metrics(ref, empty, tca_tol_s=1.0)
    assert sc.event_recall == 0.0 and sc.pair_recall == 0.0


def test_filter_metrics(ref_cand):
    ref, _ = ref_cand
    ids = [1, 2, 3, 4]
    keep = np.ones(S.pair_count(4), dtype=bool)
    keep[S.pair_index(0, 2, 4)] = False  # prune (1, 3)
    keep[S.pair_index(2, 3, 4)] = False  # prune (3, 4): no reference event
    m = S.filter_metrics(ref, S.PairMask(ids, keep))
    assert m["event_recall"] == pytest.approx(3 / 4)
    assert m["pair_recall"] == pytest.approx(2 / 3)
    assert m["lost_pairs"] == [(1, 3)]
    assert m["pruned_fraction"] == pytest.approx(2 / 6)
    assert m["n_pairs"] == 3 and m["n_events"] == 4  # reference counts
    assert m["n_pairs_total"] == 6 and m["n_pairs_kept"] == 4
    assert m["recall_by_horizon"][3 * DAY]["event_recall"] == 1.0
    assert m["recall_by_horizon"][7 * DAY]["event_recall"] == pytest.approx(3 / 4)


def test_screening_result_reports_cost():
    t = grid(1200.0)
    a = line_traj(1, [7000.0, 0, 0], [0, 7.5, 0], T0 + 600, t)
    b = line_traj(2, [7000.0, 0, 1], [0, -7.5, 0], T0 + 600, t)
    res = S.screen([a, b], 5.0)
    assert res.stats["wall_s"] > 0
    assert res.stats["rel_speed_bound_km_s"] >= 15.0
    sc = S.screening_metrics(res.events, res.events, tca_tol_s=0.0, wall_s=res.stats["wall_s"])
    assert sc.event_recall == 1.0 and sc.wall_s == res.stats["wall_s"]
    assert isinstance(res.events, pa.Table)
