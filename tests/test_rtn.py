"""Tests for the curvilinear RTN conversion in ``orbit_correction.residuals``.

The definition under test is the contract in ``orbit_correction.types`` (module
docstring). Checks:

* round trips in both directions on random LEO states (e up to 0.3) with along-track
  separations up to 2,000 km and radial and cross-track up to +-50 km, to 1e-9 km;
* agreement with a literal, row-by-row transcription of the contract's formulas
  (cross products and arcsin, as written there), which the module doesn't use;
* sign conventions;
* agreement with rectilinear RTN at small separations, with a difference that
  scales as separation^2 / r;
* a large along-track case where rectilinear RTN leaks ~81 km into radial and the
  curvilinear residual doesn't;
* errors for degenerate inputs, including a velocity a hair off radial;
* forward and inverse agreeing at the angle limits: every residual the forward map
  returns, the inverse accepts.

All states are synthetic (two-body Keplerian), so no fixture files are needed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from orbit_correction import residuals as R

MU = 398600.4418  # km^3/s^2; any value works, the conversion is purely geometric
RE = 6378.137
TOL_KM = 1e-9


def kepler_to_state(a, e, inc, raan, argp, nu, mu=MU):
    """Position and velocity (km, km/s) from classical elements (radians), vectorized."""
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu))
    r_pf = np.stack([r * np.cos(nu), r * np.sin(nu), np.zeros_like(nu)], axis=-1)
    v_pf = np.stack(
        [-np.sqrt(mu / p) * np.sin(nu), np.sqrt(mu / p) * (e + np.cos(nu)), np.zeros_like(nu)],
        axis=-1,
    )
    cO, sO, ci, si, cw, sw = (
        np.cos(raan), np.sin(raan), np.cos(inc), np.sin(inc), np.cos(argp), np.sin(argp)
    )
    rot = np.stack(
        [
            np.stack([cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si], axis=-1),
            np.stack([sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si], axis=-1),
            np.stack([sw * si, cw * si, ci], axis=-1),
        ],
        axis=-2,
    )
    return np.einsum("...ij,...j->...i", rot, r_pf), np.einsum("...ij,...j->...i", rot, v_pf)


def random_leo_states(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """n states with perigee altitude 200-1,500 km, e in [0, 0.3], any orientation."""
    rng = np.random.default_rng(seed)
    e = rng.uniform(0.0, 0.3, n)
    perigee = RE + rng.uniform(200.0, 1500.0, n)
    a = perigee / (1 - e)
    return kepler_to_state(
        a,
        e,
        np.arccos(rng.uniform(-1, 1, n)),
        rng.uniform(0, 2 * np.pi, n),
        rng.uniform(0, 2 * np.pi, n),
        rng.uniform(0, 2 * np.pi, n),
    )


def random_rtn(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [rng.uniform(-50, 50, n), rng.uniform(-2000, 2000, n), rng.uniform(-50, 50, n)]
    )


def literal_rtn(r_c, v_c, r_t):
    """One row, transcribed from the contract's text in ``types.py`` with nothing reworked."""
    h = np.cross(r_c, v_c)
    n = h / np.linalg.norm(h)
    phi_t = math.asin(np.dot(r_t, n) / np.linalg.norm(r_t))
    r_p = r_t - np.dot(r_t, n) * n
    du = math.atan2(np.dot(np.cross(r_p, r_c), n), np.dot(r_p, r_c))
    rc = np.linalg.norm(r_c)
    return np.array([rc - np.linalg.norm(r_t), rc * du, rc * (0.0 - phi_t)])


# ----------------------------------------------------------------- round trips


def test_round_trip_rtn_to_position_to_rtn():
    r_c, v_c = random_leo_states(20_000, seed=1)
    rtn = random_rtn(20_000, seed=2)
    r_t = R.states_from_rtn(r_c, v_c, rtn)
    back = R.rtn_from_states(r_c, v_c, r_t)
    err = np.abs(back - rtn).max()
    assert err < TOL_KM, err


def test_round_trip_position_to_rtn_to_position():
    r_c, v_c = random_leo_states(20_000, seed=3)
    rng = np.random.default_rng(4)
    # truth positions anywhere within the stated separations, built independently of
    # the inverse: along-track by rotating about the true normal, then offsets
    rtn = random_rtn(20_000, seed=5)
    rc = np.linalg.norm(r_c, axis=1)
    u_r = r_c / rc[:, None]
    n = np.cross(r_c, v_c)
    n /= np.linalg.norm(n, axis=1)[:, None]
    u_t = np.cross(n, u_r)
    ang = rtn[:, 1] / rc
    r_t = (
        (rc - rtn[:, 0])[:, None] * (np.cos(ang)[:, None] * u_r + np.sin(ang)[:, None] * u_t)
        + rtn[:, 2:3] * n
        + rng.normal(0, 1.0, (20_000, 3))
    )
    back = R.states_from_rtn(r_c, v_c, R.rtn_from_states(r_c, v_c, r_t))
    err = np.linalg.norm(back - r_t, axis=1).max()
    assert err < TOL_KM, err


def test_matches_literal_transcription_of_the_contract():
    r_c, v_c = random_leo_states(500, seed=6)
    r_t = R.states_from_rtn(r_c, v_c, random_rtn(500, seed=7))
    got = R.rtn_from_states(r_c, v_c, r_t)
    want = np.array([literal_rtn(a, b, c) for a, b, c in zip(r_c, v_c, r_t, strict=True)])
    assert np.abs(got - want).max() < TOL_KM


def test_single_state_shapes():
    r_c, v_c = random_leo_states(1, seed=8)
    rtn = np.array([1.0, 2.0, 3.0])
    r_t = R.states_from_rtn(r_c[0], v_c[0], rtn)
    assert r_t.shape == (3,)
    back = R.rtn_from_states(r_c[0], v_c[0], r_t)
    assert back.shape == (3,)
    np.testing.assert_allclose(back, rtn, rtol=0, atol=TOL_KM)
    assert R.rectilinear_rtn(r_c[0], v_c[0], r_t).shape == (3,)


# ----------------------------------------------------------------- sign conventions


def test_sign_conventions():
    r_c, v_c = random_leo_states(200, seed=9)
    rc = np.linalg.norm(r_c, axis=1)
    n = np.cross(r_c, v_c)
    n /= np.linalg.norm(n, axis=1)[:, None]

    # cheap higher: truth on the same line, 10 km lower -> R = +10, T = N = 0
    lower = r_c * ((rc - 10.0) / rc)[:, None]
    out = R.rtn_from_states(r_c, v_c, lower)
    np.testing.assert_allclose(out[:, 0], 10.0, atol=TOL_KM)
    np.testing.assert_allclose(out[:, 1:], 0.0, atol=TOL_KM)

    # cheap ahead: truth where the cheap object was 2 s earlier -> T > 0
    behind = r_c - 2.0 * v_c
    assert np.all(R.rtn_from_states(r_c, v_c, behind)[:, 1] > 0)
    assert np.all(R.rtn_from_states(r_c, v_c, r_c + 2.0 * v_c)[:, 1] < 0)

    # cheap on the +n side of truth: truth displaced along -n -> N > 0
    assert np.all(R.rtn_from_states(r_c, v_c, r_c - 3.0 * n)[:, 2] > 0)
    assert np.all(R.rtn_from_states(r_c, v_c, r_c + 3.0 * n)[:, 2] < 0)

    # identical positions -> zero residual
    np.testing.assert_allclose(R.rtn_from_states(r_c, v_c, r_c), 0.0, atol=TOL_KM)


# ----------------------------------------------------------------- against rectilinear RTN


@pytest.mark.parametrize("seed", [10, 11])
def test_small_separations_agree_with_rectilinear_to_second_order(seed):
    """|curvilinear - rectilinear| ~ s^2 / (2 r) at the leading order, at every scale.

    Measured: the largest component difference over s^2 / r is 0.500-0.503 for
    s = 0.1-100 km.
    """
    r_c, v_c = random_leo_states(2000, seed=seed)
    rc = np.linalg.norm(r_c, axis=1)
    rng = np.random.default_rng(seed + 100)
    direction = rng.normal(size=(2000, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    ratios = []
    for s in (100.0, 10.0, 1.0, 0.1):
        r_t = r_c - s * direction
        diff = np.abs(R.rtn_from_states(r_c, v_c, r_t) - R.rectilinear_rtn(r_c, v_c, r_t))
        ratio = (diff.max(axis=1) / (s**2 / rc)).max()
        ratios.append(ratio)
        assert 0.45 < ratio < 0.55, (s, ratio)
    # second order, not first: the scaled difference is the same at every s
    assert max(ratios) / min(ratios) < 1.02, ratios


def test_pure_along_track_leading_term():
    """Same radius, arc s behind: rectilinear R = r (1 - cos(s/r)) ~ s^2 / (2 r)."""
    r_c, v_c = random_leo_states(100, seed=12)
    rc = np.linalg.norm(r_c, axis=1)
    for s in (1.0, 10.0):
        r_t = R.states_from_rtn(r_c, v_c, np.column_stack([np.zeros(100), np.full(100, s),
                                                           np.zeros(100)]))
        rect = R.rectilinear_rtn(r_c, v_c, r_t)
        np.testing.assert_allclose(rect[:, 0], rc * (1 - np.cos(s / rc)), rtol=1e-6)
        np.testing.assert_allclose(rect[:, 0] / (s**2 / (2 * rc)), 1.0, atol=1e-3)


def test_large_along_track_leaks_into_rectilinear_radial_only():
    """342 km circular orbit, truth 1,045 km behind on the same orbit.

    A 7-day along-track error of this size at 342 km is the kind measured for
    low objects. Rectilinear RTN reports ~81 km of radial error that isn't there;
    curvilinear reports R = 0 and T = 1,045 km.
    """
    a = RE + 342.0
    r_c, v_c = kepler_to_state(
        np.array([a]), np.array([0.0]), np.array([0.9]), np.array([0.3]), np.array([0.0]),
        np.array([1.0]),
    )
    du = 1045.0 / a
    r_t, _ = kepler_to_state(
        np.array([a]), np.array([0.0]), np.array([0.9]), np.array([0.3]), np.array([0.0]),
        np.array([1.0 - du]),
    )
    curv = R.rtn_from_states(r_c, v_c, r_t)[0]
    rect = R.rectilinear_rtn(r_c, v_c, r_t)[0]
    np.testing.assert_allclose(curv, [0.0, 1045.0, 0.0], atol=TOL_KM)
    assert rect[0] == pytest.approx(a * (1 - math.cos(du)), rel=1e-9)
    assert 80.0 < rect[0] < 82.0  # ~81 km of spurious radial error
    assert rect[1] == pytest.approx(a * math.sin(du), rel=1e-9)  # and T comes out short
    assert abs(rect[2]) < TOL_KM


# ----------------------------------------------------------------- degenerate inputs


def test_degenerate_inputs_raise():
    r_c, v_c = random_leo_states(3, seed=13)
    ok_t = r_c - 1.0 * v_c
    # radial velocity: no orbit normal
    with pytest.raises(R.RTNDomainError, match="angular momentum"):
        R.rtn_from_states(r_c, r_c * 1e-3, ok_t)
    with pytest.raises(R.RTNDomainError, match="angular momentum"):
        R.states_from_rtn(r_c, np.zeros_like(v_c), np.zeros_like(r_c))
    # zero cheap or truth position
    with pytest.raises(R.RTNDomainError, match="cheap position is zero"):
        R.rtn_from_states(np.zeros_like(r_c), v_c, ok_t)
    with pytest.raises(R.RTNDomainError, match="truth position is zero"):
        R.rtn_from_states(r_c, v_c, np.zeros_like(r_c))
    # non-finite
    bad = ok_t.copy()
    bad[1, 0] = np.nan
    with pytest.raises(R.RTNDomainError, match="NaN"):
        R.rtn_from_states(r_c, v_c, bad)
    with pytest.raises(R.RTNDomainError, match="NaN"):
        R.states_from_rtn(r_c, v_c, np.full((3, 3), np.inf))
    # shapes
    with pytest.raises(ValueError, match="row counts"):
        R.rtn_from_states(r_c, v_c, ok_t[:2])
    with pytest.raises(ValueError, match="shape"):
        R.rtn_from_states(r_c[:, :2], v_c, ok_t)


def test_near_radial_velocity_is_refused():
    """A velocity 1e-7 rad from radial: angular momentum not zero, but no usable normal.

    The limit is ``|r x v| / (|r| |v|) >= H_REL_MIN`` = 1e-6, the sine of the angle
    between position and velocity. 1e-7 rad is below it and must be refused, in both
    directions; 1e-5 rad is above it and is accepted.
    """
    r_c, v_c = random_leo_states(50, seed=15)
    rc = np.linalg.norm(r_c, axis=1)
    u_r = r_c / rc[:, None]
    n = np.cross(r_c, v_c)
    n /= np.linalg.norm(n, axis=1)[:, None]
    u_t = np.cross(n, u_r)
    speed = np.linalg.norm(v_c, axis=1)[:, None]

    def tilted(angle):  # velocity `angle` rad away from the radial direction
        return speed * (math.cos(angle) * u_r + math.sin(angle) * u_t)

    # truth 1 km behind along the arc (the in-plane direction is still well defined)
    r_t = rc[:, None] * (np.cos(1.0 / rc)[:, None] * u_r - np.sin(1.0 / rc)[:, None] * u_t)
    v_bad = tilted(1e-7)
    h_rel = np.linalg.norm(np.cross(r_c, v_bad), axis=1) / (rc * speed[:, 0])
    assert np.all(h_rel > 0) and np.all(h_rel < R.H_REL_MIN)  # degenerate, not zero
    with pytest.raises(R.RTNDomainError, match="angular momentum"):
        R.rtn_from_states(r_c, v_bad, r_t)
    with pytest.raises(R.RTNDomainError, match="angular momentum"):
        R.states_from_rtn(r_c, v_bad, np.zeros_like(r_c))
    with pytest.raises(R.RTNDomainError, match="angular momentum"):
        R.rtn_from_states(r_c, tilted(-1e-7), r_t)
    with pytest.raises(R.RTNDomainError, match="angular momentum"):
        R.cheap_frame(r_c, v_bad)
    # one bad row among good ones is enough
    mixed = v_c.copy()
    mixed[7] = v_bad[7]
    with pytest.raises(R.RTNDomainError, match="1 row"):
        R.rtn_from_states(r_c, mixed, r_t)
    # just above the limit the frame exists and the conversion works
    out = R.rtn_from_states(r_c, tilted(1e-5), r_t)
    # looser than TOL_KM: the normal of so thin a triangle is good to ~1e-11 rad
    np.testing.assert_allclose(out, np.tile([0.0, 1.0, 0.0], (50, 1)), atol=1e-6)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_inverse_accepts_every_residual_at_the_forward_limits(sign):
    """A residual with |du| = pi/2 or |phi| = pi/4 exactly, as the forward map stores it.

    The forward map accepts the limit itself and returns T = |r_c| du. The inverse
    recomputes du = T/|r_c|, and that quotient is one float64 step above du for about
    3.5% of radii. With the same limit in both directions the inverse refused those;
    it now accepts one step more, so forward output is always invertible.
    """
    r_c, v_c = random_leo_states(20_000, seed=16)
    rc = np.linalg.norm(r_c, axis=1)
    zero = np.zeros_like(rc)
    for column, limit in ((1, R.MAX_ABS_DU_RAD), (2, R.MAX_ABS_PHI_RAD)):
        rtn = np.column_stack([zero, zero, zero])
        rtn[:, column] = rc * (sign * limit)  # what the forward map computes at the limit
        recomputed = np.abs(rtn[:, column] / rc)
        above = recomputed > limit
        assert 0.01 < above.mean() < 0.10  # the case exists: the quotient overshoots
        assert np.all(recomputed <= math.nextafter(limit, math.inf))  # by one step, no more
        r_t = R.states_from_rtn(r_c, v_c, rtn)  # must not raise
        assert np.all(np.isfinite(r_t))
        np.testing.assert_allclose(np.linalg.norm(r_t, axis=1), rc, rtol=1e-13)
        # still a limit: a part in 1e12 beyond it (0.01 mm of arc) is refused
        beyond = rtn.copy()
        beyond[:, column] *= 1.0 + 1e-12
        with pytest.raises(R.RTNDomainError):
            R.states_from_rtn(r_c, v_c, beyond)


def test_forward_output_near_the_along_track_limit_always_inverts():
    """Truth placed a few float steps either side of du = pi/2: accepted means invertible.

    Each state is converted on its own, so a refused row doesn't hide the others. Rows
    the forward map accepts must go through the inverse and come back to the same
    position; rows it refuses must be the ones beyond the limit.
    """
    r_c, v_c = random_leo_states(400, seed=17)
    rc = np.linalg.norm(r_c, axis=1)
    u_r = r_c / rc[:, None]
    n = np.cross(r_c, v_c)
    n /= np.linalg.norm(n, axis=1)[:, None]
    u_t = np.cross(n, u_r)
    step = math.ulp(R.MAX_ABS_DU_RAD)
    accepted = refused = 0
    for k in (-3, -2, -1, 0, 1, 2, 3):
        for sign in (1.0, -1.0):
            du = sign * (R.MAX_ABS_DU_RAD + k * step)
            r_t = rc[:, None] * (math.cos(du) * u_r - math.sin(du) * u_t)
            for i in range(r_c.shape[0]):
                try:
                    rtn = R.rtn_from_states(r_c[i], v_c[i], r_t[i])
                except R.RTNDomainError:
                    refused += 1
                    assert k >= -2  # only at or beyond the limit, give or take rounding
                    continue
                accepted += 1
                assert abs(rtn[1]) <= R.MAX_ABS_DU_RAD * rc[i]
                back = R.states_from_rtn(r_c[i], v_c[i], rtn)  # must not raise
                assert np.linalg.norm(back - r_t[i]) < TOL_KM
    assert accepted > 1000 and refused > 1000  # both sides of the limit were exercised


def test_angle_limits_raise_in_both_directions():
    r_c, v_c = random_leo_states(4, seed=14)
    rc = np.linalg.norm(r_c, axis=1)
    n = np.cross(r_c, v_c)
    n /= np.linalg.norm(n, axis=1)[:, None]
    # truth on the far side of the Earth (du = pi): ambiguous wrap
    with pytest.raises(R.RTNDomainError, match="along-track"):
        R.rtn_from_states(r_c, v_c, -r_c)
    # just inside the limit is fine, just outside raises (forward and inverse)
    inside = np.column_stack([np.zeros(4), 0.999 * R.MAX_ABS_DU_RAD * rc, np.zeros(4)])
    outside = np.column_stack([np.zeros(4), 1.001 * R.MAX_ABS_DU_RAD * rc, np.zeros(4)])
    r_in = R.states_from_rtn(r_c, v_c, inside)
    np.testing.assert_allclose(R.rtn_from_states(r_c, v_c, r_in), inside, atol=1e-8)
    with pytest.raises(R.RTNDomainError, match="along-track"):
        R.states_from_rtn(r_c, v_c, outside)
    # truth near the cheap orbit's pole
    with pytest.raises(R.RTNDomainError, match="out of the cheap orbit plane"):
        R.rtn_from_states(r_c, v_c, r_c + 2.0 * rc[:, None] * n)
    with pytest.raises(R.RTNDomainError, match="out of the cheap orbit plane"):
        R.states_from_rtn(r_c, v_c, np.column_stack([np.zeros(4), np.zeros(4), rc]))
    # inverse with no positive radius
    with pytest.raises(R.RTNDomainError, match="radius"):
        R.states_from_rtn(r_c, v_c, np.column_stack([rc, np.zeros(4), np.zeros(4)]))
