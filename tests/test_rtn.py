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
  returns, the inverse accepts;
* the window-level path (``rtn_from_window``): round trips with the along-track angle
  unwrapped beyond pi/2, pi and 2 pi, exact agreement with the single-state path
  inside (-pi, pi], continuity breaks refused, and the other limits kept.

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
        beyond = rtn.copy()
        beyond[:, column] *= 1.0 + 1e-12
        if column == 2:
            # phi is still a limit: a part in 1e12 beyond it (0.01 mm of arc) is refused
            with pytest.raises(R.RTNDomainError):
                R.states_from_rtn(r_c, v_c, beyond)
        else:
            # du is not (decided 2026-11-25): the inverse accepts any along-track angle
            assert np.all(np.isfinite(R.states_from_rtn(r_c, v_c, beyond)))


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
    # just inside the single-state limit is fine, just outside raises in the forward
    # map; the inverse accepts both (it accepts any du)
    inside = np.column_stack([np.zeros(4), 0.999 * R.MAX_ABS_DU_RAD * rc, np.zeros(4)])
    outside = np.column_stack([np.zeros(4), 1.001 * R.MAX_ABS_DU_RAD * rc, np.zeros(4)])
    r_in = R.states_from_rtn(r_c, v_c, inside)
    np.testing.assert_allclose(R.rtn_from_states(r_c, v_c, r_in), inside, atol=1e-8)
    r_out = R.states_from_rtn(r_c, v_c, outside)
    with pytest.raises(R.RTNDomainError, match="along-track"):
        R.rtn_from_states(r_c, v_c, r_out)
    # truth near the cheap orbit's pole
    with pytest.raises(R.RTNDomainError, match="out of the cheap orbit plane"):
        R.rtn_from_states(r_c, v_c, r_c + 2.0 * rc[:, None] * n)
    with pytest.raises(R.RTNDomainError, match="out of the cheap orbit plane"):
        R.states_from_rtn(r_c, v_c, np.column_stack([np.zeros(4), np.zeros(4), rc]))
    # inverse with no positive radius
    with pytest.raises(R.RTNDomainError, match="radius"):
        R.states_from_rtn(r_c, v_c, np.column_stack([rc, np.zeros(4), np.zeros(4)]))


# ------------------------------------------------------------------ one window, unwrapped


def _window(n: int, du_end: float, seed: int, wobble: float = 0.0):
    """Cheap states and the RTN residuals of a constructed window.

    ``du`` ramps from about 0 to ``du_end`` over ``n`` horizons (plus an optional
    sinusoidal ``wobble``, rad, so it is not monotone); R and N are a few km. The cheap
    states are independent random LEO states: the conversion is geometric per row,
    and only the order of the rows matters for unwrapping.
    """
    r_c, v_c = random_leo_states(n, seed=seed)
    rc = np.linalg.norm(r_c, axis=1)
    k = np.arange(n)
    du = du_end * (k + 1) / n + wobble * np.sin(k / 7.0)
    rng = np.random.default_rng(seed)
    rtn = np.column_stack([rng.uniform(-5, 5, n), rc * du, rng.uniform(-5, 5, n)])
    return r_c, v_c, rtn


@pytest.mark.parametrize("du_end", [1.0, 2.0, -2.0, 4.0, -4.0, 7.0, 13.0, -13.0])
def test_window_round_trip_beyond_a_quarter_half_and_whole_turn(du_end):
    """|du| beyond pi/2 (2 rad), beyond pi (4 rad) and beyond 2 pi (7, 13 rad)."""
    r_c, v_c, rtn = _window(200, du_end, seed=31, wobble=0.05)
    r_t = R.states_from_rtn(r_c, v_c, rtn)
    got = R.rtn_from_window(r_c, v_c, r_t)
    # T is |r_c| du with |du| up to 13 rad: the absolute tolerance scales with it
    np.testing.assert_allclose(got[:, [0, 2]], rtn[:, [0, 2]], atol=TOL_KM)
    np.testing.assert_allclose(got[:, 1], rtn[:, 1], rtol=1e-12, atol=1e-8)
    back = R.states_from_rtn(r_c, v_c, got)
    assert np.max(np.linalg.norm(back - r_t, axis=1)) < 1e-8
    if abs(du_end) > math.pi / 2:
        assert np.max(np.abs(got[:, 1] / np.linalg.norm(r_c, axis=1))) > math.pi / 2
        with pytest.raises(R.RTNDomainError, match="along-track"):
            R.rtn_from_states(r_c, v_c, r_t)  # the single-state path still refuses


def test_window_inside_the_single_state_limit_matches_it_bit_for_bit():
    """The per-row path is unchanged, and the window path agrees with it exactly there."""
    r_c, v_c, rtn = _window(199, 1.4, seed=32, wobble=0.1)
    r_t = R.states_from_rtn(r_c, v_c, rtn)
    per_row = R.rtn_from_states(r_c, v_c, r_t)
    assert np.array_equal(R.rtn_from_window(r_c, v_c, r_t), per_row)
    # between pi/2 and pi the window keeps the wrapped value too, bit for bit
    r_c, v_c, rtn = _window(199, 3.0, seed=33)
    r_t = R.states_from_rtn(r_c, v_c, rtn)
    got = R.rtn_from_window(r_c, v_c, r_t)
    rc, u_r, u_t, n = R.cheap_frame(r_c, v_c)
    du_w = np.arctan2(-np.einsum("ij,ij->i", r_t, u_t), np.einsum("ij,ij->i", r_t, u_r))
    assert np.array_equal(got[:, 1], rc * du_w)


def test_window_continuity_break_is_refused():
    r_c, v_c, rtn = _window(50, 0.5, seed=34)
    rc = np.linalg.norm(r_c, axis=1)
    jumped = rtn.copy()
    jumped[30:, 1] += 2.0 * rc[30:]  # du jumps by 2 rad between horizons 29 and 30
    r_t = R.states_from_rtn(r_c, v_c, jumped)
    with pytest.raises(R.RTNDomainError, match="continuity break"):
        R.rtn_from_window(r_c, v_c, r_t)
    # a step just under the limit is followed
    ok = rtn.copy()
    ok[30:, 1] += 0.99 * R.MAX_DU_STEP_RAD * rc[30:]
    got = R.rtn_from_window(r_c, v_c, R.states_from_rtn(r_c, v_c, ok))
    np.testing.assert_allclose(got[:, 1], ok[:, 1], rtol=1e-12, atol=1e-8)
    # a jump by a whole turn is invisible at the horizons (it is a step of 0)
    turned = rtn.copy()
    turned[30:, 1] += 2.0 * math.pi * rc[30:]
    got = R.rtn_from_window(r_c, v_c, R.states_from_rtn(r_c, v_c, turned))
    np.testing.assert_allclose(got[:, 1], rtn[:, 1], rtol=1e-12, atol=1e-8)


def test_window_must_start_near_zero_and_keeps_the_other_limits():
    r_c, v_c, rtn = _window(20, 1.0, seed=35)
    rc = np.linalg.norm(r_c, axis=1)
    late_start = rtn.copy()
    late_start[:, 1] += 1.6 * rc  # first horizon at du ~ 1.65 rad, beyond pi/2
    with pytest.raises(R.RTNDomainError, match="along-track"):
        R.rtn_from_window(r_c, v_c, R.states_from_rtn(r_c, v_c, late_start))
    _, u_r, _, n = R.cheap_frame(r_c, v_c)
    off_plane = R.states_from_rtn(r_c, v_c, rtn)
    off_plane[5] = r_c[5] + 2.0 * rc[5] * n[5]
    with pytest.raises(R.RTNDomainError, match="out of the cheap orbit plane"):
        R.rtn_from_window(r_c, v_c, off_plane)
    with pytest.raises(ValueError, match="one window"):
        R.rtn_from_window(r_c[0], v_c[0], r_c[0])
    radial = v_c.copy()
    radial[3] = r_c[3] / rc[3] * 7.5
    with pytest.raises(R.RTNDomainError, match="degenerate"):
        R.rtn_from_window(r_c, radial, R.states_from_rtn(r_c, v_c, rtn))
