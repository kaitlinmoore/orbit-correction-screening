"""Cross-check the Orekit truth propagator against brahe on sample objects.

brahe (Rust, MIT, https://github.com/duncaneddy/brahe) is an independent
implementation of the same physics: spherical-harmonic gravity, NRLMSISE-00,
IAU 2006/2000A Earth orientation and a Runge-Kutta integrator. Agreement between
the two says the truth configuration is implemented as described; disagreement
points at a convention.

Matched inputs
--------------
* **Gravity, shared zonal set.** brahe 1.9.0 cannot read the bundle's EIGEN-6S file
  (non-UTF-8 header; it misreads the time-variable terms). So both libraries get one
  static zonal set: EIGEN-6S evaluated by Orekit at the run epoch, degree
  ``truth.toml`` zonal_degree, order 0, written to a fully normalised ICGEM file for
  brahe and passed to Orekit as arrays (``GravityFieldFactory.getNormalizedProvider``).
  Same GM and reference radius in both.
* **Initial state:** the same GCRF state (SGP4 at the shifted epoch).
* **EOP:** the bundle's ``finals2000A.all`` (orekit-data 2d6edc9) in both.
* **Space weather:** Orekit reads the project's copy; brahe reads CelesTrak's
  ``SW-All.txt`` directly (it parses the published file). Same numbers.
* **Drag:** Cd 2.2, A/m from B*, mass 100 kg (cancels).
* **Integrators:** Orekit DOP853 at 1 mm / max step 30 s (converged); brahe RKF78
  at rel 1e-13, abs 1e-8, max step 30 s (converged per project research). The truth
  setting (DOP853, 1 m, 300 s) is reported too.
* **Not matched, stated:** each library's NRLMSISE-00 driver (local solar time
  convention, ap mode, which F10.7 it reads), Sun ephemeris (Orekit DE-440), and
  Earth-rotation implementation details. ``density_probe`` measures the first.

Runs: (1) no drag, isolating gravity, frames and EOP; (2) with drag. Also (0) the
project research's original setup (brahe's EGM2008 J2-J6 vs Orekit's time-variable
EIGEN-6S degree 6, no drag) to size the gravity-model part of the earlier 150 m gap.

Usage (needs the datagen and crosscheck groups):
    uv run python scripts/crosscheck_brahe.py [--quick] [--json out.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zonal_convergence as Z  # noqa: E402  (sample objects and epoch)

from orbit_correction import truth_prop as T  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "derived" / "truth"
HORIZONS = {"1 h": 3600.0, "1 d": 86400.0, "7 d": 7 * 86400.0}
QUICK = (63982, 22824)  # 500 km and 800 km
OREKIT_CONVERGED = (1e-3, 30.0)
BRAHE_TOL = dict(abs=1e-8, rel=1e-13, max_step=30.0)
#: Proposed cross-check tolerance on |brahe - Orekit| at 7 days, metres, both converged.
#: From the first full run (2026-11-23, 8 objects, 300-1200 km, June 2024 start): no drag
#: <= 5.1 mm (tolerance 0.1 m); drag with matched NRLMSISE-00 conventions <= 3.9 m, at
#: 300 km (tolerance 10 m). With the project's ap-array setting brahe differs by up to
#: 11 km at 300 km; that is a convention difference, reported, not a tolerance.
TOLERANCE_7D_M = {"no_drag": 0.1, "drag": 10.0, "drag_project_setting": None}


def bundle_dir() -> Path:
    import orekitdata

    return Path(orekitdata.__path__[0])


def shared_zonal_set(ctx, degree: int, t_tt_s: float):
    """EIGEN-6S normalised zonal C_n0 at the epoch, plus ae and mu (from Orekit)."""
    prov = ctx.gravity_provider(degree)
    h = prov.onDate(ctx.date(t_tt_s))
    c = [1.0, 0.0] + [float(h.getNormalizedCnm(n, 0)) for n in range(2, degree + 1)]
    return np.array(c), float(prov.getAe()), float(prov.getMu())


def write_gfc(path: Path, c: np.ndarray, ae: float, mu: float) -> None:
    """Fully normalised ICGEM file with zonal terms only (order 0), UTF-8."""
    n_max = len(c) - 1
    lines = [
        "begin_of_head",
        "product_type gravity_field",
        "modelname EIGEN6S_zonal_static_crosscheck",
        f"earth_gravity_constant {mu:.10e}",
        f"radius {ae:.10e}",
        f"max_degree {n_max}",
        "errors no",
        "norm fully_normalized",
        "tide_system tide_free",
        "key L M C S",
        "end_of_head",
    ]
    for n in range(n_max + 1):
        for m in range(n + 1):
            cnm = c[n] if m == 0 else 0.0
            lines.append(f"gfc {n:4d} {m:4d} {cnm: .18e} {0.0: .18e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def orekit_provider(c: np.ndarray, ae: float, mu: float):
    from jpype import JArray, JDouble
    from org.orekit.forces.gravity.potential import GravityFieldFactory, TideSystem

    cc = JArray(JDouble, 2)([[float(c[n])] for n in range(len(c))])
    ss = JArray(JDouble, 2)([[0.0] for _ in range(len(c))])
    return GravityFieldFactory.getNormalizedProvider(ae, mu, TideSystem.TIDE_FREE, cc, ss)


def rtn(r, v, dr):
    u = r / np.linalg.norm(r)
    n = np.cross(r, v)
    n /= np.linalg.norm(n)
    t = np.cross(n, u)
    return np.array([dr @ u, dr @ t, dr @ n])


def orekit_states(cfg, init, a2m, sw_copy, provider, tol, step):
    c = dataclasses.replace(cfg, position_tolerance_m=tol, max_step_s=step)
    prop, _, ctx = T.build_propagator(c, init, a2m, sw_copy, gravity_provider=provider)
    gen = prop.getEphemerisGenerator()
    t0 = init.t_tt_s[0]
    prop.propagate(ctx.date(t0), ctx.date(t0 + max(HORIZONS.values())))
    eph = gen.getGeneratedEphemeris()
    out = {}
    for name, dt in HORIZONS.items():
        pv = eph.getPVCoordinates(ctx.date(t0 + dt), ctx.gcrf)
        out[name] = (np.array(pv.getPosition().toArray()), np.array(pv.getVelocity().toArray()))
    return out


def brahe_states(init, a2m, cd, gravity, drag: bool):
    import brahe as bh

    t0 = init.t_tt_s[0]
    epoch = bh.Epoch.from_mjd(51544.5 + t0 / 86400.0, bh.TimeSystem.TT)
    x0 = np.concatenate([init.r_gcrf_km[0], init.v_gcrf_km_s[0]]) * 1000.0
    mass = 100.0
    params = np.array([mass, (a2m or 0.0) * mass, cd, 0.0, 0.0])
    forces = bh.ForceModelConfig(
        gravity=gravity,
        drag=bh.DragConfiguration(
            bh.AtmosphericModel.NRLMSISE00,
            bh.ParameterSource.parameter_index(1),
            bh.ParameterSource.value(float(cd)),
        )
        if drag
        else None,
        mass=bh.ParameterSource.parameter_index(0),
    )
    pc = (
        bh.NumericalPropagationConfig.with_method(bh.IntegrationMethod.RKF78)
        .with_abs_tol(BRAHE_TOL["abs"])
        .with_rel_tol(BRAHE_TOL["rel"])
        .with_max_step(BRAHE_TOL["max_step"])
    )
    prop = bh.NumericalOrbitPropagator(epoch, x0, pc, forces, params)
    out = {}
    for name, dt in HORIZONS.items():  # stop exactly at each horizon (no interpolation)
        prop.propagate_to(epoch + dt)
        s = np.asarray(prop.current_state())
        out[name] = (s[:3], s[3:6])
    return out


def compare(ref, other):
    """other - ref in ref's RTN frame, metres, per horizon."""
    res = {}
    for name in HORIZONS:
        r, v = ref[name]
        dr = other[name][0] - r
        c = rtn(r, v, dr)
        res[name] = {"R": c[0], "T": c[1], "N": c[2], "norm": float(np.linalg.norm(dr))}
    return res


def density_probe(ctx, sw_copy, samples):
    """brahe vs Orekit NRLMSISE-00 at the same ITRF points, for each Orekit convention."""
    import brahe as bh
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.models.earth.atmosphere import NRLMSISE00

    swd = ctx.space_weather(sw_copy)
    variants = {}
    for lst in ("MEAN", "APPARENT"):
        for apsw in (1, -1):
            variants[f"{lst}, sw9={apsw}"] = (
                NRLMSISE00(swd, ctx.sun, ctx.earth, ctx.ut1)
                .withLocalSolarTimeMode(getattr(NRLMSISE00.LocalSolarTimeMode, lst))
                .withSwitch(9, apsw)
            )
    rows = []
    for t, pos_itrf in samples:
        date = ctx.date(t)
        epc = bh.Epoch.from_mjd(51544.5 + t / 86400.0, bh.TimeSystem.TT)
        rho_b = bh.density_nrlmsise00(epc, np.asarray(pos_itrf))
        rows.append(
            {
                k: atm.getDensity(date, Vector3D(*map(float, pos_itrf)), ctx.itrf) / rho_b - 1
                for k, atm in variants.items()
            }
        )
    return {k: float(np.max(np.abs([r[k] for r in rows]))) for k in variants}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="two objects (500 and 800 km)")
    ap.add_argument("--json", type=Path, default=OUT_DIR / "crosscheck_brahe.json")
    args = ap.parse_args()

    import brahe as bh

    cfg = T.load_truth_config()
    recs = Z.load_records()
    samples = [s for s in Z.sample_states(recs) if not args.quick or s[1] in QUICK]
    t0 = samples[0][2].t_tt_s[0]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sw_copy = OUT_DIR / "sw_orekit_crosscheck.txt"
    T.write_space_weather_copy(Z.SW_FILE, sw_copy, t0, t0 + HORIZONS["7 d"])
    ctx = T._context(cfg)

    od = bundle_dir()
    bh.set_global_eop_provider(
        bh.FileEOPProvider.from_standard_file(
            str(od / "Earth-Orientation-Parameters" / "IAU-2000" / "finals2000A.all"), True, "Hold"
        )
    )
    bh.set_global_space_weather_provider(
        bh.FileSpaceWeatherProvider.from_file(str(Z.SW_FILE), "Hold")
    )

    deg = cfg.zonal_degree
    c, ae, mu = shared_zonal_set(ctx, deg, t0)
    gfc = OUT_DIR / f"eigen6s_zonal{deg}_static.gfc"
    write_gfc(gfc, c, ae, mu)
    model = bh.GravityModel.from_file(str(gfc))
    bh.set_global_gravity_model(model)
    got = np.array([model.get_c(n, 0) for n in range(deg + 1)]) if hasattr(model, "get_c") else None
    shared_check = None if got is None else float(np.max(np.abs(got - c)))
    if shared_check is not None and shared_check > 1e-20:
        raise RuntimeError(f"brahe read the shared zonal set back with error {shared_check}")
    brahe_shared = bh.GravityConfiguration.spherical_harmonic(degree=deg, order=0, use_global=True)
    provider = orekit_provider(c, ae, mu)

    results = []
    wall0 = time.monotonic()
    for band, nid, init, bstar in samples:
        a2m = T.area_to_mass_from_bstar(bstar, cfg)
        row = {"band_km": band, "norad_id": nid, "area_to_mass_m2_per_kg": a2m}
        for drag in (False, True):
            key = "drag" if drag else "no_drag"
            c_run = dataclasses.replace(cfg, atmosphere="NRLMSISE-00" if drag else "none")
            ok_conv = orekit_states(c_run, init, a2m, sw_copy, provider, *OREKIT_CONVERGED)
            ok_truth = orekit_states(
                c_run, init, a2m, sw_copy, provider, cfg.position_tolerance_m, cfg.max_step_s
            )
            br = brahe_states(init, a2m if drag else None, cfg.cd, brahe_shared, drag)
            row[key] = {
                "brahe_minus_orekit_converged": compare(ok_conv, br),
                "orekit_truth_setting_minus_converged": compare(ok_conv, ok_truth),
            }
            if drag:
                # Orekit with brahe's NRLMSISE-00 conventions (found by density_probe and
                # pymsis: previous-day observed F10.7, centred observed 81-day mean,
                # daily Ap only = switch 9 = 1, mean local solar time)
                c_b = dataclasses.replace(c_run, msis_local_solar_time="MEAN", msis_ap_switch=1)
                ok_b = orekit_states(c_b, init, a2m, sw_copy, provider, *OREKIT_CONVERGED)
                row[key]["brahe_minus_orekit_brahe_conventions"] = compare(ok_b, br)
                row[key]["orekit_sw9_1_minus_project_setting"] = compare(ok_conv, ok_b)
                # Orekit 13.1.9's constructor defaults (apparent solar time, sw9 = 1), as
                # in the project research run that found the 150 m gap
                c_d = dataclasses.replace(c_run, msis_local_solar_time="APPARENT", msis_ap_switch=1)
                ok_d = orekit_states(c_d, init, a2m, sw_copy, provider, *OREKIT_CONVERGED)
                row[key]["brahe_minus_orekit_library_defaults"] = compare(ok_d, br)
        # (0) the project research's original setup, no drag
        c6 = dataclasses.replace(cfg, zonal_degree=6, atmosphere="none")
        ok6 = orekit_states(c6, init, None, sw_copy, None, *OREKIT_CONVERGED)
        br6 = brahe_states(
            init,
            None,
            cfg.cd,
            bh.GravityConfiguration.earth_zonal(bh.ZonalHarmonicsDegree.J6),
            False,
        )
        row["original_setup_no_drag"] = compare(ok6, br6)
        results.append(row)
        print(f"{band} km {nid} done ({time.monotonic() - wall0:.0f} s)", flush=True)

    # density convention probe along one sample orbit
    band, nid, init, bstar = samples[0]
    from org.orekit.frames import FramesFactory  # noqa: F401  (JVM is up)

    c_run = dataclasses.replace(cfg, atmosphere="none")
    prop, _, _ = T.build_propagator(c_run, init, None, None)
    gen = prop.getEphemerisGenerator()
    prop.propagate(ctx.date(init.t_tt_s[0] + 86400.0))
    eph = gen.getGeneratedEphemeris()
    pts = []
    # offset from 00:00 UTC: at an exact day boundary Orekit's neighbour lookup returns
    # the previous day's row (found here), which is a boundary convention, not a model
    for dt in 420.0 + np.linspace(0, 86400.0 - 840.0, 49):
        date = ctx.date(init.t_tt_s[0] + dt)
        p = eph.getPVCoordinates(date, ctx.itrf).getPosition()
        pts.append((init.t_tt_s[0] + dt, np.array(p.toArray())))
    probe = density_probe(ctx, sw_copy, pts)

    def fmt(x):
        return f"{x:.3g}"

    print("\nbrahe minus Orekit (converged both), metres, RTN of the Orekit state")
    for key in ("no_drag", "drag"):
        print(
            f"\n{key}: | Band | NORAD | "
            + " | ".join(f"{h} R / T / N, abs" for h in HORIZONS)
            + " |"
        )
        for row in results:
            cells = []
            for h in HORIZONS:
                d = row[key]["brahe_minus_orekit_converged"][h]
                cells.append(f"{fmt(d['R'])} / {fmt(d['T'])} / {fmt(d['N'])}, {fmt(d['norm'])}")
            print(f"| {row['band_km']} | {row['norad_id']} | " + " | ".join(cells) + " |")
    print("\nWith drag, brahe minus Orekit using brahe's NRLMSISE conventions (sw9 = 1), |dr|;")
    print("and Orekit sw9 = 1 minus the project setting (sw9 = -1), |dr| / T, at 1 h / 1 d / 7 d:")
    for row in results:
        a = row["drag"]["brahe_minus_orekit_brahe_conventions"]
        b = row["drag"]["orekit_sw9_1_minus_project_setting"]
        print(
            f"  {row['band_km']} {row['norad_id']}: "
            + " / ".join(fmt(a[h]["norm"]) for h in HORIZONS)
            + "  |  "
            + " / ".join(f"{fmt(b[h]['norm'])} ({fmt(b[h]['T'])})" for h in HORIZONS)
        )
    print("\nWith drag, brahe minus Orekit with library defaults (APPARENT, sw9 = 1), |dr| / T:")
    for row in results:
        d = row["drag"]["brahe_minus_orekit_library_defaults"]
        print(
            f"  {row['band_km']} {row['norad_id']}: "
            + " / ".join(f"{fmt(d[h]['norm'])} ({fmt(d[h]['T'])})" for h in HORIZONS)
        )
    print("\nOrekit truth setting (1 m, 300 s) minus converged, |dr| at 1 h / 1 d / 7 d:")
    for row in results:
        for key in ("no_drag", "drag"):
            d = row[key]["orekit_truth_setting_minus_converged"]
            print(
                f"  {row['band_km']} {row['norad_id']} {key}: "
                + " / ".join(fmt(d[h]["norm"]) for h in HORIZONS)
            )
    print("\nOriginal setup (brahe EGM2008 J2-J6 vs Orekit EIGEN-6S deg 6, no drag), |dr| / T:")
    for row in results:
        d = row["original_setup_no_drag"]
        print(
            f"  {row['band_km']} {row['norad_id']}: "
            + " / ".join(f"{fmt(d[h]['norm'])} ({fmt(d[h]['T'])})" for h in HORIZONS)
        )
    print(
        "\nNRLMSISE-00 density, max |Orekit/brahe - 1| along one orbit-day, by Orekit convention:"
    )
    for k, v in probe.items():
        print(f"  {k}: {v:.3g}")

    def worst7(key, field):
        return max(r[key][field]["7 d"]["norm"] for r in results)

    worst = {
        "no_drag": worst7("no_drag", "brahe_minus_orekit_converged"),
        "drag": worst7("drag", "brahe_minus_orekit_brahe_conventions"),
        "drag_project_setting": worst7("drag", "brahe_minus_orekit_converged"),
    }
    tol = dict(TOLERANCE_7D_M)
    within = all(tol.get(k) is None or worst[k] <= tol[k] for k in worst)
    summary = {
        "worst_7d_m": worst,
        "tolerance_7d_m": tol,
        "density_probe": probe,
        "shared_set_readback_max_abs": shared_check,
    }
    print(f"\nWorst 7-day |dr|: {worst}; tolerance {tol}; within: {within}")
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(
            {
                "epoch_utc": Z.EPOCH_UTC,
                "zonal_degree": deg,
                "brahe": bh.__version__,
                "provenance": T.orekit_provenance(),
                "rows": results,
                "summary": summary,
                "within_tolerance": within,
            },
            indent=1,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
