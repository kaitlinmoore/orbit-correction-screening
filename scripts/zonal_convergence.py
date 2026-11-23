"""Zonal-degree convergence check for the truth propagator.

Question: how many zonal harmonics does truth need? For sample objects in four
altitude bands, propagate 7 days at increasing zonal degree and compare each with
a high-degree reference (degree 90). A degree is converged for an object when its
difference from the reference, and that of every higher degree tested, stays below
the object's threshold over the whole window. The chosen degree is the smallest one
converged for every object.

Integration: the degree runs and the reference use a converged integrator (DOP853,
position tolerance 1 mm, maximum step 30 s), so integration error doesn't blur the
degree-to-degree differences.

Threshold: the integration error of the truth setting itself (1 m tolerance, 300 s
maximum step), measured as its 7-day difference from the converged integrator at
degree ``THRESHOLD_DEGREE``. The original specification was the "1 m vs 1 cm" tolerance
effect; at a 300 s maximum step DOP853 runs at the step cap at both tolerances
(identical evaluation counts), so that difference measures a shift in step phase,
not the integration error. Both are reported; this script uses the integration error.

Every run has drag (NRLMSISE-00), as truth does.

Sample objects (``SAMPLES``): two per band from CelesTrak's active catalogue
(``data/raw/celestrak/gp_active.json``, accessed 2026-11-23), chosen by
``select_samples``: train-role objects (split seed 20261123), eccentricity
< 0.005, 0 < B* <= 0.01 /ER, no maneuvering-constellation names, not a
validation object, mean altitude within 25 km of the band centre, nearest first.
Their mean elements are started at ``EPOCH_UTC`` (solar maximum), where the
space-weather file is fully observed: realistic LEO states, not those objects'
real orbits at that date.

Usage (needs the datagen group):
    uv run python scripts/zonal_convergence.py [--workers 8] [--out PATH.json]
    (default output: data/derived/truth/zonal_convergence.json, gitignored)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from orbit_correction import truth_prop as T

REPO = Path(__file__).resolve().parents[1]
GP_FILE = REPO / "data" / "raw" / "celestrak" / "gp_active.json"
SW_FILE = REPO / "data" / "raw" / "celestrak" / "SW-All.txt"
EPOCH_UTC = "2024-06-15T00:00:00"
HORIZON_S = 7 * 86400.0
STEP_S = 600.0
DEGREES = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 36, 40, 48, 56, 64, 72)
REFERENCE_DEGREE = 90
THRESHOLD_DEGREE = 36
#: the converged integrator: tolerance (m) and maximum step (s)
CONVERGED = (1e-3, 30.0)

#: band (km) -> NORAD IDs, as select_samples() picked them on the 2026-11-23 file.
SAMPLES = {
    300: (46458, 68828),  # Jilin-1 Gaofen 3F (97.2 deg), Cosmos 2617 (96.6 deg)
    500: (63982, 67273),  # SCS-01 A (97.4 deg), 2025-313AB (97.4 deg)
    800: (22824, 24920),  # Stella (98.8 deg, passive sphere), FORTE (70.0 deg)
    1200: (40878, 38354),  # Yaogan-27 (100.1 deg), Yaogan-15 (100.7 deg)
}

_EXCLUDED_PREFIXES = (
    "STARLINK",
    "ONEWEB",
    "KUIPER",
    "IRIDIUM",
    "GLOBALSTAR",
    "ORBCOMM",
    "QIANFAN",
    "GUOWANG",
    "FLOCK",
    "LEMUR",
)


def select_samples(records: list[dict], per_band: int = 2) -> dict[int, tuple[int, ...]]:
    """Re-run the selection rule in the module docstring (for checking ``SAMPLES``)."""
    from orbit_correction import splits as Sp

    scfg = Sp.load_split_config(REPO / "config" / "splits.toml")
    roles = Sp.object_roles([r["NORAD_CAT_ID"] for r in records], scfg)
    mu = 398600.4418
    out = {}
    for band in SAMPLES:
        cands = []
        for r in records:
            n = r["MEAN_MOTION"] * 2 * np.pi / 86400.0
            alt = (mu / n**2) ** (1 / 3) - 6378.137
            ok = (
                roles[r["NORAD_CAT_ID"]] == "train"
                and r["ECCENTRICITY"] < 0.005
                and 0 < r["BSTAR"] <= 0.01
                and not r["OBJECT_NAME"].startswith(_EXCLUDED_PREFIXES)
                and r["NORAD_CAT_ID"] not in scfg.forced_heldout_norad_ids
                and abs(alt - band) < 25
            )
            if ok:
                cands.append((abs(alt - band), r["NORAD_CAT_ID"]))
        out[band] = tuple(nid for _, nid in sorted(cands)[:per_band])
    return out


def load_records() -> dict[int, dict]:
    return {r["NORAD_CAT_ID"]: r for r in json.loads(GP_FILE.read_text())}


def sample_states(records: dict[int, dict]):
    """(band, norad, initial Trajectory, B*) for every sample object."""
    out = []
    for band, ids in SAMPLES.items():
        for nid in ids:
            init, bstar = T.initial_state_from_omm(records[nid], EPOCH_UTC)
            out.append((band, nid, init, bstar))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--out", type=Path, default=REPO / "data" / "derived" / "truth" / "zonal_convergence.json"
    )
    ap.add_argument(
        "--degrees", type=lambda s: tuple(int(x) for x in s.split(",")), default=DEGREES
    )
    ap.add_argument("--reference", type=int, default=REFERENCE_DEGREE)
    args = ap.parse_args()
    degrees, reference = args.degrees, args.reference

    cfg = T.load_truth_config()
    recs = load_records()
    reselected = select_samples(list(recs.values()))
    samples = sample_states(recs)
    t0 = samples[0][2].t_tt_s[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sw_copy = args.out.parent / "sw_orekit_zonal.txt"
    T.write_space_weather_copy(SW_FILE, sw_copy, t0, t0 + HORIZON_S)

    # one batch per configuration; each batch runs every sample object
    tol, step = CONVERGED
    configs = {
        "truth_1m_300s": dataclasses.replace(cfg, zonal_degree=THRESHOLD_DEGREE),
        "truth_1cm_300s": dataclasses.replace(
            cfg, zonal_degree=THRESHOLD_DEGREE, position_tolerance_m=0.01
        ),
        "ref_step_15s": dataclasses.replace(
            cfg, zonal_degree=reference, position_tolerance_m=tol, max_step_s=step / 2
        ),
    }
    for d in sorted({*degrees, THRESHOLD_DEGREE, reference}):
        configs[d] = dataclasses.replace(
            cfg, zonal_degree=d, position_tolerance_m=tol, max_step_s=step
        )
    results: dict = {}
    wall0 = time.monotonic()
    for key, c in configs.items():
        jobs = [
            T.TruthJob(init, bstar, T.output_grid(init.t_tt_s[0], HORIZON_S, STEP_S))
            for _, _, init, bstar in samples
        ]
        res = T.propagate_batch(jobs, c, sw_copy, n_workers=args.workers)
        bad = [r for r in res if r.status != "ok"]
        if bad:
            raise RuntimeError(f"{key}: {[(r.norad_id, r.status, r.message) for r in bad]}")
        results[key] = [r.trajectory.r_gcrf_km for r in res]
        print(f"{key} done ({time.monotonic() - wall0:.0f} s)", flush=True)

    def maxdiff_m(x, y):
        return float(np.max(np.linalg.norm(x - y, axis=1)) * 1000.0)

    ref = results[reference]
    rows = []
    for i, (band, nid, _, bstar) in enumerate(samples):
        thr = maxdiff_m(results["truth_1m_300s"][i], results[THRESHOLD_DEGREE][i])
        diffs = {d: maxdiff_m(results[d][i], ref[i]) for d in degrees}
        converged = [d for d in degrees if all(diffs[e] < thr for e in degrees if e >= d)]
        rows.append(
            {
                "band_km": band,
                "norad_id": nid,
                "bstar": bstar,
                "threshold_truth_integration_error_m": thr,
                "tol_1m_vs_1cm_at_300s_m": maxdiff_m(
                    results["truth_1m_300s"][i], results["truth_1cm_300s"][i]
                ),
                "reference_step_check_m": maxdiff_m(results["ref_step_15s"][i], ref[i]),
                "max_diff_vs_ref_m": diffs,
                "smallest_converged_degree": converged[0] if converged else None,
            }
        )
    needed = [r["smallest_converged_degree"] for r in rows]
    chosen = None if None in needed else max(needed)

    print(
        f"\nReference: degree {reference}, tolerance {tol} m, max step {step} s. Values: max "
        "position difference over 7 days (m)."
    )
    print(
        "Threshold = truth integrator (1 m, 300 s) vs converged, at degree "
        f"{THRESHOLD_DEGREE}. Ref check = reference at max step {step / 2} s vs {step} s."
    )
    hdr = (
        "| Band | NORAD | Threshold | 1 m vs 1 cm (300 s) | Ref check | "
        + " | ".join(f"N={d}" for d in degrees)
        + " | Converged at |"
    )
    print(hdr)
    print("|" + "---|" * (len(degrees) + 6))
    for r in rows:
        cells = " | ".join(f"{r['max_diff_vs_ref_m'][d]:.3g}" for d in degrees)
        print(
            f"| {r['band_km']} | {r['norad_id']} | "
            f"{r['threshold_truth_integration_error_m']:.3g} | "
            f"{r['tol_1m_vs_1cm_at_300s_m']:.3g} | {r['reference_step_check_m']:.3g} | {cells} | "
            f"{r['smallest_converged_degree']} |"
        )
    print(f"\nChosen zonal degree: {chosen}")
    print(f"select_samples() on today's file: {reselected} (SAMPLES: {SAMPLES})")
    args.out.write_text(
        json.dumps(
            {
                "epoch_utc": EPOCH_UTC,
                "horizon_s": HORIZON_S,
                "step_s": STEP_S,
                "reference_degree": reference,
                "degrees": degrees,
                "converged_integrator": CONVERGED,
                "threshold_degree": THRESHOLD_DEGREE,
                "rows": rows,
                "chosen_degree": chosen,
                "provenance": T.orekit_provenance(),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
