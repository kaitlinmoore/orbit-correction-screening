"""Measure two truth settings on the sample objects: the start rule and the step cap.

Part 1, start rule. SGP4 minus truth at 0, 1 h, 1 d and 7 d, with the default
start rule (``preserve_sgp4_mean_motion``: initial state scaled by
``k = (mu_EIGEN-6S / mu_WGS-72) ** (1/3)``) and without it
(``sgp4_state_direct``). The rule removes the drift that comes only from the two
models' different GM (about 1.15 km/day). What remains is the rest of the
SGP4-truth difference: the zonal and drag models, and SGP4's mean-element theory
against an osculating start. At t0 the residual is not zero with the rule: the
scaled state sits ``(k - 1) |r|``, about 2 m, from the SGP4 state.

Part 2, step cap. The truth setting (``config/truth.toml``: DOP853, 1 m tolerance,
the configured maximum step, degree 36) against a converged run (1 mm, 30 s, same
degree and force model), at 1 h, 1 d and 7 d and as the maximum over 7 days.

SGP4 here is ``cheap_prop`` (the ``sgp4`` package with WGS-72, mode 'i', the same
element set as ``truth_prop.initial_state_from_omm``), rotated TEME -> GCRF exactly.
Components are rectilinear RTN of the truth state (T is along-track).

Sample objects and epoch: ``scripts/zonal_convergence.py`` (two per band, mean
elements started at 2024-06-15T00:00 UTC; synthetic states, train-role objects).

Usage (needs the datagen group):
    uv run python scripts/truth_settings_check.py [--workers 8] [--out PATH.json]
    (default output: data/derived/truth/truth_settings_check.json, gitignored)
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
OUT = REPO / "data" / "derived" / "truth" / "truth_settings_check.json"
HORIZON_S = 7 * 86400.0
STEP_S = 600.0
HORIZONS = {"0": 0.0, "1 h": 3600.0, "1 d": 86400.0, "7 d": HORIZON_S}
CONVERGED = (1e-3, 30.0)


def sgp4_gcrf(record: dict, epoch_utc: str, t_tt_s: np.ndarray) -> np.ndarray:
    """SGP4 positions (km, GCRF) at TT times, from the shifted-epoch element set.

    Through ``cheap_prop`` (exact rotation), the same element set and SGP4 path as the
    truth start (``truth_prop.initial_state_from_omm``).
    """
    from orbit_correction.cheap_prop import propagate_sgp4

    res = propagate_sgp4(T._omm_record_elset(record, epoch_utc), t_tt_s, rotation="exact")[0]
    if not res.status.ok:
        raise RuntimeError(f"NORAD {res.norad_id}: {res.status.message}")
    return res.trajectory.r_gcrf_km


def rtn_m(r_ref, v_ref, dr):
    """dr (km) in the RTN frame of (r_ref, v_ref), metres."""
    u = r_ref / np.linalg.norm(r_ref)
    n = np.cross(r_ref, v_ref)
    n /= np.linalg.norm(n)
    t = np.cross(n, u)
    return {"R": float(dr @ u) * 1e3, "T": float(dr @ t) * 1e3, "N": float(dr @ n) * 1e3,
            "norm": float(np.linalg.norm(dr)) * 1e3}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    cfg = T.load_truth_config()
    recs = Z.load_records()
    samples = Z.sample_states(recs)
    t0 = samples[0][2].t_tt_s[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sw_copy = T.write_space_weather_copy(
        Z.SW_FILE, args.out.parent / "sw_orekit_settings_2024-06-15_7d.txt", t0, t0 + HORIZON_S
    )
    configs = {
        "truth": cfg,
        "truth_direct_start": dataclasses.replace(cfg, start_rule="sgp4_state_direct"),
        "converged": dataclasses.replace(
            cfg, position_tolerance_m=CONVERGED[0], max_step_s=CONVERGED[1]
        ),
    }
    grids = [T.output_grid(init.t_tt_s[0], HORIZON_S, STEP_S) for _, _, init, _ in samples]
    runs = {}
    wall0 = time.monotonic()
    for key, c in configs.items():
        jobs = [T.TruthJob(init, b, g) for (_, _, init, b), g in zip(samples, grids, strict=True)]
        res = T.propagate_batch(jobs, c, sw_copy, n_workers=args.workers)
        bad = [r for r in res if r.status != "ok"]
        if bad:
            raise RuntimeError(f"{key}: {[(r.norad_id, r.status, r.message) for r in bad]}")
        runs[key] = res
        print(f"{key} done ({time.monotonic() - wall0:.0f} s)", flush=True)

    rows = []
    for i, (band, nid, _, _) in enumerate(samples):
        g = grids[i]
        idx = {h: int(round(dt / STEP_S)) for h, dt in HORIZONS.items()}
        r_sgp4 = sgp4_gcrf(recs[nid], Z.EPOCH_UTC, g)
        row = {"band_km": band, "norad_id": nid, "start_scale": runs["truth"][i].start_scale}
        for key in ("truth", "truth_direct_start"):
            tr = runs[key][i].trajectory
            row[f"sgp4_minus_{key}"] = {
                h: rtn_m(tr.r_gcrf_km[j], tr.v_gcrf_km_s[j], r_sgp4[j] - tr.r_gcrf_km[j])
                for h, j in idx.items()
            }
        a, b = runs["truth"][i].trajectory, runs["converged"][i].trajectory
        d = np.linalg.norm(a.r_gcrf_km - b.r_gcrf_km, axis=1) * 1e3
        row["truth_minus_converged_m"] = {h: float(d[j]) for h, j in idx.items() if h != "0"}
        row["truth_minus_converged_max_7d_m"] = float(d.max())
        row["evaluations"] = {"truth": runs["truth"][i].n_evaluations,
                              "converged": runs["converged"][i].n_evaluations}
        rows.append(row)

    def f(x):
        return f"{x:.3g}"

    print("\nPart 1. SGP4 minus truth, |dr| (T) in metres, at 0 / 1 h / 1 d / 7 d")
    print("| Band | NORAD | k - 1 | With rescale | Without rescale |")
    print("|---|---|---|---|---|")
    for r in rows:
        cells = []
        for key in ("sgp4_minus_truth", "sgp4_minus_truth_direct_start"):
            cells.append(" / ".join(f"{f(r[key][h]['norm'])} ({f(r[key][h]['T'])})"
                                    for h in HORIZONS))
        print(f"| {r['band_km']} | {r['norad_id']} | {r['start_scale'] - 1:.3e} | "
              + " | ".join(cells) + " |")
    print(f"\nPart 2. Truth setting (1 m, {cfg.max_step_s:g} s) minus converged "
          f"({CONVERGED[0]} m, {CONVERGED[1]:g} s), |dr| m")
    print("| Band | NORAD | 1 h | 1 d | 7 d | max over 7 d | evaluations truth / converged |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        d = r["truth_minus_converged_m"]
        print(f"| {r['band_km']} | {r['norad_id']} | {f(d['1 h'])} | {f(d['1 d'])} | "
              f"{f(d['7 d'])} | {f(r['truth_minus_converged_max_7d_m'])} | "
              f"{r['evaluations']['truth']} / {r['evaluations']['converged']} |")
    args.out.write_text(json.dumps({
        "epoch_utc": Z.EPOCH_UTC,
        "truth_config": dataclasses.asdict(cfg),
        "converged": CONVERGED,
        "provenance": T.orekit_provenance(),
        "rows": rows,
    }, indent=1))


if __name__ == "__main__":
    main()
