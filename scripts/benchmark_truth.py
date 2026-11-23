"""Throughput of the truth propagator: object-weeks per second at 1, 4 and 8 workers.

Evidence for decision 3 (catalog subset size). Uses the project code end to end:
``truth_prop.propagate_batch`` (spawned workers, one JVM each, the configured
``-XX:ActiveProcessorCount``), the configured force model and zonal degree, and
7-day windows at 60 s output, as data generation will run them.

Protocol, per worker count:
1. Start the pool; each worker starts its JVM and loads Orekit in its initializer.
2. Warm-up batch: one propagation per worker (JIT warm-up), not timed.
3. Timed batch: ``--per-worker`` propagations per worker, cycling over the
   sample objects (scripts/zonal_convergence.py ``SAMPLES``, four altitude bands).
   Throughput = propagations / wall time of the batch (``time.monotonic``).
Repeated ``--trials`` times; reported as median [min-max].

Machine load: ``os.getloadavg()`` and ``/proc/stat`` CPU busy fraction are
recorded at the start and end of each trial. WSL can't see the Windows side's
load directly; a busy Windows host shows up only as slower runs, so rerun on a
quiet machine before using the numbers.

Usage (needs the datagen group):
    uv run python scripts/benchmark_truth.py [--workers 1,4,8] [--per-worker 6] [--trials 3]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import zonal_convergence as Z  # noqa: E402  (sample objects and epoch)

from orbit_correction import truth_prop as T  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "derived" / "truth" / "benchmark_truth.json"
WEEK_S = 7 * 86400.0


def cpu_busy_fraction(interval_s: float = 1.0) -> float:
    """Busy fraction of all CPUs WSL sees, from two /proc/stat samples."""

    def snap():
        with open("/proc/stat") as f:
            vals = [int(x) for x in f.readline().split()[1:]]
        idle = vals[3] + vals[4]
        return sum(vals), idle

    t1, i1 = snap()
    time.sleep(interval_s)
    t2, i2 = snap()
    return 1.0 - (i2 - i1) / max(t2 - t1, 1)


def load_snapshot() -> dict:
    la = os.getloadavg()
    return {"loadavg_1_5_15": [round(x, 2) for x in la], "cpu_busy": round(cpu_busy_fraction(), 3)}


def run_trial(jobs, cfg, sw_copy, n_workers: int, per_worker: int) -> dict:
    sw = str(Path(sw_copy).resolve())
    apc = cfg.jvm_active_processor_count
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp.get_context("spawn"),
        initializer=T._worker_init,
        initargs=(cfg, sw, apc),
    ) as pool:
        warm = [jobs[i % len(jobs)] for i in range(n_workers)]
        list(pool.map(T._worker_run, warm, chunksize=1))
        timed = [jobs[i % len(jobs)] for i in range(n_workers * per_worker)]
        t0 = time.monotonic()
        res = list(pool.map(T._worker_run, timed, chunksize=1))
        wall = time.monotonic() - t0
    bad = [r for r in res if r.status != "ok"]
    if bad:
        raise RuntimeError([(r.norad_id, r.status, r.message) for r in bad])
    per_run = [r.wall_s for r in res]
    return {
        "propagations": len(res),
        "wall_s": wall,
        "object_weeks_per_s": len(res) / wall,
        "per_run_s_median": float(np.median(per_run)),
        "per_run_s_min": float(np.min(per_run)),
        "per_run_s_max": float(np.max(per_run)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", default="1,4,8")
    ap.add_argument("--per-worker", type=int, default=6)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    cfg = T.load_truth_config()
    recs = Z.load_records()
    samples = Z.sample_states(recs)
    jobs = [
        T.TruthJob(init, bstar, T.output_grid(init.t_tt_s[0], WEEK_S, cfg.output_step_s))
        for _, _, init, bstar in samples
    ]
    t0 = samples[0][2].t_tt_s[0]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sw_copy = args.out.parent / "sw_orekit_benchmark.txt"
    T.write_space_weather_copy(Z.SW_FILE, sw_copy, t0, t0 + WEEK_S)

    meta = {
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "zonal_degree": cfg.zonal_degree,
        "integrator": [cfg.integrator, cfg.position_tolerance_m, cfg.max_step_s],
        "jvm_active_processor_count": cfg.jvm_active_processor_count,
        "provenance": T.orekit_provenance(),
        "objects": [nid for _, nid, _, _ in samples],
    }
    try:
        meta["cpu_model"] = next(
            ln.split(":", 1)[1].strip()
            for ln in Path("/proc/cpuinfo").read_text().splitlines()
            if ln.startswith("model name")
        )
    except (OSError, StopIteration):
        pass

    rows = []
    for n in [int(x) for x in args.workers.split(",")]:
        trials = []
        for k in range(args.trials):
            before = load_snapshot()
            res = run_trial(jobs, cfg, sw_copy, n, args.per_worker)
            res["load_start"], res["load_end"] = before, load_snapshot()
            trials.append(res)
            print(
                f"workers {n} trial {k + 1}: {res['object_weeks_per_s']:.3f} object-weeks/s "
                f"(per run {res['per_run_s_median']:.2f} s); load start {before}, "
                f"end {res['load_end']}",
                flush=True,
            )
        tp = [t["object_weeks_per_s"] for t in trials]
        rows.append(
            {
                "workers": n,
                "trials": trials,
                "median": float(np.median(tp)),
                "min": float(min(tp)),
                "max": float(max(tp)),
            }
        )

    base = rows[0]["median"]
    print(
        "\n| Workers | Object-weeks/s, median [min-max] | Speedup | Per-run s |"
    )
    print("|---|---|---|---|")
    for r in rows:
        per_run = np.median([t["per_run_s_median"] for t in r["trials"]])
        print(
            f"| {r['workers']} | {r['median']:.3f} [{r['min']:.3f}-{r['max']:.3f}] | "
            f"{r['median'] / base:.2f} | {per_run:.2f} |"
        )
    args.out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
