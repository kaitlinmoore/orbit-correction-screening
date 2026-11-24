"""Scaling benchmark for the screening engine and the classical filters.

Builds a synthetic LEO catalog (``screening.synthetic_leo_elements``: two-body
plus J2 secular motion, not real objects) on a 60 s TT grid, screens it all
pairs and prints one JSON record with wall time, peak memory and system load.
Run one configuration per process, so the peak memory belongs to that run:

    uv run python scripts/benchmark_screening.py --n 500 --days 7 --substeps 2
    uv run python scripts/benchmark_screening.py --n 2000 --days 7 --substeps 2 --padded
    uv run python scripts/benchmark_screening.py --n 2000 --days 7 --filters

``--padded`` screens at 5 km + (1 km + 6 km/day x tau), up to 48 km at day 7,
which is the order of pad a cheap SGP4 screen would need (literature figures
in the methodology); without it the threshold is a constant 5 km.
``--filters`` also runs the three classical filters (pads 20 / 30 / 30 km,
time pad 10 s) on the same catalog, screens the pairs they keep, and reports
their recall against the unfiltered screen. On this catalog the elements
describe the motion exactly, so that recall is a sanity check, not evidence
about the filters on real data.

Nothing is written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import threading
import time

import numpy as np

from orbit_correction import screening as S

EPOCH_TT_S = 7.88e8  # about 2024-12-20; any epoch works for synthetic orbits


class LoadSampler:
    """Samples /proc/loadavg (1-minute load) every few seconds in a thread."""

    def __init__(self, every_s: float = 5.0) -> None:
        self.every_s = every_s
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(os.getloadavg()[0])
            self._stop.wait(self.every_s)

    def __enter__(self) -> LoadSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, required=True, help="number of objects")
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--step", type=float, default=60.0, help="sample spacing (s)")
    ap.add_argument("--substeps", type=int, default=2)
    ap.add_argument("--padded", action="store_true", help="horizon-dependent padded threshold")
    ap.add_argument("--filters", action="store_true", help="also run the classical filters")
    ap.add_argument("--seed", type=int, default=20261123)
    args = ap.parse_args()

    t = EPOCH_TT_S + np.arange(0.0, args.days * S.DAY_S + 0.5, args.step)
    rec: dict = {
        "n": args.n, "days": args.days, "step_s": args.step, "substeps": args.substeps,
        "padded": args.padded, "cpu": cpu_model(), "cpu_count": os.cpu_count(),
        "load_before": os.getloadavg(),
    }
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    with LoadSampler() as sampler:
        tg = time.perf_counter()
        el = S.synthetic_leo_elements(args.n, EPOCH_TT_S, seed=args.seed)
        trajs = S.trajectories_from_elements(el, t, j2=True)
        rec["generate_s"] = time.perf_counter() - tg
        rec["trajectory_bytes"] = int(sum(tr.r_gcrf_km.nbytes + tr.v_gcrf_km_s.nbytes
                                          + tr.t_tt_s.nbytes for tr in trajs))

        if args.padded:
            def threshold(tau):
                return 5.0 + 1.0 + 6.0 * np.maximum(tau, 0.0) / S.DAY_S
        else:
            threshold = 5.0
        u_screen0 = resource.getrusage(resource.RUSAGE_SELF)
        res = S.screen(trajs, threshold, substeps=args.substeps)
        u_screen1 = resource.getrusage(resource.RUSAGE_SELF)
        rec["screen"] = res.stats
        rec["screen_cpu_s"] = (u_screen1.ru_utime - u_screen0.ru_utime
                               + u_screen1.ru_stime - u_screen0.ru_stime)

        if args.filters:
            dmax = float(np.max(S.as_volume(threshold).bound_km(t - t[0])))
            ids, q, Q = S.apsides_from_elements(el)
            ap_res = S.apogee_perigee_filter(ids, q, Q, threshold_km=dmax, pad_km=20.0)
            op_res = S.orbit_path_filter(el, [t[0], t[t.size // 2], t[-1]], threshold_km=dmax,
                                         pad_km=30.0, prior=ap_res.mask)
            tf_res = S.time_filter(el, (t[0], t[-1]), threshold_km=dmax, pad_km=30.0,
                                   time_pad_s=10.0, prior=op_res.mask)
            kept = S.screen(trajs, threshold, substeps=args.substeps, keep=tf_res.mask)
            rec["filters"] = {}
            for f in (ap_res, op_res, tf_res):
                m = S.filter_metrics(res.events, f)
                rec["filters"][f.name] = {
                    "wall_s": f.wall_s, "n_evaluated": f.n_evaluated,
                    "n_pruned_here": f.n_pruned_here, "pruned_fraction": f.pruned_fraction,
                    "event_recall": m["event_recall"], "pair_recall": m["pair_recall"],
                }
            score = S.screening_metrics(res.events, kept.events, tca_tol_s=1.0)
            rec["filtered_screen"] = {
                "wall_s": kept.stats["wall_s"], "n_events": kept.stats["n_events"],
                "event_recall": score.event_recall, "false_alarm_events": score.false_alarm_events,
            }

    usage1 = resource.getrusage(resource.RUSAGE_SELF)
    rec["peak_rss_mib"] = usage1.ru_maxrss / 1024.0
    rec["total_cpu_s"] = (usage1.ru_utime - usage0.ru_utime + usage1.ru_stime - usage0.ru_stime)
    rec["load_samples_1min"] = {"min": min(sampler.samples), "max": max(sampler.samples),
                                "mean": float(np.mean(sampler.samples)),
                                "n": len(sampler.samples)}
    rec["load_after"] = os.getloadavg()
    print(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__":
    main()
