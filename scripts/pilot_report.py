"""Summarise a generation run (and its daily-Ap sensitivity run) for review.

    uv run python scripts/pilot_report.py \\
        --run data/derived/residuals/pilot_v2 \\
        --dailyap-run data/derived/residuals/pilot_v2_dailyap \\
        --compare-run data/derived/residuals/pilot_baseline \\
        --out data/derived/residuals/pilot_v2/report

Reads **train and dev rows only**, and refuses a run whose rows or log carry any other
split label (no statistic of held-out or test-role data is computed anywhere here).

Writes ``report.json`` and ``report.md``:

* windows by status x altitude band x solar-activity bin (``rtn_domain`` included) and
  by split;
* SGP4-minus-truth curvilinear R, T, N: median and 95th percentile of |.| at the grid
  horizons nearest 1 h, 1 d, 3 d and 7 d, by altitude band (with each band's
  ``rtn_domain`` count beside it), by solar-activity bin and by eccentricity bin;
* daily-Ap minus 3-hour-ap truth at the same horizons (the difference of the two runs'
  targets on the same windows: target_dailyap - target_baseline = truth_3h - truth_daily,
  to first order);
* the fitted start's RMS misfit by altitude band (quantiles, maximum, and the windows
  flagged ``fit_rms_above_threshold``);
* the along-track angle of kept windows by altitude band: largest ``|du|`` per window
  (and how many pass pi/2, the single-state limit) and largest step between horizons;
* with ``--compare-run`` (an earlier run of train/dev windows): the windows that run
  dropped as ``rtn_domain``, their status here, and each band's 7-day statistics with
  and without them;
* throughput (windows per second at the worker count used, from the manifest), bytes
  per window on disk, and projections for the larger stages as ratios and totals of
  disk and of truth work (worker-seconds and integrator evaluations, forward run plus the
  fitted start's passes), not durations.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from orbit_correction import generation

ALLOWED = {"train", "dev"}
HORIZONS = {"1 h": 3600.0, "1 d": 86400.0, "3 d": 259200.0, "7 d": 604800.0}
COMPONENTS = ("target_r_km", "target_t_km", "target_n_km")


def load_run(run_dir: Path):
    log = generation.read_logs(run_dir)
    labels = {r["split"] for r in log}
    if not labels <= ALLOWED:
        raise SystemExit(f"{run_dir}: log has split labels {sorted(labels)}; refusing")
    files = sorted(run_dir.glob("chunk_*.parquet"))
    rows = pa.concat_tables([pq.read_table(f) for f in files]) if files else None
    if rows is not None:
        got = set(rows.column("split").unique().to_pylist())
        if not got <= ALLOWED:
            raise SystemExit(f"{run_dir}: rows have split labels {sorted(got)}; refusing")
    man = json.loads((run_dir / "run_manifest.json").read_text())
    return log, rows, man


def nearest_horizons(dt: np.ndarray) -> dict[str, float]:
    grid = np.unique(dt)
    return {k: float(grid[np.argmin(np.abs(grid - v))]) for k, v in HORIZONS.items()}


def stats(values: np.ndarray) -> dict[str, float | int]:
    a = np.abs(values)
    return {"n": int(a.size), "median": float(np.median(a)) if a.size else None,
            "p95": float(np.percentile(a, 95)) if a.size else None}


def error_tables(rows, log, hz):
    """{grouping: {group: {horizon: {component: stats}}}} over ok windows' rows."""
    ecc = {(r["norad_id"], r["epoch_tt_s"]): r["eccentricity_bin"] for r in log}
    nid = rows.column("norad_id").to_numpy()
    t0 = rows.column("start_epoch_tt_s").to_numpy()
    dt = rows.column("dt_s").to_numpy()
    groups = {
        "altitude_band": np.array(rows.column("altitude_band").to_pylist()),
        "solar_activity_bin": np.array(rows.column("solar_activity_bin").to_pylist()),
        "eccentricity_bin": np.array([ecc[(int(a), float(b))] for a, b in zip(nid, t0,
                                                                              strict=True)]),
        "split": np.array(rows.column("split").to_pylist()),
    }
    comp = {c: rows.column(c).to_numpy() for c in COMPONENTS}
    out: dict = {}
    for gname, g in groups.items():
        out[gname] = {}
        for val in sorted(set(g.tolist())):
            out[gname][val] = {}
            for hname, h in hz.items():
                sel = (g == val) & (dt == h)
                out[gname][val][hname] = {c[7]: stats(comp[c][sel]) for c in COMPONENTS}
    out["all"] = {"all": {hname: {c[7]: stats(comp[c][dt == h]) for c in COMPONENTS}
                          for hname, h in hz.items()}}
    return out


def dailyap_diff(base, daily, hz):
    """Target difference (daily-Ap run minus baseline) on the windows both runs kept."""
    def keyed(t):
        k = list(zip(t.column("norad_id").to_pylist(), t.column("start_epoch_tt_s").to_pylist(),
                     t.column("dt_s").to_pylist(), strict=True))
        return {key: i for i, key in enumerate(k)}

    kb, kd = keyed(base), keyed(daily)
    common = sorted(set(kb) & set(kd))
    ib = np.array([kb[k] for k in common], dtype=np.int64)
    idd = np.array([kd[k] for k in common], dtype=np.int64)
    dt = np.array([k[2] for k in common])
    band = np.array(base.column("altitude_band").to_pylist())[ib] if common else np.array([])
    out = {"n_windows_compared": len({(k[0], k[1]) for k in common}), "all": {}, "by_band": {}}
    diffs = {c: daily.column(c).to_numpy()[idd] - base.column(c).to_numpy()[ib]
             for c in COMPONENTS} if common else {}
    for hname, h in hz.items():
        sel = dt == h
        out["all"][hname] = {c[7]: stats(diffs[c][sel]) for c in COMPONENTS} if common else {}
        for b in sorted(set(band.tolist())):
            s2 = sel & (band == b)
            out["by_band"].setdefault(b, {})[hname] = {c[7]: stats(diffs[c][s2])
                                                       for c in COMPONENTS}
    return out


def quantiles(values) -> dict:
    a = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not a.size:
        return {"n": 0}
    q = np.percentile(a, [50, 90, 99])
    return {"n": int(a.size), "median": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
            "max": float(a.max())}


def misfit_by_band(log):
    """Fitted-start RMS misfit (m) by altitude band, and the flagged windows."""
    out = {}
    for b in sorted({r["altitude_band"] for r in log}):
        rs = [r for r in log if r["altitude_band"] == b]
        out[b] = {**quantiles(r.get("fit_rms_m") for r in rs),
                  "flagged": sum("fit_rms_above_threshold" in (r.get("flags") or []) for r in rs)}
    return out


def du_by_band(log):
    """Largest |du| and largest du step per kept window, by altitude band (rad)."""
    out = {}
    for b in sorted({r["altitude_band"] for r in log}):
        rs = [r for r in log if r["altitude_band"] == b and r["status"] == "ok"]
        mx = [r.get("du_max_abs_rad") for r in rs]
        out[b] = {"max_abs_du": quantiles(mx),
                  "windows_beyond_pi_over_2": sum(1 for v in mx if v is not None
                                                  and v > np.pi / 2),
                  "max_step": quantiles(r.get("du_max_step_rad") for r in rs)}
    return out


def compare_previous(log, rows, prev_log, hz):
    """Windows the earlier run dropped as rtn_domain: their fate here, and 7-day stats."""
    prev = {(r["norad_id"], r["epoch_tt_s"]): r for r in prev_log}
    dropped = {k for k, r in prev.items() if r["status"] == "rtn_domain"}
    here = {(r["norad_id"], r["epoch_tt_s"]): r for r in log}
    common = set(prev) & set(here)
    fate = Counter(here[k]["status"] for k in dropped if k in here)
    out = {"windows_in_both_runs": len(common), "previous_rtn_domain": len(dropped),
           "previous_rtn_domain_in_this_run": sum(1 for k in dropped if k in here),
           "their_status_here": dict(fate),
           "previous_status_of_this_runs_rtn_domain": dict(Counter(
               prev[k]["status"] for k, r in here.items()
               if r["status"] == "rtn_domain" and k in prev)), "by_band_7d": {}}
    if rows is None or "7 d" not in hz:
        return out
    nid = rows.column("norad_id").to_numpy()
    t0 = rows.column("start_epoch_tt_s").to_numpy()
    dt = rows.column("dt_s").to_numpy()
    band = np.array(rows.column("altitude_band").to_pylist())
    was = np.array([(int(a), float(b)) in dropped for a, b in zip(nid, t0, strict=True)])
    at = dt == hz["7 d"]
    for b in sorted(set(band.tolist())):
        sel = at & (band == b)
        out["by_band_7d"][b] = {
            "all_ok": {c[7]: stats(rows.column(c).to_numpy()[sel]) for c in COMPONENTS},
            "without_previous_rtn_domain": {
                c[7]: stats(rows.column(c).to_numpy()[sel & ~was]) for c in COMPONENTS},
            "previous_rtn_domain_only": {
                c[7]: stats(rows.column(c).to_numpy()[sel & was]) for c in COMPONENTS},
        }
    return out


def fmt(x):
    if x is None:
        return "-"
    return f"{x:.3g}" if abs(x) < 1000 else f"{x:,.0f}"


def md_error_table(title, tab, hz, extra_col=None):
    lines = [f"### {title}", "",
             "| group | horizon | n | med abs R | p95 abs R | med abs T | p95 abs T | med abs N "
             "| p95 abs N |" + (f" {extra_col[0]} |" if extra_col else ""),
             "|---|---|---|---|---|---|---|---|---|" + ("---|" if extra_col else "")]
    for g, by_h in tab.items():
        for i, hname in enumerate(hz):
            s = by_h[hname]
            ex = ""
            if extra_col:
                ex = f" {extra_col[1].get(g, 0) if i == 0 else ''} |"
            lines.append(f"| {g if i == 0 else ''} | {hname} | {s['r']['n']} | "
                         f"{fmt(s['r']['median'])} | {fmt(s['r']['p95'])} | "
                         f"{fmt(s['t']['median'])} | {fmt(s['t']['p95'])} | "
                         f"{fmt(s['n']['median'])} | {fmt(s['n']['p95'])} |{ex}")
    return lines + [""]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--dailyap-run", type=Path, default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--compare-run", type=Path, default=None,
                   help="an earlier run (train/dev): how its rtn_domain windows fare here")
    p.add_argument("--stages", default="first_batch=50000,ceiling=300000")
    args = p.parse_args(argv)

    log, rows, man = load_run(args.run)
    hz = nearest_horizons(rows.column("dt_s").to_numpy()) if rows is not None else {}
    counts = Counter((r["status"], r["altitude_band"], r["solar_activity_bin"]) for r in log)
    by_split = Counter((r["split"], r["status"]) for r in log)
    rtn_by_band = Counter(r["altitude_band"] for r in log if r["status"] == "rtn_domain")
    report = {
        "run": str(args.run), "label": man.get("label"), "n_windows": len(log),
        "horizons_used_s": hz,
        "counts_status_band_bin": {"|".join(k): v for k, v in sorted(counts.items())},
        "counts_split_status": {"|".join(k): v for k, v in sorted(by_split.items())},
        "rtn_domain_by_band": dict(rtn_by_band),
        "errors": error_tables(rows, log, hz) if rows is not None else {},
        "fit_rms_m_by_band": misfit_by_band(log),
        "flags": dict(Counter(f for r in log for f in (r.get("flags") or []))),
        "du_rad_by_band": du_by_band(log),
        "rtn_conversion": man["plan"].get("rtn_conversion"),
    }
    if args.compare_run is not None:
        prev_log, _, _ = load_run(args.compare_run)
        report["compare"] = {"run": str(args.compare_run),
                             **compare_previous(log, rows, prev_log, hz)}
    if args.dailyap_run is not None:
        dlog, drows, dman = load_run(args.dailyap_run)
        report["dailyap"] = {
            "label": dman.get("label"),
            "status": dict(Counter(r["status"] for r in dlog)),
            "diff": dailyap_diff(rows, drows, hz),
        }

    # throughput, disk and projections
    n_ok = man["status"].get("ok", 0)
    truth_s = [r["truth_wall_s"] for r in log if r["truth_wall_s"]]
    # integrator evaluations per window: the forward run plus the fitted start's passes
    evals = [(r["truth_n_evaluations"] or 0) + (r.get("fit_n_evaluations") or 0)
             for r in log if r["truth_n_evaluations"] or r.get("fit_n_evaluations")]
    per_window_bytes = man["parquet_bytes"] / max(n_ok, 1)
    per_sampled_bytes = man["parquet_bytes"] / max(len(log), 1)
    proj = {}
    for item in args.stages.split(","):
        name, n = item.split("=")
        n = int(n)
        ratio = n / len(log)
        proj[name] = {
            "windows": n, "ratio_to_this_run": ratio,
            "disk_bytes": per_sampled_bytes * n,
            "truth_worker_seconds": float(np.sum(truth_s)) * ratio,
            "integrator_evaluations": float(np.sum(evals)) * ratio,
        }
    report["throughput"] = {
        "workers": man.get("last_call", {}).get("workers", man.get("n_workers")),
        "windows_per_s": man.get("last_call", {}).get("windows_per_s"),
        "truth_job_s_median": float(np.median(truth_s)) if truth_s else None,
        "truth_job_s_p95": float(np.percentile(truth_s, 95)) if truth_s else None,
        "truth_worker_seconds_total": float(np.sum(truth_s)),
        "integrator_evaluations_per_window_median": float(np.median(evals)) if evals else None,
        "fit_status": dict(Counter(str(r.get("fit_status")) for r in log)),
        "bytes_per_ok_window": per_window_bytes,
        "bytes_per_sampled_window": per_sampled_bytes,
        "parquet_bytes": man["parquet_bytes"], "n_rows": man["n_rows"],
        "projections": proj,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    md = [f"# Run report: {man['plan']['run_id']} ({man.get('label')})", "",
          f"Windows: {len(log)}; status: {dict(Counter(r['status'] for r in log))}. "
          "Train and dev rows only.", "", "## Windows by status, altitude band, solar bin", "",
          "| status | band | solar bin | windows |", "|---|---|---|---|"]
    md += [f"| {s} | {b} | {x} | {v} |" for (s, b, x), v in sorted(counts.items())]
    md += ["", f"Horizons used (s): {hz}", ""]
    if rows is not None:
        e = report["errors"]
        md += md_error_table("SGP4 minus truth by altitude band (km)", e["altitude_band"], hz,
                             ("rtn_domain windows", rtn_by_band))
        md += md_error_table("By solar-activity bin (km)", e["solar_activity_bin"], hz)
        md += md_error_table("By eccentricity bin (km)", e["eccentricity_bin"], hz)
        md += md_error_table("By split (km)", e["split"], hz)
    if "dailyap" in report:
        d = report["dailyap"]["diff"]
        md += [f"## Daily-Ap minus 3-hour-ap truth ({d['n_windows_compared']} windows)", ""]
        md += md_error_table("Target difference, dailyap run minus baseline (km)", d["all"]
                             and {"all": d["all"]}, hz)
        md += md_error_table("By altitude band (km)", d["by_band"], hz)
    md += ["## Fitted start: RMS misfit by altitude band (m)", "",
           "| band | n | median | p90 | p99 | max | flagged |", "|---|---|---|---|---|---|---|"]
    for b, q in report["fit_rms_m_by_band"].items():
        md.append(f"| {b} | {q['n']} | {fmt(q.get('median'))} | {fmt(q.get('p90'))} | "
                  f"{fmt(q.get('p99'))} | {fmt(q.get('max'))} | {q['flagged']} |")
    md += ["", "## Along-track angle of kept windows (rad)", "",
           "| band | median max abs du | p99 | max | windows beyond pi/2 | max du step |",
           "|---|---|---|---|---|---|"]
    for b, q in report["du_rad_by_band"].items():
        m, st = q["max_abs_du"], q["max_step"]
        md.append(f"| {b} | {fmt(m.get('median'))} | {fmt(m.get('p99'))} | {fmt(m.get('max'))} "
                  f"| {q['windows_beyond_pi_over_2']} | {fmt(st.get('max'))} |")
    md.append("")
    if "compare" in report:
        c = report["compare"]
        md += [f"## Compared with {c['run']}", "",
               f"- Windows in both runs: {c['windows_in_both_runs']}. The earlier run's "
               f"rtn_domain windows: {c['previous_rtn_domain']}, of which "
               f"{c['previous_rtn_domain_in_this_run']} are in this run, with status "
               f"{c['their_status_here']}.",
               f"- This run's rtn_domain windows had status "
               f"{c['previous_status_of_this_runs_rtn_domain'] or '-'} there.", "",
               "| band | rows | med abs T, all ok | p95 | med abs T, without | p95 | "
               "med abs T, previously dropped only | p95 |", "|---|---|---|---|---|---|---|---|"]
        for b, q in c["by_band_7d"].items():
            a, w, o = q["all_ok"]["t"], q["without_previous_rtn_domain"]["t"], \
                q["previous_rtn_domain_only"]["t"]
            md.append(f"| {b} | {a['n']} | {fmt(a['median'])} | {fmt(a['p95'])} | "
                      f"{fmt(w['median'])} | {fmt(w['p95'])} | {fmt(o['median'])} | "
                      f"{fmt(o['p95'])} |")
        md.append("")
    t = report["throughput"]
    md += ["## Throughput and projections", "",
           f"- {fmt(t['windows_per_s'])} windows/s with {t['workers']} workers; truth job median "
           f"{fmt(t['truth_job_s_median'])} s, p95 {fmt(t['truth_job_s_p95'])} s.",
           f"- {fmt(t['bytes_per_sampled_window'])} bytes per sampled window on disk "
           f"({fmt(t['bytes_per_ok_window'])} per ok window).", ""]
    for name, pr in proj.items():
        md.append(f"- {name}: {pr['windows']:,} windows = {pr['ratio_to_this_run']:.0f} x this "
                  f"run; disk {pr['disk_bytes'] / 1e9:.2f} GB; truth work "
                  f"{pr['truth_worker_seconds']:,.0f} worker-seconds, "
                  f"{pr['integrator_evaluations']:.3g} integrator evaluations.")
    (args.out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
