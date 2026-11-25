"""Generate residual rows for a window sample: chunked, resumable, every window logged.

    uv run --group datagen python scripts/generate_residuals.py \\
        --windows data/derived/windows/pilot/windows.parquet \\
        --space-weather data/raw/celestrak/SW-All.txt \\
        --out-root data/derived/residuals --run-id pilot_baseline

    # the labeled daily-Ap sensitivity run on the subsample:
    uv run --group datagen python scripts/generate_residuals.py \\
        --windows data/derived/windows/pilot/windows_dailyap.parquet --daily-ap \\
        --space-weather data/raw/celestrak/SW-All.txt \\
        --out-root data/derived/residuals --run-id pilot_dailyap

What runs where: SGP4 (the window grid, the epoch state and, under the fitted start
rule, the SGP4 samples over the fit span before the epoch), the rows and the files in
this process (no JVM); truth in
``--workers`` spawned processes, one JVM each, kept for the whole run
(:class:`orbit_correction.generation.TruthWorkerPool`). The training process never
runs this: generation is its own process.

Outputs in ``<out-root>/<run-id>/``: one zstd Parquet file of rows per chunk, one
JSON-lines log per chunk (one record per window: IDs, epoch, split, strata, status and
causes, timings, chunk file), a marker per finished chunk, ``run_plan.json`` and
``run_manifest.json`` (code commit, config hashes, orekit-data commit, the
space-weather copy's hash and range, last observed day, seed, counts by status, band
and solar bin). Rerunning the same command skips chunks already written and verified.

``--daily-ap`` runs truth with NRLMSISE-00 switch 9 = +1 (daily Ap) instead of the
configured -1 (3-hour ap array), labels every output ``dailyap`` and leaves
``config/truth.toml`` unchanged. A worker that dies breaks the pool: the chunk is
retried on a fresh pool (up to three attempts) and the crash is counted in the
manifest. ``--truth-config`` takes another truth configuration
(for example a variant start rule); its hash and start rule go into the manifest.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

from orbit_correction import generation, residuals, truth_prop
from orbit_correction.frames import tt_seconds_to_utc_isot
from orbit_correction.spaceweather import parse_spaceweather

REPO = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--windows", type=Path, required=True, help="windows.parquet from sampling")
    p.add_argument("--space-weather", type=Path, required=True, help="CelesTrak SW-All.txt")
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--daily-ap", action="store_true",
                   help="labeled sensitivity run: NRLMSISE-00 switch 9 = +1 (daily Ap)")
    p.add_argument("--label", default=None, help="output label (default baseline / dailyap)")
    p.add_argument("--truth-config", type=Path, default=truth_prop.CONFIG_PATH)
    p.add_argument("--config", type=Path, default=generation.CONFIG_PATH)
    p.add_argument("--residual-config", type=Path, default=residuals.CONFIG_PATH)
    p.add_argument("--workers", type=int, default=None, help="default: config n_workers")
    p.add_argument("--chunk-windows", type=int, default=None, help="default: config value")
    p.add_argument("--limit", type=int, default=None, help="first N windows only (smoke test)")
    p.add_argument("--include-observed", action="store_true", help="fill the obs_* columns")
    args = p.parse_args(argv)

    gcfg = generation.load_generation_config(args.config)
    rcfg = residuals.load_residual_config(args.residual_config)
    tcfg = truth_prop.load_truth_config(args.truth_config)
    if args.daily_ap:
        tcfg = dataclasses.replace(tcfg, msis_ap_switch=1)
    label = args.label or ("dailyap" if args.daily_ap else "baseline")
    workers = args.workers or gcfg.n_workers
    chunk = args.chunk_windows or gcfg.chunk_windows

    windows = pq.read_table(args.windows)
    if args.limit is not None:
        windows = windows.slice(0, args.limit)
    roles = set(windows.column("split").to_pylist())
    if not roles <= set(gcfg.roles):
        raise SystemExit(f"windows carry split labels {sorted(roles)}; only {gcfg.roles} allowed")
    dt_s = residuals.horizon_grid(rcfg)
    sw = parse_spaceweather(args.space_weather)
    epochs = windows.column("epoch_tt_s").to_numpy()
    # under the fitted start, truth also integrates over [epoch - fit span, epoch]
    lead = tcfg.fit_span_s if tcfg.start_rule == truth_prop.FIT_START_RULE else 0.0
    first, last = float(epochs.min() - lead), float(epochs.max() + dt_s[-1])

    run_dir = args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    d0, d1 = str(tt_seconds_to_utc_isot(first))[:10], str(tt_seconds_to_utc_isot(last))[:10]
    copy = truth_prop.write_space_weather_copy(
        args.space_weather, run_dir / f"sw_orekit_{d0}_{d1}.txt", first, last)
    strata = generation.window_strata(windows, sw, rcfg, gcfg)

    wm = args.windows.parent / "windows_manifest.json"
    extra = {
        "label": label,
        "command": sys.argv,
        "inputs": {
            "windows_file": str(args.windows.resolve()),
            "windows_sha256": generation.file_sha256(args.windows),
            "windows_manifest_sha256": generation.file_sha256(wm) if wm.exists() else None,
            "limit": args.limit,
            "space_weather_source_sha256": generation.file_sha256(args.space_weather),
        },
        "sampling_seed": (json.loads(wm.read_text()).get("seed") if wm.exists() else None),
        "configs_sha256": {
            "truth.toml (as given)": generation.file_sha256(args.truth_config),
            "residuals.toml": generation.file_sha256(args.residual_config),
            "generation.toml": generation.file_sha256(args.config),
            "catalog_filters.toml": generation.file_sha256(REPO / "config" /
                                                           "catalog_filters.toml"),
            "splits.toml": generation.file_sha256(REPO / "config" / "splits.toml"),
        },
        "truth": {
            "config_path": str(args.truth_config.resolve()),
            "settings": dataclasses.asdict(tcfg),
            "start_rule": tcfg.start_rule,
            "fit_span_s": tcfg.fit_span_s if lead else None,
            "msis_ap_switch": tcfg.msis_ap_switch,
            "overridden_at_run_time": {"msis_ap_switch": 1} if args.daily_ap else {},
            "orekit_data_commit": tcfg.orekit_data_commit,
            "orekit_stack": truth_prop.orekit_provenance(),
        },
        "space_weather_copy": {
            "path": copy.path, "sha256": copy.sha256,
            "range_utc": [copy.window_first_utc, copy.window_last_utc],
            "last_fully_observed_utc": copy.last_fully_observed_utc,
            "margin_days": copy.margin_days, "n_filled": copy.n_filled,
        },
        "horizons": {"n": int(dt_s.size), "first_s": float(dt_s[0]), "last_s": float(dt_s[-1])},
    }
    settings = generation.RunSettings(run_id=args.run_id, label=label, chunk_windows=chunk,
                                      n_workers=workers, include_observed=args.include_observed)
    start = time.perf_counter()

    def progress(m):
        el = time.perf_counter() - start
        print(f"chunk {m['chunk']:05d}: {m['n_windows']} windows, {m['n_ok']} ok, "
              f"{m['n_rows']} rows, {m['wall_s']:.1f} s ({el:.0f} s so far)", flush=True)

    with generation.TruthWorkerPool(tcfg, copy, workers) as pool:
        man = generation.run_generation(
            windows, run_dir, settings, dt_s=dt_s, residual_cfg=rcfg, space_weather=sw,
            strata=strata, truth_runner=pool.run, truth_cfg=tcfg,
            restart_runner=pool.restart, manifest_extra=extra, progress=progress)
    wall = time.perf_counter() - start
    new = len(man["chunks_written_this_call"])
    # the call's wall time (worker start-up included) goes into the manifest too
    man["last_call"] = {"wall_s": wall, "chunks_written": new, "workers": workers,
                        "windows_per_s": None}
    if new == man["n_chunks"] and man["n_windows"]:
        man["last_call"]["windows_per_s"] = man["n_windows"] / wall
    (run_dir / "run_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({"status": man["status"], "n_rows": man["n_rows"],
                      "parquet_bytes": man["parquet_bytes"], "chunks_new": new,
                      "wall_s_this_call": round(wall, 1)}, indent=1))
    if new == man["n_chunks"] and man["n_windows"]:
        print(f"throughput this call: {man['n_windows'] / wall:.3f} windows/s "
              f"with {workers} workers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
