"""Sample window starts from the labeled pool for one stage, and report their coverage.

    uv run python scripts/sample_windows.py \\
        --pool-dir data/derived/pool/pool_v1 --stage pilot \\
        --space-weather data/raw/celestrak/SW-All.txt \\
        --out data/derived/windows/pilot

Uses :func:`orbit_correction.generation.sample_windows` with ``config/generation.toml``
(train and dev rows only; starts at element-set epochs at least 7 days apart per
object; one start per object before any gets a second; seeded). The stage's target
count comes from ``[generation.stages]``; ``--n`` overrides it.

Writes into ``--out``:

* ``windows.parquet``: the chosen element sets (catalog and SATCAT columns, ``split``,
  ``window_index``, ``start_round``, ``draw``), in sample order.
* ``windows_dailyap.parquet``: the daily-Ap sensitivity subsample (the same windows,
  ``dailyap_windows`` of them).
* ``windows_manifest.json``: the pool manifest's hash, config hashes, the seed and the
  coverage of the sample by year, altitude band, solar-activity bin, eccentricity bin
  and split, plus starts per object. Labels and counts only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from orbit_correction import generation, residuals
from orbit_correction.spaceweather import parse_spaceweather

REPO = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--pool-dir", type=Path, required=True)
    p.add_argument("--stage", required=True, help="a key of [generation.stages]")
    p.add_argument("--n", type=int, default=None, help="override the stage's window count")
    p.add_argument("--space-weather", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=generation.CONFIG_PATH)
    p.add_argument("--residual-config", type=Path, default=residuals.CONFIG_PATH)
    args = p.parse_args(argv)

    gcfg = generation.load_generation_config(args.config)
    rcfg = residuals.load_residual_config(args.residual_config)
    n = args.n if args.n is not None else gcfg.stage_windows[args.stage]
    files = sorted(args.pool_dir.glob("pool_bucket_*.parquet"))
    if not files:
        raise SystemExit(f"no pool_bucket_*.parquet in {args.pool_dir}")

    # light columns of every eligible start, with where each row lives
    light = []
    for i, f in enumerate(files):
        t = pq.read_table(f, columns=["norad_id", "epoch_tt_s", "split"])
        t = t.append_column("file_no", pa.array(np.full(t.num_rows, i), type=pa.int64()))
        t = t.append_column("row_no", pa.array(np.arange(t.num_rows), type=pa.int64()))
        light.append(t.filter(pc.is_in(t.column("split"), value_set=pa.array(list(gcfg.roles)))))
    light = pa.concat_tables(light)
    chosen = generation.sample_windows(light, gcfg, n)

    # full rows, in sample order
    parts = []
    fno = chosen.column("file_no").to_numpy()
    rno = chosen.column("row_no").to_numpy()
    for i, f in enumerate(files):
        sel = np.flatnonzero(fno == i)
        if sel.size:
            full = pq.read_table(f).take(pa.array(rno[sel]))
            for c in ("window_index", "start_round", "draw"):
                full = full.append_column(c, chosen.column(c).take(pa.array(sel)))
            parts.append(full)
    windows = pa.concat_tables(parts)
    windows = windows.take(pc.sort_indices(windows, [("window_index", "ascending")]))
    assert windows.column("window_index").to_pylist() == list(range(n))

    sw = parse_spaceweather(args.space_weather)
    strata = generation.window_strata(windows, sw, rcfg, gcfg)
    cov = generation.coverage(strata, windows.column("split").to_pylist())
    per_obj = Counter(Counter(windows.column("norad_id").to_pylist()).values())
    daily = generation.dailyap_subsample(windows, gcfg.dailyap_windows, gcfg.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    pq.write_table(windows, args.out / "windows.parquet", compression="zstd")
    pq.write_table(daily, args.out / "windows_dailyap.parquet", compression="zstd")
    manifest = {
        "written_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "code": generation.code_provenance(REPO),
        "stage": args.stage,
        "n_windows": n,
        "n_eligible_starts": light.num_rows,
        "n_eligible_objects": int(len(np.unique(light.column("norad_id").to_numpy()))),
        "n_objects": int(len(np.unique(windows.column("norad_id").to_numpy()))),
        "starts_per_object": {str(k): v for k, v in sorted(per_obj.items())},
        "max_start_round": int(windows.column("start_round").to_numpy().max()),
        "seed": gcfg.seed,
        "generation_status": gcfg.status,
        "pool_manifest_sha256": generation.file_sha256(args.pool_dir / "pool_manifest.json"),
        "configs_sha256": {
            "generation.toml": generation.file_sha256(args.config),
            "residuals.toml": generation.file_sha256(args.residual_config),
        },
        "space_weather_sha256": generation.file_sha256(args.space_weather),
        "coverage": cov,
        "epoch_range_utc": [min(windows.column("epoch_utc").to_pylist()),
                            max(windows.column("epoch_utc").to_pylist())],
        "dailyap_windows": daily.num_rows,
        "dailyap_coverage": generation.coverage(
            generation.window_strata(daily, sw, rcfg, gcfg), daily.column("split").to_pylist()),
        "windows_sha256": generation.file_sha256(args.out / "windows.parquet"),
        "dailyap_sha256": generation.file_sha256(args.out / "windows_dailyap.parquet"),
    }
    (args.out / "windows_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("n_windows", "n_objects", "n_eligible_objects",
                                               "starts_per_object", "coverage")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
