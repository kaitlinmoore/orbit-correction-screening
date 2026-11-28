"""Build the training pool and label the first real split.

    uv run python scripts/build_pool.py \\
        --extract-dir data/derived/pool/bulk_extract \\
        --validation-json 'data/raw/spacetrack/gp_history/*.json.gz' \\
        --satcat data/raw/celestrak/satcat.csv \\
        --space-weather data/raw/celestrak/SW-All.txt \\
        --start 2015-01-01 --end 2026-01-01 \\
        --out-dir data/derived/pool/pool_v3

Per NORAD-ID bucket (``bulk_extract/bucket_NN``, from ``extract_bulk_tle.py``):

1. Element sets from the bulk TLE extracts, plus the validation objects' ``gp_history``
   downloads (Space-Track API, OMM JSON). The validation objects come from the API
   only; their rows in the bulk files are dropped (counted), so one object's history
   never mixes two sources.
2. :func:`orbit_correction.generation.build_pool`: the span, exact duplicates
   removed, reissues (one object's sets at most 60 s apart) collapsed to the highest
   element-set number, SATCAT attached, the approved catalog filters applied (counts per rule,
   in rows and objects; for the element-history rules also trips and the sets the spike
   rule left out, with and without the spike rule).
3. :func:`orbit_correction.generation.label_pool`: every kept element set labeled as a
   window start with ``splits.assign_split_labels``, using the space-weather file's
   last fully observed day at labeling time; windows ending after ``--end`` (the end
   of the downloaded history) are labeled ``excluded``, counted by original label.

Writes ``pool_bucket_NN.parquet`` (catalog columns, SATCAT columns, ``split``) and
``pool_manifest.json`` (inputs and their hashes, config hashes, the last observed day,
the split seed, and every count). Counts only: no residual exists at this point, and
nothing about held-out objects goes beyond row and object counts per label.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from orbit_correction import catalog, generation
from orbit_correction.spaceweather import parse_spaceweather
from orbit_correction.splits import load_split_config

REPO = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--extract-dir", type=Path, required=True)
    p.add_argument("--validation-json", default=None,
                   help="glob of the validation objects' gp_history files (.json or .json.gz)")
    p.add_argument("--satcat", type=Path, required=True)
    p.add_argument("--space-weather", type=Path, required=True, help="CelesTrak SW-All.txt")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True, help="exclusive")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--filters", type=Path, default=REPO / "config" / "catalog_filters.toml")
    p.add_argument("--splits", type=Path, default=REPO / "config" / "splits.toml")
    args = p.parse_args(argv)

    filters = catalog.load_filter_config(args.filters)
    split_cfg = load_split_config(args.splits)
    satcat = catalog.load_satcat(args.satcat)
    sw = parse_spaceweather(args.space_weather)
    last_obs = sw.last_fully_observed_date()
    forced = pa.array(list(split_cfg.forced_heldout_norad_ids), type=pa.int64())

    val_files = sorted(glob.glob(args.validation_json)) if args.validation_json else []
    validation = catalog.load_spacetrack_gp_history(val_files) if val_files else None
    bucket_dirs = sorted(d for d in args.extract_dir.glob("bucket_*") if d.is_dir())
    if not bucket_dirs:
        raise SystemExit(f"no bucket_* directories in {args.extract_dir}")
    n_buckets = len(bucket_dirs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    totals: dict[str, Counter] = {k: Counter() for k in (
        "dropped_at_rule", "rejected_by_rule", "objects_dropped_at_rule", "dedupe", "split_rows",
        "split_objects", "rows", "history_end_excluded", "reissues",
        "reissues_removed_by_source")}
    history_rules: dict[str, Counter] = {r: Counter() for r in catalog.HISTORY_RULES}
    max_span = 0.0
    outputs = []
    for d in bucket_dirs:
        b = int(d.name.split("_")[1])
        files = sorted(d.glob("*.parquet"))
        bulk = pa.concat_tables([pq.read_table(f) for f in files]) if files else None
        parts = []
        if bulk is not None:
            is_val = pc.is_in(bulk.column("norad_id"), value_set=forced)
            totals["rows"]["bulk_rows_of_validation_objects_dropped"] += int(
                pc.sum(is_val).as_py() or 0)
            parts.append(bulk.filter(pc.invert(is_val)))
        if validation is not None:
            ids = validation.column("norad_id").to_numpy()
            parts.append(validation.filter(pa.array(ids % n_buckets == b)))
        elsets = pa.concat_tables(parts, promote_options="none")
        res = generation.build_pool(elsets, satcat, filters, args.start, args.end)
        labeled, summary, moved = generation.label_pool(
            res.table, split_cfg, last_obs, history_end_utc=f"{args.end}T00:00:00")
        totals["history_end_excluded"].update(moved)
        out = args.out_dir / f"pool_bucket_{b:02d}.parquet"
        pq.write_table(labeled, out, compression="zstd")
        outputs.append(out.name)
        totals["rows"].update(loaded=res.n_loaded, in_span=res.n_in_span,
                              kept=labeled.num_rows, objects_in=res.n_objects_in,
                              objects_kept=res.n_objects_kept)
        totals["dedupe"].update(exact_duplicates=res.dedupe.exact_duplicates,
                                same_epoch_different=res.dedupe.same_epoch_different)
        rs = res.reissues
        totals["reissues"].update(groups=rs.groups, rows_removed=rs.rows_removed,
                                  same_epoch_groups=rs.same_epoch_groups,
                                  near_simultaneous_groups=rs.near_simultaneous_groups,
                                  tie_groups=rs.tie_groups)
        totals["reissues_removed_by_source"].update(rs.removed_by_source)
        max_span = max(max_span, rs.max_group_span_s)
        for k in ("dropped_at_rule", "rejected_by_rule", "objects_dropped_at_rule"):
            totals[k].update(getattr(res, k))
        for r, c in res.history_rules.items():
            history_rules[r].update(c)  # buckets are disjoint by object
        for lab, c in summary.items():
            totals["split_rows"][lab] += c["rows"]
            totals["split_objects"][lab] += c["objects"]  # buckets are disjoint by object
        print(f"bucket {b:02d}: {res.n_in_span:,} in span -> {labeled.num_rows:,} kept, "
              f"{res.n_objects_kept:,} objects", flush=True)

    access = args.satcat.with_name("ACCESS_DATE.txt")
    manifest = {
        "written_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "code": generation.code_provenance(REPO),
        "span": [args.start, args.end],
        "inputs": {
            "extract_manifest_sha256": generation.file_sha256(
                args.extract_dir / "extract_manifest.jsonl"),
            "validation_files": {Path(f).name: generation.file_sha256(f) for f in val_files},
            "satcat_sha256": generation.file_sha256(args.satcat),
            "satcat_access": access.read_text().strip() if access.exists() else None,
            "space_weather_sha256": generation.file_sha256(args.space_weather),
        },
        "configs_sha256": {
            "catalog_filters.toml": generation.file_sha256(args.filters),
            "splits.toml": generation.file_sha256(args.splits),
        },
        "filters_status": filters.status,
        "split": {
            "seed": split_cfg.seed,
            "status": split_cfg.status,
            "last_fully_observed_date_utc": last_obs,
            "rows_by_label": dict(sorted(totals["split_rows"].items())),
            "objects_by_label": dict(sorted(totals["split_objects"].items())),
            "history_end_utc": f"{args.end}T00:00:00",
            "history_end_excluded_by_original_label": dict(totals["history_end_excluded"]),
        },
        "counts": {k: dict(v) for k, v in totals.items()
                   if k not in ("split_rows", "split_objects", "history_end_excluded")},
        "history_rules": {
            "exclude_history_spikes": filters.exclude_history_spikes,
            "spike_leg_thresholds": catalog.SPIKE_LEG_THRESHOLDS,
            "counts_on_objects_alive_at_rule": {r: dict(c) for r, c in history_rules.items()},
        },
        "reissue_rule": {"max_gap_s": generation.REISSUE_MAX_GAP_S,
                         "keep": "highest element_set_no; tie: last read",
                         "max_group_span_s": max_span},
        "buckets": n_buckets,
        "outputs": outputs,
    }
    (args.out_dir / "pool_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"split": manifest["split"], "counts": manifest["counts"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
