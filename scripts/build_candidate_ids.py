"""Write the NORAD ID lists for the Space-Track ``gp_history`` download, from SATCAT alone.

    uv run python scripts/build_candidate_ids.py \\
        --satcat data/raw/celestrak/satcat.csv --out-dir data/derived/pool \\
        --start 2015-01-01 --end 2026-11-24

Writes to ``--out-dir``:

* ``candidate_ids.txt``: every candidate (the superset rule in
  :func:`orbit_correction.generation.candidate_ids_from_satcat`), forced
  validation-case objects included.
* ``ids_validation.txt``: the forced validation-case objects, downloaded on their own
  over a longer span (their events go back to 2005).
* ``ids_sample.txt``: a seeded sample of the other candidates, downloaded first to
  measure records and bytes per object.
* ``ids_main.txt``: every other candidate. The three download lists are disjoint, so
  no object's history is fetched twice (Space-Track asks for ``gp_history`` once per
  object).
* ``candidates_manifest.json``: the rule's parameters, the SATCAT file's SHA-256 and
  access date, and counts per reason.

Reads only CelesTrak's SATCAT; touches no Space-Track data and no network.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

from orbit_correction import generation
from orbit_correction.catalog import load_filter_config
from orbit_correction.splits import load_split_config

REPO = Path(__file__).resolve().parents[1]


def _write_ids(path: Path, ids, header: str) -> None:
    path.write_text(f"# {header}\n" + "\n".join(str(int(i)) for i in ids) + "\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--satcat", type=Path, required=True, help="CelesTrak satcat.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--start", required=True, help="span start, YYYY-MM-DD")
    p.add_argument("--end", required=True, help="span end, YYYY-MM-DD (exclusive)")
    p.add_argument("--filters", type=Path, default=REPO / "config" / "catalog_filters.toml")
    p.add_argument("--splits", type=Path, default=REPO / "config" / "splits.toml")
    p.add_argument("--apogee-margin-km", type=float, default=100.0)
    p.add_argument("--decay-margin-days", type=int, default=30)
    p.add_argument("--apply-name-rule", choices=("auto", "on", "off"), default="auto",
                   help="name rule: auto applies it when the filter reads SATCAT names (exact)")
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--sample-seed", type=int, default=20261124)
    args = p.parse_args(argv)

    filters = load_filter_config(args.filters)
    forced = load_split_config(args.splits).forced_heldout_norad_ids
    res = generation.candidate_ids_from_satcat(
        args.satcat, filters, args.start, args.end, forced,
        apogee_margin_km=args.apogee_margin_km,
        decay_margin_days=args.decay_margin_days,
        apply_name_rule={"auto": None, "on": True, "off": False}[args.apply_name_rule],
    )
    forced_set = {int(i) for i in forced}
    others = [int(i) for i in res.ids if int(i) not in forced_set]
    sample = generation.download_sample_ids(others, args.sample_size, args.sample_seed)
    sample_set = set(sample.tolist())
    main_ids = [i for i in others if i not in sample_set]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"span {args.start} .. {args.end} (exclusive)"
    _write_ids(args.out_dir / "candidate_ids.txt", res.ids, f"all candidates, {tag}")
    _write_ids(args.out_dir / "ids_validation.txt", sorted(forced_set), "validation-case objects")
    _write_ids(args.out_dir / "ids_sample.txt", sample, f"download sample, seed {args.sample_seed}")
    _write_ids(args.out_dir / "ids_main.txt", main_ids, f"main download list, {tag}")

    access = args.satcat.with_name("ACCESS_DATE.txt")
    manifest = {
        "written_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "satcat": str(args.satcat.resolve()),
        "satcat_sha256": hashlib.sha256(args.satcat.read_bytes()).hexdigest(),
        "satcat_access": access.read_text().strip() if access.exists() else None,
        "filters_sha256": hashlib.sha256(args.filters.read_bytes()).hexdigest(),
        "span_start_utc": args.start,
        "span_end_utc_exclusive": args.end,
        "apogee_margin_km": args.apogee_margin_km,
        "decay_margin_days": args.decay_margin_days,
        "apply_name_rule": args.apply_name_rule,
        "n_satcat": res.n_satcat,
        "dropped": res.dropped,
        "unknown_kept": res.unknown_kept,
        "forced_added": list(res.forced_added),
        "n_candidates": int(res.ids.size),
        "n_validation": len(forced_set),
        "n_sample": int(sample.size),
        "sample_seed": args.sample_seed,
        "n_main": len(main_ids),
    }
    (args.out_dir / "candidates_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in (
        "n_satcat", "dropped", "unknown_kept", "forced_added", "n_candidates", "n_validation",
        "n_sample", "n_main")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
