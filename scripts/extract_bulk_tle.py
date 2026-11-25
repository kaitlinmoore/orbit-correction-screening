"""Extract candidate objects' element sets from Space-Track's yearly bulk TLE files.

    uv run python scripts/extract_bulk_tle.py \\
        --bulk-dir data/raw/spacetrack/bulk_tle \\
        --ids data/derived/pool/candidate_ids.txt \\
        --start 2015-01-01 --end 2026-11-24 \\
        --out-dir data/derived/pool/bulk_extract

Reads every ``*.zip`` (each text member streamed, nothing unpacked to disk; macOS
``__MACOSX/`` and ``._*`` members skipped), ``*.txt`` and ``*.txt.gz`` file in
``--bulk-dir``. Files may be grouped by publication date, not epoch (Space-Track's 2015
file starts with 2014-12-31 epochs), so element sets are kept by their **epoch** and
never by the file they came in. A trailing backslash on a line (the 2015 file ends
every line 1 with one) is removed and counted; 68-character lines without a checksum
column are counted and left out. Two-line and three-line sets are both read
(``orbit_correction.generation.iter_tle_records``). Keeps element sets of the IDs in
``--ids`` with epochs in ``[--start, --end)`` and writes them in the catalog schema
(``source = "spacetrack_bulk_tle"``), split by NORAD ID into ``--buckets`` groups
(``bucket_NN/<file stem>.parquet``), so later steps can work one group of objects at a
time: every catalog filter rule is per object.

``extract_manifest.jsonl`` gets one entry per input file: size, SHA-256, members and
the read counts (lines, TLEs, bad pairs with examples, stray lines, other objects,
epochs outside the span, kept). An input whose entry matches its current SHA-256 and
whose outputs exist is skipped, so a rerun only does new files.

Space-Track data stays in ``data/`` (gitignored) and is never committed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import json
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from orbit_correction import catalog, generation

SOURCE = "spacetrack_bulk_tle"
BATCH = 250_000


def read_ids(path: Path) -> set[int]:
    out = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(int(line))
    return out


def text_streams(path: Path) -> Iterator[tuple[str, Iterator[str]]]:
    """(member name, line iterator) for each text stream in one input file."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                base = info.filename.rsplit("/", 1)[-1]
                if info.is_dir() or info.filename.startswith("__MACOSX/") or base.startswith("._"):
                    continue  # macOS resource forks, not data
                with z.open(info) as raw:
                    if info.filename.endswith(".gz"):
                        raw = gzip.GzipFile(fileobj=raw)
                    yield info.filename, io.TextIOWrapper(raw, encoding="ascii",
                                                          errors="replace")
    elif path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="ascii", errors="replace") as f:
            yield path.name, f
    else:
        with open(path, encoding="ascii", errors="replace") as f:
            yield path.name, f


def extract_one(path: Path, want: set[int], start: str, end: str, out_dir: Path,
                buckets: int) -> dict:
    stats = generation.TleReadStats()
    members, out_of_span, kept = [], 0, 0
    parts: dict[int, list[pa.Table]] = {}
    batch: list[dict] = []

    def flush():
        nonlocal batch
        if not batch:
            return
        t = catalog.records_to_table(batch, SOURCE)
        ids = t.column("norad_id").to_numpy()
        for b in range(buckets):
            sel = ids % buckets == b
            if sel.any():
                parts.setdefault(b, []).append(t.filter(pa.array(sel)))
        batch = []

    for name, lines in text_streams(path):
        members.append(name)
        for rec in generation.iter_tle_records(lines, stats, want_ids=want):
            if not (start <= rec["EPOCH"] < end):
                out_of_span += 1
                continue
            batch.append(rec)
            kept += 1
            if len(batch) >= BATCH:
                flush()
    flush()
    stem = path.name.split(".")[0]
    written = []
    for b, tables in sorted(parts.items()):
        d = out_dir / f"bucket_{b:02d}"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{stem}.parquet"
        tmp = f.with_name(f.name + ".tmp")
        pq.write_table(pa.concat_tables(tables), tmp, compression="zstd")
        tmp.replace(f)
        written.append(str(f.relative_to(out_dir)))
    return {
        "lines": stats.lines, "tles": stats.tles, "bad": stats.bad,
        "no_checksum": stats.no_checksum, "trailing_backslash": stats.trailing_backslash,
        "stray_lines": stats.stray_lines, "other_objects": stats.skipped_id,
        "outside_span": out_of_span, "kept": kept, "bad_examples": stats.examples,
        "members": members, "outputs": written,
        # every TLE's epoch day (YYDDD), wanted or not: the file's coverage by epoch
        "epoch_days_all": dict(sorted(stats.epoch_days.items())),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bulk-dir", type=Path, required=True)
    p.add_argument("--ids", type=Path, required=True, help="candidate_ids.txt")
    p.add_argument("--start", required=True, help="first epoch day, YYYY-MM-DD")
    p.add_argument("--end", required=True, help="end day, YYYY-MM-DD (exclusive)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--buckets", type=int, default=16)
    p.add_argument("--jobs", type=int, default=1, help="files read in parallel (one process each)")
    args = p.parse_args(argv)

    want = read_ids(args.ids)
    inputs = sorted(
        [*args.bulk_dir.glob("*.zip"), *args.bulk_dir.glob("*.txt"),
         *args.bulk_dir.glob("*.txt.gz")]
    )
    if not inputs:
        raise SystemExit(f"no .zip, .txt or .txt.gz files in {args.bulk_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "extract_manifest.jsonl"
    done = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            e = json.loads(line)
            done[e["file"]] = e
    todo = []
    for path in inputs:
        sha = generation.file_sha256(path)
        prev = done.get(path.name)
        if (prev and prev["sha256"] == sha and prev["buckets"] == args.buckets
                and prev["span"] == [args.start, args.end]
                and all((args.out_dir / o).exists() for o in prev["outputs"])):
            print(f"skip (done) {path.name}")
            continue
        todo.append((path, sha))

    def record(path, sha, res):
        entry = {
            "file": path.name, "bytes": path.stat().st_size, "sha256": sha,
            "extracted_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
            "ids_file": str(args.ids), "ids_sha256": generation.file_sha256(args.ids),
            "span": [args.start, args.end], "buckets": args.buckets, **res,
        }
        with open(manifest, "a", encoding="utf-8") as m:
            m.write(json.dumps(entry) + "\n")
        print(json.dumps({k: entry[k] for k in ("file", "tles", "bad", "no_checksum",
                                                "trailing_backslash", "stray_lines",
                                                "other_objects", "outside_span", "kept")}),
              flush=True)

    if args.jobs <= 1:
        for path, sha in todo:
            print(f"reading {path.name} ({path.stat().st_size:,} bytes)", flush=True)
            record(path, sha, extract_one(path, want, args.start, args.end, args.out_dir,
                                          args.buckets))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futs = {pool.submit(extract_one, path, want, args.start, args.end, args.out_dir,
                                args.buckets): (path, sha) for path, sha in todo}
            for f in as_completed(futs):
                path, sha = futs[f]
                record(path, sha, f.result())
    return 0


if __name__ == "__main__":
    sys.exit(main())
