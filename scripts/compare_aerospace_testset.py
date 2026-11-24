"""Check the screening engine against the public conjunction-assessment test set.

The Aerospace Corporation and the Department of Commerce released (March 2026,
CC0-1.0) a catalog of OCM ephemerides for 2025-01-01..08 and the events their
CSieve tool found in an all-vs-all 10 km spherical screen (Auman, Murphy and
George, "Conjunction Assessment Verification Data and Process", user's guide).
This script runs :func:`orbit_correction.screening.screen` on the set's own
ephemerides for a subset of objects and compares the events with that answer
key, restricted to pairs with both objects in the subset (valid because a
spherical screen is pair-local: whether A and B conjunct depends on A and B
only). It tests the engine alone, against an implementation it shares nothing
with; the project's propagators are not involved.

Steps (each reads ``data/raw/aerospace_ivv/`` and writes into ``--out``, by
default ``data/derived/aerospace_ivv/``; both gitignored):

``extract``
    Streams the 22 GB ``.tar.gz`` once, without unpacking it to disk. A
    :class:`gzip.GzipFile` does the decompression, so the CRC-32 and length in
    the gzip trailer are checked when the stream is drained to its end, and the
    compressed bytes are hashed (SHA-256) on the way through. Every member is
    indexed (``member_index.parquet``/``.csv``) and only the files the subset
    rule selects (below) are written to ``ocm/``.

``subset``
    Checks every extracted file against the archive's own ``MD5SUM`` member,
    applies the header checks (USEABLE span, OD epoch) and writes
    ``subset.json``: the objects of each group, with exclusions and reasons.

``interp``
    Measures the reader's resampling error on held-back samples of every
    subset file (``interpolation_check.json``).

``compare``
    Resamples every object onto uniform TT grids (8-point Lagrange on positions,
    velocity from its derivative, by default; ``--method`` and ``--velocity``
    change that), screens each group, matches the engine's events with the
    restricted key at TCA tolerances of 1 s, 10 s and pair-only, and writes
    ``results.json`` (or ``--results``):

    * counts in both, key-only and engine-only, with engine-minus-key TCA, miss
      and relative-speed differences of the matched events;
    * the key's own J2000 states at its TCAs against the files resampled there
      (a check of the time and frame handling that involves no screening);
    * a dense re-check of matched events: the range minimum found directly from
      the file samples (0.5 s scan and Brent refinement, no 60 s grid);
    * every unmatched event (10 s tolerance) with a category: ``tca_tolerance``,
      ``window_edge``, ``span_edge``, ``interpolation`` (the engine's grid and
      the file samples put the minimum on opposite sides of 10 km; found by
      re-running the engine on the pair at 50 km), ``threshold_edge`` (miss
      within 10 m of 10 km), ``interpolation_or_data`` (the file samples
      themselves disagree with the key) or ``unexplained``.

Subset rule (stated here, recorded in every output)
---------------------------------------------------
Hash sampling: ``u(id) = int(SHA-256(f"{SEED}:{id}")[:8]) / 2**64``, seed
20261124. Uniform in [0, 1) and fixed per object, so the rule is reproducible
and can be applied to each archive member as it streams past, without a second
pass.

* ``main``: TLE-derived objects (IDs 5-62461 per the guide's Table 2), nominal
  file ``<id>.ocm`` only, ``u(id) < 0.10``, and two-body semimajor-axis
  altitude of the file's first state in [500, 600) km. The 500-600 km band is
  the densest part of the catalog, so a few hundred objects give thousands of
  key events.
* ``maneuver``: every file of IDs 90006-90190 (maneuvering ephemerides and
  their purpose-built victims), screened among themselves.
* ``fictitious``: IDs 99000-99008 (purpose-built fictitious objects and
  victims) plus every object the key pairs with them, screened among
  themselves.

At ``subset``, objects whose OD epoch is 14 days or more before the screening
window start are dropped (the guide's configuration rule 4), and in ``main``
objects whose USEABLE span does not cover the whole window are dropped (so the
group can be screened on one grid). The purpose-built groups keep partial spans
and are screened on the overlaps (see ``compare``).

Run from the repo root, with the archive and the key in ``data/raw/aerospace_ivv/``
(or pass ``--archive``, ``--key`` and ``--out``)::

    uv run python scripts/compare_aerospace_testset.py extract
    uv run python scripts/compare_aerospace_testset.py subset
    uv run python scripts/compare_aerospace_testset.py interp
    uv run python scripts/compare_aerospace_testset.py compare
    # sensitivity: the files' own velocities instead of the position derivative
    uv run python scripts/compare_aerospace_testset.py compare --velocity file \
        --results results_velocity_file.json

No file from the set is written outside ``--out``; nothing here is committed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import re
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from scipy.optimize import minimize_scalar

from orbit_correction import frames
from orbit_correction import screening as scr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aerospace_ocm as ocm  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
#: Default data locations (gitignored); override with --archive, --key and --out.
RAW = REPO / "data/raw/aerospace_ivv"
ARCHIVE = RAW / "AerospaceIVVDataset_20251009a.tar.gz"
KEY = RAW / "IVV_Releasable_Dataset_Spherical_DefaultHBR.csv.gz"
OUT = REPO / "data/derived/aerospace_ivv"

SEED = 20261124
MAIN_FRACTION = 0.10
MAIN_BAND_KM = (500.0, 600.0)
TLE_IDS = (5, 62461)
MANEUVER_IDS = (90006, 90190)
FICTITIOUS_IDS = (99000, 99008)

#: Screening window from the guide (UTC), and the spherical threshold.
WINDOW_UTC = ("2025-01-01T12:00:00", "2025-01-08T12:00:00")
THRESHOLD_KM = 10.0
#: OD epochs must be less than this many days before the window start (guide rule 4).
OD_MAX_AGE_DAYS = 14.0

#: Object type by ID range, from the guide's Table 2.
ID_RANGES = [
    ((5, 62461), "tle"),
    ((90006, 90190), "maneuver_or_victim"),
    ((95000, 95407), "historical_cdm"),
    ((99000, 99008), "fictitious_or_victim"),
    ((99996, 99999), "osiris_rex_or_victim"),
]

_SAFE_PART = re.compile(r"^[A-Za-z0-9_.\-]+$")


def hash_u(object_id: int, seed: int = SEED) -> float:
    """Deterministic uniform [0, 1) draw per object (the subset rule's sampler)."""
    d = hashlib.sha256(f"{seed}:{object_id}".encode()).digest()
    return int.from_bytes(d[:8], "big") / 2.0**64


def id_type(object_id: int | None) -> str:
    if object_id is None:
        return "unknown"
    for (lo, hi), name in ID_RANGES:
        if lo <= object_id <= hi:
            return name
    return "other"


def stem_id(basename: str) -> int | None:
    """Leading integer of a file name (``43476_candidate07.ocm`` -> 43476)."""
    m = re.match(r"^(\d+)", basename)
    return int(m.group(1)) if m else None


def read_key(path: Path = KEY, columns: list[str] | None = None) -> pa.Table:
    opts = pacsv.ConvertOptions(include_columns=columns) if columns else None
    return pacsv.read_csv(path, convert_options=opts)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------------------


class _HashingReader:
    """File wrapper that hashes and counts the compressed bytes as they are read."""

    def __init__(self, fh):
        self.fh = fh
        self.sha = hashlib.sha256()
        self.n = 0

    def read(self, size=-1):
        b = self.fh.read(size)
        self.sha.update(b)
        self.n += len(b)
        return b

    def readable(self):
        return True

    def close(self):
        self.fh.close()


def fictitious_partner_files(key: pa.Table) -> set[str]:
    """File names the key pairs with IDs 99000-99008 (both sides of each row)."""
    o1 = key["obj1"].to_numpy()
    o2 = key["obj2"].to_numpy()
    f1 = np.array(key["obj1_filename"].to_pylist())
    f2 = np.array(key["obj2_filename"].to_pylist())
    lo, hi = FICTITIOUS_IDS
    s = ((o1 >= lo) & (o1 <= hi)) | ((o2 >= lo) & (o2 <= hi))
    return set(f1[s].tolist()) | set(f2[s].tolist())


def select_group(basename: str, scan: dict, fict_files: set[str]) -> str | None:
    """Which subset group (if any) an archive member belongs to; see module docstring."""
    oid = stem_id(basename)
    if oid is None:
        return None
    if basename in fict_files or FICTITIOUS_IDS[0] <= oid <= FICTITIOUS_IDS[1]:
        return "fictitious"
    if MANEUVER_IDS[0] <= oid <= MANEUVER_IDS[1]:
        return "maneuver"
    if (TLE_IDS[0] <= oid <= TLE_IDS[1] and basename == f"{oid}.ocm"
            and hash_u(oid) < MAIN_FRACTION):
        alt = scan.get("alt_a_km", float("nan"))
        if MAIN_BAND_KM[0] <= alt < MAIN_BAND_KM[1]:
            return "main"
    return None


def cmd_extract(args) -> None:
    out = Path(args.out)
    (out / "ocm").mkdir(parents=True, exist_ok=True)
    key = read_key(Path(args.key), ["obj1", "obj2", "obj1_filename", "obj2_filename"])
    fict_files = fictitious_partner_files(key)
    print(f"fictitious group: {len(fict_files)} file names from the key", file=sys.stderr)

    rows: list[dict] = []
    t0 = time.time()
    raw_size = Path(args.archive).stat().st_size
    status = {"tar_complete": False, "gzip_trailer_checked": False, "error": None}
    n_uncompressed = 0
    with open(args.archive, "rb") as fh:
        hr = _HashingReader(fh)
        gz = gzip.GzipFile(fileobj=hr, mode="rb")
        try:
            with tarfile.open(fileobj=gz, mode="r|") as tf:
                for k, m in enumerate(tf):
                    row = {"name": m.name, "size": m.size, "type": m.type.decode()}
                    if not m.isfile():
                        rows.append(row)
                        continue
                    data = tf.extractfile(m).read()
                    n_uncompressed += len(data)
                    parts = m.name.split("/")
                    base = parts[-1]
                    row["dir"] = "/".join(parts[1:-1])
                    row["basename"] = base
                    oid = stem_id(base)
                    row["file_id"] = oid
                    row["id_type"] = id_type(oid)
                    row["sha256"] = hashlib.sha256(data).hexdigest()
                    if base == "MD5SUM":  # the set's own per-file checksums
                        (out / "MD5SUM").write_bytes(data)
                    try:
                        scan = ocm.scan_ocm_header(data)
                        st = ocm.state_summary(scan.pop("first_state"))
                        scan.update({k2: v for k2, v in st.items()})
                        row.update({k2.lower(): v for k2, v in scan.items()})
                    except Exception as exc:  # recorded, never silent
                        row["scan_error"] = f"{type(exc).__name__}: {exc}"
                    group = select_group(base, {"alt_a_km": row.get("alt_a_km", np.nan)},
                                         fict_files)
                    row["group"] = group or ""
                    if group:
                        if not all(_SAFE_PART.match(p) and p not in (".", "..") for p in parts):
                            raise RuntimeError(f"unsafe member path {m.name!r}")
                        dest = out / "ocm" / Path(*parts[1:])
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if not (dest.exists() and sha256_file(dest) == row["sha256"]):
                            tmp = dest.with_suffix(dest.suffix + ".part")
                            tmp.write_bytes(data)
                            tmp.replace(dest)
                        row["extracted_to"] = str(dest.relative_to(out))
                    rows.append(row)
                    if k % 2000 == 0:
                        el = time.time() - t0
                        print(f"{k} members, {hr.n / 1e9:.2f}/{raw_size / 1e9:.2f} GB read, "
                              f"{el:.0f} s", file=sys.stderr, flush=True)
            status["tar_complete"] = True
            # drain to the end of the gzip stream so its CRC-32/ISIZE trailer is checked
            trailing = 0
            while chunk := gz.read(1 << 20):
                trailing += len(chunk)
            status["gzip_trailer_checked"] = True
            status["uncompressed_bytes_after_tar_end"] = trailing
        except Exception as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"
            status["failed_at_member"] = rows[-1]["name"] if rows else None
            status["compressed_bytes_read"] = hr.n
        # read whatever is left so the hash covers the whole file
        while chunk := hr.read(1 << 22):
            pass
    status["compressed_bytes_read"] = hr.n
    status["archive_size"] = raw_size
    status["all_bytes_read"] = hr.n == raw_size
    status["archive_sha256"] = hr.sha.hexdigest()
    status["n_members"] = len(rows)
    status["n_files"] = sum(1 for r in rows if r.get("type") == "0")
    status["uncompressed_file_bytes"] = n_uncompressed
    status["wall_s"] = time.time() - t0

    # every row gets every column (from_pylist would take the columns of the first
    # row only, which is a directory entry)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    table = pa.Table.from_pylist([{c: r.get(c) for c in columns} for r in rows])
    pq.write_table(table, out / "member_index.parquet")
    pacsv.write_csv(table, out / "member_index.csv")
    groups: dict[str, int] = {}
    for r in rows:
        if r.get("group"):
            groups[r["group"]] = groups.get(r["group"], 0) + 1
    manifest = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "archive": str(args.archive),
        "key": str(args.key),
        "key_sha256": sha256_file(Path(args.key)),
        "stream": status,
        "rule": rule_dict(),
        "extracted_per_group": groups,
    }
    (out / "extract_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def rule_dict() -> dict:
    return {
        "seed": SEED,
        "hash": "u(id) = int.from_bytes(sha256(f'{seed}:{id}')[:8], 'big') / 2**64",
        "main": {
            "ids": list(TLE_IDS),
            "nominal_file_only": True,
            "fraction": MAIN_FRACTION,
            "alt_a_band_km": list(MAIN_BAND_KM),
            "alt_a": "two-body semimajor axis of the first sample of the first TRAJ block "
                     "(mu 398600.4418) minus 6378.137 km",
            "requires_full_window_coverage": True,
        },
        "maneuver": {"ids": list(MANEUVER_IDS), "files": "all"},
        "fictitious": {"ids": list(FICTITIOUS_IDS),
                       "plus": "every file the key pairs with these IDs"},
        "od_max_age_days": OD_MAX_AGE_DAYS,
        "window_utc": list(WINDOW_UTC),
        "threshold_km": THRESHOLD_KM,
    }


# --------------------------------------------------------------------------------------
# subset
# --------------------------------------------------------------------------------------


def _dt64(s) -> np.datetime64 | None:
    if s is None or (isinstance(s, float) and np.isnan(s)) or s == "":
        return None
    return np.datetime64(str(s).rstrip("Z"), "us")


def cmd_subset(args) -> None:
    out = Path(args.out)
    idx = pq.read_table(out / "member_index.parquet").to_pylist()
    w0, w1 = (np.datetime64(w, "us") for w in WINDOW_UTC)
    groups: dict[str, list[dict]] = {"main": [], "maneuver": [], "fictitious": []}
    excluded: list[dict] = []
    # the archive's MD5SUM member lists one checksum per file ("<md5>  <member name>")
    md5 = {}
    if (out / "MD5SUM").exists():
        for line in (out / "MD5SUM").read_text().splitlines():
            h, _, name = line.partition("  ")
            md5[name.strip()] = h.strip()
    n_md5_checked = 0
    for r in idx:
        g = r.get("group") or ""
        if not g:
            continue
        rec = {
            "file": r["basename"],
            "path": r["extracted_to"],
            "id": int(r["file_id"]),
            "sha256": r["sha256"],
            "useable_start": r.get("useable_start"),
            "useable_stop": r.get("useable_stop"),
            "od_epoch": r.get("od_epoch"),
            "n_traj_blocks": r.get("n_traj_blocks"),
            "alt_a_km": r.get("alt_a_km"),
        }
        reasons = []
        if md5:
            got = hashlib.md5((out / r["extracted_to"]).read_bytes()).hexdigest()
            rec["md5_ok"] = got == md5.get(r["name"])
            n_md5_checked += 1
            if not rec["md5_ok"]:
                reasons.append(f"MD5 {got} does not match the archive's MD5SUM")
        if r.get("scan_error"):
            reasons.append(f"scan error: {r['scan_error']}")
        if r.get("time_system") not in ("UTC", "TT", "TAI"):
            reasons.append(f"TIME_SYSTEM {r.get('time_system')}")
        od = _dt64(r.get("od_epoch"))
        if od is None:
            rec["od_age_days"] = None
        else:
            age = (w0 - od).astype(np.int64) / 86400e6
            rec["od_age_days"] = float(age)
            if age >= OD_MAX_AGE_DAYS:
                reasons.append(f"OD epoch {age:.2f} days before the window (>= {OD_MAX_AGE_DAYS})")
        us, ue = _dt64(r.get("useable_start")), _dt64(r.get("useable_stop"))
        if us is None or ue is None:
            reasons.append("no USEABLE span")
        elif ue <= w0 or us >= w1:
            reasons.append("USEABLE span outside the window")
        elif g == "main" and (us > w0 or ue < w1):
            reasons.append("main group: USEABLE span does not cover the whole window")
        if reasons:
            excluded.append({**rec, "group": g, "reasons": reasons})
        else:
            groups[g].append(rec)
    # the guide: ephemerides of one object designator are never screened against each other
    for g, recs in groups.items():
        ids = [x["id"] for x in recs]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            print(f"group {g}: object IDs with several files: {dup}", file=sys.stderr)
    subset = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "rule": rule_dict(),
        "groups": groups,
        "excluded": excluded,
        "md5sum_checked_files": n_md5_checked,
        "md5sum_mismatches": sum(1 for e in excluded if any("MD5" in x for x in e["reasons"])),
        "counts": {g: len(v) for g, v in groups.items()},
        "excluded_counts": {g: sum(1 for e in excluded if e["group"] == g) for g in groups},
    }
    (out / "subset.json").write_text(json.dumps(subset, indent=1, default=str))
    print(json.dumps({k: subset[k] for k in ("md5sum_checked_files", "md5sum_mismatches",
                                             "counts", "excluded_counts")}, indent=1))
    for e in excluded:
        print(e["group"], e["file"], "; ".join(e["reasons"]), file=sys.stderr)


# --------------------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------------------


def window_tt() -> tuple[float, float]:
    t = frames.utc_to_tt_seconds(list(WINDOW_UTC))
    return float(t[0]), float(t[1])


def uniform_grid(lo: float, hi: float, max_step_s: float = 60.0) -> np.ndarray:
    """Uniform grid with both ends on [lo, hi] and step <= ``max_step_s``.

    The 7-day window is exactly 10,080 steps of 60 s; partial overlaps get the
    largest step not above 60 s that divides them, so no part of a USEABLE
    overlap is left out of the screen.
    """
    n = max(1, int(np.ceil((hi - lo) / max_step_s - 1e-9)))
    return lo + (hi - lo) * np.arange(n + 1) / n


@dataclass
class Obj:
    """One ephemeris file of a group, ready to resample."""

    file: str
    id: int
    blocks: list
    lo: float  # USEABLE span intersected with the window, TT seconds
    hi: float


#: Resampling options for the screened trajectories (``compare --method/--velocity``).
RESAMPLE = {"method": "lagrange", "n_points": 8, "velocity": "derivative"}


def load_group(out: Path, recs: list[dict], w0: float, w1: float) -> list[Obj]:
    objs = []
    for rec in recs:
        f = ocm.read_ocm(out / rec["path"])
        blocks = ocm.tt_blocks(f)
        lo, hi = ocm.useable_span_tt(blocks)
        lo, hi = max(lo, w0), min(hi, w1)
        # a USEABLE bound written exactly at the window edge can land a few hundred
        # nanoseconds off it after UTC -> TT; snap it so such objects share one grid
        lo = w0 if abs(lo - w0) < 1e-3 else lo
        hi = w1 if abs(hi - w1) < 1e-3 else hi
        objs.append(Obj(rec["file"], rec["id"], blocks, lo, hi))
    return objs


def screen_group(objs: list[Obj], w0: float, w1: float, log: list[dict]) -> pa.Table:
    """Screen every pair of a group over its USEABLE overlap (within the window).

    Pairs are grouped by their overlap interval [max lo, min hi]; each distinct
    interval gets one :func:`screen` call on a uniform grid spanning it exactly,
    with only the objects that take part and a :class:`PairMask` keeping only
    the pairs whose overlap it is. For a group whose objects all cover the
    window this is a single all-vs-all call on the 60 s window grid. Objects
    with the same ID (alternative ephemerides of one object) are never paired.
    """
    n = len(objs)
    by_interval: dict[tuple[float, float], list[tuple[int, int]]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if objs[i].id == objs[j].id:
                continue
            lo, hi = max(objs[i].lo, objs[j].lo), min(objs[i].hi, objs[j].hi)
            if hi - lo < 1.0:
                continue
            by_interval.setdefault((lo, hi), []).append((i, j))
    tables = []
    for (lo, hi), pairs in sorted(by_interval.items()):
        members = sorted({i for p in pairs for i in p})
        pos = {i: k for k, i in enumerate(members)}
        t = uniform_grid(lo, hi)
        trajs, hows = [], {}
        for i in members:
            tr, how = ocm.to_trajectory(objs[i].blocks, objs[i].id, t, **RESAMPLE)
            trajs.append(tr)
            hows[how] = hows.get(how, 0) + 1
        ids = np.array([objs[i].id for i in members])
        keep = np.zeros(scr.pair_count(len(members)), dtype=bool)
        a = np.array([pos[i] for i, _ in pairs])
        b = np.array([pos[j] for _, j in pairs])
        keep[scr.pair_index(np.minimum(a, b), np.maximum(a, b), len(members))] = True
        res = scr.screen(trajs, THRESHOLD_KM, epoch_tt_s=w0, window_tt_s=(lo, hi),
                         keep=scr.PairMask(ids, keep))
        tables.append(res.events)
        log.append({"interval_tt_s": [lo, hi], "n_objects": len(members), "n_pairs": len(pairs),
                    "step_s": float(t[1] - t[0]), "resampled": hows,
                    "n_events": res.events.num_rows, "wall_s": res.stats["wall_s"]})
    if not tables:
        return scr.empty_event_table()
    return pa.concat_tables(tables).combine_chunks()


def restrict_key(key: pa.Table, files: set[str], w0: float) -> tuple[pa.Table, pa.Table]:
    """Key rows with both files in ``files``, as an event table, plus the raw rows."""
    f1 = np.array(key["obj1_filename"].to_pylist())
    f2 = np.array(key["obj2_filename"].to_pylist())
    sel = np.isin(f1, list(files)) & np.isin(f2, list(files))
    raw = key.filter(pa.array(sel))
    epochs = [str(e).replace(" ", "T") for e in raw["epoch"].to_pylist()]
    tca = (frames.utc_to_tt_seconds(epochs) if epochs else np.empty(0))
    events = scr.make_event_table(
        raw["obj1"].to_numpy(), raw["obj2"].to_numpy(), tca, np.asarray(tca) - w0,
        raw["min_range"].to_numpy(), raw["Vrel"].to_numpy(), np.full(raw.num_rows, THRESHOLD_KM),
    )
    # make_event_table sorts by TCA then pair: carry the raw rows in the same order
    order = np.lexsort((raw["obj2"].to_numpy(), raw["obj1"].to_numpy(), np.asarray(tca)))
    return events, raw.take(pa.array(order))


def dense_minimum(oa: Obj, ob: Obj, t_center: float, half_s: float = 120.0,
                  step_s: float = 0.5) -> dict | None:
    """Range minimum near ``t_center`` from the OCM samples directly (no engine).

    Evaluates both objects with the reader's default interpolant (8-point Lagrange) of the
    file samples (not the 60 s screening grid) every ``step_s`` over
    ``t_center`` +/- ``half_s`` (clipped to both USEABLE spans), takes the
    smallest range, and refines it with a bounded Brent search. Returns None if
    the smallest range sits on the edge of the searched interval (no interior
    local minimum there).
    """
    lo = max(t_center - half_s, oa.lo, ob.lo)
    hi = min(t_center + half_s, oa.hi, ob.hi)
    if hi - lo < 2 * step_s:
        return None

    def rng(t):
        ra, _, _ = ocm.resample(oa.blocks, np.atleast_1d(t))
        rb, _, _ = ocm.resample(ob.blocks, np.atleast_1d(t))
        return np.linalg.norm(rb - ra, axis=1)

    ts = np.arange(lo, hi + step_s / 2, step_s)
    d = rng(ts)
    k = int(np.argmin(d))
    if k == 0 or k == ts.size - 1:
        return {"edge": True, "tca_tt_s": float(ts[k]), "miss_km": float(d[k])}
    # search in the offset from the scan minimum: the bounded method's tolerance has a
    # relative term (sqrt(eps) |x|), which at TT ~ 8e8 s would be ~12 s
    t_k = float(ts[k])
    res = minimize_scalar(lambda x: float(rng(t_k + x)[0]), bounds=(-step_s, step_s),
                          method="bounded", options={"xatol": 1e-7})
    t = t_k + float(res.x)
    ra, va, _ = ocm.resample(oa.blocks, np.array([t]))
    rb, vb, _ = ocm.resample(ob.blocks, np.array([t]))
    return {"edge": False, "tca_tt_s": t, "miss_km": float(np.linalg.norm(rb - ra)),
            "rel_speed_km_s": float(np.linalg.norm(vb - va))}


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    a = np.abs(x)
    return {"n": int(x.size), "median_signed": float(np.median(x)),
            "median_abs": float(np.median(a)), "p95_abs": float(np.percentile(a, 95)),
            "max_abs": float(a.max())}


EDGE_S = 60.0  # an unmatched event this close to a window or USEABLE edge is an edge case
THRESHOLD_EDGE_KM = 0.010  # "within a few metres of 10 km": 10 m


def engine_grid_minimum(oa: Obj, ob: Obj, t_center: float) -> dict | None:
    """The engine's own range minimum for one pair near ``t_center``.

    Rebuilds the pair's screening grid exactly as :func:`screen_group` does (the
    grid depends only on the pair's overlap interval) and re-runs :func:`screen`
    on the two objects with a 50 km threshold over ``t_center`` +/- 600 s, so the
    engine reports the minimum whatever side of 10 km it falls on. Returns the
    event nearest ``t_center``, or None.
    """
    lo, hi = max(oa.lo, ob.lo), min(oa.hi, ob.hi)
    if hi - lo < 1.0:
        return None
    t = uniform_grid(lo, hi)
    ta, _ = ocm.to_trajectory(oa.blocks, oa.id, t, **RESAMPLE)
    tb, _ = ocm.to_trajectory(ob.blocks, ob.id, t, **RESAMPLE)
    res = scr.screen([ta, tb], 50.0, window_tt_s=(max(lo, t_center - 600.0),
                                                   min(hi, t_center + 600.0)))
    if res.events.num_rows == 0:
        return None
    tca = res.events["tca_tt_s"].to_numpy()
    j = int(np.argmin(np.abs(tca - t_center)))
    return {"tca_tt_s": float(tca[j]), "miss_km": float(res.events["miss_km"][j].as_py())}


def classify(side: str, ev: dict, dense: dict | None, grid: dict | None,
             w: tuple[float, float], span: tuple[float, float],
             other_side_near: dict | None) -> tuple[str, str]:
    """Category and a one-line reason for an unmatched event (module docstring).

    ``dense`` is the range minimum from the file samples, ``grid`` the engine's
    own minimum for the pair on its screening grid (any distance), both near the
    event's TCA.
    """
    tca = ev["tca_tt_s"]
    if other_side_near is not None:
        return ("tca_tolerance",
                f"same pair has an event {other_side_near['dt_s']:+.3f} s away on the other side "
                f"(beyond the matching tolerance)")
    if min(abs(tca - w[0]), abs(tca - w[1])) <= EDGE_S:
        return "window_edge", f"TCA within {EDGE_S:.0f} s of the screening window edge"
    if min(abs(tca - span[0]), abs(tca - span[1])) <= EDGE_S:
        return "span_edge", f"TCA within {EDGE_S:.0f} s of a USEABLE span edge"
    if side == "key" and not (span[0] <= tca <= span[1]):
        return "span_edge", "key TCA outside the pair's USEABLE overlap"
    ok_dense = dense is not None and not dense.get("edge")
    if ok_dense and grid is not None and (
            (dense["miss_km"] <= THRESHOLD_KM) != (grid["miss_km"] <= THRESHOLD_KM)):
        return ("interpolation",
                f"the engine's grid puts this minimum at {grid['miss_km']:.4f} km and the file "
                f"samples at {dense['miss_km']:.4f} km, on opposite sides of 10 km")
    if abs(ev["miss_km"] - THRESHOLD_KM) <= THRESHOLD_EDGE_KM or (
            ok_dense and abs(dense["miss_km"] - THRESHOLD_KM) <= THRESHOLD_EDGE_KM):
        return "threshold_edge", "miss within 10 m of 10 km"
    if not ok_dense:
        return "unexplained", "no interior range minimum in the OCM samples near this TCA"
    if dense["miss_km"] > THRESHOLD_KM and side == "key":
        return ("interpolation_or_data",
                f"the file samples give {dense['miss_km']:.4f} km (> 10 km) here")
    return "unexplained", (f"file samples {dense['miss_km']:.4f} km, engine grid "
                           f"{grid['miss_km'] if grid else float('nan'):.4f} km")


def compare_group(name: str, recs: list[dict], key: pa.Table, out: Path,
                  w0: float, w1: float) -> dict:
    t_start = time.time()
    objs = load_group(out, recs, w0, w1)
    by_file = {o.file: o for o in objs}
    by_id: dict[int, list[Obj]] = {}
    for o in objs:
        by_id.setdefault(o.id, []).append(o)
    log: list[dict] = []
    engine = screen_group(objs, w0, w1, log)
    key_events, key_raw = restrict_key(key, set(by_file), w0)
    result: dict = {
        "group": name, "n_objects": len(objs), "n_files": len(by_file),
        "n_screen_calls": len(log),
        "engine_wall_s": float(sum(x["wall_s"] for x in log)),
        "resampled": {k: sum(x["resampled"].get(k, 0) for x in log)
                      for k in ("samples", "lagrange", "hermite", "mixed")},
        "n_key_events": key_events.num_rows, "n_engine_events": engine.num_rows,
        "n_key_pairs": len(set(zip(key_events["norad_id_1"].to_pylist(),
                                   key_events["norad_id_2"].to_pylist(), strict=True))),
    }
    if name != "main":
        result["screen_calls"] = log
    # files used by the key, per row (for groups where one ID has several files)
    kf1 = key_raw["obj1_filename"].to_pylist()
    kf2 = key_raw["obj2_filename"].to_pylist()

    def pair_objs(a: int, b: int, row: int | None) -> tuple[Obj, Obj]:
        if row is not None:
            return by_file[kf1[row]], by_file[kf2[row]]
        return by_id[a][0], by_id[b][0]

    def span_of(oa: Obj, ob: Obj) -> tuple[float, float]:
        return max(oa.lo, ob.lo), min(oa.hi, ob.hi)

    ka, kb, kt = (key_events[c].to_numpy() for c in ("norad_id_1", "norad_id_2", "tca_tt_s"))
    km = key_events["miss_km"].to_numpy()
    ea, eb, et = (engine[c].to_numpy() for c in ("norad_id_1", "norad_id_2", "tca_tt_s"))
    em = engine["miss_km"].to_numpy()
    ev_ = engine["rel_speed_km_s"].to_numpy()
    kv_ = key_events["rel_speed_km_s"].to_numpy()

    per_tol = {}
    for tol in (1.0, 10.0, None):
        mt = scr.match_events(key_events, engine, tca_tol_s=tol)
        both = mt.ref_to_cand >= 0
        r_i = np.flatnonzero(both)
        c_i = mt.ref_to_cand[both]
        entry = {
            "tca_tol_s": tol,
            "both": int(both.sum()),
            "key_only": int((~both).sum()),
            "engine_only": int((mt.cand_to_ref < 0).sum()),
            "dtca_s_engine_minus_key": _stats(et[c_i] - kt[r_i]),
            "dmiss_m_engine_minus_key": _stats((em[c_i] - km[r_i]) * 1e3),
            "dvrel_m_s_engine_minus_key": _stats((ev_[c_i] - kv_[r_i]) * 1e3),
        }
        score = scr.screening_metrics(key_events, engine, tca_tol_s=tol)
        entry["event_recall"] = score.event_recall
        entry["pair_recall"] = score.pair_recall
        per_tol["none" if tol is None else f"{tol:g}s"] = entry
        if tol == 10.0:
            match10 = mt
    result["matching"] = per_tol

    # State check: the key's J2000 states at its TCA against the files, resampled there.
    # Separates time and frame handling from the engine (no screening involved).
    dpos, dvel_der, dvel_file = [], [], []
    for row in range(key_events.num_rows):
        oa, ob = pair_objs(int(ka[row]), int(kb[row]), row)
        t = kt[row]
        for o, cols in ((oa, ("x1", "y1", "z1", "vx1", "vy1", "vz1")),
                        (ob, ("x2", "y2", "z2", "vx2", "vy2", "vz2"))):
            if not (o.lo <= t <= o.hi):
                continue
            kx = np.array([key_raw[c][row].as_py() for c in cols])
            r, v, _ = ocm.resample(o.blocks, np.array([t]))
            s_der = ocm.from_gcrf(np.hstack([r, v]), "EME2000")[0]
            r, v, _ = ocm.resample(o.blocks, np.array([t]), velocity="file")
            s_file = ocm.from_gcrf(np.hstack([r, v]), "EME2000")[0]
            dpos.append(np.linalg.norm(s_der[:3] - kx[:3]))
            dvel_der.append(np.linalg.norm(s_der[3:] - kx[3:]))
            dvel_file.append(np.linalg.norm(s_file[3:] - kx[3:]))
    result["key_state_vs_files_at_key_tca"] = {
        "dpos_m": _stats(np.array(dpos) * 1e3),
        "dvel_mm_s_derivative": _stats(np.array(dvel_der) * 1e6),
        "dvel_mm_s_file": _stats(np.array(dvel_file) * 1e6),
        "note": "key J2000 states at the key TCA vs the files resampled there (8-point "
                "Lagrange on positions; velocity as the derivative of that fit, or the file's "
                "velocities Lagrange-interpolated), converted back to EME2000",
    }

    # Where the key's TCAs sit relative to the files' TRAJ blocks: inside an
    # intermediate block (the burn arc of a coast-burn-coast file) or within 10 minutes
    # of a block boundary of either object. Those are the stressing cases.
    in_mid, near_bound = 0, 0
    for row in range(key_events.num_rows):
        oa, ob = pair_objs(int(ka[row]), int(kb[row]), row)
        t = kt[row]
        mid = near = False
        for o in (oa, ob):
            bounds = [b.useable_tt_s for b in o.blocks]
            for bi, (lo_b, hi_b) in enumerate(bounds):
                if 0 < bi < len(bounds) - 1 and lo_b <= t <= hi_b:
                    mid = True
            for lo_b, _ in bounds[1:]:
                if abs(t - lo_b) <= 600.0:
                    near = True
        in_mid += mid
        near_bound += near
    result["key_events_in_intermediate_traj_block"] = in_mid
    result["key_events_within_600s_of_a_block_boundary"] = near_bound
    result["key_events_miss_within_10m_of_threshold"] = int(
        np.count_nonzero(km >= THRESHOLD_KM - THRESHOLD_EDGE_KM))

    # Dense check of matched events: engine grid (60 s, cubic Hermite) vs file samples.
    rng_ = np.random.default_rng(SEED)
    matched_rows = np.flatnonzero(match10.ref_to_cand >= 0)
    sample = (matched_rows if matched_rows.size <= args_dense_n
              else np.sort(rng_.choice(matched_rows, args_dense_n, replace=False)))
    d_t, d_m, k_t, k_m = [], [], [], []
    for row in sample:
        oa, ob = pair_objs(int(ka[row]), int(kb[row]), int(row))
        c = match10.ref_to_cand[row]
        dm = dense_minimum(oa, ob, float(et[c]), half_s=60.0)
        if dm is None or dm["edge"]:
            continue
        d_t.append(et[c] - dm["tca_tt_s"])
        d_m.append((em[c] - dm["miss_km"]) * 1e3)
        k_t.append(kt[row] - dm["tca_tt_s"])
        k_m.append((km[row] - dm["miss_km"]) * 1e3)
    result["dense_check_matched"] = {
        "n_checked": len(d_t),
        "engine_minus_dense": {"dtca_s": _stats(d_t), "dmiss_m": _stats(d_m)},
        "key_minus_dense": {"dtca_s": _stats(k_t), "dmiss_m": _stats(k_m)},
        "note": "dense = range minimum from the file samples (8-point Lagrange, 0.5 s scan, "
                "Brent refinement), independent of the engine's 60 s grid",
    }

    # Every unmatched event at the 10 s tolerance, with a reason.
    unmatched = []
    for row in np.flatnonzero(match10.ref_to_cand < 0):
        a, b = int(ka[row]), int(kb[row])
        oa, ob = pair_objs(a, b, int(row))
        dm = dense_minimum(oa, ob, float(kt[row]))
        near = None
        same = np.flatnonzero((ea == a) & (eb == b))
        if same.size:
            j = same[np.argmin(np.abs(et[same] - kt[row]))]
            near = {"dt_s": float(et[j] - kt[row]), "engine_miss_km": float(em[j])}
            if abs(near["dt_s"]) > 600:
                near = None
        ev = {"tca_tt_s": float(kt[row]), "miss_km": float(km[row])}
        grid = engine_grid_minimum(oa, ob, float(kt[row]))
        cat, why = classify("key", ev, dm, grid, (w0, w1), span_of(oa, ob), near)
        unmatched.append({
            "side": "key_only", "obj1": a, "obj2": b, "files": [kf1[row], kf2[row]],
            "tca_utc": key_raw["epoch"][int(row)].as_py().isoformat(),
            "tca_tt_s": float(kt[row]), "miss_km": float(km[row]),
            "rel_speed_km_s": float(kv_[row]), "dense": dm, "engine_grid": grid,
            "nearest_engine_event": near,
            "category": cat, "reason": why,
        })
    for c in np.flatnonzero(match10.cand_to_ref < 0):
        a, b = int(ea[c]), int(eb[c])
        oa, ob = pair_objs(a, b, None)
        dm = dense_minimum(oa, ob, float(et[c]))
        near = None
        same = np.flatnonzero((ka == a) & (kb == b))
        if same.size:
            j = same[np.argmin(np.abs(kt[same] - et[c]))]
            near = {"dt_s": float(kt[j] - et[c]), "key_miss_km": float(km[j])}
            if abs(near["dt_s"]) > 600:
                near = None
        ev = {"tca_tt_s": float(et[c]), "miss_km": float(em[c])}
        grid = {"tca_tt_s": float(et[c]), "miss_km": float(em[c])}
        cat, why = classify("engine", ev, dm, grid, (w0, w1), span_of(oa, ob), near)
        unmatched.append({
            "side": "engine_only", "obj1": a, "obj2": b,
            "tca_utc": str(frames.tt_seconds_to_utc_isot(float(et[c]))),
            "tca_tt_s": float(et[c]), "miss_km": float(em[c]),
            "rel_speed_km_s": float(ev_[c]), "dense": dm, "nearest_key_event": near,
            "category": cat, "reason": why,
        })
    cats: dict[str, dict[str, int]] = {}
    for u in unmatched:
        cats.setdefault(u["side"], {}).setdefault(u["category"], 0)
        cats[u["side"]][u["category"]] += 1
    result["unmatched_by_category"] = cats
    result["unmatched"] = unmatched
    result["wall_s"] = time.time() - t_start
    return result


args_dense_n = 2000


def holdback_errors(block: ocm.TtBlock, every: int, method: str,
                    n: int) -> tuple[np.ndarray, np.ndarray]:
    """Position errors (km) at samples held back from the fit, and velocity gaps (km/s).

    Drops sample k for every k = every//2 + i*every (so ``every=2`` drops every
    other sample and doubles the spacing; ``every=10`` drops one in ten and keeps
    the native spacing around each gap), interpolates the remaining samples at the
    dropped times and compares positions. ``method`` is ``"lagrange"`` (``n``
    points, positions only) or ``"hermite"`` (``n`` samples, positions and file
    velocities). The second array is |interpolated velocity - file velocity| at
    the dropped samples: for Lagrange it measures how far the file's velocities
    are from the derivative of its positions, not an interpolation error. Only
    held-back samples with enough kept samples on both sides are used.
    """
    nn = block.t_tt_s.size
    drop = np.arange(every // 2, nn, every)
    keep = np.setdiff1d(np.arange(nn), drop)
    if keep.size < 2 * n + 2:
        return np.empty(0), np.empty(0)
    tk = block.t_tt_s[keep]
    # n kept samples on each side, so every fit is centred (no end effects)
    drop = drop[(block.t_tt_s[drop] > tk[n]) & (block.t_tt_s[drop] < tk[-n - 1])]
    if drop.size == 0:
        return np.empty(0), np.empty(0)
    s = block.states_gcrf
    if method == "lagrange":
        r, v = ocm.lagrange_interpolate(tk, s[keep, :3], block.t_tt_s[drop], n_points=n)
    else:
        r, v = ocm.hermite_interpolate(tk, s[keep, :3], s[keep, 3:], block.t_tt_s[drop], m=n)
    return np.linalg.norm(r - s[drop, :3], axis=1), np.linalg.norm(v - s[drop, 3:], axis=1)


VARIANTS = [("lagrange", 6), ("lagrange", 8), ("lagrange", 10), ("hermite", 4)]


def cmd_interp(args) -> None:
    """Interpolation error of the reader's resampling, on held-back samples."""
    out = Path(args.out)
    subset = json.loads((out / "subset.json").read_text())
    res: dict = {"created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                 "method": "errors at held-back samples, per TRAJ block: lagrangeN = "
                           "positions only, N points (degree N - 1); hermite4 = positions and "
                           "file velocities at 4 samples (degree 7). vel = |interpolated "
                           "velocity - file velocity|: for lagrange, the gap between the file's "
                           "velocities and the derivative of its positions",
                 "groups": {}}
    for name, recs in subset["groups"].items():
        g: dict = {}
        spacing = []
        for rec in recs:
            f = ocm.read_ocm(out / rec["path"])
            for b in ocm.tt_blocks(f):
                spacing.append(np.median(np.diff(b.t_tt_s)))
                for every in (10, 2):
                    for method, nn in VARIANTS:
                        ep, evv = holdback_errors(b, every, method, nn)
                        if ep.size == 0:
                            continue
                        k = f"every{every}_{method}{nn}"
                        g.setdefault(k, {"pos": [], "vel": [], "file_max_pos": []})
                        g[k]["pos"].append(ep)
                        g[k]["vel"].append(evv)
                        g[k]["file_max_pos"].append((float(ep.max()), rec["file"]))
        summary = {"n_files": len(recs),
                   "sample_spacing_s": {"median": float(np.median(spacing)),
                                        "min": float(np.min(spacing)),
                                        "max": float(np.max(spacing))}}
        for k, d in g.items():
            p = np.concatenate(d["pos"]) * 1e3
            v = np.concatenate(d["vel"]) * 1e6
            worst = sorted(d["file_max_pos"], reverse=True)[:5]
            summary[k] = {"n_points": int(p.size),
                          "pos_m": {"median": float(np.median(p)),
                                    "p95": float(np.percentile(p, 95)),
                                    "p99": float(np.percentile(p, 99)), "max": float(p.max())},
                          "vel_gap_mm_s": {"median": float(np.median(v)),
                                           "p95": float(np.percentile(v, 95)),
                                           "max": float(v.max())},
                          "worst_files_m": [(round(x * 1e3, 4), f) for x, f in worst]}
        res["groups"][name] = summary
        print(name, json.dumps({k: (v if not isinstance(v, dict) or "pos_m" not in v else
                                    v["pos_m"]) for k, v in summary.items()}), file=sys.stderr)
    (out / "interpolation_check.json").write_text(json.dumps(res, indent=1))


def cmd_compare(args) -> None:
    global args_dense_n
    args_dense_n = args.dense_n
    RESAMPLE.update(method=args.method, velocity=args.velocity)
    out = Path(args.out)
    subset = json.loads((out / "subset.json").read_text())
    key = read_key(Path(args.key))
    w0, w1 = window_tt()
    results = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "window_utc": list(WINDOW_UTC), "window_tt_s": [w0, w1],
        "threshold_km": THRESHOLD_KM,
        "threshold_note": "10 km on miss distance alone; the 0.5 m hard-body radius is used "
                          "only for Pc in the key (user's guide, spherical configuration)",
        "time_and_frame": {
            "ocm_time_system": "UTC -> TT via ERFA at each block's first epoch, plus elapsed "
                               "time; leap seconds inside a block refused (none in 2025-01)",
            "tt_minus_utc_s": float(frames.tt_minus_utc_s(WINDOW_UTC[0])),
            "frame": "EME2000 -> GCRF with the constant IAU 2006 frame bias (0.0230 arcsec); "
                     "ranges are unaffected",
            "key_tca": "key 'epoch' column (UTC) -> TT the same way",
        },
        "rule": subset["rule"],
        "resample": dict(RESAMPLE),
        "edge_s": EDGE_S, "threshold_edge_km": THRESHOLD_EDGE_KM,
        "groups": {},
    }
    for name in args.groups:
        recs = subset["groups"][name]
        print(f"group {name}: {len(recs)} files", file=sys.stderr, flush=True)
        results["groups"][name] = compare_group(name, recs, key, out, w0, w1)
        g = results["groups"][name]
        print(json.dumps({k: g[k] for k in ("n_objects", "n_key_events", "n_engine_events",
                                            "unmatched_by_category")}, default=str),
              file=sys.stderr, flush=True)
    (out / args.results).write_text(json.dumps(results, indent=1, default=float))
    summary = {n: {"objects": g["n_objects"], "key": g["n_key_events"],
                   "engine": g["n_engine_events"],
                   "matching": {t: {k: v for k, v in m.items() if k in
                                    ("both", "key_only", "engine_only")}
                                for t, m in g["matching"].items()},
                   "unmatched": g["unmatched_by_category"]}
               for n, g in results["groups"].items()}
    print(json.dumps(summary, indent=1))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("extract", help="stream the archive once: index, check, keep the subset")
    pe.add_argument("--archive", default=str(ARCHIVE))
    pe.add_argument("--key", default=str(KEY))
    pe.add_argument("--out", default=str(OUT))
    pe.set_defaults(func=cmd_extract)
    ps = sub.add_parser("subset", help="apply the header checks and write subset.json")
    ps.add_argument("--out", default=str(OUT))
    ps.set_defaults(func=cmd_subset)
    pi = sub.add_parser("interp", help="resampling error on held-back samples")
    pi.add_argument("--out", default=str(OUT))
    pi.set_defaults(func=cmd_interp)
    pc_ = sub.add_parser("compare", help="screen the subset and compare with the key")
    pc_.add_argument("--key", default=str(KEY))
    pc_.add_argument("--out", default=str(OUT))
    pc_.add_argument("--groups", nargs="+", default=["main", "maneuver", "fictitious"])
    pc_.add_argument("--method", choices=["lagrange", "hermite"], default="lagrange")
    pc_.add_argument("--velocity", choices=["derivative", "file"], default="derivative")
    pc_.add_argument("--results", default="results.json", help="output file name in --out")
    pc_.add_argument("--dense-n", type=int, default=2000,
                     help="matched events re-checked against the file samples (random, seeded)")
    pc_.set_defaults(func=cmd_compare)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
