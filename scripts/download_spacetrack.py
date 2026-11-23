"""Download Space-Track ``gp_history`` element sets for a list of NORAD IDs and a date range.

Run it yourself, with your own account: the credentials come from ``.env``
(``SPACETRACK_USER``, ``SPACETRACK_PASSWORD``; see ``.env.example``). Output goes
to ``data/raw/spacetrack/`` (gitignored). Space-Track data is never committed or
redistributed (Space-Track user agreement).

    uv run python scripts/download_spacetrack.py --ids 26998,22236 \\
        --start 2024-02-14 --end 2024-03-01
    uv run python scripts/download_spacetrack.py --ids-file ids.txt \\
        --start 2019-01-01 --end 2025-12-31 --dry-run

What it does:

* Logs in once, queries ``class/gp_history`` in batches of IDs, logs out.
* Paces requests to stay under Space-Track's published limits (30 requests per
  minute, 300 per hour) with margin: by default one request every 3 s and at
  most 250 per rolling hour.
* Writes each batch as gzipped JSON plus a manifest entry (query, time, record
  count, SHA-256). Batches already on disk are skipped, so an interrupted run
  resumes without re-downloading (Space-Track asks that ``gp_history`` be
  fetched once and stored locally).
* ``--dry-run`` prints the queries and touches neither the network nor ``.env``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import sys
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

BASE_URL = "https://www.space-track.org"
LOGIN_URL = f"{BASE_URL}/ajaxauth/login"
LOGOUT_URL = f"{BASE_URL}/ajaxauth/logout"

#: Space-Track's published limits are 30/minute and 300/hour; stay under both.
DEFAULT_MIN_INTERVAL_S = 3.0
DEFAULT_MAX_PER_HOUR = 250


def parse_ids(text: str) -> list[int]:
    """Parse NORAD IDs separated by commas, whitespace or newlines; ``#`` starts a comment."""
    ids: list[int] = []
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        for tok in line.replace(",", " ").split():
            n = int(tok)
            if n <= 0:
                raise ValueError(f"NORAD ID must be positive: {tok}")
            ids.append(n)
    return sorted(set(ids))


def batches(ids: Sequence[int], size: int) -> list[list[int]]:
    """Split IDs into consecutive batches of at most ``size``."""
    if size <= 0:
        raise ValueError("batch size must be positive")
    return [list(ids[i : i + size]) for i in range(0, len(ids), size)]


def build_query_url(ids: Iterable[int], start: str, end: str) -> str:
    """URL for ``gp_history`` element sets of ``ids`` with ``start <= EPOCH < end``.

    Dates are ``YYYY-MM-DD`` (UTC). A date-only bound means 00:00:00 that day,
    so ``end`` is effectively exclusive: pass the day after the last day you want.
    """
    dt.date.fromisoformat(start)
    dt.date.fromisoformat(end)
    if start >= end:
        raise ValueError(f"start {start} must be before end {end}")
    id_list = ",".join(str(int(i)) for i in ids)
    if not id_list:
        raise ValueError("no IDs")
    return (
        f"{BASE_URL}/basicspacedata/query/class/gp_history/NORAD_CAT_ID/{id_list}"
        f"/EPOCH/{start}--{end}/orderby/NORAD_CAT_ID%20asc,EPOCH%20asc/format/json"
    )


def batch_filename(ids: Sequence[int], start: str, end: str) -> str:
    """Stable file name for one batch (hash of its IDs, so reruns find it)."""
    h = hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:12]
    return f"gp_history_{start}_{end}_{ids[0]}-{ids[-1]}_{h}.json.gz"


class RateLimiter:
    """Waits so requests stay under a minimum spacing and an hourly cap.

    ``clock`` and ``sleep`` are injectable so the pacing logic can be tested
    without waiting.
    """

    def __init__(
        self,
        min_interval_s: float = DEFAULT_MIN_INTERVAL_S,
        max_per_hour: int = DEFAULT_MAX_PER_HOUR,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval_s = min_interval_s
        self.max_per_hour = max_per_hour
        self.clock = clock
        self.sleep = sleep
        self.sent: deque[float] = deque()

    def wait(self) -> float:
        """Block until the next request may go; record it; return seconds waited."""
        now = self.clock()
        delay = 0.0
        if self.sent:
            delay = max(delay, self.sent[-1] + self.min_interval_s - now)
        while self.sent and self.sent[0] <= now - 3600.0:
            self.sent.popleft()
        if len(self.sent) >= self.max_per_hour:
            delay = max(delay, self.sent[0] + 3600.0 - now)
        if delay > 0:
            self.sleep(delay)
        self.sent.append(now + max(delay, 0.0))
        return max(delay, 0.0)


def _login(session, user: str, password: str) -> None:
    r = session.post(LOGIN_URL, data={"identity": user, "password": password}, timeout=60)
    r.raise_for_status()
    if "Failed" in r.text:
        raise SystemExit("Space-Track login failed (check SPACETRACK_USER / SPACETRACK_PASSWORD)")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ids", help="comma-separated NORAD IDs")
    g.add_argument("--ids-file", type=Path, help="file of NORAD IDs (one per line or commas)")
    p.add_argument("--start", required=True, help="first epoch date, YYYY-MM-DD (UTC)")
    p.add_argument("--end", required=True, help="end epoch date, YYYY-MM-DD (UTC, exclusive)")
    p.add_argument("--out-dir", type=Path, default=Path("data/raw/spacetrack"))
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--min-interval-s", type=float, default=DEFAULT_MIN_INTERVAL_S)
    p.add_argument("--max-per-hour", type=int, default=DEFAULT_MAX_PER_HOUR)
    p.add_argument("--dry-run", action="store_true", help="print queries; no network, no .env")
    args = p.parse_args(argv)

    ids = parse_ids(args.ids if args.ids else args.ids_file.read_text())
    plan = [(b, build_query_url(b, args.start, args.end)) for b in batches(ids, args.batch_size)]
    if args.dry_run:
        for b, url in plan:
            print(f"{batch_filename(b, args.start, args.end)}  {url}")
        print(f"{len(plan)} requests for {len(ids)} objects (dry run; nothing sent)")
        return 0

    import requests
    from dotenv import load_dotenv

    load_dotenv()
    user, password = os.environ.get("SPACETRACK_USER"), os.environ.get("SPACETRACK_PASSWORD")
    if not user or not password:
        raise SystemExit("set SPACETRACK_USER and SPACETRACK_PASSWORD in .env")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"
    limiter = RateLimiter(args.min_interval_s, args.max_per_hour)
    with requests.Session() as s:
        limiter.wait()
        _login(s, user, password)
        try:
            for b, url in plan:
                out = args.out_dir / batch_filename(b, args.start, args.end)
                if out.exists():
                    print(f"skip (exists) {out.name}")
                    continue
                limiter.wait()
                r = s.get(url, timeout=300)
                r.raise_for_status()
                records = r.json()
                if not isinstance(records, list):
                    raise SystemExit(f"unexpected response for {out.name}: {str(records)[:200]}")
                body = r.content
                tmp = out.with_suffix(".tmp")
                with gzip.open(tmp, "wb") as f:
                    f.write(body)
                tmp.replace(out)
                entry = {
                    "file": out.name,
                    "query": url,
                    "retrieved_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                    "records": len(records),
                    "sha256_uncompressed": hashlib.sha256(body).hexdigest(),
                }
                with open(manifest_path, "a", encoding="utf-8") as m:
                    m.write(json.dumps(entry) + "\n")
                print(f"wrote {out.name}: {len(records)} element sets")
        finally:
            limiter.wait()
            s.get(LOGOUT_URL, timeout=60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
