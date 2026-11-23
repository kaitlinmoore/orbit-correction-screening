"""Regenerate the small committed test fixtures in ``tests/fixtures/``.

    uv run python scripts/make_fixtures.py            # all fixtures
    uv run python scripts/make_fixtures.py --synthetic-only

Fixtures and their provenance (also in ``tests/fixtures/README.md``):

* ``sw_quirks.txt``: synthetic space-weather file in CelesTrak's fixed-width
  format, reproducing the three quirks of the published file (blank F10.7 Q
  flags in predicted rows, blank Kp/Ap/Cp/C9 in monthly-predicted rows, no
  MONTHLY_FIT section). Values are made up. Lines are written here with explicit
  field widths, independently of ``spaceweather.py``'s formatter.
* ``spacetrack_gp_history_synthetic.json``: synthetic element sets in
  Space-Track's ``gp_history`` JSON layout (string values, extra keywords).
  Values are made up; no Space-Track data is used anywhere.
* ``celestrak_gp_sample.json`` / ``.csv`` and ``satcat_sample.csv``: a few
  records cut from CelesTrak GP and SATCAT downloads in ``data/raw/celestrak/``
  (needs those files; skipped with ``--synthetic-only``). The CSV is written
  from the JSON records with CelesTrak's OMM CSV column names.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
RAW = ROOT / "data" / "raw" / "celestrak"

# Validation-case objects plus objects chosen to exercise each catalog filter.
VALIDATION_IDS = [24946, 22675, 26998, 22236, 13777, 2828, 19826, 36123, 7219, 26207]


# ---------------------------------------------------------------- space weather


def _sw_line(d: date, kp: list[int] | None, ap: list[int] | None, cp: float | None,
             c9: int | None, isn: int, f107_adj: float, q: int | None, adj_ctr: float,
             adj_lst: float, f107_obs: float, obs_ctr: float, obs_lst: float,
             bsrn: int, nd: int) -> str:
    """One 130-column row, FORMAT(I4,I3,I3,I5,I3,8I3,I4,8I4,I4,F4.1,I2,I4,F6.1,I2,5F6.1)."""
    s = f"{d.year:4d} {d.month:02d} {d.day:02d}{bsrn:5d}{nd:3d}"
    if kp is None:
        s += " " * (8 * 3 + 4 + 8 * 4 + 4 + 4 + 2)
    else:
        s += "".join(f"{k:3d}" for k in kp) + f"{sum(kp):4d}"
        s += "".join(f"{a:4d}" for a in ap) + f"{round(sum(ap) / 8):4d}"
        s += f"{cp:4.1f}{c9:2d}"
    s += f"{isn:4d}{f107_adj:6.1f}" + ("  " if q is None else f"{q:2d}")
    s += f"{adj_ctr:6.1f}{adj_lst:6.1f}{f107_obs:6.1f}{obs_ctr:6.1f}{obs_lst:6.1f}"
    assert len(s) == 130, len(s)
    return s


def make_sw_fixture(path: Path) -> None:
    """120 observed days, 5 daily-predicted days, 3 monthly-predicted months."""
    start = date(2024, 1, 1)
    n_obs, n_daily = 120, 5
    days = [start + timedelta(days=i) for i in range(n_obs + n_daily)]
    # Smooth synthetic F10.7 with a 27-day rotation signal; Kp/Ap vary by day.
    f107 = [round(150.0 + 30.0 * math.sin(2 * math.pi * i / 27.0), 1) for i in range(len(days))]
    lines = [
        "DATATYPE CssiSpaceWeather",
        "VERSION 1.2",
        "UPDATED 2024 Apr 30 00:00:00 UTC",
        "# Synthetic test fixture in CelesTrak SW-All.txt layout. Values are made up.",
        "# FORMAT(I4,I3,I3,I5,I3,8I3,I4,8I4,I4,F4.1,I2,I4,F6.1,I2,5F6.1)",
        "",
        f"NUM_OBSERVED_POINTS {n_obs}",
        "BEGIN OBSERVED",
    ]

    def row(i: int, q: int | None) -> str:
        kp = [(i * 7 + k * 3) % 50 for k in range(8)]
        ap = [(i * 5 + k * 2) % 40 for k in range(8)]
        lst = sum(f107[max(0, i - 80) : i + 1]) / len(f107[max(0, i - 80) : i + 1])
        return _sw_line(days[i], kp, ap, 0.1 * (i % 20), i % 10, 100 + i % 50, f107[i] - 1.0, q,
                        150.0, round(lst, 1) - 1.0, f107[i], 150.0, round(lst, 1),
                        2600 + i // 27, i % 27 + 1)

    lines += [row(i, 0) for i in range(n_obs)]
    lines += ["END OBSERVED", "", f"NUM_DAILY_PREDICTED_POINTS {n_daily}", "BEGIN DAILY_PREDICTED"]
    lines += [row(i, None) for i in range(n_obs, n_obs + n_daily)]  # quirk 1: blank Q
    lines += ["END DAILY_PREDICTED", "", "NUM_MONTHLY_PREDICTED_POINTS 3",
              "BEGIN MONTHLY_PREDICTED"]
    for j, m in enumerate((6, 7, 8)):  # quirk 2: blank Kp/Ap/Cp/C9 (and Q)
        lines.append(_sw_line(date(2024, m, 1), None, None, None, None, 120 - j, 160.0 - j, None,
                              158.0, 157.0, 161.0 - j, 159.0, 158.0, 2610 + j, 1))
    lines += ["END MONTHLY_PREDICTED", ""]  # quirk 3: no MONTHLY_FIT section
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# ---------------------------------------------------------------- Space-Track layout


def make_spacetrack_fixture(path: Path) -> None:
    """Two made-up objects, six element sets each; object 90002 raises its orbit once."""
    recs = []
    for nid, name, n0, jump_at in ((90001, "SYNTH DEB A", 15.10, None),
                                   (90002, "SYNTH PAY B", 15.00, 3)):
        for k in range(6):
            n = n0 + 0.0002 * k  # slow decay: mean motion creeps up
            if jump_at is not None and k >= jump_at:
                n -= 0.02  # orbit raise: mean motion drops (about +3.7 km in a)
            recs.append({
                "CCSDS_OMM_VERS": "2.0", "COMMENT": "SYNTHETIC TEST FIXTURE",
                "CREATION_DATE": f"2024-01-0{k + 2}T00:00:00", "ORIGINATOR": "SYNTHETIC",
                "OBJECT_NAME": name, "OBJECT_ID": "2000-000" + ("A" if nid == 90001 else "B"),
                "CENTER_NAME": "EARTH", "REF_FRAME": "TEME", "TIME_SYSTEM": "UTC",
                "MEAN_ELEMENT_THEORY": "SGP4",
                "EPOCH": f"2024-01-0{k + 1}T{3 * k:02d}:00:00.000000",
                "MEAN_MOTION": f"{n:.8f}", "ECCENTRICITY": "0.00120000",
                "INCLINATION": "97.5000", "RA_OF_ASC_NODE": f"{10.0 + k:.4f}",
                "ARG_OF_PERICENTER": "90.0000", "MEAN_ANOMALY": f"{30.0 * k:.4f}",
                "EPHEMERIS_TYPE": "0", "CLASSIFICATION_TYPE": "U", "NORAD_CAT_ID": str(nid),
                "ELEMENT_SET_NO": str(999), "REV_AT_EPOCH": str(1000 + 15 * k),
                "BSTAR": "0.00012300", "MEAN_MOTION_DOT": "0.00001000",
                "MEAN_MOTION_DDOT": "0.0000000000000", "SEMIMAJOR_AXIS": "", "PERIOD": "",
                "APOAPSIS": "", "PERIAPSIS": "", "OBJECT_TYPE": "DEBRIS", "RCS_SIZE": "SMALL",
                "COUNTRY_CODE": "XX", "LAUNCH_DATE": "2000-01-01", "SITE": "XX",
                "DECAY_DATE": None, "FILE": "0", "GP_ID": str(nid * 10 + k),
                "TLE_LINE0": "0 " + name, "TLE_LINE1": "", "TLE_LINE2": "",
            })
    path.write_text(json.dumps(recs, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- CelesTrak-derived


def _pick_filter_cases(active: list[dict]) -> list[dict]:
    """One active-catalog object per filter case, chosen deterministically."""
    mu, re_km = 398600.8, 6378.135

    def alt(r: dict) -> tuple[float, float]:
        n = r["MEAN_MOTION"] * 2 * math.pi / 86400
        a = (mu / n**2) ** (1 / 3)
        return a * (1 - r["ECCENTRICITY"]) - re_km, a * (1 + r["ECCENTRICITY"]) - re_km

    by_id = sorted(active, key=lambda r: r["NORAD_CAT_ID"])
    cases = {
        "iss": lambda r: r["NORAD_CAT_ID"] == 25544,
        "starlink": lambda r: r["OBJECT_NAME"].startswith("STARLINK"),
        "geo": lambda r: abs(r["MEAN_MOTION"] - 1.0027) < 0.001,
        "heo": lambda r: alt(r)[0] < 2000 and alt(r)[1] > 20000,
        "low_perigee": lambda r: alt(r)[0] < 250 and alt(r)[1] < 2000,
        "neg_bstar": lambda r: r["BSTAR"] < 0 and alt(r)[1] < 2000,
        "big_bstar": lambda r: r["BSTAR"] > 0.01 and alt(r)[0] > 250 and alt(r)[1] < 2000,
    }
    out = []
    for _, pred in cases.items():
        out.append(next(r for r in by_id if pred(r)))
    return out


def make_celestrak_fixtures() -> None:
    active = json.loads((RAW / "gp_active.json").read_text())
    recs = []
    for nid in VALIDATION_IDS:
        recs += json.loads((RAW / f"gp_{nid}.json").read_text())
    recs += _pick_filter_cases(active)
    (FIX / "celestrak_gp_sample.json").write_text(json.dumps(recs, indent=1) + "\n")
    keys = list(recs[0].keys())
    with open(FIX / "celestrak_gp_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        w.writerows(recs)
    ids = {str(r["NORAD_CAT_ID"]) for r in recs}
    with open(RAW / "satcat.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    keep = [r for r in rows if r["NORAD_CAT_ID"] in ids]
    with open(FIX / "satcat_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(keep)
    print(f"celestrak fixtures: {len(recs)} element sets, {len(keep)} SATCAT rows")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic-only", action="store_true")
    args = p.parse_args()
    FIX.mkdir(parents=True, exist_ok=True)
    make_sw_fixture(FIX / "sw_quirks.txt")
    make_spacetrack_fixture(FIX / "spacetrack_gp_history_synthetic.json")
    if not args.synthetic_only:
        make_celestrak_fixtures()
    print("fixtures written to", FIX)


if __name__ == "__main__":
    main()
