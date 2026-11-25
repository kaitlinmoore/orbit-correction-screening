"""Compare the truth start rules: the current epoch start and the fitted start.

The current default, ``preserve_sgp4_mean_motion``, starts truth from SGP4's state at
the element-set epoch (scaled for the GM difference). That ties the truth orbit's
mean motion to where in its orbit the object is at the epoch. The labeled variant,
``fit_to_sgp4_before_epoch``, fits the epoch state to SGP4 positions over a span
*before* the epoch (``truth_prop.fit_start_to_sgp4``). This script measures both on
the same windows. It decides nothing; the numbers go to the start-rule decision.

Parts (``--part``; ``all`` runs them in order):

* ``phase``: the gravity-only phase table. The fixture element set of NORAD 25544
  with perigee 800 km, B* = 1e-7 /ER, eccentricity 0.001 / 0.02 / 0.07 and mean
  anomaly 0 / 90 / 180 / 270 deg at the epoch, plus four perigee-argument cases at
  e = 0.07; epoch 2024-06-15T00:00 UTC; 7 days on a 60 s grid; the fixture
  space-weather extract. These are the cases of an earlier review's table, rerun under
  both rules.
* ``strata``: the stratified comparison. Base objects are real element sets from
  CelesTrak's active catalogue (``raw/celestrak/gp_active.json``) that pass the
  orbital catalog rules only (ephemeris type 0, perigee >= 250 km, apogee <= 2,000 km,
  B* > 0; the payload-status and name rules would leave 2 sets, since the file is
  mostly operational payloads) and whose split role is train or dev (held-out objects
  are never used). Two per perigee band (``config/residuals.toml`` edges), drawn with a
  fixed seed. Each keeps its real perigee, inclination, node, argument of perigee and
  B*; its eccentricity is set to 0.001, 0.02 and 90% of the largest the 2,000 km
  apogee limit allows at that perigee (``e_high``), with the mean motion set to keep
  the perigee; mean anomaly 0 / 90 / 180 / 270 deg; epochs moved to 2019-06-15 (low
  solar activity), 2022-02-15 (moderate) and 2024-06-15 (high). Drag on. 7 days on a
  300 s grid. 2 x 5 x 3 x 4 x 3 = 360 windows, each under both rules. These are
  synthetic test cases built from real orbits, not those objects' real histories.
* ``span``: fit spans of 0.125, 0.25, 0.5, 1 and 2 days on a subset of the strata
  windows (every base object and eccentricity class, mean anomaly 90 and 270 deg,
  2024-06-15): fit RMS, T at 1 h, 1 d and 7 d, and the spread of 7-day T between the
  two phases.
* ``samples``: the eight truth sample objects of ``scripts/zonal_convergence.py`` (two
  per band at 300, 500, 800 and 1,200 km, their own elements at 2024-06-15) under the
  current rule and the fitted rule with 1-day and 0.25-day spans.
* ``cost``: the span-test windows under the current rule and the fitted rule at 1 and
  0.25 days, interleaved and repeated twice at one worker count: per-window wall time
  (fit included), evaluations and batch wall time.

Every truth run goes through ``truth_prop.propagate_batch`` (spawned workers, one JVM
each); this process never starts a JVM. SGP4 is ``cheap_prop`` (WGS-72, mode 'i', the
fast TEME -> GCRF rotation). Residuals are SGP4 minus truth in curvilinear RTN of the
SGP4 state (``residuals.rtn_from_states``, the training targets' definition); a window
whose along-track angle leaves that map's domain is recorded as ``rtn_domain``.
Cost is each window's ``TruthResult.wall_s`` (fit included) and evaluation count, and
each batch's wall time, both rules at the same worker count on the same machine.

Outputs go to ``<out-dir>`` (default ``<data-root>/derived/fitted_start``, gitignored):
``<part>.json`` (per-window records and summaries), ``<part>.md`` (the tables) and, for
``strata``, ``strata_rtn.npz`` (the residual series, for the practice notebook).

Usage (needs the datagen group):
    uv run python scripts/fitted_start_comparison.py --data-root /path/to/data \\
        [--part all|phase|strata|span|samples|cost] [--workers 16] [--out-dir PATH]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np

from orbit_correction import catalog, frames, residuals, splits
from orbit_correction import truth_prop as T
from orbit_correction.cheap_prop import propagate_sgp4_from_epoch, sgp4_epoch_states
from orbit_correction.spaceweather import parse_spaceweather

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
DAY = 86400.0
HORIZON_S = 7 * DAY
RE72, MU72 = 6378.135, 398600.8  # WGS-72: what SGP4's mean elements are expressed in
RE_WGS84 = 6378.137
APOGEE_MAX_KM = 2000.0
CURRENT, FITTED = "preserve_sgp4_mean_motion", T.FIT_START_RULE
RULES = (CURRENT, FITTED)
HORIZONS = {"1 h": 3600.0, "1 d": DAY, "3 d": 3 * DAY, "7 d": 7 * DAY}
PHASES_DEG = (0.0, 90.0, 180.0, 270.0)
SOLAR_EPOCHS = {"low": "2019-06-15T00:00:00", "moderate": "2022-02-15T00:00:00",
                "high": "2024-06-15T00:00:00"}
PHASE_EPOCH = "2024-06-15T00:00:00"
SEED = 20261124
PER_BAND = 2
SPANS_DAYS = (0.125, 0.25, 0.5, 1.0, 2.0)
SPAN_PHASES_DEG = (90.0, 270.0)
ORBITAL_RULES = ("sgp4_ephemeris_type", "leo_perigee", "leo_apogee", "bstar_positive",
                 "bstar_not_extreme", "not_maneuvering_constellation")


# --------------------------------------------------------------------------------------
# Element sets
# --------------------------------------------------------------------------------------


def e_high(hp_km: float) -> float:
    """90% of the largest eccentricity with this perigee and apogee <= 2,000 km."""
    e_max = (APOGEE_MAX_KM - hp_km) / (APOGEE_MAX_KM + hp_km + 2 * RE72)
    return 0.9 * e_max


def make_elset(record: dict, epoch_utc: str, *, hp_km=None, e=None, ma=None, argp=None,
               bstar=None):
    """One OMM record with its epoch and chosen elements replaced, as a catalog table.

    With ``hp_km``, the mean motion is set so the (WGS-72, mean-element) perigee
    altitude is ``hp_km`` at the record's (possibly replaced) eccentricity.
    """
    rec = dict(record)
    if e is not None:
        rec["ECCENTRICITY"] = float(e)
    if hp_km is not None:
        a = (RE72 + hp_km) / (1.0 - float(rec["ECCENTRICITY"]))
        rec["MEAN_MOTION"] = math.sqrt(MU72 / a**3) * DAY / (2 * math.pi)
    if ma is not None:
        rec["MEAN_ANOMALY"] = float(ma)
    if argp is not None:
        rec["ARG_OF_PERICENTER"] = float(argp)
    if bstar is not None:
        rec["BSTAR"] = float(bstar)
    return T._omm_record_elset(rec, epoch_utc)


def base_objects(data_root: Path) -> list[dict]:
    """Two train/dev element sets per perigee band from the active catalogue (see module)."""
    path = data_root / "raw" / "celestrak" / "gp_active.json"
    table = catalog.load_celestrak_omm(path)
    rules = [r for r in catalog.DEFAULT_RULES if r.name in ORBITAL_RULES]
    fcfg = catalog.load_filter_config(REPO / "config" / "catalog_filters.toml")
    kept = catalog.apply_filters(table, fcfg, rules).kept
    scfg = splits.load_split_config(REPO / "config" / "splits.toml")
    ids = kept.column("norad_id").to_numpy()
    roles = splits.object_roles(ids, scfg)
    rcfg = residuals.load_residual_config()
    records = {int(r["NORAD_CAT_ID"]): r for r in json.loads(path.read_text())}
    hp = kept.column("perigee_alt_km").to_numpy()
    rng = np.random.default_rng(SEED)
    out = []
    for lo, hi in zip(rcfg.altitude_edges_km[:-1], rcfg.altitude_edges_km[1:],
                      strict=True):
        # keep the perigee inside the band with room for the WGS-72/geodetic difference
        m = (hp >= lo + 5) & (hp < hi - 5)
        m &= np.array([roles[int(i)] in ("train", "dev") for i in ids])
        cand = np.sort(ids[m])
        for nid in rng.choice(cand, size=PER_BAND, replace=False):
            row = kept.filter(kept.column("norad_id").to_numpy() == nid).to_pylist()[0]
            out.append({
                "norad_id": int(nid),
                "name": row.get("object_name"),
                "role": roles[int(nid)],
                "band": residuals.altitude_band(row["perigee_alt_km"], rcfg),
                "hp_km": float(row["perigee_alt_km"]),
                "e_real": float(row["eccentricity"]),
                "bstar": float(row["bstar_per_earth_radii"]),
                "inc_deg": float(records[int(nid)]["INCLINATION"]),
                "record": records[int(nid)],
            })
    return out


# --------------------------------------------------------------------------------------
# Windows and truth runs
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class Window:
    label: str
    meta: dict
    elset: object
    dt_s: np.ndarray
    bstar: float = 0.0
    epoch_tt_s: float = 0.0
    sgp4: object = None
    init: object = None
    fit_samples: dict = dataclasses.field(default_factory=dict)  # span_s -> Trajectory
    results: dict = dataclasses.field(default_factory=dict)  # key -> TruthResult

    def prepare(self, spans_s):
        row = self.elset.to_pylist()[0]
        self.bstar = float(row["bstar_per_earth_radii"])
        self.epoch_tt_s = float(row["epoch_tt_s"])
        self.meta.update(e=float(row["eccentricity"]), hp_km=float(row["perigee_alt_km"]),
                         ha_km=float(row["apogee_alt_km"]), bstar=self.bstar)
        self.sgp4 = propagate_sgp4_from_epoch(self.elset, self.dt_s)[0]
        start = sgp4_epoch_states(self.elset)[0]
        self.init = start.trajectory
        for span in spans_s:
            res = T.fit_samples_from_elsets(self.elset, fit_cfg(span))[0]
            self.fit_samples[span] = res.trajectory if res.status.ok else None
            if not res.status.ok:
                self.meta[f"fit_samples_status_{span:g}"] = res.status.message


def fit_cfg(span_s: float) -> T.TruthConfig:
    return dataclasses.replace(T.load_truth_config(), start_rule=FITTED, fit_span_s=span_s)


def run_rule(windows, key, cfg, copies, workers, *, span_s=None):
    """Truth for every window under ``cfg``, one batch per space-weather copy.

    Windows whose epoch state or fit samples are missing get no job (recorded).
    Returns the batch wall times.
    """
    walls = []
    groups: dict[str, list[Window]] = {}
    for w in windows:
        groups.setdefault(w.meta["sw_copy"], []).append(w)
    for name, ws in groups.items():
        jobs, owners = [], []
        for w in ws:
            if w.init is None or (span_s is not None and w.fit_samples.get(span_s) is None):
                w.results[key] = None
                continue
            fs = w.fit_samples[span_s] if span_s is not None else None
            jobs.append(T.TruthJob(w.init, w.bstar, w.epoch_tt_s + w.dt_s, fit_samples=fs))
            owners.append(w)
        t0 = time.perf_counter()
        res = T.propagate_batch(jobs, cfg, copies[name], n_workers=workers) if jobs else []
        walls.append({"copy": name, "n_jobs": len(jobs), "wall_s": time.perf_counter() - t0})
        for w, r in zip(owners, res, strict=True):
            w.results[key] = r
        print(f"  {key} / {name}: {len(jobs)} jobs, {walls[-1]['wall_s']:.0f} s", flush=True)
    return walls


def write_copies(sw, out_dir: Path, epochs_utc, before_days: float):
    """One space-weather copy per epoch: from the earliest fit sample to the window end."""
    copies = {}
    for name, epoch in epochs_utc.items():
        t0 = frames.utc_to_tt_seconds(epoch)
        first = t0 - before_days * DAY
        path = out_dir / f"sw_orekit_{epoch[:10]}_minus{before_days:g}d_7d.txt"
        copies[name] = T.write_space_weather_copy(sw, path, first, t0 + HORIZON_S)
    return copies


# --------------------------------------------------------------------------------------
# Residuals and records
# --------------------------------------------------------------------------------------


def rtn_series(w: Window, res) -> tuple[np.ndarray | None, str]:
    """SGP4 minus truth, curvilinear RTN (km), on the window's grid, or (None, status)."""
    if res is None:
        return None, "no_job"
    if res.status != "ok":
        return None, f"truth_{res.status}"
    if not w.sgp4.status.ok:
        return None, "sgp4_" + ("floor" if w.sgp4.status.below_floor else "error")
    sg, tr = w.sgp4.trajectory, res.trajectory
    try:
        return residuals.rtn_from_states(sg.r_gcrf_km, sg.v_gcrf_km_s, tr.r_gcrf_km), "ok"
    except residuals.RTNDomainError:
        return None, "rtn_domain"


def window_record(w: Window, key: str) -> dict:
    res = w.results.get(key)
    rtn, status = rtn_series(w, res)
    rec = {"label": w.label, "rule_key": key, **{k: v for k, v in w.meta.items()},
           "status": status}
    if res is not None:
        rec.update(truth_status=res.status, wall_s=res.wall_s, n_evaluations=res.n_evaluations,
                   message=res.message)
        f = res.start_fit
        if f is not None:
            rec["fit"] = fit_record(w, f)
            rec["n_evaluations_total"] = (res.n_evaluations or 0) + f.n_evaluations
        else:
            rec["n_evaluations_total"] = res.n_evaluations
    if rtn is not None:
        idx = {h: int(np.argmin(np.abs(w.dt_s - s))) for h, s in HORIZONS.items()}
        rec["rtn_km"] = {h: [float(x) for x in rtn[j]] for h, j in idx.items()}
        sg = w.sgp4.trajectory
        tan_g = np.einsum("ij,ij->i", sg.r_gcrf_km, sg.v_gcrf_km_s) / np.linalg.norm(
            np.cross(sg.r_gcrf_km, sg.v_gcrf_km_s), axis=1)
        leak = rtn[:, 1] * tan_g
        rec["radial"] = {
            "max_abs_R_km": float(np.abs(rtn[:, 0]).max()),
            "rms_R_km": float(np.sqrt(np.mean(rtn[:, 0] ** 2))),
            "rms_R_minus_T_tan_gamma_km": float(np.sqrt(np.mean((rtn[:, 0] - leak) ** 2))),
            "max_abs_T_km": float(np.abs(rtn[:, 1]).max()),
        }
    return rec


def fit_record(w: Window, f: T.StartFit) -> dict:
    """The fit's record, with its offset from the SGP4 epoch state in RTN of that state."""
    r, v = w.init.r_gcrf_km[0], w.init.v_gcrf_km_s[0]
    u = r / np.linalg.norm(r)
    n = np.cross(r, v)
    n /= np.linalg.norm(n)
    t = np.cross(n, u)
    dr, dv = np.array(f.offset_r_m), np.array(f.offset_v_m_s)
    return {
        "status": f.status, "iterations": f.iterations, "n_samples": f.n_samples,
        "span_s": f.last_sample_tt_s - f.first_sample_tt_s,
        "rms_m": f.rms_m, "max_m": f.max_m, "rms_initial_m": f.rms_initial_m,
        "offset_r_rtn_m": [float(dr @ u), float(dr @ t), float(dr @ n)],
        "offset_v_rtn_m_s": [float(dv @ u), float(dv @ t), float(dv @ n)],
        "offset_r_norm_m": float(np.linalg.norm(dr)),
        "offset_v_norm_m_s": float(np.linalg.norm(dv)),
        "last_correction_r_m": f.last_correction_r_m,
        "last_correction_v_m_s": f.last_correction_v_m_s,
        "n_evaluations": f.n_evaluations, "wall_s": f.wall_s, "message": f.message,
    }


def pct(x, q):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size else float("nan")


def stats_by(records, keys, rule_keys):
    """Median and 95th percentile of |R|, |T|, |N| per horizon, by stratum and rule."""
    out = []
    groups: dict[tuple, list] = {}
    for r in records:
        if r["rule_key"] not in rule_keys:
            continue
        groups.setdefault(tuple(r[k] for k in keys) + (r["rule_key"],), []).append(r)
    for g, rs in sorted(groups.items()):
        ok = [r for r in rs if r["status"] == "ok"]
        row = {**dict(zip(keys, g[:-1], strict=True)), "rule": g[-1], "n": len(rs),
               "n_ok": len(ok), "not_ok": _count([r["status"] for r in rs if r["status"] != "ok"])}
        for h in HORIZONS:
            for c, comp in enumerate("RTN"):
                vals = [abs(r["rtn_km"][h][c]) for r in ok]
                row[f"{comp}_{h}_med"] = pct(vals, 50)
                row[f"{comp}_{h}_p95"] = pct(vals, 95)
        row["rms_R_med"] = pct([r["radial"]["rms_R_km"] for r in ok], 50)
        row["rms_R_minus_leak_med"] = pct(
            [r["radial"]["rms_R_minus_T_tan_gamma_km"] for r in ok], 50)
        out.append(row)
    return out


def _count(items):
    d: dict = {}
    for i in items:
        d[i] = d.get(i, 0) + 1
    return d


def phase_spread(records, group_keys, rule_keys, n_phases=4):
    """Spread (max - min) of 7-day T across the phases of one orbit and epoch.

    NaN for a group missing a phase (a window that wasn't ``ok``)."""
    groups: dict[tuple, list] = {}
    for r in records:
        if r["rule_key"] in rule_keys and r["status"] == "ok":
            groups.setdefault(tuple(r[k] for k in group_keys) + (r["rule_key"],), []).append(
                r["rtn_km"]["7 d"][1])
    return [{**dict(zip(group_keys, g[:-1], strict=True)), "rule": g[-1], "n_phases": len(v),
             "T7d_spread_km": float(max(v) - min(v)) if len(v) == n_phases else float("nan"),
             "T7d_mean_km": float(np.mean(v)),
             "T7d_values_km": [float(x) for x in v]} for g, v in sorted(groups.items())]


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    ax = abs(x)
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 1:
        return f"{x:.1f}" if ax >= 10 else f"{x:.2f}"
    return f"{x:.{nd}f}"


# --------------------------------------------------------------------------------------
# Parts
# --------------------------------------------------------------------------------------


def part_phase(args, out_dir: Path) -> dict:
    rec25544 = next(r for r in json.loads((FIXTURES / "celestrak_gp_sample.json").read_text())
                    if int(r["NORAD_CAT_ID"]) == 25544)
    cases = [(f"grav_e{e:g}_ma{ma:g}", dict(e=e, ma=ma)) for e in (0.001, 0.02, 0.07)
             for ma in PHASES_DEG]
    cases += [(f"grav_e0.07_argp{w:g}_ma{ma:g}", dict(e=0.07, ma=ma, argp=w))
              for w in (0.0, 90.0) for ma in (0.0, 180.0)]
    dt = 60.0 * np.arange(7 * 1440 + 1)
    windows = []
    for label, ch in cases:
        el = make_elset(rec25544, PHASE_EPOCH, hp_km=800.0, bstar=1e-7, **ch)
        w = Window(label, {"case": label, "sw_copy": "phase", **ch}, el, dt)
        w.prepare([DAY])
        windows.append(w)
    t0 = frames.utc_to_tt_seconds(PHASE_EPOCH)
    copies = {"phase": T.write_space_weather_copy(
        FIXTURES / "sw_celestrak_2024_extract.txt", out_dir / "sw_orekit_phase_2024-06-14_7d.txt",
        t0 - DAY, t0 + HORIZON_S)}
    base = T.load_truth_config()
    walls = {CURRENT: run_rule(windows, CURRENT, base, copies, args.workers),
             FITTED: run_rule(windows, FITTED, fit_cfg(DAY), copies, args.workers, span_s=DAY)}
    records = [window_record(w, k) for w in windows for k in RULES]
    lines = ["| Case | e | MA | argp | T 1 d current | T 7 d current | T 1 d fitted | "
             "T 7 d fitted | fit RMS (m) | passes |", "|---|---|---|---|---|---|---|---|---|---|"]
    for w in windows:
        a = next(r for r in records if r["label"] == w.label and r["rule_key"] == CURRENT)
        b = next(r for r in records if r["label"] == w.label and r["rule_key"] == FITTED)
        argp = w.meta.get("argp", float(rec25544["ARG_OF_PERICENTER"]))
        lines.append(
            f"| {w.label} | {w.meta['e']:g} | {w.meta['ma']:g} | {argp:g} | "
            f"{fmt(a['rtn_km']['1 d'][1])} | {fmt(a['rtn_km']['7 d'][1])} | "
            f"{fmt(b['rtn_km']['1 d'][1])} | {fmt(b['rtn_km']['7 d'][1])} | "
            f"{fmt(b['fit']['rms_m'])} | {b['fit']['iterations']} |")
    md = "\n".join(["## Phase table (gravity only: B* 1e-7, 800 km, 2024-06-15)",
                    "SGP4 minus truth, along-track T (km).", "", *lines])
    return {"records": records, "batch_walls": walls, "markdown": md}


def part_strata(args, out_dir: Path) -> dict:
    bases = base_objects(args.data_root)
    dt = 300.0 * np.arange(int(HORIZON_S / 300.0) + 1)
    rcfg = residuals.load_residual_config()
    sw = parse_spaceweather(args.data_root / "raw" / "celestrak" / "SW-All.txt")
    windows = []
    for b in bases:
        for e_class, e in (("circular", 0.001), ("moderate", 0.02), ("high", e_high(b["hp_km"]))):
            for sname, epoch in SOLAR_EPOCHS.items():
                asof = sw.as_of(frames.utc_to_tt_seconds(epoch))
                for ma in PHASES_DEG:
                    el = make_elset(b["record"], epoch, hp_km=b["hp_km"], e=e, ma=ma)
                    meta = {"norad_id": b["norad_id"], "band": b["band"], "e_class": e_class,
                            "solar": sname, "epoch_utc": epoch, "ma": ma,
                            "f107_81_sfu": asof.f107_obs_trailing81_sfu,
                            "solar_bin": residuals.solar_activity_bin(
                                asof.f107_obs_trailing81_sfu, rcfg),
                            "argp": float(b["record"]["ARG_OF_PERICENTER"]),
                            "inc_deg": b["inc_deg"], "sw_copy": sname}
                    w = Window(f"{b['norad_id']}_{e_class}_{sname}_ma{ma:g}", meta, el, dt)
                    w.prepare([DAY])
                    windows.append(w)
    print(f"strata: {len(windows)} windows from {len(bases)} base objects", flush=True)
    copies = write_copies(sw, out_dir, SOLAR_EPOCHS, before_days=max(SPANS_DAYS))
    base_cfg = T.load_truth_config()
    walls = {CURRENT: run_rule(windows, CURRENT, base_cfg, copies, args.workers),
             FITTED: run_rule(windows, FITTED, fit_cfg(DAY), copies, args.workers, span_s=DAY)}
    records = [window_record(w, k) for w in windows for k in RULES]
    # residual series for the notebook
    arrays = {}
    for w in windows:
        for k in RULES:
            rtn, st = rtn_series(w, w.results.get(k))
            if rtn is not None:
                arrays[f"{w.label}|{k}"] = rtn.astype(np.float32)
    np.savez_compressed(out_dir / "strata_rtn.npz", dt_s=dt, **arrays)
    summaries = {
        "by_band": stats_by(records, ["band"], RULES),
        "by_e_class": stats_by(records, ["e_class"], RULES),
        "by_solar_bin": stats_by(records, ["solar_bin"], RULES),
        "by_band_e_class": stats_by(records, ["band", "e_class"], RULES),
        "by_band_solar_bin": stats_by(records, ["band", "solar_bin"], RULES),
        "phase_spread": phase_spread(records, ["norad_id", "band", "e_class", "solar"], RULES),
    }
    md = strata_markdown(records, summaries, walls)
    return {"bases": [{k: v for k, v in b.items() if k != "record"} for b in bases],
            "records": records, "summaries": summaries, "batch_walls": walls, "markdown": md}


def strata_markdown(records, s, walls) -> str:
    out = ["## Stratified comparison", ""]
    st = _count([(r["rule_key"], r["status"]) for r in records])
    out.append("Window outcomes: "
               + "; ".join(f"{short(k[0])} {k[1]}: {v}" for k, v in sorted(st.items())))
    fits = [r["fit"] for r in records if r.get("fit")]
    out.append("Fit statuses: " + str(_count([f["status"] for f in fits])))
    for title, key, cols in (("By altitude band", "by_band", ["band"]),
                             ("By eccentricity class", "by_e_class", ["e_class"]),
                             ("By solar-activity bin", "by_solar_bin", ["solar_bin"])):
        out += ["", f"### {title}: median / p95 of |.| in km", "",
                "| " + " | ".join(cols) + " | rule | n ok | " +
                " | ".join(f"{c} {h}" for h in HORIZONS for c in "RTN") + " |",
                "|" + "---|" * (len(cols) + 2 + 3 * len(HORIZONS))]
        for row in s[key]:
            cells = [f"{fmt(row[f'{c}_{h}_med'])} / {fmt(row[f'{c}_{h}_p95'])}"
                     for h in HORIZONS for c in "RTN"]
            out.append("| " + " | ".join(str(row[c]) for c in cols) +
                       f" | {short(row['rule'])} | {row['n_ok']}/{row['n']} | " +
                       " | ".join(cells) + " |")
    out += ["", "### 7-day T by band and eccentricity: median / p95 of |T| (km), and the phase "
            "spread (median over orbit-epoch groups of max - min of 7-day T across the four "
            "phases)", "",
            "| band | e class | current T 7 d | fitted T 7 d | current spread | fitted spread | "
            "spread ratio |", "|---|---|---|---|---|---|---|"]
    sp = {}
    for r in s["phase_spread"]:
        sp.setdefault((r["band"], r["e_class"], r["rule"]), []).append(r["T7d_spread_km"])
    be = {(r["band"], r["e_class"], r["rule"]): r for r in s["by_band_e_class"]}
    for band, ecl in sorted({(r["band"], r["e_class"]) for r in s["by_band_e_class"]}):
        a, b = be.get((band, ecl, CURRENT)), be.get((band, ecl, FITTED))
        sa, sb = pct(sp.get((band, ecl, CURRENT), []), 50), pct(sp.get((band, ecl, FITTED), []), 50)
        out.append(f"| {band} | {ecl} | {fmt(a['T_7 d_med'])} / {fmt(a['T_7 d_p95'])} | "
                   f"{fmt(b['T_7 d_med'])} / {fmt(b['T_7 d_p95'])} | {fmt(sa)} | {fmt(sb)} | "
                   f"{fmt(sb / sa if sa else float('nan'))} |")
    out += ["", "### Radial follows along-track (median over windows, km)", "",
            "| e class | rule | max abs R | rms R | rms (R - T tan gamma) |",
            "|---|---|---|---|---|"]
    for ecl in ("circular", "moderate", "high"):
        for k in RULES:
            ok = [r for r in records if r["e_class"] == ecl and r["rule_key"] == k
                  and r["status"] == "ok"]
            out.append(f"| {ecl} | {short(k)} | "
                       f"{fmt(pct([r['radial']['max_abs_R_km'] for r in ok], 50))} | "
                       f"{fmt(pct([r['radial']['rms_R_km'] for r in ok], 50))} | "
                       f"{fmt(pct([r['radial']['rms_R_minus_T_tan_gamma_km'] for r in ok], 50))} |")
    out += ["", "### Fit (fitted rule, 1-day span)", "",
            "| band | e class | n | passes (max) | RMS initial m (median) | RMS fitted m "
            "(median / max) | offset m (median) | offset mm/s (median) |",
            "|---|---|---|---|---|---|---|---|"]
    groups: dict = {}
    for r in records:
        if r.get("fit"):
            groups.setdefault((r["band"], r["e_class"]), []).append(r["fit"])
    for (band, ecl), fs in sorted(groups.items()):
        out.append(f"| {band} | {ecl} | {len(fs)} | {max(f['iterations'] for f in fs)} | "
                   f"{fmt(pct([f['rms_initial_m'] for f in fs], 50))} | "
                   f"{fmt(pct([f['rms_m'] for f in fs], 50))} / "
                   f"{fmt(max(f['rms_m'] for f in fs))} | "
                   f"{fmt(pct([f['offset_r_norm_m'] for f in fs], 50))} | "
                   f"{fmt(pct([f['offset_v_norm_m_s'] * 1e3 for f in fs], 50))} |")
    out += ["", "### Cost (same machine, same worker count)", ""]
    out += cost_lines(records, walls)
    return "\n".join(out)


def cost_lines(records, walls) -> list[str]:
    lines = []
    w = {k: [r["wall_s"] for r in records if r["rule_key"] == k and r.get("wall_s") is not None
             and r.get("truth_status") == "ok"] for k in RULES}
    ev = {k: [r["n_evaluations_total"] for r in records if r["rule_key"] == k
              and r.get("truth_status") == "ok"] for k in RULES}
    bw = {k: sum(x["wall_s"] for x in walls[k]) for k in RULES}
    nj = {k: sum(x["n_jobs"] for x in walls[k]) for k in RULES}
    for k in RULES:
        lines.append(f"- {short(k)}: per-window wall median {fmt(pct(w[k], 50))} s (p95 "
                     f"{fmt(pct(w[k], 95))}), evaluations median {pct(ev[k], 50):.0f}, batch wall "
                     f"{bw[k]:.0f} s for {nj[k]} jobs")
    lines.append(f"- ratio fitted / current: per-window wall (median of medians) "
                 f"{pct(w[FITTED], 50) / pct(w[CURRENT], 50):.2f}, evaluations "
                 f"{pct(ev[FITTED], 50) / pct(ev[CURRENT], 50):.2f}, batch wall "
                 f"{bw[FITTED] / bw[CURRENT]:.2f}")
    return lines


def short(rule: str) -> str:
    return {CURRENT: "current", FITTED: "fitted"}.get(rule, rule)


def part_span(args, out_dir: Path) -> dict:
    bases = base_objects(args.data_root)
    dt = 300.0 * np.arange(int(HORIZON_S / 300.0) + 1)
    sw = parse_spaceweather(args.data_root / "raw" / "celestrak" / "SW-All.txt")
    epoch = SOLAR_EPOCHS["high"]
    spans = [d * DAY for d in SPANS_DAYS]
    windows = []
    for b in bases:
        for e_class, e in (("circular", 0.001), ("moderate", 0.02), ("high", e_high(b["hp_km"]))):
            for ma in SPAN_PHASES_DEG:
                el = make_elset(b["record"], epoch, hp_km=b["hp_km"], e=e, ma=ma)
                meta = {"norad_id": b["norad_id"], "band": b["band"], "e_class": e_class,
                        "solar": "high", "epoch_utc": epoch, "ma": ma, "sw_copy": "high"}
                w = Window(f"{b['norad_id']}_{e_class}_high_ma{ma:g}", meta, el, dt)
                w.prepare(spans)
                windows.append(w)
    print(f"span: {len(windows)} windows x {len(spans)} spans", flush=True)
    copies = write_copies(sw, out_dir, {"high": epoch}, before_days=max(SPANS_DAYS))
    walls = {CURRENT: run_rule(windows, CURRENT, T.load_truth_config(), copies, args.workers)}
    keys = [CURRENT]
    for s in spans:
        k = f"span_{s / DAY:g}d"
        keys.append(k)
        walls[k] = run_rule(windows, k, fit_cfg(s), copies, args.workers, span_s=s)
    records = [window_record(w, k) for w in windows for k in keys]
    spread = phase_spread(records, ["norad_id", "band", "e_class"], keys,
                          n_phases=len(SPAN_PHASES_DEG))
    out = ["## Span sensitivity (2024-06-15, drag on, mean anomaly 90 and 270 deg)", "",
           f"{len(windows)} windows: every strata base object and eccentricity class. Spans "
           f"{', '.join(f'{d:g}' for d in SPANS_DAYS)} days; 0.125 and 0.25 are additions to "
           "the requested 0.5 / 1 / 2.", "",
           "### By band: median |T| (km) at 1 h / 1 d / 7 d, median fit RMS (m), and the median "
           "phase spread of 7-day T (km, |T(90) - T(270)|)", "",
           "| band | rule | n ok | T 1 h | T 1 d | T 7 d | fit RMS | phase spread | passes max |",
           "|---|---|---|---|---|---|---|---|---|"]
    for band in sorted({w.meta["band"] for w in windows}):
        for k in keys:
            ok = [r for r in records if r["band"] == band and r["rule_key"] == k
                  and r["status"] == "ok"]
            n = sum(1 for r in records if r["band"] == band and r["rule_key"] == k)
            fits = [r["fit"] for r in ok if r.get("fit")]
            sp = [r["T7d_spread_km"] for r in spread if r["band"] == band and r["rule"] == k]
            out.append(
                f"| {band} | {short(k)} | {len(ok)}/{n} | "
                + " | ".join(fmt(pct([abs(r["rtn_km"][h][1]) for r in ok], 50))
                             for h in ("1 h", "1 d", "7 d"))
                + f" | {fmt(pct([f['rms_m'] for f in fits], 50)) if fits else '-'} | "
                f"{fmt(pct(sp, 50))} | {max(f['iterations'] for f in fits) if fits else '-'} |")
    out += ["", "### All windows: median and p95 of |T| (km), fit RMS, cost", "",
            "| rule | T 1 h med / p95 | T 1 d med / p95 | T 7 d med / p95 | fit RMS med (m) | "
            "phase spread med (km) | wall med (s) |", "|---|---|---|---|---|---|---|"]
    for k in keys:
        ok = [r for r in records if r["rule_key"] == k and r["status"] == "ok"]
        fits = [r["fit"] for r in ok if r.get("fit")]
        sp = [r["T7d_spread_km"] for r in spread if r["rule"] == k]
        cells = [f"{fmt(pct([abs(r['rtn_km'][h][1]) for r in ok], 50))} / "
                 f"{fmt(pct([abs(r['rtn_km'][h][1]) for r in ok], 95))}"
                 for h in ("1 h", "1 d", "7 d")]
        out.append(f"| {short(k)} | " + " | ".join(cells)
                   + f" | {fmt(pct([f['rms_m'] for f in fits], 50)) if fits else '-'} | "
                   f"{fmt(pct(sp, 50))} | {fmt(pct([r['wall_s'] for r in ok], 50))} |")
    statuses = _count([(r["rule_key"], r["status"]) for r in records])
    out += ["", "Outcomes: " + "; ".join(f"{short(a)} {b}: {v}"
                                         for (a, b), v in sorted(statuses.items()))]
    return {"records": records, "phase_spread": spread, "batch_walls": walls,
            "markdown": "\n".join(out)}


def part_samples(args, out_dir: Path) -> dict:
    """The eight truth sample objects (scripts/zonal_convergence.py) under both rules."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import zonal_convergence as Z

    records = {int(r["NORAD_CAT_ID"]): r for r in json.loads(
        (args.data_root / "raw" / "celestrak" / "gp_active.json").read_text())}
    dt = 600.0 * np.arange(int(HORIZON_S / 600.0) + 1)
    spans = [DAY, 0.25 * DAY]
    windows = []
    for band, ids in Z.SAMPLES.items():
        for nid in ids:
            el = T._omm_record_elset(records[nid], Z.EPOCH_UTC)
            w = Window(f"{band}_{nid}", {"band_km": band, "norad_id": nid, "sw_copy": "s"},
                       el, dt)
            w.prepare(spans)
            windows.append(w)
    sw = parse_spaceweather(args.data_root / "raw" / "celestrak" / "SW-All.txt")
    copies = write_copies(sw, out_dir, {"s": Z.EPOCH_UTC}, before_days=1.0)
    walls = {CURRENT: run_rule(windows, CURRENT, T.load_truth_config(), copies, args.workers)}
    keys = [CURRENT]
    for sp in spans:
        k = f"span_{sp / DAY:g}d"
        keys.append(k)
        walls[k] = run_rule(windows, k, fit_cfg(sp), copies, args.workers, span_s=sp)
    records_out = [window_record(w, k) for w in windows for k in keys]
    lines = ["## The eight truth sample objects (2024-06-15, drag on)", "",
             "SGP4 minus truth along-track T (km) at 1 h / 1 d / 7 d.", "",
             "| band | NORAD | B* | " + " | ".join(short(k) for k in keys) + " |",
             "|---|---|---|" + "---|" * len(keys)]
    for w in windows:
        rs = {r["rule_key"]: r for r in records_out if r["label"] == w.label}
        cells = [" / ".join(fmt(rs[k]["rtn_km"][h][1]) for h in ("1 h", "1 d", "7 d"))
                 if rs[k]["status"] == "ok" else rs[k]["status"] for k in keys]
        lines.append(f"| {w.meta['band_km']} | {w.meta['norad_id']} | {w.bstar:.2e} | "
                     + " | ".join(cells) + " |")
    return {"records": records_out, "batch_walls": walls, "markdown": "\n".join(lines)}


def part_cost(args, out_dir: Path) -> dict:
    """Truth compute per window under each rule, interleaved, at one worker count.

    The span-test windows (60), run as current, fitted 1 d, fitted 0.25 d, then the same
    three again, so a change in machine load during the run hits every rule alike.
    """
    bases = base_objects(args.data_root)
    dt = 300.0 * np.arange(int(HORIZON_S / 300.0) + 1)
    sw = parse_spaceweather(args.data_root / "raw" / "celestrak" / "SW-All.txt")
    epoch = SOLAR_EPOCHS["high"]
    spans = [DAY, 0.25 * DAY]
    windows = []
    for b in bases:
        for e_class, e in (("circular", 0.001), ("moderate", 0.02), ("high", e_high(b["hp_km"]))):
            for ma in SPAN_PHASES_DEG:
                el = make_elset(b["record"], epoch, hp_km=b["hp_km"], e=e, ma=ma)
                meta = {"norad_id": b["norad_id"], "band": b["band"], "e_class": e_class,
                        "ma": ma, "sw_copy": "high"}
                w = Window(f"{b['norad_id']}_{e_class}_ma{ma:g}", meta, el, dt)
                w.prepare(spans)
                windows.append(w)
    copies = write_copies(sw, out_dir, {"high": epoch}, before_days=1.0)
    runs = []
    for rep in (1, 2):
        for key, cfg, span in ((CURRENT, T.load_truth_config(), None),
                               ("fit_1d", fit_cfg(DAY), DAY),
                               ("fit_0.25d", fit_cfg(0.25 * DAY), 0.25 * DAY)):
            k = f"{key}#{rep}"
            walls = run_rule(windows, k, cfg, copies, args.workers, span_s=span)
            res = [w.results[k] for w in windows]
            runs.append({
                "rule": key, "repeat": rep, "batch_wall_s": sum(x["wall_s"] for x in walls),
                "n_jobs": len(res), "statuses": _count([r.status for r in res]),
                "window_wall_s": [r.wall_s for r in res],
                "fit_wall_s": [r.start_fit.wall_s if r.start_fit else 0.0 for r in res],
                "evaluations": [(r.n_evaluations or 0)
                                + (r.start_fit.n_evaluations if r.start_fit else 0) for r in res],
            })
    lines = [f"## Cost: {len(windows)} windows per run, {args.workers} workers, rules "
             "interleaved twice", "",
             "| rule | repeat | batch wall (s) | per-window wall mean / median (s) | fit part "
             "mean (s) | evaluations mean |", "|---|---|---|---|---|---|"]
    for r in runs:
        lines.append(f"| {short(r['rule'])} | {r['repeat']} | {r['batch_wall_s']:.1f} | "
                     f"{np.mean(r['window_wall_s']):.2f} / {np.median(r['window_wall_s']):.2f} | "
                     f"{np.mean(r['fit_wall_s']):.2f} | {np.mean(r['evaluations']):.0f} |")
    mean = {k: np.mean([np.mean(r["window_wall_s"]) for r in runs if r["rule"] == k])
            for k in (CURRENT, "fit_1d", "fit_0.25d")}
    batch = {k: np.mean([r["batch_wall_s"] for r in runs if r["rule"] == k])
             for k in (CURRENT, "fit_1d", "fit_0.25d")}
    lines += ["", "| rule | ratio to current, per-window mean | ratio, batch wall |",
              "|---|---|---|"]
    for k in ("fit_1d", "fit_0.25d"):
        lines.append(f"| {k} | {mean[k] / mean[CURRENT]:.2f} | {batch[k] / batch[CURRENT]:.2f} |")
    return {"runs": runs, "mean_window_wall_s": mean, "mean_batch_wall_s": batch,
            "markdown": "\n".join(lines)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, required=True,
                    help="the data/ folder holding raw/celestrak (main checkout)")
    ap.add_argument("--part", choices=("all", "phase", "strata", "span", "samples", "cost"),
                    default="all")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out_dir = args.out_dir or args.data_root / "derived" / "fitted_start"
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = ("phase", "strata", "span", "samples", "cost") if args.part == "all" else (
        args.part,)
    funcs = {"phase": part_phase, "strata": part_strata, "span": part_span,
             "samples": part_samples, "cost": part_cost}
    for p in parts:
        print(f"== {p}", flush=True)
        t0 = time.perf_counter()
        out = funcs[p](args, out_dir)
        out["meta"] = {
            "part": p, "workers": args.workers, "wall_s": time.perf_counter() - t0,
            "truth_config": dataclasses.asdict(T.load_truth_config()),
            "provenance": T.orekit_provenance(), "seed": SEED,
            "solar_epochs": SOLAR_EPOCHS, "phases_deg": PHASES_DEG,
        }
        (out_dir / f"{p}.json").write_text(json.dumps(out, indent=1, default=str))
        (out_dir / f"{p}.md").write_text(out["markdown"] + "\n")
        print(out["markdown"], flush=True)


if __name__ == "__main__":
    main()
