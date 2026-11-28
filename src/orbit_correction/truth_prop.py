"""Truth propagator: Orekit numerical integration with zonal gravity and NRLMSISE-00 drag.

Why this module exists
----------------------
This study is model-world: "truth" is not tracked positions but the output of a
high-fidelity numerical propagator, configured once in ``config/truth.toml``
(decision 1, 2026-11-23). The correction model learns SGP4 minus this
propagator, so everything this module does defines the targets. Its job is to
turn a project :class:`~orbit_correction.types.Trajectory` (one GCRF state at an
element-set epoch) into a ``Trajectory(source="truth")`` on a requested time
grid, with no hidden conversions and with every physical choice taken from the
config file.

What is library behaviour and what is ours
------------------------------------------
* **Orekit's** (13.1.9, through ``orekit-jpype``): the numerical propagator and
  DOP853 integrator (Hipparchus), the Holmes-Featherstone gravity model reading
  EIGEN-6S from the pinned ``orekit-data`` bundle, NRLMSISE-00, the drag force,
  Earth orientation (IERS 2010, bundle EOP), the Sun ephemeris (DE-440), the
  ephemeris generator and the altitude event detector.
* **Ours:** the configuration and its checks; the B* -> area-to-mass relation;
  the start rule (below); the GCRF/TT/km conventions at the boundary; the rule
  that Orekit reads only the space-weather copy written by
  :func:`write_space_weather_copy` (never CelesTrak's file, never the bundle's
  CSSI file), and the guards that refuse a window outside the range that copy was
  checked for or inside the 41-day data-end margin; the re-entry status and the
  initial-altitude guard; the JVM rules and the worker pool.

The JVM rules
-------------
Orekit runs in a Java virtual machine started through JPype. A JVM cannot be
restarted inside a process, and forking a process that holds one is unsafe. So:

1. Nothing imports ``jpype`` or Orekit at module import time. Every Java import
   happens inside a function, after :func:`start_jvm`. Importing this module (or
   any other project module) never starts a JVM; ``tests/test_truth_prop.py``
   checks that in a fresh interpreter.
2. Batches run in worker processes started with ``spawn`` (never ``fork``), one
   JVM per worker (:func:`propagate_batch`). The parent process never starts a
   JVM, so the training process, which imports PyTorch, never shares a process
   with Java. :func:`start_jvm` refuses to run in a process that has imported
   ``torch``.
3. Each JVM is capped with ``-XX:ActiveProcessorCount`` (config, default 2) so
   parallel workers don't each size their thread pools for the whole machine.

Initial state and the start rule
--------------------------------
The SGP4 state at the element-set epoch, rotated TEME -> GCRF, is the deliberate,
logged conversion from SGP4 mean elements (CLAUDE.md). How truth starts from it is
the start rule, ``[truth.start] rule``. There is no default that skips the choice:
the config file must name a rule, and :class:`TruthConfig`'s own default is the
configured one.

**Default since 2026-11-25: ``fit_to_sgp4_before_epoch``, 0.25-day span**
(decided by Kaitlin from the fitted-start comparison; see the changelog). Starting
from SGP4's state at one instant ties the truth orbit's mean motion to where in
its orbit the object is at the epoch: SGP4's short-periodic terms make that
state's osculating period differ from the period SGP4 itself keeps, by an amount
that depends on the orbital phase. Truth then drifts along-track against SGP4 by
a phase-dependent amount (about +-13 km over 7 days at 800 km near-circular,
hundreds of km at e = 0.07, with no drag) that a model would learn from its
inputs. Element sets fitted to tracking over a span don't carry that artifact.

The fitted rule fits the truth state at the epoch (position and velocity, six
numbers) by least squares to SGP4's GCRF positions at times in
``[epoch - span, epoch]`` (:func:`fit_start_to_sgp4`). Each pass propagates
backwards from the epoch with the full truth force model (the same
:func:`build_propagator`: zonals, drag with the B*-derived A/m, the run's
space-weather copy) and Orekit's state transition matrix, and takes one
Gauss-Newton step. **Only SGP4 states at or before the epoch enter the fit**;
:func:`fit_start_to_sgp4` refuses any sample after it, so nothing from the
prediction window can shape the start. The GM rescale is not needed: the fit
finds whatever state matches SGP4 under the truth GM. A fit that does not
converge returns ``TruthResult(status="fit_failed")``, never a fallback start.
Without fit samples, :func:`propagate_truth` and :func:`propagate_batch` refuse
the job rather than start from the epoch state. The fitted state sits tens of
metres from the SGP4 epoch state, so residuals at t0 are that offset, not zero:
at a 0.25-day span, median 10 / 21 / 56 m in R / T / N and p95 71 / 137 /
1,006 m (60 span-test windows, June 2024); at a 1-day span, median 31 / 45 /
21 m and p95 |T| about 1.2 km (360 stratified windows). The largest are at
250-400 km, where the fit absorbs part of the drag-model difference.

**Epoch-state rules, kept for comparison.** ``preserve_sgp4_mean_motion`` (the
default from 2026-11-23 to 2026-11-25) starts at the SGP4 epoch state with one
correction. SGP4 states are consistent with the WGS-72 GM (398600.8 km^3/s^2);
truth integrates with the EIGEN-6S GM (398600.4415), so the same position and
velocity would have a slightly longer two-body period under truth and drift
behind SGP4 by about 1.15 km per day for that reason alone. The rule multiplies
position and velocity by ``k = (mu_truth / mu_WGS72) ** (1/3)``
(:func:`apply_start_rule`). Under ``mu_truth`` the scaled state has the original
state's two-body period under ``mu_WGS72``, the same eccentricity vector and the
same orbit plane; it sits about 2 m from the SGP4 state, so under this rule
residuals at t0 are about 2 m. It keeps the phase-dependent drift above.
``sgp4_state_direct`` (k = 1) is the SGP4 state as is. Callers that start from
one state with no SGP4 samples (physics tests, the settings scripts, the brahe
cross-check) name one of these rules explicitly.

Reproducibility
---------------
The same job gives the same trajectory to about a millimetre, not bit for bit:
Orekit's interpolated frame transforms take their sample grid from the first
date a JVM requests, so a job run after another at a nearby epoch can differ
from a fresh-worker run by ~1e-4 m over days. Compare trajectories with a
tolerance; don't hash them. At 300 km with drag, runs at a 120 s maximum step
and below also stop converging at about 1-18 m over 7 days for a high-drag
object (cause not identified); treat that as the reproducibility floor there.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from orbit_correction.types import Trajectory

if TYPE_CHECKING:
    from orbit_correction.spaceweather import SpaceWeather

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "config" / "truth.toml"
#: Source of the truth data-end margin (``truth_data_end_margin_days``).
SPLITS_CONFIG_PATH = REPO / "config" / "splits.toml"
#: The fitted start rule, the configured default since 2026-11-25 (module docstring).
FIT_START_RULE = "fit_to_sgp4_before_epoch"
#: The epoch-state rule that was the default before it, kept for comparison.
EPOCH_START_RULE = "preserve_sgp4_mean_motion"
#: Start rules accepted in ``[truth.start] rule``.
START_RULES = (EPOCH_START_RULE, "sgp4_state_direct", FIT_START_RULE)
#: Outcomes of a fitted start (:class:`StartFit`); only ``converged`` gives a trajectory.
FIT_STATUSES = ("converged", "not_converged", "diverged")

_DAY_S = 86400.0

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TruthConfig:
    """Every physical and numerical choice of the truth propagator.

    Loaded from ``config/truth.toml`` by :func:`load_truth_config`; each field is
    documented there. Frozen and hashable, so a worker can cache Orekit objects
    per configuration. Tests build variants with :func:`dataclasses.replace`
    (for example ``atmosphere="none"`` to isolate gravity).
    """

    status: str
    orekit_data_commit: str
    gravity_field_file: str
    zonal_degree: int
    zonal_order: int
    atmosphere: str
    cd: float
    bstar_reference_density_kg_per_m2_er: float
    mass_kg: float
    msis_local_solar_time: str
    msis_ap_switch: int
    iers_conventions: str
    simple_eop: bool
    body_shape: str
    integrator: str
    position_tolerance_m: float
    min_step_s: float
    max_step_s: float
    output_step_s: float
    reentry_altitude_km: float
    reentry_max_check_s: float
    jvm_active_processor_count: int
    # The configured default (2026-11-25), so a config built here without naming a
    # rule doesn't reach the epoch-state rule.
    start_rule: str = FIT_START_RULE
    sgp4_mu_km3_s2: float = 398600.8
    # Fitted start (used only when start_rule == FIT_START_RULE; see [truth.start]).
    fit_span_s: float = 0.25 * 86400.0
    fit_step_s: float = 60.0
    fit_max_iterations: int = 8
    fit_tolerance_m: float = 1.0e-3
    fit_tolerance_m_s: float = 1.0e-6

    def __post_init__(self) -> None:
        if self.zonal_degree < 0 or self.zonal_degree == 1:
            raise ValueError(f"zonal_degree must be 0 or >= 2, got {self.zonal_degree}")
        if self.zonal_order != 0:
            raise ValueError("zonal_order must be 0: the truth model is zonal-only by design")
        if self.atmosphere not in ("NRLMSISE-00", "none"):
            raise ValueError(f"atmosphere must be 'NRLMSISE-00' or 'none', got {self.atmosphere!r}")
        if self.msis_local_solar_time not in ("MEAN", "APPARENT"):
            raise ValueError("msis_local_solar_time must be 'MEAN' or 'APPARENT'")
        if self.msis_ap_switch not in (1, -1):
            raise ValueError("msis_ap_switch must be 1 (daily Ap) or -1 (ap array)")
        if self.integrator != "DOP853":
            raise ValueError(f"only DOP853 is supported, got {self.integrator!r}")
        if self.iers_conventions != "IERS_2010" or self.body_shape != "WGS84":
            raise ValueError("only IERS_2010 conventions and the WGS84 ellipsoid are supported")
        for name in (
            "cd",
            "bstar_reference_density_kg_per_m2_er",
            "mass_kg",
            "position_tolerance_m",
            "min_step_s",
            "max_step_s",
            "output_step_s",
            "reentry_max_check_s",
        ):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if self.min_step_s > self.max_step_s:
            raise ValueError("min_step_s must not exceed max_step_s")
        if self.start_rule not in START_RULES:
            raise ValueError(f"start_rule must be one of {START_RULES}, got {self.start_rule!r}")
        if not self.sgp4_mu_km3_s2 > 0:
            raise ValueError("sgp4_mu_km3_s2 must be positive")
        for name in ("fit_span_s", "fit_step_s", "fit_tolerance_m", "fit_tolerance_m_s"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if self.fit_step_s > self.fit_span_s / 2:
            raise ValueError("fit_step_s must leave at least three samples in fit_span_s")
        if self.fit_max_iterations < 1:
            raise ValueError("fit_max_iterations must be >= 1")
        if self.jvm_active_processor_count < 0:
            raise ValueError("jvm_active_processor_count must be >= 0 (0 = no cap)")


def load_truth_config(path: str | Path = CONFIG_PATH) -> TruthConfig:
    """Read ``config/truth.toml`` (or another file with the same layout)."""
    with open(path, "rb") as f:
        t = tomllib.load(f)["truth"]
    start = t["start"]
    # The fit settings are optional, so files written before the fitted rule still
    # load; a missing one takes TruthConfig's default. The rule itself is required:
    # a file that names none is refused (KeyError), never given a rule.
    fit = {}
    if "fit_span_days" in start:
        fit["fit_span_s"] = float(start["fit_span_days"]) * _DAY_S
    for key, typ in (("fit_step_s", float), ("fit_max_iterations", int),
                     ("fit_tolerance_m", float), ("fit_tolerance_m_s", float)):
        if key in start:
            fit[key] = typ(start[key])
    return TruthConfig(
        status=t["status"],
        orekit_data_commit=t["data"]["orekit_data_commit"],
        gravity_field_file=t["gravity"]["field_file"],
        zonal_degree=int(t["gravity"]["zonal_degree"]),
        zonal_order=int(t["gravity"]["zonal_order"]),
        atmosphere=t["drag"]["atmosphere"],
        cd=float(t["drag"]["cd"]),
        bstar_reference_density_kg_per_m2_er=float(
            t["drag"]["bstar_reference_density_kg_per_m2_er"]
        ),
        mass_kg=float(t["drag"]["mass_kg"]),
        msis_local_solar_time=t["drag"]["msis_local_solar_time"],
        msis_ap_switch=int(t["drag"]["msis_ap_switch"]),
        iers_conventions=t["earth"]["iers_conventions"],
        simple_eop=bool(t["earth"]["simple_eop"]),
        body_shape=t["earth"]["body_shape"],
        integrator=t["integrator"]["method"],
        position_tolerance_m=float(t["integrator"]["position_tolerance_m"]),
        min_step_s=float(t["integrator"]["min_step_s"]),
        max_step_s=float(t["integrator"]["max_step_s"]),
        output_step_s=float(t["output"]["step_s"]),
        reentry_altitude_km=float(t["reentry"]["altitude_km"]),
        reentry_max_check_s=float(t["reentry"]["max_check_s"]),
        jvm_active_processor_count=int(t["jvm"]["active_processor_count"]),
        start_rule=start["rule"],
        sgp4_mu_km3_s2=float(start["sgp4_mu_km3_s2"]),
        **fit,
    )


# --------------------------------------------------------------------------------------
# Pure helpers (no JVM)
# --------------------------------------------------------------------------------------


def area_to_mass_from_bstar(bstar_per_er: float, cfg: TruthConfig) -> float:
    """Area-to-mass ratio (m^2/kg) implied by an SGP4 B* term (1/Earth radii).

    The standard B* relation is ``B* = (Cd * A/m) * rho0 / 2`` with the SGP4
    reference density ``rho0 = 0.15696615 kg/(m^2 ER)``, so
    ``A/m = 2 B* / (Cd rho0)``. Cd and rho0 come from the config. B* is fitted by
    whoever produced the element set and absorbs any drag-model error, so this is
    a model-world A/m, not a physical one.

    Raises:
        ValueError: if B* is not positive and finite. Negative or zero B* (for
            example IRAS) has no physical A/m; pass ``area_to_mass_m2_per_kg`` to
            :func:`propagate_truth` instead.
    """
    if not (math.isfinite(bstar_per_er) and bstar_per_er > 0):
        raise ValueError(
            f"B* = {bstar_per_er!r}: only positive B* maps to an area-to-mass ratio; "
            "give area_to_mass_m2_per_kg explicitly for this object"
        )
    return 2.0 * bstar_per_er / (cfg.cd * cfg.bstar_reference_density_kg_per_m2_er)


def _omm_record_elset(record: dict, epoch_utc: str | None = None):
    """One OMM record (CelesTrak/Space-Track JSON keys) as a one-row element-set table.

    ``epoch_utc``, if given, replaces the record's EPOCH (``epoch_tt_s`` follows).
    Goes through the catalog loader's record conversion, so the row is exactly what
    ``catalog.load_celestrak_omm`` would give for the same record.
    """
    from orbit_correction import catalog

    rec = dict(record)
    if epoch_utc is not None:
        rec["EPOCH"] = epoch_utc
    return catalog.records_to_table([rec], "omm_record")


def initial_state_from_omm(record: dict, epoch_utc: str | None = None) -> tuple[Trajectory, float]:
    """SGP4 state at the element-set epoch, in GCRF, plus the set's B*. Deprecated.

    A thin wrapper over :func:`orbit_correction.cheap_prop.sgp4_epoch_states`, the
    project's SGP4 wrapper (TLE-reader arithmetic, WGS-72, mode 'i', exact TEME ->
    GCRF rotation, the 120 km floor). It is kept only because tests and scripts start
    synthetic cases from a raw OMM record with a replaced epoch; new code should load
    element sets with :mod:`orbit_correction.catalog` and call ``sgp4_epoch_states``.

    Args:
        record: one OMM record (CelesTrak/Space-Track JSON keys).
        epoch_utc: if given, replaces the record's EPOCH, so the same mean elements
            start at another date. Used to build realistic LEO states at historical
            epochs where the space-weather file is fully observed; such a state is a
            synthetic test case, not that object's real orbit at that date.

    Returns:
        ``(Trajectory(source="sgp4"), bstar_per_er)``; the trajectory has one state.

    Raises:
        ValueError: if SGP4 fails at the epoch or the state is below the altitude floor.
    """
    from orbit_correction.cheap_prop import sgp4_epoch_states

    elset = _omm_record_elset(record, epoch_utc)
    res = sgp4_epoch_states(elset)[0]
    if res.trajectory is None:
        raise ValueError(
            f"NORAD {res.norad_id}: no SGP4 state at the epoch ({res.status.message})"
        )
    return res.trajectory, float(elset.column("bstar_per_earth_radii")[0].as_py())


def start_rule_scale(mu_truth_m3_s2: float, cfg: TruthConfig) -> float:
    """The factor k the start rule multiplies the initial position and velocity by.

    ``preserve_sgp4_mean_motion``: ``k = (mu_truth / mu_sgp4) ** (1/3)``. With
    ``r' = k r`` and ``v' = k v``, the vis-viva relation gives ``a' = k a``
    under ``mu_truth = k^3 mu_sgp4``, so the mean motion ``sqrt(mu / a^3)`` is
    unchanged; the eccentricity vector ``(v x h) / mu - r / |r|`` and the plane
    are unchanged too. ``sgp4_state_direct``: k = 1. ``fit_to_sgp4_before_epoch``:
    the same k as ``preserve_sgp4_mean_motion``, applied to the fit's initial guess
    only (the fitted state replaces it).

    Args:
        mu_truth_m3_s2: GM of the truth gravity field (Orekit's, m^3/s^2).
        cfg: configuration (``start_rule``, ``sgp4_mu_km3_s2``).
    """
    if cfg.start_rule == "sgp4_state_direct":
        return 1.0
    return (float(mu_truth_m3_s2) / (cfg.sgp4_mu_km3_s2 * 1.0e9)) ** (1.0 / 3.0)


def _scaled(initial: Trajectory, k: float) -> Trajectory:
    if k == 1.0:
        return initial
    return Trajectory(
        norad_id=initial.norad_id,
        t_tt_s=initial.t_tt_s.copy(),
        r_gcrf_km=initial.r_gcrf_km * k,
        v_gcrf_km_s=initial.v_gcrf_km_s * k,
        source=initial.source,
    )


def apply_start_rule(initial: Trajectory, mu_truth_m3_s2: float, cfg: TruthConfig) -> Trajectory:
    """The truth initial state: ``initial`` scaled by :func:`start_rule_scale`.

    :func:`propagate_truth` calls this; scripts that build their own Orekit
    propagators (the brahe cross-check) call it to start from the same state.
    Returns ``initial`` itself when k = 1.

    Raises:
        ValueError: for the fitted rule, whose start depends on SGP4 states before
            the epoch, not on ``initial`` alone (use :func:`fit_start_to_sgp4`).
    """
    if cfg.start_rule == FIT_START_RULE:
        raise ValueError(
            f"start rule {FIT_START_RULE!r} needs SGP4 samples before the epoch; "
            "use fit_start_to_sgp4 or propagate_truth(..., fit_samples=...)"
        )
    return _scaled(initial, start_rule_scale(mu_truth_m3_s2, cfg))


def output_grid(t0_tt_s: float, horizon_s: float, step_s: float) -> np.ndarray:
    """A regular output grid ``t0, t0 + step, ..., t0 + horizon`` (TT seconds).

    Built as ``t0 + k * step`` (no accumulated rounding). The last point is
    ``t0 + horizon`` when ``horizon`` is a multiple of ``step``.
    """
    n = int(math.floor(horizon_s / step_s + 1e-9))
    return float(t0_tt_s) + step_s * np.arange(n + 1, dtype=np.float64)


def fit_sample_offsets_s(cfg: TruthConfig) -> np.ndarray:
    """Times of the fitted start's SGP4 samples, seconds from the epoch: ``-span .. 0``.

    ``-n * step, ..., -step, 0`` with ``n = floor(span / step)``, built from the epoch
    backwards so the last sample is exactly the epoch. Every value is <= 0: the fit
    sees nothing after the epoch.
    """
    n = int(math.floor(cfg.fit_span_s / cfg.fit_step_s + 1e-9))
    return -cfg.fit_step_s * np.arange(n, -1, -1, dtype=np.float64)


def fit_samples_from_elsets(elsets, cfg: TruthConfig):
    """SGP4 states for the fitted start, one :class:`~orbit_correction.cheap_prop.Sgp4Result`
    per element set, at :func:`fit_sample_offsets_s` from each set's own epoch (GCRF).

    This is the hook a generation pipeline uses (no JVM; run it in the parent
    process beside the window's own SGP4 run): pass each result's ``trajectory`` as
    ``fit_samples`` to :func:`propagate_truth` or :class:`TruthJob`. Only results
    with ``status.ok`` are usable; any other status (an SGP4 error before the epoch,
    or the 120 km floor) means the window has no fitted start and should be counted
    under its own status, not given a fallback start.

    Args:
        elsets: element sets with the catalog schema (``catalog.ELEMENT_SET_SCHEMA``).
        cfg: configuration (``fit_span_s``, ``fit_step_s``).
    """
    from orbit_correction.cheap_prop import propagate_sgp4_from_epoch

    return propagate_sgp4_from_epoch(elsets, fit_sample_offsets_s(cfg))


# --------------------------------------------------------------------------------------
# JVM
# --------------------------------------------------------------------------------------


def jvm_started() -> bool:
    """Whether this process has started a JVM. Never imports jpype itself."""
    jp = sys.modules.get("jpype")
    return bool(jp is not None and jp.isJVMStarted())


def start_jvm(active_processor_count: int = 2, extra_vmargs: Sequence[str] = ()) -> None:
    """Start the JVM in this process and register the pinned orekit-data bundle.

    Safe to call more than once: later calls do nothing (a JVM can't be
    restarted, so the first call's options stay in force). Call it only in
    processes that will never fork afterwards and never import PyTorch: worker
    processes, scripts and the test session.

    Args:
        active_processor_count: cap for ``-XX:ActiveProcessorCount``; 0 means no cap.
        extra_vmargs: further JVM options.

    Raises:
        RuntimeError: if this process has imported ``torch`` (the training
            process must never share a process with a JVM).
        ImportError: if the ``datagen`` dependency group isn't installed.
    """
    _refuse_if_torch_loaded()
    if jvm_started():
        return
    import jdk4py  # noqa: PLC0415  (lazy by design: see the JVM rules above)

    os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
    import orekit_jpype  # noqa: PLC0415

    # --enable-native-access silences JDK 25's restricted-native-access warnings.
    vmargs = ["--enable-native-access=ALL-UNNAMED"]
    if active_processor_count:
        vmargs.append(f"-XX:ActiveProcessorCount={int(active_processor_count)}")
    vmargs += list(extra_vmargs)
    orekit_jpype.initVM(vmargs=",".join(vmargs))
    from orekit_jpype.pyhelpers import setup_orekit_data  # noqa: PLC0415

    setup_orekit_data(from_pip_library=True)


def _refuse_if_torch_loaded() -> None:
    if "torch" in sys.modules:
        raise RuntimeError(
            "refusing to use a JVM in a process that has imported torch; run truth "
            "propagation through propagate_batch (spawned workers) or in its own process"
        )


def orekit_provenance() -> dict[str, str]:
    """Versions of the Orekit stack in use (no JVM needed)."""
    from importlib.metadata import version  # noqa: PLC0415

    out = {}
    for pkg in ("orekit-jpype", "jpype1", "jdk4py", "orekitdata"):
        try:
            out[pkg] = version(pkg)
        except Exception:  # noqa: BLE001  (not installed)
            out[pkg] = "not installed"
    return out


def _check_orekit_data(cfg: TruthConfig) -> None:
    """Fail if the installed orekit-data bundle isn't the commit the config names."""
    installed = orekit_provenance()["orekitdata"]
    short = cfg.orekit_data_commit[:7]
    if f"g{short}" not in installed:
        raise RuntimeError(
            f"orekit-data bundle {installed!r} does not match config commit "
            f"{cfg.orekit_data_commit}; re-run `uv sync --group datagen`"
        )


# --------------------------------------------------------------------------------------
# Orekit objects, built once per (process, configuration)
# --------------------------------------------------------------------------------------


class _Orekit:
    """Orekit frames, bodies and models for one configuration, in one JVM.

    Built lazily by :func:`_context`. Loading frames, the gravity field and a
    space-weather file costs about two seconds, so these objects are reused for
    every propagation in the process.
    """

    def __init__(self, cfg: TruthConfig) -> None:
        _check_orekit_data(cfg)
        from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
        from org.orekit.forces.gravity.potential import GravityFieldFactory, ICGEMFormatReader
        from org.orekit.frames import FramesFactory
        from org.orekit.time import AbsoluteDate, TimeScalesFactory
        from org.orekit.utils import Constants, IERSConventions

        self.cfg = cfg
        self.utc = TimeScalesFactory.getUTC()
        self.tt = TimeScalesFactory.getTT()
        conventions = getattr(IERSConventions, cfg.iers_conventions)
        self.ut1 = TimeScalesFactory.getUT1(conventions, cfg.simple_eop)
        self.j2000 = AbsoluteDate.J2000_EPOCH  # 2000-01-01T12:00:00 TT
        self.gcrf = FramesFactory.getGCRF()
        self.teme = FramesFactory.getTEME()
        self.itrf = FramesFactory.getITRF(conventions, cfg.simple_eop)
        self.earth = OneAxisEllipsoid(
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, self.itrf
        )
        self.sun = CelestialBodyFactory.getSun()
        # Read exactly the configured gravity file (not whatever else the bundle holds).
        GravityFieldFactory.clearPotentialCoefficientsReaders()
        GravityFieldFactory.addPotentialCoefficientsReader(
            ICGEMFormatReader(cfg.gravity_field_file, False)
        )
        self._gravity: dict[int, object] = {}
        self._sw: dict[str, object] = {}
        self._atm: dict[str, object] = {}
        # mu of the gravity field, used for the central attraction in every run.
        self.mu = self.gravity_provider(2).getMu()

    # -- time
    def date(self, t_tt_s: float):
        """TT seconds since J2000.0 -> AbsoluteDate."""
        return self.j2000.shiftedBy(float(t_tt_s))

    def tt_seconds(self, date) -> float:
        return float(date.durationFrom(self.j2000))

    # -- models
    def gravity_provider(self, degree: int):
        """Normalized EIGEN-6S provider, zonal terms to ``degree`` (order 0)."""
        if degree not in self._gravity:
            from org.orekit.forces.gravity.potential import GravityFieldFactory

            self._gravity[degree] = GravityFieldFactory.getNormalizedProvider(int(degree), 0)
        return self._gravity[degree]

    def space_weather(self, sw: SpaceWeatherCopy | str | Path):
        """CssiSpaceWeatherData reading exactly this file (our written copy).

        Cached by (path, SHA-256 of the content), so a copy rewritten at the same
        path is read again rather than served from the cache. For a
        :class:`SpaceWeatherCopy` the file must still have the content it was
        written with. A bare path (low-level tests only) is hashed on every call.
        """
        key = _sw_key(sw)
        if key not in self._sw:
            from java.io import File
            from org.orekit.data import DataSource
            from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData

            path = key[0]
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            if _sha256(path) != key[1]:
                raise SpaceWeatherRangeError(
                    f"{path} changed since it was written; write a new copy"
                )
            self._sw[key] = CssiSpaceWeatherData(DataSource(File(path)), self.utc)
        return self._sw[key]

    def atmosphere(self, sw: SpaceWeatherCopy | str | Path):
        """NRLMSISE-00 fed by our space-weather copy, with the configured switches."""
        key = _sw_key(sw)
        if key not in self._atm:
            from org.orekit.models.earth.atmosphere import NRLMSISE00

            atm = NRLMSISE00(self.space_weather(sw), self.sun, self.earth, self.ut1)
            atm = atm.withLocalSolarTimeMode(
                getattr(NRLMSISE00.LocalSolarTimeMode, self.cfg.msis_local_solar_time)
            )
            atm = atm.withSwitch(9, int(self.cfg.msis_ap_switch))
            self._atm[key] = atm
        return self._atm[key]


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sw_key(sw: SpaceWeatherCopy | str | Path) -> tuple[str, str]:
    if isinstance(sw, SpaceWeatherCopy):
        return (str(Path(sw.path).resolve()), sw.sha256)
    path = str(Path(sw).resolve())
    return (path, _sha256(path) if Path(path).is_file() else "")


_CONTEXTS: dict[TruthConfig, _Orekit] = {}


def _context(cfg: TruthConfig) -> _Orekit:
    """The process's Orekit objects for ``cfg``; starts the JVM if needed."""
    _refuse_if_torch_loaded()
    if not jvm_started():
        start_jvm(cfg.jvm_active_processor_count)
    if cfg not in _CONTEXTS:
        _CONTEXTS[cfg] = _Orekit(cfg)
    return _CONTEXTS[cfg]


# --------------------------------------------------------------------------------------
# Propagation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TruthResult:
    """Outcome of one truth propagation.

    Attributes:
        norad_id: the object.
        status: ``"ok"`` (``trajectory`` holds the full grid), ``"reentry"`` (the
            geodetic altitude fell below ``cfg.reentry_altitude_km`` inside the
            window, or the start was already below it; ``trajectory`` holds the
            grid points up to the stop time, or is None if there are none or the
            start was below the threshold) or ``"error"`` (only from
            :func:`propagate_batch`, which catches per-job exceptions so one bad
            object doesn't stop a batch; ``message`` says why), or ``"fit_failed"``
            (fitted start rule only: the fit did not converge; ``start_fit`` says
            how, and there is no trajectory).
        trajectory: ``Trajectory(source="truth")`` on the requested grid (all of
            it for ``ok``, the points at or before ``stop_t_tt_s`` for
            ``reentry``), or None.
        stop_t_tt_s: when the re-entry altitude was crossed (TT seconds), the
            initial epoch for a start below it, else None.
        area_to_mass_m2_per_kg: the A/m used.
        start_scale: the start rule's factor k (1.0 for ``sgp4_state_direct``;
            None for the fitted rule, whose start is the fitted state).
        n_evaluations: right-hand-side evaluations of the forward integration
            (the fit's own are in ``start_fit``).
        wall_s: wall-clock time of the whole call: fit (if any), propagation and
            sampling.
        message: human-readable detail.
        start_fit: the fit's record (:class:`StartFit`) under the fitted rule, else
            None.
    """

    norad_id: int
    status: str
    trajectory: Trajectory | None = None
    stop_t_tt_s: float | None = None
    area_to_mass_m2_per_kg: float | None = None
    start_scale: float | None = None
    n_evaluations: int | None = None
    wall_s: float | None = None
    message: str = ""
    start_fit: StartFit | None = None


@dataclass(frozen=True)
class StartFit:
    """Record of one fitted start (:func:`fit_start_to_sgp4`).

    Attributes:
        status: ``"converged"`` (the last Gauss-Newton correction was below both
            tolerances), ``"not_converged"`` (``fit_max_iterations`` passes without
            that) or ``"diverged"`` (the RMS rose between passes, or went
            non-finite). Only ``converged`` gives a start.
        iterations: propagations with the state transition matrix (passes).
        n_samples: SGP4 positions fitted.
        first_sample_tt_s, last_sample_tt_s: their time span (TT seconds); the last
            is at or before the epoch.
        rms_m: RMS of |SGP4 - truth| position over the samples at the last pass.
        max_m: largest |SGP4 - truth| over the samples at the last pass.
        rms_initial_m: the same RMS at the initial guess (the start-rule-scaled SGP4
            epoch state), for comparison.
        offset_r_m, offset_v_m_s: fitted state minus the (unscaled) SGP4 epoch
            state, GCRF, m and m/s.
        last_correction_r_m, last_correction_v_m_s: size of the last correction.
        n_evaluations: right-hand-side evaluations over all passes.
        wall_s: wall-clock time of the fit.
        message: detail for a failed fit.
    """

    status: str
    iterations: int
    n_samples: int
    first_sample_tt_s: float
    last_sample_tt_s: float
    rms_m: float
    max_m: float
    rms_initial_m: float
    offset_r_m: tuple[float, float, float]
    offset_v_m_s: tuple[float, float, float]
    last_correction_r_m: float
    last_correction_v_m_s: float
    n_evaluations: int
    wall_s: float
    message: str = ""

    @property
    def converged(self) -> bool:
        return self.status == "converged"


def build_propagator(
    cfg: TruthConfig,
    initial: Trajectory,
    area_to_mass_m2_per_kg: float | None,
    space_weather: SpaceWeatherCopy | str | Path | None,
    *,
    atmosphere=None,
    gravity_provider=None,
):
    """Configured Orekit ``NumericalPropagator`` for one initial state.

    Exposed for tests and scripts that need Orekit objects; normal callers use
    :func:`propagate_truth`. Returns ``(propagator, integrator, context)``.

    This is the low-level builder: it uses ``initial`` exactly as given (no
    start rule) and doesn't check the window against the space-weather copy.
    :func:`propagate_truth` does both.

    Args:
        cfg: configuration.
        initial: one-state trajectory (GCRF, km, km/s, TT seconds).
        area_to_mass_m2_per_kg: A/m for drag; ignored when ``cfg.atmosphere`` is
            ``"none"``.
        space_weather: our Orekit copy (:func:`write_space_weather_copy`, or a
            path to a file written by ``spaceweather.write_orekit_cssi``);
            required with NRLMSISE-00.
        atmosphere: test hook, an Orekit ``Atmosphere`` replacing NRLMSISE-00.
        gravity_provider: cross-check hook, an Orekit
            ``NormalizedSphericalHarmonicsProvider`` replacing EIGEN-6S (for
            example a static zonal set shared with brahe). Its mu is used for the
            central attraction.
    """
    ctx = _context(cfg)
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
    from org.orekit.forces.drag import DragForce, IsotropicDrag
    from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
    from org.orekit.orbits import CartesianOrbit, OrbitType
    from org.orekit.propagation import SpacecraftState, ToleranceProvider
    from org.orekit.propagation.numerical import NumericalPropagator
    from org.orekit.utils import PVCoordinates

    if len(initial) != 1:
        raise ValueError(f"initial must be a one-state Trajectory, got {len(initial)} states")
    r = initial.r_gcrf_km[0] * 1000.0
    v = initial.v_gcrf_km_s[0] * 1000.0
    pv = PVCoordinates(Vector3D(*map(float, r)), Vector3D(*map(float, v)))
    mu = ctx.mu if gravity_provider is None else gravity_provider.getMu()
    orbit = CartesianOrbit(pv, ctx.gcrf, ctx.date(initial.t_tt_s[0]), mu)

    tol = ToleranceProvider.getDefaultToleranceProvider(float(cfg.position_tolerance_m))
    abs_tol, rel_tol = tol.getTolerances(orbit, OrbitType.CARTESIAN)
    integ = DormandPrince853Integrator(cfg.min_step_s, cfg.max_step_s, abs_tol, rel_tol)
    prop = NumericalPropagator(integ)
    prop.setOrbitType(OrbitType.CARTESIAN)
    prop.setMu(mu)  # central attraction; the zonal model below adds only the rest
    if gravity_provider is not None:
        prop.addForceModel(HolmesFeatherstoneAttractionModel(ctx.itrf, gravity_provider))
    elif cfg.zonal_degree >= 2:
        prop.addForceModel(
            HolmesFeatherstoneAttractionModel(ctx.itrf, ctx.gravity_provider(cfg.zonal_degree))
        )
    if atmosphere is not None or cfg.atmosphere == "NRLMSISE-00":
        if area_to_mass_m2_per_kg is None or not area_to_mass_m2_per_kg > 0:
            raise ValueError("drag needs a positive area_to_mass_m2_per_kg")
        if atmosphere is None:
            if space_weather is None:
                raise ValueError("NRLMSISE-00 needs a space-weather copy")
            atmosphere = ctx.atmosphere(space_weather)
        area = float(area_to_mass_m2_per_kg) * cfg.mass_kg
        prop.addForceModel(DragForce(atmosphere, IsotropicDrag(area, float(cfg.cd))))
    prop.setInitialState(SpacecraftState(orbit).withMass(float(cfg.mass_kg)))
    return prop, integ, ctx


def describe_gravity(prop) -> list[dict]:
    """The Holmes-Featherstone gravity models on a built propagator, as data.

    One dict per model: ``degree`` and ``order`` of its coefficient provider,
    ``body_frame`` (the frame name) and ``mu``. Orekit 13.1.9 doesn't expose the
    provider or the frame publicly, so they are read by Java reflection; this
    exists so tests can check that the configured degree, order 0 and the
    Earth-fixed frame reach the force model (a silent J2-only or inertial-frame
    fallback would otherwise pass every physics test).
    """
    out = []
    for f in prop.getAllForceModels():
        cls = f.getClass()
        if cls.getSimpleName() != "HolmesFeatherstoneAttractionModel":
            continue
        fp = cls.getDeclaredField("provider")
        fp.setAccessible(True)
        fb = cls.getDeclaredField("bodyFrame")
        fb.setAccessible(True)
        provider, frame = fp.get(f), fb.get(f)
        out.append(
            {
                "degree": int(provider.getMaxDegree()),
                "order": int(provider.getMaxOrder()),
                "body_frame": str(frame.getName()),
                "mu": float(provider.getMu()),
            }
        )
    return out


def initial_geodetic_altitude_km(cfg: TruthConfig, initial: Trajectory) -> float:
    """Geodetic (WGS84) altitude of a one-state trajectory's position, km."""
    ctx = _context(cfg)
    from org.hipparchus.geometry.euclidean.threed import Vector3D

    r = initial.r_gcrf_km[0] * 1000.0
    gp = ctx.earth.transform(
        Vector3D(*map(float, r)), ctx.gcrf, ctx.date(float(initial.t_tt_s[0]))
    )
    return float(gp.getAltitude()) / 1000.0


def _area_to_mass(
    cfg: TruthConfig, bstar_per_er: float | None, area_to_mass_m2_per_kg: float | None
) -> float | None:
    if cfg.atmosphere == "none":
        return None
    if area_to_mass_m2_per_kg is not None:
        return float(area_to_mass_m2_per_kg)
    return area_to_mass_from_bstar(float(bstar_per_er), cfg)


def _check_fit_samples(initial: Trajectory, samples: Trajectory | None) -> None:
    """Refuse fit samples that are missing, too few, or reach past the epoch."""
    if samples is None:
        raise ValueError(
            f"start rule {FIT_START_RULE!r} needs fit_samples: SGP4 states at or before "
            "the epoch (truth_prop.fit_samples_from_elsets)"
        )
    if not isinstance(samples, Trajectory):
        raise TypeError(f"fit_samples must be a Trajectory, got {type(samples).__name__}")
    if samples.norad_id != initial.norad_id:
        raise ValueError(
            f"fit_samples are for NORAD {samples.norad_id}, the start is {initial.norad_id}"
        )
    if len(samples) < 3:
        raise ValueError(f"the fit needs at least 3 samples, got {len(samples)}")
    t0 = float(initial.t_tt_s[0])
    late = samples.t_tt_s > t0
    if late.any():
        raise ValueError(
            f"{int(late.sum())} fit sample(s) lie after the epoch (latest "
            f"{float(samples.t_tt_s.max()) - t0:+.6f} s); only SGP4 states at or before "
            "the epoch may enter the fit"
        )
    if not samples.t_tt_s[0] < t0:
        raise ValueError("the fit samples must start before the epoch")


def _fit_pass(cfg, ctx, x_m, t0, dates, a2m, space_weather):
    """One backward propagation from state ``x_m`` (m, m/s) at ``t0`` with the STM.

    Returns the model positions at ``dates`` (N, 3) in metres, d(position)/d(x0) at
    each date (N, 3, 6), and the number of right-hand-side evaluations. The state
    transition matrix comes from Orekit's matrices harvester (Cartesian state,
    variational equations integrated with the state); the step control uses the
    main state only, so the trajectory is the one :func:`build_propagator` gives
    without the matrix.
    """
    state = Trajectory(1, np.array([t0]), x_m[None, :3] / 1000.0, x_m[None, 3:] / 1000.0, "fit")
    prop, integ, _ = build_propagator(cfg, state, a2m, space_weather)
    harvester = prop.setupMatricesComputation("stm", None, None)
    generator = prop.getEphemerisGenerator()
    prop.propagate(ctx.date(t0), dates[0])
    ephemeris = generator.getGeneratedEphemeris()
    pos = np.empty((len(dates), 3))
    jac = np.empty((len(dates), 3, 6))
    gcrf = ctx.gcrf
    for j, d in enumerate(dates):
        s = ephemeris.propagate(d)
        p = s.getPVCoordinates(gcrf).getPosition()
        pos[j] = (p.getX(), p.getY(), p.getZ())
        phi = harvester.getStateTransitionMatrix(s)
        jac[j] = [[phi.getEntry(a, b) for b in range(6)] for a in range(3)]
    return pos, jac, int(integ.getEvaluations())


def fit_start_to_sgp4(
    initial: Trajectory,
    samples: Trajectory,
    bstar_per_er: float | None,
    cfg: TruthConfig,
    space_weather: SpaceWeatherCopy | None,
    *,
    area_to_mass_m2_per_kg: float | None = None,
) -> tuple[Trajectory | None, StartFit]:
    """Fit the truth state at the epoch to SGP4 positions at or before it.

    Unknowns: the GCRF position and velocity at ``initial``'s epoch. Data: the
    positions of ``samples`` (SGP4 in GCRF, all at or before the epoch). Model: the
    truth propagator (:func:`build_propagator` with ``cfg``), integrated backwards
    from the epoch. Unweighted least squares on position by Gauss-Newton: each pass
    propagates once with the state transition matrix and solves the linearized
    problem (columns scaled 1 m : 1 mm/s). It stops when the correction is below
    ``fit_tolerance_m`` and ``fit_tolerance_m_s``; the returned state includes that
    last correction. The initial guess is ``initial`` scaled by
    :func:`start_rule_scale`.

    Own Gauss-Newton rather than Orekit's propagator converters: the fitted state
    must sit at the epoch, which is the *last* sample, with exactly the truth force
    model, and the convergence record (status, passes, RMS) is ours to report.

    The space-weather range is not checked here; :func:`propagate_truth` checks the
    whole span from the first sample to the end of the window.

    Args:
        initial: one-state ``Trajectory``, the SGP4 state at the epoch (GCRF).
        samples: SGP4 GCRF states of the same object at or before the epoch (at
            least 3), for example from :func:`fit_samples_from_elsets`. Only the
            positions are used.
        bstar_per_er, area_to_mass_m2_per_kg: as in :func:`propagate_truth`.
        cfg: configuration (force model, integrator and ``fit_*`` settings).
        space_weather: the run's copy (None only without drag).

    Returns:
        ``(state, record)``: the fitted one-state ``Trajectory`` (``source`` copied
        from ``initial``) or None if the fit did not converge, and its
        :class:`StartFit`.

    Raises:
        ValueError: if any sample lies after the epoch, there are fewer than 3, or
            they are for another object.
    """
    _check_fit_samples(initial, samples)
    wall0 = time.perf_counter()
    ctx = _context(cfg)
    a2m = _area_to_mass(cfg, bstar_per_er, area_to_mass_m2_per_kg)
    t0 = float(initial.t_tt_s[0])
    guess = _scaled(initial, start_rule_scale(ctx.mu, cfg))
    x = np.concatenate([guess.r_gcrf_km[0], guess.v_gcrf_km_s[0]]) * 1000.0
    obs = samples.r_gcrf_km * 1000.0
    dates = [ctx.date(float(t)) for t in samples.t_tt_s]
    col = np.array([1.0, 1.0, 1.0, 1.0e-3, 1.0e-3, 1.0e-3])  # 1 m ~ 1 mm/s

    status, message = "not_converged", f"no convergence in {cfg.fit_max_iterations} passes"
    rms0 = rms = prev = math.inf
    worst = math.inf
    dr = dv = math.inf
    n_eval = 0
    it = 0
    for it in range(1, cfg.fit_max_iterations + 1):
        pos, jac, nev = _fit_pass(cfg, ctx, x, t0, dates, a2m, space_weather)
        n_eval += nev
        res = obs - pos
        dist = np.linalg.norm(res, axis=1)
        rms, worst = float(np.sqrt(np.mean(dist**2))), float(dist.max())
        if it == 1:
            rms0 = rms
        # Gauss-Newton on a least-squares problem this close to linear never raises
        # the RMS by more than rounding; a rise means the linearization broke down.
        if not math.isfinite(rms) or rms > prev * (1.0 + 1.0e-6) + 1.0e-3:
            status = "diverged"
            message = f"RMS rose from {prev:.6g} m to {rms:.6g} m at pass {it}"
            break
        prev = rms
        dx, *_ = np.linalg.lstsq(jac.reshape(-1, 6) * col, res.reshape(-1), rcond=None)
        dx = dx * col
        dr, dv = float(np.linalg.norm(dx[:3])), float(np.linalg.norm(dx[3:]))
        x = x + dx
        if dr < cfg.fit_tolerance_m and dv < cfg.fit_tolerance_m_s:
            status, message = "converged", ""
            break

    r0 = initial.r_gcrf_km[0] * 1000.0
    v0 = initial.v_gcrf_km_s[0] * 1000.0
    record = StartFit(
        status=status,
        iterations=it,
        n_samples=len(samples),
        first_sample_tt_s=float(samples.t_tt_s[0]),
        last_sample_tt_s=float(samples.t_tt_s[-1]),
        rms_m=rms,
        max_m=worst,
        rms_initial_m=rms0,
        offset_r_m=tuple(float(a) for a in x[:3] - r0),
        offset_v_m_s=tuple(float(a) for a in x[3:] - v0),
        last_correction_r_m=dr,
        last_correction_v_m_s=dv,
        n_evaluations=n_eval,
        wall_s=time.perf_counter() - wall0,
        message=message,
    )
    if status != "converged":
        return None, record
    fitted = Trajectory(
        norad_id=initial.norad_id,
        t_tt_s=initial.t_tt_s.copy(),
        r_gcrf_km=x[None, :3] / 1000.0,
        v_gcrf_km_s=x[None, 3:] / 1000.0,
        source=initial.source,
    )
    return fitted, record


def propagate_truth(
    initial: Trajectory,
    bstar_per_er: float | None,
    t_out_tt_s: np.ndarray,
    cfg: TruthConfig,
    space_weather: SpaceWeatherCopy | None,
    *,
    area_to_mass_m2_per_kg: float | None = None,
    fit_samples: Trajectory | None = None,
) -> TruthResult:
    """Propagate one object with the truth model and sample it on a time grid.

    Steps: check the window against the space-weather copy (from the first fit
    sample under the fitted rule); apply the start rule (:func:`apply_start_rule`;
    under the fitted rule, :func:`fit_start_to_sgp4`, returning ``fit_failed`` if
    it does not converge); return ``reentry`` at once if the start is
    below the re-entry altitude; otherwise one integration from the initial
    epoch to the last grid time, with Orekit's ephemeris generator recording the
    dense output, read at each grid time. Never one ``propagate()`` call per
    output time (that restarts the integrator at every point).

    Starts a JVM in this process if none is running. For batches use
    :func:`propagate_batch`, which keeps the JVM out of the calling process.

    Args:
        initial: one-state ``Trajectory`` at the element-set epoch (the SGP4
            state rotated to GCRF), before the start rule.
        bstar_per_er: the element set's B* (1/ER). Converted to A/m with
            :func:`area_to_mass_from_bstar`. Ignored if ``area_to_mass_m2_per_kg``
            is given or ``cfg.atmosphere == "none"``.
        t_out_tt_s: output times, TT seconds since J2000.0, strictly increasing,
            none before the initial epoch.
        cfg: configuration (``load_truth_config()``).
        space_weather: the :class:`SpaceWeatherCopy` from
            :func:`write_space_weather_copy` covering the window; Orekit reads
            this file and nothing else for space weather. May be None only when
            ``cfg.atmosphere == "none"``.
        area_to_mass_m2_per_kg: explicit A/m, overriding B* (for objects whose
            B* has no physical meaning, such as negative B*).
        fit_samples: fitted rule only, and required there: SGP4 GCRF states of
            this object at or before the epoch (:func:`fit_samples_from_elsets`).
            Given with any other rule, it is refused rather than ignored.

    Returns:
        :class:`TruthResult` with status ``"ok"``, ``"reentry"`` or (fitted rule)
        ``"fit_failed"``.

    Raises:
        TypeError: if drag is on and ``space_weather`` isn't a
            :class:`SpaceWeatherCopy` (a bare path carries no checked range).
        SpaceWeatherRangeError: if the window (from the first fit sample under
            the fitted rule) isn't inside the copy's checked range or ends inside
            the data-end margin.
        ValueError: on other bad inputs, including fit samples after the epoch.
            Orekit exceptions propagate as raised.
    """
    t_out = np.asarray(t_out_tt_s, dtype=np.float64)
    if t_out.ndim != 1 or t_out.size == 0:
        raise ValueError("t_out_tt_s must be a non-empty 1-D array")
    if t_out.size > 1 and not np.all(np.diff(t_out) > 0):
        raise ValueError("t_out_tt_s must be strictly increasing")
    t0 = float(initial.t_tt_s[0])
    if t_out[0] < t0:
        raise ValueError("output times must not precede the initial epoch")
    fitted = cfg.start_rule == FIT_START_RULE
    if fitted:
        _check_fit_samples(initial, fit_samples)
    elif fit_samples is not None:
        raise ValueError(
            f"fit_samples given but the start rule is {cfg.start_rule!r}; they would be ignored"
        )
    t_first = float(fit_samples.t_tt_s[0]) if fitted else t0

    a2m = None
    if cfg.atmosphere != "none":
        _require_copy(space_weather)
        space_weather.check_window(t_first, float(t_out[-1]))
        a2m = _area_to_mass(cfg, bstar_per_er, area_to_mass_m2_per_kg)

    wall0 = time.perf_counter()
    ctx = _context(cfg)
    k = start_rule_scale(ctx.mu, cfg)
    # The altitude guard runs on the start-rule state (for the fitted rule, the
    # fit's initial guess, tens of metres from the fitted state), so a start below
    # the re-entry altitude is never fitted or integrated.
    start = _scaled(initial, k)
    record = None

    alt0 = initial_geodetic_altitude_km(cfg, start)
    if alt0 < cfg.reentry_altitude_km:
        return TruthResult(
            norad_id=initial.norad_id,
            status="reentry",
            stop_t_tt_s=t0,
            area_to_mass_m2_per_kg=a2m,
            start_scale=None if fitted else k,
            n_evaluations=0,
            wall_s=time.perf_counter() - wall0,
            message=(
                f"initial geodetic altitude {alt0:.1f} km is below "
                f"{cfg.reentry_altitude_km} km; not propagated"
            ),
        )

    if fitted:
        start, record = fit_start_to_sgp4(
            initial, fit_samples, bstar_per_er, cfg, space_weather,
            area_to_mass_m2_per_kg=a2m,
        )
        k = None
        if start is None:
            return TruthResult(
                norad_id=initial.norad_id,
                status="fit_failed",
                area_to_mass_m2_per_kg=a2m,
                n_evaluations=0,
                wall_s=time.perf_counter() - wall0,
                message=f"fitted start {record.status}: {record.message}",
                start_fit=record,
            )

    prop, integ, ctx = build_propagator(cfg, start, a2m, space_weather)

    from org.orekit.propagation.events import AltitudeDetector
    from org.orekit.propagation.events.handlers import StopOnDecreasing

    detector = AltitudeDetector(
        float(cfg.reentry_max_check_s), 1.0e-3, cfg.reentry_altitude_km * 1000.0, ctx.earth
    ).withHandler(StopOnDecreasing())
    prop.addEventDetector(detector)

    generator = prop.getEphemerisGenerator()
    end_date = ctx.date(t_out[-1])
    final = prop.propagate(ctx.date(t0), end_date)
    n_eval = int(integ.getEvaluations())
    stop_t = ctx.tt_seconds(final.getDate())
    reentered = final.getDate().durationFrom(end_date) < -1.0e-6
    # sample the dense output at every grid time the integration reached
    t_keep = t_out[t_out <= stop_t] if reentered else t_out
    traj = None
    if t_keep.size:
        ephemeris = generator.getGeneratedEphemeris()
        r = np.empty((t_keep.size, 3))
        v = np.empty((t_keep.size, 3))
        gcrf = ctx.gcrf
        for j, t in enumerate(t_keep):
            pv = ephemeris.getPVCoordinates(ctx.date(t), gcrf)
            p, w = pv.getPosition(), pv.getVelocity()
            r[j] = (p.getX(), p.getY(), p.getZ())
            v[j] = (w.getX(), w.getY(), w.getZ())
        traj = Trajectory(
            norad_id=initial.norad_id,
            t_tt_s=t_keep.copy(),
            r_gcrf_km=r / 1000.0,
            v_gcrf_km_s=v / 1000.0,
            source="truth",
        )
    if reentered:
        return TruthResult(
            norad_id=initial.norad_id,
            status="reentry",
            trajectory=traj,
            stop_t_tt_s=stop_t,
            area_to_mass_m2_per_kg=a2m,
            start_scale=k,
            n_evaluations=n_eval,
            wall_s=time.perf_counter() - wall0,
            message=(
                f"altitude fell below {cfg.reentry_altitude_km} km at "
                f"{(stop_t - t0) / _DAY_S:.3f} d; trajectory holds the {t_keep.size} "
                f"grid points before that"
            ),
            start_fit=record,
        )
    return TruthResult(
        norad_id=initial.norad_id,
        status="ok",
        trajectory=traj,
        area_to_mass_m2_per_kg=a2m,
        start_scale=k,
        n_evaluations=n_eval,
        wall_s=time.perf_counter() - wall0,
        start_fit=record,
    )


# --------------------------------------------------------------------------------------
# Space-weather copy
# --------------------------------------------------------------------------------------


class SpaceWeatherRangeError(ValueError):
    """A window is outside a space-weather copy's checked range or inside the data-end margin."""


@dataclass(frozen=True)
class SpaceWeatherCopy:
    """An Orekit space-weather copy and the window range it was checked for.

    Returned by :func:`write_space_weather_copy`; :func:`propagate_truth` and
    :func:`propagate_batch` take this object, not a path, so every propagation
    can be checked against what the copy was validated for. A copy holds the
    whole published file, with zeros filled in outside the checked range, so a
    window outside it would otherwise run on placeholders or forecasts.

    Attributes:
        path: the file Orekit reads (absolute).
        sha256: SHA-256 of its content when written; Orekit objects are cached
            by it, and a file whose content changed is refused.
        window_first_utc, window_last_utc: first and last UTC day of the
            propagation range the copy was written for (``YYYY-MM-DD``). Every
            date from 3 days before the first to 1 day after the last was
            checked free of placeholders.
        last_fully_observed_utc: the source file's last fully observed day
            (``SpaceWeather.last_fully_observed_date``).
        margin_days: the truth data-end margin applied
            (``truth_data_end_margin_days`` in ``config/splits.toml``).
        first_allowed_tt_s: windows must start at or after this (00:00 UTC of
            ``window_first_utc``).
        last_allowed_tt_s: windows must end at or before this: the earlier of
            00:00 UTC after ``window_last_utc`` and
            ``splits.latest_window_end_tt_s`` (the end of the last fully
            observed day minus the margin).
        n_filled: fields filled with 0 outside the checked range (from the
            writer's report).
    """

    path: str
    sha256: str
    window_first_utc: str
    window_last_utc: str
    last_fully_observed_utc: str
    margin_days: float
    first_allowed_tt_s: float
    last_allowed_tt_s: float
    n_filled: int = field(default=0, compare=False)

    def check_window(self, t_first_tt_s: float, t_last_tt_s: float) -> None:
        """Raise :class:`SpaceWeatherRangeError` unless the window fits this copy."""
        from orbit_correction.frames import tt_seconds_to_utc_isot

        if t_first_tt_s < self.first_allowed_tt_s or t_last_tt_s > self.last_allowed_tt_s:
            raise SpaceWeatherRangeError(
                f"window {tt_seconds_to_utc_isot(t_first_tt_s)} .. "
                f"{tt_seconds_to_utc_isot(t_last_tt_s)} UTC is outside what {self.path} "
                f"allows: {tt_seconds_to_utc_isot(self.first_allowed_tt_s)} .. "
                f"{tt_seconds_to_utc_isot(self.last_allowed_tt_s)} UTC (checked range "
                f"{self.window_first_utc} .. {self.window_last_utc}; last fully observed "
                f"day {self.last_fully_observed_utc} minus a {self.margin_days:g}-day "
                "margin); write a copy for this window"
            )


def _require_copy(space_weather) -> None:
    if not isinstance(space_weather, SpaceWeatherCopy):
        raise TypeError(
            "drag needs the SpaceWeatherCopy returned by write_space_weather_copy, got "
            f"{type(space_weather).__name__}; a bare path carries no checked range"
        )


def write_space_weather_copy(
    source: str | Path | SpaceWeather,
    out_file: str | Path,
    first_t_tt_s: float,
    last_t_tt_s: float,
    *,
    splits_config: str | Path = SPLITS_CONFIG_PATH,
) -> SpaceWeatherCopy:
    """Write the Orekit copy of a CelesTrak space-weather file for a time range.

    Parses ``source`` with :func:`orbit_correction.spaceweather.parse_spaceweather`
    (or takes an already parsed ``SpaceWeather``) and writes it with
    :func:`~orbit_correction.spaceweather.write_orekit_cssi`, requiring every
    date from 3 days before ``first`` to 1 day after ``last`` to be complete (no
    placeholder there). The 3 days cover NRLMSISE-00's ap history (57 hours
    back) and the previous-day F10.7; the 1 day covers Orekit reading the row
    after the current day (its lookups take the two neighbouring rows).

    Also enforces the truth data-end margin: ``last`` must be no later than
    ``splits.latest_window_end_tt_s`` for the source's last fully observed day,
    because Orekit's centred 81-day F10.7 mean at ``last`` reads up to 40 days
    ahead. This holds whatever the split labels say.

    Write each copy to its own path (for example with the range in the name).
    A copy rewritten at the same path gets a new hash and is read again, but a
    unique path keeps old results traceable to their input.

    Returns:
        :class:`SpaceWeatherCopy`, to pass to :func:`propagate_truth` or
        :func:`propagate_batch`.

    Raises:
        SpaceWeatherRangeError: if ``last`` is inside the data-end margin.
        orbit_correction.spaceweather.PlaceholderInRangeError: if a required
            date is missing or has a blank.
    """
    from orbit_correction.frames import tt_seconds_to_utc_isot, utc_to_tt_seconds
    from orbit_correction.spaceweather import (
        SpaceWeather,
        parse_spaceweather,
        write_orekit_cssi,
    )
    from orbit_correction.splits import latest_window_end_tt_s, load_split_config

    if last_t_tt_s < first_t_tt_s:
        raise ValueError("last_t_tt_s must not precede first_t_tt_s")
    sw = source if isinstance(source, SpaceWeather) else parse_spaceweather(source)
    last_obs = sw.last_fully_observed_date()
    scfg = load_split_config(splits_config)
    latest_end = latest_window_end_tt_s(last_obs, scfg)
    if last_t_tt_s > latest_end:
        raise SpaceWeatherRangeError(
            f"window end {tt_seconds_to_utc_isot(last_t_tt_s)} UTC is later than "
            f"{tt_seconds_to_utc_isot(latest_end)} UTC, {scfg.truth_data_end_margin_days:g} "
            f"days before the end of the last fully observed day {last_obs}; truth there "
            "would average forecast F10.7"
        )
    first_day = str(tt_seconds_to_utc_isot(first_t_tt_s))[:10]
    last_day = str(tt_seconds_to_utc_isot(last_t_tt_s))[:10]
    report = write_orekit_cssi(
        sw, out_file, first_day, last_day, margin_before_days=3, margin_after_days=1
    )
    after_last = str(np.datetime64(last_day, "D") + np.timedelta64(1, "D"))
    return SpaceWeatherCopy(
        path=str(Path(out_file).resolve()),
        sha256=_sha256(out_file),
        window_first_utc=first_day,
        window_last_utc=last_day,
        last_fully_observed_utc=last_obs,
        margin_days=float(scfg.truth_data_end_margin_days),
        first_allowed_tt_s=float(utc_to_tt_seconds(f"{first_day}T00:00:00")),
        # strictly before 00:00 of the day after: at that instant Orekit would read
        # the row two days after the last day, which wasn't checked
        last_allowed_tt_s=min(
            float(utc_to_tt_seconds(f"{after_last}T00:00:00")) - 1.0e-3, float(latest_end)
        ),
        n_filled=len(report.filled),
    )


# --------------------------------------------------------------------------------------
# Worker pool
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TruthJob:
    """One propagation for :func:`propagate_batch` (the arguments of :func:`propagate_truth`).

    ``fit_samples`` is required under the fitted start rule and refused under any
    other: the SGP4 states at or before the epoch from :func:`fit_samples_from_elsets`.
    """

    initial: Trajectory
    bstar_per_er: float | None
    t_out_tt_s: np.ndarray = field(repr=False)
    area_to_mass_m2_per_kg: float | None = None
    fit_samples: Trajectory | None = field(default=None, repr=False)


_WORKER: dict[str, object] = {}


def _worker_init(cfg: TruthConfig, space_weather: SpaceWeatherCopy | None, apc: int) -> None:
    """Pool initializer: one JVM per worker, Orekit objects built before the first job."""
    start_jvm(apc)
    ctx = _context(cfg)
    if space_weather is not None and cfg.atmosphere != "none":
        ctx.atmosphere(space_weather)
    _WORKER.update(cfg=cfg, sw=space_weather)


def _worker_run(job: TruthJob) -> TruthResult:
    try:
        return propagate_truth(
            job.initial,
            job.bstar_per_er,
            job.t_out_tt_s,
            _WORKER["cfg"],
            _WORKER["sw"],
            area_to_mass_m2_per_kg=job.area_to_mass_m2_per_kg,
            fit_samples=job.fit_samples,
        )
    except Exception as exc:  # noqa: BLE001  (reported per job, batch continues)
        return TruthResult(
            norad_id=job.initial.norad_id, status="error", message=f"{type(exc).__name__}: {exc}"
        )


def propagate_batch(
    jobs: Sequence[TruthJob],
    cfg: TruthConfig,
    space_weather: SpaceWeatherCopy | None,
    n_workers: int,
    *,
    active_processor_count: int | None = None,
) -> list[TruthResult]:
    """Run many propagations in spawned worker processes, one JVM per worker.

    The calling process never starts a JVM (it only pickles jobs and results),
    so this is safe to call from a process that must stay Java-free. Workers use
    the ``spawn`` start method; each pays the JVM start and Orekit setup once
    (a few seconds) in its initializer.

    Every job's window is checked against the space-weather copy before any
    worker starts (from its first fit sample under the fitted rule); one window
    outside it refuses the whole batch, and so does a job whose ``fit_samples``
    don't match the start rule (missing under the fitted rule, present under
    another, or reaching past the epoch).

    Results come back in job order. A job that raises inside a worker gives
    ``TruthResult(status="error")`` instead of stopping the batch.

    Args:
        jobs: the propagations.
        cfg: configuration, shared by all jobs.
        space_weather: the :class:`SpaceWeatherCopy` covering every job's
            window (None only without drag).
        n_workers: number of worker processes.
        active_processor_count: per-JVM processor cap; defaults to the config's.

    Raises:
        TypeError, SpaceWeatherRangeError, ValueError: as in
            :func:`propagate_truth`, for any job, before anything runs.

    Note:
        Scripts that call this need an ``if __name__ == "__main__":`` guard,
        because ``spawn`` re-imports the main module in each worker.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    fitted = cfg.start_rule == FIT_START_RULE
    for i, job in enumerate(jobs):
        try:
            if fitted:
                _check_fit_samples(job.initial, job.fit_samples)
            elif job.fit_samples is not None:
                raise ValueError(f"fit_samples given but the start rule is {cfg.start_rule!r}")
        except ValueError as exc:
            raise ValueError(f"job {i} (NORAD {job.initial.norad_id}): {exc}") from None
    if cfg.atmosphere != "none":
        _require_copy(space_weather)
        for i, job in enumerate(jobs):
            first = float(job.fit_samples.t_tt_s[0]) if fitted else float(job.initial.t_tt_s[0])
            try:
                space_weather.check_window(first, float(np.asarray(job.t_out_tt_s)[-1]))
            except SpaceWeatherRangeError as exc:
                raise SpaceWeatherRangeError(
                    f"job {i} (NORAD {job.initial.norad_id}): {exc}"
                ) from None
    apc = (
        cfg.jvm_active_processor_count if active_processor_count is None else active_processor_count
    )
    with ProcessPoolExecutor(
        max_workers=int(n_workers),
        mp_context=mp.get_context("spawn"),
        initializer=_worker_init,
        initargs=(cfg, space_weather, int(apc)),
    ) as pool:
        return list(pool.map(_worker_run, jobs, chunksize=1))


__all__ = [
    "CONFIG_PATH",
    "EPOCH_START_RULE",
    "FIT_START_RULE",
    "FIT_STATUSES",
    "START_RULES",
    "StartFit",
    "SpaceWeatherCopy",
    "SpaceWeatherRangeError",
    "TruthConfig",
    "TruthJob",
    "TruthResult",
    "apply_start_rule",
    "area_to_mass_from_bstar",
    "build_propagator",
    "describe_gravity",
    "fit_sample_offsets_s",
    "fit_samples_from_elsets",
    "fit_start_to_sgp4",
    "initial_geodetic_altitude_km",
    "initial_state_from_omm",
    "jvm_started",
    "load_truth_config",
    "orekit_provenance",
    "output_grid",
    "propagate_batch",
    "propagate_truth",
    "start_jvm",
    "start_rule_scale",
    "write_space_weather_copy",
]
