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
The initial state is the SGP4 state at the element-set epoch, rotated TEME ->
GCRF. That is the deliberate, logged conversion from SGP4 mean elements
(CLAUDE.md). SGP4 states are consistent with the WGS-72 GM (398600.8 km^3/s^2);
truth integrates with the EIGEN-6S GM (398600.4415), so the same position and
velocity would have a slightly longer two-body period under truth and drift
behind SGP4 by about 1.15 km per day for that reason alone. The default start
rule, ``preserve_sgp4_mean_motion`` (decided 2026-11-23), removes that
artefact: position and velocity are both multiplied by
``k = (mu_truth / mu_WGS72) ** (1/3)`` (:func:`apply_start_rule`). Under
``mu_truth`` the scaled state has the original state's two-body period under
``mu_WGS72``, the same eccentricity vector and the same orbit plane; it sits
about 2 m from the SGP4 state. ``sgp4_state_direct`` (k = 1) is kept for
comparison. A start fitted to SGP4 is a separate, labeled sensitivity variant
and is not implemented here.

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
#: Start rules accepted in ``[truth.start] rule``.
START_RULES = ("preserve_sgp4_mean_motion", "sgp4_state_direct")

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
    start_rule: str = "preserve_sgp4_mean_motion"
    sgp4_mu_km3_s2: float = 398600.8

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
        if self.jvm_active_processor_count < 0:
            raise ValueError("jvm_active_processor_count must be >= 0 (0 = no cap)")


def load_truth_config(path: str | Path = CONFIG_PATH) -> TruthConfig:
    """Read ``config/truth.toml`` (or another file with the same layout)."""
    with open(path, "rb") as f:
        t = tomllib.load(f)["truth"]
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
        start_rule=t["start"]["rule"],
        sgp4_mu_km3_s2=float(t["start"]["sgp4_mu_km3_s2"]),
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


def initial_state_from_omm(record: dict, epoch_utc: str | None = None) -> tuple[Trajectory, float]:
    """SGP4 state at the element-set epoch, in GCRF, plus the set's B*.

    A stop-gap for this module's tests and scripts: step 3's ``cheap_prop.py``
    is the project's SGP4 wrapper and will supply initial states in production.
    Uses the ``sgp4`` package (OMM fields, WGS72, improved mode) and
    :func:`orbit_correction.frames.teme_to_gcrf`.

    Args:
        record: one OMM record (CelesTrak/Space-Track JSON keys).
        epoch_utc: if given, replaces the record's EPOCH, so the same mean elements
            start at another date. Used to build realistic LEO states at historical
            epochs where the space-weather file is fully observed; such a state is a
            synthetic test case, not that object's real orbit at that date.

    Returns:
        ``(Trajectory(source="sgp4"), bstar_per_er)``; the trajectory has one state.
    """
    from sgp4 import omm
    from sgp4.api import Satrec

    from orbit_correction.frames import satrec_epoch_to_tt_seconds, teme_to_gcrf

    fields = {k: str(v) for k, v in record.items()}
    if epoch_utc is not None:
        # the sgp4 OMM reader wants fractional seconds
        fields["EPOCH"] = epoch_utc if "." in epoch_utc else epoch_utc + ".000000"
    sat = Satrec()
    omm.initialize(sat, fields)
    err, r, v = sat.sgp4_tsince(0.0)
    if err:
        raise ValueError(f"SGP4 error {err} for NORAD {record.get('NORAD_CAT_ID')}")
    t0 = satrec_epoch_to_tt_seconds(sat.jdsatepoch, sat.jdsatepochF)
    r_g, v_g = teme_to_gcrf(np.array(r), np.array(v), t0)
    traj = Trajectory(
        norad_id=int(record["NORAD_CAT_ID"]),
        t_tt_s=np.array([t0]),
        r_gcrf_km=r_g[None, :],
        v_gcrf_km_s=v_g[None, :],
        source="sgp4",
    )
    return traj, float(sat.bstar)


def start_rule_scale(mu_truth_m3_s2: float, cfg: TruthConfig) -> float:
    """The factor k the start rule multiplies the initial position and velocity by.

    ``preserve_sgp4_mean_motion``: ``k = (mu_truth / mu_sgp4) ** (1/3)``. With
    ``r' = k r`` and ``v' = k v``, the vis-viva relation gives ``a' = k a``
    under ``mu_truth = k^3 mu_sgp4``, so the mean motion ``sqrt(mu / a^3)`` is
    unchanged; the eccentricity vector ``(v x h) / mu - r / |r|`` and the plane
    are unchanged too. ``sgp4_state_direct``: k = 1.

    Args:
        mu_truth_m3_s2: GM of the truth gravity field (Orekit's, m^3/s^2).
        cfg: configuration (``start_rule``, ``sgp4_mu_km3_s2``).
    """
    if cfg.start_rule == "sgp4_state_direct":
        return 1.0
    return (float(mu_truth_m3_s2) / (cfg.sgp4_mu_km3_s2 * 1.0e9)) ** (1.0 / 3.0)


def apply_start_rule(initial: Trajectory, mu_truth_m3_s2: float, cfg: TruthConfig) -> Trajectory:
    """The truth initial state: ``initial`` scaled by :func:`start_rule_scale`.

    :func:`propagate_truth` calls this; scripts that build their own Orekit
    propagators (the brahe cross-check) call it to start from the same state.
    Returns ``initial`` itself when k = 1.
    """
    k = start_rule_scale(mu_truth_m3_s2, cfg)
    if k == 1.0:
        return initial
    return Trajectory(
        norad_id=initial.norad_id,
        t_tt_s=initial.t_tt_s.copy(),
        r_gcrf_km=initial.r_gcrf_km * k,
        v_gcrf_km_s=initial.v_gcrf_km_s * k,
        source=initial.source,
    )


def output_grid(t0_tt_s: float, horizon_s: float, step_s: float) -> np.ndarray:
    """A regular output grid ``t0, t0 + step, ..., t0 + horizon`` (TT seconds).

    Built as ``t0 + k * step`` (no accumulated rounding). The last point is
    ``t0 + horizon`` when ``horizon`` is a multiple of ``step``.
    """
    n = int(math.floor(horizon_s / step_s + 1e-9))
    return float(t0_tt_s) + step_s * np.arange(n + 1, dtype=np.float64)


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
            object doesn't stop a batch; ``message`` says why).
        trajectory: ``Trajectory(source="truth")`` on the requested grid (all of
            it for ``ok``, the points at or before ``stop_t_tt_s`` for
            ``reentry``), or None.
        stop_t_tt_s: when the re-entry altitude was crossed (TT seconds), the
            initial epoch for a start below it, else None.
        area_to_mass_m2_per_kg: the A/m used.
        start_scale: the start rule's factor k (1.0 for ``sgp4_state_direct``).
        n_evaluations: right-hand-side evaluations of the integrator.
        wall_s: wall-clock time of the propagation and sampling.
        message: human-readable detail.
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


def propagate_truth(
    initial: Trajectory,
    bstar_per_er: float | None,
    t_out_tt_s: np.ndarray,
    cfg: TruthConfig,
    space_weather: SpaceWeatherCopy | None,
    *,
    area_to_mass_m2_per_kg: float | None = None,
) -> TruthResult:
    """Propagate one object with the truth model and sample it on a time grid.

    Steps: check the window against the space-weather copy; apply the start
    rule (:func:`apply_start_rule`); return ``reentry`` at once if the start is
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

    Returns:
        :class:`TruthResult` with status ``"ok"`` or ``"reentry"``.

    Raises:
        TypeError: if drag is on and ``space_weather`` isn't a
            :class:`SpaceWeatherCopy` (a bare path carries no checked range).
        SpaceWeatherRangeError: if the window isn't inside the copy's checked
            range or ends inside the data-end margin.
        ValueError: on other bad inputs. Orekit exceptions propagate as raised.
    """
    t_out = np.asarray(t_out_tt_s, dtype=np.float64)
    if t_out.ndim != 1 or t_out.size == 0:
        raise ValueError("t_out_tt_s must be a non-empty 1-D array")
    if t_out.size > 1 and not np.all(np.diff(t_out) > 0):
        raise ValueError("t_out_tt_s must be strictly increasing")
    t0 = float(initial.t_tt_s[0])
    if t_out[0] < t0:
        raise ValueError("output times must not precede the initial epoch")

    a2m = None
    if cfg.atmosphere != "none":
        _require_copy(space_weather)
        space_weather.check_window(t0, float(t_out[-1]))
        a2m = (
            float(area_to_mass_m2_per_kg)
            if area_to_mass_m2_per_kg is not None
            else area_to_mass_from_bstar(float(bstar_per_er), cfg)
        )

    wall0 = time.perf_counter()
    ctx = _context(cfg)
    k = start_rule_scale(ctx.mu, cfg)
    start = apply_start_rule(initial, ctx.mu, cfg)

    alt0 = initial_geodetic_altitude_km(cfg, start)
    if alt0 < cfg.reentry_altitude_km:
        return TruthResult(
            norad_id=initial.norad_id,
            status="reentry",
            stop_t_tt_s=t0,
            area_to_mass_m2_per_kg=a2m,
            start_scale=k,
            n_evaluations=0,
            wall_s=time.perf_counter() - wall0,
            message=(
                f"initial geodetic altitude {alt0:.1f} km is below "
                f"{cfg.reentry_altitude_km} km; not propagated"
            ),
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
        )
    return TruthResult(
        norad_id=initial.norad_id,
        status="ok",
        trajectory=traj,
        area_to_mass_m2_per_kg=a2m,
        start_scale=k,
        n_evaluations=n_eval,
        wall_s=time.perf_counter() - wall0,
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
    """One propagation for :func:`propagate_batch` (the arguments of :func:`propagate_truth`)."""

    initial: Trajectory
    bstar_per_er: float | None
    t_out_tt_s: np.ndarray = field(repr=False)
    area_to_mass_m2_per_kg: float | None = None


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
    worker starts; one window outside it refuses the whole batch.

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
        TypeError, SpaceWeatherRangeError: as in :func:`propagate_truth`, for
            any job, before anything runs.

    Note:
        Scripts that call this need an ``if __name__ == "__main__":`` guard,
        because ``spawn`` re-imports the main module in each worker.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    if cfg.atmosphere != "none":
        _require_copy(space_weather)
        for i, job in enumerate(jobs):
            try:
                space_weather.check_window(
                    float(job.initial.t_tt_s[0]), float(np.asarray(job.t_out_tt_s)[-1])
                )
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
    "START_RULES",
    "SpaceWeatherCopy",
    "SpaceWeatherRangeError",
    "TruthConfig",
    "TruthJob",
    "TruthResult",
    "apply_start_rule",
    "area_to_mass_from_bstar",
    "build_propagator",
    "describe_gravity",
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
