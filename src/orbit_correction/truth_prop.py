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
  the GCRF/TT/km conventions at the boundary; the rule that Orekit reads only the
  space-weather copy written by :func:`orbit_correction.spaceweather.write_orekit_cssi`
  (never CelesTrak's file, never the bundle's CSSI file); the re-entry status;
  the JVM rules and the worker pool.

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
   with Java.
3. Each JVM is capped with ``-XX:ActiveProcessorCount`` (config, default 2) so
   parallel workers don't each size their thread pools for the whole machine.

Initial state
-------------
The initial state is the SGP4 state at the element-set epoch, rotated TEME ->
GCRF, used directly as an osculating Cartesian state. That is the deliberate,
logged conversion from SGP4 mean elements (CLAUDE.md); a start fitted to SGP4
is a separate, labeled sensitivity variant and is not implemented here.
"""

from __future__ import annotations

import math
import os
import sys
import time
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from orbit_correction.types import Trajectory

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "config" / "truth.toml"

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
        ImportError: if the ``datagen`` dependency group isn't installed.
    """
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

    def space_weather(self, path: str | Path):
        """CssiSpaceWeatherData reading exactly this file (our written copy)."""
        key = str(Path(path).resolve())
        if key not in self._sw:
            from java.io import File
            from org.orekit.data import DataSource
            from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData

            if not Path(key).is_file():
                raise FileNotFoundError(key)
            self._sw[key] = CssiSpaceWeatherData(DataSource(File(key)), self.utc)
        return self._sw[key]

    def atmosphere(self, path: str | Path):
        """NRLMSISE-00 fed by our space-weather copy, with the configured switches."""
        key = str(Path(path).resolve())
        if key not in self._atm:
            from org.orekit.models.earth.atmosphere import NRLMSISE00

            atm = NRLMSISE00(self.space_weather(key), self.sun, self.earth, self.ut1)
            atm = atm.withLocalSolarTimeMode(
                getattr(NRLMSISE00.LocalSolarTimeMode, self.cfg.msis_local_solar_time)
            )
            atm = atm.withSwitch(9, int(self.cfg.msis_ap_switch))
            self._atm[key] = atm
        return self._atm[key]


_CONTEXTS: dict[TruthConfig, _Orekit] = {}


def _context(cfg: TruthConfig) -> _Orekit:
    """The process's Orekit objects for ``cfg``; starts the JVM if needed."""
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
            window; no trajectory is returned, not even a partial one) or
            ``"error"`` (only from :func:`propagate_batch`, which catches per-job
            exceptions so one bad object doesn't stop a batch; ``message`` says why).
        trajectory: ``Trajectory(source="truth")`` on the requested grid, or None.
        stop_t_tt_s: when the re-entry altitude was crossed (TT seconds), else None.
        area_to_mass_m2_per_kg: the A/m used.
        n_evaluations: right-hand-side evaluations of the integrator.
        wall_s: wall-clock time of the propagation and sampling.
        message: human-readable detail.
    """

    norad_id: int
    status: str
    trajectory: Trajectory | None = None
    stop_t_tt_s: float | None = None
    area_to_mass_m2_per_kg: float | None = None
    n_evaluations: int | None = None
    wall_s: float | None = None
    message: str = ""


def build_propagator(
    cfg: TruthConfig,
    initial: Trajectory,
    area_to_mass_m2_per_kg: float | None,
    space_weather_file: str | Path | None,
    *,
    atmosphere=None,
    gravity_provider=None,
):
    """Configured Orekit ``NumericalPropagator`` for one initial state.

    Exposed for tests and scripts that need Orekit objects; normal callers use
    :func:`propagate_truth`. Returns ``(propagator, integrator, context)``.

    Args:
        cfg: configuration.
        initial: one-state trajectory (GCRF, km, km/s, TT seconds).
        area_to_mass_m2_per_kg: A/m for drag; ignored when ``cfg.atmosphere`` is
            ``"none"``.
        space_weather_file: our Orekit copy (``spaceweather.write_orekit_cssi``);
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
            if space_weather_file is None:
                raise ValueError("NRLMSISE-00 needs space_weather_file (our Orekit copy)")
            atmosphere = ctx.atmosphere(space_weather_file)
        area = float(area_to_mass_m2_per_kg) * cfg.mass_kg
        prop.addForceModel(DragForce(atmosphere, IsotropicDrag(area, float(cfg.cd))))
    prop.setInitialState(SpacecraftState(orbit).withMass(float(cfg.mass_kg)))
    return prop, integ, ctx


def propagate_truth(
    initial: Trajectory,
    bstar_per_er: float | None,
    t_out_tt_s: np.ndarray,
    cfg: TruthConfig,
    space_weather_file: str | Path | None,
    *,
    area_to_mass_m2_per_kg: float | None = None,
) -> TruthResult:
    """Propagate one object with the truth model and sample it on a time grid.

    One integration from the initial epoch to the last grid time, with Orekit's
    ephemeris generator recording the dense output; positions and velocities are
    then read from that ephemeris at each grid time. Never one ``propagate()``
    call per output time (that restarts the integrator at every point).

    Starts a JVM in this process if none is running. For batches use
    :func:`propagate_batch`, which keeps the JVM out of the calling process.

    Args:
        initial: one-state ``Trajectory`` at the element-set epoch (the SGP4
            state rotated to GCRF).
        bstar_per_er: the element set's B* (1/ER). Converted to A/m with
            :func:`area_to_mass_from_bstar`. Ignored if ``area_to_mass_m2_per_kg``
            is given or ``cfg.atmosphere == "none"``.
        t_out_tt_s: output times, TT seconds since J2000.0, strictly increasing,
            none before the initial epoch.
        cfg: configuration (``load_truth_config()``).
        space_weather_file: the copy written by
            :func:`orbit_correction.spaceweather.write_orekit_cssi` (or
            :func:`write_space_weather_copy`) covering the window. Orekit reads
            this file and nothing else for space weather.
        area_to_mass_m2_per_kg: explicit A/m, overriding B* (for objects whose
            B* has no physical meaning, such as negative B*).

    Returns:
        :class:`TruthResult` with status ``"ok"`` or ``"reentry"``.

    Raises:
        ValueError: on bad inputs; Orekit exceptions (for example a date outside
            the space-weather copy) propagate as raised.
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
        a2m = (
            float(area_to_mass_m2_per_kg)
            if area_to_mass_m2_per_kg is not None
            else area_to_mass_from_bstar(float(bstar_per_er), cfg)
        )

    wall0 = time.perf_counter()
    prop, integ, ctx = build_propagator(cfg, initial, a2m, space_weather_file)

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
    if final.getDate().durationFrom(end_date) < -1.0e-6:
        return TruthResult(
            norad_id=initial.norad_id,
            status="reentry",
            stop_t_tt_s=stop_t,
            area_to_mass_m2_per_kg=a2m,
            n_evaluations=n_eval,
            wall_s=time.perf_counter() - wall0,
            message=(
                f"altitude fell below {cfg.reentry_altitude_km} km at "
                f"{(stop_t - t0) / _DAY_S:.3f} d; no trajectory returned"
            ),
        )

    ephemeris = generator.getGeneratedEphemeris()
    r = np.empty((t_out.size, 3))
    v = np.empty((t_out.size, 3))
    gcrf = ctx.gcrf
    for k, t in enumerate(t_out):
        pv = ephemeris.getPVCoordinates(ctx.date(t), gcrf)
        p, w = pv.getPosition(), pv.getVelocity()
        r[k] = (p.getX(), p.getY(), p.getZ())
        v[k] = (w.getX(), w.getY(), w.getZ())
    traj = Trajectory(
        norad_id=initial.norad_id,
        t_tt_s=t_out.copy(),
        r_gcrf_km=r / 1000.0,
        v_gcrf_km_s=v / 1000.0,
        source="truth",
    )
    return TruthResult(
        norad_id=initial.norad_id,
        status="ok",
        trajectory=traj,
        area_to_mass_m2_per_kg=a2m,
        n_evaluations=n_eval,
        wall_s=time.perf_counter() - wall0,
    )


# --------------------------------------------------------------------------------------
# Space-weather copy
# --------------------------------------------------------------------------------------


def write_space_weather_copy(
    source_file: str | Path, out_file: str | Path, first_t_tt_s: float, last_t_tt_s: float
):
    """Write the Orekit copy of a CelesTrak space-weather file for a time range.

    Parses ``source_file`` with :func:`orbit_correction.spaceweather.parse_spaceweather`
    and writes it with :func:`~orbit_correction.spaceweather.write_orekit_cssi`,
    requiring every date from 3 days before ``first`` to 1 day after ``last`` to be
    complete (no placeholder there). The 3 days cover NRLMSISE-00's ap history
    (57 hours back) and the previous-day F10.7; the 1 day covers Orekit reading
    the row after the current day (its lookups take the two neighbouring rows).

    Returns the writer's ``WriteReport``.
    """
    from orbit_correction.frames import tt_seconds_to_utc_isot
    from orbit_correction.spaceweather import parse_spaceweather, write_orekit_cssi

    sw = parse_spaceweather(source_file)
    return write_orekit_cssi(
        sw,
        out_file,
        str(tt_seconds_to_utc_isot(first_t_tt_s))[:10],
        str(tt_seconds_to_utc_isot(last_t_tt_s))[:10],
        margin_before_days=3,
        margin_after_days=1,
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


def _worker_init(cfg: TruthConfig, space_weather_file: str | None, apc: int) -> None:
    """Pool initializer: one JVM per worker, Orekit objects built before the first job."""
    start_jvm(apc)
    ctx = _context(cfg)
    if space_weather_file is not None and cfg.atmosphere != "none":
        ctx.atmosphere(space_weather_file)
    _WORKER.update(cfg=cfg, sw=space_weather_file)


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
    space_weather_file: str | Path | None,
    n_workers: int,
    *,
    active_processor_count: int | None = None,
) -> list[TruthResult]:
    """Run many propagations in spawned worker processes, one JVM per worker.

    The calling process never starts a JVM (it only pickles jobs and results),
    so this is safe to call from a process that must stay Java-free. Workers use
    the ``spawn`` start method; each pays the JVM start and Orekit setup once
    (a few seconds) in its initializer.

    Results come back in job order. A job that raises gives
    ``TruthResult(status="error")`` instead of stopping the batch.

    Args:
        jobs: the propagations.
        cfg: configuration, shared by all jobs.
        space_weather_file: our Orekit copy covering every job's window.
        n_workers: number of worker processes.
        active_processor_count: per-JVM processor cap; defaults to the config's.

    Note:
        Scripts that call this need an ``if __name__ == "__main__":`` guard,
        because ``spawn`` re-imports the main module in each worker.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    apc = (
        cfg.jvm_active_processor_count if active_processor_count is None else active_processor_count
    )
    sw = None if space_weather_file is None else str(Path(space_weather_file).resolve())
    with ProcessPoolExecutor(
        max_workers=int(n_workers),
        mp_context=mp.get_context("spawn"),
        initializer=_worker_init,
        initargs=(cfg, sw, int(apc)),
    ) as pool:
        return list(pool.map(_worker_run, jobs, chunksize=1))


__all__ = [
    "CONFIG_PATH",
    "TruthConfig",
    "TruthJob",
    "TruthResult",
    "area_to_mass_from_bstar",
    "build_propagator",
    "initial_state_from_omm",
    "jvm_started",
    "load_truth_config",
    "orekit_provenance",
    "output_grid",
    "propagate_batch",
    "propagate_truth",
    "start_jvm",
    "write_space_weather_copy",
]
