# Methodology

Skeleton. Sections fill in as components land; the planning package's locked decisions are
summarised here once the README results exist.

## Question

How much does a learned correction reduce SGP4's position error against a high-fidelity
numerical propagator, and do corrected predictions screen LEO conjunctions better than
uncorrected SGP4 and classical geometric filters? Truth is a model (a numerical propagator),
not tracked positions.

## Data

- Element sets: CelesTrak GP (development, tests) and Space-Track `gp_history` (training and
  test epochs; validation cases). Loading and filtering: `src/orbit_correction/catalog.py`.
- Space weather: CelesTrak `SW-All.txt`, read by `src/orbit_correction/spaceweather.py`.
- Filters and their values: `config/catalog_filters.toml` (approved 2026-11-23). LEO
  (perigee >= 250 km, apogee <= 2000 km), 0 < B* <= 0.01 /ER, SGP4 element sets only,
  payloads, rocket bodies and debris. Non-maneuvering: operational payloads (current SATCAT
  status) and maneuvering constellations by name are dropped, and so is any object whose
  element history shows a mean semimajor-axis rise, a drop larger than drag explains, or an
  inclination step. Each rule's rejections are reported by name.

## Model inputs

The headline model sees the SGP4 element set (mean elements, B*, mean-motion derivative), the
time since the start, and exactly four space-weather values known at the start (approved
2026-11-23): the last complete day's observed F10.7, the trailing 81-day mean of observed
F10.7, the last complete day's daily Ap, and the last completed 3-hour ap. A day counts as
known from 00:00 UTC the next day and a 3-hour ap value from the end of its interval, with no
extra lag; only observed rows are read. Space weather observed over the prediction horizon
(`obs_*` columns) is optional and used only for a labeled comparison of how much perfect
space-weather foresight would add.

## Conventions

- Frame GCRF, time TT seconds since J2000.0, km and km/s (`src/orbit_correction/types.py`).
- SGP4 TEME output converted with astropy (`src/orbit_correction/frames.py`); EOP pinned via
  `astropy-iers-data`.
- Residual targets: curvilinear RTN, cheap minus truth, in the frame of the SGP4 state
  (definition in `types.py`).

## Splits

By object and by time (`src/orbit_correction/splits.py`, `config/splits.toml`, approved
2026-11-23):

- Objects: a seeded hash per NORAD ID (seed 20261123) holds out 20% of objects; 15% of the
  rest form a development set. The ten validation-case objects are always held out.
- Time: training and development windows (start plus 7 days) end before 2025-01-01T00:00:00
  UTC; test rows start at or after it.
- Training and development windows overlapping [TCA - 7 d, TCA + 1 d] of a validation event
  are excluded.
- Windows of any split that end later than 41 days before the end of the last fully observed
  space-weather day are excluded: beyond that day the file holds only forecasts, and the truth
  propagator's centred 81-day F10.7 mean reaches 40 days ahead (`truth_data_end_margin_days`).

Row labels: `train` (fitted on), `dev` (development evaluation), `test` (held-out evaluation,
unexamined until a confirmatory run), `reserved_object`, `reserved_time` and `excluded`.
`reserved_object` rows (held-out objects before the cutoff) and `reserved_time` rows
(training objects after the cutoff) enter neither training nor evaluation: each is held out
on only one axis, so training on them would leak held-out objects or epochs, and evaluating on
them would mix partly seen data into the held-out result.

## Cheap propagator (SGP4)

`src/orbit_correction/cheap_prop.py`, a wrapper of the `sgp4` package (Vallado's C++ SGP4/SDP4
through its Python bindings; the theory and the propagation code are the library's).

- Constants and mode: WGS-72 and operation mode `'i'`, the constants element sets are fitted
  with and the mode of Vallado et al. (2006)'s verification output. The wrapper reproduces
  that output (`tcppver.out`, 33 cases, 2e-7 km and km/s, the package's own tolerance) and the
  error codes of its seven failing cases.
- Initialisation: the element-set fields are converted to SGP4's units exactly as the
  library's TLE reader converts them and passed to `Satrec.sgp4init`, so an OMM element set
  and the same TLE give the same propagation.
- Time: SGP4's time since epoch is `(t_TT - epoch_TT) / 60` minutes, with the epoch converted
  from its UTC string by astropy, so elapsed time is physical across leap seconds. Output
  trajectories are on a TT grid: one shared absolute grid (screening) or offsets from each
  element set's epoch (data generation).
- Errors: a trajectory keeps the states before SGP4's first error flag (1-4 invalid or
  out-of-range elements, 6 decayed: radius under one Earth radius) and nothing after; each
  object carries a status. Truncated and empty propagations are counted, not dropped
  silently. SGP4's decay flag only fires at one Earth radius, so states just before it can be
  far below any physical altitude.
- Frames: TEME -> GCRF with the exact astropy chain or an interpolated rotation
  (`frames.teme_to_gcrf_fast`, the default in the wrapper): exact rotations at nodes every
  6 h in TT, degree-5 Lagrange interpolation. Against the exact path over 7 days at 60 s it
  agrees to under a micrometre and a micrometre per second.
- Truth initial state: the SGP4 state at the element-set epoch, rotated to GCRF exactly
  (`sgp4_epoch_states`). Treating that SGP4 state as an osculating state is the project's
  deliberate conversion from mean elements.

## Truth propagation

Truth is a numerical model, not tracked positions (`src/orbit_correction/truth_prop.py`,
configuration in `config/truth.toml`). The settings below define it; they were decided on
2026-11-23 (decision 1 plus the truth-definition decisions after review).

- **Library behaviour:** Orekit 13.1.9 (through orekit-jpype) does the integration (DOP853),
  gravity (Holmes-Featherstone with EIGEN-6S from the pinned orekit-data bundle), the
  NRLMSISE-00 atmosphere and drag force, Earth orientation (IERS 2010, bundle EOP), the Sun
  (DE-440), the ephemeris output and the altitude event.
- **Project code:** the configuration and its checks; the B* to area-to-mass relation
  (A/m = 2 B* / (Cd rho0), Cd 2.2, rho0 = 0.15696615 kg/(m^2 ER)); the start rule; the
  GCRF/TT/km boundary; reading space weather only from the copy `truth_prop.write_space_weather_copy`
  writes, and refusing windows outside the range that copy was checked for or inside the
  data-end margin; the re-entry status and initial-altitude guard; the JVM rules and the
  spawned worker pool.
- **Force model:** central body plus EIGEN-6S zonal harmonics (order 0) to degree 36;
  NRLMSISE-00 drag with mean local solar time and the 3-hourly ap array (switch 9 = -1); a
  cannonball with Cd 2.2. No tesserals, Sun or Moon gravity, radiation pressure, tides or
  relativity; truth is "zonal plus drag" by design.
- **Zonal degree 36 is a chosen truncation**, part of the truth definition, not a converged
  value. Against a degree-90 reference the omitted zonals move a 7-day trajectory by up to
  about 70 m at 300 km, 18 m at 500 km, 8 m at 800 km and 1 m at 1,200 km
  (`scripts/zonal_convergence.py`; background, see the changelog).
- **NRLMSISE-00 conventions.** Mean local solar time (UT + longitude/15) is what pymsis and
  brahe compute, which makes the density cross-checks possible (pymsis to 1.1e-5). The
  reference notes name the input local apparent solar time and call equation-of-time
  departures of minor importance; along real orbits apparent and mean differ in density by up
  to 6.8% (800 km, November 2024). Switch 9 = -1 feeds the 3-hourly ap history; daily Ap
  (+1) differs by up to 7.8% along an orbit and about 20% at storm points, up to 11 km at
  300 km over 7 days. Switch 9 stays configurable for a labeled daily-Ap sensitivity
  subsample in residual generation.
- **Initial state and start rule:** the SGP4 state at the element-set epoch, rotated TEME to
  GCRF by `frames.teme_to_gcrf`, used as an osculating state, then scaled by
  k = (mu_truth / mu_WGS72)^(1/3) in position and velocity (`preserve_sgp4_mean_motion`).
  SGP4 states are consistent with the WGS-72 GM (398600.8 km^3/s^2) and truth integrates
  with EIGEN-6S's (398600.4415); without the rule the GM difference alone makes truth lag
  SGP4 by about 1.15 km per day, an artefact a model would learn trivially and that tracked
  orbits wouldn't have. The scaled state keeps the two-body period (each under its own GM),
  eccentricity and plane, and sits about 2 m from the SGP4 state, so residuals at t0 are
  about 2 m, not zero. On eight sample objects (300-1,200 km, June 2024 start;
  `scripts/truth_settings_check.py`) the rule moves the 7-day SGP4-minus-truth along-track
  difference by about 8 km in every band. What remains at 7 days is the rest of the model
  difference: 170-970 km at 300 km and 35-86 km at 500 km, where drag dominates, and
  3-15 km at 800 and 1,200 km, where drag is small and the zonal model, SGP4's theory and
  the osculating start account for it (not separated here).
  A start fitted to SGP4 is a later, labeled sensitivity variant.
- **Integrator:** DOP853, 1 m position tolerance, 180 s maximum step. At this cap DOP853 runs
  at the step limit, so the step sets the integration error. Against a converged run (1 mm, 30 s) its
  7-day error on the eight sample objects is 0.8-7.3 m at 300 km, 0.12-0.18 m at 500 km,
  0.13-0.15 m at 800 km and 0.06 m at 1,200 km (at the earlier 300 s step: 92-195, 53, 17-22
  and 8 m), for 50,447 right-hand-side evaluations per object-week instead of 30,302. This is part
  of the truth definition and is reported as a limit.
- **Reproducibility:** the same job gives the same trajectory to about a millimetre, not bit
  for bit: Orekit's interpolated frame transforms depend on the first date a JVM requests, so
  job order within a worker can change a trajectory by ~1e-4 m over days. At 300 km with drag,
  runs at a 120 s maximum step and below stop converging at about 1-18 m over 7 days for a
  high-drag object (cause not identified); that is the reproducibility floor there.
- **Space weather:** Orekit's NRLMSISE-00 input is the previous day's observed F10.7, the
  centred 81-day observed mean for the day containing each instant, and the Ap history
  (Orekit source, `CssiSpaceWeatherData`). Windows therefore end at least 41 days before
  the end of the last fully observed space-weather day (see Splits). The rule is enforced
  twice: in the split labels and in `truth_prop`, which refuses to write a copy for, or
  propagate, a window ending inside the margin, whatever the labels say. Each copy records
  the dates it was checked for; a window outside them is refused, because outside them the
  copy holds forecasts or zero placeholders.
- **Drag in the residual.** For an object whose A/m comes from B*, Cd has no effect: the
  drag acceleration depends on Cd A/m = 2 B* / rho0 whatever Cd is (Cd matters only when A/m
  is given directly). Truth drag is then SGP4's own ballistic coefficient flown through
  NRLMSISE-00, so the drag part of the residual is largely the ratio of NRLMSISE-00 density
  to the density SGP4's theory assumes. That ratio depends on altitude and solar activity: it
  runs from about 1.6 (300 km, November 2024) to 0.01 (650 km and above, 2019). The model
  learns this function of altitude and F10.7 as much as anything about real drag.
- **Re-entry:** a window in which the geodetic altitude falls below 120 km returns status
  `reentry` with the truth trajectory up to the crossing, so close approaches before
  re-entry stay usable. A start below 120 km returns `reentry` at once, without propagating.
  The matching floor for SGP4 belongs to the cheap propagator.
- **Checks** (`tests/test_truth_prop.py`): two-body energy and angular momentum over 7 days;
  J2 node, perigee and mean-anomaly rates against first-order theory over 12 days; zonal
  truncation (no tesseral terms; degree 4 equal to closed-form J2-J4 to 1e-13); the
  configured degree, order 0 and Earth-fixed frame on the built force model; drag work
  equal to the energy lost; circular decay under an exponential atmosphere against the
  analytic rate (6e-6); NRLMSISE-00 against pymsis (msis00) to 1.1e-5; Orekit's
  space-weather values against `spaceweather.observed`; Orekit's TEME to GCRF against
  `frames.teme_to_gcrf` (0.5-1.1 m); the start rule's period in a two-body run; the
  integration error of the truth setting per case; the range and margin guards; the
  re-entry outputs; reproducibility to 1 mm across workers and job order. A library check
  compares Orekit's TLEPropagator with the sgp4 package (9 micrometres); the project doesn't
  use the former.
- **Independent check:** brahe 1.9.0 with a shared static zonal set (`scripts/crosscheck_brahe.py`).
  Without drag the two agree to millimetres at 7 days (tolerance 0.1 m). With drag they
  agree once both use the same NRLMSISE-00 conventions (daily Ap, mean solar time), within
  per-band tolerances of 10 m at 300 km, 1 m at 500 km and 0.1 m at 800 km and above.
  Rerun on all eight sample objects with the start rule and the 180 s setting
  (2026-11-23): no drag at most 5.1 mm; drag with matched conventions at most 4.1 m at
  300 km, 0.058 m at 500 km, 4.8 mm at 800 km and 1.8 mm at 1,200 km. With the project's
  ap-array setting brahe differs by up to 11 km at 300 km, a convention difference.

## Residual dataset

_To be written (build step 3)._

## Models

_To be written (build steps 4 and 7)._

## Screening and metrics

Engine, classical filters and metric functions: `src/orbit_correction/screening.py`, tested
on synthetic encounters with known answers (`tests/test_screening.py`). The screening
comparison itself (truth vs SGP4 vs corrected vs classical filters) has not been run yet.

**Event.** A local minimum of the range between two objects, with miss distance within the
threshold at its TCA and TCA inside the window. A pair can have several events; a pair
inside the threshold at the window edge with its range still falling has none there. This
is the Aerospace / Department of Commerce test-set definition. Pairs are unordered.

**Threshold.** Truth screens use the truth threshold (5 km headline; 1 and 10 km
sensitivity). Cheap screens use truth threshold + pad(tau), with tau the time from the
screening epoch to TCA. The engine takes the threshold as a function of tau, through a
two-method volume interface that an RTN volume could implement later without changing the
event table. The pad rule is open.

**Engine.** Cubic Hermite interpolation of each trajectory's 60 s states. k-d tree candidate
search at two sub-samples per interval, with search radius = local threshold + V h/4 + 1 m.
V bounds the relative speed on the intervals next to the sub-sample. It comes from the
interpolant, not from orbital physics: each object's interpolated velocity on an interval
is a quadratic Bezier curve with control points v_k, 3 (r_k+1 - r_k)/h - v_k - v_k+1 and
v_k+1, so its speed is at most the largest of their norms, and V is the sum of the two
largest such bounds over all objects (about 15 km/s in LEO). The bound holds for any state
sequence, including states whose velocities don't match their positions. Any minimum is
within h/4 of a sub-sample and a pair closes at most V h/4 in that time, so no
between-sample minimum of the interpolated motion under the threshold is skipped, at any
relative speed. The threshold must not peak between sub-samples (constants, ramps and steps
are safe); a narrow bump between two sub-samples can lose the event under it. Each
candidate interval gets a convex-hull lower bound on its range; survivors have their range
minima located from sign changes of the degree-5 range-rate polynomial (16-point bracket,
bisection). Accuracy at 60 s sampling: straight-line encounters exact; 40 random LEO
encounters (e <= 0.05) within 7 us in TCA and 2.6 cm in miss distance of a dense
minimisation of the closed-form range. Limit: a range minimum within 3.75 s of a range
maximum (near-zero relative speed) can be missed.

An independent review compared the engine with a dense brute-force screen (1 s sampling,
every pair, no k-d tree or speed bound) on 10 synthetic catalogs and 10 threshold shapes:
111,322 of 111,322 events matched, none unmatched on either side. After the change to the
speed bound, three of those catalogs were rerun (8,233 events, all matched) and the event
tables were unchanged. A smaller version of the check (40 objects, 1 day, ramped threshold)
is a committed test.

**Classical filters** (Hoots, Crawford and Roehrich 1984, as described by Woodburn,
Coppola and Stoner 2009 and Alfano and Finkleman 2013; the 1984 paper itself was not
read). All work from orbital elements, with a distance pad each:

- Apogee/perigee: prune when the radial shells are further apart than threshold + pad.
  Perigee and apogee from elements (one or more epochs) or from the sampled radius of each
  trajectory.
- Orbit path: prune when the minimum distance between the orbit ellipses exceeds
  threshold + pad at every sample time (Newton iteration from the two relative-node
  points). Pairs are kept without a distance when they are near-coplanar (under 0.057 deg)
  or when sin I < 1.5 (e1 + e2). In that regime Newton from the nodes can settle on a
  local minimum and overstate the distance by hundreds of km. In a sweep against a global
  search over about 5,700 LEO pairs it did so only below sin I / (e1 + e2) = 0.83.
- Time filter: prune when the times each object spends within threshold + pad of the
  other's orbit plane, near the same relative node, never overlap. Windows come from
  Kepler's equation with J2 secular rates. The geometry is recomputed from the same
  elements every hour and held fixed in between (it is not checked against the
  trajectory), and a time pad widens the windows.

What the filters lose when the elements describe the motion exactly:

- With pure two-body motion, none of the three loses an encounter at zero pad.
- With J2 secular drift, over 7 days, the orbit-path filter (sampled at the start, middle
  and end) and the time filter (3,600 s segments) both lose a few encounters at zero pad,
  because the geometry moves between samples or within a segment. The review found, for
  example, 1 of 33 events at 5 km on this project's generator for each, and 16 of 67
  (orbit path) and 7 of 67 (time filter) at 5 km on a 60 km thick shell with e <= 0.02
  (rerun with the current code: same counts). Some orbit-path losses remain at a 30 km
  pad; the time-filter losses went away with 2-10 km of distance pad or a 10 s time pad.
- The committed test of exact elements covers 1 day only; it is a check of the
  geometry, not a lossless guarantee.

Misses from what the elements leave out come on top of these. The tests document one per
filter (drag decay for apogee/perigee and orbit path, drag-driven timing drift for the
time filter). Pads are an open decision.

**Metrics.** Event recall (reference events matched one to one by a candidate event of the
same pair with |dTCA| <= a tolerance) and pair recall (reference pairs with at least one
matched event); false alarms per event and per pair; recall for reference events with
TCA within 1, 3 and 7 days; for filters, event and pair recall (an event survives if its
pair does) and the fraction of pairs pruned; wall-clock cost of each run. The headline
counting rule and the TCA tolerance are open decisions.

**Scaling** (synthetic LEO catalog, two-body plus J2 secular motion, 7 days at 60 s, one
process, Intel Core Ultra 9 275HX, 24 logical CPUs; `scripts/benchmark_screening.py`). Run
with the control-polygon speed bound on a nearly idle machine (1-minute load 0.1-0.9):

| Objects | Threshold | Engine wall time | Peak memory | Events |
|---|---|---|---|---|
| 500 | 5 km | 5.3 s | 657 MiB | 103 |
| 500 | 5 km + (1 + 6 km/day x tau) | 5.3 s | 658 MiB | 3,722 |
| 2,000 | 5 km | 28.0 s | 1,475 MiB (1,077 MiB of it the trajectories) | 1,702 |
| 2,000 | 5 km + (1 + 6 km/day x tau) | 28.3 s | 1,474 MiB | 57,938 |

The engine is single-threaded (CPU time about equal to wall time). Event counts match the
runs with the previous speed bound, which took 4-5 s and 31-49 s with other jobs running
(1-minute load 5-13); the new bound lists 9% fewer candidate intervals at 2,000 objects.

At 2,000 objects the classical filters as implemented (numpy) took 0.1-0.4 s (apogee/
perigee), 42-71 s (orbit path, on the pairs apogee/perigee kept) and 123-150 s (time filter,
on the pairs orbit path kept), longer than the engine itself. Their compute cost reflects
this implementation, not an optimised one. Engine cost grows roughly as n^2 at this size;
trajectory memory is 0.565 MB (0.538 MiB) per object per 7-day set, time arrays included
(1.13 GB for 2,000 objects).

Triangulation against the public Aerospace / Department of Commerce screening test set
(Auman, Murphy and George, 2026; CC0) is pending: the files are on Google Drive behind an
access request, and the ephemerides are a 20.7 GB archive of OCM files.

Implemented here vs library: the k-d tree is `scipy.spatial.cKDTree` and tables are
pyarrow. Interpolation, search bound, minimum finding, element conversions, filters and
metrics are this project's code, written with AI assistance and checked by the tests above.

## Limitations

_To be written._
