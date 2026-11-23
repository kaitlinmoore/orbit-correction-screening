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
- Windows of any split that end after the last fully observed space-weather day are excluded:
  beyond it the space-weather file holds only forecasts.

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

_To be written (build step 2)._

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
search at two sub-samples per interval, with search radius = local threshold + V h/4 + 1 m,
where V = 2 x (largest sampled speed + mu / r_min^2 x h/2), about 16 km/s in LEO. Any
minimum is within h/4 of a sub-sample and a pair closes at most V h/4 in that time, so no
between-sample minimum under the threshold is skipped at any relative speed up to V. Each
candidate interval gets a convex-hull lower bound on its range; survivors have their range
minima located from sign changes of the degree-5 range-rate polynomial (16-point bracket,
bisection). Accuracy at 60 s sampling: straight-line encounters exact; 40 random LEO
encounters (e <= 0.05) within 7 us in TCA and 2.6 cm in miss distance of a dense
minimisation of the closed-form range. Limit: a range minimum within 3.75 s of a range
maximum (near-zero relative speed) can be missed.

**Classical filters** (Hoots, Crawford and Roehrich 1984, as described by Woodburn,
Coppola and Stoner 2009 and Alfano and Finkleman 2013; the 1984 paper itself was not
read). All work from orbital elements, with a distance pad each:

- Apogee/perigee: prune when the radial shells are further apart than threshold + pad.
  Perigee and apogee from elements (one or more epochs) or from the sampled radius of each
  trajectory.
- Orbit path: prune when the minimum distance between the orbit ellipses exceeds
  threshold + pad at every sample time (Newton iteration from the two relative-node
  points); near-coplanar pairs are kept.
- Time filter: prune when the times each object spends within threshold + pad of the
  other's orbit plane, near the same relative node, never overlap. Windows come from
  Kepler's equation with J2 secular rates, geometry is recomputed every hour, and a time
  pad widens them.

With elements that describe the motion exactly, none of the three loses an encounter at
zero pad (test). Their misses come from what the elements leave out; the tests document
one per filter (drag decay for apogee/perigee and orbit path, drag-driven timing drift for
the time filter). Pads are an open decision.

**Metrics.** Event recall (reference events matched one to one by a candidate event of the
same pair with |dTCA| <= a tolerance) and pair recall (reference pairs with at least one
matched event); false alarms per event and per pair; recall for reference events with
TCA within 1, 3 and 7 days; for filters, event and pair recall (an event survives if its
pair does) and the fraction of pairs pruned; wall-clock cost of each run. The headline
counting rule and the TCA tolerance are open decisions.

**Scaling** (synthetic LEO catalog, two-body plus J2 secular motion, 7 days at 60 s, one
process, Intel Core Ultra 9 275HX; other jobs were running, 1-minute load 5-13 on 24
logical CPUs):

| Objects | Threshold | Engine wall time | Peak memory | Events |
|---|---|---|---|---|
| 500 | 5 km | 4-5 s | 0.66 GB | 103 |
| 500 | 5 km + (1 + 6 km/day x tau) | 5 s | 0.66 GB | 3,722 |
| 2,000 | 5 km | 31-47 s | 1.5 GB (1.1 GB of it the trajectories) | 1,702 |
| 2,000 | 5 km + (1 + 6 km/day x tau) | 41-49 s | 1.5 GB | 57,938 |

At 2,000 objects the classical filters as implemented (numpy) took 0.1-0.4 s (apogee/
perigee), 42-71 s (orbit path, on the pairs apogee/perigee kept) and 123-150 s (time filter,
on the pairs orbit path kept), longer than the engine itself. Their compute cost reflects
this implementation, not an optimised one. Engine cost grows roughly as n^2 at this size;
trajectory memory is 0.54 MB per object per 7-day set.

Triangulation against the public Aerospace / Department of Commerce screening test set
(Auman, Murphy and George, 2026; CC0) is pending: the files are on Google Drive behind an
access request, and the ephemerides are a 20.7 GB archive of OCM files.

Implemented here vs library: the k-d tree is `scipy.spatial.cKDTree` and tables are
pyarrow. Interpolation, search bound, minimum finding, element conversions, filters and
metrics are this project's code, written with AI assistance and checked by the tests above.

## Limitations

_To be written._
