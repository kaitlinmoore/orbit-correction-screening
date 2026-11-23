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

_To be written (build step 5)._

## Limitations

_To be written._
