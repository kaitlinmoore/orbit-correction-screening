# Methodology

Skeleton. Sections fill in as components land; the planning package's locked decisions are
summarised here once the README results exist.

## Question

How much does a learned correction reduce SGP4's position error against a high-fidelity
numerical propagator, and do corrected predictions screen LEO conjunctions better than
uncorrected SGP4 and classical geometric filters? Truth is a model (a numerical propagator),
not tracked positions.

## Data

- Element sets: CelesTrak GP (development, tests); Space-Track's yearly bulk TLE files for
  training and test epochs (decided 2026-11-24, following Space-Track's guidance to use the
  bulk files for many objects or long date ranges): the 2015 to 2025 files, element sets
  kept by epoch from 2015-01-01 to 2026-01-01 (the files are grouped by publication date,
  not epoch); and Space-Track `gp_history` from 2004-01-01 to 2026-11-23 for the ten
  validation-case objects only (one download, 148,602 element sets). **The test period is
  2025 for now**: windows of any role must end within the downloaded history (before
  2026-01-01). 2026 will be added when Space-Track publishes its 2026 file, as a
  test-period extension only (training epochs end at the 2025-01-01 cutoff either way).
  Loading and filtering: `src/orbit_correction/catalog.py`; the bulk TLE reader and the
  download list: `src/orbit_correction/generation.py`.
- Space weather: CelesTrak `SW-All.txt`, read by `src/orbit_correction/spaceweather.py`.
- Filters and their values: `config/catalog_filters.toml` (approved 2026-11-23). LEO
  (perigee >= 250 km, apogee <= 2000 km), 0 < B* <= 0.01 /ER, SGP4 element sets only,
  payloads, rocket bodies and debris. Non-maneuvering: operational payloads (current SATCAT
  status) and payloads of maneuvering constellations by their current SATCAT name are
  dropped (decided 2026-11-24: the name rule reads SATCAT names, since bulk TLEs carry
  none and an element set's own name can predate the object's naming, and it applies to
  payloads only, so constellation debris such as Iridium 33's is kept), and so is any object whose
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
  (definition in `types.py`, code in `residuals.py`).
- SGP4 Julian dates count every UTC day as 86,400 s; `frames.satrec_epoch_to_tt_seconds`
  reads them that way (midnight plus seconds), so epochs on leap-second days are exact. The
  day is taken from the pair's two parts separately: their float sum rounds to the next
  midnight within about 20 microseconds of the end of a day.
- Geodetic altitudes (the 120 km floor) are WGS-84, matching the truth propagator's body
  shape, although SGP4 itself uses WGS-72 constants.

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
  far below any physical altitude; the wrapper therefore also truncates before the first
  state below a 120 km geodetic altitude, the truth propagator's re-entry altitude (status
  code -1, `altitude_floor`). Both tests are made at the requested times only, so where a
  trajectory is cut depends on the sampling grid; the status's `first_error_t_tt_s` is the
  first sampled time that failed, not a re-entry time.
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
  (`scripts/zonal_convergence.py`; background, see the changelog). At the 180 s step that
  script's own convergence rule (each degree within the truth setting's integration error
  of the reference) gives degree 48-64 at 800-1,200 km and no degree at 300 and 500 km, so
  its "chosen degree" is None (rerun 2026-11-24). That is consistent with degree 36 being
  chosen, not converged.
- **NRLMSISE-00 conventions.** Mean local solar time (UT + longitude/15) is what pymsis and
  brahe compute, which makes the density cross-checks possible (pymsis to 1.1e-5). The
  reference notes name the input local apparent solar time and call equation-of-time
  departures of minor importance; along real orbits apparent and mean differ in density by up
  to 6.8% (800 km, November 2024). Switch 9 = -1 feeds the 3-hourly ap history; daily Ap
  (+1) differs by up to 7.8% along an orbit and about 20% at storm points, up to 11 km at
  300 km over 7 days. Switch 9 stays configurable for a labeled daily-Ap sensitivity
  subsample in residual generation.
- **Initial state and start rule:** the SGP4 state at the element-set epoch, rotated TEME to
  GCRF by `frames.teme_to_gcrf`, is the deliberate conversion from mean elements; the start
  rule says how truth starts from it. **Since 2026-11-25 the configured rule is the fitted
  start, `fit_to_sgp4_before_epoch`, with a 0.25-day span at 60 s** (decided by Kaitlin from
  the comparison below; changelog). It replaced `preserve_sgp4_mean_motion`, which is kept
  for comparison and is reached only by naming it: a config built without a rule gets the
  fitted one, a config file without a rule is refused, and a truth job without SGP4 fit
  samples is refused rather than started from the epoch state (`tests/test_truth_prop.py`).
  Physics tests, the settings scripts and the brahe cross-check, which start from one shared
  state, name the epoch-state rule explicitly.
- **Epoch-state rule (`preserve_sgp4_mean_motion`, the default 2026-11-23 to 2026-11-25).**
  The SGP4 epoch state, used as an osculating state, scaled by k = (mu_truth / mu_WGS72)^(1/3)
  in position and velocity. SGP4 states are consistent with the WGS-72 GM
  (398600.8 km^3/s^2) and truth integrates with EIGEN-6S's (398600.4415); without the
  scaling the GM difference alone makes truth lag SGP4 by about 1.15 km per day, an artefact
  a model would learn trivially and that tracked orbits wouldn't have. The scaled state
  keeps the two-body period (each under its own GM), eccentricity and plane, and sits about
  2 m from the SGP4 state, so **under this rule** residuals at t0 are about 2 m, not zero.
  On eight sample objects (300-1,200 km, June 2024 start; `scripts/truth_settings_check.py`)
  the scaling moves the 7-day SGP4-minus-truth along-track difference by about 8 km in every
  band. What remains at 7 days under this rule is 170-970 km at 300 km and 35-86 km at
  500 km, where drag dominates, and 3-15 km at 800 and 1,200 km. The fitted-start
  comparison below splits the last figure: most of it is the start (the phase artifact)
  plus the drag-model difference, and only a few hundred metres is SGP4's theory.
- **Fitted start (`fit_to_sgp4_before_epoch`, the configured rule since 2026-11-25).** An
  independent review found that a start at one SGP4 state ties the truth orbit's mean
  motion to where in its orbit the object is at the epoch: SGP4's short-periodic terms
  make that state's osculating period differ from the period SGP4 keeps, by an amount
  that depends on the phase. Truth then drifts along-track against SGP4 by a
  phase-dependent amount that tracked element sets, fitted over a span, would not show,
  and that a model would learn from its inputs. The fitted rule fits the six-number epoch
  state by unweighted least squares to SGP4's GCRF positions at 60 s spacing over
  [epoch - span, epoch] (`truth_prop.fit_start_to_sgp4`). Each Gauss-Newton pass
  propagates backwards from the epoch with the full truth force model and Orekit's state
  transition matrix; it stops when the correction is below 1 mm and 1 micrometre/s
  (typically 3 passes). **Only SGP4 states at or before the epoch enter the fit**, so no
  information from the prediction window shapes the start; the code refuses later
  samples and a test runs the whole path under a spy on SGP4. The GM rescale is not
  needed: the fit finds the state that matches SGP4 under the truth GM. A fit that does
  not converge is a truth status (`fit_failed`), never a fallback. Project code: the fit,
  its convergence rule and its record; Orekit's: the integration and the variational
  equations. Measured with `scripts/fitted_start_comparison.py` (2026-11-24; synthetic
  phases and moved epochs on real train/dev orbits, details in the changelog; the phase,
  strata and sample-object parts ran at a 1-day span, the span part at 0.125-2 days):
  - On the review's gravity-only cases (800 km, B* 1e-7, e 0.001 to 0.07) the epoch-state
    rule's 7-day T of up to 678 km becomes at most 0.32 km.
  - On 360 stratified windows with drag (five perigee bands, three eccentricities up to
    90% of the apogee limit, three solar bins, four phases), the spread of 7-day T across
    the four phases falls by a factor of 8 to 2,700 per stratum (medians). Above 800 km the
    fitted start leaves a 7-day T of about 1 km (800-1,200 km) and 0.3 km
    (1,200-2,000 km), the same in every solar bin.
  - **Gravity-only split at 800-1,200 km.** On the four truth sample objects there, the
    epoch-state rule leaves 3-15 km at 7 days. Run gravity-only on both sides (B* 0 in
    SGP4, no drag in truth), the fitted start leaves 0.18-0.28 km on all four, against
    3.5-13.3 km under the epoch-state rule: that is SGP4's gravity theory against the
    degree-36 zonals. With drag, the fitted start leaves 0.5 km on the one with low B*
    (the stratified windows: 0.69 km median at 800-1,200 km) and 4-11 km on the
    three whose B* is large for their altitude (6.7e-5 at 800 km, 7.7e-4 to 1.1e-3 at
    1,200 km), which is the drag-model difference. So the 3-15 km is
    mostly the start plus the drag-model difference, not SGP4's theory.
  - Below 600 km, where the drag models differ, the fit also absorbs part of that
    difference over the span into the start. The 7-day T of near-circular orbits at
    250-400 km grows by about 2.5% (0.125-day span), 4.6% (0.25 d) and 17% (1 d) against
    the epoch-state rule, and T at 1 h from about 0.09 km to 0.06, 0.14 and 1.3 km. Shorter
    spans absorb less; two revolutions already average the short-periodic terms.
  - **Residuals at t0 under the fitted rule** are the fitted state's offset from the SGP4
    epoch state, not the epoch-state rule's 2 m. At the configured 0.25-day span: median
    10 / 21 / 56 m in R / T / N, p95 71 / 137 / 1,006 m (60 span-test windows at solar
    maximum). At a 1-day span: median 31 / 45 / 21 m, p95 |T| about 1.2 km (360
    stratified windows). The largest offsets are at 250-400 km.
  - Truth compute per window, with the rules interleaved on 60 windows at one worker
    count, is 1.47 times the epoch-state rule's at a 0.25-day span and 2.69 times at
    1 day (2.57 times at 1 day over the 360 stratified windows).
  - Why 0.25 days: the phase spread is about 0.9 km (median) against 0.8 km at 1 day,
    with a quarter of the 1-day span's drag absorption at 250-400 km and just over half of
    its compute.
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
  `frames.teme_to_gcrf` (0.5-1.1 m); the epoch-state start rule's period in a two-body run; the
  integration error of the truth setting per case; the range and margin guards; the
  re-entry outputs; reproducibility to 1 mm across workers and job order. A library check
  compares Orekit's TLEPropagator with the sgp4 package (9 micrometres); the project doesn't
  use the former.
- **Independent check:** brahe 1.9.0 with a shared static zonal set (`scripts/crosscheck_brahe.py`).
  Without drag the two agree to millimetres at 7 days (tolerance 0.1 m). With drag they
  agree once both use the same NRLMSISE-00 conventions (daily Ap, mean solar time), within
  per-band tolerances of 10 m at 300 km, 1 m at 500 km and 0.1 m at 800 km and above.
  Rerun on all eight sample objects with the epoch-state start rule and the 180 s setting
  (2026-11-23): no drag at most 5.1 mm; drag with matched conventions at most 4.1 m at
  300 km, 0.058 m at 500 km, 4.8 mm at 800 km and 1.8 mm at 1,200 km. With the project's
  ap-array setting brahe differs by up to 11 km at 300 km, a convention difference.

## Residual dataset

`src/orbit_correction/residuals.py`, configuration in `config/residuals.toml` (bins and
horizon grid approved 2026-11-24 for the pilot). Contract: `RESIDUAL_SCHEMA` in `types.py`.

- **Window.** One element set, propagated from its epoch (the prediction start in v1) by
  SGP4 and by truth on the same TT instants, `epoch + dt_s`. The row builder refuses a
  trajectory whose times differ from those by any amount, so the two propagators are
  never compared at different instants.
- **Targets.** The curvilinear RTN residual, cheap minus truth, in the frame of the SGP4
  state (definition in `types.py`): R is the difference of radii, T the along-track angle
  times the SGP4 radius, N the out-of-plane angle times the SGP4 radius. Because T is an
  arc, a large along-track error does not leak into R: a 1,045 km along-track error at
  342 km altitude shows up as about 81 km of spurious radial error in rectilinear RTN and
  none here. At small separations the two agree to `s^2 / (2 r)`. The inverse rebuilds a
  position from the SGP4 state and (R, T, N); it is how a model's prediction becomes a
  corrected position in screening. Round trip on 20,000 random LEO states (e up to 0.3,
  along-track up to 2,000 km): 8e-12 km.
- **Domain.** The along-track angle is defined modulo a full turn, so the conversion
  refuses angles above pi/2 (about 10,500 km of along-track error in LEO) rather than
  wrap; it also refuses truth positions more than pi/4 out of the SGP4 orbit plane and
  SGP4 states with (near-)radial velocity (limits approved 2026-11-24). The out-of-plane
  and radial-velocity limits are far from anything measured. **The pi/2 limit is not:** it
  is reachable at the low, high-drag edge of the catalog with both propagators still `ok`.
  With real truth on synthetic element sets at 2024-06-15, 350 km perigee with B* = 0.01
  /ER passes it at 6.7 days (T = 11,600 km at 7 days, 1.12 x the limit), and neighbouring
  windows that are kept reach 4,500-8,100 km (43-77% of it). Such a window is refused, not
  wrapped; generation counts it as its own drop status (`rtn_domain`) by stratum and
  reports the count beside the lowest band's error statistics, because the windows it
  removes are that band's largest errors. The inverse accepts one float64 step above the
  forward limits, so every residual the forward conversion returns can be inverted.
- **Features.** The element set's mean elements, B* and mean-motion derivative; the four
  start-of-prediction space-weather values (see Model inputs). The `obs_*` columns, when
  computed, take each row's UTC day from OBSERVED rows only; a day without an observed row
  stays null.
- **Labels.** Altitude band from the element set's perigee altitude (edges 250, 400, 600,
  800, 1,200 and 2,000 km); solar-activity bin from the trailing 81-day observed F10.7
  mean known at the start (edges 90 and 150 sfu; a missing mean is labeled `unknown`).
  Both are constant over a window. Approved 2026-11-24 for the pilot.
- **Horizon grid.** Approved 2026-11-24 for the pilot: 100 log-spaced horizons from 60 s
  to 7 days plus 100 evenly spaced every 6,048 s, rounded to whole seconds: 199 rows per
  window, about 41 kB in memory and 7.7 kB as zstd Parquet. Dense early horizons follow
  the fast initial growth of the residual; the uniform part covers the screening window
  evenly. dt = 0 is left out.
- **Drop rule (120 km).** One 120 km geodetic floor for both propagators: truth stops and
  returns `reentry`; SGP4 is truncated before its first state below 120 km. SGP4's
  altitude is WGS-84 from the TEME position, which ignores polar motion at a cost of
  under 10 cm (at most (a - b) x the pole offset, 7 cm at the largest offsets in the EOP
  table); on sampled states it is within 5 cm of Orekit's. A window gives no rows if truth
  returned `reentry` or `error`, or SGP4 stopped early for any reason (its own error codes
  or the floor). Dropped windows keep their altitude band and solar bin and are counted by
  status (`count_windows`), so the drop rate by stratum is reported, never hidden. SGP4's
  floor is tested at the window's horizons only and truth's continuously, so on the
  199-horizon grid the rule leans on truth, and the time at which SGP4 was cut depends on
  the grid.

Measured on six fixture objects moved to 2024-06-15 (solar maximum), with truth under the
epoch-state start rule (before the fitted start became the default): the residual at 60 s
is 1.8-2.8 m (that rule's start offset), and the 7-day along-track residual runs from 9.7 km
(770 km altitude) to 963 km (355 km), with R and N within a few km.

## Residual generation

Code: `src/orbit_correction/generation.py`; scripts `build_candidate_ids.py`,
`extract_bulk_tle.py`, `build_pool.py`, `sample_windows.py`, `generate_residuals.py`,
`pilot_report.py`; settings in `config/generation.toml` (pending approval).

- **Download list.** A superset of the objects the filters can keep, from SATCAT alone:
  allowed object types and not an operational payload (both exact, the filter reads the
  same columns), not launched after the span, not decayed more than 30 days before it,
  SATCAT apogee at most 2,100 km (the LEO bound plus 100 km: apogee only falls under drag,
  so the one assumption is that no object raised its orbit after the span; the margin also
  covers SATCAT apogees 39 km below to 97 km above the loader's values), and the name rule
  (exact, since the filter reads the same SATCAT names). No perigee test: perigee falls
  under drag, so an object low now may have been above 250 km earlier. The validation-case
  objects are always added.
- **Bulk TLE reading.** TLE fields are read with the `sgp4` package's column layout and
  arithmetic, so each element set gives the same `Satrec` as the library's reader except
  for the epoch, which is converted to an ISO string rounded to the microsecond (the TLE
  resolves 0.864 ms). Bad pairs (checksum, layout) and stray lines are counted per file.
  File quirks: the 2015 file ends every line 1 with a backslash (removed and counted) and
  carries macOS `__MACOSX` members (skipped); 68-character lines without a checksum
  column (analyst objects and some old sets) are counted and left out. Each file's epoch
  coverage by day is recorded.
- **Pool.** Element sets in the span, exact duplicates removed (same object, epoch and
  elements; same-epoch rows that differ are kept and counted), SATCAT attached, the
  catalog filters applied with rows and objects removed reported per rule. The pool is
  built per group of NORAD IDs, which changes nothing because every rule is per object.
- **Split.** Every pool element set is labeled as a window start with
  `splits.assign_split_labels`; the last fully observed space-weather day, the seed and
  the config hashes are recorded with it. Then one rule of the download (decided
  2026-11-25): a window of any role ending after the end of the downloaded history
  (2026-01-01 today) is labeled `excluded`, counted by its original label. Only counts per
  label are reported.
- **Window sampling (proposed).** Starts at element-set epochs, train and dev rows only.
  Per object, its eligible epochs are tried in a seeded random order and a start is kept
  if it is at least 7 days from the object's other starts, up to a cap. Across objects,
  round k takes every object's k-th start (objects in a seeded order), so every object
  has one start before any has two, and a smaller stage is the first part of a larger one.
  Uniform over element sets means better-tracked periods get more starts; coverage is
  reported by year, altitude band, solar bin, eccentricity bin and split.
- **Runs.** Chunks of windows: SGP4, its epoch states and, under the fitted start, the SGP4
  samples over the 0.25 days before each epoch (`truth_prop.fit_samples_from_elsets`) in
  the parent process (no JVM), passed to each truth job; truth in spawned workers that
  keep their JVM for the whole run, rows from
  `build_window_rows`, one zstd Parquet file per chunk. A chunk is done when its marker,
  written last, names the same windows and both of its files still hash to what the
  marker recorded; a rerun skips done chunks and redoes the rest. One space-weather copy
  covers the run from the earliest epoch minus the fit span to the latest window end,
  checked by the existing range and 41-day margin guards (from each job's first fit
  sample). A worker that dies breaks the pool (`BrokenProcessPool`): nothing of that chunk
  has been written, the chunk is retried on a fresh pool up to three attempts, and every
  crash is recorded (`crashes.jsonl`, counted in the manifest).
- **Window fates.** Every sampled window gets one log record and one status: `ok`, the
  row builder's drops (`truth_error`, `truth_fit_failed` for a fitted start that did not
  converge, `sgp4_error`, `truth_reentry`, `sgp4_floor`), the conversion's refusal
  (`rtn_domain`), or one of three statuses for windows that get no truth job: no SGP4
  state at the epoch (`start_below_floor`, `start_sgp4_error`), or no usable SGP4 samples
  over the fit span (`fit_samples_unavailable`). A window without fit samples is never
  started some other way. The log also carries each fit's status, passes and misfit. Counts by
  status, altitude band and solar bin go into the run manifest, with the code commit,
  config hashes, orekit-data commit, the space-weather copy's hash and range, the last
  observed day and the seed.
- **Daily-Ap sensitivity.** A run option sets NRLMSISE-00 switch 9 to +1 (daily Ap) for a
  subsample of the same windows and labels every output `dailyap`; `config/truth.toml`
  is not changed.

## Models

_To be written (build steps 4 and 7)._

## Screening and metrics

Engine, classical filters and metric functions: `src/orbit_correction/screening.py`, tested
on synthetic encounters with known answers (`tests/test_screening.py`) and checked against
an external answer key (the Aerospace / Department of Commerce test set, below). The
screening comparison itself (truth vs SGP4 vs corrected vs classical filters) has not been
run yet.

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

**External check: the Aerospace / Department of Commerce test set.** The Aerospace
Corporation and the Department of Commerce released a conjunction-assessment verification
set (Auman, Murphy and George, user's guide March 2026; CC0-1.0). It contains OCM
ephemerides of 26,811 objects for the first week of January 2025, and an answer key from
Aerospace's CSieve tool: 913,330 events from an all-vs-all screen with a 10 km sphere,
2025-01-01T12:00 to 2025-01-08T12:00 UTC. The sphere applies to miss distance alone. The
0.5 m hard-body radius is used only for collision probability. The guide's event
definition is this engine's: a local range minimum within the volume, with TCA inside
the window. `scripts/compare_aerospace_testset.py` runs `screen` on the set's own
ephemerides and compares the events with the key. Neither side uses the other's code or
this project's propagators, so this checks the engine alone.

- *Subset* (seed 20261124, hash sampling per object, so it was applied while the 22 GB
  archive streamed past once):
  - **main:** 10% of the TLE-derived objects whose first state has a two-body
    semimajor-axis altitude of 500-600 km. That gives 550 objects, all covering the whole
    window on a single 60 s grid.
  - **maneuver:** the 87 purpose-built maneuvering ephemerides and victims (IDs
    90006-90190). Most have three TRAJ blocks (coast, finite burn, coast), and many start
    or stop inside the window.
  - **fictitious:** the 9 purpose-built fictitious objects (IDs 99000-99008) and the 81
    objects the key pairs them with.

  Each group is screened among itself, and the key is restricted to rows with both files
  in the group. That restriction is valid because a sphere is pair-local. Objects with
  partial spans are screened on each pair's USEABLE overlap. All 727 files matched the
  archive's own MD5 list. None failed the guide's 14-day OD-epoch rule.
- *Reading* (`scripts/aerospace_ocm.py`):
  - **Time:** UTC to TT is TT - UTC = 69.184 s. No leap second falls in the window, and
    the reader refuses a block that spans one.
  - **Frame:** EME2000 to GCRF uses the constant IAU 2006 frame bias (0.0230 arcsec,
    about 0.8 m at LEO radius). A rotation cannot change any range.
  - **Resampling:** onto a uniform TT grid, at 60 s over the window, and at the largest
    step of at most 60 s that spans a partial overlap exactly. The interpolant is
    8-point Lagrange on positions, with velocity from its derivative. On held-back
    samples its error is 2.8 mm median and 8 mm maximum in the main group, and 2 mm
    median in the maneuver group. The worst cases are a 1.6 m maximum in the maneuver
    group and 213 m in one fictitious-group file whose single TRAJ block contains a
    maneuver.
- *Result:* every event matched, in both directions, at a 1 s TCA tolerance:

  | Group | Objects | Key events | Matched | Key only | Engine only | TCA diff., max | Miss diff., median / p95 / max |
  |---|---|---|---|---|---|---|---|
  | main | 550 | 3,815 | 3,815 | 0 | 0 | 0.44 ms | 0.7 mm / 3.7 mm / 25 mm |
  | maneuver | 87 | 111 | 111 | 0 | 0 | 86 ms | 0.3 mm / 10 mm / 0.33 m |
  | fictitious | 90 | 237 | 237 | 0 | 0 | 24 ms | 2.4 mm / 0.10 m / 0.39 m |

  There are no unmatched events, so none needed explaining. Two checks of the check:
  - The key's own J2000 states at its TCAs match the files resampled there to 2-4 mm
    (median). That is a test of the time and frame handling that involves no screening.
  - A dense re-check finds the range minimum straight from the file samples, without the
    60 s grid. It agrees with the key to 0.2 mm (median) in the main group.

  The engine's relative speeds differ from the key's by 14 mm/s (median) in the main
  group and up to 15 m/s in the maneuver group. The key's velocities are the files' own,
  and those are not the derivatives of the files' positions (next point).
- *Velocities in the files are not the derivatives of their positions:* by 16 mm/s
  (median; up to 0.37 m/s) in the TLE-derived files and 4.6 m/s (median; up to 9.5 m/s)
  in the maneuvering ones. SGP4 itself shows 5-55 mm/s on the project's fixture objects.
  The engine's cubic Hermite uses velocities between grid points.
  - Rerun with the files' velocities (`--velocity file`), the main group still matches
    3,815 of 3,815, but miss differences grow to 2 cm median (0.45 m maximum).
  - In the maneuver group, miss differences reach 54 m, and one key event (9.988 km) is
    lost: the engine's grid puts that minimum just above 10 km.

  For screening SGP4 or corrected trajectories, this means between-sample accuracy
  depends on how consistent their velocities are with their positions.
- *Limits:*
  - **What the agreement shows:** the two tools find the same range minima of the same
    sampled motion. It says nothing about the physical encounters.
  - **Coverage:** one altitude band of TLE-derived objects and two purpose-built groups.
    Not covered: the historical-CDM objects (95000-95407; a sample would need thousands
    of partners), Osiris-REx, the candidate maneuver OCMs of 43476, and the SFSH
    rectangular-volume key.
  - **The threshold edge is barely exercised:** only 6 key events in the subset lie
    within 10 m of 10 km, all in the main group, and all matched.
  - **Unknowns:** CSieve's interpolation method is not documented, and the guide states
    that its key may not be exhaustive.

Implemented here vs library: the k-d tree is `scipy.spatial.cKDTree` and tables are
pyarrow. Interpolation, search bound, minimum finding, element conversions, filters and
metrics are this project's code, written with AI assistance and checked by the tests above.

## Limitations

_To be written._
