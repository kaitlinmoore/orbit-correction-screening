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
- Filters and their values: `config/catalog_filters.toml` (pending approval).

## Conventions

- Frame GCRF, time TT seconds since J2000.0, km and km/s (`src/orbit_correction/types.py`).
- SGP4 TEME output converted with astropy (`src/orbit_correction/frames.py`); EOP pinned via
  `astropy-iers-data`.
- Residual targets: curvilinear RTN, cheap minus truth, in the frame of the SGP4 state
  (definition in `types.py`).

## Splits

By object and by time (`src/orbit_correction/splits.py`, `config/splits.toml`, pending
approval). Validation-case objects are always held out.

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
