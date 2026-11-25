# orbit-correction-screening

A PhysicsNeMo model that learns the error of SGP4 against a high-fidelity numerical propagator, tested on whether corrected predictions screen low-Earth-orbit conjunctions better than uncorrected SGP4 and classical geometric filters.

Status: data pipeline built (catalog, splits, SGP4 and Orekit truth propagators, curvilinear RTN residuals, the screening engine with classical filters, and residual-dataset generation); a generation pilot on real historical element sets is in progress. No model and no model results yet.

## Question

Cheap propagators (SGP4 on public element sets) are fast but drift; high-fidelity numerical propagation is accurate but expensive across a catalog. This project trains a model in NVIDIA PhysicsNeMo to predict SGP4's error relative to a numerical propagator (Orekit: zonal gravity and NRLMSISE-00 drag driven by observed space weather), then asks: (1) how much the correction reduces position error as a function of prediction horizon; (2) whether corrected predictions recover the close approaches (under 5 km over 7 days) that numerical screening finds, with fewer false alarms than uncorrected SGP4; and (3) whether that beats classical geometric filters. "Truth" is a numerical model, not tracked positions, and missed conjunctions count as the costly error.

## Data sources

Data is gitignored and never committed. Downloads go under `data/raw/`.

| Source | URL | Version | Access date | License / terms |
|---|---|---|---|---|
| CelesTrak GP element sets (OMM JSON) | https://celestrak.org/NORAD/elements/gp.php (e.g. `?CATNR=26998&FORMAT=json`, `?GROUP=active&FORMAT=json`) | live data; epochs as of the access date | 2026-11-23 | Free, no account. Small extracts are committed as test fixtures (`tests/fixtures/README.md`) |
| CelesTrak SATCAT | https://celestrak.org/pub/satcat.csv | live data | 2026-11-23 | Free, no account |
| CelesTrak space weather (F10.7, Kp, Ap) | https://celestrak.org/SpaceData/SW-All.txt (format: https://celestrak.org/SpaceData/SpaceWx-format.php) | file header `UPDATED 2026 Nov 23 10:32:51 UTC` | 2026-11-23 | Free, no account |
| Space-Track yearly bulk TLE files (historical element sets, training history) | https://www.space-track.org/documentation (API Use Guidelines, `GP_HISTORY` row: "For queries of many objects or large date ranges, download TLEs bundled as zip files by year from our cloud storage site instead") | `tle2015.txt.zip` to `tle2025.txt.zip` (11 files, 7,341,206,490 bytes; two-line TLEs without names, grouped by publication date; the 2025 file exported 2026-01-07; no 2026 file published yet). Element sets kept by epoch, 2015-01-01 to 2025-12-31. Per file size, SHA-256 and read counts in `data/derived/pool/bulk_extract/extract_manifest.jsonl` | 2026-11-25 | Free account; Space-Track User Agreement. This project never commits or redistributes Space-Track data |
| Space-Track `gp_history` (validation-case objects only) | https://www.space-track.org (`basicspacedata/query/class/gp_history`) | 5 requests, 148,602 element sets of the 10 objects, epochs 2004-01-01 to 2026-11-23 (`data/raw/spacetrack/gp_history/manifest.jsonl`); downloaded once, not to be repeated | 2026-11-25 | As above. Space-Track asks for each object's history once ("1 / lifetime") |
| IERS Earth orientation data and leap seconds, via `astropy-iers-data` | https://pypi.org/project/astropy-iers-data/ (data from https://www.iers.org) | 0.2026.11.23.0.35.53, pinned in `uv.lock` | 2026-11-23 | Package BSD-3-Clause; IERS data freely available |
| Conjunction-assessment verification test set (The Aerospace Corporation with the Department of Commerce; Auman, Murphy and George, user's guide "Conjunction Assessment Verification Data and Process") | Dataset page https://space.commerce.gov/?p=13145; user's guide https://space.commerce.gov/wp-content/uploads/2026/03/Conjunction_Screening_Testset_Users_Guide.pdf | Released March 2026. Ephemerides `AerospaceIVVDataset_20251009a.tar.gz` (22,255,966,812 bytes, SHA-256 `0e990091659c9c564c731882eb213938d32455d4d294995b6fec22e405e13d66`); 10 km spherical answer key `IVV_Releasable_Dataset_Spherical_DefaultHBR.csv.gz` (SHA-256 `3cb76041785ab0302589a7331823fc0e902a825573a8d97a5291232258d755dd`). The SFSH-volume key and volume mappings were not downloaded | 2026-11-23 | CC0-1.0. Not redistributed here: the files stay in `data/` (gitignored); used only for the external check of the screening engine (`scripts/compare_aerospace_testset.py`) |
| Orekit data bundle (`orekit-data`) | https://gitlab.orekit.org/orekit/orekit-data | git commit `2d6edc9118983d95c51303df05db16ada90b7ee2`, installed by uv from that commit (`orekitdata` in `uv.lock`, `datagen` group) | 2026-11-23 | The bundle has no license file of its own; each file stays under its producer's terms (rows below). Not redistributed here: uv fetches it from the pinned commit. The truth propagator does not read the bundle's CSSI space-weather, MSAFE, JB2008 (`Space-Environment-Data`) or FES 2004 files |
| ↳ `Earth-Orientation-Parameters/IAU-2000/finals2000A.all` (Earth orientation) | IERS Rapid Service/Prediction Center at USNO, https://maia.usno.navy.mil/ser7/readme | as in the bundle commit | checked 2026-11-25 | The data.gov record for these products lists its license as U.S. Government Works (https://www.usa.gov/government-works); USNO's own pages state no terms |
| ↳ `tai-utc.dat` (leap seconds) | USNO, https://maia.usno.navy.mil/ser7/tai-utc.dat | as in the bundle commit | checked 2026-11-25 | No terms stated by the source (the file and USNO's readme carry none) |
| ↳ `DE-440-ephemerides/lnxp1990.440` (Sun and Moon positions) | JPL Solar System Dynamics, https://ssd.jpl.nasa.gov/planets/eph_export.html; Park et al. (2021), AJ 161, 105, https://doi.org/10.3847/1538-3881/abd414 | DE440 | checked 2026-11-25 | No license or usage statement found on JPL's export page or FTP README; cited by the paper above |
| ↳ `Potential/eigen-6s.gfc` (gravity field) | GFZ Potsdam and GRGS Toulouse, distributed by ICGEM (https://icgem.gfz.de/tom_longtime); Förste et al. (2011), EGU2011-3242-2 (from the file header) | EIGEN-6S (2011) | checked 2026-11-25 | No license or DOI found for this model; ICGEM's imprint says data from other sources is under its owner's terms ("please contact the owner of the data for the terms of use"). Not yet asked |
| ↳ `itrf-versions.conf` | Orekit team, in the bundle | as in the bundle commit | checked 2026-11-25 | Configuration file of the bundle; no separate terms |

Software that produces or checks data (installed by uv from `uv.lock`; nothing is redistributed here):

| Software | URL | Version | License | Used for |
|---|---|---|---|---|
| Orekit | https://www.orekit.org | 13.1.9 | Apache-2.0 | truth propagator (`truth_prop.py`) |
| orekit-jpype | https://pypi.org/project/orekit-jpype/ | 13.1.9.0 | Apache-2.0 | Python bindings to Orekit through JPype (`datagen` group) |
| jdk4py | https://pypi.org/project/jdk4py/ | 25.0.5.1 (JDK 25) | GPL-2.0 with the Classpath exception | the Java runtime orekit-jpype runs on (`datagen` group) |
| brahe | https://github.com/duncaneddy/brahe | 1.9.0 | MIT | independent cross-check of the truth propagator (`crosscheck` group) |
| pymsis | https://github.com/SWxTREC/pymsis | 0.13.0 | MIT | reference densities in tests only (`dev` group) |
| astropy-iers-data | https://pypi.org/project/astropy-iers-data/ | 0.2026.11.23.0.35.53 | BSD-3-Clause | Earth orientation data for TEME -> GCRF (row above) |

CelesTrak downloads used here (`curl` with a generic user agent):

```bash
mkdir -p data/raw/celestrak && cd data/raw/celestrak
curl -sSfO https://celestrak.org/SpaceData/SW-All.txt
curl -sSfO https://celestrak.org/pub/satcat.csv
curl -sSf -o gp_active.json "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json"
for id in 24946 22675 26998 22236 13777 2828 19826 36123 7219 26207; do
  curl -sSf -o gp_$id.json "https://celestrak.org/NORAD/elements/gp.php?CATNR=$id&FORMAT=json"; sleep 1
done
```

## Setup

WSL2 (Ubuntu), because PhysicsNeMo supports Windows only through WSL. The repo stays under `C:\dev\GitHub`; the uv environment lives in the WSL filesystem.

```bash
cd /mnt/c/dev/GitHub/orbit-correction-screening
direnv allow                 # .envrc sets UV_PROJECT_ENVIRONMENT=$HOME/.venvs/orbit-correction-screening
uv sync                      # Python 3.12 from .python-version; locked deps from uv.lock
uv sync --group datagen --group crosscheck   # for data generation (Orekit, JVM) and the brahe cross-check
cp .env.example .env         # then fill in Space-Track credentials (only for the download script)
```

Without direnv, `export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/orbit-correction-screening` before any `uv` command, so no `.venv/` is created in the repo.

## How to run

```bash
uv run pytest                # all tests run offline
uv run ruff check
# Historical element sets (your own Space-Track account; paced under Space-Track's rate limits):
uv run python scripts/download_spacetrack.py --ids 26998,22236 --start 2024-02-14 --end 2024-03-01 --dry-run
uv run python scripts/download_spacetrack.py --ids 26998,22236 --start 2024-02-14 --end 2024-03-01
# Rebuild the test fixtures:
uv run python scripts/make_fixtures.py
```

Residual-dataset generation (data paths are arguments; outputs stay in `data/`):

```bash
# 1. Download lists from SATCAT (the superset of objects the filters can keep)
uv run python scripts/build_candidate_ids.py --satcat data/raw/celestrak/satcat.csv \
    --out-dir data/derived/pool --start 2015-01-01 --end 2026-01-01
# 2. Validation objects' histories through the API (done once; don't repeat); everything
#    else from the yearly bulk TLE files saved under data/raw/spacetrack/bulk_tle/
uv run python scripts/download_spacetrack.py --ids-file data/derived/pool/ids_validation.txt \
    --start 2004-01-01 --end 2026-11-24 --batch-size 2 --out-dir data/raw/spacetrack/gp_history
uv run python scripts/extract_bulk_tle.py --bulk-dir data/raw/spacetrack/bulk_tle \
    --ids data/derived/pool/candidate_ids.txt --start 2015-01-01 --end 2026-01-01 \
    --out-dir data/derived/pool/bulk_extract --jobs 6
# 3. Pool (filters) and split labels; window sample for a stage
uv run python scripts/build_pool.py --extract-dir data/derived/pool/bulk_extract \
    --validation-json 'data/raw/spacetrack/gp_history/*.json.gz' \
    --satcat data/raw/celestrak/satcat.csv --space-weather data/raw/celestrak/SW-All.txt \
    --start 2015-01-01 --end 2026-01-01 --out-dir data/derived/pool/pool_v1
uv run python scripts/sample_windows.py --pool-dir data/derived/pool/pool_v1 --stage pilot \
    --space-weather data/raw/celestrak/SW-All.txt --out data/derived/windows/pilot
# 4. Generation (spawned Orekit workers; resumable), the daily-Ap sensitivity run, report
uv run --group datagen python scripts/generate_residuals.py \
    --windows data/derived/windows/pilot/windows.parquet \
    --space-weather data/raw/celestrak/SW-All.txt --out-root data/derived/residuals \
    --run-id pilot_baseline
uv run --group datagen python scripts/generate_residuals.py --daily-ap \
    --windows data/derived/windows/pilot/windows_dailyap.parquet \
    --space-weather data/raw/celestrak/SW-All.txt --out-root data/derived/residuals \
    --run-id pilot_dailyap
uv run python scripts/pilot_report.py --run data/derived/residuals/pilot_baseline \
    --dailyap-run data/derived/residuals/pilot_dailyap \
    --out data/derived/residuals/pilot_baseline/report
```

## Results

No model results yet.

Engine verification (not a model result): engine check against the Aerospace/DoC 10 km key on a 727-file subset: 4,163 of 4,163 events, none extra, miss agreement 0.7 mm median (`scripts/compare_aerospace_testset.py`).

## Methodology

- [`docs/methodology.md`](docs/methodology.md): approach, conventions, data, splits.
- [`docs/changelog.md`](docs/changelog.md): every instrument and design change with its reason.
- Contracts between components: [`src/orbit_correction/types.py`](src/orbit_correction/types.py), checked by `tests/test_contracts.py`.
- Catalog filters and split values (approved 2026-11-23): [`config/`](config/).

What is implemented here and what is library behaviour:
- SGP4 is the `sgp4` package (Vallado's implementation). TEME to GCRF rotations and time-scale arithmetic are astropy and ERFA. This project adds the conventions and wrappers, EOP pinning and range checks, and tests against the Vallado et al. (2006) reference examples.
- The space-weather parser, the leak-free as-of lookup, the Orekit-compatible writer, the catalog loaders and filters, and the split logic are this project's own code.
- The curvilinear RTN residual, its inverse and the residual-row builder (`residuals.py`), and SGP4's 120 km altitude floor (`cheap_prop.py`, geodetic altitude from ERFA's `gc2gd`), are this project's own code.
- Residual-dataset generation (`generation.py` and its scripts: the candidate rule, the TLE reader for Space-Track's bulk files, the pool, window sampling, the chunked and resumable runs, the generation log and manifests) is this project's own code. The TLE column layout, its number arithmetic and the Alpha-5 decoding follow the `sgp4` package's reader (`sgp4.io.twoline2rv`, `sgp4.alpha5`), and the tests check the result against that reader.

## Limitations and scope

- Truth is a numerical model (Orekit), not tracked positions.
- v1 covers LEO, non-maneuvering objects only, and the non-maneuvering rule is approximate. None of its parts is a historical maneuver record:
  - The status part uses each object's *current* SATCAT status, not its status at each epoch.
  - It misclassifies non-propulsive operational satellites as maneuverable. TIMED is one example (held out anyway as a validation object).
  - The constellation name list covers only the constellations named in `config/catalog_filters.toml`, and is tested on each object's current SATCAT name and on payloads only.
  - The element-history detector misses small maneuvers: semimajor-axis changes under about 1 km plus the drag allowance, inclination changes under 0.05 deg, and anything hidden in a long gap between element sets.
- Prediction windows ending after the last fully observed space-weather day are excluded, so the most recent epochs are never used.
- The space-weather file is a snapshot of final, revised values. The as-of lookup prevents reading values after the prediction start, but cannot reproduce what an operator knew in real time.
- The test period is 2025 only for now: windows must end within the downloaded history (the 2015-2025 bulk files). 2026 will be added as a test-period extension when Space-Track publishes its 2026 file.
- Catalog filter values and split parameters are approved (2026-11-23); generation settings (`config/generation.toml`) are pending approval.

## AI assistance

This repository is built with AI assistance. Code, tests, fixtures and documentation in the current state were written by Claude (Anthropic) through Claude Code, under Kaitlin Moore's direction; each change is reviewed before merge. Checks so far: the pytest suite (reference examples from Vallado et al. 2006 for frames; leap-second and TT-UTC cases; a byte-for-byte format check and a comparison of the computed trailing 81-day F10.7 mean against CelesTrak's own column for the space-weather parser), and ruff.
