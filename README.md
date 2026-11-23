# orbit-correction-screening

A PhysicsNeMo model that learns the error of SGP4 against a high-fidelity numerical propagator, tested on whether corrected predictions screen low-Earth-orbit conjunctions better than uncorrected SGP4 and classical geometric filters.

Status: core modules only (contracts, frames and time scales, catalog loading and filters, space weather, splits). No propagation, model or results yet.

## Question

Cheap propagators (SGP4 on public element sets) are fast but drift; high-fidelity numerical propagation is accurate but expensive across a catalog. This project trains a model in NVIDIA PhysicsNeMo to predict SGP4's error relative to a numerical propagator (Orekit: zonal gravity and NRLMSISE-00 drag driven by observed space weather), then asks: (1) how much the correction reduces position error as a function of prediction horizon; (2) whether corrected predictions recover the close approaches (under 5 km over 7 days) that numerical screening finds, with fewer false alarms than uncorrected SGP4; and (3) whether that beats classical geometric filters. "Truth" is a numerical model, not tracked positions, and missed conjunctions count as the costly error.

## Data sources

Data is gitignored and never committed. Downloads go under `data/raw/`.

| Source | URL | Version | Access date | License / terms |
|---|---|---|---|---|
| CelesTrak GP element sets (OMM JSON) | https://celestrak.org/NORAD/elements/gp.php (e.g. `?CATNR=26998&FORMAT=json`, `?GROUP=active&FORMAT=json`) | live data; epochs as of the access date | 2026-11-23 | Free, no account. Small extracts are committed as test fixtures (`tests/fixtures/README.md`) |
| CelesTrak SATCAT | https://celestrak.org/pub/satcat.csv | live data | 2026-11-23 | Free, no account |
| CelesTrak space weather (F10.7, Kp, Ap) | https://celestrak.org/SpaceData/SW-All.txt (format: https://celestrak.org/SpaceData/SpaceWx-format.php) | file header `UPDATED 2026 Nov 23 10:32:51 UTC` | 2026-11-23 | Free, no account |
| Space-Track `gp_history` (historical element sets) | https://www.space-track.org (`basicspacedata/query/class/gp_history`) | per download, recorded in `data/raw/spacetrack/manifest.jsonl` | not yet downloaded | Free account; Space-Track User Agreement. This project never commits or redistributes Space-Track data |
| IERS Earth orientation data and leap seconds, via `astropy-iers-data` | https://pypi.org/project/astropy-iers-data/ (data from https://www.iers.org) | 0.2026.11.23.0.35.53, pinned in `uv.lock` | 2026-11-23 | Package BSD-3-Clause; IERS data freely available |

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

## Results

None yet.

## Methodology

- [`docs/methodology.md`](docs/methodology.md): approach, conventions, data, splits.
- [`docs/changelog.md`](docs/changelog.md): every instrument and design change with its reason.
- Contracts between components: [`src/orbit_correction/types.py`](src/orbit_correction/types.py), checked by `tests/test_contracts.py`.
- Catalog filters and split values (approved 2026-11-23): [`config/`](config/).

What is implemented here and what is library behaviour:
- SGP4 is the `sgp4` package (Vallado's implementation). TEME to GCRF rotations and time-scale arithmetic are astropy and ERFA. This project adds the conventions and wrappers, EOP pinning and range checks, and tests against the Vallado et al. (2006) reference examples.
- The space-weather parser, the leak-free as-of lookup, the Orekit-compatible writer, the catalog loaders and filters, and the split logic are this project's own code.

## Limitations and scope

- Truth is a numerical model (Orekit), not tracked positions.
- v1 covers LEO, non-maneuvering objects only, and the non-maneuvering rule is approximate. None of its parts is a historical maneuver record:
  - The status part uses each object's *current* SATCAT status, not its status at each epoch.
  - It misclassifies non-propulsive operational satellites as maneuverable. TIMED is one example (held out anyway as a validation object).
  - The constellation name list covers only the constellations named in `config/catalog_filters.toml`.
  - The element-history detector misses small maneuvers: semimajor-axis changes under about 1 km plus the drag allowance, inclination changes under 0.05 deg, and anything hidden in a long gap between element sets.
- Prediction windows ending after the last fully observed space-weather day are excluded, so the most recent epochs are never used.
- The space-weather file is a snapshot of final, revised values. The as-of lookup prevents reading values after the prediction start, but cannot reproduce what an operator knew in real time.
- Catalog filter values and split parameters are approved (2026-11-23); no split has been generated from real data yet.

## AI assistance

This repository is built with AI assistance. Code, tests, fixtures and documentation in the current state were written by Claude (Anthropic) through Claude Code, under Kaitlin Moore's direction; each change is reviewed before merge. Checks so far: the pytest suite (reference examples from Vallado et al. 2006 for frames; leap-second and TT-UTC cases; a byte-for-byte format check and a comparison of the computed trailing 81-day F10.7 mean against CelesTrak's own column for the space-weather parser), and ruff.
