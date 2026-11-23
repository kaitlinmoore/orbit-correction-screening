# Test fixtures

Small committed inputs for the tests. Regenerate with `uv run python scripts/make_fixtures.py`
(the CelesTrak-derived files need the downloads in `data/raw/celestrak/`; the synthetic ones
don't: `--synthetic-only`). **No Space-Track data is used in any fixture.**

| File | Provenance | Used by |
|---|---|---|
| `sw_quirks.txt` | Synthetic. CelesTrak `SW-All.txt` layout, made-up values. Reproduces the three quirks of the published file: blank F10.7 Q flag in DAILY_PREDICTED rows, blank Kp/Ap/Cp/C9/Q in MONTHLY_PREDICTED rows, no MONTHLY_FIT section. 120 observed days (2024-01-01 to 2024-04-29), 5 daily-predicted days, 3 monthly rows (June to August 2024). Written with explicit field widths, independently of `spaceweather.py`'s formatter. | `test_spaceweather.py` |
| `spacetrack_gp_history_synthetic.json` | Synthetic. Space-Track `gp_history` JSON layout (string values, extra keywords). Two made-up objects (NORAD 90001, 90002), six element sets each; 90002 raises its orbit by about 37 km between its third and fourth sets. | `test_catalog.py` |
| `celestrak_gp_sample.json`, `celestrak_gp_sample.csv` | Derived from CelesTrak GP data (`gp.php?CATNR=...&FORMAT=json` and `GROUP=active&FORMAT=json`), accessed 2026-11-23. The ten validation-case objects plus one object per filter case (ISS, two Starlinks, OneWeb, TDRS 3, IMAGE, Calsphere 2). The CSV is written from the same records with CelesTrak's OMM CSV column names. | `test_catalog.py`, `practice/teme_gcrf.ipynb` |
| `satcat_sample.csv` | Derived from CelesTrak SATCAT (`https://celestrak.org/pub/satcat.csv`), accessed 2026-11-23: the rows for the objects above. | `test_catalog.py` |
