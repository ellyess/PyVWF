# Spain (WindStats) region — processing runbook

**⚠ MIXED LICENCE.** Generation is **CONFIDENTIAL / commercial WindStats**;
coordinates are **open GWPT** (CC-BY-4.0). Neither the data nor the derived
observation/metadata tables may be committed or redistributed. `input/` is
git-ignored; the script below only reshapes files you already hold.

**Data caveat — read first.** The WindStats Spain extract is **1998-2000
only** (an old ~600 kW fleet). This is a *historical, three-year* region, not a
contemporary of the 2015-2019 reference set — useful as a legacy check, not a
modern benchmark. It also needs ERA5 for 1998-2000 (not the repo's 2015+
`era5/EU`). The region code is **`ES-WS`** so it does not collide with the
country-level ENTSO-E `ES`.

## Build the inputs

```bash
python scripts/process/windstats.py --country ES --src "<WindStats folder>"
```

Reads `ES_md.csv`, `ES_data.csv`, `geolocate.spain.csv` from `--src` and the
Global Wind Power Tracker, and writes to `input/turbine_level_data/ES/`:
`es_md.csv`, `es_obs.csv`, `es_join_report.md`.

The pipeline: WindStats per-turbine monthly `Output` (kWh) → CF against
nameplate kW; coordinates via **WindStats id → `geolocate` thewindpower name →
GWPT project name** (fuzzy match). On the current extract: **1,158 of 1,450
turbines geolocate (80%)**; the 291 turbines whose farm has no GWPT match are
dropped and reported. Canary-Islands turbines (lat < 30) fall outside the
mainland ERA5 box and are excluded. WindStats manufacturer strings are kept as
provenance (`windstats_manufacturer`) but are not curve keys — the curve key is
the uniform default (as BR/CL/AR do), matched by specific power downstream via
`diameter`.

## Train (needs 1998-2000 ERA5)

```bash
python scripts/fetch/era5.py --region es-ws          # 1998-2000 mainland Spain
PYVWF_INPUT=input python scripts/analysis/validate_region.py train    --region configs/regions/es_ws.toml
PYVWF_INPUT=input python scripts/analysis/validate_region.py evaluate --region configs/regions/es_ws.toml --train-run output/validation/ES-WS/train-<stamp>
```

## Sweden & Finland

`SE`/`FI` share the WindStats format and this exact pipeline, but **GWPT only
matches 7-9% of their farms** (it under-covers the fragmented Nordic fleets),
so they are not usable via the open coordinate path. Supply the
thewindpower.net coordinate table the `twp` names point at, register `SE-WS`/
`FI-WS` in `vwf/sources/windstats.py`, and run
`scripts/process/windstats.py --country SE`. Their extracts cover SE 1998-2013,
FI 2005-2012.
