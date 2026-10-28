# Germany (WindStats)

**Data source:** WindStats per-turbine register, monthly kWh. **Adapter:** `european-turbine` · turbine-level · unit = turbine · confidential.

Germany is a turbine-level region on a licensed commercial extract, located by
postcode. Its result is the DE row of the [scorecard](../scorecard.md).

## Source and licence

**CONFIDENTIAL / COMMERCIAL DATA.** The German wind data is a licensed
**WindStats** extract, not open data. It (and anything derived from it) must
never be committed or redistributed. `input/` is git-ignored; the script below
only moves files you already hold locally.

Unlike the openly-scriptable regions (NZ, CL, AR, DK-register, UK-REPD),
Germany's inputs are commercial. Results computed from them are fine for
research per the WindStats terms, but the data and derived observation tables
stay out of the repository.

## 1. Observations

```bash
python scripts/process/de.py --src "<path to the WindStats folder>"
python scripts/process/de.py --src "<...>" --check-only   # validate only
```

The script validates the three files, prints aggregates only, and copies them
as they are into `input/observations/turbine/DE/` (or `--out-dir`):

| File | Content |
| --- | --- |
| `DE_md.csv` | Per-turbine metadata: `V1` (id; first 5 chars are the postcode), `Manufacturer`, `kW`, `Rotor..m.`, `Tower..m.` |
| `geolocate.germany.csv` | `postcode -> lon/lat`, so no external coordinate source is needed |
| `DE_data.csv` | Monthly generation: `ID`, `Year`, `Month`, `Output` (kWh), `Downtime` |

`--check-only` on the current extract reports 11,433 turbines, 15.72 GW, 88.9%
postcode coverage, 1,949,625 turbine-months, 1998-2019; another count means
another extract.

## 2. ERA5

Germany slices the shared European box, `era5/EU_2026-09`, to
`bbox = [5.0, 16.0, 47.0, 55.2]` ([data sources, section 3](../guides/data-sources.md#3-shared-inputs-all-regions)):

```bash
python scripts/fetch/era5.py --code eu --file-tag EU_2026-09 \
    --bbox -12 31.5 36 72 --years 2015 2016 2017 2018 2019 2020 2021 2022 2023
```

## 3. Train and evaluate

Germany trains on 2015-2018 and tests on 2019, on the licensed curve library
([input root](../guides/training.md#choose-the-input-root)). The scorecard row
reruns from `configs/regions/scorecard/de_k100.toml` (k=100, fixed slice), at
the row's commit through `scripts/dev/run_locked.py` ([Reproducing a row](../scorecard.md#reproducing-a-row)):

```bash
PYVWF_INPUT=input/combined pyvwf-validate train \
    --region configs/regions/scorecard/de_k100.toml
PYVWF_INPUT=input/combined pyvwf-validate evaluate \
    --region configs/regions/scorecard/de_k100.toml --train-run output/validation/DE/train-<stamp>
```

The maintained config, `configs/regions/de.toml`, sweeps `cluster_list = [1, 50]`
over fixed and season slices, so it does not reproduce the row's k=100.

## Caveats

- **Postcode locations, specific-power curves.** Turbines sit at their postcode
  (`location_resolution = "postcode"`; unmatched ones are dropped), and with no
  model designation their curves go by nearest specific power.
- **One test year, degenerate fit.** The row rests on 2019 alone and carries the
  scorecard's † marker; check `fit_quality` before using its factors.
