# Denmark (Danish Energy Agency)

**Data source:** Danish Energy Agency master data register, per-turbine monthly kWh. **Adapter:** `european-turbine` · turbine-level · unit = turbine · open.

Denmark is the reference turbine-level region: an open national register with
exact per-turbine coordinates and monthly output, so it needs no GWPT join.
Its result is the DK row of the [scorecard](../scorecard.md); the choice of
cluster count is in [Denmark's cluster count](../results.md#denmarks-cluster-count).

## Source and licence

The **Master data register of wind turbines** ("Stamdataregister for
vindkraftanlæg"), published by the Danish Energy Agency (Energistyrelsen) at
<https://ens.dk/en/analyses-and-statistics/overview-energy-sector>. Open public
data, free to use with attribution. Nothing from it is committed: `input/` is
git-ignored.

| File | ens.dk media | Content |
| --- | --- | --- |
| `anlaeg.xlsx` | `/media/4945/download` | One row per turbine (GSRN id, connection date, capacity kW, rotor, hub height, UTM X/Y, model), "existing" and "decommissioned" sheets. Header on row 9. |
| `maanedsdata_2002_2020.xlsx` | `/media/4948/download` | Monthly kWh per turbine, one sheet per year (`Månedsprod_<year>`, 2002-2020). Header on row 7. |

The pinned media ids are the snapshot of ultimo January 2022 the inputs were
built on. ens.dk names the monthly file `maanedsdata_2002_2017.xlsx`, but it
holds 2002-2020; the fetch saves it as `_2020`. A newer register (pass
`--anlaeg-url <current link>`) only adds post-2022 turbines, so keep the default.

## 1. Observations

No credentials are needed.

```bash
python scripts/fetch/dk.py       # anlaeg.xlsx + maanedsdata -> input/observations/turbine/DK/
python scripts/process/dk.py     # -> dk_md.csv, dk_obs_2002_2020.csv
```

`--dry-run` on the fetch prints the plan; downloads resume. The processor takes
`--metadata-only` or `--observations-only`. On the pinned files `dk_md.csv` has
6,296 turbines and `dk_obs_2002_2020.csv` 1,229,633 turbine-month rows.

## 2. ERA5

Denmark slices the shared European box, `era5/EU_2026-09`, to
`bbox = [7.5, 13.5, 54.0, 58.2]` ([data sources, section 3](../guides/data-sources.md#3-shared-inputs-all-regions)):

```bash
python scripts/fetch/era5.py --code eu --file-tag EU_2026-09 \
    --bbox -12 31.5 36 72 --years 2015 2016 2017 2018 2019 2020 2021 2022 2023
```

## 3. Train and evaluate

Denmark trains on 2015-2019 and tests on 2020, on the licensed curve library
([input root](../guides/training.md#choose-the-input-root)). The scorecard row
reruns from `configs/regions/scorecard/dk_k100.toml` (k=100, season slices), at
the row's commit through `scripts/dev/run_locked.py` ([Reproducing a row](../scorecard.md#reproducing-a-row)):

```bash
PYVWF_INPUT=input/combined pyvwf-validate train \
    --region configs/regions/scorecard/dk_k100.toml
PYVWF_INPUT=input/combined pyvwf-validate evaluate \
    --region configs/regions/scorecard/dk_k100.toml --train-run output/validation/DK/train-<stamp>
```

The maintained config, `configs/regions/dk.toml`, sweeps `cluster_list = [1, 100]`
over fixed and season slices. It does not set `allow_extrapolation`, so it
stops with `ExtrapolationError` on the Bornholm units.

## Caveats

- **Bornholm lies outside the box.** Bornholm is near 14.9°E, beyond the box's
  13.5°E edge. `dk_k100.toml` sets `allow_extrapolation = true`, so its winds
  there are extrapolated (the scorecard's § note); `dk.toml` keeps the default,
  `false`, and stops with an `ExtrapolationError`
  ([output structure](../guides/output-structure.md)).
- **One test year, licensed curves.** The row rests on 2020 alone and needs the
  licensed library, so it is not third-party reproducible.
