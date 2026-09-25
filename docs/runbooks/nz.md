# New Zealand (EMI)

**Data source:** EA EMI `Generation_MD`, half-hourly metered injection.
**Adapter:** `emi-nz` · turbine-level · unit = farm · open.
**Fleet:** 13 dispatched farms, about 1.5 GW, with per-farm hub heights.

All inputs are open Electricity Authority data, which needs no registration, or
come from your own CDS account for ERA5. No raw data or processed input is
committed, because `input/` is git-ignored. This runbook is how you build them.

NZ was the first pick of the July 2026 dataset survey
(`docs/findings/dataset-survey.md`). It has open half-hourly metered generation
per plant. Its Southern-Hemisphere temperate-westerly climate over complex
terrain is not yet in the validation set. It also has per-farm hub heights, which no other region outside
Europe has, apart from Canada.

## 0. What is already committed

- `configs/curation/nz_wind_farms.csv`: the curated farm table, 13 dispatched
  farms. It holds each farm's Gen_Code and POC keys into the EMI files, its
  coordinates, final-build capacity, turbine model and hub height.
  `height_source` marks the three unverified hub heights: tararua_3,
  mill_creek and kaiwera_downs_2. The table was compiled in July 2026 from
  NZWEA, operator, Wikipedia and EMI register sources. It has no per-row
  source column yet.
- `configs/curation/nz_capacity_stages.csv`: stable capacity plateaus for
  staged builds. At present this is Turitea, from 118.8 MW (North only) to
  221.4 MW (full).
- `configs/curation/nz_mask_windows.csv`: commissioning months to mask at load,
  for Waipipi, Turitea (twice), Harapaki, and Kaiwera Downs 1 and 2.
- The adapter (`pyvwf/sources/emi_nz.py`), the transforms
  (`pyvwf/datasets/emi_nz.py`) and their tests. Must-distinguish tests pin the
  trading-period mapping on daylight-saving (DST) days.

Three known exclusions are documented:

- **Mahinerangi** is metered inside the Waipori hydro scheme. It never appears
  as wind in `Generation_MD`.
- **Seven small embedded farms** connect to the distribution network, so they
  are outside the dispatched dataset. They are Brooklyn, Hau Nui, Mt Stuart,
  Flat Hill, Horseshoe Bend, Weld Cone and Lulworth, about 28 MW in total.
- **Te Rere Hau** is included, but degraded late in the window: 5 turbines
  stopped and 2 derated. Its observed CF understates the resource. This is a
  standing caveat, and it could be excluded later.

## 1. Observations (user-executed, no credentials)

```bash
python scripts/fetch/emi_nz.py            # 72 monthly CSVs 2019-2024 + register
python scripts/process/emi_nz.py          # -> input/observations/turbine/NZ/
```

The fetch uses plain HTTP. Each URL returns an HTTP 302 redirect to an open
Azure blob, about 0.4 to 0.9 MB per month.

The processing step does the following:

- It selects wind rows, where `Fuel_Code` is Wind or WIN.
- It keys on `Gen_Code`, normalised for case. `Site_Code` is not stable across
  years.
- It maps trading periods to UTC through `Pacific/Auckland`. Days of 46, 48 and
  50 periods (DST days) are handled, and tests pin them.
- It sums farms with several POCs: West Wind, and Tararua I and II.
- It computes monthly CF against the capacity history described below.
- It writes the build mask of commissioning-ramp months from `nz_mask_windows.csv`.

An unmapped wind `Gen_Code` stops processing with an error. It means a new farm
needs a curated row. Kaiwaikawe (Northland, 77 MW) is expected to appear around
mid-2026.

Next, read `input/observations/turbine/NZ/join_report.md`. Check the farm count
(13), the capacity (about 1.5 GW), the matched-curve count and the masked
months. Trust nothing before this check.

### Capacity-factor denominator

Each farm's monthly CF is divided by a stable-plateau capacity history. It is
built from the curated tables, not from the EMI plant register:

- For a staged build, the capacity comes from `nz_capacity_stages.csv`.
- For any other farm, it is the farm's final capacity in `nz_wind_farms.csv`,
  from its first generation.
- Months in `nz_mask_windows.csv` are masked.

The EMI register is fetched for the join report only. The curated tables have
no per-unit confidence column yet.

## 2. ERA5 (user-executed, your CDS key)

```bash
python scripts/fetch/era5.py --region nz           # 72 months, 2019-2024, NZ box
```

The box is small, 53 by 49 cells. So it needs no daily combine step, unlike the
BR and US boxes.

## 3. Train and evaluate

```bash
PYVWF_INPUT=<input root> python scripts/analysis/validate_region.py train \
    --region configs/regions/nz.toml
PYVWF_INPUT=<input root> python scripts/analysis/validate_region.py evaluate \
    --region configs/regions/nz.toml --train-run output/validation/NZ/train-<stamp>
```

Set `PYVWF_INPUT` on both commands. Evaluation resolves the curve library again.
Without the variable, it silently uses the open library. The manifest records
which library each run used.

Notes for reading the result:

- **Bias-structure diagnosis first** (the D2 lesson). Check the uncorrected MBE
  before judging the correction. NZ capacity factors are among the world's
  highest, about 40%. ERA5 probably under-resolves the resource in complex
  terrain, such as the Manawatu Gorge and Cook Strait funnelling. This is the
  Tehachapi-like regime the ML transfer re-test identified as under-represented
  globally.
- **The cluster count is limited.** Only 8 of the 13 farms reach the clusterer
  in the 2019-2023 training years. Harapaki and Kaiwera Downs 2 commission
  later, and some farms with sparse coverage drop in `train_set`. k-means needs
  a cluster count no larger than 8.
  - k=10 crashed for that reason (`region-nz.md`).
  - A cluster count near 8 puts one farm in each cluster: the fake-plateau
    regime (`region-us-br.md`).
  - The maintained config sweeps `cluster_list = [1, 5]`.
  - The scorecard row is k7 fixed. Its scorecard config is
    `configs/regions/scorecard/nz_k7.toml`.
- **Training years 2019-2023, test year 2024.** Turitea contributes 2022 (the
  North plateau) and 2024. Its 2021 and 2023 ramps are masked. Harapaki
  effectively enters in the test year, masked until July 2024. Watch its months
  in evaluation.
- **Curtailment is not screened.** Metered injection is net of availability. NZ
  has little economic wind curtailment in this window, because hydro dominates
  the system. No curtailment screen exists, unlike for BR. This is a standing
  caveat.

## 4. Refresh

- **New months:** re-run both fetch scripts. They skip files that already
  exist.
- **New farms** (Kaiwaikawe, Mt Munro, a Te Rere Hau repowering): add a row to
  `configs/curation/nz_wind_farms.csv`.
  - For a staged farm, also add rows to `nz_capacity_stages.csv` and
    `nz_mask_windows.csv`.
  - Processing fails with an error until the row exists.
- **New register files:** the register filename carries its publication date,
  about every six months. The fetch script finds the newest in the directory.
- **Dataset changes:** EMI plans to replace `Generation_MD` with a richer
  dataset.
  - A fetch that returns 404 is the sign. Then check the EMI dataset page.

## Licence

The generation data are open Electricity Authority EMI datasets, downloaded
without registration (the [source table](../guides/data-sources.md#source-urls)). Nothing from them is committed. The committed
curation tables are `configs/curation/nz_wind_farms.csv`,
`nz_capacity_stages.csv` and `nz_mask_windows.csv`. The farm table records
where each hub height came from in `height_source`, and other notes in
`notes`; it has no general per-row source column.
