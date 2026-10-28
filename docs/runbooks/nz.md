# New Zealand (EMI)

**Data source:** Electricity Authority EMI `Generation_MD`, half-hourly metered
injection per plant. **Adapter:** `emi-nz` · turbine-level · unit = farm ·
open.

New Zealand's dispatched wind fleet: 13 farms, about 1.35 GW at final build,
with per-farm hub heights. Training years 2019-23, test year 2024. Its result
is the NZ row of the [scorecard](../scorecard.md) and the
[results entry](../results.md#new-zealand).

## Source and licence

The generation data are open Electricity Authority EMI datasets, downloaded
without registration (the
[source table](../guides/data-sources.md#source-urls)). Nothing from them is
committed. ERA5 needs your own CDS account. The committed curation tables are:

- `configs/curation/nz_wind_farms.csv`: the 13 dispatched farms, with each
  farm's Gen_Code and POC keys into the EMI files, coordinates, final-build
  capacity, turbine model and hub height. `height_source` marks the three
  unverified hub heights (tararua_3, mill_creek, kaiwera_downs_2); other notes
  are in `notes`. There is no general per-row source column.
- `configs/curation/nz_capacity_stages.csv`: capacity plateaus for staged
  builds; at present Turitea, 118.8 MW (North only) then 221.4 MW (full).
- `configs/curation/nz_mask_windows.csv`: commissioning months to mask at load,
  for Waipipi, Turitea (twice), Harapaki, and Kaiwera Downs 1 and 2.

## 1. Observations

```bash
python scripts/fetch/emi_nz.py            # 72 monthly CSVs 2019-2024 + register
python scripts/process/emi_nz.py          # -> input/observations/turbine/NZ/
```

Processing sums multi-POC farms (West Wind, Tararua I and II), takes monthly
CF against the curated capacity history (the final capacity, or the stage table
for a staged build; not the EMI register),
masks the commissioning months, and assigns curves by rating band, then
specific power, from the library of the input root it runs under. Run it under
the default root: the scorecard row's farms are on curves the open library
holds. An unmapped wind `Gen_Code` stops it with an error: add a curated row
for the new farm, or an entry in `KNOWN_ABSENT` in
`src/pyvwf/datasets/emi_nz.py` if it never carries wind rows.

Then read `input/observations/turbine/NZ/join_report.md`: check the farm count
(13), the final-build capacity (1.35 GW), the matched-curve count and the
masked months.

**Known exclusions.** Mahinerangi is metered inside the Waipori hydro scheme,
never as wind in `Generation_MD`. Seven small embedded farms (Brooklyn, Hau
Nui, Mt Stuart, Flat Hill, Horseshoe Bend, Weld Cone, Lulworth; about 28 MW)
are on the distribution network, outside the dispatched dataset.

### Refresh

- **New months:** re-run `scripts/fetch/emi_nz.py` and
  `scripts/fetch/era5.py --region nz`; both skip existing files. A year outside
  2019-2024 needs `--years` on each fetch and on the processing step.
- **New farms** (Kaiwaikawe, Mt Munro, a Te Rere Hau repowering): add a row to
  `configs/curation/nz_wind_farms.csv`, and for a staged farm rows to
  `nz_capacity_stages.csv` and `nz_mask_windows.csv`.
- **Dataset changes:** EMI plans to replace `Generation_MD`; a fetch returning
  404 across the board is the sign to check the EMI dataset page.

## 2. ERA5

Box `[166.0, 179.0, -48.0, -36.0]`, west of the antimeridian, written to
`<input root>/era5/NZ/`, the config's ERA5 path, so no combine step. The rest
is in [Shared inputs](../guides/data-sources.md#3-shared-inputs-all-regions).

```bash
python scripts/fetch/era5.py --region nz           # 72 months, 2019-2024
```

## 3. Train and evaluate

The scorecard row reruns from `configs/regions/scorecard/nz_k7.toml` (k7, fixed
slice) on the combined library
([Reproducing a row](../scorecard.md#reproducing-a-row),
[Choose the input root](../guides/training.md#choose-the-input-root)). The
maintained `configs/regions/nz.toml` sweeps `cluster_list = [1, 5]` over fixed
and season slices. Seasons are Southern-Hemisphere (winter June-August).

```bash
PYVWF_INPUT=input/combined pyvwf-validate train \
    --region configs/regions/scorecard/nz_k7.toml
PYVWF_INPUT=input/combined pyvwf-validate evaluate \
    --region configs/regions/scorecard/nz_k7.toml \
    --train-run output/validation/NZ/train-<stamp>
```

**Keep the cluster count at 8 or below.** Only 8 of the 13 farms reach the
clusterer in 2019-2023 (Harapaki and Kaiwera Downs 2 commission later; sparse
farms drop in `train_set`), and k=10 crashed. Near 8 each cluster holds one
farm, the fake-plateau regime under
[United States and Brazil](../results.md#united-states-and-brazil); the
scorecard's k7 is close. Turitea contributes 2022 (the North plateau) and
2024, its 2021 and 2023 ramps masked; Harapaki enters in the test year, masked
until July 2024.

## Caveats

- **The gain is not resolved (‡):** resampling the test farms gives an
  interval that includes zero; see the [scorecard](../scorecard.md).
- **Te Rere Hau is degraded late in the window** (5 turbines stopped, 2
  derated), so its observed CF understates the resource. It is included.
- **Read the uncorrected bias first.** ERA5 may under-resolve complex terrain
  (Cook Strait, the Manawatu Gorge); judge the correction against that bias.
- **Curtailment is not screened**, unlike for BR. Metered injection is net of
  availability; hydro dominance keeps economic wind curtailment low in this
  window.
