# US (EIA) region — data acquisition and validation runbook

End-to-end reproduction for the US region (`configs/regions/us.toml`,
source `eia-us`). All inputs are public-domain US government data or your own
CDS/ERA5 credentials; none of the raw or derived data is committed (`/input`
is git-ignored), so this runbook is how you rebuild it.

The steps split into two halves: the **credential-free** half (EIA/USWTDB
acquisition + processing, fully scripted and verified against the real 2021
files) and the **user-executed** half (ERA5 via your CDS key, and the run
itself with a real power-curve library).

## 1. Observations + metadata (credential-free)

Download the public files (sizes are the 2021 vintage):

| File | Source |
| --- | --- |
| `f923_<year>.zip` (EIA-923) | `eia.gov/electricity/data/eia923/` (recent) or `.../archive/xls/` (older) |
| `eia860<year>.zip` (EIA-860) | `eia.gov/electricity/data/eia860/` (recent) or `.../archive/xls/` (older) |
| `uswtdbCSV.zip` (USWTDB) | `energy.usgs.gov/uswtdb/data/` |

Then build the two on-disk tables the `eia-us` source reads:

```bash
python scripts/process_eia_us.py \
    --eia923 input/eia_raw/f923_2019/EIA923_Schedules_2_3_4_5_M_12_2019_*.xlsx \
             input/eia_raw/f923_2020/EIA923_Schedules_2_3_4_5_M_12_2020_*.xlsx \
             input/eia_raw/f923_2021/EIA923_Schedules_2_3_4_5_M_12_2021_*.xlsx \
    --eia860-plants     input/eia_raw/eia860_2021/2___Plant_Y2021.xlsx \
    --eia860-generators input/eia_raw/eia860_2021/3_1_Generator_Y2021.xlsx \
    --uswtdb            input/eia_raw/uswtdb/uswtdb_V9_0_*.csv \
    --out input/turbine_level_data/US
```

Writes `us_md.csv`, `us_eia923_netgen.csv`, and `join_report.md`. Pass every
year of EIA-923 you intend to train/test on (`--eia923` takes several
workbooks); one EIA-860 vintage supplies the static nameplate + coordinates.

**What the real 2021 file produces** (sanity anchors — verified on the
`Final_Revision` workbook + USWTDB V9):

- 1,279 wind plants in EIA-923; 1,278 join to EIA-860 coordinates (a handful,
  e.g. the `99999` placeholder code, do not — listed in the join report).
- Fleet nameplate ≈ 133 GW; per-plant hub height from USWTDB for 1,168 / 1,278
  plants (mean ≈ 82 m), the rest on the 100 m uniform default (`height_source`
  records which, per plant).
- **777 of the 1,279 plants are annual (`A`) respondents**, whose monthly cells
  are EIA-imputed, not measured. The source drops them by default, leaving
  ≈ 499 monthly-reporting plants as the training fleet (still ~5x DK or AU).
- Fleet capacity-weighted mean CF ≈ 0.34 — the published 2021 US wind number.

## 2. ERA5 (user-executed — needs your CDS key)

```bash
pip install cdsapi          # not a PyVWF dependency; ~/.cdsapirc holds your key
python scripts/fetch_era5_us.py --dry-run     # inspect the request plan
python scripts/fetch_era5_us.py               # 48 months, 2019-2022, CONUS box
python scripts/combine_era5_us_daily.py       # reduce to yearly daily files
```

The CONUS box is ~7x the AU-NEM box, so the monthly files are large and a
multi-year hourly load will not fit in memory — the daily pre-combine is not
optional here. Point `--in-dir/--out-dir` (or the region's `era5_path`) at
wherever the harness should read.

## 3. Power curves (user-executed — licensed, not committed)

The bundled `input/power_curves.csv` / `models.csv` are SYNTHETIC placeholders;
runs against them are not physically meaningful (the loader warns, loudly).
Supply a real curve library as `input/power_curves.real.csv` (git-ignored) and
set the region's `correction.model` / metadata `model` to real curve keys
before drawing any conclusion. The USWTDB manufacturer/model string is carried
in `us_md.csv`'s `uswtdb_model` column for the future vintage-aware assignment,
but is never itself used as the curve key.

## 4. Run the validation harness

```bash
python scripts/validate_region.py train    --region configs/regions/us.toml
python scripts/validate_region.py evaluate --region configs/regions/us.toml \
    --train-run output/validation/US/train-<timestamp>
```

`train` fits `(cluster, slice)` correction factors for every combination in the
config (10 clusters × {fixed, season}); `evaluate` scores the held-out test
year (2022) with the standard skill table. Transfer runs stay out of scope on
this branch (the driver's approved-pair guard is AU↔Europe only).

## Caveats carried into any US result

- **Net, not gross.** EIA-923 `Netgen` is net of station use; the correction
  absorbs a small parasitic-load loss the European monthly *generation* data
  does not.
- **Curtailment is not yet screened.** ERCOT/SPP curtailment contaminates
  observed CF where it occurs. The shared `pyvwf.qc` module (research doc §6),
  generalised from the AU heuristics, is the right home — not yet applied.
- **Static nameplate.** Staged build-outs bias a plant's early months low; the
  commissioning mask removes pre-operating months but not partial staging.
