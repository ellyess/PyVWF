# United States (EIA)

**Data source:** EIA-923 net generation, EIA-860 nameplate and coordinates,
USWTDB hub heights and rotors. **Adapter:** `eia-us` · turbine-level ·
unit = plant · open.

The scorecard row covers 520 plants (512 scored) in the contiguous states. The
result is in the [scorecard](../scorecard.md) and the [results page](../results.md#united-states-and-brazil).

## Source and licence

The observations and metadata are US public-domain data: EIA-923 and EIA-860
from the Energy Information Administration, and hub heights and rotor
diameters from the US Wind Turbine Database (USGS). The
[source table](../guides/data-sources.md#source-urls) records each. Nothing
from them is committed, because `input/` is git-ignored; this runbook rebuilds
it. Results computed from them may be shared.

## 1. Observations and metadata

| File | Source |
| --- | --- |
| `f923_<year>.zip` (EIA-923) | `eia.gov/electricity/data/eia923/` (recent) or `.../archive/xls/` (older) |
| `eia860<year>.zip` (EIA-860) | `eia.gov/electricity/data/eia860/` (recent) or `.../archive/xls/` (older) |
| `uswtdbCSV.zip` (USWTDB) | `energy.usgs.gov/uswtdb/data/` |

Pass one EIA-923 workbook per year, 2019 to 2022 (`configs/regions/us.toml`),
and one EIA-860 vintage. Run it under the run's input root, because curve
matching reads that root's `models.csv`:

```bash
PYVWF_INPUT=input/combined python scripts/process/eia_us.py \
    --eia923 input/raw/eia/f923_2019/EIA923_Schedules_2_3_4_5_M_12_2019_*.xlsx \
             input/raw/eia/f923_2020/EIA923_Schedules_2_3_4_5_M_12_2020_*.xlsx \
             input/raw/eia/f923_2021/EIA923_Schedules_2_3_4_5_M_12_2021_*.xlsx \
             input/raw/eia/f923_2022/EIA923_Schedules_2_3_4_5_M_12_2022_*.xlsx \
    --eia860-plants     input/raw/eia/eia860_2022/2___Plant_Y2022.xlsx \
    --eia860-generators input/raw/eia/eia860_2022/3_1_Generator_Y2022.xlsx \
    --uswtdb            input/raw/eia/uswtdb/uswtdb_V9_0_*.csv \
    --bbox -125 -66 24 50
```

It writes `us_md.csv`, `us_eia923_netgen.csv` and `join_report.md` under
`<input root>/observations/turbine/US/` (override with `--out`). Read the join
report afterwards for unmatched plants and hub-height and curve counts.

- **`--bbox` must match the region's `[era5] bbox`,** or Alaska and Hawaii
  plants are silently simulated with the nearest in-domain cell's wind.
- **Hub heights are binned to 10 m** (`--height-bin`, default 10; 0 disables);
  unbinned USWTDB heights run out of memory.
- **Annual respondents are dropped.** Plant-years flagged `A` in EIA-923 carry
  EIA-imputed monthly values and are set to NaN by default; months before a
  plant's commissioning date are masked too.
- **Curves are matched** by default (`--curve-assignment match`): each plant gets
  the onshore curve of similar rating nearest its specific power. Plants with no
  USWTDB rotor or turbine count fall back to `--model` (default
  `2019COE_Market_Average_2.6MW_121`), recorded in `us_md.csv`'s `model_source`.
  `--curve-assignment uniform` gives every plant `--model`.

## 2. ERA5

```bash
python scripts/fetch/era5.py --region us --dry-run   # inspect the request plan
python scripts/fetch/era5.py --region us             # 48 months, 2019-2022
python scripts/era5/combine.py --region us           # yearly daily files
```

The box is `bbox = [-125.0, -66.0, 24.0, 50.0]` ([W, E, S, N]). The config reads
the combined daily files in `era5/US_daily` with `roughness = "stored"`; the rest
is in [data sources, section 3](../guides/data-sources.md#3-shared-inputs-all-regions).

## 3. Train and evaluate

```bash
# The scorecard row (k = 250; the row reports k250 fixed)
PYVWF_INPUT=input/combined python scripts/dev/run_locked.py -- \
    pyvwf-validate train --region configs/regions/scorecard/us_k250.toml
PYVWF_INPUT=input/combined python scripts/dev/run_locked.py -- \
    pyvwf-validate evaluate --region configs/regions/scorecard/us_k250.toml \
    --train-run output/validation/US/train-<timestamp>
# The maintained config (cluster_list = [1, 25]; no k = 250)
PYVWF_INPUT=input/combined pyvwf-validate train --region configs/regions/us.toml
PYVWF_INPUT=input/combined pyvwf-validate evaluate --region configs/regions/us.toml \
    --train-run output/validation/US/train-<timestamp>
```

The scorecard row needs `input/combined`: most of its capacity is on curves only
the licensed library holds. Both use the fixed and season slices, train on
2019-2021 and test on 2022. The maintained sweep is the [sweep knee](../results.md#cluster-count-per-region);
transfer runs are out of scope; the rest is in [training](../guides/training.md).

## Caveats

- **The scorecard row's fit is degenerate (†):** check `fit_quality` before
  using a per-cluster factor; see the [scorecard](../scorecard.md).
- **Curtailment is not screened:** ERCOT and SPP curtailment stays in observed CF.
- **Curves are matched by specific power, not machine identity.** The USWTDB
  model string (`us_md.csv`'s `uswtdb_model`) is never the curve key.
- **Net, not gross:** EIA-923 `Netgen` excludes station use, which the correction absorbs.
- **Static nameplate.** Capacity is the EIA-860 nameplate, fixed over the
  window, so staged build-outs bias a plant's early months low.
