# Brazil (ONS)

**Data source:** ONS `FATOR_CAPACIDADE` hourly capacity factor per complex, plus
the ONS constrained-off series. **Adapter:** `ons-br` · turbine-level ·
unit = complex · open.

The scorecard row covers 151 complexes (140 scored) on the ONS interconnected
system. The result is in the [scorecard](../scorecard.md) and the
[results page](../results.md#united-states-and-brazil).

## Source and licence

The capacity factors, the complex coordinates and capacities, and the
constrained-off series come from the ONS open-data portal, under ONS Open Data
terms; the optional commissioning dates come from ANEEL SIGA (Dados Abertos).
The [source table](../guides/data-sources.md#source-urls) records each. Nothing
from them is committed, because `input/` is git-ignored; this runbook rebuilds
it. Results computed from them may be shared.

## 1. Observations and metadata

| File | Source | Notes |
| --- | --- | --- |
| `FATOR_CAPACIDADE-2_<YYYY>_<MM>.csv` | ONS Open Data (`dados.ons.org.br`, dataset `fator-capacidade-2`; mirrored on AWS S3) | capacity factor, capacity and coordinates per complex |
| `RESTRICAO_COFF_EOLICA_<YYYY>_<MM>.csv` | ONS Open Data (dataset `restricao_coff_eolica_usi`; 2021 onwards) | the constrained-off series, for the curtailment mask |
| `siga-empreendimentos-geracao.csv` | ANEEL Dados Abertos (SIGA) | optional commissioning dates |

The ONS CKAN API lists every resource URL, for example
`dados.ons.org.br/api/3/action/package_show?id=fator-capacidade-2`. Process
every year from the first training year to the test year (2021 to 2024 in
`configs/regions/br.toml`):

```bash
python scripts/process/ons_br.py \
    --fc   input/raw/ons/FATOR_CAPACIDADE*_2021*.csv \
           input/raw/ons/FATOR_CAPACIDADE-2_2022_*.csv \
           input/raw/ons/FATOR_CAPACIDADE-2_2023_*.csv \
           input/raw/ons/FATOR_CAPACIDADE-2_2024_*.csv \
    --coff input/raw/ons/RESTRICAO_COFF_EOLICA_2021_*.csv \
           input/raw/ons/RESTRICAO_COFF_EOLICA_2022_*.csv \
           input/raw/ons/RESTRICAO_COFF_EOLICA_2023_*.csv \
           input/raw/ons/RESTRICAO_COFF_EOLICA_2024_*.csv \
    --siga input/raw/ons/siga-empreendimentos-geracao.csv
```

It writes `br_md.csv`, `br_fc.csv`, `br_curtailment_mask.csv` and
`join_report.md` under `<input root>/observations/turbine/BR/` (override with
`--out`). The shell expands these patterns, and in zsh one that matches no
file stops the command. The 2021 pattern also matches an annual
`FATOR_CAPACIDADE_2021.csv`, so keep only one form of the 2021 files in the
directory. Check that `br_fc.csv` runs from January 2021 to December 2024. A complex-month is masked
when its curtailed fraction exceeds `--curtailment-threshold` (default 0.05).
ONS timestamps are Brasilia time (UTC-3), converted to UTC at ingest.

ONS carries no turbine model or hub height, so every complex gets one uniform
curve, the `--model` default `2019COE_Market_Average_2.6MW_121`, at the
`--height` default of 100 m. Override with `--model <key>` (any column of the
input root's `power_curves.csv`) or `--height`.

## 2. ERA5

```bash
python scripts/fetch/era5.py --region br --dry-run   # inspect the request plan
python scripts/fetch/era5.py --region br             # 48 months, 2021-2024
python scripts/era5/combine.py --region br           # yearly daily files
```

The box is `bbox = [-74.0, -34.0, -34.0, 6.0]` ([W, E, S, N]). The config reads
the combined daily files in `era5/BR_daily` with `roughness = "stored"`; the rest
is in [data sources, section 3](../guides/data-sources.md#3-shared-inputs-all-regions).

## 3. Train and evaluate

```bash
# The scorecard row (k = 60; the row reports k60 fixed)
PYVWF_INPUT=input/combined python scripts/dev/run_locked.py -- \
    pyvwf-validate train --region configs/regions/scorecard/br_k60.toml
PYVWF_INPUT=input/combined python scripts/dev/run_locked.py -- \
    pyvwf-validate evaluate --region configs/regions/scorecard/br_k60.toml \
    --train-run output/validation/BR/train-<timestamp>
# The maintained config (cluster_list = [1, 20]; no k = 60)
PYVWF_INPUT=input/combined pyvwf-validate train --region configs/regions/br.toml
PYVWF_INPUT=input/combined pyvwf-validate evaluate --region configs/regions/br.toml \
    --train-run output/validation/BR/train-<timestamp>
```

Both use the fixed and season slices, train on 2021-2023 and test on 2024. The
maintained sweep is the [sweep knee](../results.md#cluster-count-per-region);
transfer runs are out of scope; the rest is in [training](../guides/training.md).

## Caveats

- **The scorecard row's fit is degenerate (†):** check `fit_quality` before
  using a per-cluster factor; see the [scorecard](../scorecard.md).
- **Curtailment is the dominant contaminant,** hardest in the Nordeste. The
  mask removes only complex-months above the threshold.
- **Uniform curve.** The curve is a fleet representative, not any complex's
  own machine, and is not matched by specific power.
- **Complex, not plant.** A complex's coordinate is its collector substation.
  The ONS `ceg` field is often `-`, so the SIGA commissioning join is best-effort.
- **Southern Hemisphere and tropical.** Read the season slice against the
  Nordeste's trade-wind maximum (about June to November), not a thermal season.
