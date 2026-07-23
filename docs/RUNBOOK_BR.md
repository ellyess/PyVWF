# Brazil (ONS) region — data acquisition and validation runbook

End-to-end reproduction for the Brazil region (`configs/regions/br.toml`,
source `ons-br`). All inputs are open government data (ONS + ANEEL) or your own
CDS/ERA5 credentials; none of the raw or derived data is committed (`/input`
is git-ignored), so this runbook is how you rebuild it.

The steps split into the **credential-free** half (ONS/ANEEL acquisition +
processing, scripted and verified against the real June-2023 files) and the
**user-executed** half (ERA5 via your CDS key, and the run itself with a real
power-curve library).

## 1. Observations + metadata (credential-free)

ONS publishes wind per *conjunto de usinas* (complex), one hourly capacity-factor
series per `id_ons`. The `FATOR_CAPACIDADE` file is self-contained — it carries
the CF, the time-varying installed capacity, and the collector-substation
coordinates — so it supplies both the observations and the metadata. ANEEL SIGA
is optional (commissioning dates via CEG).

| File | Source | Notes |
| --- | --- | --- |
| `FATOR_CAPACIDADE-2_<YYYY>_<MM>.csv` | ONS Open Data (`dados.ons.org.br`, dataset `fator-capacidade-2`; mirrored on AWS S3) | monthly files, semicolon-delimited, UTF-8, dot decimal |
| `RESTRICAO_COFF_EOLICA_<YYYY>_<MM>.csv` | ONS Open Data (dataset `restricao_coff_eolica_usi`; 2021+) | the constrained-off series, for the curtailment mask |
| `siga-empreendimentos-geracao.csv` | ANEEL Dados Abertos (SIGA) | semicolon, latin-1, decimal-comma; optional commissioning |

The ONS CKAN API lists every resource URL, e.g.
`dados.ons.org.br/api/3/action/package_show?id=fator-capacidade-2`.

Build the on-disk tables the `ons-br` source reads:

```bash
python scripts/process/ons_br.py \
    --fc   input/ons_raw/FATOR_CAPACIDADE-2_2021_*.csv \
           input/ons_raw/FATOR_CAPACIDADE-2_2022_*.csv \
           input/ons_raw/FATOR_CAPACIDADE-2_2023_*.csv \
    --coff input/ons_raw/RESTRICAO_COFF_EOLICA_2021_*.csv \
           input/ons_raw/RESTRICAO_COFF_EOLICA_2022_*.csv \
           input/ons_raw/RESTRICAO_COFF_EOLICA_2023_*.csv \
    --siga input/ons_raw/siga-empreendimentos-geracao.csv \
    --out  input/turbine_level_data/BR
```

Writes `br_md.csv`, `br_fc.csv`, `br_curtailment_mask.csv`, and `join_report.md`.

**What the real June-2023 file produces** (sanity anchors — verified on the
`FATOR_CAPACIDADE-2_2023_06.csv` + `RESTRICAO_COFF_EOLICA_2023_06.csv`):

- 152 wind complexes with coordinates; fleet ≈ 26 GW installed.
- June CF ≈ 0.42 fleet-mean (p5-p95 0.18-0.69) — the Nordeste austral-winter
  trade-wind maximum; no impossible values.
- **Curtailment is severe**: mean curtailed fraction ≈ 0.11, and ~90% of the
  fleet (142/157 complexes) was constrained off > 5% in that month alone. This
  is exactly why the curtailment mask exists — without it the correction is
  fitted to absorb curtailment as if it were reanalysis bias.

## 2. ERA5 (user-executed — needs your CDS key)

```bash
pip install cdsapi          # not a PyVWF dependency; ~/.cdsapirc holds your key
python scripts/fetch/era5.py --region br --dry-run     # inspect the request plan
python scripts/fetch/era5.py --region br               # 48 months, 2021-2024, whole-Brazil box
python scripts/era5/combine.py --region br       # reduce to yearly daily files
```

The box spans the whole country and crosses the equator, so the monthly files
are large and the daily pre-combine is not optional.

## 3. Power curves (bundled open library, or your own)

The bundled `input/power_curves.csv` / `models.csv` are now a REAL open curve
library (NatLabRockies/turbine-models, DOI 10.11578/dc.20210112.1, BSD-3-Clause,
VWF-smoothed), so runs against them are physically meaningful. `process_ons_br.py`
assigns every complex a UNIFORM representative curve — the default
`2019COE_Market_Average_2.6MW_121` (the most recent market-average utility curve),
recorded in `br_md.csv`'s `model_source` column. Override with `--model <key>`
(any column of `power_curves.csv`) for a different representative.

The caveat is identity, not realism: the fleet is matched by specific power, not
by actual machine. ONS carries no hub height or turbine model, so both height and
model are uniform defaults recorded in the `*_source` columns (vintage-aware
per-complex assignment is the same named follow-up as AU/US). For identity-matched
curves, supply your own library as `input/power_curves.real.csv` (git-ignored) and
pass its key.

## 4. Run the validation harness

```bash
python scripts/analysis/validate_region.py train    --region configs/regions/br.toml
python scripts/analysis/validate_region.py evaluate --region configs/regions/br.toml \
    --train-run output/validation/BR/train-<timestamp>
```

`train` fits `(cluster, slice)` correction factors for every combination in the
config (8 clusters × {fixed, season}); `evaluate` scores the held-out test year
(2024). Transfer runs stay out of scope on this branch (the driver's
approved-pair guard is AU↔Europe only).

## Caveats carried into any Brazil result

- **Curtailment is the dominant contaminant.** The constrained-off mask (2021+)
  removes the worst months; pre-2021 CF carries curtailment unscreened. The
  Nordeste is where it bites hardest. This is also the region's scientific draw:
  the constrained-off series lets you *quantify* curtailment (delivered vs
  constrained-off energy) rather than merely caveat it — the natural US/AU
  follow-up is folding this into a shared `pyvwf.qc` module.
- **Complex, not plant.** Corrections are derived at the ONS conjunto level; a
  complex's coordinate is its collector substation, and its capacity is the ONS
  installed value (max over the window). The plant-vs-complex mapping to ANEEL
  CEGs is imperfect (ONS `ceg` is often blank at conjunto level), so SIGA
  commissioning is best-effort.
- **Southern Hemisphere + tropical.** Seasons are stated explicitly (SH) in the
  config, never inherited from the NH default. In the Nordeste the meaningful
  cycle is the trade-wind maximum (~Jun-Nov), not a thermal season — read the
  seasonal-slice result against that, and consider a custom slicing.
- **A verified hub-height wind speed** is reportedly available in a `_detail`
  variant of the constrained-off dataset; this adapter uses the usina-level
  energy account (delivered/curtailed). Wiring the wind-speed column for
  *direct* wind-speed validation is the standout Phase-2 science task (research
  doc §1) and is not yet done.
