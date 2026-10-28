# United Kingdom (Ofgem ROC + REPD)

**Data source:** Ofgem Renewables Obligation certificates (observations) and REPD or curated metadata. **Adapter:** `european-turbine` · turbine-level · unit = farm (ROC station, pseudo-replicated across turbine rows) · mixed: open REPD and RER export, confidential certificate warehouse.

The UK is only partly reproducible from open data: the curated inputs the
adapter reads from `input/observations/turbine/UK/`, `uk_md.csv` and `ukobs.csv`,
came from gated or not openly redistributable sources. Each `process/uk.py`
command below writes a differently named file beside them and never overwrites
them. Its result is the UK row of the [scorecard](../scorecard.md).

## Source and licence

| Component | Curated file | Open and scriptable? |
| --- | --- | --- |
| **Observations**: Ofgem RO certificates → energy | `ukobs.csv` | **Not end-to-end.** Per-station monthly issuance is a gated Ofgem RER export (manual); the conversion and pseudo-replication are scripted. |
| **Metadata**: location, capacity, turbine specs | `uk_md.csv` | **Partly.** REPD (open, OGL) gives location, capacity and turbine count, but no model or rotor and a tip height for only about 13% of sites. |

The **confidential** Ofgem "CertificatesExternalPublicDataWarehouse" per-year
exports are differently licensed, NOT the open RER/REPD data; never commit or
redistribute them or anything derived. REPD is under the Open Government
Licence. `input/` is git-ignored.

## 1. Observations

### Metadata: REPD (scriptable, open)

```bash
python scripts/fetch/uk.py             # scrapes gov.uk, downloads the current REPD CSV
python scripts/process/uk.py metadata  # -> uk_md_open.csv + uk_md_open_divergence.md
```

The fetch saves the current quarterly CSV from the
[REPD publication page](https://www.gov.uk/government/publications/renewable-energy-planning-database-quarterly-extract)
as `input/raw/repd/repd_wind.csv` (`--dry-run` shows the plan) and prints the
manual Ofgem steps below. `metadata` fills the missing specs with a uniform
height (`--height`, 100 m) where REPD has no tip height and a uniform model
(`--model`, default `2019COE_Market_Average_2.6MW_121`). The result is today's
operational fleet, not the 2015-2019 RO-accredited set, and cannot be matched
to `uk_md.csv` row by row; keep `uk_md.csv` for the scorecard row.

### Observations, confidential path: the Ofgem certificate warehouse

If you hold the confidential warehouse exports, they reproduce `ukobs.csv` exactly:

```bash
python scripts/process/uk.py observations \
    --ofgem-confidential "<path>/CertificatesExternalPublicDataWarehouse Wind 20*.csv"
```

Output is `ukobs_confidential.csv`. RO years run April-March, so cover a
calendar year by passing both adjacent RO-year files. Check: station
R00116SQSC's January 2015 total is 1709 MWh, the curated 569,666.7 kWh per
turbine across its three turbine rows.

### Observations, public path: the RER export

1. Go to <https://rer.ofgem.gov.uk/> (View Public Reports, no login), or email
   `renewable.enquiry@ofgem.gov.uk` for the current access link.
2. Export the ROCs-issued report for the Renewables Obligation, filtered to
   wind and your output-period window. Save it as
   `input/ofgem_raw/roc_issuance.xlsx` (or `.csv`).
3. Process it:

```bash
python scripts/process/uk.py observations --roc input/ofgem_raw/roc_issuance.xlsx
```

This converts ROCs to MWh at each technology's default band in
`pyvwf.datasets.uk_roc` (onshore 0.9 ROC/MWh, offshore 2.0), not per-vintage
bands, then equal-splits each station's energy across its turbines using the
counts in `uk_md.csv` (or `--turbine-counts`), into `ukobs_open.csv`. The RER
column names are unverified: pass `--station-col`, `--period-col` and
`--certs-col` if matching fails. RER keeps about 7 years of issuance, so the
full 2015-2019 window may need archived reports.

## 2. ERA5

The UK slices the shared European box, `era5/EU_2026-09`, to
`bbox = [-11.0, 3.0, 49.0, 61.0]` ([data sources, section 3](../guides/data-sources.md#3-shared-inputs-all-regions)):

```bash
python scripts/fetch/era5.py --code eu --file-tag EU_2026-09 \
    --bbox -12 31.5 36 72 --years 2015 2016 2017 2018 2019 2020 2021 2022 2023
```

## 3. Train and evaluate

The UK trains on 2015-2018 and tests on 2019, on the licensed curve library
([input root](../guides/training.md#choose-the-input-root)). The scorecard row
reruns from `configs/regions/scorecard/uk_k50.toml` (k=50, fixed slice), at the
row's commit through `scripts/dev/run_locked.py` ([Reproducing a row](../scorecard.md#reproducing-a-row)):

```bash
PYVWF_INPUT=input/combined pyvwf-validate train \
    --region configs/regions/scorecard/uk_k50.toml
PYVWF_INPUT=input/combined pyvwf-validate evaluate \
    --region configs/regions/scorecard/uk_k50.toml --train-run output/validation/UK/train-<stamp>
```

The maintained config, `configs/regions/uk.toml`, sweeps `cluster_list = [1, 50]`
over fixed and season slices; swap it into the same commands.

## Caveats

- **Pseudo-replicated rows.** Each station's generation is split equally across
  its turbine rows, so the rows are not independent. Both configs set
  `pseudo_replicated_rows = true` and `station_id_regex`; distribution comparisons
  must collapse to stations (`pyvwf.harness.skill.collapse_pseudo_replicates`).
- **One test year.** The row rests on 2019 alone; its fit carries the
  scorecard's degenerate-fit marker (†) and its gain the resampling marker (‡).
