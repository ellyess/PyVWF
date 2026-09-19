# Australia (AEMO National Electricity Market)

**Data source:** AEMO NEMWEB MMSDM archives: 5-minute unit SCADA, plus the
Generation Information workbook.
**Adapter:** `aemo-nem` · turbine-level · unit = farm · open.
**Config:** `configs/regions/au_nem.toml`; the scorecard row is
`configs/regions/scorecard/au_nem_k45.toml`.

All observation inputs are open AEMO data. ERA5 needs your own CDS account. No
raw data or processed input is committed, because `input/` is git-ignored.
This runbook builds them. The results are in
[`region-au-nem.md`](../findings/region-au-nem.md).

## 1. Observations

1. Fetch the MMSDM archives for 2020 to 2023. The script writes under
   `<input root>/raw/aemo/`, about 0.9 GB zipped, and resumes where it stopped:

   ```bash
   bash scripts/fetch/aemo_au.sh
   ```

2. Download the AEMO Generation Information workbook by hand. Its URL changes
   each quarter; the script's header gives the page.
3. Build the adapter's inputs. The script needs the `data` extra, for the
   workbook:

   ```bash
   python scripts/process/aemo_au.py \
       --gen-info "input/raw/aemo/<Generation Information workbook>.xlsx" \
       --gwpt input/reference/gwpt/Global-Wind-Power-Tracker-February-2026.xlsx
   ```

   It writes `au_nem_md.csv`, the per-month SCADA partials and a
   `join_report.md` to `<input root>/observations/turbine/AU_NEM/`.
4. Read `join_report.md` before going on. It lists the farms that did not
   match and the capacity checks.

The SCADA archives are cut on market-time month boundaries. The script reduces
each archive to energy per UTC month, and the adapter finalises each month at
load time.

## 2. Power curves

Each farm's turbine model comes from `configs/curation/au_turbine_models.csv`.
`scripts/region_tools/assign_au_curves.py` assigns a curve from each library
and writes `au_nem_md_real.csv` and `au_nem_md_open.csv` beside
`au_nem_md.csv`. Run it with `PYVWF_INPUT` pointing at the input root whose
`models.csv` you want matched; its `--help` lists the arguments.

## 3. ERA5

Fetch ERA5 for the config's box and years. The files go to
`<input root>/era5/AU/`:

```bash
python scripts/fetch/era5.py --region AU-NEM
```

## 4. Train and evaluate

The scorecard row ran on the combined library. Set the input root on both
commands; see
[Choose the input root](../guides/training.md#choose-the-input-root).

```bash
PYVWF_INPUT=<input root> python scripts/analysis/validate_region.py train \
    --region configs/regions/scorecard/au_nem_k45.toml
PYVWF_INPUT=<input root> python scripts/analysis/validate_region.py evaluate \
    --region configs/regions/scorecard/au_nem_k45.toml \
    --train-run output/validation/AU-NEM/train-<stamp>
```

The seasons in the config are the Southern-Hemisphere ones: winter is June to
August.

## Licence

The SCADA and the Generation Information workbook are open AEMO data (the
[source table](../guides/data-sources.md#source-urls)). Nothing from them is
committed. Coordinates come from the Global Wind Power Tracker (Global Energy
Monitor, CC-BY-4.0), which requires attribution. The committed curation table
is `configs/curation/au_turbine_models.csv`. It has a `source_url` column, and
16 of its 104 rows leave it blank. `configs/curation/aemo_au_aliases.csv` maps
AEMO unit codes (DUIDs) to GWPT projects, with the evidence for each. Results computed from these data may be shared.
