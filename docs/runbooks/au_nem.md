# Australia (AEMO NEM)

**Data source:** AEMO NEMWEB MMSDM archives (5-minute unit SCADA) and the AEMO
Generation Information workbook. **Adapter:** `aemo-nem` · turbine-level ·
unit = farm · open.

The wind fleet of Australia's National Electricity Market: training years
2020-22, test year 2023. Its result is the AU-NEM row of the
[scorecard](../scorecard.md) and the
[results entry](../results.md#australia-nem).

## Source and licence

The SCADA and the Generation Information workbook are open AEMO data (the
[source table](../guides/data-sources.md#source-urls)). Nothing from them is
committed. Coordinates come from the Global Wind Power Tracker (Global Energy
Monitor, CC-BY-4.0), which requires attribution. The committed curation is
`configs/curation/au_turbine_models.csv` (each farm's turbine model;
`source_url` is blank in 16 of 104 rows) and
`configs/curation/aemo_au_aliases.csv` (AEMO unit codes, DUIDs, to GWPT
projects with evidence; aliases override the automatic name match). Results
computed from these data may be shared. ERA5 needs your own CDS account.

## 1. Observations

1. Fetch the 2020-2023 MMSDM archives (unit SCADA, `DUDETAILSUMMARY`, and
   `DUDETAIL` for the capacity history) into `<input root>/raw/aemo/`, about
   0.9 GB zipped; the script resumes: `bash scripts/fetch/aemo_au.sh`.
2. Download the Generation Information workbook by hand. Its URL changes each
   quarter; the script's header gives the page.
3. Build the adapter's inputs (needs the `data` extra, for the workbook):

   ```bash
   python scripts/process/aemo_au.py \
       --gen-info "input/raw/aemo/<Generation Information workbook>.xlsx" \
       --gwpt input/reference/gwpt/Global-Wind-Power-Tracker-February-2026.xlsx
   ```

   It writes `au_nem_md.csv`, `au_nem_scada_monthly_partials.csv`,
   `au_nem_capacity_mask.csv` (only when the `DUDETAIL` archives are present)
   and `join_report.md` to `<input root>/observations/turbine/AU_NEM/`. Read
   the report before going on: it lists the unmatched DUIDs (excluded for
   lack of coordinates) and the capacity checks.

## 2. ERA5

Box `[129.0, 154.0, -44.0, -10.0]`, into `<input root>/era5/AU/` (the config's
ERA5 path, so no combine step): `python scripts/fetch/era5.py --region AU-NEM`.
The rest is in [Shared inputs](../guides/data-sources.md#3-shared-inputs-all-regions).

## 3. Power curves

`scripts/region_tools/assign_au_curves.py` replaces the uniform curve that
`scripts/process/aemo_au.py` gives every farm, from the curated models. It
writes `au_nem_md_real.csv` (matched against the `models.csv` of the root in
`PYVWF_INPUT`) and `au_nem_md_open.csv` (the open library by specific-power
class, with an IEC Class 2 composite where specific power is missing). The
adapter reads only `au_nem_md.csv`; the scorecard row copied the open one over
it ([curve matching](../scorecard.md#curve-matching-the-last-three-turbine-level-columns)):

```bash
PYVWF_INPUT=input/combined python scripts/region_tools/assign_au_curves.py \
    --open-models src/pyvwf/resources/models.csv \
    --open-curves src/pyvwf/resources/power_curves.csv
cp input/combined/observations/turbine/AU_NEM/au_nem_md_open.csv \
   input/combined/observations/turbine/AU_NEM/au_nem_md.csv
```

## 4. Train and evaluate

The scorecard row reruns from `configs/regions/scorecard/au_nem_k45.toml`
(k45, season slices) on the combined library
([Reproducing a row](../scorecard.md#reproducing-a-row),
[Choose the input root](../guides/training.md#choose-the-input-root)). The
maintained `configs/regions/au_nem.toml` sweeps `cluster_list = [1, 15]` over
fixed and season slices. Seasons are Southern-Hemisphere (winter June-August).

```bash
PYVWF_INPUT=input/combined pyvwf-validate train \
    --region configs/regions/scorecard/au_nem_k45.toml
PYVWF_INPUT=input/combined pyvwf-validate evaluate \
    --region configs/regions/scorecard/au_nem_k45.toml \
    --train-run output/validation/AU-NEM/train-<stamp>
```

## Caveats

- **SCADA records curtailed output**, which the correction absorbs; in South
  Australia it cannot be separated from resource bias, so farm-level behaviour
  there is not established.
- **The scorecard row's fit is degenerate.** Check `fit_quality` on the
  factors file before using any per-cluster factor.
- **Hub height is a uniform 100 m** (`--height` in `scripts/process/aemo_au.py`)
  until there is per-farm height data.
