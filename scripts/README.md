# Scripts

Data acquisition, processing, and analysis for the bias-correction workflow.
See [PIPELINE.md](../PIPELINE.md) for execution order. All data-download
scripts are **user-executed** (they use your credentials / CDS key); none of
the raw or derived data is committed (`input/` is git-ignored).

The per-region flow is always the same three steps (**fetch observations →
fetch ERA5 → process**), then train/evaluate with the harness. Each region has
a runbook in `docs/runbooks/<CC>.md`.

## Layout

```
scripts/
  fetch/          download raw inputs (user-executed)
    era5.py         ERA5 for ANY region: --region <cc> (bbox + years from the TOML)
    aemo_au.sh      AU  observations (AEMO SCADA)
    cammesa_ar.py   AR  observations (CAMMESA renewables ZIP)
    cen_cl.py       CL  observations (Coordinador SIP API; CEN_API_KEY env)
    dk.py           DK  observations (Danish Energy Agency register .xlsx)
    emi_nz.py       NZ  observations (EA EMI Generation_MD)
    epias_tr.py     TR  observations (EPİAŞ; input/.epias_credentials); demoted, see runbook
    uk.py           UK  metadata (REPD, auto) + Ofgem ROC export steps (manual); see runbook
  era5/
    combine.py      reduce monthly ERA5 to yearly DAILY files: --region <cc>
                    (only the big boxes, US, BR, AU, need this)
  process/          raw inputs -> the adapter's input CSVs (input/observations/turbine/<CC>/)
    aemo_au.py  cammesa_ar.py  cen_cl.py  de.py  dk.py  eia_us.py  emi_nz.py  ons_br.py  uk.py
    windstats.py --country ES   Spain (WindStats gen + GWPT coords, mixed licence)
    (de.py + windstats.py handle CONFIDENTIAL WindStats data; docs/runbooks/de.md and ES.md)
  analysis/         train, evaluate, and one-off studies
    validate_region.py             train / evaluate / transfer a region (the main driver)
    train_all_bias_corrections.py  batch trainer across configurations
    evaluate_all_pyvwf_runs.py     MAE/RMSE/MBE across runs
    ml_transfer_retest.py          leave-one-region-out ML transfer study
    audit_country_observations.py  physical-bound check on country-level CF series
    regression_compare.py  regression_run_harness.py  regression_run_legacy.py   refactor regression check
  region_tools/     region-specific helpers
    assign_au_curves.py  export_au_grid_netcdf.py
    weight_country_grid_points.py  real GWPT capacity weights for country grids
    repair_country_capacity.py     rebuild a country CF series on a GWPT capacity register
    assign_country_zones.py        reassign grid points to the bidding zone that contains them
                                   (real polygons in configs/curation/zones/)
```

Region curation data (coordinate/capacity overrides, curated farm tables,
turbine-model maps) lives in [`configs/curation/`](../configs/curation);
region configs are in [`configs/regions/`](../configs/regions).

## Typical run (any region)

```bash
# 1. observations (region-specific fetch script)
python scripts/fetch/cen_cl.py --years 2021 2024      # example: Chile
# 2. ERA5 (one script, region from the config)
python scripts/fetch/era5.py --region cl
#    big boxes only: python scripts/era5/combine.py --region us
# 3. process into the adapter's inputs
python scripts/process/cen_cl.py
# 4. train + evaluate
PYVWF_INPUT=input/combined python scripts/analysis/validate_region.py train    --region configs/regions/cl.toml
PYVWF_INPUT=input/combined python scripts/analysis/validate_region.py evaluate --region configs/regions/cl.toml --train-run output/validation/CL/train-<stamp>
```
