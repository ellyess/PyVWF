# Scripts

Entry points over the `pyvwf` package: data acquisition and processing, tools
that run across regions, and the drivers of each study. Logic that more than
one script needs belongs in `src/pyvwf/`. Every Python script parses its
arguments with `argparse`, so `--help` lists its options. The exceptions are
`pinn/` and the two ML transfer scripts, which move together (issue #12).

The download scripts are user-executed: they use your own credentials, passed
in the environment for one command. No raw or derived data is committed
(`input/` is git-ignored).

How to use them lives elsewhere, once each:

- adding a region, in order: [`adding-a-region.md`](../docs/guides/adding-a-region.md);
- running a region, and choosing the input root: [`training.md`](../docs/guides/training.md);
- each data source and ERA5: [`data-sources.md`](../docs/guides/data-sources.md);
- one region's data: `docs/runbooks/<code>.md`;
- adding a study: [`adding-a-study.md`](../docs/guides/adding-a-study.md).

## Layout

```
scripts/
  fetch/            download raw inputs (user-executed)
    era5.py           ERA5 for any region: --region <code> (box and years from its config)
    aemo_au.sh        AU  observations (AEMO SCADA)
    cammesa_ar.py     AR  observations (CAMMESA renewables ZIP)
    cen_cl.py         CL  observations (Coordinador SIP API; CEN_API_KEY in the environment)
    dk.py             DK  observations (Danish Energy Agency register .xlsx)
    emi_nz.py         NZ  observations (EA EMI Generation_MD)
    epias_tr.py       TR  observations (EPIAS; credentials in the environment); not shipped, see the runbook
    uk.py             UK  metadata (REPD) and the Ofgem ROC export steps; see the runbook
  era5/
    combine.py        monthly ERA5 to yearly daily files, with per-timestep roughness: --region <code>
  process/          raw inputs to the adapter's input CSVs (input/observations/turbine/<CODE>/)
    aemo_au.py  cammesa_ar.py  cen_cl.py  de.py  dk.py  eia_us.py  emi_nz.py  ons_br.py  uk.py
    windstats.py      WindStats generation with GWPT coordinates (mixed licence; runbooks/es.md)
                      de.py and windstats.py read CONFIDENTIAL WindStats data (runbooks/de.md, es.md)
  region_tools/     one-region helpers
    apply_turbine_specs.py         power curves and hub heights from a spec table (CL, AR)
    assign_au_curves.py            curve assignment for the AU fleet, per curve library
    assign_country_zones.py        grid points to the bidding zone that contains them
    repair_country_capacity.py     a country CF series rebuilt on a capacity register
    weight_country_grid_points.py  GWPT capacity weights for country-level grid points
  dev/              development guards (docs/design/agent-guards.md)
    run_locked.py                  run a command holding the run lock: -- <command>
    stamp.py                       run a local-only suite (realdata, pinn) and stamp the code it tested
  analysis/         tools that run across regions
    validate_region.py             train, evaluate or transfer one region (also pyvwf-validate)
    run_hindcast.py                national monthly CF hindcast against the record
    export_correction_field.py     a region's gridded factor field, as NetCDF
    audit_country_observations.py  physical-bound check of every country-level series
    curve_match_audit.py           the curve-match audit, per fitted fleet
    extent_audit.py                each scorecard row's fleet against its loaded extent
    baseline_bootstrap.py          paired bootstrap intervals for a scorecard row
    common_row_rescore.py          a scorecard row rescored on common rows
    eu_rerun_compare.py            paired comparison of evaluate runs of one row
    regression_compare.py          frame-level diff of two runs' outputs
    export_voronoi_frames.py       cluster maps for TouchDesigner (the touchdesigner extra)
    ml_transfer_retest.py          ML transfer, round one (moves with pinn/, issue #12)
    ml_transfer_expanded.py        ML transfer, round two (moves with pinn/, issue #12)
  studies/          one directory per study, named by its findings document (studies/README.md)
  pinn/             the physics-informed study; moves to studies/ after its turbine-only run (issue #12)
```

Region curation data (coordinate and capacity overrides, curated farm tables,
turbine-model maps, fleet exclusions) lives in
[`configs/curation/`](../configs/curation). Region configs are in
[`configs/regions/`](../configs/regions).
