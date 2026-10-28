# Scripts

Entry points over the `pyvwf` package: data acquisition and processing, and
tools that run across regions. Logic that more than one script needs belongs in
`src/pyvwf/`. Every Python script parses its arguments with `argparse`, so
`--help` lists its options. The drivers of the finished studies are at tag
`archive/research-2026-10-27`; a new study's drivers go in
`scripts/studies/<stem>/` ([`adding-a-study.md`](../docs/guides/adding-a-study.md)).

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
    backfill_gwpt_start_years.py   start years for undated GWPT records, from national registers
    repair_country_capacity.py     a country CF series rebuilt on a capacity register
    weight_country_grid_points.py  GWPT capacity weights for country-level grid points
  dev/              development guards (docs/design/ai-native-development.md)
    run_locked.py                  run a command holding the run lock: -- <command>
    stamp.py                       run a local-only suite (realdata, pinn) and stamp the code it tested
    lock.py                        write the CI constraints files (constraints/), or --check them
    agent_evals/evals.py           score the review agents on pre-registered cases: prepare, judge, score
  analysis/         tools that run across regions
    run_hindcast.py                national monthly CF hindcast against the record
    export_correction_field.py     a region's gridded factor field, as NetCDF
    audit_country_observations.py  physical-bound check of every country-level series
    curve_match_audit.py           the curve-match audit, per fitted fleet
    extent_audit.py                each scorecard row's fleet against its loaded extent
    baseline_bootstrap.py          paired bootstrap intervals for a scorecard row
    common_row_rescore.py          a scorecard row rescored on common rows
    unit_concentration.py          how concentrated a turbine-level row's error is across units
    missing_value_audit.py         where a row's corrected capacity factors are missing, and why
    off_curve_sensitivity.py       how much a row's result depends on dropping off-curve days
    training_objective_check.py    whether a country-level fit's objective drops off-curve days
    export_voronoi_frames.py       cluster maps for TouchDesigner (the touchdesigner extra)
```

## Importing another script

Scripts import `pyvwf` as an installed package (`pip install -e .` in a
checkout); none puts `src/` on `sys.path`. Logic that more than one script
needs belongs in `src/pyvwf/`. A script may
still import a sibling in its own directory, or a tool in `analysis/`, by
putting that directory on `sys.path`. A new script does not reach further:
move the shared code into the package instead.

## Retiring a script

A script is retired when nothing runs it any more: no test, no guide or
runbook, no other script, and no command a findings document or pre-registration
records as current. Delete it in its own commit, and say in that commit why,
and which commit last had it. A script whose output a findings document reports
stays as long as that document stands, so its numbers can be regenerated, or
until an archive tag preserves both. Pre-registrations keep the command lines they
registered, whatever happens to the script.

Region curation data (coordinate and capacity overrides, curated farm tables,
turbine-model maps, fleet exclusions) lives in
[`configs/curation/`](../configs/curation). Region configs are in
[`configs/regions/`](../configs/regions).
