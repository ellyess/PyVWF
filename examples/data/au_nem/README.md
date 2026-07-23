# AU-NEM derived data (bundled for the validation notebook)

Small DERIVED datasets so `examples/notebooks/au_nem_validation.ipynb` runs
without re-downloading the raw AEMO archives. Raw 5-minute SCADA is not
redistributed — these are monthly aggregates and metadata.

| file | content | built by |
|---|---|---|
| `au_nem_scada_monthly_partials.csv` | per-(DUID, UTC month) energy sums + interval counts, 2020–2023 | `scripts/process/aemo_au.py` |
| `au_nem_capacity_mask.csv` | farm-months with unreliable registered capacity | same |
| `au_nem_md_open.csv` | farm metadata with OPEN-library curve assignment | `scripts/region_tools/assign_au_curves.py` |

## Attribution
- Market data: **Source: AEMO** (aggregated derivatives of NEMWeb
  DISPATCH_UNIT_SCADA / DUDETAIL / Generation Information), used and
  redistributed with attribution per AEMO's Copyright Permissions.
  AEMO makes no representation as to the accuracy or completeness of the data.
- Coordinates: derived from the **Global Wind Power Tracker, Global Energy
  Monitor** (CC BY 4.0), February 2026 release.
- Turbine identities: `configs/curation/au_turbine_models.csv` (public sources,
  per-row provenance).

NOT bundled (size): the ERA5-AU subset — fetch with
`scripts/fetch/era5.py --region au_nem` (~6–12 GB hourly download via your CDS account;
queue time is the bottleneck), then reduce to ~29 MB/year daily files with
`scripts/era5/combine.py --region au_nem`. The open curve library is the separately
distributed `power_curves_open` set; point the notebook at your unzipped copy.
