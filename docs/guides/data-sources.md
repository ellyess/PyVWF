# Data sources and preprocessing

The single reference for the data PyVWF uses: what each source is, its licence,
how it is fetched, and **what the processing does to it** for every region.
Full step-by-step acquisition lives in the per-region runbooks
(`docs/runbooks/<cc>.md`); the adapter contract is in
[`adding-an-adapter.md`](adding-an-adapter.md).

**Licence key**, used in the `Lic.` column below: **open** = redistributable;
**mixed** = open coordinates over a confidential generation series;
**confidential** = confidential or commercial, never redistributed.

Nothing under `input/` is committed (`/input` is git-ignored) except the open
curve library in `input/reference/`. Every fetch script is **user-executed**: it
runs under your own credentials (CDS, ENTSO-E, CEN keys), never bundled data.
Results *derived* from confidential inputs are shareable; the inputs are not.

## Input layout

Paths are defined centrally in [`src/pyvwf/config.py`](../../src/pyvwf/config.py)
(`PyVWFPaths`); `PYVWF_INPUT` redirects the root.

| Data | Format | Location |
|---|---|---|
| ERA5 winds | NetCDF | `input/era5/<TAG>/*.nc` |
| Turbine metadata + generation | CSV | `input/observations/turbine/<CC>/` |
| Country grid points + observations | CSV | `input/observations/country/{grid_points,observations}/<cc>/` |
| Power curves + models | CSV | `input/reference/{power_curves,models}.csv` |

`<CC>` is the upper-case code, `<cc>` lower-case.

---

## 1. Data sources by region

Each region declares its data source in `configs/regions/<code>.toml`; the
harness resolves it to an adapter in `src/pyvwf/sources/`. Every data source is
reduced to a **monthly capacity factor** before correction.

| Region | Code | Adapter | Source | Lic. | Fetch → Process | Runbook |
|---|---|---|:---:|:---:|---|---|
| Denmark | DK | `european-turbine` | Danish Energy Agency turbine register (monthly kWh) | open | `fetch/dk.py` → `process/dk.py` | [DK](../runbooks/dk.md) |
| United Kingdom | UK | `european-turbine` | REPD metadata + Ofgem ROC certificates | mixed | `fetch/uk.py` → `process/uk.py` | [UK](../runbooks/uk.md) |
| Germany | DE | `european-turbine` | WindStats per-turbine register (commercial) | confidential | (none) → `process/de.py` | [DE](../runbooks/de.md) |
| Spain (hist.) | ES-WS | `windstats` | WindStats generation + GWPT coordinates | mixed | (none) → `process/windstats.py` | [ES](../runbooks/es.md) |
| United States | US | `eia-us` | EIA-923 netgen + EIA-860 + USWTDB heights | open | (none) → `process/eia_us.py` | [US](../runbooks/us.md) |
| Brazil | BR | `ons-br` | ONS `FATOR_CAPACIDADE` hourly CF + constrained-off | open | (none) → `process/ons_br.py` | [BR](../runbooks/br.md) |
| Australia (NEM) | AU-NEM | `aemo-nem` | AEMO NEMWEB 5-min SCADA + Generation Information | open | `fetch/aemo_au.sh` → `process/aemo_au.py` | [AU-NEM](../runbooks/au_nem.md) |
| New Zealand | NZ | `emi-nz` | EA EMI `Generation_MD` half-hourly injection | open | `fetch/emi_nz.py` → `process/emi_nz.py` | [NZ](../runbooks/nz.md) |
| Chile | CL | `cen-cl` | Coordinador SIP `generacion-real` hourly | open | `fetch/cen_cl.py` → `process/cen_cl.py` | [CL](../runbooks/cl.md) |
| Argentina | AR | `cammesa-ar` | CAMMESA/MEM monthly generation | open | `fetch/cammesa_ar.py` → `process/cammesa_ar.py` | [AR](../runbooks/ar.md) |
| 4 ENTSO-E countries | ES IE IT NL | `entsoe-country` | ENTSO-E national generation | open | `datasets/generate_country_level_training_data.py` | [ENTSO-E](../runbooks/entsoe.md) |
| Sweden | SE | `entsoe-country` | ENTSO-E generation over Energimyndigheten's year-end wind fleet | open | generator → `region_tools/build_se_capacity_from_energimyndigheten.py` | [ENTSO-E](../runbooks/entsoe.md) |
| Norway | NO | `entsoe-country` | NVE metered farm output over NVE's turbine register (2015-21); ENTSO-E generation over the register (2023) | open, attribution | generator → `region_tools/build_no_obs_from_nve.py` | [ENTSO-E](../runbooks/entsoe.md) |
| Portugal | PT | `entsoe-country` | ENTSO-E generation over REN Data Hub's month-end wind fleet | mixed | generator → `region_tools/build_pt_capacity_from_ren.py` | [ENTSO-E](../runbooks/entsoe.md) |
| Belgium | BE | `entsoe-country` | Elia Open Data `ods031` output and monitored capacity | open | `region_tools/build_be_obs_from_elia.py` | [ENTSO-E](../runbooks/entsoe.md) |
| France | FR | `entsoe-country` | RTE eCO2mix definitive output and year-end fleet, from ODRE | open | `region_tools/build_fr_obs_from_rte.py` | [ENTSO-E](../runbooks/entsoe.md) |
| Sweden (per zone) | SE-BZ | `entsoe-zonal` | The same fetch, read per bidding zone | open | `datasets/generate_country_level_training_data.py` | [ENTSO-E](../runbooks/entsoe.md) |

## 2. What the processing does (per region)

The common shape: raw generation → align to UTC monthly bins → divide by
installed capacity → monthly CF, plus a coordinate/curve/height join where the
source lacks them. The region-specific work:

- **DK**: `european-turbine`. Convert ETRS89/UTM 32N to lon/lat, map
  `Land`/`Hav` to onshore/offshore, reshape per-year sheets to long
  `ID,year,month,generation_kwh`. Coordinates are exact and per-turbine, so no
  GWPT join. kWh → CF at load.
- **UK**: REPD gives open metadata; Ofgem ROC certificates give energy, banded
  to generation. Open RER export or the confidential certificate warehouse (the
  mixed-licence split is flagged in the runbook and the processor).
- **DE**: WindStats per-turbine register; postcode → coordinate via
  `geolocate.germany.csv`. Confidential: writes only to git-ignored `input/`.
- **ES-WS**: WindStats generation joined to GWPT coordinates by a
  thewindpower name map. Historical 1998-2000 fleet (old ~600 kW machines).
- **US**: EIA-923 net generation, EIA-860 nameplate + coordinates, USWTDB hub
  heights. Drops annual (`A`) respondents (imputed months); `--bbox` must match
  the ERA5 box or out-of-domain plants snap to the wrong cell. Curves matched to
  the curve library by rating band and specific power.
- **BR**: ONS publishes CF per *complex* (collector-substation coordinates and
  time-varying capacity in the same file). A constrained-off (`RESTRICAO_COFF`)
  series builds a curtailment mask; the Nordeste curtails heavily. One default
  curve for every unit.
- **AU-NEM**: 5-min SCADA summed to monthly energy, AEST → UTC, over
  Generation Information nameplate. Real turbine specs
  (`configs/curation/au_turbine_models.csv`) drive matched curves.
- **NZ**: half-hourly metered injection, trading-period → UTC through
  `Pacific/Auckland` (DST 46/48/50-period days pinned by tests), summed over
  multi-POC farms, against a staged-capacity history. Curated farm table gives
  coordinates, hub heights, and turbine models; commissioning-ramp months masked.
- **CL**: hourly `generacion-real` (delivered energy) → monthly CF against
  registered capacity, fixed UTC-4 (no DST). Coordinates joined from GWPT;
  turbine specs (`configs/curation/cl_turbine_specs.csv`) drive matched curves
  and hub heights; leading pre-operational months stripped.
- **AR**: monthly GWh (native resolution, no timezone logic). CAMMESA carries
  no capacity, so the CF denominator is joined from GWPT, corrected to the real
  per-phase nameplate (turbine count × unit MW) in
  `configs/curation/ar_coord_overrides.csv` where GWPT gave a phase-split
  full-farm value. Zero-generation and steady-underperforming codes are dropped
  (`configs/curation/ar_fleet_exclusions.csv`). Turbine specs drive matched
  curves.
- **Country-level (ENTSO-E)**: see §4.

Turbine-level sources report at different native units (turbine / farm / plant /
complex); the per-adapter unit, time convention, and curtailment handling are in
the [built-in adapters](adding-an-adapter.md#built-in-adapters) table.

### Source URLs

| Source | Endpoint | Access | Licence |
|---|---|---|---|
| Danish Energy Agency | `ens.dk/media/4945/download`, `.../4948/download` | download | open, attribution |
| UK REPD | `gov.uk/.../renewable-energy-planning-database-quarterly-extract` | download | OGL |
| Ofgem ROC | `rer.ofgem.gov.uk` (open) / certificate warehouse (confidential) | export / licensed | mixed |
| EIA-923 / EIA-860 | `eia.gov/electricity/data/` | download | US public domain |
| USWTDB | `energy.usgs.gov/uswtdb/data/` | download | US public domain |
| ONS Brazil | `dados.ons.org.br` (`fator-capacidade-2`) | open portal | ONS Open Data |
| AEMO NEM | `nemweb.com.au/Data_Archive/.../MMSDM` | download | open |
| EA EMI (NZ) | `emi.ea.govt.nz/Wholesale/Datasets/Generation/Generation_MD/` | download | open |
| CEN Chile | `sipub.api.coordinador.cl` (SIP) | API (free key) | public transparency |
| CAMMESA (AR) | `cammesaweb.cammesa.com/erenovables/` | download | public |
| ENTSO-E | `transparency.entsoe.eu` | API (free key) | public-by-regulation |
| Elia (BE) | `opendata.elia.be`, dataset `ods031` | open portal | Elia Open Data Licence |
| ODRE (FR) | `odre.opendatasoft.com`, datasets `eco2mix-national-cons-def` and `parc-prod-par-filiere` | open portal | Licence Ouverte v2.0 |
| Energimyndigheten (SE) | `pxexternal.energimyndigheten.se`, table `Vindkraftsstatistik/EN0105_1` | API | Swedish official statistics; no licence stated |
| NVE (NO) | `nve.no/media/13399/vindprod2002-2021_kraftverk_utcplus1.xlsx`; `api.nve.no/web/WindPowerplant/GetWindPowerPlants` | download / API | NVE credited, content unchanged; register NLOD |
| REN Data Hub (PT) | `servicebus.ren.pt/datahubapi/electricity/ElectricityInstalledPowerMonthly` | API | no reuse licence published ("All rights reserved"); kept local |
| Eurostat (audit) | `nrg_bal_peh`, gross wind production | download | reuse with the source acknowledged |
| WindStats (DE, ES) | commercial subscription | (none) | confidential, commercial |

---

## 3. Shared inputs (all regions)

**ERA5 reanalysis (the wind field).** Copernicus CDS
`reanalysis-era5-single-levels`, hourly 100 m + 10 m u/v winds at 0.25°. Free,
needs a CDS account (`~/.cdsapirc`). `scripts/fetch/era5.py --region <cc>` takes
the box and years from the region TOML, batches 3 months per CDS request (the
cost-limit ceiling), and writes per-month `era5_<tag>_<YYYY>_<MM>.nc`. Big boxes
(US, BR) are reduced to yearly daily means by `scripts/era5/combine.py`.

**The `era5/EU` archive (annual-mean roughness).** The European rows published
before 2026-09-13 read `era5/EU`. It holds one hourly file per year, 2015 to
2023, named `era5_combined_<YYYY>_EU.nc`. Each file merges the 10 m and 100 m
winds and carries `z0`, one static field per year. That field is the annual
mean of the roughness derived from the 10 m to 100 m shear, so a run on it
applies the annual-mean treatment. `src/pyvwf/datasets/combine_era5_files.py`
built the archive from the per-year `era5_u10_v10_<YYYY>_months01-12_EU.nc` and
`era5_u100_v100_<YYYY>_months01-12_EU.nc` downloads:

```bash
python src/pyvwf/datasets/combine_era5_files.py --all-years --add-roughness \
    --roughness-source pyvwf
```

The archive is kept unchanged, so the rows published on it stay reproducible.
No shipped configuration reads it any more. Every European config, maintained
and scorecard alike, reads `era5/EU_2026-09`, which carries no roughness field
and a wider box, so `prep_era5` derives the roughness per timestep. That is
also the default now, so a region asks for nothing. The reasons, and the three
routes, are in
[`roughness-temporal-treatment.md`](../design/roughness-temporal-treatment.md).

**Coordinates and capacity: Global Wind Power Tracker (GWPT).** Global Energy
Monitor, CC-BY-4.0, at `input/reference/gwpt/`. Supplies coordinates (CL, AR,
ES-WS) and per-phase capacity (AR). Also supplies capacity weights for every
country-level grid via `scripts/region_tools/weight_country_grid_points.py`
(§4 and the region guide); start/retirement years let the fleet be reconstructed by year. One known
spurious record is excluded in `configs/curation/gwpt_exclusions.csv`.
Records the tracker leaves undated take a start year from
`configs/curation/gwpt_start_years.csv` where a national register dates them;
the rest are counted in every year.

**Registers that date the tracker's records.** Written into
`configs/curation/gwpt_start_years.csv` by
`scripts/region_tools/backfill_gwpt_start_years.py`, which reads them from
`input/reference/registers/` (local, git-ignored):

- `nve_wind_in_operation_<date>.json`: NVE's wind plants in operation, with
  each plant's turbines and their dates, from
  `https://api.nve.no/web/WindPowerplant/GetWindPowerPlantsInOperation`
  (NLOD).
- `odre_registre_eolien_<date>.csv`: the ODRE national register of production
  installations, filtered to wind (`codefiliere = "EOLIE"`), from
  `odre.opendatasoft.com`, dataset
  `registre-national-installation-production-stockage-electricite-agrege`
  (Licence Ouverte). Most installation names are confidential, so matching is
  by commune and capacity.
- `fr_communes_centre_<date>.json`: French commune centres, from
  `https://geo.api.gouv.fr/communes?fields=code,nom,centre` (Licence Ouverte).

**Bidding-zone geometry.** Real polygons vendored under
`configs/curation/zones/` (SE/DK/IT from entsoe-py, MIT; NO from NVE, NLOD).
Used by `scripts/region_tools/assign_country_zones.py` and the zone-aware
weighting so a grid-point cluster is the bidding zone its observations describe.
That directory's README carries provenance and caveats. Not used:
electricitymaps-contrib (AGPL) or PyPSA-Eur's derived zones.

**Hub heights: GOWIRES.** Global hub-height dataset (Zenodo, CC-BY-4.0), ~32%
coverage, a fallback where the source omits height. NZ, CL and AR use curated
per-farm heights instead.

**Power curves and turbine models.**
- *Open (shipped, committed):* `input/reference/{power_curves,models}.csv`, the
  NREL/turbine-models library (BSD-3-Clause, DOI 10.11578/dc.20210112.1,
  VWF-smoothed). Bundled in the package too, so an installed PyVWF runs outside
  a checkout. Provenance in `power_curves_provenance.csv`.
- *Licensed (local only, confidential):* `input/reference/{power_curves,models}.real.csv`,
  the renewables.ninja/VWF library from thewindpower.net. Swap in for production
  runs; the merged set is under `input/combined/reference/` (`PYVWF_INPUT=input/combined`).

**Region shapes.** `input/reference/shapes/*.geojson`, onshore/offshore outlines
for clustering (committed).

---

## 4. Country-level (ENTSO-E) layout

The nine country-level regions use grid-sampled ERA5 against a national
observed series. `python -m pyvwf.datasets.generate_country_level_training_data`
builds the grid points of all nine and fetches the ENTSO-E observations of
seven; the harness then trains and evaluates like any region. The steps,
including the capacity weighting and the observation audit, are in
[A country-level region](adding-a-region.md#a-country-level-region), and the
order for the shipped regions is in the [ENTSO-E runbook](../runbooks/entsoe.md).

- **ENTSO-E (ES, IE, IT, NL, and the files the SE, NO and PT builders start
  from).** The fetch takes every wind type, onshore and offshore, unless
  `--psr-type` names one; until 2026-10-25 it took onshore alone. Generation is
  the wind types' output, with any consumption column left out. An hour in
  which a wind type that reports is missing is dropped, so the remaining types
  are never taken for the whole fleet; outside a type's first and last reported
  hour it is taken not to exist yet. The capacity-factor denominator is
  ENTSO-E's installed capacity.
- **SE: Energimyndigheten.**
  `scripts/region_tools/build_se_capacity_from_energimyndigheten.py` keeps
  ENTSO-E's generation and divides it by Energimyndigheten's 31 December wind
  fleet, onshore and offshore, interpolated linearly through the year. The
  per-zone files are not changed.
- **NO: NVE.** `scripts/region_tools/build_no_obs_from_nve.py` builds 2015-21
  from NVE's hourly output per farm, metered at the grid connection, over
  NVE's turbine register. A farm counts in a month only when it is metered
  every hour, its registered capacity is above zero and fixed, and its output
  stays within 1.05 times that capacity. NVE's metered file ends in 2021, so
  2023 keeps ENTSO-E's generation over the register. The per-zone files are
  not changed.
- **PT: REN.** `scripts/region_tools/build_pt_capacity_from_ren.py` keeps
  ENTSO-E's generation and divides it by REN Data Hub's month-end wind fleet,
  onshore and offshore, interpolated linearly through each month.
- **BE: Elia.** `scripts/region_tools/build_be_obs_from_elia.py` sums Elia's
  onshore and offshore output and monitored capacity every 15 minutes,
  averages each hour, and divides output by monitored capacity, so both cover
  the same fleet in every hour. A step with no monitored capacity is a gap.
- **FR: RTE.** `scripts/region_tools/build_fr_obs_from_rte.py` averages RTE's
  half-hourly national wind output (onshore and offshore) to hours and divides
  it by RTE's 31 December wind fleet, interpolated linearly through the year.
  It reads both from ODRE. Three hours of each year's two clock-change days are
  left empty, because RTE's series cannot place them.

The generator run for BE or FR without `--skip-observations` overwrites the
builders' files with ENTSO-E's ([runbook](../runbooks/entsoe.md)); for SE,
NO and PT, a new fetch is followed by their builders again. The generator and
the builders write this layout:

```
input/observations/country/
├── grid_points/<cc>/<cc>_grid_points[_<year>].csv   # ID,lat,lon,height,model,capacity,cluster
└── observations/<cc>/<cc>_{train_<a>_<b>,test_<y>}.csv   # capacity_factor,generation_mw,capacity_mw
```

Zonal regions (NO, SE) also carry per-zone files and an `_aggregated` national
series.

`scripts/analysis/audit_country_observations.py` divides each year's observed
energy by Eurostat's gross wind production, from
`configs/curation/eurostat_wind_gwh.csv`. The physical gates cannot see a
series that measures only part of the fleet; this comparison can.

---

## 5. Evaluated but not shipped

- **Turkey (TR):** EPİAŞ Şeffaflık. The per-plant licensed-realtime endpoint
  returns 403 under the public subscription and aggregates ignore
  `powerPlantId`, so per-plant wind is not obtainable openly.
  `fetch/epias_tr.py` + [TR runbook](../runbooks/tr.md) document the gap.
- **Sweden & Finland (WindStats):** the format supports them, but GWPT matches
  only 7-9% of their fragmented fleets. Parked pending a thewindpower.net
  coordinate table ([ES runbook](../runbooks/es.md)).

## 6. Redistribution summary

| May be committed / shared | Must stay local (confidential or mixed) |
|---|---|
| Open curve library, region shapes, zone polygons | WindStats DE/ES generation and per-turbine derivatives |
| All fetch/process scripts, configs, curation tables | Ofgem confidential certificate-warehouse exports |
| Model results / metrics from any source | The licensed `*.real.csv` curve library |
| Provenance and audit tables | Raw `input/` of every region (git-ignored) |

Scripts that touch confidential data print a banner and write only to
git-ignored `input/`. Mixed-licence regions (UK confidential path, ES-WS) flag
the split in their runbook and processor.
