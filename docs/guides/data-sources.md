# Data sources and preprocessing

The single reference for the data PyVWF uses: what each source is, its licence,
how it is fetched, and **what the processing does to it** for every region.
Full step-by-step acquisition lives in the per-region runbooks
(`docs/runbooks/<CC>.md`); the adapter contract is in
[`adding-an-observation-source.md`](adding-an-observation-source.md).

**Licence key:** 🟢 open (redistributable) · 🟡 mixed (open coordinates over a
confidential generation series) · 🔴 confidential / commercial.

Nothing under `input/` is committed (`/input` is git-ignored) except the open
curve library in `input/reference/`. Every fetch script is **user-executed**: it
runs under your own credentials (CDS, ENTSO-E, CEN keys), never bundled data.
Results *derived* from confidential inputs are shareable; the inputs are not.

## Input layout

Paths are defined centrally in [`src/vwf/config.py`](../../src/vwf/config.py)
(`PyVWFPaths`); `PYVWF_INPUT` redirects the root.

| Data | Format | Location |
|---|---|---|
| ERA5 winds | NetCDF | `input/era5/<TAG>/*.nc` |
| Turbine metadata + generation | CSV | `input/observations/turbine/<CC>/` |
| Country grid points + observations | CSV | `input/observations/country/{grid_points,observations}/<cc>/` |
| Power curves + models | CSV | `input/reference/{power_curves,models}.csv` |

`<CC>` is the upper-case code, `<cc>` lower-case.

---

## 1. Observation sources by region

Each region declares its source in `configs/regions/<code>.toml`; the harness
resolves it to an adapter in `src/vwf/sources/`. Every source is reduced to a
**monthly capacity factor** before correction.

| Region | Code | Adapter | Source | Lic. | Fetch → Process | Runbook |
|---|---|---|:---:|:---:|---|---|
| Denmark | DK | `european-turbine` | Danish Energy Agency turbine register (monthly kWh) | 🟢 | `fetch/dk.py` → `process/dk.py` | [DK](../runbooks/dk.md) |
| United Kingdom | UK | `european-turbine` | REPD metadata + Ofgem ROC certificates | 🟡 | `fetch/uk.py` → `process/uk.py` | [UK](../runbooks/uk.md) |
| Germany | DE | `european-turbine` | WindStats per-turbine register (commercial) | 🔴 | (none) → `process/de.py` | [DE](../runbooks/de.md) |
| Spain (hist.) | ES-WS | `windstats` | WindStats generation + GWPT coordinates | 🟡 | (none) → `process/windstats.py` | [ES](../runbooks/es.md) |
| United States | US | `eia-us` | EIA-923 netgen + EIA-860 + USWTDB heights | 🟢 | (none) → `process/eia_us.py` | [US](../runbooks/us.md) |
| Brazil | BR | `ons-br` | ONS `FATOR_CAPACIDADE` hourly CF + constrained-off | 🟢 | (none) → `process/ons_br.py` | [BR](../runbooks/br.md) |
| Australia (NEM) | AU-NEM | `aemo-nem` | AEMO NEMWEB 5-min SCADA + Generation Information | 🟢 | `fetch/aemo_au.sh` → `process/aemo_au.py` | (none) |
| New Zealand | NZ | `emi-nz` | EA EMI `Generation_MD` half-hourly injection | 🟢 | `fetch/emi_nz.py` → `process/emi_nz.py` | [NZ](../runbooks/nz.md) |
| Chile | CL | `cen-cl` | Coordinador SIP `generacion-real` hourly | 🟢 | `fetch/cen_cl.py` → `process/cen_cl.py` | [CL](../runbooks/cl.md) |
| Argentina | AR | `cammesa-ar` | CAMMESA/MEM monthly generation | 🟢 | `fetch/cammesa_ar.py` → `process/cammesa_ar.py` | [AR](../runbooks/ar.md) |
| 9 ENTSO-E countries | BE ES FR IE IT NL NO PT SE | `entsoe-country` | ENTSO-E national generation | 🟢 | `datasets/generate_country_level_training_data.py` | (none) |
| Sweden (per zone) | SE-BZ | `entsoe-zonal` | The same fetch, read per bidding zone | 🟢 | `datasets/generate_country_level_training_data.py` | (none) |

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
  the library by scale + specific power.
- **BR**: ONS publishes CF per *complex* (collector-substation coordinates and
  time-varying capacity in the same file). A constrained-off (`RESTRICAO_COFF`)
  series builds a curtailment mask; the Nordeste curtails heavily. Uniform
  representative curve.
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
  (`EXCLUDE`). Turbine specs drive matched curves.
- **Country-level (ENTSO-E)**: see §4.

Turbine-level sources report at different native units (turbine / farm / plant /
complex); the per-adapter unit, time convention, and curtailment handling are in
[`adding-an-observation-source.md`](adding-an-observation-source.md).

### Source URLs

| Source | Endpoint | Access | Licence |
|---|---|---|---|
| Danish Energy Agency | `ens.dk/media/4945/download`, `.../4948/download` | download | open, attribution |
| UK REPD | `gov.uk/.../renewable-energy-planning-database-quarterly-extract` | download | OGL |
| Ofgem ROC | `rer.ofgem.gov.uk` (open) / certificate warehouse (🔴) | export / licensed | mixed |
| EIA-923 / EIA-860 | `eia.gov/electricity/data/` | download | US public domain |
| USWTDB | `energy.usgs.gov/uswtdb/data/` | download | US public domain |
| ONS Brazil | `dados.ons.org.br` (`fator-capacidade-2`) | open portal | ONS Open Data |
| AEMO NEM | `nemweb.com.au/Data_Archive/.../MMSDM` | download | open |
| EA EMI (NZ) | `emi.ea.govt.nz/Wholesale/Datasets/Generation/Generation_MD/` | download | open |
| CEN Chile | `sipub.api.coordinador.cl` (SIP) | API (free key) | public transparency |
| CAMMESA (AR) | `cammesaweb.cammesa.com/erenovables/` | download | public |
| ENTSO-E | `transparency.entsoe.eu` | API (free key) | public-by-regulation |
| WindStats (DE, ES) | commercial subscription | (none) | 🔴 commercial |

---

## 3. Shared inputs (all regions)

**ERA5 reanalysis (the wind field).** Copernicus CDS
`reanalysis-era5-single-levels`, hourly 100 m + 10 m u/v winds at 0.25°. Free,
needs a CDS account (`~/.cdsapirc`). `scripts/fetch/era5.py --region <cc>` takes
the box and years from the region TOML, batches 3 months per CDS request (the
cost-limit ceiling), and writes per-month `era5_<tag>_<YYYY>_<MM>.nc`. Big boxes
(US, BR) are reduced to yearly daily means by `scripts/era5/combine.py`.

**Coordinates and capacity: Global Wind Power Tracker (GWPT).** Global Energy
Monitor, CC-BY-4.0, at `input/reference/gwpt/`. Supplies coordinates (CL, AR,
ES-WS) and per-phase capacity (AR). Also supplies capacity weights for every
country-level grid via `scripts/region_tools/weight_country_grid_points.py`
(§4); start/retirement years let the fleet be reconstructed by year. One known
spurious record is excluded in `configs/curation/gwpt_exclusions.csv`.

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
- *Licensed (local only, 🔴):* `input/reference/{power_curves,models}.real.csv`,
  the renewables.ninja/VWF library from thewindpower.net. Swap in for production
  runs; the merged set is under `input/combined/reference/` (`PYVWF_INPUT=input/combined`).

**Region shapes.** `input/reference/shapes/*.geojson`, onshore/offshore outlines
for clustering (committed).

---

## 4. Country-level (ENTSO-E) workflow

The nine ENTSO-E regions use grid-sampled ERA5 against a national observed
series. `datasets/generate_country_level_training_data.py` fetches ENTSO-E
generation (needs `ENTSOE_API_KEY`) and builds the grid points; the harness then
trains and evaluates like any region. Layout:

```
input/observations/country/
├── grid_points/<cc>/<cc>_grid_points[_<year>].csv   # ID,lat,lon,height,model,capacity,cluster
└── observations/<cc>/<cc>_{train_<a>_<b>,test_<y>}.csv   # capacity_factor,generation_mw,capacity_mw
```

Zonal regions (NO, SE) also carry per-zone files and an `_aggregated` national
series.

Three steps make the grid trustworthy, and each has a script:

1. **Capacity-weight the grid.** The generator writes a uniform synthetic
   capacity, which weights the country aggregate by land area, not fleet. Fix it
   with real GWPT capacity (drops empty grid points):
   ```bash
   PYTHONPATH=src python scripts/region_tools/weight_country_grid_points.py --all --per-year 2015 2024
   ```
   `--per-year` writes one grid per year so train and test see the fleet as it
   stood (NL's more than quadrupled). Add `--zone-aware` for NO/SE so capacity
   is never summed across a zone boundary.

2. **Set `cluster_list` correctly.** It must be `1` (one national cluster, an
   exactly-determined fit) or the grid's own cluster count; anything else raises.

3. **Audit the observations** before trusting them:
   ```bash
   PYTHONPATH=src python scripts/analysis/audit_country_observations.py
   ```
   The affine correction absorbs a constant observation error into the scalar,
   so a uniformly-wrong series still fits well in sample. **NL and IE fail this
   audit** (see [method-country-level.md](../findings/method-country-level.md)).

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

| May be committed / shared | Must stay local (🔴 / 🟡) |
|---|---|
| Open curve library, region shapes, zone polygons | WindStats DE/ES generation and per-turbine derivatives |
| All fetch/process scripts, configs, curation tables | Ofgem confidential certificate-warehouse exports |
| Model results / metrics from any source | The licensed `*.real.csv` curve library |
| Provenance and audit tables | Raw `input/` of every region (git-ignored) |

Scripts that touch confidential data print a banner and write only to
git-ignored `input/`. Mixed-licence regions (UK confidential path, ES-WS) flag
the split in their runbook and processor.
