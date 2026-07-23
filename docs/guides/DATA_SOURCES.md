# Data sources

Every dataset PyVWF uses, where it comes from, its licence, and the scripts
that fetch and process it. This is the provenance reference for the validation
regions; per-region acquisition detail lives in `docs/runbooks/<CC>.md`, and the
adapter contract in [`ADDING_AN_OBSERVATION_SOURCE.md`](ADDING_AN_OBSERVATION_SOURCE.md).

**Licence key:** 🟢 open (redistributable) · 🟡 mixed (open coordinates over a
confidential generation series) · 🔴 confidential / commercial (never committed
or redistributed; `input/` is git-ignored). Results *derived* from confidential
inputs are shareable; the inputs and derived per-turbine tables are not.

Nothing under `input/` is committed to the repository (`/input` is git-ignored)
except the open, redistributable curve library in `input/reference/`. Every
download script is **user-executed**: it runs under your own credentials
(CDS key, ENTSO-E key, CEN key), never bundled data.

---

## 1. Observation sources by region

Each region declares its source in `configs/regions/<code>.toml`; the harness
resolves it to an adapter in `src/vwf/sources/`. Generation is converted to a
monthly capacity factor before correction.

| Region | Code | Adapter | Observation source | Lic. | Fetch → Process | Runbook |
| --- | --- | --- | --- | :---: | --- | --- |
| Denmark | DK | `european-turbine` | Danish Energy Agency master register (per-turbine monthly kWh) | 🟢 | `fetch/dk.py` → `process/dk.py` | [DK](../runbooks/DK.md) |
| United Kingdom | UK | `european-turbine` | REPD metadata (open) + Ofgem ROC certificates (open RER export **or** confidential warehouse) | 🟡 | `fetch/uk.py` → `process/uk.py` | [UK](../runbooks/UK.md) |
| Germany | DE | `european-turbine` | WindStats per-turbine register (commercial) | 🔴 | (none) → `process/de.py` | [DE](../runbooks/DE.md) |
| Spain (hist.) | ES-WS | `windstats` | WindStats per-turbine generation (commercial) + GWPT coordinates (open) | 🟡 | (none) → `process/windstats.py` | [ES](../runbooks/ES.md) |
| United States | US | `eia-us` | EIA-923 net generation, EIA-860 capacity/coords, USWTDB hub heights | 🟢 | (none) → `process/eia_us.py` | [US](../runbooks/US.md) |
| Brazil | BR | `ons-br` | ONS `FATOR_CAPACIDADE-2` hourly CF (+ constrained-off mask) | 🟢 | (none) → `process/ons_br.py` | [BR](../runbooks/BR.md) |
| Australia (NEM) | AU-NEM | `aemo-nem` | AEMO NEMWEB MMSDM 5-min SCADA + Generation Information | 🟢 | `fetch/aemo_au.sh` → `process/aemo_au.py` | (none) |
| New Zealand | NZ | `emi-nz` | EA EMI `Generation_MD` half-hourly metered injection | 🟢 | `fetch/emi_nz.py` → `process/emi_nz.py` | [NZ](../runbooks/NZ.md) |
| Chile | CL | `cen-cl` | Coordinador Eléctrico Nacional SIP API `generacion-real` hourly | 🟢 | `fetch/cen_cl.py` → `process/cen_cl.py` | [CL](../runbooks/CL.md) |
| Argentina | AR | `cammesa-ar` | CAMMESA / MEM monthly renewables generation | 🟢 | `fetch/cammesa_ar.py` → `process/cammesa_ar.py` | [AR](../runbooks/AR.md) |
| BE ES FR IE IT NL NO PT SE | (9) | `entsoe-country` | ENTSO-E Transparency actual generation per bidding zone | 🟢 | `datasets/generate_country_level_training_data.py` | (none) |

Turbine-level sources report at different native units (turbine / farm / plant /
complex), see the adapter table in
[`ADDING_AN_OBSERVATION_SOURCE.md`](ADDING_AN_OBSERVATION_SOURCE.md) for the
per-adapter unit, time convention, and curtailment handling.

### Source URLs

| Source | Endpoint | Access | Licence |
| --- | --- | --- | --- |
| Danish Energy Agency register | `ens.dk/media/4945/download`, `.../4948/download` | direct download | open public data, free with attribution |
| UK REPD (metadata) | `gov.uk/.../renewable-energy-planning-database-quarterly-extract` | direct download | Open Government Licence |
| Ofgem ROC (open) | `rer.ofgem.gov.uk` | RER export | open |
| Ofgem ROC (confidential) | "CertificatesExternalPublicDataWarehouse" | licensed warehouse | 🔴 differently licensed, not the open RER data |
| EIA-923 / EIA-860 | `eia.gov` survey forms | direct download | US public domain |
| USWTDB (hub heights) | USGS U.S. Wind Turbine Database | direct download | US public domain |
| ONS Brazil | `dados.ons.org.br` (dataset `fator-capacidade-2`; S3 mirror) | open data portal | ONS Open Data |
| AEMO NEM | `nemweb.com.au/Data_Archive/.../MMSDM`; `aemo.com.au/.../generation-information` | direct download | open |
| EA EMI (NZ) | `emi.ea.govt.nz/Wholesale/Datasets/Generation/Generation_MD/` | direct download | open |
| CEN Chile | `sip.api.coordinador.cl`, `portal.api.coordinador.cl` | API (free key) | public transparency data |
| CAMMESA (AR) | `cammesaweb.cammesa.com/erenovables/` | direct download | public |
| ENTSO-E | `transparency.entsoe.eu` | API (free key) | public-by-regulation (verify redistribution) |
| WindStats (DE, ES) | commercial subscription | (none) | 🔴 confidential / commercial |

---

## 2. Shared inputs (all regions)

### ERA5 reanalysis: the wind field
- **Source:** Copernicus Climate Data Store, `reanalysis-era5-single-levels`
  (hourly 100 m + 10 m u/v winds, 0.25°). Free, requires a CDS account
  (`~/.cdsapirc`). Licence: Copernicus, free use with attribution.
- **Fetch:** `scripts/fetch/era5.py --region <cc>` (box + years from the region
  TOML). Requests are batched by month (default 3; the CDS cost limit rejects
  6) and split back to per-month `era5_<tag>_<YYYY>_<MM>.nc` files.
- **Big boxes (US, BR)** are reduced to yearly daily means by
  `scripts/era5/combine.py`; small boxes are used raw.

### Coordinates: Global Wind Power Tracker (GWPT)
- **Source:** Global Energy Monitor, Global Wind Power Tracker.
  `input/reference/gwpt/Global-Wind-Power-Tracker-February-2026.xlsx`.
- **Licence:** CC-BY-4.0 (redistributable with attribution).
- **Used by:** CL, AR (coordinates, and capacity for AR), ES-WS (coordinates via
  the thewindpower name mapping). This is what makes the ES-WS and AR series
  usable when the generation source carries no location.

### Hub heights: GOWIRES
- **Source:** GOWIRES global hub-height dataset (Zenodo). Licence: CC-BY-4.0.
  ~32% global coverage, used as a fallback where the generation source omits hub
  height (e.g. Chile). NZ uses a curated per-farm height table instead.

### Power curves and turbine models
- **Open library (shipped, committed):** `input/reference/power_curves.csv`
  (76 curves) + `input/reference/models.csv`: NREL/turbine-models
  (NatLabRockies/turbine-models), **BSD-3-Clause**, DOI 10.11578/dc.20210112.1,
  VWF-smoothed. Also bundled inside the package (`vwf/resources/`) so an
  installed PyVWF runs outside a checkout. Provenance:
  `input/reference/power_curves_provenance.csv`.
- **Licensed library (local only, never redistributed):**
  `input/reference/power_curves.real.csv` / `models.real.csv`: the
  renewables.ninja / VWF library derived from thewindpower.net. 🔴 Swap in for
  production runs (see `input/README.md`); the merged 236-curve set lives under
  `input/combined/reference/` (run with `PYVWF_INPUT=input/combined`).

### Region shapes
- `input/reference/shapes/*.geojson`: onshore/offshore outlines and bidding
  zones for clustering. `country_shapes.geojson` and `offshore_shapes.geojson`
  are committed.

---

## 3. Evaluated but not shipped

- **Turkey (TR)**: EPİAŞ Şeffaflık (`seffaflik.epias.com.tr`). Demoted: the
  per-plant licensed-realtime-generation endpoint returns 403 under the public
  subscription, and UEVM aggregates ignore `powerPlantId`, so per-plant wind
  output is not obtainable openly. `fetch/epias_tr.py` + [RUNBOOK_TR](../runbooks/TR.md)
  document the gap.
- **Sweden & Finland (WindStats)**: the `windstats` format supports them, but
  GWPT matches only 7–9% of their fragmented fleets, so the open-coordinate path
  cannot place them. Parked pending a thewindpower.net coordinate table
  (see [RUNBOOK_ES](../runbooks/ES.md)).

---

## 4. Redistribution summary

| May be committed / shared | Must stay local (🔴 / 🟡) |
| --- | --- |
| The open curve library (`input/reference/{power_curves,models}.csv`) | WindStats DE/ES generation and anything derived per-turbine |
| Region shapes, provenance table | Ofgem confidential certificate-warehouse exports |
| All fetch/process **scripts** and configs | The licensed `*.real.csv` curve library |
| Model **results / metrics** derived from any source | Raw `input/` data of every region (`/input` is git-ignored) |

When a script touches confidential data it prints a confidentiality banner and
writes only to git-ignored `input/`. Mixed-licence regions (UK confidential
path, ES-WS) flag the split in their runbook and in the processing script.
