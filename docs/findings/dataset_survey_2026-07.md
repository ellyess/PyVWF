# Global survey — candidate observation datasets beyond DK/DE/UK/US/BR/AU

July 2026. Web survey of market-operator and registry data for new validation
regions, scoped by the adapter requirements in
`docs/ADDING_AN_OBSERVATION_SOURCE.md` (per-plant monthly CF or finer +
coordinates/capacity metadata; or zone/country aggregate + plant register).
Verification status per item: **verified** = primary portal/file/API inspected
during the survey; **reported** = primary-page claim not inspected file-level.

Context for prioritisation: the ML transfer re-test
(`ml_transfer_retest.md`) found the binding constraint on cross-region
prediction is **regime coverage, not sample count** — new regions should be
chosen to cover unrepresented physical regimes (monsoon, complex terrain,
Mediterranean, cold-steppe westerlies, coastal desert), not to add more
temperate-maritime plants.

## Tier 1 — per-plant observed generation, verified

| Region | Source | Granularity / resolution | History | Metadata | Climate added | Fleet | Notes |
|---|---|---|---|---|---|---|---|
| **New Zealand** | EA EMI `Generation_MD` (open CSV, no registration; Azure blob) | per plant, half-hourly | **1997→** | Register has capacity/commissioning, **no coords/heights** — but ~21 farms: hand-compile from NZWEA; coords from GWPT | SH temperate westerlies ("Roaring Forties"), **complex terrain** | ~1.3 GW, ~21 farms | Best value per unit effort. License page unverified. |
| **Chile** | Coordinador (CEN), bulk yearly per-central hourly + REST API (`portal.api.coordinador.cl`, registration) | per plant, hourly | 2000→ (wind ~2009→) | Infotécnica + Energía Abierta (capacity, commissioning); no heights | **Atacama coastal desert (BW), Mediterranean (Csb), maritime south (Cfb)** — 3 regimes in one system | ~4.5–5 GW, ~70 parks | Cloudflare blocks plain scraping; use API/bulk. |
| ~~**Turkey**~~ | ~~EPİAŞ/EXIST Transparency~~ | **DEMOTED — see below** | | | | | Per-plant access not reachable on a Data Consultation subscription (verified 2026-07-22). |
| **Canada (ON+AB)** | IESO Generator Output & Capability (per-generator hourly, 2010→, verified to CSV header) + AESO metered volumes (per-asset hourly, **2001–2025**, verified) | per generator/asset, hourly | 2001/2010→ | **Canadian Wind Turbine Database (NRCan): per-turbine coords, model, hub height, rotor, commissioning** — only candidate with native heights | Cold continental (Dfb/Dfa), icing, Chinook | ON ~5.5 GW + AB ~6 GW | Best data quality + metadata; least climate novelty (overlaps US Midwest). |
| **Argentina** | CAMMESA renewables base (Excel, open) | per plant, **monthly** verified **since 2011** (hourly claimed in GRV database, unverified) | 2011→ | Capacity per plant; coords from GWPT; no heights | **Patagonian cold-steppe westerlies (BSk/BWk)** + Pampas (Cfa) | ~4.3 GW, ~60 parks | Monthly per-plant = exactly PyVWF's native training resolution. Ramping fleet → commissioning mask needed (AU DUDETAIL pattern). |
| **Uruguay** | ADME_Data: per-plant **10-min raw SCADA incl. on-site wind speed**, free CLI, no registration | per plant, 10-min | mid-2010s→ (depth unverified) | No coords/heights from ADME; MIEM registry + GWPT | Humid subtropical Cfa, ~30–40% penetration | ~1.5 GW, ~45 parks | Wind-speed channels are a rare bonus for bias-correction diagnostics; raw/unvalidated SCADA. |
| **Peru** | COES portal, per-plant 30-min via open JSON endpoint (verified live) + daily IEOD Excels | per plant, 30-min | wind 2014→ (portal depth not pinned) | Fichas técnicas (capacity, turbine count); no coords/heights | **Hyper-arid coastal desert (BWh/BWn), steady Humboldt coastal jet** — most distinctive single regime | ~1.1 GW, ~10 plants | Tiny fleet → low effort. |
| **Australia WEM** | AEMO facility SCADA, open CSV 2006–Oct 2023 (verified directory) + post-reform 5-min via OpenElectricity API | per facility, 30-min/5-min | 2006→ | Same style as NEM | Mediterranean SW Australia (Csa/Csb), sea breeze | ~17 farms | Reuses existing AEMO tooling; note Oct-2023 regime break. |

## Verification outcomes (survey claims tested against live APIs)

The survey above was desk research. Two candidates have since been tested
against the real APIs, with opposite results — which is the argument for
verifying before committing adapter work.

- **Chile: CONFIRMED and better than surveyed.** Per-plant hourly wind runs
  from at least 2010 (5 plants) to 2025 (65 plants), with `potencia_maxima`
  on every row so the CF denominator tracks the build-out for free. Details
  and the gotcha list: `docs/RUNBOOK_CL.md`.
- **Turkey: DEMOTED.** The claim "per-plant hourly via free API" does not
  hold on a Data Consultation subscription. Authentication and the
  2,584-plant list work, but the UEVM endpoint **silently ignores
  `powerPlantId`** and returns the national fuel-mix series, and the
  renewables licensed-generation endpoint returns 403. A
  must-distinguish test (same request, two different plant ids → identical
  bytes) is what exposed it; the endpoint otherwise answers 200 with a
  plausible 24 rows and reads exactly like success. Recovery path in
  `docs/RUNBOOK_TR.md`.

The general lesson, consistent with the branch's evidence discipline: **a
200 response with plausibly-shaped data is not verification.** Only a test
that could have failed — a different id, a different year, a different
region — distinguishes real per-plant data from an ignored parameter.

## Tier 2 — zone/country aggregates (usable via the country-level path)

- **India — CEA RE portal (verified):** state-wise wind, daily + monthly, PDF **and Excel**, archives back to ≥2022 (deeper scattered). No public per-plant generation anywhere. **Monsoon (Aw/BSh) is the single biggest climate gap in the programme**; 8–9 wind states ≈ 8–9 zonal series; coords/capacity from GWPT. Country-level path fits (state = "country" unit).
- **Japan — OCCTO / TSO area data (verified for Kyushu):** 10 balancing areas, hourly incl. curtailment column, CSV, April 2016→. Wind concentrated in Hokkaido/Tohoku/Kyushu. Cold maritime-continental + East Asian winter monsoon + typhoons.
- **South Africa — Eskom data portal:** national hourly wind (REIPPPP fleet), current months on portal, 5-year bulk via free data-request form; community archive to ~2017. Semi-arid Karoo + coastal Cape, SH.
- **Ember monthly electricity (CC-BY-4.0, ~88 countries, API):** the only genuine multi-country observed monthly shortcut for countries that will never publish per-plant data. Start ~2015 typical (verify per country).

## Not viable now (checked)

China (provincial monthly behind commercial terminals, no redistribution),
Mexico/CENACE (technology-aggregate national only — climate-rich, data-poor),
Colombia (open hourly per-plant API but ~40 MW fleet until La Guajira lands —
revisit ~2028), South Korea (per-plant but annual-only; renewables partially
omitted), Taiwan (per-unit 10-min but real-time only, no official archive),
Vietnam/Thailand/Kazakhstan/Middle East (aggregates in reports at best),
Philippines (metered data members-only), Costa Rica / Dominican Republic
(small, partially unverified), Faroe (aggregate curiosity), Hydro-Québec
(provincial aggregate only).

**Not usable as ground truth:** Climate TRACE per-plant "generation"
(satellite-ML **estimates**, not meters); EMHIRES and renewables.ninja country
CFs (simulated).

## Metadata gap-fillers

- **GOWIRES** (Scientific Data 2026; Zenodo, CC-BY-4.0): 416k onshore
  turbines, 89 countries; coords + specs; **hub-height completeness ~32%**.
  Best single open global height layer.
- **Global Wind Power Tracker** (verified against the local Feb-2026 file):
  coords/capacity/status/phase only — **no hub heights, models, or turbine
  counts**. CC-BY-4.0.
- **Canadian Wind Turbine Database** (OGL-Canada): per-turbine heights/models
  — pairs natively with IESO/AESO.
- **OSM `height:hub`**: ~34k turbines globally (<10% coverage), DE/FR/PL
  concentrated; ODbL share-alike.
- **The Wind Power (commercial)**: per-farm models/heights at scale, but
  subscription + no republication of raw fields (reproducibility hazard —
  cf. the open-library finding on this branch).
- Academic open SCADA (Hill of Towie 21 turbines/8.7 y CC-BY-4.0, Kelmarsh,
  Penmanshiel): UK-only — validation-grade, no new climate.

## Recommended order of attack

1. **New Zealand (EMI)** — tier-1, trivially open, complex terrain + SH
   westerlies; one afternoon of hand metadata.
2. **Chile (CEN)** — three new regimes in one adapter, hourly since ~2009;
   register on the API portal first.
3. **Turkey (EPİAŞ)** — DEMOTED 2026-07-22: per-plant access is not reachable
   on a Data Consultation subscription (the UEVM endpoint ignores
   powerPlantId). Needs a renewables-service subscription; see RUNBOOK_TR.md.
4. **Argentina (CAMMESA)** — monthly per-plant since 2011 matches the training
   resolution exactly; Patagonia is a regime nothing else offers.
5. **Canada (IESO+AESO+CWTDB)** — when hub-height-sensitive questions (RQ7)
   come up: it is the only region where heights are native and per-turbine.
6. **India (CEA, state-level)** — through the country-level path; the monsoon
   gap is worth an "acceptable-tier" ingest.
7. **WEM / Uruguay / Peru** — cheap adds when marginal effort is available.
