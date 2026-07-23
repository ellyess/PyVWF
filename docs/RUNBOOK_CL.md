# Chile (Coordinador Eléctrico Nacional) — data acquisition runbook

**Status: adapter BUILT and tested against the real data (2026-07-23); only
the ERA5 download and the train/evaluate remain.** Per-plant hourly wind runs
back to at least 2010 (5 plants → 65 in 2025); the 2021-2024 pull is in
`input/cen_raw/` (48 files, 1.93M rows). The `cen-cl` adapter, its transforms
(`vwf/datasets/cen_cl.py`), the processing + GWPT join
(`scripts/process_cen_cl.py`), and 13 tests are committed. Chile spans three
climates absent from the current set — Atacama coastal desert (BW),
Mediterranean central Chile (Csb), maritime south (Cfb) — 60 plants, ~6.5 GW.

## Build the region inputs (done; re-run to refresh)

```bash
python scripts/process_cen_cl.py     # -> input/turbine_level_data/CL/
```

Produces `cl_obs.csv` (monthly CF, commissioning prefix stripped),
`cl_md.csv` (metadata with GWPT coordinates), `join_report.md`, and
`cl_coord_residual.csv`. The coordinate join: 54 plants auto-matched to GWPT
by name confirmed on capacity, 6 hand-curated in
`configs/cl_coord_overrides.csv` (Horizonte N/S, Kallpa, San Pedro II, El
Maiten, El Nogal — with per-row confidence notes), and **4 tiny PMGD plants
excluded** (Raki, Huajache, Las Peñas, Lebu III; 5-9 MW each, ~30 MW total,
absent from GWPT — the `EXCLUDE` list in the script, add coordinates to
reinstate). A plant reaching training with no coordinate is a hard error.

**Two caveats that shape any Chile result:**
- **Curtailment.** `gen_real_mw` is *delivered* energy, and Chile's northern
  grid curtails wind heavily; the fleet-mean delivered CF is ~0.24, suppressed
  below the resource. This is the Brazil-Nordeste contaminant — the affine
  correction will partly absorb curtailment as reanalysis bias. A CEN
  wind-curtailment ("vertimiento") series would let it be screened; not wired.
- **Static nameplate.** `potencia_maxima` is full from day one, so
  pre-commissioning months read CF~0; `strip_commissioning_prefix` removes the
  leading run per plant (108 months, 25 plants). Later low months are kept as
  real delivered CF.

Everything below is the original user-executed acquisition path (already done).

## 1. What to get

| Data | Source | Notes |
| --- | --- | --- |
| Per-plant hourly generation ("Generación Real por Central") | coordinador.cl → Operación → Gráficos → Generación Real; bulk "Histórico Generación Horaria por Central" in year blocks (2000-2015, 2016-2019, 2020-2023, 2024, 2025) | The page embeds Qlik dashboards with CSV export; the site is behind Cloudflare, so scripted scraping of the WordPress pages fails — use the bulk files or the API |
| REST API (robust route) | https://portal.api.coordinador.cl/ | 3scale developer portal; **free registration required** (this is the step only you can do). After registering, an API key gives programmatic access |
| Plant registry (capacity, commissioning) | CEN Infotécnica + CNE Energía Abierta (http://energiaabierta.cl, explicitly open data) | Installed-capacity datasets with commissioning dates — the registered-capacity history the harness wants |
| Coordinates | Infotécnica (reported) or Global Wind Power Tracker (CC-BY-4.0, on disk) | Verify Infotécnica coordinates against GWPT for a few parks before trusting either |
| Hub heights | Not in CEN data | Chilean SEA environmental filings per project, or GOWIRES (Zenodo, CC-BY-4.0, ~32% global coverage) |

## 1b. Gotchas that cost time (all verified against the live API)

| Trap | Reality |
| --- | --- |
| Host | `sipub.api.coordinador.cl`. The SIP OpenAPI document has **no `servers` block**, and `sip.api.coordinador.cl` does not resolve. Found by DNS. |
| Which subscription | **Información Pública (SIP)**, not Operación. Operación exposes aggregate GWh only. |
| `page` / `limit` | Return **HTTP 502 "Internal server error"** on both `/centrales/v4` and `/generacion-real/v3`. Send `pageSize` alone and size it above the day's row count; the endpoint cannot be paged. |
| Response envelope | `data` (SIP) or `results`+`count` (Infotécnica resources) — **not** the `content` the schema advertises. |
| `tipoTecnologia` | Must be **`Eólica`, accented**. `Eolica` returns an empty list rather than an error — reads exactly like "no data exists". |
| Filtering wind in code | Match the label exactly. A substring test on `"lica"` also matches **`Hidráulica`** (this produced a 4x row overcount during verification). |
| `/centrales/v4/findByDate` | Caps at **10 records** for any window and its wind entry is a `[NO_MOSTRAR]` test record with `"Sin información"` coordinates. Not a fleet source. |
| Coordinates | **Absent from the whole API** — `/centrales` advertises `coordenada_este`/`norte`/`zona_huso` but they are unpopulated, and Infotécnica (1,373 plants) has none. Join the Global Wind Power Tracker. |
| Metadata | Comes free on every generation row: `id_central`, `central`, `propietario`, `potencia_maxima` (populated for 62/62 plants), `tipo_tecnologia`. |
| Timestamps | **Fixed offset, NO DST** (verified 2026-07-23 on the real data). `fecha_hora` runs a clean 24 hours every day including Chile's 2024 spring-forward (7 Sep 2024 has 24 rows, not 23), so CEN publishes in fixed Chilean standard time, not civil `America/Santiago`. `hora` is 1-24 with `hora N` at `(N-1):00`. This is *simpler* than the NZ/BR cases — a single fixed shift to UTC, no DST branch. Offset is almost certainly **UTC-4** (continental standard); it only matters at month boundaries for a monthly mean, but confirm UTC-4 vs -3 against one known reference before the adapter bins. |

## 1a. Endpoint map (verified July 2026 from the public OpenAPI specs)

**The per-plant data is in the SIP service, not Operation.** Operation's
`/reportes/v3/generation` returns aggregate GWh by classification only. The
service specs are public at
`https://portal.api.coordinador.cl/api_docs/services/<n>.json` (3 = Operaciones,
4 = sip). Base URL `https://sip.api.coordinador.cl`; auth is `user_key` as a
**query parameter**.

| Endpoint | Gives |
| --- | --- |
| `/generacion-real/v3/findByDate` | Per-plant real generation. Params `startDate`, `endDate`, **`idCentral`**, `tipoTecnologia`, `page`, `pageSize`. Paged response (`content`, `totalElements`, `totalPages`) |
| `/centrales/v4/findByDate` | Plant registry: `id_central`, `central`, `propietario`, `tipo_tecnologia`, `cant_und_gen`, `pot_max_bruta`, `capac_max`, `fecha_ent_oper`, `region`/`provincia`/`comuna` |
| `/capacidad-instalada/v4/findByDate` | Installed-capacity history (then-current CF denominator) |
| `/api/v2/recursos/infotecnica/centrales/` | Infotécnica registry (alternative plant list) |

**The registry carries no latitude/longitude** — only region/province/commune.
Coordinates must be joined from the Global Wind Power Tracker (CC-BY-4.0,
already on disk at `input/Global-Wind-Power-Tracker-February-2026.xlsx`).

## 2. Verification checklist before writing the adapter

Run the probe — it answers 1-3 in one go and writes nothing:

```bash
export CEN_API_KEY=<your SIP user_key>
python scripts/fetch_cen_cl.py --probe
```

It prints the registry field list, the wind-plant count, whether coordinates
are present, and the row count for one plant across 2015/2018/2020/2022/2024/
2025 — the earliest year with rows is the per-plant history depth, which is
the open question that decides whether Chile is a full tier-1 region.

1. One bulk year-block file opened: confirm per-central hourly MWh (or MW),
   column names, encoding, and how wind plants are identified (technology
   column vs name prefix).
2. Timezone of timestamps (Chile has DST — `America/Santiago`; confirm
   whether CEN publishes local or UTC) — this decides the UTC-binning step,
   same hazard as the AEMO market-time and NZ trading-period cuts.
3. Match rate plant-registry ↔ generation file IDs, and against GWPT
   coordinates.
4. Curtailment: the northern grid curtails solar heavily; check whether CEN
   publishes wind curtailment/reduction series ("vertimiento") — if so it
   feeds the BR-style curtailment mask.
5. License/terms on the API portal at registration.

## 3. Expected shape of the region

- Adapter: `cen-cl`, `obs_level = "turbine"`, `obs_unit = "plant"`, monthly
  CF from hourly energy against registered capacity (then-current if the
  registry history supports it).
- ERA5 box: mainland Chile is long and thin — roughly lon [-76, -66],
  lat [-46, -17] covers the fleet (check the actual park list; Magallanes has
  one park near -53 that would double the box for one site — the Hawaii/bbox
  lesson from the US run applies).
- Seasons: Southern Hemisphere explicit month lists (copy `nz.toml`).
- Train/test: hourly data since ~2009 supports a long window; match the
  ERA5 fetch years to whatever window is chosen.
