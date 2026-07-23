# Argentina (CAMMESA) region — data acquisition and processing runbook

**Status: adapter BUILT and tested against the real data (2026-07-23); only
the ERA5 download and the train/evaluate remain.** Argentina was the cheapest
tier-1 acquisition of the whole survey — one ~2 MB ZIP, no credentials, no
API, no rate limit — and its data is already at PyVWF's native monthly
resolution, so the adapter has no timezone/DST logic at all. The scientific
draw is Patagonia (Chubut + Santa Cruz) in the cold-steppe westerlies, a
regime nothing else in the validation set covers. 65 plants, ~5.5 GW.

## 0. What is already committed

- The `cammesa-ar` adapter (`vwf/sources/cammesa_ar.py`), transforms
  (`vwf/datasets/cammesa_ar.py`), processing with the GWPT join
  (`scripts/process/cammesa_ar.py`), region config, and 10 tests.
- `configs/curation/ar_coord_overrides.csv` — hand-curated capacity fixes (see join).

## 1. Acquire + process (user-executed, no credentials)

```bash
python scripts/fetch/cammesa_ar.py     # downloads the ZIP, writes ar_wind_monthly.csv
python scripts/process/cammesa_ar.py   # -> input/turbine_level_data/AR/
```

`fetch` pulls `Energía Renovables - Base de Datos` and extracts the wind rows
of "Tabla Resumen x Central" (header on row 4) to monthly GWh per central.
`process` computes CF and joins coordinates + capacity from GWPT.

## 2. The capacity/coordinate join (the one hard part)

CAMMESA carries **no capacity** — every sheet is energy — so the CF
denominator is external. Each central is matched to a GWPT operating farm by
normalised name, which supplies BOTH lon/lat and capacity. Because capacity is
the *output* of the join it cannot confirm the match, so a guard runs on the
result: a plant whose **median monthly CF exceeds 0.65** almost certainly
matched a too-small capacity (real Argentine wind tops out near 0.5-0.6) and
is dropped to the residual for re-curation.

Outcome on the real data: **66 auto-matched, 2 capacity-fixed, 10 excluded**
(→ 65 plants, CF mean 0.35, median 0.38, max 0.76, none > 1).

- **Capacity fixes** (`configs/curation/ar_coord_overrides.csv`): GWPT located these but
  at a partial-phase capacity. *Manantiales Behr (YPF)* → 99 MW (30×3.3,
  Chubut). *Mataco 3 Picos* → 200 MW (El Mataco + Tres Picos complex near
  Bahía Blanca; GWPT splits it three ways).
- **Exclusions** (`EXCLUDE` in `scripts/process/cammesa_ar.py`): 10 plants, mostly
  **self-generation autoproducers** absent from a confident GWPT match — wind
  at a cement works (Cementos Avellaneda), an oil field (YPF El Tordillo), and
  an aluminium smelter (ALUAR), plus small/ambiguous farms. **Fin del Mundo**
  is excluded for a different reason: it is in Tierra del Fuego, **south of the
  −49 ERA5 box** (the Chile-Magallanes pattern). Three sizeable ones —
  **Cañadón León / Casa YPF Luz (~123 MW, Santa Cruz)**, ALUAR, and La Elbita
  — are flagged in the code as worth reinstating via the override table if
  their coordinates and capacity can be verified. Each override/exclude is a
  documented judgment call; **review before publishing.**

Read `input/turbine_level_data/AR/join_report.md` and
`ar_join_residual.csv` after processing.

## 3. ERA5 + run

```bash
python scripts/fetch/era5.py --region ar        # 48 months 2021-2024, Patagonia+Pampas box
PYVWF_INPUT=<combined-library root> \
python scripts/analysis/validate_region.py train    --region configs/regions/ar.toml
python scripts/analysis/validate_region.py evaluate --region configs/regions/ar.toml \
    --train-run output/validation/AR/train-<stamp>
```

## Caveats carried into any Argentina result

- **Static nameplate.** GWPT capacity is the final build; a farm partway
  through its turbine install reads low against it. `strip_commissioning_prefix`
  removes the leading pre-operational months, but a *partial-build* farm
  generating for a year still reads at a fraction of true CF — a below-plateau
  mask is the named follow-up. Over 2021-2024 the fleet is mostly built out
  (~56→68 plants), so this bites fewer plants than the 2011 start would.
- **Capacity provenance.** Every CF depends on a GWPT capacity that is itself a
  tracker estimate; the two hand-fixed plants aside, treat the denominators as
  good-but-not-authoritative.
- **Curtailment** is far lighter than Chile/Brazil (Argentina's grid curtails
  little wind over this window), so delivered CF is closer to the resource —
  but it is unscreened, a standing caveat.
- **Southern Hemisphere.** Seasons are explicit SH in the config. Patagonia's
  westerlies have modest seasonal amplitude; read the seasonal slice against
  that.
