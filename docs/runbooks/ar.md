# Argentina (CAMMESA)

**Source:** CAMMESA / MEM monthly renewables generation (public, no credentials).
**Adapter:** `cammesa-ar` · turbine-level · unit = plant · 🟢 open.
**Fleet:** 59 plants, ~5 GW. The scientific draw is Patagonia (Chubut, Santa
Cruz) in the cold-steppe westerlies, a regime nothing else in the set covers.

CAMMESA reports monthly GWh, already at PyVWF's native resolution, so the
adapter has no timezone or DST logic. It carries no capacity or coordinates, so
both are joined from the Global Wind Power Tracker; the capacity join is the
one hard part (see step 2).

## 1. Acquire and process

```bash
python scripts/fetch/cammesa_ar.py     # downloads the ZIP -> ar_wind_monthly.csv
python scripts/process/cammesa_ar.py   # -> input/observations/turbine/AR/
python scripts/region_tools/apply_turbine_specs.py AR \
    --specs configs/curation/ar_turbine_specs.csv   # real curves + hub heights
```

`fetch` pulls *Energía Renovables - Base de Datos* and extracts the wind rows of
"Tabla Resumen x Central" to monthly GWh. `process` computes CF and joins
coordinates + capacity from GWPT. `apply_turbine_specs` replaces the uniform
default curve with a real curve per plant, matched by scale and specific power
from the per-farm turbine table (manufacturer, model, count, rotor diameter, hub
height).

Read `input/observations/turbine/AR/join_report.md` and `ar_join_residual.csv`
after processing.

## 2. The capacity join

CAMMESA has no capacity, so the CF denominator is external. Each plant is
matched to a GWPT operating farm by normalised name, which supplies lon/lat and
capacity. GWPT is per-*farm*, so a phase code (e.g. Loma Blanca 1/2/3) matched
the parent's full nameplate and read an impossibly low CF (0.03 to 0.07).

Fixed with the turbine research: real nameplate = turbine count × unit MW, from
`configs/curation/ar_turbine_specs.csv`, written as capacity overrides in
`configs/curation/ar_coord_overrides.csv` (20 rows) where GWPT diverged by more
than 15%. Six codes are hard-dropped in `EXCLUDE` (`scripts/process/cammesa_ar.py`):
three with zero generation across the window (duplicate MEM codes whose output
is reported elsewhere) and three with a steady CF near 0.07 (the state-owned
Arauco farm's old IMPSA turbines in weak La Rioja wind, and a La Castellana II
anomaly). Self-generation autoproducers (cement works, oil fields, the ALUAR
smelter) and Tierra del Fuego (south of the ERA5 box) are also excluded.

Result: 59 plants, observed CF median 0.47, none below 0.12 or above 0.65. Every
override and exclusion is a documented judgment call; review before publishing.

## 3. ERA5 and run

```bash
python scripts/fetch/era5.py --region ar        # 48 months 2021-2024, Patagonia+Pampas box
PYVWF_INPUT=<combined-library root> \
python scripts/analysis/validate_region.py train    --region configs/regions/ar.toml
python scripts/analysis/validate_region.py evaluate --region configs/regions/ar.toml \
    --train-run output/validation/AR/train-<stamp>
```

Trains 2021-2023, tests 2024. On the cleaned fleet the affine correction helps
(uncorrected RMSE 0.151 / r 0.41 → corrected 0.133 / r 0.44).

## Caveats

- **Northern under-prediction.** La Rioja / Cuyo plants sit where ERA5's 0.25°
  grid under-resolves the wind (the same limit as Chile's Atacama); their
  cluster fits an extreme scalar. This is the reanalysis-wind limit, fixable
  only with a higher-resolution wind product. See
  [`region-south-america.md`](../findings/region-south-america.md).
- **Capacity provenance.** Denominators are GWPT estimates corrected by the
  turbine research; good but not authoritative.
- **Curtailment** is light in Argentina over this window but unscreened
  (standing caveat).
- **Southern Hemisphere.** Seasons are explicit SH in the config; Patagonia's
  westerlies have modest seasonal amplitude.
