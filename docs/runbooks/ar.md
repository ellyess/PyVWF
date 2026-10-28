# Argentina (CAMMESA)

**Data source:** CAMMESA / MEM monthly renewables generation (public, no
credentials). **Adapter:** `cammesa-ar` · turbine-level · unit = plant · open.

The scorecard row's 59 plants run from Patagonia (Chubut, Santa Cruz) to the
Pampas; capacity and coordinates come from the Global Wind Power Tracker
(GWPT). Result: [scorecard](../scorecard.md), [results](../results.md#chile-and-argentina).

## Source and licence

The generation data are public CAMMESA downloads (the
[source table](../guides/data-sources.md#source-urls)). Nothing from them is
committed. Coordinates and capacities come from the Global Wind Power Tracker
(Global Energy Monitor, CC-BY-4.0), which requires attribution. The committed
curation tables are `configs/curation/ar_turbine_specs.csv`, every row of which
cites its source in `source_url`; `ar_coord_overrides.csv`, which carries a
`note` per row but no source column; and `ar_fleet_exclusions.csv`, which gives
each excluded code's reason.

## 1. Observations

```bash
python scripts/fetch/cammesa_ar.py     # downloads the ZIP -> input/raw/cammesa/ar_wind_monthly.csv
python scripts/process/cammesa_ar.py   # -> input/observations/turbine/AR/
```

The fetch's `--probe` reports coverage and writes nothing. `process` computes
CF for 2021-2024 and joins the GWPT workbook at
`input/reference/gwpt/Global-Wind-Power-Tracker-February-2026.xlsx`; read
`input/observations/turbine/AR/join_report.md` and `ar_join_residual.csv`
afterwards. GWPT is per farm, so a phase code (for example Loma Blanca 1/2/3)
would take the parent's full nameplate; where GWPT differs by more than 15%
from turbine count × unit MW, `configs/curation/ar_coord_overrides.csv`
overrides it. The build fails on a plant with no coordinate or capacity, and a
median monthly CF above 0.65 goes to the residual as a sign of too small a
capacity. `configs/curation/ar_fleet_exclusions.csv` drops zero-generation
duplicate codes, three plants at a steady CF of 0.06 to 0.08, self-generation
autoproducers, plants with no confident GWPT match, and Tierra del Fuego (south
of the ERA5 box). Review these judgment calls before publishing.

## 2. ERA5 and power curves

```bash
python scripts/fetch/era5.py --region ar   # 48 months 2021-2024
python scripts/region_tools/apply_turbine_specs.py AR \
    --specs configs/curation/ar_turbine_specs.csv
```

The box, `bbox = [-71.0, -56.0, -49.0, -27.0]` ([W, E, S, N]) in
`configs/regions/ar.toml`, goes to `era5/AR` ([data sources](../guides/data-sources.md#3-shared-inputs-all-regions)).
After `process`, `apply_turbine_specs.py` gives each plant an open-library curve
by scale and specific power, and its hub height; `--dry-run` writes nothing.

## 3. Train and evaluate

The scorecard row uses the default input root, `input/`, with no `PYVWF_INPUT`
([Choose the input root](../guides/training.md#choose-the-input-root)). Both
configs train 2021-2023 and test 2024. The maintained config sweeps
`cluster_list = [1, 12]`; the scorecard row (`cluster_list = [10]`) reruns at
the commit and through the lock that [Reproducing a row](../scorecard.md#reproducing-a-row) gives.

```bash
pyvwf-validate train --region configs/regions/ar.toml               # maintained
pyvwf-validate evaluate --region configs/regions/ar.toml \
    --train-run output/validation/AR/train-<timestamp>
pyvwf-validate train --region configs/regions/scorecard/ar_k10.toml # scorecard row
pyvwf-validate evaluate --region configs/regions/scorecard/ar_k10.toml \
    --train-run output/validation/AR/train-<timestamp>
```

## Caveats

- **Northern under-prediction.** ERA5's 0.25° grid under-resolves the La Rioja
  / Cuyo wind; only a finer wind product would fix it.
- **Degenerate fit.** The scorecard row is flagged † (one failed offset); check
  `fit_quality` on the factors table before using its per-cluster factors.
- **Capacity.** Denominators are GWPT estimates corrected by turbine research.
- **Curtailment** is unscreened, though Argentina curtails less than Chile.
- **Seasons.** The config lists Southern Hemisphere months explicitly; they
  must not inherit the Northern Hemisphere mapping.
