# Chile (Coordinador Eléctrico Nacional)

**Data source:** Coordinador SIP API `generacion-real`, per-plant hourly
delivered energy (free API key). **Adapter:** `cen-cl` · turbine-level · unit =
plant · open.

The scorecard row's 59 plants span the Atacama, central Chile and the maritime
south; CEN has no coordinates or hub heights, so both are joined externally.
Result: [scorecard](../scorecard.md), [results](../results.md#chile-and-argentina).

## Source and licence

The generation data are public transparency data from the Coordinador
Eléctrico Nacional, through its SIP API with a free key (the
[source table](../guides/data-sources.md#source-urls)). Nothing from them is
committed. Plant coordinates come from the Global Wind Power Tracker (Global
Energy Monitor, CC-BY-4.0), which requires attribution. The committed curation
tables are `configs/curation/cl_turbine_specs.csv`, every row of which cites
its source in `source_url`; `cl_coord_overrides.csv`, which carries a `note`
per row but no source column; and `cl_fleet_exclusions.csv`, which gives each
excluded plant's reason.

## 1. Observations

```bash
export CEN_API_KEY=<your Información Pública (SIP) user_key>
python scripts/fetch/cen_cl.py --probe            # verify access, writes nothing
python scripts/fetch/cen_cl.py --years 2021 2024  # -> input/raw/cen/
python scripts/process/cen_cl.py                  # -> input/observations/turbine/CL/
```

The key must be for **Información Pública (SIP)**, not Operación (aggregate
GWh only); the API's traps are in the fetch's docstring, and a re-run skips
completed months. `process` makes monthly CF against registered capacity
(fixed UTC-4, no DST), strips leading pre-operational months, and joins
coordinates from `input/reference/gwpt/Global-Wind-Power-Tracker-February-2026.xlsx`
by name, confirmed on capacity; co-located phases (for example Malleco Norte
and Sur) take the parent's coordinate. `configs/curation/cl_coord_overrides.csv`
holds six hand-curated plants; `configs/curation/cl_fleet_exclusions.csv` drops
four tiny PMGD plants absent from GWPT (add a coordinate override to reinstate
one). A plant with no coordinate is a hard error. Read `join_report.md` and
`cl_coord_residual.csv` afterwards.

## 2. ERA5 and power curves

```bash
python scripts/fetch/era5.py --region cl   # 48 months 2021-2024
python scripts/region_tools/apply_turbine_specs.py CL \
    --specs configs/curation/cl_turbine_specs.csv
```

The box, `bbox = [-75.0, -68.0, -44.0, -21.0]` ([W, E, S, N]) in
`configs/regions/cl.toml`, goes to `era5/CL` ([data sources](../guides/data-sources.md#3-shared-inputs-all-regions));
the lone Magallanes plant, off the SEN, falls outside it. After `process`,
`apply_turbine_specs.py` gives each plant an open-library curve by scale and
specific power, and its hub height where known; the rest keep the uniform
default, recorded in `model_source` and `height_source`. `--dry-run` writes
nothing.

## 3. Train and evaluate

The scorecard row uses the default input root, `input/`, with no `PYVWF_INPUT`
([Choose the input root](../guides/training.md#choose-the-input-root)). Both
configs train 2021-2023 and test 2024. The maintained config sweeps
`cluster_list = [1, 8]`; the scorecard row (`cluster_list = [10]`) reruns at
the commit and through the lock that [Reproducing a row](../scorecard.md#reproducing-a-row) gives.

```bash
pyvwf-validate train --region configs/regions/cl.toml               # maintained
pyvwf-validate evaluate --region configs/regions/cl.toml \
    --train-run output/validation/CL/train-<timestamp>
pyvwf-validate train --region configs/regions/scorecard/cl_k10.toml # scorecard row
pyvwf-validate evaluate --region configs/regions/scorecard/cl_k10.toml \
    --train-run output/validation/CL/train-<timestamp>
```

## Caveats

- **The gain is not resolved (‡):** resampling the test plants gives an
  interval that includes zero; see the [scorecard](../scorecard.md).
- **Northern under-prediction.** Atacama ERA5 wind is too calm; matched curves
  made it worse, so the limit is the reanalysis wind, fixable only with a finer
  wind product.
- **Cluster count does not predict fit health.** The sweep avoids 3 and 5,
  which blew up in the north, but the k10 row is flagged † with two refused
  clusters. Check `fit_quality` on the factors table.
- **Curtailment.** `gen_real_mw` is delivered energy and the northern grid
  curtails wind; the SIP API has no vertimiento series (`factor_ernc` is flat
  at 1.0), so CF is unscreened and sits below the resource.
- **Static nameplate.** `potencia_maxima` is full from day one, so a
  partial-build farm reads low after the pre-operational months are stripped.
