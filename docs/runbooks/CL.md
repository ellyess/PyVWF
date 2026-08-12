# Chile (Coordinador Eléctrico Nacional)

**Source:** Coordinador SIP API `generacion-real`, per-plant hourly delivered
energy (free API key). **Adapter:** `cen-cl` · turbine-level · unit = plant ·
🟢 open. **Fleet:** 60 plants, ~6.5 GW, spanning three climates absent from the
rest of the set: Atacama coastal desert (BW), Mediterranean central Chile (Csb),
maritime south (Cfb).

CEN carries no coordinates or hub heights, so both are joined externally.

## 1. Fetch and process

```bash
export CEN_API_KEY=<your Información Pública (SIP) user_key>
python scripts/fetch/cen_cl.py --probe            # verify access, writes nothing
python scripts/fetch/cen_cl.py --years 2021 2024  # -> input/raw/cen/
python scripts/process/cen_cl.py                  # -> input/observations/turbine/CL/
python scripts/region_tools/apply_turbine_specs.py CL \
    --specs configs/curation/cl_turbine_specs.csv # real curves + hub heights
```

`process` reduces the hourly delivered energy to monthly CF against registered
capacity (fixed UTC-4, no DST), strips leading pre-operational months, and joins
coordinates from GWPT. `apply_turbine_specs` assigns each plant a real curve by
scale and specific power from the per-farm turbine table, and its real hub
height where known.

The coordinate join: 54 plants auto-matched to GWPT by name confirmed on
capacity, 6 hand-curated in `configs/curation/cl_coord_overrides.csv`, and 4 tiny
PMGD plants excluded (5-9 MW, absent from GWPT; the `EXCLUDE` list, add
coordinates to reinstate). A plant reaching training with no coordinate is a
hard error. Read `join_report.md` and `cl_coord_residual.csv` after processing.

## 2. Run

```bash
python scripts/fetch/era5.py --region cl          # 48 months 2021-2024, mainland box
PYVWF_INPUT=<combined-library root> \
python scripts/analysis/validate_region.py train    --region configs/regions/cl.toml
python scripts/analysis/validate_region.py evaluate --region configs/regions/cl.toml \
    --train-run output/validation/CL/train-<stamp>
```

Trains 2021-2023, tests 2024. With matched curves the affine correction helps
(uncorrected RMSE 0.123 → corrected 0.104, r 0.32 → 0.45).

## Caveats

- **Northern under-prediction is the ceiling.** In the Atacama ERA5 gives ~3.9
  m/s, which no correctly-specified turbine turns into the observed 21% CF;
  matched curves confirmed it is the reanalysis wind, not the turbine. Fixable
  only with a higher-resolution wind product. See
  [`south_america_spatial_bias.md`](../findings/south_america_spatial_bias.md).
- **Curtailment.** `gen_real_mw` is delivered energy, and Chile's northern grid
  curtails wind, so delivered CF sits below the resource. No screen: the SIP API
  exposes no vertimiento series (`factor_ernc` is flat at 1.0), and there is no
  known endpoint under the public key.
- **Static nameplate.** `potencia_maxima` is full from day one;
  `strip_commissioning_prefix` removes the leading pre-operational run, but a
  partial-build farm still reads low against the static value.

## API notes (verified July 2026)

| Trap | Reality |
| --- | --- |
| Host | `sipub.api.coordinador.cl` (no `servers` block in the spec; found by DNS). |
| Subscription | **Información Pública (SIP)**, not Operación (which is aggregate GWh only). |
| `page`/`limit` | Return HTTP 502; send `pageSize` alone, above the day's row count. |
| Envelope | Results in `data`, not the `content` the schema advertises. |
| `tipoTecnologia` | Must be `Eólica`, accented; the unaccented spelling returns an empty list. |
| Coordinates | Absent from the whole API; joined from GWPT. |
| Timestamps | Fixed UTC-4, no DST (7 Sep 2024 has 24 rows, not 23). |

Endpoints: `/generacion-real/v3/findByDate` (per-plant generation),
`/centrales/v4/findByDate` (registry), `/capacidad-instalada/v4/findByDate`
(capacity history). Auth is `user_key` as a query parameter.
