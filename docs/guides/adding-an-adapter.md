# Adding an adapter

An adapter is an `ObservationSource` subclass. It loads one data source for
the harness, and the registry finds it by name. The correction,
interpolation, power curves and clustering do not depend on the data source.
So a new adapter changes no core module.

A new region with a new data source needs an adapter and more.
[`adding-a-region.md`](adding-a-region.md) lists every file in order. This
guide covers the adapter itself.

## What an adapter must provide

Subclass `vwf.sources.ObservationSource`. Implement the two methods below.

### `load_metadata() -> pd.DataFrame`

The units to simulate. Required columns:

| Column | Meaning |
| --- | --- |
| `ID` | Unit identifier, a string. It joins metadata to observations and to simulated output. |
| `lon`, `lat` | Location in degrees. |
| `height` | Hub height in metres, greater than 1. |
| `capacity` | Rated capacity in kW. |
| `model` | Model key, matching a column of `power_curves.csv` under the input root. |
| `type` | `onshore` or `offshore`. `cluster_mode` uses it to filter the fleet. |

Other columns pass through. Keep `model_source` and each unit's own
manufacturer or model string; see [Curve assignment](adding-a-region.md#curve-assignment).

A country-level adapter must also supply `cluster`, each grid point's cluster.
Country-level corrections are fitted per cluster. No clustering step runs on
that path.

### `load_observations(year_start=None, year_end=None) -> pd.DataFrame`

Observed generation, already converted to capacity factor. Both year bounds
are inclusive. Both may be `None`. In that case, return your default training
years. An adapter whose data the caller has already scoped may ignore both
arguments.

The shape depends on the adapter's `obs_level`:

- `obs_level = "turbine"`: wide, one row per `(ID, year)`. The columns are
  `ID`, `year`, and `obs_1` to `obs_12`, each a monthly mean capacity factor.
  A missing month is `NaN`.
- `obs_level = "country"`: a frame with a `DatetimeIndex` and a
  `capacity_factor` column, at its native resolution. The pipeline resamples to monthly means for
  training. It keeps the native resolution for validation.

`obs_level` is the pipeline branch, not the unit. A data source with farm or
plant units still sets `obs_level = "turbine"`. The unit goes in the region
config's `obs_unit`.

Capacity factors are dimensionless and normally lie in `[0, 1]`.

## Registering it

Three class attributes drive the registry:

```python
from vwf.sources import ObservationSource, register

@register
class ExampleFarmSource(ObservationSource):
    name = "example-farms"   # unique registry key, used as the config's `source`
    obs_level = "turbine"    # "turbine" or "country"
    countries = ("XX",)      # region codes this adapter is resolved for

    def __init__(self, country: str) -> None:
        self.country = country.upper()

    def load_metadata(self):
        ...

    def load_observations(self, year_start=None, year_end=None):
        ...
```

Import the module once, so the decorator runs. An import line in
`src/vwf/sources/__init__.py`, beside the built-in adapters, is enough.

The registry resolves an adapter listed in `countries` from a region code. So
the adapter must take that code as its only constructor argument. Leave
`countries` empty for an adapter the caller always builds itself, as
`InMemoryCountrySource` does.

The harness then finds the adapter from the region config's `source` field.
Nothing else changes. [`adding-a-region.md`](adding-a-region.md) gives the
commands that run it.

## Built-in adapters

| Name | Level | Region codes | Notes |
| --- | --- | --- | --- |
| `european-turbine` | turbine | DK, DE, UK | Reads the per-turbine CSVs under `input/observations/turbine/` and converts monthly kWh to capacity factor. DK acquisition is scripted by `scripts/fetch/dk.py` and `scripts/process/dk.py`, from the Danish Energy Agency register (`docs/runbooks/dk.md`). UK is partly scripted by `scripts/fetch/uk.py` and `scripts/process/uk.py` (`docs/runbooks/uk.md`). UK metadata comes from the open REPD. UK ROC observations come from the open RER export or the CONFIDENTIAL Ofgem certificate warehouse. DE is staged from CONFIDENTIAL WindStats data by `scripts/process/de.py` (`docs/runbooks/de.md`). |
| `aemo-nem` | turbine | AU-NEM, AU | Per-farm (DUID) monthly CF from 5-minute AEMO SCADA, binned from AEST to UTC. The unit is the farm (`obs_unit = "farm"`). |
| `eia-us` | turbine | US, USA | Per-plant monthly CF from EIA-923 net generation. Capacity and coordinates come from EIA-860, and hub heights from USWTDB. The unit is the plant (`obs_unit = "plant"`). |
| `ons-br` | turbine | BR, BRA | Per-complex monthly CF from the ONS `FATOR_CAPACIDADE` hourly series, which carries its own coordinates and installed capacity. An optional ONS curtailment mask removes constrained-off hours. The unit is the complex (`obs_unit = "complex"`). |
| `emi-nz` | turbine | NZ, NZL | Per-farm monthly CF from EMI `Generation_MD` half-hourly kWh. NZ trading periods, with daylight saving (DST), are converted to UTC bins. The curated farm table (`configs/curation/nz_wind_farms.csv`) gives coordinates, capacities and hub heights. Commissioning-ramp months are masked. The unit is the farm (`obs_unit = "farm"`). |
| `cen-cl` | turbine | CL, CHL | Per-plant monthly CF from CEN SIP `generacion-real` hourly `gen_real_mw`, against `potencia_maxima`. Fixed UTC-4, with no daylight saving (DST), is converted to UTC bins. The fleet comes from the generation stream. Coordinates are joined from GWPT, with `configs/curation/cl_coord_overrides.csv` for the rest. Leading commissioning months are removed. The unit is the plant (`obs_unit = "plant"`). |
| `cammesa-ar` | turbine | AR, ARG | Per-plant monthly CF from CAMMESA monthly GWh, which is already monthly. CAMMESA carries no capacity. So coordinates and capacity are joined from GWPT, with curated overrides in `configs/curation/ar_coord_overrides.csv`. A median-CF guard flags doubtful capacity matches. Leading commissioning months are removed. The unit is the plant (`obs_unit = "plant"`). |
| `windstats` | turbine | ES-WS (SE-WS and FI-WS wait for coordinates) | Per-turbine monthly CF from the CONFIDENTIAL WindStats extract. Coordinates are joined from open GWPT through a farm-name mapping, so the licence is mixed. Built by `scripts/process/windstats.py`; see `docs/runbooks/es.md`. |
| `client-csv-turbine` | turbine | none | Your own fleet, from two CSVs: metadata and monthly generation. Column names can be remapped. Build it yourself and pass it to the driver as `source=`. See [`your-own-data.md`](your-own-data.md). |
| `entsoe-country` | country | none | The nine ENTSO-E regions' national series, from the `observations/country/` layout that `generate_country_level_training_data` writes. The driver builds one per split. |
| `entsoe-zonal` | country | none | Per-bidding-zone observations over the same layout, one per cluster. Zonal fits are therefore exactly determined. |
| `in-memory-country` | country | none | Wraps grid points and an observed capacity-factor series that the caller supplies. It backs `PyVWF.load_country_data()`. |

## Country-level data supplied by the caller

Country-level observations, such as ENTSO-E series, are fetched and cached
outside the library. They are then handed to PyVWF:

```python
model.load_country_data(grid_points, obs_train, obs_test)
```

That call wraps each frame in an `InMemoryCountrySource`. You might ask for
`obs_level="country"` without loading data this way, and without an adapter
for the region. Resolution then raises `NotImplementedError`, which explains
both options. Nothing falls back silently.

The ENTSO-E regions use their own path. See
[A country-level region](adding-a-region.md#a-country-level-region).

## Testing a new adapter

`tests/test_sources.py` shows the contract pattern. Its tests stub the loaders
instead of reading real files, because `input/` is not tracked by git. The last
test registers a throwaway adapter and drives it through `train_set` to
`gen_cf`. That is the cheapest end-to-end check that a new adapter is wired
correctly.

For a region, also add the processing tests of step 10 in
[`adding-a-region.md`](adding-a-region.md). `tests/test_emi_nz_processing.py`
is the fuller model.
