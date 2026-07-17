# Adding an observation source

PyVWF corrects reanalysis wind speeds against observed generation. Where that
observed generation comes from is pluggable: each region is served by an
`ObservationSource` adapter. The correction maths, interpolation, power curves,
and clustering are all source agnostic, so adding a region means writing one
adapter and registering it. No core module changes.

## What an adapter must provide

Subclass `vwf.sources.ObservationSource` and implement two methods.

### `load_metadata() -> pd.DataFrame`

The sites to simulate. Required columns:

| Column | Meaning |
| --- | --- |
| `ID` | Site identifier, string. Joins metadata to observations and to simulated output. |
| `lon`, `lat` | Location in degrees. |
| `height` | Hub height in metres, strictly greater than 1. |
| `capacity` | Rated capacity in kW. |
| `model` | Power curve key, matching a column of `input/power_curves.csv`. |
| `type` | `onshore` or `offshore`. Used when `cluster_mode` filters the fleet. |

Country-level sources must also supply `cluster`, the cluster assignment for each
grid point. Country-level corrections are derived per cluster, and no clustering
step runs on that path.

### `load_observations(year_start=None, year_end=None) -> pd.DataFrame`

Observed generation, already converted to capacity factor. Both year bounds are
inclusive. When both are `None`, return your default training window. Sources
whose data is already scoped by the caller may ignore both arguments.

The shape depends on the adapter's `obs_level`:

- `obs_level = "turbine"`: wide, one row per `(ID, year)`, with columns `ID`,
  `year`, and `obs_1` through `obs_12` holding the monthly mean capacity factor.
  Missing months are `NaN`.
- `obs_level = "country"`: a `DatetimeIndex`ed frame with a `capacity_factor`
  column, at whatever native resolution you have. The pipeline resamples to
  monthly means for training and keeps the native resolution for validation.

Capacity factors are dimensionless and normally lie in `[0, 1]`.

## Registering it

Three class attributes drive the registry:

```python
from vwf.sources import ObservationSource, register

@register
class AemoSource(ObservationSource):
    name = "aemo"            # unique registry key
    obs_level = "country"    # "turbine" or "country"
    countries = ("AU",)      # codes this adapter is resolved for

    def __init__(self, country: str) -> None:
        self.country = country.upper()

    def load_metadata(self):
        ...

    def load_observations(self, year_start=None, year_end=None):
        ...
```

Import the module once so the decorator runs. Adding it to
`src/vwf/sources/__init__.py` alongside the built-ins is enough.

Adapters listed in `countries` are resolved automatically from a country code,
so they must accept that code as their only constructor argument. Leave
`countries` empty for adapters the caller always constructs itself, as
`InMemoryCountrySource` does.

Once registered, the pipeline picks it up with no further changes:

```python
from vwf.vwf import PyVWF

model = PyVWF("", "AU", True, True, "all", [5], ["season"], obs_level="country")
model.train(check=False)
```

## Built-in adapters

| Name | Level | Countries | Notes |
| --- | --- | --- | --- |
| `european-turbine` | turbine | DK, DE, UK | Reads the per-turbine CSVs under `input/turbine_level_data/`. Converts monthly kWh to capacity factor. |
| `aemo-nem` | turbine | AU-NEM, AU | Per-farm (DUID) monthly CF from 5-minute AEMO SCADA, AEST→UTC binned. The unit is the farm (`obs_unit = "farm"`). |
| `eia-us` | turbine | US, USA | Per-plant monthly CF from EIA-923 net generation, with EIA-860 capacity/coordinates and USWTDB hub heights. The unit is the plant (`obs_unit = "plant"`). |
| `in-memory-country` | country | none | Wraps caller-supplied grid points and an observed capacity-factor series. Backs `PyVWF.load_country_data()`. |

## Country-level data supplied by the caller

Country-level observations (ENTSO-E derived, for example) are fetched and cached
outside the library, then handed to PyVWF:

```python
model.load_country_data(grid_points, obs_train, obs_test)
```

That wraps each frame in an `InMemoryCountrySource` internally. If you ask for
`obs_level="country"` without either loading data this way or registering an
adapter for the country, resolution raises `NotImplementedError` explaining both
options. There is no silent fallback.

## Testing a new adapter

`tests/test_sources.py` shows the pattern. The contract tests stub the underlying
loaders rather than reading real files, since `input/` is not tracked by git. The
last test in that file registers a throwaway adapter and drives it all the way to
`gen_cf` through `train_set`, which is the cheapest way to confirm a new source
is wired correctly end to end.
