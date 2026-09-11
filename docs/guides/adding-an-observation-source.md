# Adding an observation source

PyVWF corrects reanalysis wind speeds against observed generation. Where that
observed generation comes from is pluggable: each region is served by an
`ObservationSource` adapter. The correction maths, interpolation, power curves
and clustering are source agnostic, so a new region changes no core module. It
does touch about fifteen files, in a fixed order, because every region carries
its own acquisition, processing, tests, documentation and provenance.

## A turbine or plant-level region, file by file

The order below is dependency order. New Zealand (`emi-nz`) is the template to
copy and Chile (`cen-cl`) the cross-check; both follow it cleanly. Country-level
regions built from ENTSO-E do not follow it; see
[Country-level data supplied by the caller](#country-level-data-supplied-by-the-caller).

| # | File | What it holds |
|---|---|---|
| 1 | [`data-sources.md`](data-sources.md) row | The source, its access route and its licence key (open, mixed or confidential), recorded before anything is fetched. Nothing from a confidential source is committed; it stays under the git-ignored `input/`. |
| 2 | `configs/regions/<code>.toml` | Source, `obs_level`, `obs_unit`, train and test years, ERA5 box and `file_tag`, correction model, `cluster_list`, time slices, and explicit season month lists. See [`training.md`](training.md). It comes first because the ERA5 fetch reads its box and years. |
| 3 | `scripts/fetch/<source>.py` | Downloads raw observations into `<input-root>/raw/<source>/`, honouring `PYVWF_INPUT`. Run by the user, with their own credentials where needed. |
| 4 | (no new file) `scripts/fetch/era5.py --region <code>` | ERA5 for the config's box and years, into `<input-root>/era5/<file_tag>/`. Large boxes are reduced with `scripts/era5/combine.py --region <code>`. |
| 5 | `configs/curation/<code>_*.csv` | Hand-curated tables: farm coordinates, turbine specifications, capacity stages, months to mask. Give each row a source (a `source_url` column), and check the source's terms before committing cells transcribed from it. |
| 6 | `src/vwf/datasets/<source>.py` | Pure frame-to-frame transforms (time conventions, monthly capacity factor, masks), with no file I/O, so they are testable without the raw data. |
| 7 | `scripts/process/<source>.py` | The I/O wrapper. Writes `<code>_md.csv` and `<code>_obs.csv` (plus any mask and a `join_report.md`) to `input/observations/turbine/<CODE>/`, and assigns each unit a curve (see [Curve assignment](#curve-assignment)). |
| 8 | `src/vwf/sources/<source>.py` | The adapter, decorated with `@register`. See [What an adapter must provide](#what-an-adapter-must-provide). |
| 9 | `src/vwf/sources/__init__.py` | One import line, so the registration runs. |
| 10 | `tests/test_<source>_processing.py` | Synthetic tests of the transforms and of the adapter's resolve and load. NZ's and Chile's are the models. |
| 11 | `tests/test_harness_regions.py` | Bump the shipped-config count in `test_all_shipped_configs_load`, and pin the region's `obs_unit` and `obs_level` in `test_shipped_granularity_classification`. |
| 12 | This guide's [built-in adapters](#built-in-adapters) table | One row. |
| 13 | `docs/runbooks/<code>.md` | Acquisition and processing steps, the licence, and how to refresh. |
| 14 | `docs/findings/region-<code>.md` | The validation write-up, after the runs. Named and shaped per [`docs/README.md`](../README.md). |
| 15 | `docs/findings/scorecard.md` row, and `configs/regions/scorecard/<code>_k<N>.toml` | The reported row, and a byte-identical copy of the exact configuration that produced it. The maintained config in step 2 carries a sweep; the scorecard file fixes one cluster count and time slice. |

Between steps 13 and 14, run the region through the harness:

```bash
PYVWF_INPUT=<input root> python scripts/analysis/validate_region.py train \
    --region configs/regions/<code>.toml
PYVWF_INPUT=<input root> python scripts/analysis/validate_region.py evaluate \
    --region configs/regions/<code>.toml --train-run output/validation/<CODE>/train-<stamp>
```

Set `PYVWF_INPUT` for both commands. The curve library is resolved at each
step, and the manifest records which one each run used.

What to read in the outputs, per [`output-structure.md`](output-structure.md):

- `metrics.csv`: the skill table, with the uncorrected row first.
- The `fit_quality` columns: `max_scalar`, `n_implausible_scalar` and
  `n_failed_offset`.
- `substituted_capacity_share`, and each run's `curve_resolution.csv`: which
  curve every unit was actually simulated on.
- The cross-manufacturer audit. Add the region to `RUNS` and
  `own_manufacturer` in `scripts/analysis/curve_match_audit.py`, then run it.

## Curve assignment

Each unit's `model` key selects its power curve. Three routes exist today. Two
match on specific power rather than on the turbine's make; the third assigns
one curve to every unit:

| Route | Where | Used by |
|---|---|---|
| `vwf.datasets.eia_us.assign_curves_from_library` | processing time | US, NZ, and CL and AR through `scripts/region_tools/apply_turbine_specs.py` |
| `vwf.data.add_models` | load time, inside the adapter; processing time through `scripts/region_tools/assign_au_curves.py` | DE, DK, UK (`european-turbine`), `client-csv-turbine`, and AU-NEM |
| One uniform default curve | processing time | BR |

The first route restricts to onshore models rated 0.5 to 2 times the unit's
per-turbine rating, then takes the nearest specific power. It is imported from
the US module, even by NZ. It writes `model_source`, recording how each unit was
matched, and falls back to a named uniform curve where a unit cannot be matched.

The second route matches a fuzzily named manufacturer first, then nearest
specific power, and records the tier it used in `model_match`.

Whichever route a region takes:

- **Record the unit's own manufacturer or model string in its metadata**
  (NZ's `true_model`, US's `uswtdb_model`), or in a curation table keyed by
  `ID`. Otherwise the cross-manufacturer audit can only report the region as
  unverifiable, as it does for BR.
- **A key the curve table lacks does not stop a run.** The unit is simulated on
  the table's first column, and `curve_resolution.csv` records it as
  substituted. Check `substituted_capacity_share` is zero before reading a
  result.

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
| `model` | Power curve key, matching a column of `power_curves.csv` under the input root. |
| `type` | `onshore` or `offshore`. Used when `cluster_mode` filters the fleet. |

Other columns pass through; `model_source` and the unit's own manufacturer or
model string are worth keeping (see [Curve assignment](#curve-assignment)).

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

`obs_level` is the pipeline branch, not the unit of measurement. A per-farm or
per-plant source is still `obs_level = "turbine"`; the true unit goes in the
region config's `obs_unit`.

Capacity factors are dimensionless and normally lie in `[0, 1]`.

## Registering it

Three class attributes drive the registry:

```python
from vwf.sources import ObservationSource, register

@register
class ExampleFarmSource(ObservationSource):
    name = "example-farms"   # unique registry key, used as the config's `source`
    obs_level = "turbine"    # "turbine" or "country"
    countries = ("XX",)      # codes this adapter is resolved for

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

Once registered, the harness picks it up from the region config's `source`
field, with no further changes (see the commands above).

## Built-in adapters

| Name | Level | Countries | Notes |
| --- | --- | --- | --- |
| `european-turbine` | turbine | DK, DE, UK | Reads the per-turbine CSVs under `input/observations/turbine/`. Converts monthly kWh to capacity factor. DK acquisition is fully scripted (`scripts/fetch/dk.py` + `scripts/process/dk.py`, Danish Energy Agency register, see `docs/runbooks/dk.md`). UK is partly scripted (`scripts/fetch/uk.py` + `scripts/process/uk.py`: REPD metadata is open+auto, ROC observations from the open RER export or the CONFIDENTIAL Ofgem certificate warehouse, see `docs/runbooks/uk.md`). DE is staged from CONFIDENTIAL WindStats data via `scripts/process/de.py` (`docs/runbooks/de.md`). |
| `aemo-nem` | turbine | AU-NEM, AU | Per-farm (DUID) monthly CF from 5-minute AEMO SCADA, AEST→UTC binned. The unit is the farm (`obs_unit = "farm"`). |
| `eia-us` | turbine | US, USA | Per-plant monthly CF from EIA-923 net generation, with EIA-860 capacity/coordinates and USWTDB hub heights. The unit is the plant (`obs_unit = "plant"`). |
| `ons-br` | turbine | BR, BRA | Per-complex monthly CF from the ONS `FATOR_CAPACIDADE` hourly series (which carries coordinates + installed capacity itself); optional ONS constrained-off curtailment mask. The unit is the complex (`obs_unit = "complex"`). |
| `emi-nz` | turbine | NZ, NZL | Per-farm monthly CF from EMI `Generation_MD` half-hourly kWh, NZ trading periods (DST-aware) converted to UTC bins; curated farm table (`configs/curation/nz_wind_farms.csv`) supplies coordinates, capacities, and per-farm hub heights; commissioning-ramp months masked. The unit is the farm (`obs_unit = "farm"`). |
| `cen-cl` | turbine | CL, CHL | Per-plant monthly CF from CEN SIP `generacion-real` hourly `gen_real_mw` against `potencia_maxima`; fixed UTC-4 (no DST) converted to UTC bins; wind fleet from the generation stream; coordinates joined from GWPT (`configs/curation/cl_coord_overrides.csv` for the residual); leading commissioning months stripped. The unit is the plant (`obs_unit = "plant"`). |
| `cammesa-ar` | turbine | AR, ARG | Per-plant monthly CF from CAMMESA monthly GWh (native monthly, no time conversion); coordinates AND capacity joined from GWPT (`configs/curation/ar_coord_overrides.csv`), since CAMMESA carries no capacity; a median-CF guard flags bad-capacity matches; leading commissioning months stripped. The unit is the plant (`obs_unit = "plant"`). |
| `windstats` | turbine | ES-WS (SE-WS/FI-WS pending coords) | Per-turbine monthly CF from the CONFIDENTIAL WindStats extract; coordinates joined from open GWPT via the thewindpower name mapping (mixed licence). Built by `scripts/process/windstats.py`, see `docs/runbooks/es.md`. |
| `client-csv-turbine` | turbine | none | Your own fleet from two CSVs (metadata and monthly generation), with remappable column names; constructed explicitly and passed to the driver as `source=`. See [`your-own-data.md`](your-own-data.md). |
| `entsoe-country` | country | none | The nine ENTSO-E regions' national series, read from the `observations/country/` layout that `generate_country_level_training_data` writes; the driver builds one per split. |
| `entsoe-zonal` | country | none | Per-bidding-zone observations over the same layout, one per cluster, so zonal fits are exactly determined. |
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

The ENTSO-E countries do not follow the file-by-file sequence above. Their
fetch and processing live in `vwf.datasets.generate_country_level_training_data`
rather than in `scripts/fetch/` and `scripts/process/`; see
[`data-sources.md`](data-sources.md).

## Testing a new adapter

`tests/test_sources.py` shows the contract pattern. Its tests stub the underlying
loaders rather than reading real files, since `input/` is not tracked by git. The
last test in that file registers a throwaway adapter and drives it all the way to
`gen_cf` through `train_set`, which is the cheapest way to confirm a new source
is wired correctly end to end. For a region, add the processing tests of step 10
as well; `tests/test_emi_nz_processing.py` is the fuller model.
