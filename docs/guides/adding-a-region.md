# Adding a region

PyVWF's bias correction adjusts reanalysis wind speeds against observed
generation. Each region reads its observations through an adapter. The
correction, interpolation, power curves and clustering do not depend on the
data source.

A new turbine-level region touches about fifteen files, in a fixed order. Each
region carries its own acquisition, processing, tests, documentation and
provenance. A new country-level region from ENTSO-E has a shorter path. It
edits up to three tables in `src/pyvwf/datasets/`; see
[A country-level region](#a-country-level-region).

## A turbine-level region, file by file

This section covers a turbine-level region. Its units may be turbines, farms,
plants or complexes. Country-level regions built from ENTSO-E follow a
different path; see
[A country-level region](#a-country-level-region).

The table is in dependency order. New Zealand (`emi-nz`) is the template to
copy. Chile (`cen-cl`) is the cross-check.

| # | File | What it holds |
|---|---|---|
| 1 | [`data-sources.md`](data-sources.md) row | The data source, its access route and its licence key: open, mixed or confidential. Write it before fetching anything. Data from a confidential source stays under the git-ignored `input/`. |
| 2 | `configs/regions/<stem>.toml` | The adapter's registry name (`source`), `obs_level`, `obs_unit`, training years, test year, reanalysis box, `file_tag`, correction model, `cluster_list`, time slices and season months. See [`training.md`](training.md). The reanalysis fetch reads the box and years from this file, so it comes first. |
| 3 | `scripts/fetch/<source>.py` | Downloads raw data into `<input-root>/raw/<source>/`. It honours `PYVWF_INPUT`. The user runs it, with their own credentials where needed. |
| 4 | (no new file) `scripts/fetch/era5.py --region <stem>` | Fetches ERA5 for the config's box and years into `<input-root>/era5/<file_tag>/`. A large box is then reduced with `scripts/era5/combine.py --region <stem>`. |
| 5 | `configs/curation/<stem>_*.csv` | Curated tables: farm coordinates, turbine specifications, capacity stages, months to mask. Each row carries its source in a `source_url` column. Check each source's terms before committing cells transcribed from it. |
| 6 | `src/pyvwf/datasets/<source>.py` | Pure frame-to-frame transforms: time conventions, monthly capacity factor, masks. They do no file I/O, so they are testable without the raw data. |
| 7 | `scripts/process/<source>.py` | The I/O wrapper. It writes `<stem>_md.csv`, `<stem>_obs.csv`, any mask and a `join_report.md` to `input/observations/turbine/<CODE>/`. It also assigns each unit a model key; see [Curve assignment](#curve-assignment). |
| 8 | `src/pyvwf/sources/<source>.py` | The adapter, decorated with `@register`. See [Adding an adapter](#adding-an-adapter). |
| 9 | `src/pyvwf/sources/__init__.py` | One import line, so the registration runs. |
| 10 | `tests/test_<source>_processing.py` | Synthetic tests of the transforms, and of the adapter's resolve and load. NZ's and Chile's are the models. |
| 11 | `tests/test_harness_regions.py` | Raise the shipped-config count in `test_all_shipped_configs_load`. Pin the region's `obs_unit` and `obs_level` in `test_shipped_granularity_classification`. |
| 12 | The [built-in adapters](#built-in-adapters) table | One row. |
| 13 | `docs/runbooks/<stem>.md` | Acquisition and processing steps, the licence, the capacity-factor denominator and its source, and how to refresh. |
| 14 | `docs/findings/region-<stem>.md` | The region's findings document, written after the runs. [`docs/README.md`](../README.md) sets its name and shape. |
| 15 | `docs/scorecard.md` row, and its scorecard config | The scorecard row, and an unchanged copy of the configuration its run was made from, `configs/regions/scorecard/<stem>_k<N>.toml`. |

Between steps 13 and 14, run the region through the harness:

```bash
PYVWF_INPUT=<input root> pyvwf-validate train \
    --region configs/regions/<stem>.toml
PYVWF_INPUT=<input root> pyvwf-validate evaluate \
    --region configs/regions/<stem>.toml --train-run output/validation/<CODE>/train-<stamp>
```

Set `PYVWF_INPUT` on both commands. Each step resolves the curve library again,
and the manifest records which one it used.

Then read these outputs; [`output-structure.md`](output-structure.md) describes
each one:

- `metrics.csv`: the skill table, with the uncorrected row first.
- The `fit_quality` columns: `max_scalar`, `n_implausible_scalar` and
  `n_failed_offset`.
- The substituted share (`substituted_capacity_share`), and each run's
  `curve_resolution.csv`. Together they show which power curve every unit was
  simulated on.
- The curve-match audit. Add the region to `RUNS` and `own_manufacturer` in
  `scripts/analysis/curve_match_audit.py`. Then run the script.

## Curve assignment

Each unit's model key selects its power curve. Three routes assign model keys.
Two match on specific power, not on the turbine's make. The third gives every
unit the same curve.

| Route | Where | Used by |
|---|---|---|
| `pyvwf.datasets.eia_us.assign_curves_from_library` | processing time | US, NZ, and CL and AR through `scripts/region_tools/apply_turbine_specs.py` |
| `pyvwf.curves.add_models` | load time, inside the adapter; processing time through `scripts/region_tools/assign_au_curves.py` | DE, DK, UK (`european-turbine`), `client-csv-turbine`, and AU-NEM |
| One default curve for every unit | processing time | BR |

The first route keeps onshore models rated 0.5 to 2 times the unit's
per-turbine rating. It then takes the nearest specific power. NZ imports it
from the US module. It records how each unit was matched in `model_source`. A
unit it cannot match gets the default curve.

The second route first matches a fuzzily named manufacturer, then the nearest
specific power. It records the tier it used in `model_match`.

For every route:

- **Record each unit's own manufacturer or model string.** Keep it in the
  metadata, as NZ's `true_model` and US's `uswtdb_model` do. A curated table
  keyed by `ID` also works. Without it, the curve-match audit reports the
  region as unverifiable, as it does for BR.
- **A model key missing from `power_curves.csv` does not stop a
  turbine-level run.** The unit is simulated on the fallback curve, the first
  column of `power_curves.csv`. `curve_resolution.csv` records the unit as
  substituted. A country-level run refuses it instead.
- **Read the substituted share (`substituted_capacity_share`) before any
  result.** A share above zero means
  some model keys were missing. In that case, list the missing keys before
  you use the result.

## A country-level region

A country-level region fits against one national or zonal series per period.
Its grid points stand in for the fleet. The nine ENTSO-E regions follow this
path. Their inputs are built by one module, not by `scripts/fetch/` and
`scripts/process/`.

| # | File or command | What it does |
|---|---|---|
| 1 | [`data-sources.md`](data-sources.md) row | The data source, its access route and its licence. ENTSO-E is open. |
| 2 | Three tables in `src/pyvwf/datasets/` | For a country outside the nine: `country_grid.COUNTRY_CONFIGS` (outline box, grid resolution, representative turbine, cluster count), `gwpt.COUNTRY_NAME` (its GWPT country name, for step 5) and, if absent, `fetch_entsoe_capacity_factors.COUNTRY_CODES` (its ENTSO-E area code). |
| 3 | `configs/regions/<stem>.toml` | `source = "entsoe-country"`, `obs_level = "country"`, `obs_unit = "country"`, the years, the reanalysis box and the correction settings. |
| 4 | `python -m pyvwf.datasets.generate_country_level_training_data --countries <CODE>` | Writes the grid points and the ENTSO-E observations under `<input-root>/observations/country/`. The fetch needs the `data` extra. Pass `ENTSOE_API_KEY` in the environment for this one command. |
| 5 | `scripts/region_tools/weight_country_grid_points.py` | Replaces the uniform synthetic capacity with GWPT capacity. Add `--per-year <start> <end>`, an inclusive pair, so each year sees its own fleet. Add `--zone-aware` beside it for a region with bidding zones; it needs `--per-year`. |
| 6 | `scripts/analysis/audit_country_observations.py` | Checks the observed series. Run it before you trust any fit. |
| 7 | `tests/test_harness_regions.py` | Raise the shipped-config count. Pin the region's `obs_level`. |
| 8 | `docs/scorecard.md` row, and its scorecard config | As for a turbine-level region: `configs/regions/scorecard/<stem>_country.toml`. |

Set `cluster_list` to `1`, or to the grid's own cluster count. Any other value
raises. The value `1` gives one national cluster, an exactly determined fit.

The audit in step 6 matters because the affine correction absorbs a constant
observation error into the scalar. A uniformly wrong series therefore still
fits well in sample. NL fails this audit and is excluded. IE failed it once,
on a frozen capacity denominator, and was repaired; it is now a scorecard row.
Both are on the [results page](../results.md#the-country-level-estimator).

[`data-sources.md`](data-sources.md) describes the file layout the generator
writes.

## Adding an adapter

An adapter is an `ObservationSource` subclass. It loads one data source for
the harness, and the registry finds it by name. The correction,
interpolation, power curves and clustering do not depend on the data source.
So a new adapter changes no core module.


### What an adapter must provide

Subclass `pyvwf.sources.ObservationSource`. Implement the two methods below.

#### `load_metadata() -> pd.DataFrame`

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
manufacturer or model string; see [Curve assignment](#curve-assignment).

A country-level adapter must also supply `cluster`, each grid point's cluster.
Country-level corrections are fitted per cluster. No clustering step runs on
that path.

#### `load_observations(year_start=None, year_end=None) -> pd.DataFrame`

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

### Registering it

Three class attributes drive the registry:

```python
from pyvwf.sources import ObservationSource, register

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
`src/pyvwf/sources/__init__.py`, beside the built-in adapters, is enough.

The registry resolves an adapter listed in `countries` from a region code. So
the adapter must take that code as its only constructor argument. Leave
`countries` empty for an adapter the caller always builds itself, as
`InMemoryCountrySource` does.

The harness then finds the adapter from the region config's `source` field.
Nothing else changes; the commands after the file-by-file table above run it.

### Built-in adapters

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
| `client-csv-turbine` | turbine | none | Your own fleet, from two CSVs: metadata and monthly generation. Column names can be remapped. Build it yourself and pass it to the driver as `source=`. See [your own fleet data](training.md#your-own-fleet-data). |
| `entsoe-country` | country | none | The nine ENTSO-E regions' national series, from the `observations/country/` layout that `generate_country_level_training_data` writes. The driver builds one per split. |
| `entsoe-zonal` | country | none | Per-bidding-zone observations over the same layout, one per cluster. Zonal fits are therefore exactly determined. |
| `in-memory-country` | country | none | Wraps grid points and an observed capacity-factor series that the caller supplies. Pass one per split to the driver as `source=`. |

### Country-level data supplied by the caller

Country-level observations, such as ENTSO-E series, are fetched and cached
outside the library. Wrap the grid points and each split's observations in an
`InMemoryCountrySource`. Pass it to the driver as `source=`:

```python
from pyvwf.harness.driver import run_evaluate, run_train
from pyvwf.sources import InMemoryCountrySource

train_dir = run_train(spec, out, source=InMemoryCountrySource(grid_points, obs_train))
run_evaluate(spec, train_dir, out, source=InMemoryCountrySource(grid_points, obs_test))
```

You might ask for `obs_level="country"` without a source and without an
adapter for the region. Resolution then raises `NotImplementedError`, which
explains both options. Nothing falls back silently.

*[Changed 2026-10-16: this section described `PyVWF.load_country_data()`,
which was removed with the legacy path on 2026-09-24.]*

The ENTSO-E regions use their own path. See
[A country-level region](#a-country-level-region).

### Testing a new adapter

`tests/test_sources.py` shows the contract pattern. Its tests stub the loaders
instead of reading real files, because `input/` is not tracked by git. The last
test registers a throwaway adapter and drives it through `train_set` to
`gen_cf`. That is the cheapest end-to-end check that a new adapter is wired
correctly.

For a region, also add the processing tests of step 10 above. `tests/test_emi_nz_processing.py`
is the fuller model.
