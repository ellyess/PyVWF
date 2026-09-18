# Adding a region

PyVWF's bias correction adjusts reanalysis wind speeds against observed
generation. Each region reads its observations through an adapter. The
correction, interpolation, power curves and clustering do not depend on the
data source.

A new turbine-level region touches about fifteen files, in a fixed order. Each
region carries its own acquisition, processing, tests, documentation and
provenance. A new country-level region from ENTSO-E has a shorter path. It
edits up to three tables in `src/vwf/datasets/`; see
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
| 6 | `src/vwf/datasets/<source>.py` | Pure frame-to-frame transforms: time conventions, monthly capacity factor, masks. They do no file I/O, so they are testable without the raw data. |
| 7 | `scripts/process/<source>.py` | The I/O wrapper. It writes `<stem>_md.csv`, `<stem>_obs.csv`, any mask and a `join_report.md` to `input/observations/turbine/<CODE>/`. It also assigns each unit a model key; see [Curve assignment](#curve-assignment). |
| 8 | `src/vwf/sources/<source>.py` | The adapter, decorated with `@register`. See [`adding-an-adapter.md`](adding-an-adapter.md). |
| 9 | `src/vwf/sources/__init__.py` | One import line, so the registration runs. |
| 10 | `tests/test_<source>_processing.py` | Synthetic tests of the transforms, and of the adapter's resolve and load. NZ's and Chile's are the models. |
| 11 | `tests/test_harness_regions.py` | Raise the shipped-config count in `test_all_shipped_configs_load`. Pin the region's `obs_unit` and `obs_level` in `test_shipped_granularity_classification`. |
| 12 | The [built-in adapters](adding-an-adapter.md#built-in-adapters) table | One row. |
| 13 | `docs/runbooks/<stem>.md` | Acquisition and processing steps, the licence, the capacity-factor denominator and its source, and how to refresh. |
| 14 | `docs/findings/region-<stem>.md` | The region's findings document, written after the runs. [`docs/README.md`](../README.md) sets its name and shape. |
| 15 | `docs/findings/scorecard.md` row, and its scorecard config | The scorecard row, and a byte-identical copy of the configuration behind it, `configs/regions/scorecard/<stem>_k<N>.toml`. |

Between steps 13 and 14, run the region through the harness:

```bash
PYVWF_INPUT=<input root> python scripts/analysis/validate_region.py train \
    --region configs/regions/<stem>.toml
PYVWF_INPUT=<input root> python scripts/analysis/validate_region.py evaluate \
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
| `vwf.datasets.eia_us.assign_curves_from_library` | processing time | US, NZ, and CL and AR through `scripts/region_tools/apply_turbine_specs.py` |
| `vwf.data.add_models` | load time, inside the adapter; processing time through `scripts/region_tools/assign_au_curves.py` | DE, DK, UK (`european-turbine`), `client-csv-turbine`, and AU-NEM |
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
- **A model key missing from `power_curves.csv` does not stop a run.** The
  unit is simulated on the fallback curve, the first column of
  `power_curves.csv`. `curve_resolution.csv` records the unit as substituted.
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
| 2 | Three tables in `src/vwf/datasets/` | For a country outside the nine: `country_grid.COUNTRY_CONFIGS` (outline box, grid resolution, representative turbine, cluster count), `gwpt.COUNTRY_NAME` (its GWPT country name, for step 5) and, if absent, `fetch_entsoe_capacity_factors.COUNTRY_CODES` (its ENTSO-E area code). |
| 3 | `configs/regions/<stem>.toml` | `source = "entsoe-country"`, `obs_level = "country"`, `obs_unit = "country"`, the years, the reanalysis box and the correction settings. |
| 4 | `python -m vwf.datasets.generate_country_level_training_data --countries <CODE>` | Writes the grid points and the ENTSO-E observations under `<input-root>/observations/country/`. The fetch needs the `data` extra. Pass `ENTSOE_API_KEY` in the environment for this one command. |
| 5 | `scripts/region_tools/weight_country_grid_points.py` | Replaces the uniform synthetic capacity with GWPT capacity. Add `--per-year` so each year sees its own fleet. Add `--zone-aware` for a region with bidding zones. |
| 6 | `scripts/analysis/audit_country_observations.py` | Checks the observed series. Run it before you trust any fit. |
| 7 | `tests/test_harness_regions.py` | Raise the shipped-config count. Pin the region's `obs_level`. |
| 8 | `docs/findings/scorecard.md` row, and its scorecard config | As for a turbine-level region: `configs/regions/scorecard/<stem>_country.toml`. |

Set `cluster_list` to `1`, or to the grid's own cluster count. Any other value
raises. The value `1` gives one national cluster, an exactly determined fit.

The audit in step 6 matters because the affine correction absorbs a constant
observation error into the scalar. A uniformly wrong series therefore still
fits well in sample. NL and IE fail this audit; see
[`method-country-level.md`](../findings/method-country-level.md).

[`data-sources.md`](data-sources.md) describes the file layout the generator
writes.

