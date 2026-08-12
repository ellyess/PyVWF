# PyVWF documentation

Reference documentation for the PyVWF bias-correction workflow. New to the
project? Read the [project README](../README.md) first (what PyVWF is, install,
Denmark quickstart), then come here.

## Guides

| Guide | Covers |
|---|---|
| [Data sources and preprocessing](guides/DATA_SOURCES.md) | Every data source per region, its licence, and how it is fetched and preprocessed. The data reference. |
| [Training and evaluation](guides/TRAINING_GUIDE.md) | The region config and running `validate_region.py` (train / evaluate / transfer). |
| [Output structure](guides/OUTPUT_STRUCTURE.md) | The files a run produces and how to read them. |
| [Adding an observation source](guides/ADDING_AN_OBSERVATION_SOURCE.md) | The adapter contract for a new region. |
| [Using your own data](guides/USING_YOUR_OWN_DATA.md) | Run a correction on your own fleet from two CSVs, without writing an adapter. |

## Region runbooks

Step-by-step acquisition and processing for each validated region, one file per
country, in [`runbooks/`](runbooks): AR, BR, CL, DE, DK, ES, NZ, TR, UK, US. The
nine ENTSO-E country-level regions are covered by the country-level section of
[Data sources](guides/DATA_SOURCES.md#4-country-level-entso-e-workflow).

## Design and findings

- [`design/HARNESS_DESIGN.md`](design/HARNESS_DESIGN.md): the validation-harness design.
- [`findings/`](findings): dated research records (regression checks, method
  reviews, per-region first runs).

## By task

- **Run a region.** [Training guide](guides/TRAINING_GUIDE.md), with its data
  from [Data sources](guides/DATA_SOURCES.md) and the region's runbook.
- **Country-level (national) corrections.** The country-level section of
  [Data sources](guides/DATA_SOURCES.md#4-country-level-entso-e-workflow).
- **Understand a run's outputs.** [Output structure](guides/OUTPUT_STRUCTURE.md).
- **Add a region.** [Adding an observation source](guides/ADDING_AN_OBSERVATION_SOURCE.md).
