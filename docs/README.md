# PyVWF Documentation

Start here. This folder holds reference documentation for the core PyVWF
bias-correction workflow. If you are new to the project, read in the order
below.

## Start here

1. [Project README](../README.md) - what PyVWF is, how it works, installation,
   and a Denmark quickstart. Read this first.
2. [Input data](../README.md#input-data) - what data you need, where to get it,
   and the expected `input/` directory layout.

## Reading order

Once you can run the quickstart, work through these in order:

| # | Document | What it covers |
|---|----------|----------------|
| 1 | [Data requirements](DATA_REQUIREMENTS.md) | The input files PyVWF expects and how to obtain the ERA5 winds. |
| 2 | [Data pipeline](DATA_PIPELINE.md) | How turbine-level and country-level data flow through the model. |
| 3 | [Training guide](TRAINING_GUIDE.md) | Configuring and running bias-correction training across countries and resolutions. |
| 4 | [Output structure](OUTPUT_STRUCTURE.md) | The layout of a run directory and the files each run produces. |
| 5 | [ENTSO-E API guide](ENTSOE_API_GUIDE.md) | Fetching national generation data for the country-level workflow. |

For the end-to-end script execution order, see
[PIPELINE.md](../PIPELINE.md) at the repository root.

## Reference by task

- **I want to run PyVWF on turbine data.** Read the [README quickstart](../README.md#quickstart),
  then [Data requirements](DATA_REQUIREMENTS.md) and [Training guide](TRAINING_GUIDE.md).
- **I want national (country-level) corrections.** Read [Data pipeline](DATA_PIPELINE.md)
  and [ENTSO-E API guide](ENTSOE_API_GUIDE.md), then [Training guide](TRAINING_GUIDE.md).
- **I want to understand a run's outputs.** Read [Output structure](OUTPUT_STRUCTURE.md).
