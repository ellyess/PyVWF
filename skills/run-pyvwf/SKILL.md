---
name: run-pyvwf
description: Help a user install PyVWF, run its synthetic example, run a shipped region or their own fleet through the validation harness, and read what the run wrote. Use when someone wants to use PyVWF to correct ERA5 wind or simulate wind capacity factors. Do not use for changing PyVWF's own code, adding a region to the repository (use new-region), or writing up a result (use findings-doc).
---

# run-pyvwf

You are helping someone use PyVWF, not develop it. Work from a checkout's
root. Run what you can, show the output, and explain it in the user's terms.

## 1. Install

```bash
git clone https://github.com/ellyess/PyVWF.git && cd PyVWF
pip install -e .
```

Python 3.10 or newer. Add an extra only when the task needs it:
`docs/guides/installation.md` says what each is for. `conda env create -f
environment.yaml` is the alternative.

## 2. The synthetic example

```bash
python examples/run_minimal.py
```

No downloads, a few seconds. It fits the correction on bundled synthetic
weather and observations and prints a per-cluster table, then the mean absolute
capacity-factor error before and after correction. The data is synthetic, so
the numbers show the workflow, not PyVWF's skill. Real results are in
`docs/scorecard.md`.

## 3. A shipped region

Each region is one config in `configs/regions/`. Its data is not in the
repository: the region's runbook in `docs/runbooks/` says how to fetch and
process it (the country-level regions share `entsoe.md`), and
`docs/guides/data-sources.md` gives each source's licence. Confirm the user
has the inputs before running anything.

```bash
pyvwf-validate train --region configs/regions/<code>.toml
pyvwf-validate evaluate --region configs/regions/<code>.toml \
    --train-run output/validation/<CODE>/train-<timestamp>
```

**Choose the input root first.** Most regions need
`PYVWF_INPUT=input/combined` on both commands, a root holding a licensed curve
library the user must supply themselves. CL, AR and ES-WS run on the default
`input/`; the guide's table and each runbook say which root a region uses.
Without the licensed library a country-level run stops, and a turbine-level
run simulates on a fallback curve and records that it did. See
`docs/guides/training.md#choose-the-input-root`.

A maintained config does not always reproduce a scorecard row. To reproduce
one, use its file in `configs/regions/scorecard/`.

## 4. Their own fleet

Two CSV files, fleet metadata and monthly generation, through
`ClientCsvTurbineSource`. No adapter code. Follow the worked example in
`docs/guides/training.md#your-own-fleet-data`. Check three things with the
user before running:

- the metadata columns map to `ID`, `lon`, `lat`, `capacity`, `height`, plus
  either `model` or `diameter`. `column_map` renames metadata columns only;
- the generation file already has `ID`, `year`, `month` and a value column
  (`generation_mwh` unless `generation_column` names another);
- the capacity unit (kW or MW), and whether the value column is energy, on the
  same basis as the capacity (MWh against MW), or a capacity factor;
- their fleet lies inside the loaded ERA5 extent, the files' extent after the
  config's `bbox` slice. If it does not, the run stops with
  `ExtrapolationError`. Fetch the box with `scripts/fetch/era5.py` and point
  the config's `[era5]` `path` and `bbox` at it, rather than opting in to
  extrapolation.

Keep private data under the input root, which git ignores.

## 5. Read the run

A shipped region's outputs land under `output/validation/<CODE>/`, one
directory per run, and an own-fleet run writes where its script says (the
worked example uses `output/my_run`);
`docs/guides/output-structure.md` documents every file. Read `metrics.csv` in
this order, and tell the user what each step shows:

1. **The uncorrected row.** The bias the correction starts from.
2. **The corrected rows' `rmse` and `mbe`**, against the uncorrected row.
3. **Fit quality:** `n_implausible_scalar`, `n_failed_offset` and
   `degenerate_clusters`. A row can improve while the fit behind it is
   degenerate. Report such a row with that caveat, and do not reuse its
   per-cluster factors.
4. **The shares:** `substituted_capacity_share` (units on a fallback curve),
   `extrapolated_capacity_share` and `excluded_share`. A non-zero share
   changes what the score means. Say so.

`run_manifest.json` records the package version, git state, config and curve
library behind the numbers.

## What to tell the user, every time

- The score rests on one held-out test year. Name the training years and the
  test year.
- PyVWF gives screening-level estimates, not an accredited yield assessment.
- Ask before downloading data, and never put a credential in a file: pass it
  in the environment for one command.
