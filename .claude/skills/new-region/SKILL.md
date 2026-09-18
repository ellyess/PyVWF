---
name: new-region
description: Add a turbine-level validation region to PyVWF (units may be turbines, farms, plants or complexes) as a gated pipeline from licence check to scorecard row, stopping for human sign-off at every phase. Use when asked to add a region, a country, or a new data source for validation.
argument-hint: "<region-stem> <source-module>"
disable-model-invocation: true
---

# new-region

Add a region. Arguments:

- `$0`: the region stem, lower-case, used in file names (`nz`). The config's
  `code` is its upper-case form, the region code (`NZ`).
- `$1`: the adapter's module name in snake case (`emi_nz`). Its registry name
  is the hyphenated form (`emi-nz`).

Work through the phases in order. Each phase ends with a check you run
yourself. Report the result. Then **stop for sign-off before the next phase**.
Never skip a stop because a check passed.

Terms follow `docs/CONTEXT.md`. Use only its approved terms.

## Scope

- A turbine-level region with a new adapter. Its units may be turbines, farms,
  plants or complexes.
- `obs_level` is the pipeline branch, not the unit. A region with plant units
  still sets `obs_level = "turbine"`, and records the unit in `obs_unit`. The
  shipped configs do this, and `tests/test_harness_regions.py` pins it.
- **Out of scope:** country-level regions. They are built by
  `vwf.datasets.generate_country_level_training_data`, not by this sequence. If
  `$0` is one, stop and say so.

## Before starting

- Read `docs/guides/adding-an-observation-source.md`. Its table lists the files
  this region touches, in order. Do not keep a separate list.
- The template is New Zealand: `scripts/fetch/emi_nz.py`,
  `scripts/process/emi_nz.py`, `src/vwf/datasets/emi_nz.py`,
  `src/vwf/sources/emi_nz.py`, `configs/regions/nz.toml`,
  `tests/test_emi_nz_processing.py` and `docs/runbooks/nz.md`. Chile (`cen-cl`)
  is the cross-check.
- NZ's processing step imports `assign_curves_from_library` from
  `vwf.datasets.eia_us`. Reuse it the same way. Do not move or refactor it.
- Work on a branch. Never push, merge, tag or release; those are the
  maintainer's. Show the diff and wait for approval before every commit.

## Standing rules

- **Commit nothing from `input/`.** Raw data and processed inputs stay under the
  git-ignored input root.
- **Give every curated table a per-row source column** (`source_url`, or
  `source` for a non-web source). Leave no cell blank.
- **Check each source's terms** before committing cells transcribed from it.
  Keep anything licensed or confidential local.
- **Run under `output/`, from the repository root, on a clean tree.** Never run
  from a temporary or session directory. Do not edit tracked files while a run
  writes its manifest.

## Phases

### 0. Licence

- Write the data source's row in `docs/guides/data-sources.md`, with its access
  route and licence key: open, mixed or confidential.
- Check: the row exists. Nothing in the code checks licences.
- Stop: the human confirms the licence, and whether the data can be
  redistributed. This gate is entirely theirs.

### 1. Config

- Write `configs/regions/$0.toml`, following `docs/guides/training.md`. Give
  explicit month lists for the seasons.
- In `tests/test_harness_regions.py`, raise the count in
  `test_all_shipped_configs_load`. Pin `obs_unit` and `obs_level` in
  `test_shipped_granularity_classification`.
- Check: `vwf.harness.load_region("configs/regions/$0.toml")` succeeds.
  `tests/test_harness_regions.py` passes.
- Stop: the human reviews the seasons, years, box and `cluster_list`. The
  largest usable cluster count is the number of units that reach the clusterer
  in the training years. See `docs/guides/training.md` and the k=10 crash in
  `docs/findings/region-nz.md`.

### 2. Fetch

- Write `scripts/fetch/$1.py`. It writes to `<input-root>/raw/<source>/` and
  honours `PYVWF_INPUT`.
- Fetch reanalysis with `scripts/fetch/era5.py --region $0`. For a large box,
  then run `scripts/era5/combine.py --region $0`.
- Check: raw data exists for every month of the training years and the test
  year.
- Stop: the user runs any fetch that needs credentials. Never ask for a key.
  Never write a key into a file.

### 3. Process

- Write the curated tables, `configs/curation/$0_*.csv`, each with its per-row
  source column.
- Write `src/vwf/datasets/$1.py`: pure frame-to-frame transforms, no file I/O.
- Write `scripts/process/$1.py`. It writes `$0_md.csv`, `$0_obs.csv` and a
  `join_report.md`.
- Choose the curve-assignment route from the guide's "Curve assignment"
  section. Record how each unit was matched, in `model_source`.
- Keep each unit's own manufacturer or model string in the metadata. Without
  it, the curve-match audit can only report the region as unverifiable.
- **State the capacity-factor denominator** in a "Capacity-factor denominator"
  section of `docs/runbooks/$0.md`:
  - what it is, for example a nameplate or a staged capacity history;
  - which data source or curated table it comes from;
  - for a curated denominator, a per-unit `confidence` column in that table.
- Check:
  - `tests/test_$1_processing.py` passes;
  - the metadata carries the adapter's required columns;
  - every curated table has a source column with no blank cells;
  - the runbook's denominator section exists and names its source;
  - `join_report.md` is shown to the human.
- Stop: the human reviews the join report, the curated tables and the
  denominator section.

### 4. Adapter

- Write `src/vwf/sources/$1.py` with `@register`, and its import line in
  `src/vwf/sources/__init__.py`.
- Write the adapter tests, and a row in the guide's built-in adapter table.
- Check: `vwf.sources.get_source("<registry-name>", "<CODE>")` resolves. The
  tests pass. `ruff check src/vwf tests` passes.
- Stop: code review.

### 5. Runs

- **Before running,** write down which configuration will be reported, and
  why. Fix that rule before seeing any result.
- Run `scripts/analysis/validate_region.py` train, then evaluate, as in the
  guide. Set `PYVWF_INPUT` on both commands.
- Add the region to `RUNS` and `own_manufacturer` in
  `scripts/analysis/curve_match_audit.py`. Run the audit.
- Check:
  - both manifests record `git_dirty: false`;
  - `metrics.csv` carries its fit quality; list any degenerate fit;
  - report the substituted share (`substituted_capacity_share`) and the
    `open` and `external` shares from `curve_resolution.csv`;
  - report the curve-match audit's classes for the region.
- If the substituted share is not zero, stop. List the model keys missing from
  `power_curves.csv`, the fallback curve they used, and their share of
  capacity. Do not change the region to make the share zero.
- Stop: show the full metrics table (uncorrected row first), the fit quality,
  and the curve shares, before any prose.

### 6. Write-up

- Write the region's findings document and its scorecard row with the
  `findings-doc` skill. It owns their shape, the scorecard columns, the
  scorecard config and the CHANGELOG convention.
- Complete `docs/runbooks/$0.md`: acquisition, processing, licence, and how to
  refresh. Write it under the procedural rules in `docs/README.md`.
- Check: the `findings-doc` checks pass. The runbook keeps its
  capacity-factor denominator section.
- Stop: final review. Then propose the commits.

## Done when

- The config loads, and the regions test passes with the raised count.
- The adapter resolves, and the processing and adapter tests pass.
- The capacity-factor denominator is stated in the runbook, with its source.
- A clean-tree train and evaluate run exist under `output/`, with curve
  resolution recorded and the audit run.
- The human saw the full metrics table before any prose was written.
- The `findings-doc` checks pass for the findings document and scorecard row.
