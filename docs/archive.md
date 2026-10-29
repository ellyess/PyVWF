# The research archive

PyVWF's main branch shows the package and how it is built. The research that
shaped it is a separate record, preserved whole at tag
[`archive/research-2026-10-27`](https://github.com/ellyess/PyVWF/tree/archive/research-2026-10-27)
(commit `e378015`) and on Zenodo as
[10.5281/zenodo.23696129](https://doi.org/10.5281/zenodo.23696129), deposited
by Ellyess F. Benmoufok as its sole author. At the tag it holds 48 findings
documents (18 of them preregistrations), 22 study directories with their
configurations and tests, and the raw 2026-10-16 run of the review-agent
evaluation. Every result in it, negative ones included, has an entry on the
{doc}`results page <results>`, with its training years, its test year and a
link to the full record.

To cite the archive: Benmoufok, E. F. (2026). *PyVWF research archive
(2026-10-27)* (archive/research-2026-10-27). Zenodo.
https://doi.org/10.5281/zenodo.23696129. The record has one author and no
contributors; the PyVWF software it supplements is cited as `CITATION.cff`
says, with its own authors.

## What left the main branch, and where it is

| What | Path at the tag |
|---|---|
| Findings and preregistrations | `docs/findings/`. Main keeps only an active study's documents, `agent-critics-prereg.md` and its baseline extension, and a copy of `method-wake-unit-prereg.md` as the preregistration test's fixture, `tests/data/prereg/complete.md` |
| The full scorecard, with every correction notice | `docs/findings/scorecard.md`. Main has the short form, {doc}`scorecard` |
| Study drivers | `scripts/studies/<study>/`, mapped in `scripts/studies/README.md`: 21 of the 22 directories. The scorecard audits moved to `scripts/analysis/` |
| Finished analysis scripts | `scripts/analysis/{eu_rerun_compare,ml_transfer_retest,ml_transfer_expanded,regression_compare,check_issue155}.py`, `scripts/dev/repo_stats.py` |
| Study and superseded configurations | `configs/regions/study/`, `configs/regions/scorecard/superseded/`, `configs/regions/pinn_loco/`, `configs/regions/nl.toml` |
| The builder of the superseded `era5/EU` archive, removed on 2026-10-28 | `src/pyvwf/datasets/combine_era5_files.py` and `tests/test_combine_era5_files.py`. `docs/guides/data-sources.md` gives its command |
| Review-agent evaluation run | `scripts/dev/agent_evals/runs/2026-10-16/`: 30 critic prompts and reviews, 40 judge prompts and judgements, and the run record. The harness, its five cases and a results summary stay on main |
| Turkey evaluation, not shipped | `docs/runbooks/tr.md`, `scripts/fetch/epias_tr.py`. `docs/guides/data-sources.md`, section 5, keeps what was found |
| The gridded correction and the transfer model, removed from the package on 2026-10-28 | `src/pyvwf/extensions/` (`grid`, `ml`), `src/pyvwf/geospatial.py`, and `calculate_error`, `prepare_monthly_data` and `weighted_average_vectorized` in `src/pyvwf/metrics.py`, with their tests and the `grid` extra. The sections below say how to run them |
| The TouchDesigner export, removed on 2026-10-29 and now kept locally | `scripts/analysis/export_voronoi_frames.py` and the `touchdesigner` extra in `pyproject.toml`. `docs/guides/visualisation.md` at the tag says how to run it |
| Agent definitions now kept locally | `.claude/agents/prior-work-scout.md`, `.claude/agents/simplicity-critic.md`, `.claude/skills/idea-review/`. The three evaluated critics stay on main |
| Their tests | `tests/`, as named in each finding's reproduction record |

## Reproducing a result from the tag

```bash
git clone https://github.com/ellyess/PyVWF.git && cd PyVWF
git checkout archive/research-2026-10-27
pip install -e ".[dev,grid,pinn,data,parallel]" -c constraints/py312-extras.txt
```

Each findings document opens with a reproduction record naming its drivers and
the commit they ran at. The exact command line of every driver is in
`tests/test_script_command_lines.py` at the tag. Runs go through
`scripts/dev/run_locked.py`, from the repository root, into `output/`. The
inputs are not in the archive: `docs/guides/data-sources.md` says where each
comes from and whether it can be shared, and turbine-level European rows need
`PYVWF_INPUT=input/combined` with a licensed curve library.

The record configurations under `configs/regions/scorecard/superseded/` and
`configs/regions/study/dk_k100_stored.toml` name `roughness = "stored"`
explicitly (commit `8105277`). Without that key they would load as `derived`,
the default since 2026-09-20.

### Grid interpolation (thesis chapter 4)

`pyvwf.extensions.grid` needs the `grid` extra. The synthetic tests run
without data:

```bash
pytest -m "not realdata" tests/test_grid_evaluate.py tests/test_grid_geodataframes.py \
    tests/test_grid_interpolation.py tests/test_grid_surface.py tests/test_geospatial.py tests/test_metrics.py
```

The studies (distance mask, domain split, leave-one-country-out interpolation,
offshore pool, pool as training set, chapter control-point refit) read the
control-point pool `output/pyvwf_to_grid/all_corrections_centroids.csv`. No
script at the tag produces it: it predates the repository's run records. It is
deposited with the Zenodo record
([10.5281/zenodo.23696129](https://doi.org/10.5281/zenodo.23696129), sole
author Ellyess F. Benmoufok) as
`pyvwf-research-archive-2026-10-27-control-points.csv`; check it against the
record's `.sha256` file and copy it to that path before running a driver.

### Machine-learning transfer

`scripts/analysis/ml_transfer_retest.py` (round one) and
`ml_transfer_expanded.py` (round two) read the training runs
`output/validation/{DK,DE,UK}/train-sweep3` and `{US,BR}/train-sweep2`, made at
commit `8a032d6`, plus `input/reference/terrain/etopo_global.nc` (NOAA ETOPO).
The runs are not deposited, because they carry turbine metadata from
confidential sources. To regenerate them, check out `8a032d6` (2026-07-22, when
the package was still `vwf`) and train each region with that commit's
`scripts/validate_region.py`. Then return to the tag and run the retest.
`tests/test_pin_ml_transfer.py` pins the outputs: its synthetic cases run
without data and agree under scikit-learn 1.7.2 and 1.9.1, while the
real-data scores move in the third decimal between those two versions, so its
real-data layer carries one digest per version.

### Review-agent evaluation

The harness and case definitions stay on main, so the evaluation reruns there
(`scripts/dev/agent_evals/evals.py`). The 2026-10-16 run's reviews and
judgements are at the tag; rescoring that run reads `run.json`, `responses/`
and `judgements/` from the run folder there.
