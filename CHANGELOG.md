# Changelog

All notable changes to PyVWF are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
PyVWF adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, the public API may change in a minor release.

The version is defined once, in `pyvwf/_version.py`, and re-exported as
`pyvwf.__version__`; `pyproject.toml` reads it from there, and
`tests/test_packaging.py` asserts that `CITATION.cff` and the newest release in
this file stay in step with it.

Released versions are summarised here. The full notes of every release up to
0.6.0 are at tag
[`archive/research-2026-10-27`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/CHANGELOG.md),
and each summary from 0.2.0 on links to its own.

## [Unreleased]

The repository is simplified to the package and how it is built. The research
record, every finished study's scripts, configs and findings among it, is at
tag `archive/research-2026-10-27`, and `docs/archive.md` says what left, where
it is and how to reproduce from it. Several country-level rows move, because
their observations are rebuilt.

### Breaking

- **`prep_era5`, `train_set` and `val_set` take `era5_dir` as a required
  keyword**, and `bbox=None` reads the files' whole extent.
  `PyVWFPaths.ERA5_DATA`, `BoundingBoxes`, `get_country_shapes` and
  `get_turbine_metadata` are removed. Harness runs are unchanged, since a
  region config names its ERA5 `path` and `bbox`.
- **`pyvwf.extensions`, `pyvwf.geospatial` and the `grid` extra are removed**,
  with `calculate_error`, `prepare_monthly_data` and
  `weighted_average_vectorized` from `pyvwf.metrics`. The gridded correction
  and the machine-learning transfer served studies whose results are void,
  negative or never run; `pyvwf.harness.export` still writes a gridded
  correction field. `rasterio` moves to the `pinn` extra.
- **`pyvwf.datasets.combine_era5_files`**, the builder of the old `era5/EU`
  archive, is removed; no configuration on main reads that archive, and the
  data-sources guide gives the builder's command at the tag.
- **`pyvwf.viz.style`'s `apply_spatial_resolution_axis`,
  `add_resolution_markers` and `format_axes_standard`, and
  `pyvwf.viz.palettes.EXISTING_NEW_COLOURS`**, are removed. Nothing in the
  package, scripts, examples, tests or docs used them.
- **The `touchdesigner` extra and `scripts/analysis/export_voronoi_frames.py`**
  are removed. Nothing in the package used either; they are at the tag.

### Added

- **A `run-pyvwf` skill for users**, which takes a coding agent through
  installing PyVWF, the example, a region or the user's own fleet, and
  reading the output.
- **`prep_era5(years=(start, end))`**, which returns those years identical to
  the whole record cut the same way.
- **An evaluation of the review agents**, pre-registered
  (`docs/findings/agent-critics-prereg.md`), with a no-agent baseline arm and
  a docs suite for `docs-critic`, each pre-registered too. Its first run did
  not pass its bar; its record is at the tag, and run folders are no longer
  committed. **The baseline and docs runs found no measurable difference**
  between the critic definitions and a generic reviewer on the same model
  (`docs/findings/agent-critics-baseline.md`,
  `docs/findings/agent-critics-docs.md`); no training or test year applies.
- **The country-level observation audit compares each year's energy with
  Eurostat's**, which catches a series that measures only part of the fleet.
  Its bands were set after seeing the data and screen for a missing segment.
  NL and SE-BZ have no Eurostat reference and are not compared.
- **`docs/results.md`, `docs/archive.md` and a short-form scorecard** on the
  documentation site. The scorecard names each row's run, commit, roughness
  fill and curve library, and says that the reported configuration is not
  the lowest-error variant of its run in most rows and that some scorecard
  configs were edited after their runs.

### Changed

- **The derived roughness is filled within each calendar year**, so a
  training year no longer takes roughness from the next year, which can be its
  test year. A cell with no defined hour in a year stays missing, with a
  warning, and the January margin read for the old fill is gone. The manifest
  records `fill`. Results move slightly; the offset-fit pins are re-recorded,
  and the route-B scorecard rows lag the code until the next refresh.
- **Runs read ERA5 for the years they need:** the test year for evaluate and
  transfer, the training years for turbine-level training. Country-level
  training still reads every year. The manifest records `loaded_years`.
- **The open curve library has one copy, in `pyvwf.resources`**, with the same
  bytes and therefore the same numbers. An input root without curve tables
  uses it without a warning, and every run records the library it used by hash
  and each unit's curve in `curve_resolution.csv`. A test fails if
  `input/reference/` holds a curve table. A licensed library still goes in its
  own input root, and the country-level rows run on `input/combined`, as
  they have since 0.6.0.
- **The country-level generator fetches every wind type by default**;
  `--psr-type onshore` restores the old behaviour. For Belgium and France,
  run it with `--skip-observations`, or it overwrites their builders' files
  with ENTSO-E's; after a new fetch, move aside the backups that the Sweden,
  Norway and Portugal builders read their generation from
  (`docs/runbooks/entsoe.md`).
- **The country builders' shared steps live in
  `pyvwf.datasets.country_obs_rebuild`.** A rerun of the Portugal or Sweden
  builder over a file it already rebuilt now needs `--replace`, as the other
  builders did and as their own docstrings already said.
- **The scorecard audits move** from `scripts/studies/scorecard/` to
  `scripts/analysis/`.
- **The `slow` test marker is removed**; the fast set is
  `pytest -m "not realdata"`.

### Removed

- **Two superseded scripts**, `scripts/analysis/validate_region.py` (identical
  to `pyvwf-validate`) and `scripts/region_tools/assign_country_zones.py`
  (replaced by `weight_country_grid_points.py --zone-aware`), and the Danish
  bidding-zone polygons, which nothing read.
- **Archived at the tag:** `configs/regions/nl.toml`, a region with no
  scorecard row, and the Turkey runbook and probe; the data-sources guide
  keeps what the Turkey evaluation found.

### Fixed

- **Belgium's, France's, Sweden's, Norway's and Portugal's country-level
  observations are rebuilt so that generation and capacity cover one fleet**,
  except Norway's test year. Belgium's are rebuilt from Elia, France's from
  RTE's series on ODRE (under Licence Ouverte, where RTE's own files may not
  be redistributed), and Norway's training years from NVE's metered farm
  output. Sweden and Portugal keep ENTSO-E's generation over
  Energimyndigheten's and REN's fleets, interpolated through each year or
  month, and Norway's test year keeps ENTSO-E's generation over NVE's register.
  Italy's and Portugal's test files for 2023 now include offshore output; both
  rows stay suspended. Each old file is kept beside the new one as a backup,
  and the scorecard's corrections log has the figures.
- **The ENTSO-E fetcher leaves consumption out of wind generation, and drops
  an hour in which a wind type that reports is missing**, instead of scoring
  the remaining types against the whole fleet.
- **The review-agent scorer no longer excludes a review for a note under its
  files list.** It read every line after "Files read:" as a path, so a closing
  verdict or a note such as "(tried, not found: ...)" counted as a read outside
  the snapshot; every exclusion in the runs of 2026-10-29 was one, and one of
  them also had `/dev/null` in a command. A note now contributes only the
  absolute paths it names, so one naming a path outside the snapshot still
  excludes the review, and `/dev/null` in a command is not a read. The
  recorded results of those runs stand as scored; they are not rescored.
- **Germany's source is named correctly in the scorecard**: its row reads the
  licensed WindStats extract, not a public register. The results page says
  its Denmark figure belongs to the run it reports, not to the scorecard row.

### Documentation

- **The findings that rest on Belgium's and France's old observation files
  carry correction notices of 2026-10-25**, which withdraw the figures that
  read those files and say what stands. Those findings are at the tag.
- **The README leads with installing and running the package**, and points
  coding-agent users to `run-pyvwf`. `CONTRIBUTING.md` and the pull-request
  template are rewritten, with a section on AI-assisted contributions.
- **Fewer, current pages:** the region runbooks share one template and quote
  no results; guides are merged in pairs; the agent design pages are one; the
  harness design page and `docs/README.md` describe the code as it is now; and
  the released versions from 0.2.0 on are summarised, each with a link to its
  full notes.

## [0.6.0] - 2026-10-16

[Full notes](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/CHANGELOG.md#060---2026-10-16)

The harness becomes the one path, the package is imported as `pyvwf`, and the
power curve behind every number becomes a recorded fact. The version was first
dated 2026-09-19 and was published on 2026-10-16 with everything merged since.
Several changes move results: a factor rests on one set of accepted years or is
refused, the offset search returns a root or nothing, national fits with more
than one cluster take the joint fit again, and a country-level run simulates its
grid's own curve and refuses one the library lacks. Every scorecard row comes
from one refresh, and the scorecard's dated notices give what each change moved.
The research record shows that the affine fit's two parameters are not
separately identified, eliminates four explanations of why corrections do not
transfer across borders, and closes two registered studies without results.

The notes as first written said the residual test added to the offset search on
2026-09-15 moved no fit in this repository, because the golden regression test
was unmoved. [Corrected 2026-09-19: false. The residual test also refused
genuine fits that the search had left short of the root, and its fallback
returned a closer root, so fitted offsets outside the golden test moved by small
refinements, the Chile row's among them. See issue #18.]

PyVWF is not published on PyPI, so the removals below have no downstream
consumers and go without a deprecation period.

### Breaking

- **The package is imported as `pyvwf`, not `vwf`.** There is no `vwf` alias.
- **The legacy `PyVWF` class, `pyvwf-train` and the batch scripts are
  removed.** `pyvwf` exports `load_region`, `run_train`, `run_evaluate` and
  `run_transfer`; `docs/publications.md` records the last commit with the
  removed code.
- **Modules moved:** `pyvwf.datasets.era5` to `pyvwf.era5`,
  `pyvwf.harness.provenance` to `pyvwf.provenance`, and `load_power_curves` and
  `add_models` from `pyvwf.data` to `pyvwf.curves`.
- **The per-timestep roughness is the default.** `prep_era5` derives the
  roughness length from the 10 m to 100 m shear unless `roughness="stored"` is
  asked for, and a file with no 10 m winds now raises.
- **`find_offset` has one bracketed root search** and loses `max_iter`,
  `tolerance`, `initial_step` and `use_scipy_fallback`; `find_offset_iterative`
  and `MAX_OFFSET_RESIDUAL` are removed.
- **Every European configuration reads `era5/EU_2026-09`**, and
  `configs/regions/se_zonal.toml` is `se_bz.toml`.
- **`pyvwf.viz.load_results` reads a harness evaluate run**, and
  `align_to_obs` is gone.
- **`calculate_error` keeps only its `"total"` mode**, and `overall_error` is
  removed.
- **The `ml` extra is removed**, with several functions, constants and
  arguments nothing called; the full notes list them.

### Added

- **`pyvwf-validate`**, a console entry that trains, evaluates or transfers one
  region from its config.
- **Run records.** Every manifest records the environment, the loaded ERA5
  extent and the share of capacity outside it, and the requested and applied
  roughness. Every train, evaluate and transfer run writes
  `curve_resolution.csv`, the curve each model key actually used and its
  sha256, and `metrics.csv` gains the substituted, extrapolated, off-curve and
  excluded shares.
- **Fit diagnostics** (`fit_diagnostics_<slice>_<k>.csv`): what each fitted
  pair does to its own training speeds. Recorded, not yet used to mark a fit.
- **The `parallel` and `touchdesigner` extras.**
- **Development tooling:** pre-commit hooks, Dependabot, a CI job that runs
  every example, a Markdown link test, deptry, vulture and import contracts,
  the packaging and committed-files tests, `AGENTS.md`, `docs/CONTEXT.md`, three
  Claude Code skills, and `scripts/studies/`.

### Changed

- **A unit outside the loaded ERA5 extent stops the run** with
  `ExtrapolationError`; `[era5] allow_extrapolation = true` opts in and records
  the share.
- **A factor rests on one set of accepted years, or is refused (#28).** The
  factors table gains `n_years`.
- **A run refuses a run directory that already holds files.**
- **Offshore farms get their own points in the country grids**, and the grids
  are rebuilt on the corrected fleet register.
- **`pyvwf.data`, `pyvwf.clustering` and the harness driver are split.** Every
  name is still importable from its old module except
  `create_sampling_points`, `cluster_with_geometries` and
  `add_turbine_metadata`, now in `pyvwf.sampling`.
- **One helper for every weighted mean**, so a missing value leaves both the sum
  and the weights.
- **Scripts:** default paths follow `PYVWF_INPUT`, every study driver and
  analysis script parses its command line with `argparse` except the
  physics-informed and two transfer scripts, and `epias_tr.py` reads its
  credentials from the environment only.
- **pandas 2.2 is the declared floor.** CI tests Python 3.10 to 3.13, with
  3.10 to 3.12 installed through constraints files, and one job installs the
  `pinn`, `grid` and `data` extras.

### Fixed

- **The offset search returns a root or nothing (#18).**
- **National fits with more than one cluster take the joint fit again**, and
  the joint fit closes a small error instead of stopping at zero and refuses a
  failed fit.
- **A country-level run on a curve the library lacks is refused.**
- **The variants of a run are compared on the same rows.**
- **Corrected simulations keep each unit's own power curve and capacity**;
  the AU-NEM and NZ rows moved.
- **Turbine-level training fits the config's `train_years`**, evaluation
  refuses a training run from another config, and a config with an unknown key
  is refused.
- **KMeans partitions no longer depend on the machine's thread count**, and the
  power-curve cache cannot serve another table's curves.
- **A year's GWPT fleet is the fleet as it stood that year**, retired projects
  included.
- **The ENTSO-E fetcher refuses a series with no capacity register**, and a
  failed query raises instead of coming back empty.
- **Only four scorecard rows depend on the licensed curve library** (DE, DK,
  UK, US), not seven, and the UK and NZ gains cannot be distinguished from zero
  when the test year's units are resampled.
- **Published readings are corrected:** the daggered rows' worst damage never
  reached their scores; `method-cluster-count-dk.md` is not a reproduction of
  the published results; the rows do not share one roughness treatment; and
  Denmark's row carries the § marker. The IT, PT and ES rows were suspended
  for winds extrapolated outside the ERA5 download. *[Resolved 2026-09-13: the
  wider download covers them and they were re-run.]*

### Documentation

The research record adds the terrain studies, of which the turbine-only one
(#12) fails three of its five skill gates, the negative control among them,
so neither terrain gate reads as evidence about terrain; the country weighting
registration (#67), which stops at its row rule with nothing fitted; a
correction to the physics-informed comparison; and an account of how coding
agents are used. Every scorecard region has a runbook.

## [0.5.1] - 2026-09-11

[Full notes](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/CHANGELOG.md#051---2026-09-11)

A corrections-only release; the only source edit is a docstring. Every
country-level result had been simulated on a 100 kW fallback curve, because the
grid points name Vestas keys the bundled open library lacks; how much of the
country-level correction absorbs that mismatch rather than ERA5 bias is not
quantified. The scorecard said every region except CL and AR used the licensed
library, when the country rows used the open one, and 0.5.1 states the rows
that use the licensed library as not reproducible by a third party, seven of
them.
*[Correction, 2026-09-11: four, not seven. AU-NEM, BR and NZ were run with the
licensed library but simulate only on curves the open library contains, and
reproduce byte for byte on it.]* For seven of the nine turbine-level regions
the maintained config cannot produce the reported cluster count, so the exact
configuration behind each scorecard row was committed to
`configs/regions/scorecard/`; every evaluation re-run at its original commit
reproduced its `metrics.csv` byte for byte.

## [0.5.0] - 2026-09-01

[Full notes](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/CHANGELOG.md#050---2026-09-01)

Docker support (a multi-stage image that runs the synthetic example with no
data, built and exercised in CI), and `vwf.pinn`, an experimental
physics-informed correction for regions with no observed generation, behind the
new `pinn` extra. The README was cut to a front door, with the visualisation
gallery moved to its own guide.

## [0.4.0] - 2026-08-24

[Full notes](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/CHANGELOG.md#040---2026-08-24)

A region stopped being a code change. The validation harness (`vwf.harness`)
makes a region one TOML file plus one observation adapter, and ran the method
against observed generation in seventeen regions on four continents. It added
twenty region configs, fourteen adapters *[Corrected 2026-10-28: twelve
adapter classes, ten of them new in 0.4.0; fourteen counted the modules,
`base` and `registry` among them.]*, run provenance manifests, gridded
correction export, optional parallel offset fitting and k-means++
initialisation; it fixed scalar dilution in `calculate_scalar`, a country-level
path whose errors partly cancelled, and ERA5 `expver` handling. The merged open
curve library became the uniform default, so the validated rows reproduce
without the licensed library. *[Correction, 2026-09-11: this was never true.
Four scorecard rows (DE, DK, UK, US) simulate most of their capacity on curves
only the licensed library contains, so they cannot be reproduced without it.
The eight country-level rows did run on the open library, but every unit fell
back to a 100 kW curve. This note first said seven licensed rows: AU-NEM, BR
and NZ were run with the licensed library, yet reproduce byte for byte on the
open one. See [0.5.1](#051---2026-09-11) and `docs/findings/scorecard.md`.]*
Its known limitations: one held-out test year per region; Norway worse under
correction; the Netherlands excluded for a coverage defect; limited skill in
Chile and Argentina, where ERA5 exaggerates the north-south gradient; an
unscreened curtailment confound in the US; under-determined country-level
offsets; and implausible wind scalars in four of the nine turbine-level rows.

## [0.3.0] - 2026-07-17

[Full notes](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/CHANGELOG.md#030---2026-07-17)

The bundled power curves became real: the open library of 69 machines and 7
composites from NREL's turbine-models archive (BSD-3-Clause) replaced the
synthetic placeholders. Season slices took an optional month mapping, so a
Southern-Hemisphere fleet can be given its own seasons instead of inheriting
Northern ones; ERA5 longitudes are normalised on load; and the documentation
is hosted on Read the Docs.

## [0.2.0] - 2026-07-14

[Full notes](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/CHANGELOG.md#020---2026-07-14)

Pluggable observation sources (`vwf.sources`), four correction and evaluation
diagnostics in `vwf.viz`, the `data` extra, which takes `entsoe-py`, `openpyxl`
and `pyarrow` out of the core dependencies,
`PYVWF_INPUT` and `PYVWF_OUTPUT`, bundled reference tables so an installed copy
runs outside a checkout, and an API reference, mypy and a package CI job. It
fixed a silent year relabelling in the evaluation layer: error metrics computed
for training windows other than 2015 to 2019 were wrong. Version 0.1.2 was
never released, and its changes are folded in here.

## [0.1.1] - 2026-07-07

Paper revisions ahead of the JOSS submission.

## [0.1.0] - 2026-07-06

Initial public release: turbine- and country-level wind simulation from ERA5,
the per-cluster linear wind-speed bias correction at configurable spatial and
temporal resolution, evaluation metrics, the `pyvwf-train` console script, and
distributional diagnostics in `vwf.viz`.

[Unreleased]: https://github.com/ellyess/PyVWF/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/ellyess/PyVWF/compare/v0.5.1...v0.6.0
[0.5.1]: https://github.com/ellyess/PyVWF/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/ellyess/PyVWF/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ellyess/PyVWF/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ellyess/PyVWF/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ellyess/PyVWF/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ellyess/PyVWF/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ellyess/PyVWF/releases/tag/v0.1.0
