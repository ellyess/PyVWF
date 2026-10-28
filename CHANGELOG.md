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

### Added

- An evaluation of the three review agents (`method-critic`,
  `implementation-critic`, `red-team`) on proposals carrying defects this
  repository has already found, each matched with the same proposal fixed.
  The cases, the scoring rule and the pass bar are pre-registered in
  `docs/findings/agent-critics-prereg.md`; the cases and the driver are in
  `scripts/dev/agent_evals/`, and `tests/test_agent_evals.py` checks the
  scoring.
- The evaluation's first run, recorded in `docs/findings/agent-critics.md`
  with every review and judgement under `scripts/dev/agent_evals/runs/`. It did
  not pass its pre-registered bar. `docs/design/ai-native-development.md`
  gains a section on it.
- **The country-level observation audit compares each year's energy with
  Eurostat's** (`scripts/analysis/audit_country_observations.py`, #144). Its
  physical gates cannot see a series that measures only part of the fleet:
  when the capacity-factor denominator is partial too, the capacity factors
  look ordinary, as Belgium's did. Each covered year's observed energy is now
  divided by Eurostat's gross wind production
  (`configs/curation/eurostat_wind_gwh.csv`). A ratio outside a wide band
  fails the series and one outside a narrower band is noted. The bands were
  set after seeing the data, and screen for a missing segment; they are not a
  precision check. NL and SE-BZ have no Eurostat reference and are not
  compared.

### Changed

- **The `slow` test marker is removed.** Every test it marked is also
  `realdata`, so it selected nothing on its own. The fast set is now
  `pytest -m "not realdata"`, in CI, `scripts/dev/ci_local.py` and the
  contributor guides.
- **The finished studies' scripts are archived** at tag
  `archive/research-2026-10-27`: `scripts/studies/` (21 study directories),
  the ML transfer, EU re-run, regression-compare and #155 check scripts,
  `scripts/dev/repo_stats.py`, and their tests. The four scorecard audits
  move from `scripts/studies/scorecard/` to `scripts/analysis/`.
- **The study, superseded and LOCO region configs are archived** at the same
  tag: `configs/regions/study/`, `configs/regions/scorecard/superseded/`,
  `configs/regions/pinn_loco/`, and `configs/regions/nl.toml`, a region with no
  scorecard row.
- **The 2026-10-16 run of the review-agent evaluation is archived** at the same
  tag. The harness, its five cases and the run's results summary
  (`scripts/dev/agent_evals/results-2026-10-16.md`) stay.
- **The review-agent evaluation gains a no-agent baseline arm**, pre-registered
  in `docs/findings/agent-critics-baseline-prereg.md`: three generic reviewers
  on the critics' models, all six reviewers run from one `--agents` JSON
  without `AGENTS.md` loaded, and a gate on the critics' margin over the baseline.
  Run folders under `scripts/dev/agent_evals/runs/` are no longer committed.
- **The evaluation gains a docs suite for `docs-critic`**, pre-registered in
  `docs/findings/agent-critics-docs-prereg.md`: held-out page changes, each a
  change carrying a defect and the same change with it fixed, drawn from
  mistakes the definition's text does not draw on, and a generic baseline with
  the same model, tools and Bash hook. `evals.py prepare --suite docs` prepares
  it. Its session starts outside any git repository, so the reviewers' git
  commands do not reach the history, and a review that lists a file or command
  path outside its checkout is excluded. A baseline now takes its matched
  critic's tools and hooks as well as its model.
- **The scorecard is published in short form at `docs/scorecard.md`**, on the
  documentation site. It keeps both results tables, the fit-quality,
  suspended and superseded tables and the overclaim list, and adds a table
  naming each row's run, commit, roughness fill and curve library. Each dated
  notice becomes one line of a corrections log linking the full scorecard at
  tag `archive/research-2026-10-27`. It states that the reported
  configuration is not the lowest-error variant of its run in most rows, and
  that some scorecard configurations were edited after their runs, which
  `configs/regions/scorecard/README.md` had called byte-identical.
- **Finished studies' findings documents are archived** at tag
  `archive/research-2026-10-27`, and each result has an entry on a new
  results page, `docs/results.md`, with its years, its gates and a link to the
  full record; negative results sit beside positive ones, and a closing section
  lists where two records disagree. A new `docs/archive.md` says what left the
  main branch, where it is, how to cite the Zenodo record and how to reproduce
  from the tag. `docs/findings/` keeps only the active review-agent study.
  Links, comments and the preregistration test move with them; the test's
  complete-document case reads a fixture copy.
- **Germany's source is named correctly in the scorecard.** Its provenance
  line called it a public turbine register; Germany's row reads only the
  licensed WindStats extract, for metadata and generation alike, and the
  corrections log records the fix. The results page now says its Denmark
  figure belongs to the run it reports, not to the scorecard row. The
  data-sources guide gains Ireland's capacity repair, the region guide says the
  country-level generator writes to a fixed path, and the scorecard configs'
  comments point at documents that are still on the main branch.
- **The region runbooks share one template** (source and licence,
  observations, ERA5, train and evaluate, caveats) and quote no results; each
  points to the scorecard instead. Stale instructions are fixed: test-year
  files missing from the US and Brazil steps, the country-level generator's
  years, the Ireland repair and Norway's manual inputs, cluster counts that did
  not match the configs, and curve-assignment claims that did not match the
  processing scripts. Each runbook with a scorecard row names its scorecard
  config and the maintained one. The Turkey runbook and its probe are archived at tag
  `archive/research-2026-10-27`; `docs/guides/data-sources.md` keeps what the
  evaluation found.
- **The README leads with installing and running the package.** The
  quickstart starts from `git clone`, says that a shipped region needs its
  runbook's inputs, and points a user with their own fleet to the CSV route.
  The results table is unchanged, and the negative transfer result is named
  above it. The five-minute list is folded into "How this is built", which now
  names the pre-registered studies and the research archive, whose citation is
  linked.
- **The agent design docs are one page.** `docs/design/agent-guards.md` is
  merged into `docs/design/ai-native-development.md`: the guard table with
  what each guard misses, the incident behind each rule in brief, the
  critics' models, the mistake log's summary and rates, and the mistakes that
  reached publication. The full log stays at tag
  `archive/research-2026-10-27`. References outside dated records point at the
  merged page; the pre-registrations and evaluation cases keep theirs.
- **`CONTRIBUTING.md` and the pull-request template are rewritten.** The
  contributing guide gives each command once, describes the CI jobs correctly
  (it said CI installs no optional extras, beside a line describing the job
  that installs them, and credited the test jobs with running the example), and gains a section on AI-assisted contributions. The
  template's real-data section names the code the pins now reach, not the
  narrower set they covered before, and the template asks whether a coding
  agent was used.
- **Guides are merged in pairs:** the Docker guide into
  `installation.md`, running on your own fleet data into `training.md`, and
  the adapter contract into `adding-a-region.md`. The installation guide lists
  every extra, and states what the conda environment and continuous
  integration install. The input-root table in `training.md` puts every
  country-level row on `input/combined`, where it has run since country-level
  runs began refusing a missing curve. `adding-a-study.md` says what happens
  when a study finishes: a results entry, and its record leaves at the next
  archive tag.
- **Evaluate and transfer runs read ERA5 for their test year only, and
  turbine-level training runs for their training years only.** Country-level
  training still reads every year.
  `prep_era5` takes `years=(start, end)` and returns those years identical to
  the whole record cut the same way. The manifest's `era5_extent` block
  records `loaded_years`. Loading is faster and holds less in memory.
- **The derived roughness is filled within each calendar year.** An hour whose
  wind profile cannot be inverted was back-filled along the whole loaded
  record, so a year's last hours could take the next year's roughness, which
  for a training run can be its test year. `prep_era5` now back-fills within
  the year and fills a year's trailing gap forward; a cell with no defined
  hour in a year stays missing, with a warning. The January margin
  `prep_era5` read for that reason is gone. The manifest's `era5_roughness`
  block records `fill`. Results move slightly; the offset-fit pins are
  re-recorded and the route-B scorecard rows lag the code until the next
  refresh (`docs/findings/method-roughness-fill-prereg.md`).
- **The country-level generator fetches every wind type by default**
  (`generate_country_level_training_data --psr-type`, #145). It defaulted to
  onshore while the grids simulate every farm, offshore included, so the
  files built with the default left out offshore output in some or all of
  their years. `--psr-type onshore` restores the old behaviour.
- **France's observations are read from ODRE, under Licence Ouverte v2.0.**
  `scripts/region_tools/build_fr_obs_from_rte.py` read RTE's annual eCO2mix
  files, whose terms forbid redistributing them; it now reads the same
  definitive national series from ODRE (`eco2mix-national-cons-def`), RTE's
  open-data portal. ODRE's UTC stamps inherit the series' clock-change
  ambiguity, so the builder also leaves March's clock-change hour empty, which
  it kept before. Every other hour is unchanged except the last of 2021, which
  RTE's 2022 file and ODRE give differently.

### Removed

- **`scripts/analysis/validate_region.py`**, a wrapper identical to the
  `pyvwf-validate` console entry. The runbooks and guides now use
  `pyvwf-validate`.
- **`scripts/region_tools/assign_country_zones.py`**, superseded by
  `weight_country_grid_points.py --zone-aware`, which derives the zones itself.
- **The Danish bidding-zone polygons** (`configs/curation/zones/DK_1`, `DK_2`),
  which nothing read. The zones README says how to fetch them if a Danish
  zonal run is added.

### Fixed

- **Belgium's country-level observations are rebuilt from Elia** (#144). The
  ENTSO-E files held onshore generation alone in most years and onshore plus
  offshore in others, each over a matching capacity, so the correction was
  fitted to a fleet that changed from year to year while the grid simulates
  the whole one. `scripts/region_tools/build_be_obs_from_elia.py` builds every
  split from Elia's Open Data series `ods031`: onshore and offshore output
  over the monitored capacity behind it, so both cover the same fleet in
  every hour. The old files are kept beside the new ones as
  `.entsoe-mixed.bak.csv`. The country row moves; the scorecard's notice of
  2026-10-22 has the figures.
- **France's country-level observations are rebuilt from RTE** (#144). The
  ENTSO-E test file was onshore alone, since ENTSO-E holds no French offshore
  output before June 2023, and the training capacity fell behind the fleet in
  some training years, which overstated their capacity factors.
  `scripts/region_tools/build_fr_obs_from_rte.py` divides RTE's definitive
  eCO2mix national wind output, onshore and offshore, by RTE's year-end wind
  fleet from ODRE (`parc-prod-par-filiere`), interpolated through the year.
  The old files are kept as `.entsoe.bak.csv`, and the France offset-fit pin
  is re-recorded. The country row moves; the scorecard's notice of 2026-10-23
  has the figures.
- **Sweden's country-level observations are divided by Energimyndigheten's
  wind fleet.** Their generation from ENTSO-E was sound, but their
  capacity-factor denominator was not: flat over 2015-19, built by an old
  fetcher from the generation itself, and elsewhere ENTSO-E's start-of-year
  onshore register, which leaves out each year's growth and the offshore
  output the generation includes.
  `scripts/region_tools/build_se_capacity_from_energimyndigheten.py` keeps the
  generation and divides it by Energimyndigheten's year-end fleet, onshore and
  offshore, interpolated through the year. The old files are kept as
  `.entsoe-capacity.bak.csv`. The zonal files are not changed. The country row
  moves; the scorecard's notice of 2026-10-25 has the figures.
- **Norway's country-level observations are rebuilt from NVE.** ENTSO-E's
  zone-summed generation fell short of the official totals by a different
  share in each training year, and its capacity was ENTSO-E's register.
  `scripts/region_tools/build_no_obs_from_nve.py` builds the training years
  from NVE's hourly output per farm, metered at the grid connection, over
  NVE's turbine register, counting a farm in a month only when its output
  and registered capacity agree; the test year keeps ENTSO-E's generation over
  the register. The old files are kept as `.entsoe.bak.csv`. The zonal files
  are not changed. The country row moves; the scorecard's second notice of
  2026-10-25 has the figures.
- **Italy's and Portugal's test files for 2023 include offshore output**
  (#144). Built with the generator's onshore default, they left out the
  Taranto and WindFloat Atlantic offshore farms. They are rebuilt from
  ENTSO-E with every wind type, and the old files are kept as
  `.entsoe-onshore.bak.csv`. Both rows stay suspended.
- **Portugal's country-level observations are divided by REN's wind fleet.**
  Their generation from ENTSO-E was sound, but their capacity-factor
  denominator was a register ENTSO-E stopped updating in 2015-19 and, from
  2020, an onshore register while the generation includes offshore.
  `scripts/region_tools/build_pt_capacity_from_ren.py` keeps the generation and
  divides it by REN Data Hub's month-end wind fleet, onshore and offshore,
  interpolated through each month. The old files are kept as
  `.entsoe-capacity.bak.csv`. The row stays suspended; the scorecard's note of
  2026-10-27 has the figures.
- **The ENTSO-E fetcher leaves consumption columns out of wind generation**
  (`pyvwf.datasets.fetch_entsoe_capacity_factors`, #144). entsoe-py splits a
  type that also draws power into its output and its consumption, and every
  generation path summed the two. Belgian offshore reports consumption in some
  hours. Only output is kept now. No stored file was rebuilt for this change
  alone.
- **The ENTSO-E fetcher drops an hour in which a wind type that reports is
  missing** (#145). It summed whichever types had a value, so an hour missing
  onshore kept the offshore output alone against the whole fleet's capacity,
  and an hour with no value at all became zero output. Each type is now
  required between its first and last reported hour. Outside that span it is
  taken not to exist yet, which cannot tell a farm not yet built from one not
  yet reported, so that case is left to the audit. Rebuilt with it, Italy's
  and Portugal's test files are unchanged.

### Documentation

- **The CHANGELOG summarises each released version** and links to its full
  notes at tag `archive/research-2026-10-27`. Each dated correction of a
  claim the summary keeps stays beside it.
- **The README gives a reviewer a short route through the project.** A line
  near the top links how it is built, an "If you have five minutes" section
  points to the results, the scorecard, the account of development with
  coding agents and its case study, one study from registration to result and
  one negative result, and the results section leads with a table of
  uncorrected and corrected error per region, its markers explained directly
  under it. The table is a copy of the scorecard's, and
  `tests/test_readme_results.py` fails when the two disagree.
- **The findings that rest on Belgium's and France's old observation files
  carry correction notices of 2026-10-25.** The physics-informed (leave one
  country out, and turbine), country-level, curve-library, country weighting,
  roughness treatment, EU re-run, correction identifiability, transfer,
  scalar bounds and joint-fit reachability findings, with their
  pre-registrations where those are affected, withdraw the figures that read
  the old files, re-score what can be re-scored exactly from stored frames,
  and say what stands and what waits on a rerun. The roughness-fill
  pre-registration carries a notice too, and amends its refresh checks for the
  rows whose files were rebuilt.
- **The ENTSO-E runbook and the data-sources guide build Belgium's and
  France's observations with their builders** (`docs/runbooks/entsoe.md`,
  `docs/guides/data-sources.md`). The generator still builds both grids, with
  `--skip-observations`, since without it the generator overwrites the
  builders' files with ENTSO-E's. The guide says what the fetch keeps and
  what each builder does, and its source table gains Elia, RTE, ODRE and
  Eurostat with their terms; RTE's do not allow its eCO2mix files to be
  redistributed.

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
