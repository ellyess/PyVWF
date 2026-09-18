# Changelog

All notable changes to PyVWF are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
PyVWF adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, the public API may change in a minor release.

The version is defined once, in `vwf.__version__`; `pyproject.toml` reads it
from there, and `tests/test_packaging.py` asserts that `CITATION.cff` and the
newest release in this file stay in step with it.

## [Unreleased]

Changes since 0.5.1 make the power curve behind every number a recorded fact: a
run says which curve each unit was actually simulated on, and the scorecard
says how much of each fleet runs on a curve from a different manufacturer from
the turbine's own. No correction, clustering or curve-matching behaviour
changes, and the golden regression test is untouched. One scoring change: the
variants of a run are now compared on the same rows (see Fixed). Every
scorecard row reproduces exactly on this code except CL, AR and the US, where
a corrected variant lacks some values.

Since 2026-09-15 two behaviour changes join them. The offset search now tests
the residual it leaves rather than the size of its last step, so a search that
never reaches the root refuses instead of reporting success; the golden
regression test is unmoved, so no fit in this repository was affected. And the
gridded correction surface answers at every cell instead of filling some with
the identity, carrying per-cell distance, support count, kriging variance and a
plausibility flag so that a user can tell a correction that declined to answer
from one that happens to be the identity.

The country-level observation gates gain a register test that asks when the
capacity moved rather than how far it moved in the end, tiered severities so
that one bad hour and a broken denominator no longer read alike, and clipped
rows that travel into the manifest and the metrics table.

The research record grew faster than the code. The affine fit is shown to solve
one equation in two unknowns, so its two parameters are not separately
identified, and four candidate explanations of why corrections do not transfer
across borders are tested and eliminated. Two registered studies are closed
without results, one blocked on observation files that do not exist and one
void because its gates were written against comparators that did not.

### Added

- **A switch for the temporal treatment of the roughness.**
  `[era5] roughness = "stored"` (the default, and what every existing run did)
  or `"derived"`. `"derived"` ignores a stored roughness field and inverts the
  log profile per timestep from the 10 m and 100 m winds. Every harness run
  records both the requested and the applied treatment in an `era5_roughness`
  manifest block, since the two differ when the files carry no stored field.
  Which treatment is better is under test, and the comparison is
  pre-registered; nothing changes until it reports.
- **Fit diagnostics: what a fitted pair does to its own training speeds.**
  `fit_quality` bounds the scalar and checks each offset converged, but an
  affine pair with a negative offset sends every speed below
  `-offset / scalar` off the curve, and out of its own objective. Train runs
  now write `fit_diagnostics_<slice>_<k>.csv`: per cluster, slice value and
  training year, the zero-crossing speed and the capacity-weighted training
  steps sent below 0 m/s and above the curve. `fit_quality` reports the worst
  shares, and they reach `metrics.csv`. They are recorded beside the dagger
  and do not set it until a bound has its own pre-registered calibration.
- **Off-curve values are counted.** A speed below 0 m/s or above the curve
  table's last speed has no value on the curve, so the capacity factor is
  missing, not zero. A monthly mean skips it. Every variant of an evaluate or
  transfer run now records the capacity-weighted shares of unit-steps below
  the curve, above it and with no speed. It also records the unit-months
  scored on only some of their steps, which the common-row scoring does not
  catch. They are `metrics.csv` columns and an `off_curve` manifest block, and
  a non-zero share warns. Recording only: off-curve values stay missing.
- **The loaded ERA5 extent is recorded for every harness run.** The manifest
  gains an `era5_extent` block: the loaded extent, the requested bbox, the
  units outside and their capacity share, and whether the region opted in.
  `metrics.csv` gains `extrapolated_capacity_share`, and `prep_era5` warns
  when the ERA5 files stop short of the requested bbox. `CONTEXT.md` defines
  the loaded extent and the extrapolated share, and `docs/README.md` states
  the scorecard markers and their rules.
- **Curve resolution logging.** Every harness train, evaluate and transfer run
  writes `curve_resolution.csv`, recording for each model key the fleet
  requests:
  - whether `power_curves.csv` has it;
  - which curve was actually used, and that curve's sha256;
  - whether that curve is an open-library curve.

  The manifest carries a summary, and `metrics.csv` gains
  `substituted_capacity_share` on every row. A model missing from the table
  used to be visible only as a one-off warning, which is how every
  country-level run on the bundled library came to simulate a 100 kW
  distributed-wind turbine unnoticed. Recording only: the fallback itself is
  unchanged, and a test now pins its identity. `run_hindcast` and the legacy
  `PyVWF` path are not covered; `docs/guides/output-structure.md` says so.
- `add_models` adds a `model_match` column naming the tier that matched each
  turbine: `fuzzy-manufacturer+specific-power` or `specific-power-only`. The
  manufacturer tier is loose (a difflib cutoff of 0.3 lets "ewt" match
  "vestasv"), and the tier name says so; matching itself is unchanged.
- **Cross-manufacturer curve audit** (`scripts/analysis/curve_match_audit.py`),
  comparing each unit's own manufacturer with the manufacturer of the curve it
  was assigned. The scorecard's turbine table gains Other brand, Reference
  curve and Unverifiable columns from it. Whether held-out skill survives
  specific-power matching is not assessed; it is an open question for a curve
  library study.
- `tests/test_packaging.py` ties the newest CHANGELOG release, its compare links
  and the `CITATION.cff` release date to `vwf.__version__`.
- **Two Claude Code skills** in `.claude/skills/`. `findings-doc` checks a
  findings document or scorecard row against the evidence rules before it
  lands. `new-region`, invoked by name, adds a turbine-level region in
  seven gated phases, stopping for human sign-off after each.
- `tests/test_committed_files.py` fails if a tracked file is excluded by
  `.gitignore`, apart from a stated allowlist, if a licensed-library file is
  tracked, or if either copy of the open library differs from its recorded
  sha256. A third skill, `provenance-guard`, runs it with the packaging and
  curve-library tests before any release-shaped action.
- **`AGENTS.md`**, the standing rules for coding agents, imported by
  `.claude/CLAUDE.md` so Claude Code loads it with `CONTEXT.md`. A committed
  `.claude/settings.json` denies tagging, releases, and pushes to `main` or
  of tags; `AGENTS.md` gives the reason and the limits.
- **`CONTEXT.md`, the project's controlled vocabulary.** One approved term per
  concept, one sense per term, with the rejected synonyms listed. It resolves
  collisions that had already caused errors, among them four senses of
  "fallback" and "curve table" meaning two different files. Procedural
  documents use only its terms. Nothing loads it automatically yet.

### Changed

- **A unit outside the loaded ERA5 extent stops the run.** `interpolate_wind`
  extrapolated winds linearly past the grid without a warning, and the IT, PT
  and ES country rows were simulated that way. It now raises
  `ExtrapolationError`, naming the units, their capacity share, how far
  outside they lie, and the two remedies. A region can opt in with
  `[era5] allow_extrapolation = true` (`allow_extrapolation=True` on the
  legacy `PyVWF` class). The run then finishes and records the share, and any
  scorecard row from it carries the § marker by rule (`docs/README.md`). The
  permission travels with the loaded dataset, so every path that simulates is
  covered. Passing the check means the units lie inside the loaded extent. It
  does not verify the data in those cells.

### Deprecated

- **Four documented names that nothing in PyVWF uses.** Each still works and
  now raises a `DeprecationWarning`; each will be removed in a future
  release.
  - `vwf.data.sim_turbines_to_country_cf`.
  - `vwf.loaders.country_level_loaders.country_gen_to_cf`.
  - `PyVWF.from_config`, which imported generated code by inserting a
    directory into `sys.path`. Use the `entsoe-country` adapter with a region
    config instead.
  - The module constants `vwf.data.COUNTRY_DIR`, `TURBINE_DIR` and
    `COUNTRY_LEVEL_DIR`. Use `vwf.config.PyVWFPaths` instead. They are now
    read when accessed, so they no longer go stale when the input root
    changes after import.

### Fixed

- **The variants of a run are compared on the same rows.** `run_evaluate` and
  `run_transfer` scored each variant on its own complete rows. So a corrected
  variant with no value for some units, such as those in a cluster whose offset
  fit failed, was compared with the uncorrected variant on a different set of
  rows, and the rows it dropped were the hard ones. Every variant of a run is
  now scored on the rows that all of them can score. The excluded rows, and
  the variants that lacked them, are written to `scoring_exclusions.csv`.
  `metrics.csv` gains `excluded_share`, and the manifest gains a
  `common_row_scoring` summary. Of the scorecard rows, CL moves materially and
  AR and the US marginally.
- **Only four scorecard rows depend on the licensed curve library** (DE, DK, UK,
  US), not seven. AU-NEM, BR and NZ were run with it but simulate only on
  curves the open library also contains, and reproduce byte for byte on the
  open library. The scorecard, the 0.4.0 correction note and the 0.5.1 entry
  are corrected in place.
- **The section 7 caveat in `method-country-level.md` is reopened.** The DK and
  NZ correlations it had cleared rest mostly on other-maker curves. The notice
  now leaves open whether they hold on each unit's own curves.
- **The UK and NZ rows do not show that the correction improves those
  regions.** When the test year's units are resampled, neither row's gain can
  be distinguished from zero, and in each a few units that the correction
  makes worse decide the result. For NZ the limit is structural: the fleet has
  too few farms. The scorecard carries a dated correction notice and marks
  both rows, and so does the README. `method-cluster-count.md`,
  `method-physics-informed.md` and `region-nz.md` carry notices where they
  report UK or NZ skill. The UK fleet is counted in farms, not turbines. AR
  and AU-NEM are named as resolved only narrowly. The resampling scripts are
  `scripts/analysis/baseline_bootstrap.py` and `unit_concentration.py`.
- **The daggered rows' worst damage never reached their scores.** A degenerate
  cluster loses corrected values through a failed offset, or through speeds
  pushed off the power curve, where the interpolator gives no value rather
  than zero output. Those values dropped out of the score, so the corrected
  score left out exactly what a degenerate fit damaged most. That is the
  reverse of the scorecard's reading that the fleet average absorbed the
  damage. The scorecard's notice sizes it for CL, AR, the US and BR. CL also
  compared its uncorrected and corrected scores on different plants. Its row,
  and AR's, now come from a re-run on the common-row harness, and CL carries
  the unresolved-gain marker in the scorecard and the README.
  `region-south-america.md`, `method-scalar-bounds.md` (whose pre-registered
  gates still pass, with smaller margins), `method-hourly-resolution.md`,
  `method-physics-informed.md` and `method-cluster-count.md` carry notices
  where they report CL skill.
- **`method-cluster-count-dk.md` is not a reproduction of the published
  results.** Its scope line said the paper's research grid was reproduced. The
  run shares the grid, the onshore-only fleet and the years, but differs in the
  roughness treatment, the fleet size after the paper's exclusions, and the
  curve library. The notice states all three, and the sweep's own numbers
  stand. `method-harness-regression.md`, which already kept its comparison with
  the paper loose on purpose, gains the roughness difference in its list.
- **The rows do not share one roughness treatment.** Every region derives the
  surface roughness by inverting the log wind profile from the 10 m and 100 m
  winds. The European files carry a single annual mean of it, so DE, DK, UK and
  the eight country-level rows run on one static field per year, while the
  other six rows derive it hour by hour. The difference is not the formula but
  whether its result varies in time, and it confounds any comparison between
  the two halves of the scorecard. Which treatment is better is not
  established, and is under test on Denmark; no figure changes until that
  reports. The scorecard carries the notice, and two documentation errors are
  corrected with it: `COMBINED_ERA5_USAGE.md` called the European roughness
  terrain-derived, and `combine_era5_files.py` offered a `terrain` source that
  does not exist.
- **The IT, PT and ES country-level rows are suspended.** *[Resolved
  2026-09-13: the wider download covers them and they were re-run; see the
  entry above.]* The European ERA5
  download never covered their southern grid points, and the harness
  extrapolated winds to them without a warning. In ES and IT the fit then
  drove offsets to values that push most of those days off the curve, and the
  dropped days made the corrected capacity factor read high. PT's figures
  barely move under the same check, and that is the warning: nothing in its
  metrics shows its winds were never in the input. The scorecard moves the three
  rows to a table of suspended rows, with a notice giving the capacity share
  outside the data for each. They return once ERA5 covers them and they are
  re-run. NO and SE keep their rows, with their smaller extrapolated shares
  stated. `method-country-level.md` carries the notice, and the README drops
  the three from its national-level line.

### Documentation

- **The eleven European rows were re-run on the per-timestep roughness and a
  wider ERA5 box, and Spain, Italy and Portugal have returned from
  suspension.** The plan was registered before the download completed and
  measured the two changes apart: the treatment moved every row by a
  negligible amount, and the returning rows improved because their winds are
  real rather than extrapolated past the data, which the record locates rather
  than assumes. None of the three carries a degenerate fit. One registered
  prediction is refuted, and the record says its reasoning confused the
  symptom for the cause. Published rows are superseded rather than deleted.
  The re-run also found that an ordinary affine correction drops calm days
  that `fit_quality` cannot see, which is logged as candidate work.
  (`docs/findings/method-eu-rerun.md`.) Writing it up exercised the
  `findings-doc` rule added the day before, that a changed decision means
  re-reading whole documents rather than the edited part: four statements
  elsewhere still described the three rows as suspended, including a caveat in
  the scorecard itself quoting Norway's superseded figure.
- **The per-timestep roughness derivation is adopted as the method.** The two
  temporal treatments were compared on Denmark and France under a
  pre-registration, and the record (`docs/findings/method-roughness-treatment.md`)
  states that the change rests on method fidelity and comparability, not on
  accuracy: the measured effect is resolved by the registered gate and too
  small to matter. No scorecard figure changes on the comparison; the European
  rows are re-run only after the extended ERA5 download. Recording the
  treatments turned up a third route that nobody had named, which a manifest
  cannot yet tell apart from the annual mean, and the scorecard's tables gain a
  roughness column carrying all three. Both consequences of that route are
  logged as candidate work.
- **The scorecard's Denmark row carries the § marker.** The extent check of
  2026-09-11 compared the European rows against the extent of the ERA5 files
  rather than against each row's bbox-sliced extent, which is what a run
  loads, so it missed a row whose box stops inside the files.
  `scripts/analysis/extent_audit.py` now asks the second question for every
  row, read-only, and the suspension notice and the guard commit's claim about
  how many rows lie inside their grids are corrected in place, with a date.
  The row's own figures are unchanged.
- **Two design notes on surface roughness.**
  `docs/design/roughness-temporal-treatment.md` records that every region
  derives the roughness the same way but eleven rows apply an annual mean of it
  and six apply it per timestep, by two different routes, what that does to
  comparisons between regions, which treatment the method uses, and where the
  hub-height geometry lets either treatment matter at all.
  `docs/design/undefined-roughness-in-complex-terrain.md` records where the
  shear-derived estimator has no value at all: with no shear or inverted shear,
  which in the Andes cells of the Brazilian box holds for every hour of a
  month.
- `docs/README.md` separates procedural documents (guides, runbooks), which
  follow three writing rules adapted from Simplified Technical English
  principles, from argumentative ones (findings, design), which follow three
  structural rules. A findings document now answers one question and is
  revised in place, matching practice. New pre-registration records use a
  `-prereg` suffix.
- `docs/guides/adding-an-observation-source.md` now covers every file a turbine
  or plant-level region touches, in dependency order, with New Zealand as the
  template. It adds the three curve-assignment routes and what a region must
  record for its match to be checkable. Its registration example was a stale
  country-level `aemo` adapter, and its run example used the legacy path;
  both now show the harness. The built-in adapter table gains the three
  registered adapters it was missing.
- The adding-a-region guide and the New Zealand runbook follow the procedural
  writing rules: one instruction per sentence, short sentences, and only the
  terms in `CONTEXT.md`. The guide's heading is now "Adding a region and its
  adapter"; its file name is unchanged. The runbook gains a capacity-factor
  denominator section. The README's uses of rejected terms are fixed. The
  rules simplify sentence structure only: they never remove a proper noun, a
  document reference, an identifier or a standard technical term.
- The New Zealand documentation no longer misstates how the region is built.
  The adapter and transform docstrings and the `nz.toml` comment said the
  capacity-factor denominator comes from the EMI plant register; it comes
  from the curated capacity stages and farm table, and the register is only
  reported on. The runbook's cluster counts were two versions out of date,
  and it set `PYVWF_INPUT` on the train line only.

## [0.5.1] - 2026-09-11

A corrections-only release. The 0.5.0 archive carries a scorecard preamble and
a country-level findings document that state things the runs behind them do not
support; this release corrects them. No behaviour changes: the only source edit
is a docstring.

### Fixed

- **Every country-level result was simulated on a 100 kW fallback curve.** The
  country grid points name `Vestas.V80.2000`, `Vestas.V90.2000` or
  `Vestas.V90.3000`, none of which is in the bundled open library these runs
  used, so every grid point fell back, with a warning and no record, to the
  library's first column, `2019COE_DW100_100kW_27.6`, a 100 kW
  distributed-wind turbine. Confirmed by re-running the eight scorecard
  country rows. `method-country-level.md` carries a dated correction notice
  naming which of its figures rest on that curve, the scorecard states it, and
  the README's national-level line says so. How much of the country-level
  correction absorbs this mismatch rather than ERA5 bias is not quantified.
- **The scorecard's curve-library claim.** It said every region except CL and
  AR used the licensed library; the eight country-level rows used the bundled
  open library. The seven rows that do use the licensed library are now stated
  as not reproducible by a third party. *[Correction, 2026-09-11: four, not
  seven. AU-NEM, BR and NZ were run with the licensed library but simulate only
  on curves the open library contains, and reproduce byte for byte on it.]*
- **The scorecard's reproducibility claim.** The configurations behind its rows
  had never been committed, and for seven of the nine turbine-level regions the
  maintained config cannot produce the reported cluster count. The exact
  configurations are now in `configs/regions/scorecard/`, one per row. The
  links from each evaluation to its training run, recorded as a path that no
  longer exists, were verified by re-running every evaluation at the original
  commit: all seventeen `metrics.csv` files are byte-identical.

### Documentation

- `paper.md` is marked as archived: submitted to the Journal of Open Source
  Software, review closed, not currently under submission.
- The `vwf.viz` docstring no longer refers to a JOSS paper.

## [0.5.0] - 2026-09-01

### Added

- **Docker support.** A multi-stage `Dockerfile` builds the scientific stack
  into a virtualenv and copies it into a slim runtime that runs as a non-root
  user; `docker run pyvwf` executes the bundled synthetic example with no data
  and no arguments. Inputs and outputs are mounted rather than baked in, wired
  up by `docker-compose.yml`. `torch` is behind `--build-arg EXTRAS="[pinn]"`
  and the interpreter behind `--build-arg PYTHON_VERSION`. A CI job builds the
  image from a clean checkout and exercises it, so it cannot rot unnoticed.
  The `.dockerignore` is load-bearing: the working tree carries 43 GB of
  inputs, 7 GB of outputs and a 12 GB `.git`, and excluding them takes the
  build context from roughly 62 GB to 8 MB.

- **`vwf.pinn`, a physics-informed correction** for regions with no observed
  generation to fit against. Four bounded physical quantities (terrain
  speed-up, shear-exponent offset, conversion efficiency, sub-daily wind
  spread) are learned inside a differentiable forward operator and supervised
  directly on observed capacity factor, so the two-stage estimation of free
  per-cluster factors is removed. Zero-shot on nine regions it never saw, it
  improves on uncorrected ERA5 where a statistical transfer of the affine
  factors does harm. Not wired into the harness and no stable API yet. Needs
  the new optional `[pinn]` extra (`torch`), which CI does not install;
  `tests/test_pinn_physics.py` skips itself when torch is absent.
  Method, gates and results in `docs/findings/method-physics-informed.md`.

### Changed

- The README is cut from 683 lines to 217, with the visualisation gallery moved
  to a new `docs/guides/visualisation.md` and the CI detail to
  `CONTRIBUTING.md`. The harness design document and the physics-informed
  findings are rewritten to state what is true rather than narrate how the work
  unfolded.

## [0.4.0] - 2026-08-24

The theme of this release is that a region stopped being a code change. PyVWF
0.3.0 could correct Denmark, Germany, the UK and nine ENTSO-E countries, but
each of those was wired into the pipeline. 0.4.0 adds a validation harness in
which a region is one TOML file plus one observation adapter, and uses it to
run the method against observed generation in seventeen regions on four
continents. The correction maths, the clustering and the power curves are
unchanged and remain pinned bit-for-bit by a golden regression test.

### Added

- **The multi-region validation harness** (`vwf.harness`). Seven modules:
  `driver` (`run_train`, `run_evaluate`, `run_transfer`), `regions` (reads
  `configs/regions/*.toml`), `corrections` (a `CorrectionModel` registry with
  the affine baseline plus two controls), `skill` (the metrics), `provenance`,
  `export` and `hindcast`. The affine model delegates to the existing
  `vwf.correction` / `vwf.wind` / `vwf.data` code rather than reimplementing
  it, and a regression test pins the harness output against the legacy path to
  machine precision on four regions.
- **Twenty region configurations** in `configs/regions/`, and **fourteen
  observation adapters** behind the existing `ObservationSource` contract:
  AEMO (Australia), EIA (United States), ONS (Brazil), EMI (New Zealand),
  Coordinador/CEN (Chile), CAMMESA (Argentina), WindStats (Germany, Spain),
  Ofgem ROC (United Kingdom), per-zone and file-backed ENTSO-E, and a
  user-supplied CSV source for running the correction on your own fleet.
- **Explicit season month lists in every region config.** Season names
  previously resolved through a hardcoded Northern-Hemisphere mapping, which a
  Southern-Hemisphere region would have inherited silently. A config without
  season definitions is now refused rather than defaulted.
- **Run provenance.** Every run writes `run_manifest.json` recording the
  package version, git state and dirtiness, the region config and its hash,
  observation granularity, and the identity, path and hash of the curve
  library. Provenance is diagnostic and never aborts a run.
- **Gridded correction export.** Corrected wind and capacity-factor fields as
  NetCDF, applied across the archive, plus a hindcast entry point.
- **Ten region runbooks and fourteen findings documents** under `docs/`,
  including a per-region validation scorecard giving the source path for every
  reported number.
- **Optional parallel offset fitting.** `PYVWF_OFFSET_WORKERS` above 1 fans
  the row-wise fit across dask workers, roughly four times faster at high
  cluster counts. The default of 0 keeps the sequential path that the golden
  test pins; the parallel path is verified bit-identical.
- **Optional clustering variants**, both off by default because the evidence
  did not support making either standard: capacity weighting, and geographic
  distance on the unit sphere.
- Region shape repair (islands that `country_shapes.geojson` omits) and
  bidding-zone polygons for the zonal regions.

### Changed

- **k-means clustering now uses k-means++ initialisation.** The previous
  initialisation made the partition a seed lottery, and an apparent skill
  curve against cluster count turned out to be partition noise rather than
  signal. Measured across five regions before adoption.
- **The merged open curve library is the uniform default**, so the validated
  rows reproduce without the licensed library. *[Correction, 2026-09-11: this
  was never true. Four scorecard rows (DE, DK, UK, US) simulate most of their
  capacity on curves only the licensed library contains, so they cannot be
  reproduced without it. The eight country-level rows did run on the open
  library, but every unit fell back to a 100 kW curve. This note first said
  seven licensed rows: AU-NEM, BR and NZ were run with the licensed library,
  yet reproduce byte for byte on the open one. See
  [0.5.1](#051---2026-09-11) and `docs/findings/scorecard.md`.]*
- `input/` is reorganised by pipeline stage (`raw/`, `observations/`,
  `reference/`), `scripts/` and `configs/` by function, and `docs/` by purpose
  (`guides/`, `runbooks/`, `findings/`, `design/`).
- The six separate ERA5 fetchers are consolidated into one script, batching
  months per CDS request; the default `--chunk-months` drops to 3, the CDS
  cost-limit ceiling.
- ERA5 daily averaging is now optional, so the correction can be tested at
  hourly resolution.

### Fixed

- **Scalar dilution in `calculate_scalar`.** The weighted observed mean took
  its numerator over reporting units but its denominator over every unit's
  capacity, including plants that reported nothing. Simulated values are never
  NaN, so only the observed side was scaled down, by exactly the reporting
  fraction, and the fitted scalar was wrong by that factor.
- **`calculate_scalar` compared observed and simulated over different
  samples.** Both sides are now taken over the same units.
- **The country-level path**: the estimator, the observation handling and the
  fleet weighting were each wrong in ways that partly cancelled.
- **ERA5 combine dropped `expver`**, which recent CDS responses mix (ERA5 and
  ERA5T) and which silently broke the concatenation.
- Degenerate fits are surfaced rather than returned quietly, and clusters too
  small to fit are guarded.
- Argentina's capacity denominators are rebuilt from turbine specifications;
  New Zealand's `Trading_date` header drift (2019 files use lowercase) is
  normalised.

### Known limitations

Every region result rests on a **single held-out test year** and is
screening-level, not an accredited yield assessment. The correction does not
help everywhere and the cases where it does not are reported rather than
dropped: **Norway gets worse** under correction (RMSE 0.034 to 0.039), and the
**Netherlands is excluded** because an ENTSO-E coverage defect caps its
reported capacity factor. Chile and Argentina remove the mean bias but add
limited skill, because ERA5 exaggerates the north-south wind gradient in both.
The United States carries an unscreened ERCOT/SPP curtailment confound.
Country-level offsets are fitted against one national series per month and are
therefore under-determined.

All seventeen regions were re-run on this release so the reported figures are
reproducible against this tag. The refresh moved almost nothing in aggregate:
fourteen of the seventeen reproduced to four decimal places, and only Chile
(0.104 to 0.105), the United States (0.098 to 0.097) and Norway's uncorrected
baseline (0.025 to 0.034) changed at the reported precision.

One caveat does survive the refresh. Four of the nine turbine-level rows
(Chile, the United States, Argentina, Brazil) rest on fits containing
implausible wind scalars: the aggregate metrics are real but those per-cluster
factors should not be reused. The United States is the instructive case, since
the `calculate_scalar` fix moved its headline RMSE by 0.0005 while more than
doubling its worst fitted scalar, from 20.54 to 46.39. A skill metric does not
reveal this, which is why `fit_quality` now travels with every corrected row.
See `docs/findings/scorecard.md`.

## [0.3.0] - 2026-07-17

### Added

- **Optional `seasons` mapping through the four season-handling sites**
  (`parse_time_slice`, `add_time_resolution_columns`, `correct_wind_speed` via
  `simulate_wind`, `find_offset`, and `find_offsets_country_level`). Named
  season slices previously resolved through a hardcoded Northern-Hemisphere
  month mapping, so a Southern-Hemisphere user got silently inverted seasonal
  corrections. The default (`seasons=None`) preserves the legacy NH behaviour
  byte-for-byte, pinned by a frame-equality test.
- **Read the Docs hosting.** `.readthedocs.yaml` builds the existing Sphinx
  site (same `docs` extra, warnings as errors) at pyvwf.readthedocs.io.

### Fixed

- **ERA5 longitudes are normalised to [-180, 180] on load.** A 0..360 ERA5
  file sliced with a [-180, 180] bounding box previously returned an empty or
  wrong subset silently. Normalisation is a pinned no-op for data already in
  range, so EU downloads are unaffected.
- The README no longer claims the repository ships turbine data with
  licensing "being confirmed": it ships none (the directory is gitignored),
  and the provenance story lives in `input/README.md`.

### Changed

- **The bundled power curves are now real.** The synthetic placeholder curves
  and turbine models are replaced by the open turbine curve library: 69 real
  machines plus 7 normalized composites from NREL/turbine-models
  (BSD-3-Clause, DOI 10.11578/dc.20210112.1), Gaussian-smoothed to
  capacity-factor curves with the published VWF method. Per-column sources and
  licenses ship in `power_curves_provenance.csv`, and `tests/test_curve_library.py`
  pins the library's invariants. Capacity-weighted coverage through
  `add_models` for the Danish and German fleets: 99.5% and 94.5% of capacity
  assigned a curve within 20% of the turbine's true specific power (previously
  90.7% and 79.9% on the synthetic placeholders). Matching is by specific
  power, not machine identity, and the fallback warning says so. The default
  sampling-point model is now the library's 2.6 MW market-average composite,
  and the bundled example data is regenerated against the new curves.
- **Public API docstrings brought to reference quality.** Nine below-bar
  docstrings (one-line stubs and undocumented parameters, including
  `PyVWF.simulate_cf` and the sources registry) rewritten with behaviour,
  typed Args, Returns, and usage notes; the generated API reference renders
  them.
- `paper.md` describes the bundled open curve library and its provenance.

### Documentation

- JOSS pre-submission audit: corrected the method-paper citation to its full
  six-author list, fixed referenced file paths and a broken README anchor, and
  removed doc claims the code did not back up (no PyPI release, not a fully
  pinned environment). Made the `input/turbine_level_data/` gitignore intent
  explicit.
- Trimmed the `paper.md` Functionality section to the released feature set,
  corrected the MERRA-2 wording, and revised the prose for submission.

## [0.2.0] - 2026-07-14

Version 0.1.2 was bumped in the source but never tagged or released; its
changes are folded in here. This release is a minor rather than a patch bump
because it adds public API, removes packages from the core dependency set, and
changes the numbers the evaluation layer reports.

### Added

- **Pluggable observation sources (`vwf.sources`).** Observed generation and site
  metadata now come from `ObservationSource` adapters resolved through a
  registry, so supporting a new region means writing an adapter rather than
  editing the core pipeline. Ships `EuropeanTurbineSource` (DK, DE, UK) and
  `InMemoryCountrySource` for caller-supplied frames. `train_set` and `val_set`
  take a `source=` argument; the existing `external_grid_points` /
  `external_obs_data` arguments still work and are wrapped automatically. See
  [docs/guides/adding-an-observation-source.md](docs/guides/adding-an-observation-source.md).
- **Correction-factor and evaluation diagnostics in `vwf.viz`.** Four figures
  promoted from the thesis plotting scripts, generalised (no hard-coded country
  or paths) and matplotlib-only:
  - `plot_correction_factor_map`: per-cluster Voronoi choropleth of the learned
    scalar and offset, on a diverging scale centred at the neutral value.
  - `plot_factor_joint`: the scalar-vs-offset joint distribution with marginal
    histograms.
  - `plot_error_vs_clusters`: error against cluster count, one line per temporal
    resolution, with the uncorrected error as a reference. The model-selection
    plot for choosing `n_clu` and `time_res`.
  - `plot_sim_vs_obs`: per-turbine mean simulated vs observed capacity factor
    against the `y = x` diagonal, annotated with fleet-level MBE and RMSE.
- `Results.train_turb_info`: the training fleet the correction factors were
  fitted on, which `plot_correction_factor_map` needs to reproduce cluster IDs.
- A `data` extra for the data-acquisition dependencies, and a `py.typed` marker
  so type information is exported to downstream users.
- `PYVWF_INPUT` / `PYVWF_OUTPUT` environment variables, and
  `PyVWFPaths.reference_file()`, which resolves the small static reference
  tables: your own copy under the input root wins, falling back to synthetic
  placeholders bundled in `vwf.resources` so an installed PyVWF runs anywhere.
- An API reference built with Sphinx (`pip install -e ".[docs]"`), and a CI job
  that builds it with warnings-as-errors.
- A `CHANGELOG.md` and a standalone `CODE_OF_CONDUCT.md`.
- Static type checking with mypy, and a `package` CI job that builds the sdist
  and wheel, validates the distribution metadata, and imports the installed
  wheel from a clean environment.
- Tests for the scientific core: known-answer tests for the error metrics, and
  an end-to-end test that drives the real `PyVWF.train` / `simulate_cf` over a
  synthetic fleet with a planted bias.

### Changed

- **`entsoe-py`, `openpyxl` and `pyarrow` moved out of the core dependencies**
  into the new `data` extra. They are only needed to *fetch* input data; nothing
  in the simulation, correction, evaluation or plotting path imports them.
  Install with `pip install "pyvwf[data]"` if you use `vwf.datasets`.
- `vwf.viz` is now imported unconditionally by `vwf/__init__.py`. It was
  previously guarded by a `try`/`except ImportError` that rebound `Results` and
  the `plot_*` functions to `None`, which turned a missing dependency into a
  confusing `AttributeError` deep in user code. `HAS_VIZ` remains, and is always
  `True`.
- The version is now single-sourced from `vwf.__version__` rather than
  duplicated into `pyproject.toml`.
- CI installs the package from `pyproject.toml` instead of a hand-curated pip
  list, so the declared dependency metadata is actually exercised.

### Fixed

- **Silent year-relabelling in the evaluation layer.**
  `metrics.prepare_monthly_data(train=True)` discarded the training
  observations' real `(year, month)` index and overwrote it with a hard-coded
  `2015-01`–`2019-12` range. Any training window that was not exactly those 60
  months raised a `ValueError`; worse, a 60-month window starting in a different
  year was silently relabelled, merging every observation against the wrong
  year's simulation and reporting plausible but incorrect error metrics. The
  real calendar labels are now preserved. **Error metrics computed for training
  windows other than 2015–2019 were wrong and should be recomputed.**
- `metrics.calculate_error` and `metrics.prepare_monthly_data` no longer mutate
  the caller's DataFrames in place. They previously assigned ID and time columns
  onto the shared fleet table, which callers reuse across many evaluations.
- `bottleneck` is a real dependency (xarray's `.bfill()` requires it, and
  `prep_era5` uses it for the surface-roughness field) and is now installed in
  CI, where the end-to-end example had been failing.
- Dropped `infer_objects(copy=False)` in the turbine loaders: the keyword is
  deprecated under pandas 3's copy-on-write and is slated for removal in
  pandas 4.
- Corrected several implicit-`Optional` annotations and initialised the
  country-level attributes on `PyVWF` that previously sprang into existence only
  when the right loader was called.
- **The country-level path could never run without externally supplied data.**
  `prep_country` accepted an `obs_level` argument and never read it, so the
  fallback branch always reached `country_gen_to_cf` with turbine-shaped columns
  and raised a confusing `ValueError` about a missing `output_kwh` column. It now
  raises `NotImplementedError` naming both ways to supply the data.
- **PyVWF could not be used outside a repository checkout.**
  `load_power_curves()` and `add_models()` read the literal relative paths
  `input/reference/power_curves.csv` and `input/reference/models.csv`, and the region-shape loader
  read `input/reference/shapes/*.geojson`, so an installed copy raised
  `FileNotFoundError` unless the working directory happened to be a checkout.
  The declared package data also matched no files, so the wheel shipped none.
  Paths now resolve through `PyVWFPaths`, and the reference tables are bundled.
  PyVWF warns loudly whenever it falls back to the synthetic placeholders, since
  simulating with invented power curves yields plausible, meaningless numbers.

## [0.1.1] - 2026-07-07

### Changed

- Paper revisions ahead of the JOSS submission.

## [0.1.0] - 2026-07-06

### Added

- Initial public release: turbine- and country-level wind simulation from ERA5
  reanalysis, the per-cluster linear wind-speed bias correction
  (`w' = scalar * w + offset`) at configurable spatial and temporal resolution,
  evaluation metrics, the `pyvwf-train` console script, and the distributional
  diagnostics (`plot_cf_distribution`, `plot_qq`) in `vwf.viz`.

[Unreleased]: https://github.com/ellyess/PyVWF/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/ellyess/PyVWF/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/ellyess/PyVWF/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ellyess/PyVWF/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ellyess/PyVWF/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ellyess/PyVWF/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ellyess/PyVWF/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ellyess/PyVWF/releases/tag/v0.1.0
