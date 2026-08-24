# Changelog

All notable changes to PyVWF are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
PyVWF adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, the public API may change in a minor release.

The version is defined once, in `vwf.__version__`; `pyproject.toml` reads it
from there and `tests/test_packaging.py` asserts `CITATION.cff` stays in step.

## [Unreleased]

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
  rows reproduce without the licensed library.
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
dropped: **Norway gets worse** under correction (RMSE 0.025 to 0.037), and the
**Netherlands is excluded** because an ENTSO-E coverage defect caps its
reported capacity factor. Chile and Argentina remove the mean bias but add
limited skill, because ERA5 exaggerates the north-south wind gradient in both.
The United States carries an unscreened ERCOT/SPP curtailment confound.
Country-level offsets are fitted against one national series per month and are
therefore under-determined. See
`docs/findings/multi_region_validation_scorecard.md`.

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
  [docs/guides/ADDING_AN_OBSERVATION_SOURCE.md](docs/guides/ADDING_AN_OBSERVATION_SOURCE.md).
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

[Unreleased]: https://github.com/ellyess/PyVWF/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ellyess/PyVWF/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ellyess/PyVWF/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/ellyess/PyVWF/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/ellyess/PyVWF/releases/tag/v0.1.0
