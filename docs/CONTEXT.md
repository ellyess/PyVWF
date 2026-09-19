# CONTEXT: the project's controlled vocabulary

This file lists the approved term for each concept in PyVWF, and the words not to
use for it. Each term has one sense only.

- **Procedural documents** (`docs/guides/`, `docs/runbooks/`) use only these
  terms. The procedural writing rules are enforced against this file.
- **Findings and design documents** (`docs/findings/`, `docs/design/`) define a
  term once, by citing this file, and then use it consistently.
- **Code identifiers** in backticks name code, not concepts. They are exempt,
  even where they use a rejected word (for example `fallback_model` or
  `default_curve_key`).

To add a concept, add one row: the approved term, its meaning, and the synonyms
it replaces. Do not give an existing term a second sense. A rejected word listed
under two rows, such as "curve table", was retired because it named both.

In "Do not use", an entry with no qualifier is always wrong and can be checked
by search. An entry with a qualifier in brackets is wrong only in that sense;
a reader, not a search, decides it. Enforcement is therefore a search for the
unqualified entries plus a reader's review of the rest. A clean search is not a
pass. The search also catches exact forms only: an inflected form, such as
"falls back" for "fell back", needs a reader.

`AGENTS.md` imports this file, so an agent working in the repository reads it
with the standing rules. A human writer reads it from here.

## The method

| Term | Meaning | Do not use |
|---|---|---|
| bias correction | The fitted adjustment of reanalysis wind speed, per cluster and time slice, learned from observed generation. Use the full term on first use in outward-facing documents (the README, guide introductions), then the short form "correction". Findings and runbooks may use "correction" throughout. | calibration (for the bias correction) |
| affine correction | The default correction model, `affine-wind`: corrected speed equals scalar times speed plus offset. | linear correction, scale-and-offset |
| correction model | A registered way to fit and apply a correction: `affine-wind`, or its controls `scalar-only` and `scaled-affine`. | method (for a correction model), variant (except the `variant` column) |
| scalar | The multiplicative parameter of the affine correction. | slope |
| offset | The additive parameter of the affine correction, in m/s. | intercept |
| factors | The fitted scalar and offset table of one run, one file per time slice and cluster count. | correction factors, bias factors, coefficients (for factors) |
| cluster | A spatial group of units that share one set of factors. | group (for a cluster), zone (except a bidding zone) |
| cluster count | The number of clusters in a fit: `k` for turbine-level regions, `N` for country-level ones. | number of clusters, `num_clu` in prose |
| cluster sweep | A fit over several cluster counts in one config (`cluster_list` with more than one entry). | k-sweep, k-swept |
| time slice | The time grouping the factors are fitted over: `fixed`, `season`, `bimonth` or `month`. | time resolution, temporal resolution, `time_res` in prose |
| uncorrected | Simulated from reanalysis with no correction applied. The `uncorrected` row of `metrics.csv`. | raw (for uncorrected output), baseline (for uncorrected output) |
| corrected | Simulated with a correction applied. | adjusted (for corrected output), calibrated (for corrected output) |
| fit quality | The `fit_quality` diagnostics of a factors table: scalar range, implausible scalars, failed offsets. | fit health |
| degenerate fit | A fit with any scalar outside 0.2 to 3.0, or any offset that did not converge. Marked with a dagger in the scorecard. | bad fit, implausible fit |
| accepted years | The training years whose offset was fitted and accepted. A factor averages its scalar and its offset over them and carries their count as `n_years` (issue #28). A year with no usable observation is not one. | valid years, fitted years, usable years |
| refused factor | A factor whose fits were attempted but whose accepted years are not a strict majority of the training years. Its scalar and offset are NaN, its units get no corrected values, and `fit_quality` counts it as a failed offset. | rejected factor, dropped cluster |
| unfitted cluster | A cluster with no usable observation in any training year, so no fit was attempted. It carries the identity, scalar 1 and offset 0, with `n_years` 0, and is left out of a transfer's collapse. Not a refused factor. | empty cluster, identity cluster |
| transfer | Applying one region's factors, collapsed to one set, to another region. | extrapolation (for a transfer), generalisation (for a transfer) |
| hindcast | Applying trained factors over a long reanalysis window to rank one period against the record. | back-cast, reanalysis replay |
| roughness | The surface roughness length z0, derived by inverting the log wind profile between the reanalysis 10 m and 100 m winds. The hub-height wind is `w100 ln(h/z0) / ln(100/z0)`. | roughness length (except on first use), z0 in prose |
| roughness treatment | How a run's roughness varies in time: an annual mean, or per timestep. It names the temporal behaviour, not where the value is computed. | roughness method, roughness mode |
| annual-mean roughness | The treatment that applies one static field per year, computed by `src/vwf/datasets/combine_era5_files.py` and carried by `era5/EU`. | climatological roughness, static roughness |
| per-timestep roughness | The treatment that derives z0 for every timestep and averages it to daily with the winds. The method since 2026-09-12 (`docs/findings/method-roughness-treatment.md`). | hourly roughness (for the treatment), derived roughness (for the treatment) |
| roughness route | Where a run's roughness comes from. **Three routes, 2026-09-16:** a stored annual-mean field the file carries, which `prep_era5` uses when present; a per-timestep derivation at load in `prep_era5`, used when no stored field exists; and a per-timestep derivation ahead of time in `scripts/era5/combine.py`, which the file then carries. The first route produces the annual-mean treatment and the other two the per-timestep one, so three routes give two treatments. This row previously said two routes and one treatment, counting only the per-timestep ones. | roughness source, roughness pipeline |

## Regions and observations

| Term | Meaning | Do not use |
|---|---|---|
| region | One config in `configs/regions/`, and the fleet and observations it names. | country (for a turbine-level region), area (for a region) |
| region code | The upper-case identifier in a config's `code` field, such as `NZ` or `AU-NEM`. | country code (for a region code) |
| region stem | The lower-case file-name form of a region, such as `nz` or `au_nem`. | slug |
| data source | The provider and dataset observations come from, such as EMI `Generation_MD`. | observation source, provider data |
| raw data | Downloads as received from a data source, under `<input-root>/raw/`. | source data, original data |
| adapter | An `ObservationSource` subclass that loads one data source for the harness, registered by name. | observation source, loader (for an adapter), source (alone, for an adapter) |
| turbine-level | The pipeline branch for per-unit observations (`obs_level = "turbine"`). It names the branch, not the unit; "plant-level" and "farm-level" may describe a region's units. | plant-level (as a pipeline level), farm-level (as a pipeline level) |
| country-level | The pipeline branch for one national or zonal series per period (`obs_level = "country"`). | national-level |
| unit | The entity one observation measures (`obs_unit`): turbine, farm, plant, complex or country. | site (for a unit), asset |
| fleet | The set of units a run simulates: the training fleet or the test fleet. | portfolio, stock (for a fleet) |
| training years | The years a correction is fitted on (`train_years`). | training window, fit period |
| test year | The single year a correction is evaluated on and never fitted on (`test_years`). | held-out year, validation year, evaluation year |
| capacity factor | Generation divided by capacity times hours. Abbreviated CF after first use. | load factor, utilisation |
| capacity-factor denominator | The capacity a unit's generation is divided by, and the source it comes from. | nameplate (unless the source is a nameplate), installed capacity (unless that is the source) |

## Curves

| Term | Meaning | Do not use |
|---|---|---|
| power curve | Capacity factor as a function of wind speed for one model. One column of `power_curves.csv`. | turbine curve |
| model key | The name that selects a unit's power curve: its `model` value, matching a column of `power_curves.csv`. | curve key, power curve key, model name |
| `power_curves.csv` | The file of power curves: a speed column and one column per model key. Always named by file. | curve table, power-curve table, curve file |
| `models.csv` | The model catalogue: one row per model key, with manufacturer, rating, rotor diameter and specific power. Always named by file. | curve table, models table, catalogue (alone) |
| curve library | One `power_curves.csv` with its `models.csv`, under an input root's `reference/`. | curve set, library (alone, where ambiguous) |
| open library | The curve library shipped with PyVWF, derived from the NREL turbine-models archive (BSD-3). | bundled library, synthetic library, open curve library |
| licensed library | A curve library whose terms do not allow redistribution. It is kept local and identified by sha256. | real library, real curve library, proprietary library, external library |
| combined library | The open and licensed libraries merged under `input/combined`. Which curve wins for a model key in both is not yet defined. | merged library |
| specific power | Rated power divided by rotor swept area, in W/m2. | power density, rotor loading, `p_density` in prose |
| curve assignment | Giving each unit a model key, by any route. | curve matching (for the whole act) |
| specific-power match | Assignment to the nearest specific power within a rating band (`assign_curves_from_library`), or the `specific-power-only` tier of `add_models`. | fallback (for this tier) |
| default curve | The single curve assigned to units that cannot be matched (`model_source` `default-uniform`). | fallback model, uniform model |
| fallback curve | The curve a unit is simulated on when its model key is missing from `power_curves.csv`: that file's first column. Not the default curve. | default model, substitute curve |
| substituted | The status of a unit simulated on the fallback curve. Its share of capacity is the substituted share (`substituted_capacity_share`). | fell back, missing curve |
| curve resolution | The record of which power curve each model key was simulated on (`curve_resolution.csv`). | curve log, resolution log |
| origin | Whether a power curve's values match the open library (`open`) or not (`external`). | provenance (for this field) |
| curve-match audit | The comparison of each unit's own manufacturer with its assigned model's manufacturer (`scripts/analysis/curve_match_audit.py`). | cross-manufacturer audit, manufacturer audit |
| same brand | Curve-match audit class: the assigned model's manufacturer is the unit's own. | correct curve, right curve |
| other brand | Curve-match audit class: the assigned model is another manufacturer's. | other-maker, cross-manufacturer, different-brand, wrong curve |
| reference curve | Curve-match audit class: the assigned model is a research reference design or generic composite, never a unit's own machine. | generic curve, composite (alone) |
| unverifiable | Curve-match audit class: either the unit's manufacturer or the assigned model's is not recorded. | unknown (for this class), unmatched |

## Runs and outputs

| Term | Meaning | Do not use |
|---|---|---|
| input root | The directory `PYVWF_INPUT` points at; `input/` by default. | input tree, data root, data directory |
| run | One train, evaluate or transfer invocation of the harness, with its own run directory. | job, experiment (for a single run) |
| run directory | The self-contained output folder of one run, under `output/validation/<CODE>/`. | run folder, results folder |
| manifest | A run's `run_manifest.json`: version, git state, config, and curve library identity. | metadata file, run log |
| maintained config | A region's `configs/regions/<stem>.toml`, which may carry a cluster sweep. | main config, live config |
| scorecard config | The byte-identical copy of the configuration behind one scorecard row, in `configs/regions/scorecard/` (`<stem>_k<N>.toml` or `<stem>_country.toml`). | refresh config |
| third-party reproducible | Regenerable by anyone from committed files, public data and the open library. | open (for a result), public (for a result) |
| loaded extent | The lon/lat range of the ERA5 grid a run loaded, after its bbox slice. A unit inside it has its winds interpolated. Inside is a statement about position, not a check of the data in those cells. | ERA5 coverage, covered (for inside the loaded extent) |
| off-curve value | A simulated speed below the power curve table's first speed or above its last. It has no value on the curve, so the capacity factor is missing, not zero. | out-of-range value, clipped value |
| extrapolated share | `extrapolated_capacity_share`: the share of a fleet's capacity outside the loaded extent, simulated from winds extrapolated past the grid. Marked § in the scorecard. | extrapolation share, outside share |

## Evidence and documents

| Term | Meaning | Do not use |
|---|---|---|
| gate | A pass or fail criterion fixed before the run it judges. | trigger (for a gate), threshold (for a pre-registered criterion) |
| pre-registration | Recording gates and predictions before the run, in a commit that precedes the results. | pre-specification (except the existing file name) |
| correction notice | A dated statement that a published claim was wrong: at the top of a findings document, or bracketed in the CHANGELOG. | correction (alone, in this sense), erratum, limitation (for a correction notice) |
| screening-level | Validation that ranks and diagnoses, and is not an accredited yield assessment. | preliminary (for validation status), indicative (for validation status) |
| findings document | A dated research record under `docs/findings/`, named `<type>-<subject>.md`. | report (as a document type), write-up (as a document type) |
| scorecard | `docs/findings/scorecard.md`: the index of per-region results, and the entry point to the findings. | results table, league table |
| scorecard row | One region's reported result in the scorecard: one configuration, with its scorecard config and run directory. | headline row, best row |
| study | One research question, with its own findings document, usually a pre-registration, its drivers in `scripts/studies/<stem>/` (the stem of the findings document) and its run directories under `output/<name>_<date>/`. | experiment (for a study), analysis (for a study) |
| driver | A script that produces a study's numbers: a thin entry point over `vwf`, in the study's directory, whose recorded command line is pinned by a test. | runner, analysis script (for a driver) |
