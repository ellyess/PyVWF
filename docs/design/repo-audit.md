# Repository audit, 2026-09-18

Phase 1 of the extensibility restructure. It records what every tracked file
is for, who uses it, and what should happen to it. It changes no code. Terms
follow [`CONTEXT.md`](../../CONTEXT.md). "Study" is not yet a term there; here
it means one question with its own findings document, pre-registration and
driver scripts. Section 7 proposes adding it.

Scope: the 403 tracked files outside `docs/findings/` at `3dbb83b`, on branch
`manuscript-chapters-45`. Every file has exactly one row in section 2.

## 1. Summary

**No file is dead.** Every file flagged as orphaned by the grep has a live
user. Most are the registered driver of a study. The driver names its
findings document, but the document does not name the driver back, so a
grep for the file name finds nothing. The two `src/` leads are reached by
registry name and by a re-export (section 3).

Verdicts, after the adjudication in section 2:

| verdict | files |
|---|---|
| keep | 330 |
| move | 53 |
| promote | 5 |
| merge | 3 |
| delete | 1 (`STATUS.md` untracked, as Phase 4 instructs) |
| unclear | 11 (questions Q3 to Q6 in section 10) |

**The structural finding.** `scripts/analysis/` holds two kinds of file under
one name. Some are reusable tools, such as `validate_region.py`,
`run_hindcast.py` and `curve_match_audit.py`. Others are the driver of one
study, such as `roughness_treatment_study.py` and `pivot_probe.py`. 28 of the
45 move as study drivers. `scripts/pinn/` is one study family. Nothing
tells a newcomer where a new study's code goes, and no guide covers adding a
study. The proposal is `scripts/studies/<findings-stem>/`, one directory per
findings document, with the tools left in `scripts/analysis/`.

**The cost that decides how far to go.** Of the 50 script files the verdicts
move, 21 (15 in `scripts/analysis/`, 6 in `scripts/pinn/`) are cited by exact
path in a dated findings document, usually in a command line. Moving them leaves those commands stale, and rule
6 forbids editing the findings. Section 10 sets out the options.

**Defects found in passing.** Section 11 lists them. Three need action
whatever else is decided:

- `scripts/analysis/run_hindcast.py:2` ("offers 068 / 015") and
  `scripts/analysis/export_correction_field.py:2` ("offer 103 artifact")
  carry the commercial framing that `CLAUDE.md` forbids in this public repo.
- The docs build fails with `-W` on this branch.
  `docs/design/manuscript-chapters-45.md` is in no toctree and is not
  excluded. This was confirmed by a build, not inferred.
- The fetch scripts resolve paths under `PYVWF_INPUT`, but the process scripts
  default to literal `input/`. With `PYVWF_INPUT=input/combined`, a fetch
  writes to one tree and the matching process step reads the other.

## Baseline verification, before any change

The same checks run at every gate. These are the numbers any later phase is
compared against.

| check | environment | result |
|---|---|---|
| test suite, one file per process | base, Python 3.13, pandas 2.3.3 | all 66 test files exit 0 |
| `ruff check src/vwf tests` (the CI scope) | venv, Python 3.11 | clean |
| `ruff check src/vwf tests scripts examples` (the requested scope) | venv, Python 3.11 | 55 errors: 15 E402, 11 F541, 8 E702, 7 F811, 7 F401, 4 F404, 2 E401, 1 F841 |
| `mypy` | venv with `[dev]` (pandas 3.0.6, pandas-stubs) | 1 error: `src/vwf/data.py:195`, return type `Series \| DataFrame` where `DataFrame` is declared |
| `python examples/run_minimal.py` | venv, Python 3.11 | passes |
| `sphinx-build -b html docs docs/_build/html -W` | venv with `[docs]` | **fails**: `design/manuscript-chapters-45.md` is in no toctree |
| `docker build` and `docker run` | none | **not run**: Docker is not installed on this machine |

The mypy error and the docs failure exist before any restructure. CI would
report both on a pull request from this branch. The venv is a scratch
Python 3.11 environment with `pip install -e ".[dev,docs]"`, which matches
the pandas-3 side of the CI matrix.

This document is excluded from the site build (`docs/conf.py`), as the
maintainer decided for Q9.

## 2. File table

Columns are as requested. `referenced by` lists real referrers, each checked
by hand, because a raw grep confuses stems: `scripts/process/eia_us.py`,
`vwf.datasets.eia_us` and `vwf.sources.eia_us` share the stem `eia_us`.
"findings:" marks a referrer under `docs/findings/`.

The proposed layout the verdicts use:

- Pipeline entry points stay in `scripts/fetch/`, `scripts/process/`,
  `scripts/region_tools/` and `scripts/era5/`.
- Reusable tools stay in `scripts/analysis/`.
- A driver of one study moves to `scripts/studies/<findings-stem>/`. The stem
  is the findings document's name without its `method-` or `region-` prefix
  and without `-prereg`. For example, `method-roughness-treatment-prereg.md`
  gives `scripts/studies/roughness-treatment/`.
- `scripts/pinn/` moves as one unit to `scripts/studies/physics-informed/`,
  but only after the turbine-only study has run. That study's registered
  commands name `scripts/pinn/` paths, and it has not run yet.

Every `move` row whose evidence says "path cited" carries the stale-command
cost described in section 10.

### Root, .claude, .github (28 files)

| path | one-line purpose | referenced by | verdict | evidence |
|---|---|---|---|---|
| .claude/CLAUDE.md | One line: `@../AGENTS.md` | CHANGELOG.md:115 | keep | Loads AGENTS.md, and CONTEXT.md through it, for Claude Code. |
| .claude/launch.json | Preview-pane config: `http.server 8931` over docs/_build/html | none | keep | ad8e833 "Add the docs preview launch configuration"; used by the preview tool only. |
| .claude/settings.json | Deny list: git tag, gh release, gh pr merge, bare and main-named pushes | AGENTS.md "Blocked commands"; CHANGELOG.md | keep | 13 deny rules; AGENTS.md says they are not a security boundary. .claude/settings.local.json is untracked (global git ignore). |
| .claude/skills/findings-doc/SKILL.md | Skill: shape, evidence, corrections and scorecard rows for findings docs | AGENTS.md; new-region SKILL.md:152; scripts/analysis/curve_library_tables.py; findings (1) | keep | `paths: "docs/findings/**"` in frontmatter. Covers the document, not the study's code layout (section C). |
| .claude/skills/new-region/SKILL.md | Skill: gated phases 0-6 for a turbine-level region | AGENTS.md; CHANGELOG.md; findings (1) | keep | Defers to the guide's file table (SKILL.md:36-37 "Do not keep a separate list"). Overlap is deliberate; see section C. |
| .claude/skills/provenance-guard/SKILL.md | Skill: read-only pre-release checks | AGENTS.md; CHANGELOG.md | keep | Sections 1-4: tests, guards, scorecard claims, licence. |
| .dockerignore | Build-context exclusions | none (read by docker) | keep | Excludes `outputs/` (which .gitignore does not); `*.nc` is root-only, so examples/data/era5/era5_example.nc still ships. |
| .github/workflows/ci.yml | CI: lint, test matrix, docs, docker, package | CONTRIBUTING.md:47; README.md:3 badge; STATUS.md:350 | keep | Triggers only on PR, push to main, dispatch (ci.yml:3-7). See section E. |
| .gitignore | Never-commit rules with reasons | tests/test_committed_files.py; data-sources.md:13 | keep | Missing `/outputs`: README.md:82, examples/quick_run.py:7 and visualisation.md:17 all write there. .gitignore:104-105 ignore two findings files that a tracked finding cites (see section B). |
| .mailmap | Collapses five author identities to one | .dockerignore | keep | Display only, rewrites nothing (.mailmap:5-6). |
| .readthedocs.yaml | RTD build: docs/conf.py, `[docs]` extra, fail_on_warning | CHANGELOG.md | keep | fail_on_warning: true matches CI `-W`. So the toctree warning in section B would fail RTD too. |
| .zenodo.json | Zenodo deposit metadata | .mailmap | keep | No version field; Zenodo takes the version from the GitHub release. |
| AGENTS.md | Rules for coding agents: standing rules, blocked commands, skills | .claude/CLAUDE.md:1 (`@../AGENTS.md`); CHANGELOG.md; STATUS.md; scripts/analysis/national_single_cluster_study.py; findings (2) | keep | Its "PIPELINE.md describes the older batch path" (:14) is the only statement that PIPELINE.md is legacy. |
| CHANGELOG.md | Keep-a-changelog release history, [Unreleased] at top | tests/test_packaging.py:83; .dockerignore; CLAUDE.md | keep | [Unreleased] spans lines 13-289 (29 bullets); grep for decimals and percentages in it returns nothing, so the "no numbers" rule holds. |
| CITATION.cff | Citation metadata, version 0.5.1, concept DOI | tests/test_packaging.py:63-69; README.md:273; docs/index.md:96; provenance-guard skill | keep | version 0.5.1 equals src/vwf/__init__.py:18. |
| CODE_OF_CONDUCT.md | Contributor Covenant 2.1 | CONTRIBUTING.md:101; CHANGELOG.md | keep | Standard text, CODE_OF_CONDUCT.md:91-93. |
| CONTEXT.md | Controlled vocabulary: approved term, meaning, rejected synonyms | AGENTS.md:13,130 (`@CONTEXT.md` import); docs/README.md (3); 3 skills; CHANGELOG.md (4); 21 findings docs; scripts/analysis/extent_audit.py:4; src/vwf/extensions/grid/surface.py:284; docs/design/manuscript-chapters-45.md | move to docs/CONTEXT.md | Phase 4 instruction. Keep the name: 21 dated findings cite it by name. AGENTS.md:130 (`@CONTEXT.md`) must change in the same commit (Q8). Lines 25-27 ('Nothing loads this file') are stale. |
| CONTRIBUTING.md | Public contributor guide: setup, tests, CI summary, PR steps | README.md:279; AGENTS.md:12; .gitignore comment; .dockerignore | keep | CONTRIBUTING.md:75 "NumPy-style docstrings, as used throughout vwf/" is false: src/vwf has 193 `Args:` sections and 1 `Parameters`; docs/conf.py:59 sets `napoleon_numpy_docstring = False`. |
| Dockerfile | Two-stage image, non-root, CMD runs examples/run_minimal.py | CI docker job; README.md:107-145; docker-compose.yml | keep | Dockerfile:86 CMD is meaningful; ci.yml:118-123 greps the demo's "reduced the mean capacity-factor error". |
| LICENSE | BSD 3-Clause | pyproject.toml:11 (license-files); Dockerfile:27 | keep | Header "BSD 3-Clause License". |
| PIPELINE.md | Legacy batch order: generate_country_level_training_data, train_all_bias_corrections, evaluate_all_pyvwf_runs | README.md:218; scripts/README.md:4; AGENTS.md:14 | merge into docs/guides/training.md | Stale: PIPELINE.md:53 set `turbine_fixed_2015_2019` absent from train_all_bias_corrections.py (sets at :49-220); :16-17 says grid code lives on `development`, but src/vwf/extensions/grid/ is on this branch. |
| README.md | Project front page: install, quickstart, Docker, validated regions, docs index, citation | pyproject.toml:8 (readme); Dockerfile:27; docs/README.md:3; docs/index.md:21; CONTRIBUTING.md | keep | Two defects: README.md:82 writes to `outputs/`, which .gitignore does not ignore (`git check-ignore outputs/x` empty); README.md:130-132 compose command writes to /app/output, not the mounted /data/output (see docker-compose.yml row). |
| STATUS.md | Session handover log: in flight, queue, open items, settled items | findings: method-cluster-selection.md:135 (relative link `../../STATUS.md`), method-offshore-pool-prereg.md:75, method-physics-informed-loco-prereg.md:48; CLAUDE.md (local) | delete | Untrack and add to .gitignore, as Phase 4 instructs; keep the file locally. Cost: findings method-cluster-selection.md:135 links it by relative path, and two findings name it (Q7). |
| docker-compose.yml | Volume mounts for a real run | README.md:127-136 | keep | Bug: mounts ./output at /data/output, but validate_region.py:27 defaults `--out output/validation`, relative to WORKDIR /app (Dockerfile:71); no PYVWF_OUTPUT set. The documented command's results stay inside the container and are deleted by `--rm`. |
| environment.yaml | Conda env `pyvwf` | README.md:48; CONTRIBUTING.md:29 | keep | Diverges from the dev extra: includes the `[data]` libs (entsoe-py, openpyxl, pyarrow) and black; lacks mypy, pytest-cov, pandas-stubs, cdsapi; python and numpy unpinned though README.md:251 says "pinned". |
| paper.bib | Bibliography for paper.md | paper.md:24; .dockerignore | move to paper/paper.bib | Only referrer is paper.md; moves with it. |
| paper.md | Archived JOSS submission, frozen at submission | .dockerignore; CHANGELOG.md | move to paper/paper.md | paper.md:26-29 "not currently under submission ... not updated"; commit 44e7cf9 "Mark the JOSS paper as archived". Nothing reads it at the root. |
| pyproject.toml | Build, deps, extras, pytest, coverage, ruff, mypy config | CI; Dockerfile:27; tests/test_packaging.py; .readthedocs.yaml | keep | pyproject.toml:88 `[ml]` extra documents `vwf.extensions.ml`, which does not exist (src/vwf/extensions has only grid; STATUS.md:99 "unported"). |

### configs/ (84 files)

| path | one-line purpose | referenced by | verdict | evidence |
|---|---|---|---|---|
| configs/curation/aemo_au_aliases.csv | AEMO DUID to GWPT farm alias table, with evidence per row (41 rows) | scripts/process/aemo_au.py | keep | Default `--aliases` at scripts/process/aemo_au.py:68, read by `pd.read_csv` at :87. aa0ec91 2026-07-23 Reorganise scripts/ and configs/ by function |
| configs/curation/ar_coord_overrides.csv | Hand-curated lon/lat/capacity overrides for CAMMESA plants where GWPT is phase-split (22 rows) | scripts/process/cammesa_ar.py; docs/runbooks/ar.md; docs/guides/data-sources.md | keep | Default `--overrides` scripts/process/cammesa_ar.py:121, read :146; runbooks/ar.md:41. 7260c43 2026-08-12 South America: real turbine curves |
| configs/curation/ar_turbine_specs.csv | Per-plant turbine model, rotor, hub height and capacity for Argentina (65 rows) | scripts/region_tools/apply_turbine_specs.py (via --specs); scripts/analysis/curve_match_audit.py; findings: region-south-america.md | keep | curve_match_audit.py:138 reads it at :141; runbooks/ar.md:19 `--specs configs/curation/ar_turbine_specs.csv`; cited docs/findings/region-south-america.md:126 |
| configs/curation/au_turbine_models.csv | Per-DUID turbine identity and specific power for AU-NEM (104 rows) | scripts/region_tools/assign_au_curves.py; scripts/analysis/curve_match_audit.py; findings: region-au-nem.md; examples/data/au_nem/README.md | keep | Default `--models-csv` assign_au_curves.py:37, read :46; curve_match_audit.py:137; cited docs/findings/region-au-nem.md:45 |
| configs/curation/cl_coord_overrides.csv | Hand-curated coordinates for CEN plants GWPT cannot place (6 rows) | scripts/process/cen_cl.py; docs/runbooks/cl.md | keep | Default `--overrides` scripts/process/cen_cl.py:126, read :143; runbooks/cl.md:29 |
| configs/curation/cl_turbine_specs.csv | Per-plant turbine model, rotor, hub height for Chile (60 rows) | scripts/region_tools/apply_turbine_specs.py (via --specs); scripts/analysis/curve_match_audit.py; findings: region-south-america.md | keep | apply_turbine_specs.py:32-34 usage; runbooks/cl.md:19; curve_match_audit.py:137; cited docs/findings/region-south-america.md:126 |
| configs/curation/gwpt_exclusions.csv | GWPT phases excluded from country fleets (one row: Hordavind, a NO5 placeholder) | scripts/region_tools/weight_country_grid_points.py; findings: method-country-level.md; docs/guides/data-sources.md | keep | `EXCLUSIONS_PATH` weight_country_grid_points.py:62, `load_exclusions` :65; cited docs/findings/method-country-level.md:408 |
| configs/curation/nz_capacity_stages.csv | Stable capacity plateaus for staged NZ builds (Turitea) | scripts/process/emi_nz.py; docs/runbooks/nz.md; nz.toml comment; src/vwf/sources/emi_nz.py docstring | keep | `pd.read_csv(configs / "nz_capacity_stages.csv")` scripts/process/emi_nz.py:58; src mentions (sources/emi_nz.py:17) are docstrings only |
| configs/curation/nz_mask_windows.csv | NZ commissioning-ramp months masked at load (6 rows) | scripts/process/emi_nz.py; docs/runbooks/nz.md; src/vwf/sources/emi_nz.py docstring | keep | Read at scripts/process/emi_nz.py:59; runbooks/nz.md:29,65,83 |
| configs/curation/nz_wind_farms.csv | Curated NZ farm table (13 farms: codes, coords, capacity, turbine model) | scripts/process/emi_nz.py; docs/runbooks/nz.md; src/vwf/datasets/emi_nz.py and sources/emi_nz.py docstrings | keep | Read at scripts/process/emi_nz.py:56; error message :151 names it |
| configs/curation/zones/DK_1.geojson | Bidding-zone polygon DK1 (entsoe-py, MIT) | glob in scripts/region_tools/assign_country_zones.py and weight_country_grid_points.py (only when called with DK); zones/README.md | keep | Loaded by `glob(f"{country}_*.geojson")` assign_country_zones.py:52, weight_country_grid_points.py:82. README: "no Danish zonal run exists yet". dd87232 2026-08-12 |
| configs/curation/zones/DK_2.geojson | Bidding-zone polygon DK2 (entsoe-py, MIT; lacks Bornholm) | as DK_1 | keep | Same loaders as DK_1; README caveat "DK2 is missing Bornholm", unpatched because no Danish zonal run exists |
| configs/curation/zones/IT_CALA.geojson | Bidding-zone polygon IT_CALA (post-2021) | zones/README.md only; no code reads it | unclear | Both loaders skip non-numbered stems: assign_country_zones.py:61-62 "named zones (Italy's IT_NORD etc.) are not numbered"; weight_country_grid_points.py:84 `isdigit()`. Need: is an Italian zonal path planned? |
| configs/curation/zones/IT_CNOR.geojson | Bidding-zone polygon IT_CNOR | zones/README.md only | unclear | Same as IT_CALA: skipped by assign_country_zones.py:61-62 and weight_country_grid_points.py:84; assign_country_zones IT would raise ValueError ":65 none are numbered" |
| configs/curation/zones/IT_CSUD.geojson | Bidding-zone polygon IT_CSUD | zones/README.md only | unclear | Same as IT_CALA |
| configs/curation/zones/IT_NORD.geojson | Bidding-zone polygon IT_NORD | zones/README.md only | unclear | Same as IT_CALA |
| configs/curation/zones/IT_SARD.geojson | Bidding-zone polygon IT_SARD | zones/README.md only | unclear | Same as IT_CALA |
| configs/curation/zones/IT_SICI.geojson | Bidding-zone polygon IT_SICI | zones/README.md only | unclear | Same as IT_CALA |
| configs/curation/zones/IT_SUD.geojson | Bidding-zone polygon IT_SUD | zones/README.md only | unclear | Same as IT_CALA |
| configs/curation/zones/NO_1.geojson | Bidding-zone polygon NO1 (NVE, NLOD) | scripts/region_tools/assign_country_zones.py; weight_country_grid_points.py --zone-aware; findings: method-country-level.md (directory) | keep | assign_country_zones.py:23 usage `SE NO`; docs/guides/data-sources.md:188 "--zone-aware for NO/SE"; method-country-level.md:332 cites `configs/curation/zones/` |
| configs/curation/zones/NO_2.geojson | Bidding-zone polygon NO2 | as NO_1 | keep | As NO_1 |
| configs/curation/zones/NO_3.geojson | Bidding-zone polygon NO3 | as NO_1 | keep | As NO_1 |
| configs/curation/zones/NO_4.geojson | Bidding-zone polygon NO4 | as NO_1 | keep | As NO_1 |
| configs/curation/zones/NO_5.geojson | Bidding-zone polygon NO5 | as NO_1 | keep | As NO_1; gwpt_exclusions.csv rationale relies on NO5 polygon |
| configs/curation/zones/README.md | Sources, licences, refresh commands and caveats for the zone polygons | scripts/region_tools/assign_country_zones.py (:16, error text :56); docs/guides/data-sources.md:140 | keep | Only licence record for vendored MIT/NLOD geometry; dd87232 2026-08-12 Country-level: fix the estimator |
| configs/curation/zones/SE_1.geojson | Bidding-zone polygon SE1 (entsoe-py, MIT, land-only) | as NO_1 | keep | assign_country_zones.py:22-23 usage `SE`; data-sources.md:188 |
| configs/curation/zones/SE_2.geojson | Bidding-zone polygon SE2 | as SE_1 | keep | As SE_1 |
| configs/curation/zones/SE_3.geojson | Bidding-zone polygon SE3 | as SE_1 | keep | As SE_1 |
| configs/curation/zones/SE_4.geojson | Bidding-zone polygon SE4 | as SE_1 | keep | As SE_1 |
| configs/regions/ar.toml | Maintained config AR: turbine, `cammesa-ar`, cluster_list [1, 12], test 2024 | T-all; tests/test_harness_regions.py:158; docs/runbooks/ar.md; scripts/fetch/era5.py docstring | keep | `shipped("ar")` test_harness_regions.py:158 pins obs_unit plant; fetch/era5.py:11 `--region ar` resolves `configs/regions/ar.toml` (:169) |
| configs/regions/au_nem.toml | Maintained config AU-NEM: turbine, `aemo-nem`, [1, 15], test 2023 | T-all; test_harness_regions.py:145,169; test_harness_provenance.py:90; validate_region.py docstring; docs/design/harness.md; docs/guides/training.md | keep | Loaded by name in tests/test_harness_provenance.py:90 and test_harness_regions.py:169 (hemisphere pin) |
| configs/regions/be.toml | Maintained config BE: country, `entsoe-country`, [1, 3], test 2023 | T-all; T-country; scripts/analysis/national_single_cluster_study.py; refit_control_points.py | keep | `INCLUDED` national_single_cluster_study.py:55 (driver of method-national-single-cluster-prereg.md); `DEFAULT_STEMS` refit_control_points.py:51 |
| configs/regions/br.toml | Maintained config BR: turbine, `ons-br`, [1, 20], era5/BR_daily, test 2024 | T-all; test_harness_regions.py:149; docs/runbooks/br.md; scripts/era5/combine.py | keep | combine.py:41 resolves `configs/regions/{code}.toml` for `--region br` |
| configs/regions/cl.toml | Maintained config CL: turbine, `cen-cl`, [1, 8], test 2024 | T-all; test_harness_regions.py:155; scripts/analysis/hourly_resolution_test.py; min_cluster_size_tradeoff.py; docs/runbooks/cl.md; scripts/README.md | keep | Hard-coded at hourly_resolution_test.py:166 and min_cluster_size_tradeoff.py:66 (docstring cites method-scalar-bounds.md, method-hourly-resolution.md) |
| configs/regions/de.toml | Maintained config DE: turbine, `european-turbine`, [1, 50], test 2019 | T-all; test_harness_regions.py:138; test_harness_provenance.py:83; cluster_selection_study.py; pivot_probe.py | keep | Stem "de" in cluster_selection_study.py:56 (driver of method-cluster-selection-prereg.md) and pivot_probe.py:65 |
| configs/regions/dk.toml | Maintained config DK: turbine, `european-turbine`, [1, 100], test 2020 | T-all; test_harness_regions.py:137; findings: method-offshore-pool-prereg.md; runbooks/dk.md; guides/your-own-data.md; validate_region.py; export_correction_field.py; examples/dk_raw_vs_corrected.ipynb | keep | Cited by path docs/findings/method-offshore-pool-prereg.md:69; cluster_selection_study.py:57-58 |
| configs/regions/es.toml | Maintained config ES: country, `entsoe-country`, [1, 4], test 2023 | T-all; T-country; national_single_cluster_study.py | keep | national_single_cluster_study.py:55 `INCLUDED`; differs from scorecard/es_country.toml only in era5 path/file_tag/roughness and a comment |
| configs/regions/es_ws.toml | Variant ES-WS: turbine, `windstats` (confidential WindStats), [10], train 1998-1999, test 2000, era5/ES | T-all (count of 20, comment :131); docs/runbooks/es.md | keep | Adapter `windstats` registered src/vwf/sources/windstats.py:65 and tested (tests/test_windstats_processing.py:98). Never run: `input/era5/ES` absent, no `output/**/ES-WS` |
| configs/regions/fr.toml | Maintained config FR: country, `entsoe-country`, [1, 10], test 2023 | T-all; T-country; national_single_cluster_study.py | keep | national_single_cluster_study.py:55 |
| configs/regions/ie.toml | Maintained config IE: country, [1, 3], train 2017-2021, test 2023 | T-all; T-country; national_single_cluster_study.py | keep | national_single_cluster_study.py:55 |
| configs/regions/it.toml | Maintained config IT: country, [1, 3], test 2023 | T-all; T-country; national_single_cluster_study.py; refit_control_points.py | keep | national_single_cluster_study.py:55; refit_control_points.py:51 |
| configs/regions/nl.toml | Maintained config NL: country, `entsoe-country`, [1, 5], test 2023; no scorecard row | T-all; T-country; national_single_cluster_study.py; weight_country_grid_points.py docstring; base of pinn_loco/nl_country.toml; findings: method-physics-informed-loco-prereg.md:51 | keep | See section F. Driver national_single_cluster_study.py:55 loads it (NL reported with defect, prereg :165); output/validation/country_baseline_2026-07-23/NL exists |
| configs/regions/no.toml | Maintained config NO: country, [1, 4] (one per bidding zone), test 2023 | T-all; T-country; national_single_cluster_study.py | keep | national_single_cluster_study.py:55 |
| configs/regions/nz.toml | Maintained config NZ: turbine, `emi-nz`, [1, 5], test 2024 | T-all; test_harness_regions.py:152; findings: region-nz.md; README.md; docker-compose.yml; runbooks/nz.md; run_hindcast.py; .claude/skills/new-region | keep | Cited by path docs/findings/region-nz.md:21 |
| configs/regions/pinn_loco/nl_country.toml | NL fold of the physics-informed LOCO study: nl.toml on era5/EU_2026-09 with derived roughness | findings: method-physics-informed-loco-prereg.md; method-physics-informed-turbine-prereg.md; output/pinn_loco_2026-09-16 manifests | keep | Cited by path in command lines at loco-prereg.md:212,240,261 and turbine-prereg.md:321. 66a68f7 2026-09-16 Add the leave-one-country-out driver and the NL study config |
| configs/regions/pt.toml | Maintained config PT: country, [1, 2], test 2023 | T-all; T-country; refit_control_points.py | keep | refit_control_points.py:51 `DEFAULT_STEMS = ("be", "it", "pt")` |
| configs/regions/scorecard/README.md | Explains scorecard configs, the superseded/<date>/ layout and re-run command | CONTEXT.md (scorecard config term); .claude/skills/findings-doc, provenance-guard (directory) | keep | Corrected 2026-09-13 in ef5d499 Point the canonical scorecard names at the rows standing today |
| configs/regions/scorecard/ar_k10.toml | Scorecard row AR: k10, [10], fixed+season | T-sc; BB; PL; findings: scorecard.md | keep | 1dfb786 2026-09-11 Commit the configuration behind every scorecard row; BB maps "AR": "ar_k10" |
| configs/regions/scorecard/au_nem_k45.toml | Scorecard row AU-NEM: k45, season | T-sc; BB; PL; findings: scorecard.md | keep | 1dfb786 2026-09-11; BB "AU-NEM": "au_nem_k45" |
| configs/regions/scorecard/be_country.toml | Scorecard row BE: N [1, 3], EU_2026-09, derived | T-sc; T-eu; BB; PL; findings: scorecard.md, method-eu-rerun | keep | ef5d499 2026-09-13 took canonical name; T-eu asserts derived roughness |
| configs/regions/scorecard/br_k60.toml | Scorecard row BR: k60 | T-sc; BB; PL; findings: scorecard.md, method-physics-informed-rerun-prereg.md:46 | keep | 1dfb786 2026-09-11 |
| configs/regions/scorecard/cl_k10.toml | Scorecard row CL: k10 (degenerate-fit row) | T-sc; BB; PL; findings: scorecard.md | keep | 1dfb786 2026-09-11 |
| configs/regions/scorecard/de_k100.toml | Scorecard row DE: k100 fixed, EU_2026-09, derived | T-sc; T-eu; BB; PL; findings: scorecard.md | keep | ef5d499 2026-09-13; cited 6 times in findings |
| configs/regions/scorecard/dk_k100.toml | Scorecard row DK: k100 season, derived, allow_extrapolation | T-sc; T-eu; tests/test_roughness_treatment.py:93; BB; PL; findings: scorecard.md, method-roughness-treatment-prereg.md:39 | keep | Read as base text by test_roughness_treatment.py:93; ef5d499 2026-09-13 |
| configs/regions/scorecard/es_country.toml | Scorecard row ES: N [1, 4], EU_2026-09, derived | T-sc; T-eu; BB; PL; findings: scorecard.md | keep | ef5d499 2026-09-13 |
| configs/regions/scorecard/fr_country.toml | Scorecard row FR: N [1, 10], EU_2026-09, derived | T-sc; T-eu; tests/test_era5_extent.py:108,113; BB; PL; findings: scorecard.md, method-roughness-treatment-prereg.md:41 | keep | test_era5_extent.py:113 asserts allow_extrapolation False on it |
| configs/regions/scorecard/ie_country.toml | Scorecard row IE: N [1, 3], train 2017-2021 | T-sc; T-eu; BB; PL; findings: scorecard.md; scorecard/README re-run example | keep | ef5d499 2026-09-13 |
| configs/regions/scorecard/it_country.toml | Scorecard row IT: N [1, 3] | T-sc; T-eu; BB; PL; findings: scorecard.md | keep | ef5d499 2026-09-13 |
| configs/regions/scorecard/no_country.toml | Scorecard row NO: N [1, 4] (correction does not help) | T-sc; T-eu; BB; PL; findings: scorecard.md | keep | ef5d499 2026-09-13 |
| configs/regions/scorecard/nz_k7.toml | Scorecard row NZ: k7 fixed | T-sc; BB; PL; findings: scorecard.md | keep | 1dfb786 2026-09-11 |
| configs/regions/scorecard/pt_country.toml | Scorecard row PT: N [1, 2] | T-sc; T-eu; BB; PL; findings: scorecard.md | keep | ef5d499 2026-09-13 |
| configs/regions/scorecard/se_country.toml | Scorecard row SE: N [1, 4] | T-sc; T-eu; BB; PL; findings: scorecard.md | keep | ef5d499 2026-09-13 |
| configs/regions/scorecard/superseded/2026-09-13/be_country.toml | Superseded BE row config (era5/EU, stored roughness) | T-sc (loads); findings: scorecard.md:504, method-eu-rerun-prereg.md:55 | keep | Moved unchanged in ef5d499; diff vs current: only path/file_tag EU vs EU_2026-09 and added roughness = "derived" |
| configs/regions/scorecard/superseded/2026-09-13/de_k100.toml | Superseded DE row config | as be | keep | Same three-line diff as be |
| configs/regions/scorecard/superseded/2026-09-13/dk_k100.toml | Superseded DK row config | as be; base of study/dk_k100_* | keep | Diff vs current adds roughness and allow_extrapolation = true; R0/R1 study configs equal this plus stated lines |
| configs/regions/scorecard/superseded/2026-09-13/es_country.toml | Superseded ES row config | as be | keep | Same three-line diff |
| configs/regions/scorecard/superseded/2026-09-13/fr_country.toml | Superseded FR row config; R0 of the roughness study ran from it | as be; findings: method-roughness-treatment-prereg.md:41 (by its then-canonical path) | keep | Same three-line diff; roughness prereg names scorecard/fr_country.toml, which held this content on 2026-09-12 |
| configs/regions/scorecard/superseded/2026-09-13/ie_country.toml | Superseded IE row config | as be | keep | Same three-line diff |
| configs/regions/scorecard/superseded/2026-09-13/it_country.toml | Superseded IT row config | as be | keep | Same three-line diff |
| configs/regions/scorecard/superseded/2026-09-13/no_country.toml | Superseded NO row config | as be | keep | Same three-line diff |
| configs/regions/scorecard/superseded/2026-09-13/pt_country.toml | Superseded PT row config | as be | keep | Same three-line diff |
| configs/regions/scorecard/superseded/2026-09-13/se_country.toml | Superseded SE row config | as be | keep | Same three-line diff |
| configs/regions/scorecard/superseded/2026-09-13/uk_k50.toml | Superseded UK row config | as be | keep | Same three-line diff |
| configs/regions/scorecard/uk_k50.toml | Scorecard row UK: k50 fixed, EU_2026-09, derived | T-sc; T-eu; BB; PL; findings: scorecard.md | keep | ef5d499 2026-09-13 |
| configs/regions/scorecard/us_k250.toml | Scorecard row US: k250 (degenerate-fit row) | T-sc; BB; PL; findings: scorecard.md, method-physics-informed-rerun-prereg.md:46 | keep | 1dfb786 2026-09-11 |
| configs/regions/se.toml | Maintained config SE: country, [1, 4], train 2015-2021, test 2023 | T-all; T-country; se_zonal.toml comment | keep | Baseline national path that se_zonal.toml compares against (se_zonal.toml:11-12) |
| configs/regions/se_zonal.toml | Variant SE-BZ: country, `entsoe-zonal`, [1, 4], train 2015-2019 | T-all (count, comment :132); docs/guides/data-sources.md:53 (by code); findings: method-country-level.md (results, not by path) | keep | Zonal-fit rows in docs/findings/method-country-level.md:373-378 ("trained 2015-2019"); runs output/validation/rq4_realzones_2026-07-24/SE-BZ. dd87232 2026-08-12 |
| configs/regions/study/dk_k100_derived.toml | R1 condition (DK) of the roughness-treatment study | findings: method-roughness-treatment-prereg.md:262; run output/roughness_treatment_2026-09-12/R1/DK | keep | Cited by path in dated record. 3ee5319 2026-09-12 Add the roughness treatment switch; 2a6da49 registered deviation D2 |
| configs/regions/study/dk_k100_stored.toml | R0 condition (DK) of the roughness-treatment study | findings: method-roughness-treatment-prereg.md:260; run output/roughness_treatment_2026-09-12/R0/DK | keep | Cited by path; 2a6da49 2026-09-12 Register two deviations before the DK roughness comparison runs |
| configs/regions/study/fr_country_derived.toml | R1 condition (FR) of the roughness-treatment study | header names method-roughness-treatment-prereg.md; run output/roughness_treatment_2026-09-12/R1/FR; results method-roughness-treatment.md:107-110 | keep | Not cited by path (grep of docs/findings: 0 hits), but is the reproduction record of FR R1. 3ee5319 2026-09-12 |
| configs/regions/study/no_country_oldfiles_derived.toml | Condition B (NO) of the EU re-run: old files, derived roughness | header names method-eu-rerun-prereg.md (:143 condition B); run output/eu_rerun_2026-09-12/oldfiles_derived/NO; scripts/analysis/eu_rerun_compare.py label | keep | eu_rerun_compare.py:19,43 `oldfiles_derived=<dir>`; db772d7 2026-09-12 Add the thirteen configurations the European re-run needs |
| configs/regions/study/se_country_oldfiles_derived.toml | Condition B (SE) of the EU re-run | as NO; run output/eu_rerun_2026-09-12/oldfiles_derived/SE | keep | As NO; db772d7 2026-09-12 |
| configs/regions/uk.toml | Maintained config UK: turbine (farms, pseudo-replicated), [1, 50], test 2019 | T-all; test_harness_regions.py:141,170; test_harness_provenance.py:68; validate_region.py; cluster_selection_study.py; pivot_probe.py | keep | tests/test_harness_provenance.py:68 loads by name; station_id_regex pinned by test_harness_regions.py:141-144 |
| configs/regions/us.toml | Maintained config US: turbine, `eia-us`, [1, 25], era5/US_daily, test 2022 | T-all; test_harness_regions.py:146; docs/runbooks/us.md | keep | `shipped("us")` pins obs_unit plant at test_harness_regions.py:146-148 |

### docs/ (outside findings) (30 files)

| path | one-line purpose | referenced by | verdict | evidence |
|---|---|---|---|---|
| docs/README.md | Docs folder conventions, naming, procedural rules, scorecard markers | README.md:207; docs/index.md; skills; AGENTS.md:10 | keep | Excluded from the site (conf.py:37). |
| docs/api.md | autodoc API reference | docs/index.md:27,76 | keep | Gap: no `vwf.harness`, `vwf.geospatial`, and only 4 of 14 `vwf.sources` modules (base, european, in_memory, registry). Opens with the legacy `PyVWF` class as "The entry point" (:11). |
| docs/conf.py | Sphinx config, excludes README.md and findings/** | CI docs job; .readthedocs.yaml | keep | Does not exclude design/manuscript-chapters-45.md, which is in no toctree (section B). |
| docs/design/harness.md | Explanation: why the harness seams are where they are | README.md:216; training.md:7; index.md toctree | keep | Stale driver block :192-199: flags before the subcommand, a nonexistent `--factors-from` (real: `--source-region`, `--source-run`, validate_region.py:38-39), and a "config hash" in run-id that driver.py:184 does not add. |
| docs/design/manuscript-chapters-45.md | Working plan: decisions for merging thesis chapters 4 and 5 | 9 findings docs; src/vwf/extensions/grid/evaluate.py; STATUS.md | keep | Keep tracked (9 dated findings cite it) but exclude from the site. docs/README.md:102-104 says design/ has "no assumed context"; this cites commit hashes and "Spiral". |
| docs/design/roughness-temporal-treatment.md | Explanation: three roughness routes, two treatments | index.md toctree; CONTEXT.md roughness route row; configs/regions/*.toml; src/vwf/datasets/combine_era5_files.py | keep | ecf0cc3 "Say the same thing about the roughness treatment in every document". |
| docs/design/undefined-roughness-in-complex-terrain.md | Explanation: where shear-derived z0 is undefined | index.md toctree; roughness-temporal-treatment.md | keep | dce5323 "Record the two roughness facts in docs/design". |
| docs/guides/adding-an-observation-source.md | How-to plus reference: 15-file region table, curve assignment, adapter contract, built-in adapters | README.md:213; training.md:98; data-sources.md; new-region skill; docs/api.md; src/vwf/__init__.py, sources/__init__.py, sources/base.py; findings: dataset-survey.md | keep | Its filename uses a rejected term: CONTEXT.md lists "observation source" under Do not use for adapter and data source; its own H1 is "Adding a region and its adapter". A rename breaks 1 findings citation. |
| docs/guides/data-sources.md | Reference: data sources, licences, input layout, processing per region, country-level workflow | README.md:69,209; index.md; api.md; new-region and provenance-guard skills; scripts/region_tools/weight_country_grid_points.py | keep | :13 says only the open library is committed under input/, but README.md and shapes are too (:158 concedes shapes). :48 AU-NEM has no runbook. :154-156 gives both curve-library routes (section A). |
| docs/guides/output-structure.md | Reference: harness run directory contents | README.md:211; index.md; configs/regions/scorecard/README.md; tests/test_scorecard_configs.py; CHANGELOG.md | keep | Covers `output/validation/` only; the legacy `output/runs/` layout is in PIPELINE.md:58 alone. |
| docs/guides/training.md | How-to: region config, train, evaluate, transfer | README.md:98,210; AGENTS.md:15; new-region skill; index.md | keep | Uses rejected terms: "training window" and "correction factors" (:5), "open bundled library" (:58). Has no legacy section, so it is the proposed home for PIPELINE.md. |
| docs/guides/visualisation.md | How-to: `vwf.viz` figures | README.md:102,212; index.md; CHANGELOG.md | keep | `load_results` reads only the legacy layout (`results/capacity-factor`, src/vwf/viz/distribution.py:185-187), not harness runs. The guide's "any PyVWF run directory" (:8) is wrong for harness output. |
| docs/guides/your-own-data.md | How-to: run the correction on a client CSV fleet with no adapter | README.md:214; index.md; src/vwf/sources/client_csv.py | keep | Writes to `output/my_run` (:66), outside `output/validation/`. |
| docs/img/viz_distribution.png | Figure for the visualisation guide | docs/guides/visualisation.md; examples/viz_demo.py | keep | Tracked through `!docs/img/*.png` (.gitignore:73). |
| docs/img/viz_error_vs_clusters.png | Figure | visualisation.md; examples/viz_demo.py | keep | As above. |
| docs/img/viz_factor_joint.png | Figure | visualisation.md; examples/viz_demo.py | keep | As above. |
| docs/img/viz_factor_map.png | Figure, README hero image | README.md:105; visualisation.md; examples/viz_demo.py | keep | As above. |
| docs/img/viz_qq.png | QQ figure written by viz_demo | examples/viz_demo.py | keep | No doc embeds it (visualisation.md:12 names QQ but shows only the distribution image). Regenerated by viz_demo, so harmless. |
| docs/img/viz_sim_vs_obs.png | Figure | visualisation.md; examples/viz_demo.py | keep | As above. |
| docs/index.md | Site front page and toctree (guides, runbooks, design, API), citation block | docs/README.md:6-7 ("canonical contents list"); Sphinx root | keep | :91-92 method citation omits Bhaskaran and Staffell (CITATION.cff:50-60, README.md:266 list six authors); :10 "linear correction" is a rejected term; toctree omits design/manuscript-chapters-45.md (section B). |
| docs/runbooks/ar.md | How-to: Argentina (CAMMESA) acquisition, capacity join, run | data-sources.md; index.md; src/vwf/sources/cammesa_ar.py; tests/test_cammesa_ar_processing.py; STATUS.md | keep | :56-58 sets PYVWF_INPUT on train only, not evaluate (section A). |
| docs/runbooks/br.md | How-to: Brazil (ONS) | data-sources.md; index.md | keep | Sets no PYVWF_INPUT (:93). |
| docs/runbooks/cl.md | How-to: Chile (CEN) | data-sources.md; index.md; scripts/fetch/cen_cl.py; src/vwf/{datasets,sources}/cen_cl.py; tests/test_cen_cl_processing.py; findings (1) | keep | :38-40 sets PYVWF_INPUT on train only. |
| docs/runbooks/de.md | How-to: Germany (confidential WindStats) staging | adding-an-observation-source.md; data-sources.md; index.md; scripts/README.md | keep | 33 lines, no run section. |
| docs/runbooks/dk.md | How-to: Denmark (DEA register) | adding-an-observation-source.md; data-sources.md; index.md | keep | :61-63 sets PYVWF_INPUT on train only. |
| docs/runbooks/es.md | How-to: Spain historical WindStats (ES-WS) | configs/regions/es_ws.toml; adding-an-observation-source.md; data-sources.md; index.md; src/vwf/sources/windstats.py; tests/test_windstats_processing.py | keep | :38 `fetch/era5.py --region es-ws` exits: era5.py:169 builds `es-ws.toml`, which does not exist (`es_ws` would work). :39 `PYVWF_INPUT=input` is the default. |
| docs/runbooks/nz.md | How-to: New Zealand (EMI), the template region | new-region skill; data-sources.md; index.md; src/vwf/sources/emi_nz.py; tests/test_emi_nz_processing.py | keep | The only runbook stating "Set PYVWF_INPUT on both commands" (:106). |
| docs/runbooks/tr.md | Record: Turkey evaluated, not shipped; what to check if access is obtained | data-sources.md:209; index.md; scripts/fetch/epias_tr.py; findings (1) | keep | Title "evaluated, not shipped": a feasibility record in how-to form (section B). |
| docs/runbooks/uk.md | How-to: UK (REPD plus Ofgem ROC) | adding-an-observation-source.md; data-sources.md; index.md; scripts/fetch/uk.py; scripts/process/uk.py | keep | No run section. |
| docs/runbooks/us.md | How-to: US (EIA) | data-sources.md; index.md | keep | :91-92 gives a third curve route, "supply ... power_curves.real.csv and pass its key". Nothing in src reads `*.real.csv` (grep). |

### examples/ (15 files)

| path | one-line purpose | referenced by | verdict | evidence |
|---|---|---|---|---|
| examples/data/README.md | Describes the synthetic example data set | examples/run_minimal.py (docstring) | keep | Documents that `run_minimal.py` data is fabricated; needed for the JOSS-facing example. |
| examples/data/au_nem/README.md | Describes bundled derived AU-NEM data and its builders | examples/notebooks/au_nem_validation.ipynb | keep | Names builders `scripts/process/aemo_au.py`, `assign_au_curves.py`; attribution for AEMO and GWPT. |
| examples/data/au_nem/au_nem_capacity_mask.csv | Derived AU farm-month capacity mask for the notebook | au_nem_validation.ipynb; au README | keep | Only consumer is the notebook; tests write their own file of that name in tmp dirs (tests/test_aemo_au_processing.py:337). |
| examples/data/au_nem/au_nem_md_open.csv | AU farm metadata with open-library curves | au_nem_validation.ipynb; au README | keep | Output of assign_au_curves.py:117; the notebook's open-stack fleet. |
| examples/data/au_nem/au_nem_scada_monthly_partials.csv | Monthly AEMO SCADA aggregates 2020-2023 | au_nem_validation.ipynb; au README | keep | Lets the notebook skip raw AEMO archives; read by `AEMONemSource` fast path (src/vwf/sources/aemo.py:316). |
| examples/data/era5/era5_example.nc | Synthetic ERA5-shaped NetCDF (569 KB) | run_minimal.py; generate_example_data.py; .gitignore:70 | keep | Force-included by `.gitignore:70` over the `*.nc` ignore; CI example input. |
| examples/data/generate_example_data.py | Regenerates the synthetic example data | examples/data/README.md; run_minimal.py docstring | keep | Only way to rebuild the three example files; reads tracked `input/reference/power_curves.csv`. |
| examples/data/observations_example.csv | Synthetic per-cluster observed CF with known bias | run_minimal.py; data README | keep | CI input (`ci.yml:69`). |
| examples/data/turbines_example.csv | Six synthetic control points | run_minimal.py; data README | keep | CI input (`ci.yml:69`). |
| examples/dk_raw_vs_corrected.ipynb | DK onshore train/evaluate walk-through via harness | none | unclear | No incoming reference anywhere. Intro claims "only open data" and the open library, but cell 1 defaults `PYVWF_INPUT=input/combined` (licensed). Need: is it the intended open DK demo (then fix) or superseded by README quickstart? |
| examples/notebooks/au_nem_validation.ipynb | AU-NEM seasonal validation on the open stack | examples/data/au_nem/README.md | keep | Named deliverable in method-generalisation.md:163 ("validation notebook that runs on the open stack"); commit 8da53a6. |
| examples/notebooks/northsea_data.ipynb | Legacy FR/NL/NO/BE country data preparation | none | unclear | No incoming references; self-described as not runnable; uses legacy `input/country-data/`. Need: does any current country-level run or thesis reproduction still read its outputs? If not, delete. |
| examples/quick_run.py | Thin wrapper over `vwf.cli.train.main` | src/vwf/cli/train.py:3 | keep | Docstring of the console entry point cites it; 14 lines; needs private country data and ERA5. |
| examples/run_minimal.py | End-to-end synthetic example with assert | CI ci.yml:69; Dockerfile:86; README; CONTRIBUTING; docs/index.md | keep | Executed in CI and as the Docker CMD; offline. |
| examples/viz_demo.py | Renders vwf.viz figures on synthetic data | README; docs/guides/visualisation.md | keep | Offline, but writes six TRACKED files `docs/img/viz_*.png` (viz_demo.py:66), so running it dirties the tree. |

### input/ (6 files)

| path | one-line purpose | referenced by | verdict | evidence |
|---|---|---|---|---|
| input/README.md | Open curve library, the licensed-library swap, input/ layout | tests/test_committed_files.py:28; src/vwf/resources/__init__.py:3; examples/data/README.md:19; examples/run_minimal.py; .gitignore:87 | keep | Force-added (allowlist, test_committed_files.py:27-34). :44-47 tells users to `cp` the licensed library over power_curves.csv. test_committed_files.py:38-39 calls this "the mistake input/README.md walks a user towards" (section A). |
| input/reference/models.csv | Open library model catalogue | PyVWFPaths.TURBINE_MODELS (config.py:43); tests/test_committed_files.py; tests/test_curve_library.py | keep | `cmp` identical to src/vwf/resources/models.csv; sha256 pinned (test_committed_files.py:45). |
| input/reference/power_curves.csv | Open library power curves | config.py:42; test_committed_files.py; test_curve_library.py:26-37 | keep | `cmp` identical to src/vwf/resources/power_curves.csv; sha256 pinned. |
| input/reference/power_curves_provenance.csv | Per-column source and licence of the open library | test_committed_files.py; test_curve_library.py:65,87; scripts/analysis/curve_match_audit.py:154 (reads the resources copy) | keep | `cmp` identical to src/vwf/resources/power_curves_provenance.csv; sha256 pinned. |
| input/reference/shapes/country_shapes.geojson | Onshore region outlines for clustering | config.py:47 (COUNTRY_SHAPES); src/vwf/clustering.py:157,205; 4 scripts/analysis studies; export_voronoi_frames.py:72 | keep | Allowlisted in test_committed_files.py:32. Not in the wheel (pyproject.toml:129-130, "13 MB ... deliberately not shipped"). |
| input/reference/shapes/offshore_shapes.geojson | Offshore outlines for clustering | config.py:48; clustering.py:135; same studies | keep | Allowlisted; not in the wheel. |

### scripts/ (92 files)

| path | one-line purpose | referenced by | verdict | evidence |
|---|---|---|---|---|
| scripts/README.md | Layout and typical-run guide for scripts/ | none (no markdown links to it; refs.json empty) | keep | Stale: lists 7 of 45 analysis scripts, omits scripts/pinn/ (21 files) and region_tools/apply_turbine_specs.py, names `ES.md` (file is `es.md`). Needs updating, not removing. |
| scripts/analysis/audit_country_observations.py | Run the `vwf.loaders.country_obs_checks` gates over every country-level region config, train and test | docs/guides/data-sources.md; scripts/README.md:37; scripts/region_tools/repair_country_capacity.py; findings: method-national-single-cluster-prereg.md | keep | General argparse tool over `configs/regions/*.toml`, not tied to one study; thin wrapper over `check_country_cf`. b578bd2 (2026-09-15). |
| scripts/analysis/baseline_bootstrap.py | Paired bootstrap CIs for one scorecard row from 2026-09-11 backfill frames; also the de facto scorecard-row registry (`CONFIGS`, `REPORTED`) | 10 scripts (section D); findings: scorecard.md:26; CHANGELOG.md:167 | keep | Path cited in scorecard.md:26 as evidence for the 2026-09-11 correction notice; imported by 10 scripts. Its library parts should be promoted (section D) and this file left as the record. |
| scripts/analysis/chapter_capacity_weights.py | Rebuild thesis chapter 4 country-grid capacity weights from GWPT (radius sum), with wrong-parameter controls | findings: method-country-level.md:186; docs/design/manuscript-chapters-45.md:100 | move to scripts/studies/country-level/chapter_capacity_weights.py | One commit (bff4467), one findings doc. Path is cited in method-country-level.md:186, so moving stales that dated command. |
| scripts/analysis/cluster_selection_gaps.py | Train and score the counts the one-SE rule passed over, per row, on the test year | none by name; produces the gap table in findings: method-cluster-selection.md:83-93 | move to scripts/studies/cluster-selection/cluster_selection_gaps.py | 1ae99a3 "score the counts the rule passed over". Loads cluster_selection_study by file path (l.30-31); move both together. No path citation, so the move is clean. |
| scripts/analysis/cluster_selection_study.py | Registered driver: forward-chaining folds, one-SE rule, final refit against B1 and B2 | tests/test_cluster_selection_study.py; scripts/analysis/cluster_selection_gaps.py; scripts/analysis/national_single_cluster_study.py:46-47 | move to scripts/studies/cluster-selection/cluster_selection_study.py | Docstring names method-cluster-selection-prereg.md; findings cite `output/cluster_selection_2026-09-15/`, not the path. Three importers load it by file path and need updating. |
| scripts/analysis/cluster_sweep_cost.py | Time one registered cluster sweep (train plus evaluate) to cost the selection protocol | none by name; its run is cited in findings: method-cluster-selection-prereg.md:412 and docs/design/manuscript-chapters-45.md:772 | move to scripts/studies/cluster-selection/cluster_sweep_cost.py | f7caa8c, 4af1ba4. Docstring records the finding that a country row has two candidates. The output path is cited, not the script path. |
| scripts/analysis/common_row_rescore.py | Rescore a row's backfill frames through `driver._score_on_common_rows`, compare with metrics.csv; `--joint` across runs; common-row bootstrap | findings: scorecard.md:92; method-scalar-bounds.md | keep | Reproduction record for the CL, UK and NZ notices (docstring l.46-48), cited by two findings docs, so not one study. Uses private driver API (`_score_on_common_rows`, `_country_pairs`, `_tidy_eval_frame`). |
| scripts/analysis/correction_identifiability.py | Scalar and offset collinearity, pencil pivot speed and CV at reference winds, from the 1,729-point pool | findings: method-correction-identifiability.md:44 | move to scripts/studies/correction-identifiability/correction_identifiability.py | 074f648 adds only this file. Path cited at findings:44 (a move stales it). `pivot()` is duplicated in scripts/analysis/pivot_probe.py:72. |
| scripts/analysis/curve_library_assign.py | Library: T2 rule, move same-brand units to the nearest in-band other-brand model | scripts/analysis/curve_library_tables.py:49; tests/test_curve_library_assign.py | move to scripts/studies/curve-library/curve_library_assign.py | Study-only rule "fixed before any T2 run" (34d9c3a). No findings path citation. Imports curve_match_audit via sibling `sys.path` insert (l.34), which the move must keep working. |
| scripts/analysis/curve_library_match.py | Library: exact normalised match of a register designation to a licensed-library model key (T1) | scripts/analysis/curve_library_tables.py:50; tests/test_curve_library_match.py; findings: method-curve-library-prereg.md; scorecard.md:356 | move to scripts/studies/curve-library/curve_library_match.py | T1-only (b7ed5fd); no src/vwf equivalent. Path cited in scorecard.md:356 and the prereg, so a move stales two records. |
| scripts/analysis/curve_library_study.py | Registered driver: one condition of one row; override applied in `prep_country`, refusal if not applied | tests/test_curve_library_study.py; findings: method-curve-library-prereg.md:409 | move to scripts/studies/curve-library/curve_library_study.py | Prereg "Tooling this needs" names the path. Docstring usage (l.71) shows 3 args but `main()` requires 4 (`overrides_csv`): stale. `variant_root` is called only by tests, not by `main`. |
| scripts/analysis/curve_library_tables.py | Build C1, C2, T1, T2 override tables over the union of both fleets; report reach and distance | tests/test_curve_library_tables.py; scripts/analysis/curve_library_study.py (docstring only); findings: method-curve-library-prereg.md | move to scripts/studies/curve-library/curve_library_tables.py | 2f26208, 83ba607, 9aaa9d5, all curve-library. Cited in prereg. `train_fleet_of` takes `sorted(glob(...))[0]` (l.100), the list-and-pick pattern AGENTS.md warns against. |
| scripts/analysis/curve_match_audit.py | Scorecard audit: each unit's own manufacturer against its assigned curve's (same, other brand, reference, unverifiable) | CONTEXT.md; .claude/skills/new-region, findings-doc; docs/guides/adding-an-observation-source.md; CHANGELOG.md:97; findings: scorecard.md:328; curve_library_assign.py; curve_library_tables.py | keep | CONTEXT.md defines "curve-match audit" as this path; feeds the scorecard's Other brand column; cross-region tool (1166ece). `RUNS` hardcoded to refresh_2026-08-24 (l.86-91). |
| scripts/analysis/domain_split_study.py | S0, S1, S2 kriged correction surfaces (undivided vs split pool) scored on 14 chapter configurations | docs/design/manuscript-chapters-45.md:773,781 | move to scripts/studies/domain-split/domain_split_study.py | Registered driver of method-domain-split-prereg.md (docstring l.3). The study is void (prereg l.9-13) but S1 and S2 were run, so this is still the record. 9e0f73d, f305b28. |
| scripts/analysis/era5_overlap_check.py | Gate G0: compare u10, v10, u100, v100 between era5/EU and era5/EU_2026-09 on a fixed sample | findings: method-eu-rerun.md:35 | move to scripts/studies/eu-rerun/era5_overlap_check.py | 3898bdc "committed before it runs"; sample fixed by method-eu-rerun-prereg.md. Path cited in method-eu-rerun.md:35. |
| scripts/analysis/eu_rerun_compare.py | Paired bootstrap comparison of two or three evaluate runs of one row, on common rows | tests/test_eu_rerun_compare.py; used as the curve library study's comparison step (docstring l.7-8, usage l.45-47) | keep | Serves two studies (eu-rerun, curve-library) and is tested; conditions are named on the command line. The name is study-specific, the use is not. Imports roughness_treatment_study (a study driver) for `_frames` and `_score`. |
| scripts/analysis/evaluate_all_pyvwf_runs.py | Legacy: metrics across `output/runs/<prefix>` run dirs via `vwf.metrics.overall_error` | PIPELINE.md:63,81,93; docs/guides/visualisation.md:86; scripts/README.md:35; src/vwf/viz/evaluation.py:10,63 | keep | Its CSV schema is consumed by `vwf.viz.evaluation.plot_error_vs_clusters`. Legacy path only. `_country_level_metrics` duplicates `driver._country_pairs` plus `_error_metrics` (driver.py:584-712). |
| scripts/analysis/export_correction_field.py | CLI over `vwf.harness.export.export_correction_field` (gridded factor NetCDF) | docs/guides/your-own-data.md:74; src/vwf/harness/export.py (function name) | keep | Thin reusable argparse CLI. Docstring l.2 says "(offer 103 artifact)": commercial framing CLAUDE.md forbids in this public repo; should be scrubbed (also run_hindcast.py:2, outside this range). |
| scripts/analysis/export_voronoi_frames.py | Export per-k Voronoi cell geometry and factor colours to .npz for TouchDesigner | none | unclear | Zero referrers (git grep, both ways); only commit bca93e4. Imports `mapbox_earcut`, not declared in pyproject.toml. Need to know: is the TouchDesigner visualisation still in use? If not, delete; `vwf.viz.factors` covers static Voronoi maps only. |
| scripts/analysis/extent_audit.py | Every scorecard row's test fleet against its bbox-sliced loaded ERA5 extent | CHANGELOG.md:249; findings: scorecard.md:216,431; method-eu-rerun-prereg.md | keep | Row-generic audit over `bb.CONFIGS`, thin over `vwf.wind.loaded_extent_coverage`; path cited in scorecard (e797589). |
| scripts/analysis/hourly_resolution_test.py | CL 2024: apply the trained k=10 correction to hourly ERA5; score hourly, daily, monthly against CEN; gates G1 to G3 | findings: method-hourly-resolution.md | move to scripts/studies/hourly-resolution/hourly_resolution_test.py | One findings doc; path cited there (a move stales it). No argv; writes fixed `output/hourly_test/` (d2017f3). |
| scripts/analysis/loco_interpolation.py | Registered: leave-one-country-out scores for IDW, kriging, nearest, RBF on the 1,729-point pool | scripts/analysis/loco_reference_wind.py:40-43 (file-path import) | move to scripts/studies/loco-interpolation/loco_interpolation.py | Docstring names method-loco-interpolation-prereg.md; 4a1fd79 "commit its runner before it runs". Findings cite `output/loco_2026-09-13/`, not the path. |
| scripts/analysis/loco_reference_wind.py | Leave-one-country-out scored on corrected speed at 8 and 12 m/s, against the coefficient route | none by name; output cited in findings: method-correction-identifiability.md:259; output read by scripts/analysis/regime_coverage.py:44 | move to scripts/studies/correction-identifiability/loco_reference_wind.py | a284d4d; its data is the "holdout result" section of method-correction-identifiability.md. Loads loco_interpolation by file path, so moving either needs that path updated. |
| scripts/analysis/min_cluster_size_tradeoff.py | Chile k=10: does `min_cluster_size` 1/3/5 cap the worst scalar, and at what RMSE cost (gates G1 to G3) | findings: method-scalar-bounds.md L154 (path), L155 data `output/min_cluster_size/cl_tradeoff.csv` | move to scripts/studies/scalar-bounds/min_cluster_size_tradeoff.py | Driver of one finding; output exists. Path cited in method-scalar-bounds.md L154, so a move makes that command stale. Commit 569a00a "guard against tiny clusters". |
| scripts/analysis/missing_value_audit.py | Classify every missing corrected CF of a scorecard row by route (input, failed_factor, above/below curve) | findings: scorecard.md L91 (path); off_curve_sensitivity.py docstring | move to scripts/studies/scorecard/missing_value_audit.py | Scorecard 2026-09-11 notice driver; data `output/curve_library_study_2026-09-11/missing_value_audit/`. Imports `baseline_bootstrap` by module name, so must move with it. Path cited in scorecard.md. |
| scripts/analysis/ml_transfer_expanded.py | ML transfer round two: LORO on 5 vs 8 regions (NZ/CL/AR added), gates G1 to G3, capped variant | findings: method-ml-transfer.md L146, L191 (path); its output cited by method-loco-interpolation-prereg.md L23 | move to scripts/studies/ml-transfer/ml_transfer_expanded.py | Driver of round two only. Path cited in method-ml-transfer.md L191. On disk only `expanded_loro_scalar.csv` (Jul 24), older than first commit 3dd8498 (Aug 12); capped CSV absent. |
| scripts/analysis/ml_transfer_retest.py | ML transfer round one: centroids, ETOPO terrain features, RF LORO/CV, variance split | findings: method-ml-transfer.md L14, L192; method-physics-informed-rerun-prereg.md L65; imported by scripts/pinn/{d0,d1,d2,d3,d4,e1_loro,prep_rf_features}.py and analysis/{ml_transfer_expanded,pool_as_training_set,regime_coverage}.py; scripts/README.md | promote to src/vwf/extensions/ml | Ten importers treat it as a library (`from analysis.ml_transfer_retest import ...`, importlib by path). pyproject L88 already names `vwf.extensions.ml`; STATUS L97 queues that port as phase 3. See section A. |
| scripts/analysis/national_single_cluster_study.py | Country-level: one national cluster vs the grid's own count, forward chaining + one-SE rule | none (prereg does not name it back) | move to scripts/studies/national-single-cluster/national_single_cluster_study.py | Docstring names method-national-single-cluster-prereg.md; a5c51d6 "committed before it runs". Prereg "Blocked on data, 2026-09-16" (eecf252); `output/national_single_cluster_2026-09-16/` empty. Live, awaiting unblock. |
| scripts/analysis/off_curve_sensitivity.py | Rescore a scorecard row with off-curve corrected (and uncorrected) days as zero output | findings: scorecard.md L92 (basename) | move to scripts/studies/scorecard/off_curve_sensitivity.py | Data `output/curve_library_study_2026-09-11/off_curve_sensitivity/` cited in scorecard.md L93-94. Commits b7826d3, 56a78d2. Imports `baseline_bootstrap` by module name. |
| scripts/analysis/offshore_pool_study.py | Offshore pool P0 (declared mode) vs P1 (shape classification) kriging on DK/UK rows | none | move to scripts/studies/offshore-pool/offshore_pool_study.py | Names method-offshore-pool-prereg.md; bd09937. Prereg VOID 2026-09-13 (e955e17) but "its numbers are kept, in output/offshore_pool_2026-09-13/" (exists). Only generator of those kept numbers. |
| scripts/analysis/pivot_probe.py | Is the ~4 m/s pencil pivot physics or the offset search's initial step? Re-solve offsets from 6 steps x 2 caps | findings: method-correction-identifiability.md L117-118 (path) | move to scripts/studies/correction-identifiability/pivot_probe.py | Output `output/pivot_probe_2026-09-16/` exists and is cited. Commits d69ece8, 1051a6c. Path cited, so a move makes that record stale. |
| scripts/analysis/pool_as_training_set.py | Control-point pool as supervised training set: DK 884/200/100 feature coverage, label quality | none by name | move to scripts/studies/why-corrections-do-not-transfer/pool_as_training_set.py | Its output `output/pool_training_2026-09-16/dk_onshore_coverage.csv` (340 vs 114 cells, 2.6 vs 1.75) is reported unnamed in method-why-corrections-do-not-transfer.md elimination 1 (L31-37). Commits a0f4ea5, 8c7e556, 6b0afc3. |
| scripts/analysis/refit_control_points.py | Refit BE (control), IT, PT control points on era5/EU_2026-09 and screen scalar bounds | none by name | move to scripts/studies/manuscript-chapters-45/refit_control_points.py | 3930f26, clean manifest at 3930f26. BE cluster 0 scalar 0.598 and IT scalars 4.07/3.01 in output match docs/design/manuscript-chapters-45.md L791-803. No findings doc. Aggregate `refit_control_points.csv` absent. |
| scripts/analysis/regime_coverage.py | Pool feature-space coverage, candidate regions, and fold distance vs LOCO error | findings: method-why-corrections-do-not-transfer.md L83-84 (path) | move to scripts/studies/why-corrections-do-not-transfer/regime_coverage.py | Output `output/regime_coverage_2026-09-16/` exists, cited. refs.json hit from scripts/pinn/d5_regime_coverage.py is a stem collision (d5's own filename, L26). 8a7bcb8. |
| scripts/analysis/regression_compare.py | Frame-level diff of factors_*/cor_cf_* between two run dirs, pass/fail at atol | tests/test_regression_compare.py (loads by path); findings: method-harness-regression.md L33, L101; scripts/README.md; both runners' docstrings | keep | Generic, tested comparator; path cited by findings and by a test via `spec_from_file_location`. See section B. |
| scripts/analysis/regression_run_harness.py | D1 harness half: build RegionSpec from args, run_train + run_evaluate, flatten frames | findings: method-harness-regression.md L100; scripts/README.md | keep | Must run under the branch's PYTHONPATH while the legacy half runs under a main worktree's; method doc says "Each runner's docstring carries its invocation". 120f5aa. See section B. |
| scripts/analysis/regression_run_legacy.py | D1 legacy half: PyVWF.train(dask_n_workers=0) + simulate_cf, flatten frames | findings: method-harness-regression.md L99; scripts/README.md | keep | Reference half of D1, run from a main worktree at 53c4330. `--train-start/--train-end` only reach the country path (L58-59), not PyVWF(). See section B. |
| scripts/analysis/roughness_treatment_study.py | Registered R1 vs R0 roughness comparison per scorecard row, paired bootstrap, gates G1/G2 | scripts/analysis/eu_rerun_compare.py L58 (`import roughness_treatment_study as rts`, uses `_frames`, `_score`) | move to scripts/studies/roughness-treatment/roughness_treatment_study.py | Names method-roughness-treatment-prereg.md; 1ff1d54. Output `output/roughness_treatment_2026-09-12/analysis/` cited by method-roughness-treatment.md L99. Move with eu_rerun_compare or promote `_frames`/`_score` first. |
| scripts/analysis/run_hindcast.py | CLI over vwf.harness.hindcast: monthly national CF over all ERA5 years, ranked | docs/guides/your-own-data.md L76 (path). CHANGELOG L91 and hindcast.py hits are the function name | keep | Thin argparse wrapper, a reusable tool. Docstring L2 "(offers 068 / 015)" is commercial framing CLAUDE.md forbids in this public repo; strip it. |
| scripts/analysis/surface_flag_report.py | Share of the product surface's cells the plausibility flag rejects, by distance band | none | unclear | Run (output/surface_flag_2026-09-15: 1,339 of 23,989 implausible) but reported in no doc; says it is not the distance-mask finding's surface. Need: is the product flag share meant to be recorded? See C. |
| scripts/analysis/train_all_bias_corrections.py | Legacy PyVWF batch trainer over named configuration sets, writes output/runs/<prefix>/ | PIPELINE.md L43, L46, L78, L90; scripts/README.md; src/vwf/cli/train.py L9; findings: TURBINE_GRID_EVALUATION_ANALYSIS.md L319 | keep | Produced `output/runs/turbine_grid`, which the pool and several studies read. Findings cites "lines 84-150 (turbine_grid config)"; that config is now at L161, so the citation is already stale. |
| scripts/analysis/training_objective_check.py | Country-level fit: which training days the fitted factors push off the curve, objective vs zero fill | findings: scorecard.md L185 (path); method-eu-rerun-prereg.md L167 (path) | move to scripts/studies/scorecard/training_objective_check.py | train_dir hardcoded to refresh_2026-08-24 (L54), so it cannot target the re-run fits eu-rerun-prereg L167 registers. method-eu-rerun.md G5 (L248) reports numbers with no data path. |
| scripts/analysis/unit_concentration.py | Capacity-effective units, top-unit error shares, leave-one-out gain range for a turbine-level scorecard row | findings: scorecard.md L27 (path); CHANGELOG.md L167; scripts/analysis/common_row_rescore.py L30 (docstring) | move to scripts/studies/scorecard/unit_concentration.py | Commit 4b6d143 "scripts behind the UK and NZ notice". Imports `baseline_bootstrap` and private `driver._tidy_eval_frame`. Path cited in scorecard.md L27. |
| scripts/analysis/unmasked_surface_bands.py | Undivided pool kriged with no mask: bands, variance, 8 m/s reading, two plausibility screens | findings: method-distance-mask.md L54-56 (path, "at commit bdb5f66") | move to scripts/studies/distance-mask/unmasked_surface_bands.py | Output `output/unmasked_bands_2026-09-15/` exists and cited. Doc pins bdb5f66, but the second screen was added later in 171f530, so the doc's commit pin predates its correction-notice table. |
| scripts/analysis/validate_region.py | Thin argparse CLI over vwf.harness.driver train/evaluate/transfer | README.md; AGENTS.md; .claude/skills/new-region/SKILL.md; docs/guides/training.md, adding-an-observation-source.md; docs/design/harness.md; 7 runbooks; configs/regions/scorecard/README.md; src/vwf/harness/driver.py L3; docker-compose.yml L7; scripts/README.md | keep | The primary documented entry point. Candidate for a `[project.scripts]` console entry (see F); keep the script path because 15+ procedural docs cite it. |
| scripts/era5/combine.py | Monthly ERA5 to yearly daily files with per-timestep roughness | findings: method-roughness-treatment.md:261, scorecard.md:236, region-au-nem.md:176; CONTEXT.md; configs br/us; runbooks | keep | The per-timestep roughness route C named in CONTEXT.md. Its z0 inversion (lines 66-75) duplicates prep_era5 (src/vwf/datasets/era5.py:224-247); see section C. |
| scripts/fetch/aemo_au.sh | Download AEMO MMSDM SCADA and DUDETAIL archives 2020-2023 | scripts/README.md; docs/guides/data-sources.md | keep | Only fetcher for AU observations; `bash -n` clean; years hardcoded 2020-2023. |
| scripts/fetch/cammesa_ar.py | Fetch CAMMESA renewables database (AR) | findings: dataset-survey.md:71; runbooks/ar.md; process/cammesa_ar.py | keep | Path cited in a findings doc; process script exits pointing at it. |
| scripts/fetch/cen_cl.py | Probe and fetch CEN SIP API (CL), key from env | runbooks/cl.md; configs cl.toml, scorecard/cl_k10.toml | keep | Scorecard config cites it; key read from `CEN_API_KEY` env (line 58). |
| scripts/fetch/dk.py | Download ens.dk turbine workbooks (DK) | runbooks/dk.md; guides; process/dk.py | keep | Only DK fetcher. Writes raw xlsx into `observations/turbine/DK` rather than `raw/` (line 57). |
| scripts/fetch/emi_nz.py | Download EMI Generation_MD and register (NZ) | runbooks/nz.md; .claude/skills/new-region | keep | Named in the new-region skill; only NZ fetcher. |
| scripts/fetch/epias_tr.py | Probe EPIAS (TR) endpoints, must-distinguish plant test | runbooks/tr.md; scripts/README.md | keep | runbooks/tr.md records the negative result via `--probe`; reproduction of "TR not shipped". See section F on its credentials file. |
| scripts/fetch/era5.py | ERA5 CDS fetch for any region or bare bbox | tests/test_fetch_era5.py (file import); configs ar/cl/es_ws/scorecard; runbooks; pyproject.toml:67 | keep | Test loads it via `spec_from_file_location` (test_fetch_era5.py:23). No src module imports it (section C). |
| scripts/fetch/uk.py | Download REPD, print Ofgem ROC steps | runbooks/uk.md; guides; process/uk.py | keep | Only UK fetcher. |
| scripts/pinn/build_cache.py | Build per-region tensor caches for the PINN | findings: rerun-prereg.md:93, loco-prereg.md:199; src/vwf/pinn/cache.py | move to scripts/studies/physics-informed/build_cache.py | Shared by three studies (original, rerun, loco); exact command lines in two preregistrations. Move with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/d0_metric_reframe.py | D0: rescore ML transfer against identity correction | none by path; label in rerun-prereg.md:37 | move to scripts/studies/physics-informed/d0_metric_reframe.py | Numbers in method-physics-informed.md:98-105 (1/5 to 3/5); commit 85ae53a. Not dead. Moves with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/d1_subgrid_terrain.py | D1: scalar vs sub-grid elevation excess and relief | none by path; label rerun-prereg.md:37 | move to scripts/studies/physics-informed/d1_subgrid_terrain.py | Numbers in method-physics-informed.md:88-97 (+0.76, +0.70, ... pooled +0.55); commit 85ae53a. Moves with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/d2_reparameterise.py | D2: rank-1 ridge and pivot reparameterisation | none by path; label rerun-prereg.md:37 | move to scripts/studies/physics-informed/d2_reparameterise.py | Numbers in method-physics-informed.md:70-76 (pearson -0.999, pivot 3.8-4.2 m/s); commit 85ae53a. Moves with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/d3_multiscale_terrain.py | D3: multi-scale terrain features for transfer | none by path; label rerun-prereg.md:37 | move to scripts/studies/physics-informed/d3_multiscale_terrain.py | Numbers in method-physics-informed.md:83-86 (relief 28 km 0.19 vs 0.02); commit 85ae53a. Moves with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/d4_target_noise_floor.py | D4: variogram nugget of per-cluster scalar | none by path; label rerun-prereg.md:37 | move to scripts/studies/physics-informed/d4_target_noise_floor.py | Numbers in method-physics-informed.md:78-81 (UK 59%, BR 6%, DE 15%); commit 85ae53a. Moves with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/d5_regime_coverage.py | D5: physiographic coverage of each holdout | none by path; label rerun-prereg.md:37 | move to scripts/studies/physics-informed/d5_regime_coverage.py | Numbers in method-physics-informed.md:221 and 260-263 (93%, 50%, 33.3%, 91.5%); commits 11926ab, e9ccc64. Moves with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/d6_identifiability.py | D6: efficiency vs speed-up separability probe | none by path; rerun-prereg.md:264 "D6 found" | move to scripts/studies/physics-informed/d6_identifiability.py | Numbers in method-physics-informed.md:248-253 (starts 0.70-0.98, span 0.097, loss spread 0.0004); commit 1b3fbc6. Moves with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/d7_residual_anatomy.py | D7: residual sliced by physical axis | none by path; label rerun-prereg.md:37 | move to scripts/studies/physics-informed/d7_residual_anatomy.py | Section 5 table, method-physics-informed.md:324-340 (span/sd 1.14 ...); commits 5437e52, 58021f2. Moves with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/d8_hub_height_sensitivity.py | D8: cost of Brazil's uniform 100 m hub height | none by path; label rerun-prereg.md:37 | move to scripts/studies/physics-informed/d8_hub_height_sensitivity.py | method-physics-informed.md:317-321 (0.0006, 0.0001 vs 0.0028); commit 0e9a315 (its own doc deleted in 8808cc8, merged). Moves with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/e1_figures.py | Three PDF figures from E1 outputs | run_overnight.sh; run_remaining.sh | move to scripts/studies/physics-informed/e1_figures.py | Stage 08 of both runners; figures go to `output/pinn/figures/`, none are embedded in docs/findings. Moves with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/e1_loro.py | E1 leave-one-region-out driver, PINN vs incumbent | findings: rerun-prereg.md:75,112; src/vwf/pinn/runs.py:3; both runners | move to scripts/studies/physics-informed/e1_loro.py | Exact command line in a preregistration (rerun-prereg.md:112); the section-3 headline reproduces from it at 9faa192. Moving stales dated commands. Move with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/e1_report.py | Assemble E1 table, score gates P1-P3 | e1_loro.py:425 (hint text); run_overnight.sh | move to scripts/studies/physics-informed/e1_report.py | No findings path citation. Reads fixed `output/pinn/e1` and `output/validation/cluster_sweep_2026-07-24`; no `--out`, so cannot score the rerun dir. Update e1_loro.py:425 if moved. Moves with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/e2_physics_audit.py | E2: physical credibility of fitted terms | both runners; label "physics audit E2" rerun-prereg.md:37 | move to scripts/studies/physics-informed/e2_physics_audit.py | Section 3 physics figures (method-physics-informed.md:240-246, Tehachapi 1.74). Commits a0ddcff, becdbf7, 471d4d6. Moves with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/g0_cache_reproduction.py | Compare rebuilt vs published PINN caches | findings: rerun-prereg.md:77,104 | move to scripts/studies/physics-informed/g0_cache_reproduction.py | Gate R1 is read from its `g0_cache_reproduction.csv` (rerun-prereg.md:150). Exact command in a preregistration. Move with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/gwa_ratio.py | Global Wind Atlas speed-up ratio per cache unit | findings: turbine-prereg.md:293; turbine_loro.py | move to scripts/studies/physics-informed/gwa_ratio.py | Registered driver of the turbine-only study queued in STATUS (3dbb83b), not yet run; moving now breaks its registered commands. Move with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/loco.py | Leave-one-country-out PINN driver | findings: loco-prereg.md:227,248; src/vwf/pinn/runs.py:3 | move to scripts/studies/physics-informed/loco.py | Exact commands in a preregistration; results in method-physics-informed-loco.md. Commit 66a68f7. Move with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/prep_rf_features.py | Precompute SET_A terrain features for the RF arm | e1_loro.py:82,123 (hint strings) | move to scripts/studies/physics-informed/prep_rf_features.py | Input to E1's rf-transfer arm (gate P2), which the rerun did not cover (rerun-prereg.md:63). Update e1_loro strings if moved. Moves with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/run_overnight.sh | Unattended serial runner for E1, E2, E3-E7, figures | none | move to scripts/studies/physics-informed/run_overnight.sh | Record of the 2026-08-23 programme (cae8a9a). Not broken, but stale: see section A. Moves with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/run_remaining.sh | Concurrent runner for the sensitivity stages | none | move to scripts/studies/physics-informed/run_remaining.sh | Commit ff7919d; invokes only existing scripts and flags; same staleness as run_overnight.sh. Moves with the whole directory, after the turbine-only study has run (Q2). |
| scripts/pinn/turbine_loro.py | Leave-one-turbine-region-out terrain driver | findings: turbine-prereg.md:283,300; STATUS.md | move to scripts/studies/physics-informed/turbine_loro.py | Registered driver of a study not yet run (00a99d7, 3dbb83b); moving before the run stales the registration. Move with the whole directory, after the turbine-only study has run (Q2). |
| scripts/process/aemo_au.py | AEMO SCADA + Gen Info + GWPT to AU-NEM inputs | findings: region-au-nem.md:175; src datasets/aemo_au.py:3, sources/aemo.py:256; au README | keep | Thin over vwf.datasets.aemo_au; only the alias-override block (lines 90-103) is domain logic worth promoting. |
| scripts/process/cammesa_ar.py | CAMMESA GWh + GWPT join to AR inputs | findings: dataset-survey.md; runbooks/ar.md; src datasets/sources cammesa_ar docstrings | promote to src/vwf/datasets/cammesa_ar | Carries `norm`, `gwpt_argentina`, `join_coords_caps` and the 16-plant `EXCLUDE` curation list; untested (tests import only vwf.datasets.cammesa_ar). Keep a thin CLI at this path. |
| scripts/process/cen_cl.py | CEN generation + GWPT join to CL inputs | configs cl.toml, scorecard/cl_k10.toml; runbooks/cl.md | promote to src/vwf/datasets/cen_cl | Carries `norm`, `gwpt_chile`, `match` (CAP_TOL 0.35), `load_generation`, `EXCLUDE`; untested. Keep a thin CLI at this path (scorecard config cites it). |
| scripts/process/de.py | Validate and stage confidential WindStats DE files | runbooks/de.md; guides | keep | Schema check plus copy; DE is read raw by `EuropeanTurbineSource`; no datasets module needed. |
| scripts/process/dk.py | ens.dk xlsx to dk_md/dk_obs CSVs | runbooks/dk.md; guides; fetch/dk.py | keep | Already a thin CLI over vwf.datasets.process_dk_raw_data (whose own `main()` at line 286 duplicates this entry point). |
| scripts/process/eia_us.py | EIA-923/860 + USWTDB to US inputs | runbooks/us.md; src datasets/sources eia_us docstrings | keep | Thin: only file reading, one drop_duplicates and the report. |
| scripts/process/emi_nz.py | EMI Generation_MD + curated tables to NZ inputs | runbooks/nz.md; new-region skill; src datasets/sources emi_nz docstrings | promote to src/vwf/datasets/emi_nz | Production capacity history and mask (`capacity_history`, `mask_from_windows`, `gen_code_map`, fuel/Gen_Code loop) live only here, untested; src's tested register versions are "not used" (datasets/emi_nz.py:35-38). |
| scripts/process/ons_br.py | ONS FC + COFF + SIGA to BR inputs | runbooks/br.md; src datasets/sources ons_br docstrings | keep | Thin over vwf.datasets.ons_br; only a glob reader and the report. |
| scripts/process/uk.py | REPD metadata and ROC observations for UK | runbooks/uk.md; src datasets/uk_roc.py:3 | keep | Thin over vwf.datasets.uk_roc; minor station-ID regex and column picker only. |
| scripts/process/windstats.py | Confidential WindStats + GWPT coords for ES/SE/FI | runbooks/es.md; src sources/windstats.py:7,50,89,113 | keep | Thin over vwf.datasets.windstats; adapter error messages name this path. |
| scripts/region_tools/apply_turbine_specs.py | Real curves and hub heights from a per-plant spec table | runbooks/ar.md, cl.md; guides/adding-an-observation-source.md | keep | Reusable across CL, AR, AU-NEM; delegates to `assign_curves_from_library`. Commit 7260c43. |
| scripts/region_tools/assign_au_curves.py | Per-library curve assignment for AU fleet | findings: region-au-nem.md:175; src datasets/eia_us.py:400; au README | keep | Exact reproduction chain in region-au-nem.md; builder of examples au_nem_md_open.csv. Docstring still uses the retired "D2" label. |
| scripts/region_tools/assign_country_zones.py | Grid points to bidding zones by real polygons | configs/curation/zones/README.md; guides/data-sources.md; weight_country_grid_points.py | keep | Reusable for any zonal region (SE, NO). Commit dd87232. |
| scripts/region_tools/export_au_grid_netcdf.py | AU-NEM gridded corrected-wind and CF NetCDF demo | src/vwf/harness/export.py:23; scripts/README.md | move to scripts/studies/generalisation/export_au_grid_netcdf.py | One-off deliverable of method-generalisation.md:165; generalised (static field only) by vwf.harness.export. Docstring label "D4" retired by 5697ebc. |
| scripts/region_tools/repair_country_capacity.py | Rebuild country CF on a GWPT capacity register | findings: method-country-level.md:202; STATUS.md:237 | keep | Not fixed upstream (section D). STATUS names it for PT. Imports weight_country_grid_points via sys.path (line 62, 66). |
| scripts/region_tools/weight_country_grid_points.py | GWPT capacity weights for country grid points | findings: cluster-selection-prereg.md:252, country-level.md, national-single-cluster-prereg.md; src generate_country_level_training_data.py:677, sources/entsoe_files.py:73 | promote to src/vwf/datasets/gwpt.py | `load_gwpt`, `fleet_for`, `GWPT_PATH` are imported by another script via sys.path hack; src tells users to run it. Keep a thin CLI at this path (cited command lines). |

### src/vwf/ (81 files)

| path | one-line purpose | referenced by | verdict | evidence |
|---|---|---|---|---|
| src/vwf/__init__.py | Package root: `__version__`, re-exports of PyVWF, train_set/val_set, loaders, sources, config, viz | pyproject (`version = {attr = "vwf.__version__"}`); tests/test_packaging.py; docs/conf.py:17; harness/provenance.py:29; harness/export.py:39; scripts/region_tools/export_au_grid_netcdf.py:38 | keep | The single source of truth for the version. No tracked .py, .ipynb or .md file does `from vwf import <re-exported name>`. Only `vwf.__version__` is read (section D). |
| src/vwf/cli/__init__.py | Docstring-only package for the console scripts | pyproject `[project.scripts]` (through vwf.cli.train) | keep | Needed for the `vwf.cli.train:main` entry point to resolve. |
| src/vwf/cli/train.py | `pyvwf-train`: legacy PyVWF.train plus simulate_cf for one country and year | pyproject.toml:118; tests/test_cli.py; examples/quick_run.py; README.md:79-82,122; Dockerfile:12; docker-compose.yml:5; PIPELINE.md:11 | keep | The only `[project.scripts]` entry. test_cli.py:81 patches `vwf.cli.train.PyVWF`. |
| src/vwf/clustering.py | k-means turbine clustering, region-shape load and repair, Voronoi cluster geometry, country sampling points | data.py:57; harness/driver.py:24; harness/hindcast.py:29; vwf.py:28; datasets/generate_country_level_training_data.py:45; viz/factors.py:183 (fn); 3 tests; 4 scripts/analysis; docs/api.md:80 | keep | Core. Three groups with few cross-calls, a real split seam (section C). |
| src/vwf/config.py | PyVWFPaths (input root, reference_file with bundled fallback), BoundingBoxes | 20 src modules; 20 tests; 9 scripts; examples/run_minimal.py; ci.yml | keep | Imported by 20 src files (AST). config.py:84 resolves `vwf.resources`. |
| src/vwf/correction.py | Affine correction: calculate_scalar, find_offset, find_offsets_country_level | data.py:58; harness/corrections.py:24; vwf.py:26; tests/test_correction.py; tests/test_harness_corrections.py; examples/run_minimal.py:26; scripts/analysis/pivot_probe.py:42; findings cite its functions | keep | Pinned: tests/test_harness_corrections.py:63 `test_affine_golden_regression_bit_for_bit` calls `correction.find_offset`. |
| src/vwf/data.py | Legacy orchestration: train_set, val_set, prep_country, cluster_train_set, curve assignment (add_models, load_power_curves), country-level CF helpers | harness driver/corrections/hindcast; vwf.py; pinn/cache.py; sources/european.py:79 and client_csv.py:133 (fn); 9 tests; 10 scripts; notebook northsea_data; findings cite vwf.data.* | keep | Golden test imports train_set, cluster_train_set, format_bc_factors. Dead inside it: `sim_turbines_to_country_cf` (only docs/api.md:48) and COUNTRY_DIR/TURBINE_DIR/COUNTRY_LEVEL_DIR (0 users). |
| src/vwf/datasets/COMBINED_ERA5_USAGE.md | May-era usage guide for combine_era5_files.py and the era5/EU combined files | findings: scorecard.md:281 (by path); CHANGELOG.md:198 | merge into docs/guides/data-sources.md | User guide inside the package; not package-data, so never shipped; no docs page covers era5/EU. findings: scorecard.md:281 cites the path, so the merge leaves that citation stale. |
| src/vwf/datasets/__init__.py | One-line docstring package marker | every `vwf.datasets.*` import | keep | Required for the subpackage. |
| src/vwf/datasets/aemo_au.py | Pure transforms for the AU-NEM inputs (AEMO + GWPT) | scripts/process/aemo_au.py:35; tests/test_aemo_au_processing.py | keep | Follows the pattern of pure logic in vwf.datasets with a thin CLI in scripts/process. |
| src/vwf/datasets/cammesa_ar.py | Pure transforms for the Argentina inputs | scripts/process/cammesa_ar.py:37; tests/test_cammesa_ar_processing.py | keep | Same pattern as aemo_au. |
| src/vwf/datasets/cen_cl.py | Pure transforms for the Chile inputs | scripts/process/cen_cl.py:37; scripts/analysis/hourly_resolution_test.py:75; tests/test_cen_cl_processing.py | keep | Same pattern as aemo_au. |
| src/vwf/datasets/combine_era5_files.py | CLI: merges EU 10 m and 100 m files per year and stores the annual-mean z0 (the annual-mean roughness route) | CONTEXT.md:51; docs/design/roughness-temporal-treatment.md:15; findings: scorecard.md:230,283; pyproject coverage omit | keep | Produces era5/EU, which DE, DK, UK and the 8 country-level rows read. Its path is cited in scorecard.md:230 and CONTEXT.md, so a move to scripts/era5/ would make a findings citation stale. |
| src/vwf/datasets/eia_us.py | Pure transforms for the US inputs (EIA + USWTDB) | scripts/process/eia_us.py:33; scripts/process/emi_nz.py:40; scripts/region_tools/apply_turbine_specs.py:49; tests/test_eia_us_processing.py | keep | Also a helper source for emi_nz and apply_turbine_specs. |
| src/vwf/datasets/emi_nz.py | Pure transforms for the NZ inputs | scripts/process/emi_nz.py:41; tests/test_emi_nz_processing.py; docs/runbooks/nz.md:32 | keep | Same pattern as aemo_au. |
| src/vwf/datasets/era5.py | prep_era5: ERA5 load, bbox slice, roughness treatment, hub-height wind | data.py:55; harness/export.py:41; harness/hindcast.py:34; pinn/era5_stats.py:25; 5 tests; 6 scripts; examples; findings: method-why-corrections-do-not-transfer.md | keep | A core runtime module inside `datasets/`. The pyproject mypy override `module = "vwf.datasets.*"`, `ignore_errors = true`, also silences it. |
| src/vwf/datasets/fetch_entsoe_capacity_factors.py | ENTSO-E fetch CLI plus pure CF-ratio transforms | datasets/generate_country_level_training_data.py:44; tests/test_entsoe_fetch_consistency.py; findings: method-country-level.md:343 | keep | The entsoe import is guarded by try/except (L34-45), so CI collects without the `data` extra. |
| src/vwf/datasets/generate_country_level_training_data.py | CLI: grid points, ENTSO-E observations and pyvwf_config.py for the 9 ENTSO-E countries | docs/guides/data-sources.md:52-53,166; docs/guides/adding-an-observation-source.md:191,210; PIPELINE.md:28,75; configs/curation/zones/README.md; .claude/skills/new-region | keep | The only documented acquisition path for the country-level regions. It is an entry point, so nothing imports it. sys.path hack at L42 (section B). Split proposal in section C. |
| src/vwf/datasets/ons_br.py | Pure transforms for the Brazil inputs | scripts/process/ons_br.py:36; sources/ons_br.py:33; tests/test_ons_br_processing.py | keep | The only datasets module an adapter imports. |
| src/vwf/datasets/process_dk_raw_data.py | DK Excel transforms, plus its own argparse main | scripts/process/dk.py:24; docs/runbooks/dk.md; findings: method-curve-library-prereg.md:264; pyproject coverage omit | keep | main() at L286-373 duplicates the scripts/process/dk.py CLI (b689fed). The module is needed; the in-module main is redundant. |
| src/vwf/datasets/uk_roc.py | Pure transforms for the UK inputs (ROC + REPD) | scripts/process/uk.py:27; tests/test_uk_roc_processing.py | keep | Same pattern as aemo_au. |
| src/vwf/datasets/windstats.py | Pure transforms for the confidential WindStats regions | scripts/process/windstats.py:27; tests/test_windstats_processing.py | keep | Same pattern as aemo_au. |
| src/vwf/extensions/__init__.py | Docstring-only marker for the optional extensions | every `vwf.extensions.*` import | keep | Required for the subpackage. |
| src/vwf/extensions/grid/__init__.py | Re-exports the gridded-correction API (28 names) | scripts/analysis: domain_split_study, loco_interpolation, offshore_pool_study, surface_flag_report, unmasked_surface_bands (all via `from vwf.extensions.grid import ...`) | keep | Its docstring names the two registered studies (loco-interpolation, offshore-pool prereg). |
| src/vwf/extensions/grid/evaluate.py | Scores a gridded correction at the observations | scripts/analysis/domain_split_study.py:52; offshore_pool_study.py:44; tests/test_grid_evaluate.py | keep | Imported by two study drivers and one test. |
| src/vwf/extensions/grid/geodataframes.py | Joins factors to cluster geometries | tests/test_grid_geodataframes.py; re-exported in grid/__init__ | keep | Ported in dcc5e6b (2026-09-13) for the chapters 4-5 work, and tested. No script uses it yet. |
| src/vwf/extensions/grid/interpolation.py | IDW, kriging, RBF and nearest-neighbour interpolation of factors | extensions/grid/surface.py:52; 3 scripts/analysis; tests/test_grid_interpolation.py; findings cite degree_distances | keep | The single definition that both registered studies use. |
| src/vwf/extensions/grid/surface.py | Builds a correction surface from control points | 5 scripts/analysis; tests/test_grid_surface.py; findings: method-grid-nl-holdout-prereg.md | keep | Imported by 5 scripts (AST). |
| src/vwf/geospatial.py | Labels points onshore or offshore against region geometries | extensions/grid/surface.py:53; scripts/analysis/offshore_pool_study.py:45; tests/test_geospatial.py | keep | Overlaps with the clustering.py shape loader (section E). |
| src/vwf/harness/__init__.py | Re-exports the harness API | tests (2); 12+ scripts via `from vwf.harness import driver, regions` | keep | Every harness entry point goes through this package. |
| src/vwf/harness/corrections.py | CorrectionModel registry; the affine-wind delegate | harness driver, export, hindcast; extensions/grid/surface.py:54; 5 tests incl. the golden test; 4 scripts | keep | Carries the golden pin (test_harness_corrections.py:63). |
| src/vwf/harness/driver.py | run_train, run_evaluate, run_transfer | 18 scripts; 10 tests; pinn/cache.py:33; pinn/runs.py:17; 2 notebooks | keep | The preferred path (AGENTS.md). |
| src/vwf/harness/export.py | Exports an atlite-ready gridded correction field | scripts/analysis/export_correction_field.py:20; docs/design/harness.md | keep | Has a live CLI caller. Imports the root `vwf` at L39 (section B). |
| src/vwf/harness/hindcast.py | Applies trained factors over a long ERA5 window | scripts/analysis/run_hindcast.py:28; docs/design/harness.md; docs/guides/output-structure.md | keep | Has a live CLI caller. |
| src/vwf/harness/provenance.py | Writes the run manifest (git state, curve library sha256) | harness/driver.py:28; vwf.py:243 (fn); 5 scripts/pinn; 3 tests | keep | Imports the root `vwf` at L29 just for `__version__`, which closes a cycle (section B). |
| src/vwf/harness/regions.py | Loads the region TOML configs (RegionSpec) | 31 scripts; 8 tests; 7 src modules | keep | Has no internal imports: the lowest layer. |
| src/vwf/harness/skill.py | Skill metrics, common rows, pseudo-replicate collapse | 9 scripts; 2 tests; driver.py:35; pinn/runs.py:18 | keep | Imported by 9 scripts and 2 tests (AST). |
| src/vwf/loaders/__init__.py | Re-exports the turbine loaders, the year-specific grid loader and the country obs checks | vwf/__init__.py:26; data.py:65; vwf.py:356 (fn); scripts/analysis/curve_library_tables.py:301; curve_match_audit.py:129 | keep | Imported by 3 src modules and 2 scripts (AST). |
| src/vwf/loaders/country_level_loaders.py | load_year_specific_grid_points for the legacy country-level path; two dead helpers | loaders/__init__.py:12 (re-exported as `vwf.load_year_specific_grid_points`); vwf.py:356,380; docs/api.md:122-123 | keep | Not orphaned: the legacy path uses it via PyVWF.load_country_data_with_year_specific, which scripts/analysis/train_all_bias_corrections.py:487 calls. country_gen_to_cf has 0 callers (section A). |
| src/vwf/loaders/country_obs_checks.py | Plausibility gates for country-level CF series | sources/entsoe_files.py:24; sources/entsoe_zonal.py:34; scripts/analysis/audit_country_observations.py:28; scripts/region_tools/repair_country_capacity.py:65; 2 tests | keep | Live in the harness country path. |
| src/vwf/loaders/turbine_loaders.py | CSV loaders for the DK/DE/UK turbine files | sources/european.py:14; loaders/__init__.py:8; tests/test_sources.py:88-89 (patched) | keep | The european-turbine adapter reads its data through it. |
| src/vwf/metrics.py | Legacy error metrics (calculate_error, overall_error) | scripts/analysis/evaluate_all_pyvwf_runs.py:22; extensions/grid/evaluate.py:40; tests/test_metrics.py; tests/test_pipeline.py:31; ci.yml:178 (clean-install import) | keep | The CI wheel smoke test imports it. It overlaps harness/skill.py by concept, not by code (section E). |
| src/vwf/pinn/__init__.py | Docstring-only package | every `vwf.pinn.*` import | keep | Required for the subpackage. |
| src/vwf/pinn/cache.py | Assembles the per-region tensors the physics-informed model trains on | pinn/train.py:25; scripts/pinn: build_cache, g0_cache_reproduction, gwa_ratio, prep_rf_features; 3 tests | keep | Last changed 735f413 (2026-09-16). |
| src/vwf/pinn/era5_stats.py | Daily ERA5 statistics at turbine points | pinn/cache.py:36; tests/test_pinn_era5_record.py | keep | Imports the private `_normalise_longitudes` and `_slice_bbox` from datasets.era5 (L25). |
| src/vwf/pinn/gwa.py | Wind-atlas speed-up ratio per unit | scripts/pinn/gwa_ratio.py:29; tests/test_pinn_level_spatial_gwa.py | keep | Added in 00a99d7 (2026-09-18) for the study registered in 4d671c7. |
| src/vwf/pinn/model.py | The learned part: four bounded physical quantities | pinn/train.py:26; scripts/pinn/d6_identifiability.py:35; tests/test_pinn_physics.py | keep | Imported by pinn/train, one script and one test (AST). |
| src/vwf/pinn/physics.py | Differentiable forward operator, ERA5 wind to monthly CF | pinn/train.py:27; scripts/pinn/e1_loro.py:56; tests/test_pinn_physics.py | keep | Imported by pinn/train, one script and one test (AST). |
| src/vwf/pinn/runs.py | Shared pieces of the physics-informed run drivers | scripts/pinn/e1_loro.py, loco.py, turbine_loro.py; tests/test_pinn_level_spatial_gwa.py:12 | keep | Last changed 00a99d7 (2026-09-18). |
| src/vwf/pinn/terrain.py | Multi-scale terrain descriptors | pinn/cache.py:37; pinn/train.py:31; scripts/pinn/d5, d7; tests/test_pinn_physics.py | keep | Imported by 2 src modules, 2 scripts and 1 test (AST). |
| src/vwf/pinn/train.py | Fits the physics-informed correction to observed generation | 7 scripts/pinn; tests/test_pinn_physics.py; findings: method-physics-informed-rerun-prereg.md, -turbine-prereg.md | keep | Named in two pre-registrations. |
| src/vwf/py.typed | PEP 561 marker | pyproject package-data `vwf = ["py.typed"]`; tests/test_packaging.py; CONTRIBUTING.md | keep | The `Typing :: Typed` classifier depends on it. |
| src/vwf/resources/__init__.py | Docstring; makes the open library importable through importlib.resources | config.py:84; harness/provenance.py:47,117; tests/test_curve_library.py:20; scripts/analysis/curve_match_audit.py:154 | keep | package-data `"vwf.resources" = ["*.csv"]`. |
| src/vwf/resources/models.csv | Open library `models.csv` | PyVWFPaths.reference_file fallback; tests/test_curve_library.py; tests/test_committed_files.py:80; scripts/analysis/curve_match_audit.py:170 | keep | `cmp` shows it byte-identical to input/reference/models.csv. test_curve_library.py:25 enforces that. It differs from input/combined (the licensed merge). |
| src/vwf/resources/power_curves.csv | Open library `power_curves.csv` (76 curves) | same as models.csv | keep | `cmp` shows it byte-identical to input/reference/power_curves.csv, and different from input/combined/reference. |
| src/vwf/resources/power_curves_provenance.csv | Per-column source and licence of the open library | scripts/analysis/curve_match_audit.py:154; tests/test_committed_files.py | keep | Byte-identical to both the input/reference and input/combined/reference copies. |
| src/vwf/sources/__init__.py | Imports every adapter so its registration runs; re-exports 18 names | data.py:69; vwf.py:32; harness/driver.py:41; vwf/__init__.py:31; 15 tests; 2 scripts; notebook au_nem_validation | keep | Registration depends on these imports. |
| src/vwf/sources/aemo.py | `aemo-nem` adapter | configs au_nem (`source = "aemo-nem"`, 2 files); scripts/process/aemo_au.py:45; 3 tests | keep | Selected by name from its config. |
| src/vwf/sources/base.py | The ObservationSource contract | 15 src modules; docs/api.md:27 | keep | Every adapter subclasses it. |
| src/vwf/sources/cammesa_ar.py | `cammesa-ar` adapter | 2 AR configs; tests/test_cammesa_ar_processing.py | keep | Selected by name from its config. |
| src/vwf/sources/cen_cl.py | `cen-cl` adapter | 2 CL configs; tests/test_cen_cl_processing.py | keep | Selected by name from its config. |
| src/vwf/sources/client_csv.py | `client-csv-turbine` adapter for user-supplied CSVs | tests/test_client_csv_source.py; docs/guides/your-own-data.md | keep | Reachable only through get_source by name (`countries = ()`). Its lazy import of vwf.data at L133 is an upward edge (section B). |
| src/vwf/sources/eia_us.py | `eia-us` adapter | 2 US configs; tests/test_harness_eia_us.py | keep | Selected by name from its config. |
| src/vwf/sources/emi_nz.py | `emi-nz` adapter | 2 NZ configs; tests/test_emi_nz_processing.py | keep | Selected by name from its config. |
| src/vwf/sources/entsoe_files.py | `entsoe-country` adapter over observations/country | 29 configs; harness/driver.py:75-76; pinn/cache.py:35; tests/test_harness_entsoe_files.py | keep | Every country-level scorecard row uses it. |
| src/vwf/sources/entsoe_zonal.py | `entsoe-zonal` adapter, one series per bidding zone | 1 config (SE-BZ); harness/driver.py:75-76; tests/test_country_zonal.py | keep | Selected by name from its config. |
| src/vwf/sources/european.py | `european-turbine` adapter for DK, DE and UK | 11 configs incl. scorecard/de_k100, dk_k100, uk_k50; harness/driver.py:78 (by name); data.py:160 (resolve); vwf/__init__.py:32; tests/test_sources.py (13 tests); docs/api.md:33; docs/runbooks/de, dk, uk | keep | Not orphaned: three scorecard rows run through it (section A). |
| src/vwf/sources/in_memory.py | `in-memory-country` adapter backed by caller-supplied frames | data.py:69; vwf.py:32; vwf/__init__.py; tests/test_country_zonal.py:197 | keep | The legacy country path and the tests use it. |
| src/vwf/sources/ons_br.py | `ons-br` adapter | 2 BR configs; tests/test_harness_ons_br.py | keep | Selected by name from its config. |
| src/vwf/sources/registry.py | register, get_source, resolve, available_sources | 13 src modules; tests/test_sources.py:34 | keep | The harness selects adapters through it. |
| src/vwf/sources/windstats.py | `windstats` adapter (ES-WS, confidential) | 1 config; tests/test_windstats_processing.py | keep | Selected by name from its config. |
| src/vwf/time_utils.py | Time-slice mapping and columns | correction.py:6; data.py:64; wind.py:18; tests/test_time_utils.py; scripts/region_tools/export_au_grid_netcdf.py:40 | keep | Has no internal imports: the lowest layer. |
| src/vwf/utils.py | ensure_numeric (32 lines) | wind.py:19; loaders/turbine_loaders.py:9; docs/api.md:135 | keep | Tiny, but documented API; folding it into another module would need a shim. |
| src/vwf/viz/__init__.py | Re-exports 8 plotting functions | vwf/__init__.py:50; examples/viz_demo.py; tests/test_viz.py; docs/guides/visualisation.md | keep | The documented import surface for plots. |
| src/vwf/viz/distribution.py | CF histogram, ECDF, QQ; the Results loader | viz/__init__.py:18; docs/api.md:103 | keep | Reached through the package re-export. |
| src/vwf/viz/evaluation.py | Error vs cluster count, sim-vs-obs scatter | viz/__init__.py:24; docs/api.md:109 | keep | Reached through the package re-export. |
| src/vwf/viz/factors.py | Factor maps and the scalar/offset joint plot | viz/__init__.py:25; docs/api.md:106 | keep | Reached through the package re-export. |
| src/vwf/viz/palettes.py | Okabe-Ito colours and time-slice labels | viz/evaluation.py:30; scripts/pinn/e1_figures.py:29 | keep | Imported by one src module and one script (AST). |
| src/vwf/viz/style.py | Shared matplotlib style | scripts/pinn/e1_figures.py:30 | keep | One script uses it and it is not re-exported, but the viz/__init__ docstring lists it as public. |
| src/vwf/vwf.py | Legacy PyVWF class (train, simulate_cf, country loaders) | vwf/__init__.py:24; cli/train.py:17; tests/test_pipeline.py:32; scripts/analysis/train_all_bias_corrections.py:25, regression_run_legacy.py:42; examples; docs/api.md:15 | keep | The legacy reference. Stays whole (section C). The lazy harness import at L243 is an upward edge. |
| src/vwf/wind.py | Wind interpolation, simulate_wind, curve lookup | 10 src modules; 10 tests; 8 scripts; examples; ci.yml:178 | keep | Pinned: the golden test calls `wind.simulate_wind` (test_harness_corrections.py:81). |

### tests/ (67 files)

| path | one-line purpose | referenced by | verdict | evidence |
|---|---|---|---|---|
| tests/conftest.py | Shared synthetic fixtures (grid, reanalysis, make_reanalysis, turbines, power_curve) | CI (pytest); CONTRIBUTING.md:77; used by about 30 test files | keep | 4e416b1 2026-05-20 Prepare PyVWF for JOSS submission (only commit). Fixtures `grid` and `reanalysis` used across harness, pipeline, pinn and grid tests. |
| tests/test_aemo_au_processing.py | AU-NEM raw-data processing (vwf.datasets.aemo_au): MMS parsing, fleet join, chunked partials | CI (pytest) | keep | Tests vwf.datasets.aemo_au plus vwf.sources.aemo partials; complementary to test_harness_aemo.py (adapter). Origin ca0beba 2026-07-16; last df03475 2026-07-23 Docs and prose. |
| tests/test_cammesa_ar_processing.py | Argentina (CAMMESA) processing: GWh to CF, capacity join guards, masks | CI (pytest) | keep | Covers both vwf.datasets.cammesa_ar and the adapter in one file. ea2df4d 2026-07-23 Argentina: cammesa-ar adapter built and tested against the real data. |
| tests/test_cen_cl_processing.py | Chile (CEN) processing: wind filter, fixed-offset UTC binning, CF, masks | CI (pytest) | keep | Datasets and adapter (vwf.sources.cen_cl) in one file. 180c564 2026-07-23 Chile: cen-cl adapter built and tested against the real 2021-2024 data. |
| tests/test_cli.py | The `pyvwf-train` console script parser and entry point | CI (pytest) | keep | Imports vwf.cli.train `_build_parser, main` (resolve). f6e7bf5 2026-07-14 Make PyVWF usable outside a repository checkout. |
| tests/test_client_csv_source.py | Client CSV observation adapter (ClientCsvTurbineSource) | CI (pytest) | keep | dcbd877 2026-08-12 Sources: add per-zone ENTSO-E and user-supplied CSV observation adapters (only commit). |
| tests/test_cluster_selection_study.py | The cluster selection runner's fold and selection rules | CI (pytest) | keep | SCRIPT-TEST: loads scripts/analysis/cluster_selection_study.py by spec_from_file_location (lines 13-15). Single-study driver (method-cluster-selection-prereg.md); moving it to scripts/studies/cluster-selection/ breaks line 15. cc9fc14 2026-09-15. |
| tests/test_clustering.py | Spatial clustering is deterministic (k-means++), optional capacity and geographic weighting | CI (pytest); src/vwf/clustering.py:318; findings: region-us-br.md:152 | keep | Imports vwf.clustering.cluster_turbines (resolves). Origin d289e34 2026-07-22 Clustering: k-means++ init, so the partition is not a seed lottery. |
| tests/test_committed_files.py | Tracked tree obeys .gitignore; open library sha256 in input/reference and vwf/resources | CI (pytest); AGENTS.md; .claude/skills/provenance-guard/SKILL.md; CHANGELOG.md | keep | ea0d6df 2026-09-11 Add provenance-guard. Skips (lines 60, 85) only outside a git checkout; input/reference/*.csv are tracked, so it runs in CI. |
| tests/test_common_row_scoring.py | Conditions compared with each other are scored on the same rows | CI (pytest) | keep | Imports `make_spec, synthetic_dk` from test_harness_driver (line 15). 513f57c 2026-09-11 Score every variant of a run on the same rows. |
| tests/test_config.py | PyVWFPaths resolution in a checkout versus an installed copy (PYVWF_INPUT) | CI (pytest); tests/test_emi_nz_processing.py:16 (comment) | keep | Skip at line 128 only outside a checkout (input/reference/power_curves.csv is tracked). 0e6f78d 2026-08-24 Stop test_config leaking a stale PyVWFPaths class into the whole session. |
| tests/test_correction.py | The affine (scalar plus offset) correction maths in vwf.correction | CI (pytest); findings: region-us-br.md:59 | keep | 724ab1b 2026-09-16 Test the offset search on the residual, not on the step size; 53d667b 2026-08-21 Fix calculate_scalar comparing obs and sim over different samples. |
| tests/test_country_obs_checks.py | Plausibility gates for country-level observed CFs | CI (pytest) | keep | Imports vwf.loaders.country_obs_checks (resolves); docstring cites method-country-level.md. b578bd2 2026-09-15 Three gates: a register judged on when it moved, tiers, and clipped counts. |
| tests/test_country_offset_controls.py | Controls for the country-level offset identifiability question | CI (pytest) | keep | Imports private driver helpers `_country_skill, _zonal_skill` (both exist). 5697ebc 2026-08-24 Make the research record self-contained. |
| tests/test_country_training_path.py | Country-level training path: fleet prep, monthly aggregation, clusters | CI (pytest) | keep | Imports vwf.data (resolves). dd87232 2026-08-12 Country-level: fix the estimator, the observations, and the fleet weighting; 585a2e3 pandas 3 fix. |
| tests/test_country_zonal.py | Per-zone country observations and the exactly-determined country-level fit | CI (pytest); pyproject.toml:138; CLAUDE.md:77 (local, OOM note) | keep | Line 170 `from tests import test_pipeline as tp`, the reason for pytest `pythonpath = ["."]`. dcbd877 2026-08-12. |
| tests/test_curve_library.py | Invariants of the bundled open library; input/reference and vwf/resources identical | CI (pytest); .claude/skills/provenance-guard/SKILL.md; CHANGELOG.md; tests/test_committed_files.py | keep | Partial overlap with test_committed_files.py sha256 test (both copies). Skip line 32 only outside checkout. f50e0a7 2026-07-16 Replace the synthetic placeholder curves with the open turbine library. |
| tests/test_curve_library_assign.py | T2 other-brand assignment rules of the curve library study | CI (pytest) | keep | SCRIPT-TEST: scripts/analysis/curve_library_assign.py (line 16), which imports sibling curve_match_audit via sys.path (line 34-35). Single-study (method-curve-library-prereg.md); a move breaks line 16. 34d9c3a 2026-09-13. |
| tests/test_curve_library_match.py | Brand-and-spec matcher rules for the curve library study's T1 | CI (pytest) | keep | SCRIPT-TEST: scripts/analysis/curve_library_match.py (line 21). Path cited in method-curve-library-prereg.md:209 and scorecard.md:356; a move to scripts/studies/curve-library/ breaks line 21. b7ed5fd 2026-09-13. |
| tests/test_curve_library_study.py | The curve library study driver's refusal and condition wiring (21 tests) | CI (pytest) | keep | SCRIPT-TEST: scripts/analysis/curve_library_study.py (line 20), cited method-curve-library-prereg.md:409. Driver lazily imports sibling baseline_bootstrap (line 315). A move breaks line 20. 6dd7af2 2026-09-13. |
| tests/test_curve_library_tables.py | The curve library study's override-table construction | CI (pytest) | keep | SCRIPT-TEST: scripts/analysis/curve_library_tables.py (line 18), which imports siblings curve_library_assign and curve_library_match (lines 49-50). Cited prereg:174. Moving any of the three breaks it. 83ba607 2026-09-13. |
| tests/test_curve_resolution.py | Which power curve each unit was actually simulated on (curve resolution) | CI (pytest); src/vwf/wind.py:223 | keep | Imports helpers from test_pipeline (line 17) and test_harness_driver (line 19). 5c11310 2026-09-11 Record which power curve every unit was actually simulated on. |
| tests/test_eia_us_processing.py | EIA-US processing (vwf.datasets.eia_us): EIA-923 reshape, EIA-860 capacity, USWTDB join | CI (pytest) | keep | Complementary to test_harness_eia_us.py (adapter); no shared test names. 1c19846 2026-07-17 origin; 05838a7 2026-07-22 US: per-plant power curves. |
| tests/test_emi_nz_processing.py | NZ (EMI) processing: trading-period mapping, monthly CF, masks | CI (pytest); .claude/skills/new-region/SKILL.md:41; docs/guides/adding-an-observation-source.md:222 | keep | The guide names it "the fuller model" for new adapters. b0df5de 2026-07-22 NZ region: emi-nz adapter, curated farm table, fetch/process scripts, runbooks. |
| tests/test_entsoe_fetch_consistency.py | ENTSO-E CF numerator and denominator cover one fleet | CI (pytest); src/vwf/datasets/fetch_entsoe_capacity_factors.py:32 | keep | Source comment keeps this importable without the [data] extra (entsoe-py) CI omits. dd87232 2026-08-12 Country-level: fix the estimator. |
| tests/test_era5.py | ERA5 longitude normalisation on load (vwf.datasets.era5) | CI (pytest) | keep | Receives the merge below. 028b375 2026-07-16 ported the fix to main from 4d70a3c; lines 20-53 byte-identical to test_harness_era5.py 19-52. Unique: test_prep_era5_normalises_0_360_on_load. |
| tests/test_era5_extent.py | Units outside the loaded extent are refused unless a run opts in | CI (pytest) | keep | Reads tracked configs/regions/scorecard/fr_country.toml (line 108). c480f46 2026-09-11 Refuse units outside the loaded ERA5 extent, and record the extent every run. |
| tests/test_eu_rerun_compare.py | Self-check of the paired per-row comparison driver | CI (pytest) | keep | SCRIPT-TEST: scripts/analysis/eu_rerun_compare.py (line 17); script imports siblings baseline_bootstrap and roughness_treatment_study (57-58), so moving either breaks this test. Serves eu-rerun and curve-library studies. dfec0e7 2026-09-13. |
| tests/test_fetch_era5.py | ERA5 fetch script's box, tag and year resolution | CI (pytest) | keep | SCRIPT-TEST: scripts/fetch/era5.py (line 25), a pipeline entry point that stays in place per the layout. 31b1b30 2026-09-13 Fix two tests CI caught. |
| tests/test_fit_diagnostics.py | Where a fitted pair sends its own training speeds | CI (pytest) | keep | Imports vwf.wind.fit_diagnostics and test_harness_driver helpers. 631f0bc 2026-09-11 Record where a fitted pair sends its own training speeds, beside the dagger. |
| tests/test_fit_quality_and_min_cluster.py | Guards for fit_quality and the minimum cluster size | CI (pytest) | keep | Docstring cites method-hourly-resolution.md. 569a00a 2026-08-12 Fit robustness: surface degenerate fits, and guard against tiny clusters. |
| tests/test_geospatial.py | Onshore and offshore classification (src/vwf/geospatial.py) | CI (pytest) | keep | 25e7b4a 2026-09-13 Port vwf/geospatial.py from development, with the tests it never had. |
| tests/test_grid_evaluate.py | Scoring a gridded correction at observations (vwf.extensions.grid.evaluate) | CI (pytest) | keep | a067634 2026-09-13 Port the evaluation driver, counting what the original filled in silently. |
| tests/test_grid_geodataframes.py | Joining factors to cluster geometries (vwf.extensions.grid.geodataframes) | CI (pytest) | keep | dcc5e6b 2026-09-13 Port geodataframes.py, refusing the joins that used to fail silently. |
| tests/test_grid_interpolation.py | Spatial interpolation of factors (vwf.extensions.grid.interpolation) | CI (pytest) | keep | 4 tests importorskip pykrige ([grid] extra, not in CI). Line 118-126 published-CV test gated on git-ignored output/pyvwf_to_grid/, never runs in CI. 3908f5e 2026-09-13. |
| tests/test_grid_surface.py | Gridded correction surfaces (vwf.extensions.grid.surface) | CI (pytest) | keep | 2 of 24 tests importorskip pykrige (lines 182, 271), skipped in CI. be1e4ae 2026-09-15 Withdraw the claim that geometry does not select the risky cells. |
| tests/test_harness_aemo.py | AEMO NEM adapter (vwf.sources.aemo): AEST to UTC binning, CF, masks | CI (pytest) | keep | Tests vwf.sources, not vwf.harness (prefix is a misnomer); no overlap with test_aemo_au_processing.py test names. 8700d1e 2026-07-15 Add the AEMO NEM observation source. |
| tests/test_harness_corrections.py | CorrectionModel interface and the affine delegate (golden pin) | CI (pytest); src/vwf/harness/corrections.py:337 | keep | Imports test_pipeline helper (line 13). 56ab195 2026-08-12 Corrections: per-cluster country offsets, two control models, parallel fit. |
| tests/test_harness_driver.py | Driver end-to-end on synthetic data: train, evaluate, country-level fit | CI (pytest); imported as helper by 7 test files | keep | `make_spec`/`synthetic_dk` imported by test_common_row_scoring, test_curve_resolution, test_era5_extent, test_fit_diagnostics, test_off_curve_counts, test_pinn_country_cache, test_roughness_treatment. 703041c 2026-07-15 origin. |
| tests/test_harness_eia_us.py | EIA-US adapter (vwf.sources.eia_us): net-generation CF, respondent and commissioning screens | CI (pytest) | keep | Tests vwf.sources, pairs with test_eia_us_processing.py. 1c19846 2026-07-17 US (EIA) observation source, Phase 1 (only commit). |
| tests/test_harness_entsoe_files.py | EntsoeFileSource: file-backed country-level loading | CI (pytest) | keep | f26e124 2026-07-15 Wire file-backed country-level loading; dd87232 2026-08-12 Country-level: fix the estimator. |
| tests/test_harness_era5.py | ERA5 longitude normalisation and config-driven paths (vwf.datasets.era5) | CI (pytest) | merge into tests/test_era5.py | Lines 19-52 (three tests, same names) byte-identical to test_era5.py; both came from 4d70a3c/028b375 via merge e15ba4a. Subject is vwf.datasets.era5, not harness. Keep its two unique tests (lines 53, 77). |
| tests/test_harness_ons_br.py | ONS-BR adapter (vwf.sources.ons_br): FC to monthly CF, curtailment mask | CI (pytest) | keep | Pairs with test_ons_br_processing.py (datasets); no shared test names. 2f30370 2026-07-17 Brazil (ONS) observation source, Phase 1 (only commit). |
| tests/test_harness_provenance.py | Run-manifest provenance (docs/design/harness.md section 6) | CI (pytest) | keep | Reads tracked configs/regions via CONFIG_DIR (line 19). Origin 05debb1 2026-07-15 Add the multi-region validation harness core. |
| tests/test_harness_regions.py | Region-config loading and validation | CI (pytest); .claude/skills/new-region/SKILL.md:29,74; docs/guides/adding-an-observation-source.md:34 | keep | Guide step 11 edits its shipped-config count. 05debb1 2026-07-15; 7d634dc 2026-08-12 Docs: consolidate the guides. |
| tests/test_harness_skill.py | Skill metrics and pseudo-replicate handling (vwf.harness.skill) | CI (pytest) | keep | 05debb1 2026-07-15 Add the multi-region validation harness core: region configs, provenance, skill metrics. |
| tests/test_harness_transfer.py | Transfer semantics: collapse, pair enforcement, name matching | CI (pytest) | keep | Imports `check_transfer_pair, collapse_factors` (resolve). 703041c 2026-07-15 Add the harness driver. |
| tests/test_metrics.py | Error metrics behind the published numbers (vwf.metrics) | CI (pytest) | keep | 3775bb5 2026-07-14 Test the scientific core; fix a silent year-relabelling bug in metrics. |
| tests/test_off_curve_counts.py | Off-curve values are counted, not hidden | CI (pytest) | keep | Imports vwf.wind.off_curve_record and test_harness_driver helpers. 03222dc 2026-09-11 Count the simulated values the power curves cannot convert, per variant. |
| tests/test_ons_br_processing.py | Brazil (ONS) processing (vwf.datasets.ons_br): FC reshape, metadata, curtailment account | CI (pytest) | keep | 2f30370 2026-07-17 origin; 3866d23 2026-07-17 Adopt the merged open curve library as the uniform default. |
| tests/test_packaging.py | Packaging invariants: one version, semver, no drift | CI (pytest); CONTRIBUTING.md:68; pyproject.toml; .claude/skills/provenance-guard/SKILL.md; CHANGELOG.md | keep | cfcc634 2026-09-11 Record the curve-provenance work under [Unreleased], and test the release tie. |
| tests/test_pinn_country_cache.py | Country-level caches for the physics-informed model | CI (pytest) | keep | vwf.pinn.cache has no torch import, so it runs in CI. 735f413 2026-09-16 Build country-level PINN caches from national series and per-year grids. |
| tests/test_pinn_era5_record.py | PINN ERA5 reduction: roughness treatment and loaded extent | CI (pytest); findings: method-physics-informed.md:14 | keep | 1a31656 2026-09-16 Apply and record the roughness treatment and loaded extent in the PINN caches. |
| tests/test_pinn_level_spatial_gwa.py | Level and spatial split of fleet error, and the wind-atlas ratio | CI (pytest) | keep | 2 of 4 tests need rasterio via `_raster` (line 46 importorskip, [grid] extra), skipped in CI. 00a99d7 2026-09-18 Add the turbine-only study's terrain switches. |
| tests/test_pinn_physics.py | Differentiable forward operator and learned corrections (52 tests) | CI (pytest, skipped); findings: method-physics-informed.md; scripts/pinn/run_overnight.sh; CHANGELOG.md; pyproject.toml | keep | Module-level `pytest.importorskip("torch")` (line 16); torch is the [pinn] extra, which CI does not install, so no test here ever runs in CI. 00a99d7 2026-09-18. |
| tests/test_pipeline.py | End-to-end legacy PyVWF orchestration on a synthetic fleet | CI (pytest); CLAUDE.md:48 (local); pyproject.toml:139; imported as helper by 6 test files | keep | Imported as `test_pipeline` (5 files) and `tests.test_pipeline` (test_country_zonal.py:170). Origin 3775bb5 2026-07-14; ebba5fb 2026-07-23 Reorganise input/. |
| tests/test_region_shape_repair.py | Region shapes must not silently omit a country's islands | CI (pytest) | keep | Bornholm test (lines 139-142) skipif on input/reference/terrain/coastlines.geojson, git-ignored by `/input` (.gitignore:94), so it never runs in CI. c53bc55 2026-08-12. |
| tests/test_regression_compare.py | The D1 regression frame comparator | CI (pytest) | keep | SCRIPT-TEST: scripts/analysis/regression_compare.py (line 10). Path cited in method-harness-regression.md:33,101; a move to scripts/studies/harness-regression/ breaks line 10 and makes that record stale. 5697ebc 2026-08-24. |
| tests/test_roughness_treatment.py | Which roughness treatment a run applies, and what it records | CI (pytest) | keep | Reads tracked configs/regions/scorecard/dk_k100.toml (line 93); docstring cites method-roughness-treatment-prereg.md. 3ee5319 2026-09-12 Add the roughness treatment switch. |
| tests/test_scorecard_configs.py | Every scorecard row's config is committed and loads | CI (pytest); .claude/skills/provenance-guard/SKILL.md:29; configs/regions/scorecard/README.md:26 | keep | Parses docs/findings/scorecard.md (line 21). 6b7c247 2026-09-13 Check that every scorecard row has a configuration; ef5d499 2026-09-13. |
| tests/test_sources.py | The pluggable ObservationSource layer and registry | CI (pytest); docs/guides/adding-an-observation-source.md:215 | keep | d4706cd 2026-07-10 Refactor observation loading behind a pluggable ObservationSource interface; f16aaa4 2026-07-15. |
| tests/test_southern_hemisphere_pipeline.py | Synthetic Southern Hemisphere ground truth through the full pipeline | CI (pytest); findings: region-au-nem.md:38 | keep | 5697ebc 2026-08-24 Make the research record self-contained (only commit). |
| tests/test_time_utils.py | Time slice parsing and column assignment (vwf.time_utils) | CI (pytest) | keep | 4e416b1 2026-05-20 Prepare PyVWF for JOSS submission; a839786 2026-07-16 Thread an optional seasons mapping. |
| tests/test_uk_roc_processing.py | UK (Ofgem ROC plus REPD) transforms: banding, certificates to MWh | CI (pytest) | keep | Synthetic only (no confidential warehouse data). d4fc7b9 2026-07-23 UK: script the open half (REPD metadata) and the ROC banding pipeline. |
| tests/test_viz.py | vwf.viz distributional and diagnostic plotting | CI (pytest) | keep | 5ebc741 2026-05-28 Remove quantile-mapping extension; 3abad0d 2026-07-14 Add correction-factor and evaluation diagnostics to vwf.viz. |
| tests/test_wind.py | Wind interpolation, height extrapolation, power conversion (vwf.wind) | CI (pytest) | keep | 4e416b1 2026-05-20 Prepare PyVWF for JOSS submission (only commit). |
| tests/test_windstats_processing.py | WindStats (ES/SE/FI) transforms: metadata, output to CF, coords match | CI (pytest) | keep | Synthetic inputs only. bc89826 2026-07-23 Spain (ES-WS): WindStats generation plus open GWPT coordinates. |

## 3. Dead or orphaned

Each lead was checked in both directions: who names the file, and what the
file names. The check covered imports, registry names in configs, re-exports,
`spec_from_file_location` loads in tests, findings citations by path and by
label, commit messages, and whether the run's output is on disk and cited.

**Result: no lead is dead.** The grep missed three kinds of reference. A
driver names its own pre-registration, but the pre-registration does not name
the driver back. A findings document reports a script's numbers without
naming the script. An adapter is selected by its registry name, not its
module path.

### scripts/pinn: d0 to d8, run_overnight.sh, run_remaining.sh

| file | where its output is reported | commit | status |
|---|---|---|---|
| d0_metric_reframe.py | `method-physics-informed.md` section 1, lines 98-105 | 85ae53a | reproduction record |
| d1_subgrid_terrain.py | same document, lines 88-97 (relief +0.76, +0.70, +0.51, +0.44) | 85ae53a | reproduction record |
| d2_reparameterise.py | lines 70-76 (rank-1 ridge, pivot 3.8 to 4.2 m/s) | 85ae53a | reproduction record |
| d3_multiscale_terrain.py | lines 83-86 | 85ae53a | reproduction record |
| d4_target_noise_floor.py | lines 78-81 (nugget shares UK 59%, BR 6%, DE 15%) | 85ae53a | reproduction record |
| d5_regime_coverage.py | section 3 line 221, section 4 lines 260-263 | 11926ab, e9ccc64 | reproduction record |
| d6_identifiability.py | section 3 lines 248-253; `method-physics-informed-rerun-prereg.md:264` ("D6 found") | 1b3fbc6 | reproduction record |
| d7_residual_anatomy.py | section 5 table, lines 324-340 | 5437e52, 58021f2 | reproduction record |
| d8_hub_height_sensitivity.py | section 4 lines 317-321 | 0e9a315 | reproduction record |
| run_overnight.sh | none by name; it is the runner of the 2026-08-23 programme (E1 to E7 and the figures) | cae8a9a | reproduction record, stale |
| run_remaining.sh | none by name; the concurrent form of the sensitivity stages | ff7919d | reproduction record, stale |

The findings cite these diagnostics as a set, "the diagnostics D0 to D8"
(`method-physics-informed-rerun-prereg.md:37`), and say their figures stay
unattributable to a commit. A label grep alone misleads, because D0 to D5
also name decisions in the manuscript plan and deviations in the EU re-run.
The runners are not broken: every script and flag they call still exists,
and `bash -n` passes. They are stale in three ways. Nothing in them builds
`output/pinn/cache`, they write to the unversioned `output/pinn/`, and they
pass no `--registration`. d0 to d4 also import `RUNS` and helpers from
`scripts/analysis/ml_transfer_retest.py`.

### scripts/analysis leads

| file | named study | reported where | output on disk | status |
|---|---|---|---|---|
| refit_control_points.py | none | `docs/design/manuscript-chapters-45.md:791-803` (BE scalar 0.598; IT 4.07 and 3.01, which match the run) | `output/refit_control_points_2026-09-15/`, manifests clean at 3930f26 | driver of a recorded measurement |
| cluster_sweep_cost.py | `method-cluster-selection-prereg.md` | its timing run at prereg line 412 | yes | driver of a registered cost measurement |
| loco_reference_wind.py | `method-correction-identifiability.md` | the holdout section, line 259; `regime_coverage.py` reads its CSV | yes | driver of a live finding |
| offshore_pool_study.py | `method-offshore-pool-prereg.md` | the prereg is void (e955e17) but says "its numbers are kept" | `output/offshore_pool_2026-09-13/` | only generator of kept numbers |
| national_single_cluster_study.py | `method-national-single-cluster-prereg.md` | not run: the prereg is "blocked on data" (eecf252) | directory exists and is empty | driver of a live, blocked study |
| export_voronoi_frames.py | none | nowhere | not checked | **unclear**, see section 10 |
| pool_as_training_set.py | none by name | `method-why-corrections-do-not-transfer.md:31-37` (884 points in 340 cells against 200 in 114), matching `dk_onshore_coverage.csv` | `output/pool_training_2026-09-16/` | only reproduction of a live finding |
| surface_flag_report.py | cites `method-distance-mask.md` only as motivation | nowhere: 1,339 of 23,989 cells flagged, in no document | `output/surface_flag_2026-09-15/` | **unclear**, see section 10 |
| roughness_treatment_study.py | `method-roughness-treatment-prereg.md` | `method-roughness-treatment.md:99` cites its data directory | yes | not an orphan: `eu_rerun_compare.py:58` imports it |
| cluster_selection_gaps.py | `method-cluster-selection-prereg.md` | the gap table in `method-cluster-selection.md:83-93` | yes | driver of a live finding |

### src leads

- **`src/vwf/sources/european.py` is live.** It is the `european-turbine`
  adapter. 11 configs select it by name, including the DE, DK and UK
  scorecard rows. `harness/driver.py:78` resolves it through `get_source`. It
  is imported by `sources/__init__.py:25`, re-exported as
  `vwf.EuropeanTurbineSource`, and covered by about 13 tests in
  `tests/test_sources.py`.
- **`src/vwf/loaders/country_level_loaders.py` is partly live.**
  `load_year_specific_grid_points` is re-exported as
  `vwf.load_year_specific_grid_points` and called by `vwf.py:356`. That method
  is called by `scripts/analysis/train_all_bias_corrections.py:487`.
  `country_gen_to_cf` and `_aggregate_observations_to_monthly` have no caller.

### Dead code inside live files

No whole file is dead, but these functions and constants have no caller in
the tracked tree:

| item | evidence | note |
|---|---|---|
| `loaders/country_level_loaders.country_gen_to_cf`, `_aggregate_observations_to_monthly`, `_hours_in_month` | `git grep`: no caller; CHANGELOG records the path that reached it as a removed bug | `docs/api.md:123` lists the member, so the same commit edits it |
| `data.sim_turbines_to_country_cf` | no caller; only `docs/api.md:48` | same api.md constraint |
| `data.COUNTRY_DIR`, `TURBINE_DIR`, `COUNTRY_LEVEL_DIR` (data.py:73-75) | no user | |
| `PyVWF.from_config` (vwf.py:402-484) | no caller; inserts a path into `sys.path` and imports generated code | public API, so deprecate rather than delete |
| try/except ImportError guards in `clustering.py:20-35` | shapely and geopandas are core dependencies | |
| `datasets/process_dk_raw_data.main` (lines 286-373) | duplicates `scripts/process/dk.py` | a workaround from before the script existed (b689fed) |
| `sys.path.insert` in `generate_country_level_training_data.py:42` | inserts `src/` so the file runs by path | redundant once it runs as a module |

## 4. Merge candidates

### The curve-library family: not one capability

```
curve_library_match.py    (T1 rule: exact designation match; stdlib only)
curve_library_assign.py   (T2 rule: nearest in-band other-brand model) -> curve_match_audit
curve_library_tables.py   (builds C1, C2, T1 and T2 override tables)   -> match, assign, curve_match_audit, baseline_bootstrap
curve_library_study.py    (runs one condition of one row)              -> baseline_bootstrap; monkeypatches vwf.data.prep_country
eu_rerun_compare.py       (paired comparison, shared with the EU re-run)
curve_match_audit.py      (the scorecard's curve-match audit, CONTEXT.md)
```

These are the rules, the builder and the runner of one study, plus one
independent tool. Each of the four `curve_library_*` files has its own test
file, which pins the boundaries. **Do not merge.** Move the four together to
`scripts/studies/curve-library/`, because they import each other as
siblings. `curve_library_assign.py` also imports `curve_match_audit` as a
sibling (line 34), so the move must change that import: either load it by
path, or first promote `classify` and `curve_side_manufacturer` to `vwf`. `curve_match_audit.py` stays in `scripts/analysis/`: it predates
the study (1166ece, two days earlier), `CONTEXT.md` defines the curve-match
audit by this path, and two skills and a guide cite it.

Duplication with `src/vwf`: `curve_library_assign` restates the
specific-power formula (`vwf.data.add_models`, data.py:940) and `SCALE_BAND`
(`vwf.datasets.eia_us`, eia_us.py:56). Its in-band nearest search is the
core loop of `assign_curves_from_library` (eia_us.py:452-462). The
T2 rule is registered and study-specific, so it stays in the study. Its
constant should be imported rather than restated, or a test should check the
two agree. The monkeypatching in `curve_library_study.py` is a sign that the
harness lacks a fleet-override hook.

### ml_transfer_retest.py and ml_transfer_expanded.py: one family, two drivers

`ml_transfer_retest.py` is two things in one file:

- a library: `build_centroids`, `terrain_features`, `rf_eval`, `loro`,
  `random_cv`, `variance_decomposition`, `RF_KW`, `SEEDS`, `SET_A/B/C`;
- the round-one driver: `run_suite`, `main`, `RUNS`, the 3-of-5 gate.

Ten scripts import the library half: `scripts/pinn/` d0 to d4, `e1_loro.py`
and `prep_rf_features.py`, and `scripts/analysis/` `ml_transfer_expanded.py`,
`pool_as_training_set.py` and `regime_coverage.py`.
`ml_transfer_expanded.py` is purely the round-two driver, with its own gates.
The two share a findings document, `method-ml-transfer.md`, but are two
registered experiments.

**Verdict:** promote the library half to `src/vwf/extensions/ml/`. The `ml`
extra in `pyproject.toml` already names that module, and it does not exist.
Keep `ml_transfer_retest.py` at its path as the round-one driver still
defining `RUNS`. Three findings documents cite that path, and
`method-physics-informed-rerun-prereg.md:65` names the July factor files "in
`scripts/analysis/ml_transfer_retest.py`". Round one must reproduce bit for
bit, so a test pinning `terrain_features` and `loro` on a small synthetic
frame comes first. The name `roughness` in `terrain_features` is terrain
elevation roughness, not the z0 that `CONTEXT.md` calls roughness. The
promoted name should differ.

### regression_run_harness.py and regression_run_legacy.py: one capability, three files, keep

The legacy/harness regression check (`method-harness-regression.md`) runs the
legacy path, runs the harness path, and diffs the frames. The two runners
must run under different code: the legacy half from a worktree of `main`, the
harness half from the branch. A merged file would still need three
invocations, and it would be executed from a checkout whose own code the
legacy subcommand does not use. `regression_compare.py` is a general,
tested frame comparator (`tests/test_regression_compare.py`). The findings
document cites all three paths. **Keep all three in `scripts/analysis/` as
tools.** They are also the right instrument for proving that Phases 2 and 3
preserve numbers, beside the golden test.

### Genuine merges

| merge | evidence |
|---|---|
| `tests/test_harness_era5.py` into `tests/test_era5.py` | three tests byte-identical, same names (test_era5.py:20-53 and test_harness_era5.py:19-52, confirmed with `diff`); both came through merge e15ba4a; the subject is `vwf.datasets.era5`, not the harness. Keep the two tests unique to test_harness_era5.py. |
| `PIPELINE.md` into `docs/guides/training.md`, as a "Legacy batch path" section | section 6 |
| `src/vwf/datasets/COMBINED_ERA5_USAGE.md` into `docs/guides/data-sources.md`, the ERA5 section | a user guide inside the package, never shipped (not package-data); no docs page covers `era5/EU`. `scorecard.md:281` cites the path, so the move leaves that citation stale. |

## 5. Logic that belongs in the package

### The 11 analysis scripts that import nothing from vwf

| script | duplicates in src/vwf | verdict |
|---|---|---|
| chapter_capacity_weights.py | `haversine_km` duplicates `vwf.extensions.grid.interpolation.degree_distances(metric="great_circle")` | **Do not swap it in.** The script reproduces the chapter's own rule, with an arctan2 form where vwf uses arcsin. A point on the 50 km boundary could flip and break a reproduction that is exact to 1e-6 MW. Study-specific. |
| correction_identifiability.py | none (`git grep -i "pivot\|reference_wind" src/vwf` is empty) | study-specific; `pivot()` is copied in `pivot_probe.py:72`, a script-to-script duplicate |
| curve_library_assign.py | specific power (data.py:940), `SCALE_BAND` (eia_us.py:56), in-band nearest (eia_us.py:452-462) | import the constant; the rule itself is registered |
| curve_library_match.py | none; `add_models` does fuzzy manufacturer matching, which this module deliberately rejects | study-specific, new logic |
| era5_overlap_check.py | partial: `_standardise` overlaps `vwf.datasets.era5.unify_time_coordinate` and the rename in `prep_era5` | cannot use `prep_era5`, which slices, normalises and derives roughness: the raw comparison must avoid all three. Study-specific. |
| loco_reference_wind.py | none; `loco_interpolation.skill` has no exact counterpart (`vwf.harness.skill` has no R2) | study-specific |
| ml_transfer_expanded.py | none | driver only |
| ml_transfer_retest.py | `build_centroids` duplicates the centroid block in `vwf.harness.export` (export.py:138-142); `terrain_features`, RF and leave-one-out have no src equivalent | promote to `vwf.extensions.ml` (section 4) |
| pool_as_training_set.py | the plausibility screen (lines 144-148) restates `vwf.extensions.grid.surface.correction_surface` (surface.py:371-377) with literals in place of `PLAUSIBLE_SCALAR` and `MAX_ZERO_CROSSING_SPEED` | promote one `plausibility()` helper to `vwf.extensions.grid.surface`; the same screen is also in `refit_control_points.py:54-63` and `unmasked_surface_bands.py:202-203` |
| regime_coverage.py | none in src; its coverage bins duplicate `pool_as_training_set.coverage()` | study-specific |
| regression_compare.py | none; src has no frame comparator | keep as a tool |

### Reusable logic in scripts that do import vwf

| logic | where now | proposed home | constraint |
|---|---|---|---|
| paired bootstrap (unit and month resampling, weighted RMSE and MAE, percentile interval) | `baseline_bootstrap.py:198-226`, and rewritten inline in `common_row_rescore`, `eu_rerun_compare`, `roughness_treatment_study`, `unit_concentration` | new `vwf/harness/bootstrap.py` beside `vwf.harness.skill` | Keep the RNG call order bit-identical (`default_rng(SEED)`, `rng.integers` of shape `(N_DRAWS, n)`, one draw per call site), or the cited intervals stop reproducing. Pin one row's interval in a test first. |
| `load_obs_and_fleet` ("`val_set` without `prep_era5`") | `baseline_bootstrap.py`, used by 5 scripts | split `vwf.data.val_set` into an observation-and-fleet stage and an ERA5 stage, or expose `vwf.harness.driver.evaluation_inputs` | `val_set` is on the golden path |
| `country_monthly` | `baseline_bootstrap.py` | delete for `vwf.harness.driver._country_pairs`, the same arithmetic | |
| scorecard lookups `CONFIGS`, `REPORTED` | `baseline_bootstrap.py`, imported by 10 scripts | a resolver over `configs/regions/scorecard/*.toml`, which carry `code` | |
| "rebuild an evaluate run's variant frames and check them against the saved ones" | `missing_value_audit.py:75-109`, `off_curve_sensitivity.py:88-164` | a public `vwf.harness` helper | |
| GWPT loading, filtering and name normalising | `load_gwpt` and `fleet_for` in `region_tools/weight_country_grid_points.py` (imported by `repair_country_capacity.py` through `sys.path`); four normalisers (`process/cammesa_ar.norm`, `process/cen_cl.norm`, `datasets/windstats._norm`, `datasets/aemo_au.normalise_farm_name`); the country-and-operating filter in three process scripts | new `src/vwf/datasets/gwpt.py` | each normaliser may differ in detail; pin current outputs first |
| NZ capacity history and mask: `gen_code_map`, `capacity_history`, `mask_from_windows`, the fuel-code loop | `scripts/process/emi_nz.py` only, untested | `vwf.datasets.emi_nz` | **the production path is the untested one**; the tested register-based versions in src are unused (datasets/emi_nz.py:35-38) |
| AR and CL join logic: `norm`, `gwpt_argentina` or `gwpt_chile`, `join_coords_caps` or `match`, `load_generation`, the `EXCLUDE` lists | `scripts/process/cammesa_ar.py`, `cen_cl.py`, untested | `vwf.datasets.cammesa_ar`, `vwf.datasets.cen_cl`; the `EXCLUDE` lists go to `configs/curation/` | pin the processed CSVs first |
| AEMO alias overrides | `scripts/process/aemo_au.py:90-103` | `vwf.datasets.aemo_au` | small |
| z0 inversion from the 10 m and 100 m winds | three copies: `prep_era5` (era5.py:224-247), `combine_era5_files.calculate_roughness_from_winds`, `scripts/era5/combine.py:66-75` | one `derive_roughness` in `vwf.datasets.era5` | the US and BR scorecard rows rest on `combine.py`'s output: bit-identity test first |
| `region_spec(code)` | identical in `scripts/fetch/era5.py:168` and `scripts/era5/combine.py:40` | `vwf.harness.regions.load_region_by_code` | |
| generation-to-CF conversion, hours in month times capacity | written out 12 times across adapters and dataset transforms | `vwf.time_utils.hours_in_month` | keep each call site's order of operations: `european.py` divides by `days * 24.0 * capacity` in one expression |
| cluster centroids | `ml_transfer_retest.build_centroids`, `harness/export.py:138-142` | `vwf.harness` | |

### Private harness API used by scripts

Scripts reach into `driver._era5_dir`, `_tidy_eval_frame`, `_country_pairs`,
`_score_on_common_rows`, `_SCOPE_KEYS`, `_error_metrics`, `_country_skill`,
`vwf.correction._find_offset_iterative` and `vwf.wind._get_power_curve_cache`.
Each is a promotion signal: make it public or wrap it.

## 6. Interface inconsistency

### Analysis scripts without argparse (34 of 45)

baseline_bootstrap, chapter_capacity_weights, cluster_selection_gaps,
cluster_selection_study, cluster_sweep_cost, common_row_rescore,
correction_identifiability, curve_library_assign, curve_library_match,
curve_library_study, curve_library_tables, domain_split_study,
era5_overlap_check, eu_rerun_compare, extent_audit, hourly_resolution_test,
loco_interpolation, loco_reference_wind, min_cluster_size_tradeoff,
missing_value_audit, ml_transfer_expanded, ml_transfer_retest,
national_single_cluster_study, off_curve_sensitivity, offshore_pool_study,
pivot_probe, pool_as_training_set, refit_control_points, regime_coverage,
roughness_treatment_study, surface_flag_report, training_objective_check,
unit_concentration, unmasked_surface_bands.

Two of them, `curve_library_assign` and `curve_library_match`, are libraries
with no `main()` and no `__main__` guard. Elsewhere in `scripts/`, seven files
lack argparse: `pinn/` d0 to d4, d6 and `prep_rf_features.py`. Every other
`.py` file under `scripts/` and `examples/` has a `__main__` guard.

Most of the 34 take positional `sys.argv` (usually `<out_dir>`). The common
hardcoded inputs are these:

- the control-point pool `output/pyvwf_to_grid/all_corrections_centroids.csv`,
  in 11 scripts;
- the chapter grid `np.arange(-10, 30.01, 0.25)` by `np.arange(35, 72.01, 0.25)`,
  in 4;
- dated run directories: `refresh_2026-08-24`,
  `curve_resolution_backfill_2026-09-11`, `cluster_selection_2026-09-15` and
  `cluster_sweep_2026-07-24`;
- `input/...` paths written literally, so they bypass `PYVWF_INPUT`:
  `pivot_probe.py:94`, the ETOPO path in `ml_transfer_retest.py`, and the
  shape paths in three grid scripts.

Two are defects rather than style:

- `training_objective_check.py:54` hardcodes the refresh training directory.
  So the committed script cannot run the re-run on the new fits that
  `method-eu-rerun-prereg.md:166-168` registers.
- `hourly_resolution_test.py` writes a fixed `output/hourly_test/` path, so a
  re-run overwrites the record.

**Registered constants must stay in code.** Several drivers state that the
pre-registration fixes their parameters and that this is deliberate: seeds,
draw counts, gates, cluster grids, reference winds, fold definitions. A flag
would let a run differ from its record without trace. The rule for Phase 3:

- paths become arguments, with the recorded path as the default, so the
  recorded invocation still reproduces;
- registered parameters stay as module constants;
- the docstring names the registration.

### Proposed shared helper: `vwf.cli.common`

One small module, used by every entry point in `scripts/` and `src/vwf/cli/`:

```python
def make_parser(doc: str) -> argparse.ArgumentParser
    """Parser whose description is the module docstring's first paragraph."""
def add_out_dir(p, *, default: Path | None = None) -> None
    """--out, refusing a path outside output/ and a session scratch dir (AGENTS.md)."""
def add_region(p, *, scorecard: bool = False) -> None
    """--region CODE or --config PATH, resolved by vwf.harness.regions."""
def add_input_root(p) -> None
    """--input-root, defaulting to PYVWF_INPUT, resolved through PyVWFPaths."""
def input_path(*parts) -> Path
    """Resolve under the input root the way the loaders do, never a literal input/."""
def run(main: Callable[[argparse.Namespace], int]) -> NoReturn
    """Parse, configure logging, call main, exit with its code."""
```

`input_path` fixes the `PYVWF_INPUT` split between the fetch and process
scripts. `add_out_dir` puts the rule "Run under `output/`" into code. Existing
parsers stay as they are until a file is touched for another reason. The
harness CLI `scripts/analysis/validate_region.py` moves to
`src/vwf/cli/validate.py` behind a `pyvwf-validate` console entry, and the
script stays as a shim, because 15 or more procedural documents cite its path.
Today `[project.scripts]` exposes only `pyvwf-train`, the legacy path, so a
user who installs with pip does not get the preferred harness at all.

## 7. Documentation overlap

### One home per fact

| fact | stated today in | contradictions | proposed home | others become |
|---|---|---|---|---|
| install | README.md:43-59; CONTRIBUTING.md:22-34; environment.yaml | README.md:251 says dependencies are pinned in `environment.yaml`; they are not. `environment.yaml` installs the `[data]` extra that CI omits, and lacks mypy, pytest-cov and pandas-stubs. README lists five extras and omits `[grid]` and `[ml]`, and `[ml]` points at a module that does not exist. | README.md for users; CONTRIBUTING.md for developers | environment.yaml gets a header saying which extras it mirrors |
| run the tests | CONTRIBUTING.md:36-45; ci.yml; STATUS.md:348-368 | none in public files | CONTRIBUTING.md | STATUS goes (Phase 4) |
| choose the input root (`PYVWF_INPUT`) | README, training.md, input/README.md, data-sources.md, the adding guide, five runbooks, scripts/README.md, Dockerfile, compose | **three routes to the licensed library.** `input/README.md:44-46` copies it over `power_curves.csv`, which `tests/test_committed_files.py:38-39` calls a mistake. `training.md` and `data-sources.md` use `PYVWF_INPUT=input/combined`. `runbooks/us.md:91-92` names `power_curves.real.csv`, which no code reads. `ar.md`, `cl.md` and `dk.md` set the variable on `train` only. | a "Choose the input root" section in `docs/guides/training.md` | the rest link to it; the copy route is removed |
| run a region through the harness | README, training.md, scripts/README.md, the adding guide, every runbook, design/harness.md:189-199, compose | `harness.md` puts `--region` before the subcommand, uses a nonexistent `--factors-from` and claims a config hash in the run id | `docs/guides/training.md` | README keeps one example; `harness.md` drops its command block |
| the legacy batch path | PIPELINE.md; README.md:79-87; scripts/README.md; visualisation.md; api.md; the adding guide; data-sources.md section 4 | PIPELINE.md:53 names a set that no longer exists; PIPELINE.md:16-17 says the grid code lives on `development`, but it is on this branch; `visualisation.md:8` says `load_results` reads any run directory, but it reads only the legacy layout; `api.md:11` calls `PyVWF` "the entry point" | a "Legacy batch path" section at the end of `training.md` | PIPELINE.md is deleted |
| output layout | output-structure.md, PIPELINE.md, README, visualisation.md, your-own-data.md | three roots in use: `output/validation/`, `output/runs/` and `outputs/`. `outputs/` is not git-ignored, so the README quickstart dirties the tree and sets `git_dirty` in every later manifest. | `docs/reference/output-structure.md` (section 7, Diataxis), with a legacy section | examples write to `output/` |
| add a region | the adding guide (15 files); README.md:38-39; training.md:96-97; the new-region skill | README and training.md say "one adapter and one config", while the guide lists about fifteen files | the adding guide | "no core module changes; the guide lists the files" |
| add an adapter | the adding guide; your-own-data.md for the no-adapter route | none; the guide's filename uses the rejected term "observation source" | the adding guide, renamed | |
| ERA5 fetch and combine | data-sources.md, scripts/README.md, the adding guide, runbooks, `COMBINED_ERA5_USAGE.md` | which boxes need combining differs by file; `es.md:38 --region es-ws` exits (the config is `es_ws.toml`) | `data-sources.md`, ERA5 section | |
| the curve library | input/README.md, README, data-sources.md, resources/__init__.py, the adding guide | the licensed-library route (above); "bundled library" is a rejected term in two places | `input/README.md` for content and licence; the route goes to `training.md` | |

**PIPELINE.md** says four true things that exist nowhere else:

- that `generate_country_level_training_data.py` writes
  `input/observations/country/pyvwf_config.py`;
- the `train_all_bias_corrections.py --list` and `--sets` interface, whose
  set table is stale;
- the `output/runs/<prefix>/` layout;
- the `--prefix` output of `evaluate_all_pyvwf_runs.py`.

These move into the legacy section. The rest is covered or wrong.
`AGENTS.md:14` is the only place that says PIPELINE.md is legacy.

### Diataxis placement

| file | kind | now in | proposed |
|---|---|---|---|
| guides/data-sources.md | reference, with one how-to (country-level workflow) | guides | `docs/reference/data-sources.md`; the country-level how-to moves into the adding guide |
| guides/output-structure.md | reference | guides | `docs/reference/output-structure.md` |
| api.md | reference, but it misses `vwf.harness` and documents 4 of 14 `vwf.sources` modules | root of docs | `docs/reference/api.md`, extended |
| runbooks/tr.md | a record of a data source evaluated and not shipped | runbooks | keep; retitle as feasibility |
| design/manuscript-chapters-45.md | a working plan with open decisions | design (published, in no toctree) | keep the path (9 findings cite it) and add it to `exclude_patterns` |
| (none) | tutorial | gap | a short tutorial page over `examples/run_minimal.py` |

Two tracked files link to git-ignored artefacts, so the links are dead in any
clone:

- `method-harness-regression.md:77` cites `TURBINE_GRID_EVALUATION_ANALYSIS.md`;
- `examples/dk_raw_vs_corrected.ipynb:31` embeds
  `docs/findings/figures/dk_onshore_rmse_vs_k.png`.

### Extension guides

| question | answered by | gap |
|---|---|---|
| add a turbine-level region | `docs/guides/adding-an-observation-source.md`, driven by the new-region skill | the filename uses a rejected term |
| add a country-level region | nobody, end to end | the guide and the skill defer to `generate_country_level_training_data`; zone assignment is only a script list |
| add an adapter | the same guide | the API reference omits 10 adapter modules |
| add a study | **nobody** | where the driver lives, where its configs live (`configs/regions/study/` and `pinn_loco/` have no README), the output convention `output/<study>_<date>/`, how a test loads a driver, and the order: pre-registration commit, driver commit, run, findings |

**Proposed vocabulary row for `CONTEXT.md`:** "study": one question with its
own findings document, pre-registration, driver directory
(`scripts/studies/<stem>/`) and run directories
(`output/<stem>_<date>/`). Do not use: experiment (for a study), analysis
(for a study).

The new-region skill and the guide overlap on purpose. The skill says "Do
not keep a separate list", and its phases match the guide's table row for
row. It adds human stops, `git_dirty` checks and the rule to stop on a
non-zero substituted share.

## 8. Module size

| module | lines | proposal | why |
|---|---|---|---|
| `datasets/generate_country_level_training_data.py` | 1434 | **split** into grid-point generation (lines 190-682, to `vwf/datasets/country_grid.py`), the ENTSO-E observation fetch (lines 685-1027, into `fetch_entsoe_capacity_factors.py` or a new `entsoe_country_obs.py`), and `pyvwf_config.py` writing (lines 1030-1207, deprecate with the legacy country path). `main` stays as a thin CLI at this path. | Three groups that never call each other; only `main` joins them. Nothing imports the file, and the golden test does not touch it. It is excluded from coverage and mypy today, so the split also makes the grid group testable. |
| `data.py` | 1042 | **split two seams.** Curve assignment (`load_power_curves`, `add_models`, `_default_power_curve`, about 150 lines) goes to a new `vwf/curves.py`. The country-level helpers (`prepare_country_fleet`, `country_cf_to_monthly`, the zonal helpers, `assign_country_clusters`, about 330 lines) go to `vwf/country_level.py`. Keep re-exports in `data.py`. | `vwf.curves` removes both upward imports from adapters to `vwf.data` (section 9), and the lazy-import workarounds go with them. The remaining 550 lines are the pinned orchestration and stay. Patch targets move: `tests/test_sources.py:91` patches `vwf.data.add_models`. |
| `vwf.py` | 922 | **keep whole.** Deprecate `from_config` only. | One class, the legacy reference. The golden test rebuilds "PyVWF.train as it runs" from its parts, so the class body is the specification. Moving `train` would only relocate a method. |
| `clustering.py` | 915 | **split.** Runtime clustering (`cluster_turbines` and helpers, about 165 lines) stays. Cluster geometry (region shapes, repair, `get_country_shape`, `cluster_with_geometries`, about 450 lines) goes to `vwf/cluster_geometry.py`, or merges with `vwf.geospatial`, which loads the same region geometry with a stricter policy. Country sampling points (about 260 lines, used only by the country generator) go to `vwf/datasets/country_grid.py`. | One cross-call between the groups. The geometry dependencies leave the core import. Patches in `tests/test_region_shape_repair.py:83-84` must follow the module globals. |

Every move keeps each name importable from its old module, because
`docs/api.md` documents these modules and the docs build runs with `-W`.

## 9. Architecture, for the import-linter contract

This layering is derived from an AST pass over all 78 `.py` files in
`src/vwf`, with module-level and function-level imports kept separate. It is
not guessed. No module in `src/` imports from `scripts/`.

```ini
[importlinter]
root_package = vwf

[importlinter:contract:vwf-layers]
name = PyVWF layers
type = layers
layers =
    vwf.cli | vwf.extensions | vwf.pinn | vwf.viz
    vwf.harness
    vwf.vwf
    vwf.data
    vwf.sources
    vwf.loaders | vwf.datasets
    vwf.correction
    vwf.clustering | vwf.wind
    vwf.config | vwf.utils | vwf.time_utils | vwf.metrics | vwf.geospatial | vwf.resources
ignore_imports =
    vwf.sources.european -> vwf.data
    vwf.sources.client_csv -> vwf.data
    vwf.vwf -> vwf.harness.provenance
    vwf.harness.provenance -> vwf
    vwf.harness.export -> vwf
```

Two inner contracts cover `vwf.harness`
(`driver | export | hindcast` over `provenance | corrections | skill` over
`regions`) and `vwf.pinn`
(`train | runs` over `cache` over `era5_stats | model | physics | terrain | gwa`).
Checked by hand against every edge, the contract has zero violations with
the five ignores. Each ignore is a real defect to remove, not a permanent
exemption:

1. `sources/european.py:79` and `sources/client_csv.py:133` import
   `vwf.data.add_models` lazily to dodge a cycle. `vwf.curves` removes both.
2. `vwf.py:243`: the legacy class imports `vwf.harness.provenance`. Move
   manifest writing below both paths, to `vwf.provenance`, with the harness
   re-exporting it.
3. `harness/provenance.py:29` and `harness/export.py:39` import the root
   package only for `__version__`. That pulls in the legacy class and
   matplotlib, and closes the cycle in item 2. The fix is a `vwf/_version.py`,
   which also means changing `pyproject.toml`'s version attr and the
   assertion in `tests/test_packaging.py:58`.

`vwf.datasets` is not one layer. `datasets.era5` is core runtime: `data`,
`harness` and `pinn` import it, and it imports `wind`. The other `datasets`
modules are acquisition-time leaves. Moving `era5.py` to `vwf/era5.py`, with
a shim, would make `datasets` a clean acquisition layer.

Separately, `vwf.metrics` (legacy) and `vwf.harness.skill` (harness) each
define the error metrics, and the code is not shared. Both are live. Leave
them, but a change of metric definition must touch both.

## 10. Decisions needed before Phase 2

**Q1. Move the study drivers, and at what cost?** 50 script files would move.
21 of them are cited by exact path in a dated findings document, usually in a
command: 15 in `scripts/analysis/` and 6 in `scripts/pinn/`. Five tests load
a moving driver by path and change with it. A sixth,
`test_eu_rerun_compare.py`, breaks indirectly, because
`eu_rerun_compare.py` imports `roughness_treatment_study.py`. The options:

- (a) Move everything to `scripts/studies/<stem>/`. Add
  `scripts/studies/README.md` with an old-path to new-path table and the last
  commit at the old path, so every stale command in a findings document
  resolves in one lookup, and `git checkout <commit>` still runs it as
  recorded. **Recommended.**
- (b) Freeze the existing drivers in place, adopt the convention for new
  studies only, and index them all in `scripts/README.md`. This is cheaper,
  but `scripts/analysis/` stays a mix of tools and studies.
- (c) Move only the uncited drivers. **Not recommended:** it splits families,
  such as the curve-library study and `scripts/pinn/`, across two trees.

**Q2. `scripts/pinn/`: when?** The turbine-only study's registered commands
name `scripts/pinn/turbine_loro.py` and `gwa_ratio.py`, and it runs after the
MaStR download. Moving before that run makes its registration wrong on the
day it is executed. The proposal is to move the directory as one unit after
the run.

**Q3. `export_voronoi_frames.py`: unclear.** No referrer anywhere, and it
imports `mapbox_earcut`, which `pyproject.toml` does not declare. It is the
TouchDesigner visualisation's geometry exporter. Is that visualisation still
in use? If yes: keep, and declare the dependency in an extra. If no: delete.

**Q4. `surface_flag_report.py`: unclear.** It was run (1,339 of 23,989 cells
flagged) but the figure is in no document. Do you want the product flag's
share recorded? If yes, it moves with that record. If no, delete: the run log
is kept, and git keeps 4e34eb8.

**Q5. The two unreferenced notebooks: unclear.**

- `examples/dk_raw_vs_corrected.ipynb` claims open data only, but defaults
  to the licensed `input/combined` library. It also embeds a git-ignored
  figure. Fix it, or is it superseded by the README quickstart?
- `examples/notebooks/northsea_data.ipynb` is self-described as not
  runnable and reads the legacy `input/country-data/`. Does anything still
  read its outputs? If not, delete.

**Q6. The seven Italian bidding-zone polygons: unclear.** Both zone loaders
skip non-numbered stems (`assign_country_zones.py:61-62`,
`weight_country_grid_points.py:84`), so nothing reads `IT_*.geojson`. Is an
Italian zonal path planned? If not, delete them and their `zones/README.md`
entry.

**Q7. `STATUS.md`.** Phase 4 untracks it. The cost is permanent:
`method-cluster-selection.md:135` links to it by relative path, and two other
findings name it. It also names local paths and a detached download in a
public file. The alternative is a tracked one-line stub that says session
state is kept locally.

**Q8. `CONTEXT.md` under `docs/`.** The proposal is `docs/CONTEXT.md`,
keeping the name, because 21 dated findings cite it by that name. The docs
naming rule gets an explicit exception, like `README.md`. `AGENTS.md:130`
(`@CONTEXT.md`) must change in the same commit, or the vocabulary silently
drops out of agent context. Should it be published on the site as
reference?

**Q9. This audit on the published site?** It carries `orphan: true` today.
The alternative is excluding it with the manuscript plan.

**Q10. `docs/reference/`?** Section 7 proposes moving `data-sources.md`,
`output-structure.md` and `api.md` into a `docs/reference/` folder. Your
stated split names `api.md` alone as reference. Adopt the folder, or keep
the two reference pages under `guides/` and say so in `docs/README.md`?

**Q11. Commercial framing.** Remove "offers 068 / 015" and "offer 103
artifact" from the two docstrings as the first Phase 2 commit. They remain
in history. Rewriting history is a separate decision, and it is yours.

## 11. Defects found in passing

Neither is fixed here. Each is queued for the phase that touches the file.

**Correctness and reproducibility**

- `training_objective_check.py:54` cannot reach the re-run that
  `method-eu-rerun-prereg.md:166-168` registers. `method-eu-rerun.md:248`
  reports G5 passed with no data path.
- `method-distance-mask.md:56` pins `unmasked_surface_bands.py` at `bdb5f66`,
  but the screen behind its correction-notice table was added later in
  `171f530`.
- `baseline_bootstrap.py` and `common_row_rescore.py` read the 2026-09-11
  run outputs against scorecard configs replaced on 2026-09-13. A 1e-12
  check guards the worst case.
- `cluster_selection_study.py` and `hourly_resolution_test.py` read the
  maintained region configs. An edit to a maintained config silently changes
  what these recorded drivers reproduce.
- `curve_library_tables.py:100` and `curve_match_audit.py:178` pick a file
  with `sorted(glob)[0]`. The scorecard family uses
  `next(glob("evaluate-*-backfill"))`. AGENTS.md forbids resolving files this
  way. Each is harmless today, with exactly one match.
- `emi_nz`: the production capacity history and mask are the untested code.
  The tested versions are unused.
- `output/ml_retest/` lacks two files the scripts write. The one file present
  predates the first commit of the script that writes it.

**Environment and docs**

- `docker-compose.yml` mounts `/data/output`, but `validate_region.py`
  writes under `/app/output`. The documented command's results are deleted
  by `--rm`.
- `outputs/` is not git-ignored, but the README quickstart,
  `examples/quick_run.py` and `visualisation.md` write there.
- `examples/viz_demo.py` overwrites six tracked `docs/img/*.png` files, so
  running it dirties the tree.
- `scripts/fetch/epias_tr.py:14-20` recommends a plaintext credentials file.
  That conflicts with "Write no credential into a file". The environment
  variable route already exists.
- The `[ml]` extra names a module that does not exist.
- `CONTRIBUTING.md:75` says NumPy docstrings, but the code and `conf.py` use
  Google style: 193 `Args:` sections against one `Parameters`.
- `docs/index.md:91-92` omits two authors from the method citation.
- `scripts/README.md` lists 7 of 45 analysis scripts, omits `scripts/pinn/`,
  and names `ES.md` where the file is `es.md`.
- The local `CLAUDE.md` says 471 tests in 42 files. The tree has 66 test
  files with 720 test functions.

**CI coverage gaps** (for Phase 5)

- All 52 tests in `test_pinn_physics.py` skip in CI, because torch is not
  installed. Tests needing pykrige or rasterio also skip.
- CI lints `src/vwf tests` only, and runs only `run_minimal.py`.
- No pre-commit configuration and no dependabot.
- Actions are pinned by major tag, not by SHA.

**A process note from this audit.** An audit helper briefly wrote
`edges.json` into the repository root, and it was moved out within seconds.
No run that writes a manifest was in flight: only the MaStR download, which
records no git state. The tree was clean when this document was written.
