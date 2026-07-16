# Multi-Region Validation & Method Expansion — Progress Log

Running log for the multi-region validation research task. Updated at each
checkpoint. Captures what was done, what was learned, what surprised us, and
open questions — not just a change list.

---

## 2026-07-15 — Phase 0: Diagnosis and scoping (checkpoint 1)

**Status:** Phase 0 audit complete. Awaiting sign-off on: branch name,
validation matrix, research agenda. No code written, no branch created yet.

### What was done
- Read-only audit of the pipeline (`src/vwf/`), test suite (`tests/`),
  training/evaluation scripts (`scripts/`), and available data (`input/`).
- Full sweep for hardcoded UK/Europe assumptions (see Phase 0 report in
  conversation; key items summarised below).
- Baseline test-suite run in a clean venv to confirm main is green.

### What was learned
- The correction is genuinely affine **on wind speed**, not on CF:
  `cor_ws = scalar · unc_ws + offset` (wind.py:336), per
  (cluster × time-slice × year). The scalar is a capacity-weighted obs/sim CF
  ratio; the offset is found numerically by pushing wind through the power
  curve until simulated CF matches observed. So the "affine" correction is
  already non-linear in CF space — extensions should be framed in wind-speed
  space to stay comparable.
- Two distinct training branches exist: turbine-level (DK/DE/UK, monthly
  per-turbine observations) and country-level (9 countries, ENTSO-E country
  CF, grid-point sampling, joint L-BFGS-B offset optimisation across
  clusters). These share the simulation core but not the training loop.
- The `ObservationSource` abstraction (sources/base.py) is already the right
  seam for new regions: a new country = one adapter. The test suite even
  contains a fictional "ZZ" country adapter demonstrating this.
- All 141 tests are synthetic; no real ERA5/ENTSO-E data in CI. Real-data
  validation happens only through the scripts, manually.

### What surprised us
- The offset for country-level mode is optimised *jointly* across all
  clusters against a single country CF — an underdetermined problem (many
  cluster offsets, one observation per period). Regularisation is only
  implicit via L-BFGS-B bounds (±10 m/s) and maxiter=50. Worth a research
  question: how identifiable are these offsets, and do they overfit?
- Northern-Hemisphere seasons are hardcoded in two places
  (time_utils.py:12-24 and 77-90): `winter=[12,1,2]`. Any Southern-Hemisphere
  region gets physically inverted "seasonal" corrections silently.
- `models.csv` (turbine catalog, user compilation) and `power_curves.csv`
  (VWF-lineage curves, redistribution-sensitive) are joined only by a shared
  string key; country-level grid points get assigned whatever the *first
  column* of the power-curve file is (`_default_power_curve`,
  data.py:99-104). The real/synthetic distinction is a manual
  `cp *.real.csv` step, not code.
- The DK cleaning rule (drop turbines with any month CF ≤ 1%,
  data.py:216-219) is a country-specific branch inside shared code.

### Open questions (carried forward)
- What is the correction actually absorbing — ERA5 wind bias, power-curve
  mismatch, availability/curtailment losses? With synthetic placeholder
  curves this is confounded; real curves needed for any attribution claim.
- Is the country-level joint-offset optimisation identifiable, or do cluster
  offsets trade off against each other?
- How should seasons be defined for a global tool — hemisphere-aware named
  seasons, or purely month-based slices?
- ERA5 longitude convention: code assumes [-180, 180]; a 0–360 wrap is
  commented out in era5.py. Matters for any region crossing the antimeridian
  and for some ERA5 download routes.

### Next checkpoint
Phase 1 framework design — after sign-off on branch name, validation matrix,
and research agenda.

---

## 2026-07-15 — Phase 0 sign-off received; Phase 1 design (checkpoint 2)

**Status:** Branch `multi-region-validation` created off main at 53c4330.
Phase 1 design written (`docs/HARNESS_DESIGN.md`), awaiting review. No
implementation code yet. Nothing committed.

### Scope decisions recorded (from sign-off)
- Matrix: Europe re-runs (DK/DE/UK turbine + 9 ENTSO-E countries) + AU/NEM.
  US and Brazil deferred. Transfer restricted to the AU↔Europe pair only.
- RQ1–RQ4 committed. RQ5 hard-gated on RQ1 residual curvature (flat
  everywhere ⇒ RQ5 closed as "not needed" and documented). RQ6 local-only
  against the real curve library, bounded to a two-way synthetic-vs-real
  comparison of fitted corrections; no real curves in repo or CI. RQ7 is a
  descriptive covariate notebook only.
- Phase 1 blocker: run provenance (curve-library identity in every run's
  output) must land before any RQ6 run.
- Commits: user is sole author. User executes all merges/pushes/releases.

### Design decisions proposed (see docs/HARNESS_DESIGN.md)
- Region = one TOML file in `configs/regions/`; seasons are explicit month
  lists per region (kills the NH-hardcoding hazard by construction and
  supports non-meteorological splits later for RQ2).
- `CorrectionModel` interface with the affine baseline as a thin delegate to
  the untouched legacy code, pinned by a golden bit-for-bit regression test.
- `run_manifest.json` with SHA-256-based synthetic-vs-external curve-library
  detection; the manual `cp *.real.csv` workflow is preserved.
- ERA5: longitude normalisation to [-180,180] on load; config-driven paths;
  legacy constant kept as default.
- AEMO adapter contract fixed in Phase 1 against synthetic fixtures; real
  ingest and ERA5-AU download are Phase 2 items needing user approval.

### What was learned / flagged
- NEM SCADA records curtailed output, so AU corrections will absorb
  curtailment more than European monthly generation data does. Recorded as a
  standing caveat for every AU finding; correcting via semi-dispatch cap
  flags is explicitly out of scope.
- TOML costs no new dependency (tomli already shipped for py3.10).

### Open questions (for design review)
- `vwf.harness` vs `vwf.validation` naming; configs at repo root vs input/;
  hemisphere flag vs explicit season months (explicit proposed); whether
  legacy PyVWF.train also writes the provenance manifest; script vs console
  entry point for the driver.

### Next checkpoint
Phase 1 implementation — after design sign-off.

---

## 2026-07-15 — Design review amendments A–D applied (checkpoint 3)

**Status:** `docs/HARNESS_DESIGN.md` rev 2 written, awaiting confirmation
before implementation. Decisions 1–5 resolved (vwf.harness; configs/ at repo
root; explicit season months; legacy manifest writing with own-commit +
never-abort conditions; driver as script).

### What was done
- Transfer semantics made normative (§7): capacity-weighted collapse of
  source factors to one (scalar, offset) per time-slice, applied uniformly to
  the target; seasonal slices match by season NAME per the target config's
  definitions; no spatial matching schemes on this branch.
- Added `obs_unit` granularity field and verified it against the raw
  observation files (§2). Added the AEMO timezone decision: UTC monthly bins
  pipeline-wide, recorded in the manifest (§2, §6).
- Test plan gained two "must-distinguish" tests: the mirrored-hemisphere
  transfer fixture (name- vs month-matching must produce different outputs,
  name-matching correct) and the weighted-vs-unweighted collapse test (§8).

### What was learned (data verification)
- **UK is farm-level in turbine costume.** 360 ROC stations expanded to
  6,604 turbine rows; every multi-turbine station in 2015 has *identical*
  monthly values across its rows. Per-row CF median 0.275 confirms the farm
  generation was equally pre-split across turbine rows (the farm-total
  interpretation gives an impossible 0.008). Effective sample size for UK
  skill claims is 360, not 6,604 — skill.py must count independent units.
- **DE is genuinely unit-metered, contra the review's expected "farm"
  label**: 11,433 unit IDs; outputs within a station prefix are all distinct
  (checked Jun 2016: 532 multi-unit prefixes, zero with identical values).
  But geolocation is postcode-centroid only, so DE is classified
  `obs_unit="turbine"` with `location_resolution="postcode"`. Spatial claims
  for DE are bounded by postcode resolution (affects RQ3).
- DK confirmed per-turbine (per-GSRN metering, distinct values).

### What surprised us
- The UK pseudo-replication has been silently inflating the apparent number
  of independent UK observations in every historical evaluation. None of the
  existing metrics code deduplicates to stations; capacity-weighted country
  aggregates are unaffected (weights sum the same), but per-unit scatter,
  counts, and any i.i.d.-flavoured reasoning were optimistic. Worth a note in
  any Europe re-run write-up.

### Open questions
- Whether DE postcode-centroid geolocation materially damps DE cluster
  corrections relative to DK's true coordinates (testable in Phase 2: DK vs
  DE skill-vs-cluster-count curves).

### Next checkpoint
Phase 1 implementation batches — after rev 2 confirmation.

---

## 2026-07-15 — Phase 1 Batch 1 implemented (checkpoint 4)

**Status:** Rev 2 confirmed. Batch 1 (harness core) implemented and green;
diff shown, awaiting commit approval. Batches remaining: (2) corrections
interface + ERA5, (3) AEMO skeleton + driver/transfer, (4) legacy manifest
(own commit).

### What was done
- `vwf/harness/` package: `regions.py` (RegionSpec + validating TOML loader),
  `provenance.py` (run_manifest.json, SHA-256 curve-library identity,
  never-abort writer), `skill.py` (MBE/MAE/RMSE/r/EMD/seasonal-cycle RMSE,
  station collapse for pseudo-replicated regions).
- 13 region configs under `configs/regions/` with explicit season months
  (NH for Europe, SH for AU), verified granularity fields, and the AU
  curtailment/timezone caveats as comments.
- 37 tests. Full suite 178 passed; ruff and mypy clean.

### Notable implementation decisions
- Config loader does NOT check registry membership of source/model names —
  configs are declarative; the driver resolves at run time (so `aemo-nem`
  and `entsoe-country` can be named before their adapters exist).
- Train/test year overlap is a hard config error, not a warning.
- Pearson r stays unweighted (documented in skill.py); MBE/MAE/RMSE/EMD are
  capacity-weighted by default.
- Test-suite pins added for the review findings: AU winter=[6,7,8] vs UK
  winter=[12,1,2] (hemisphere pin), UK farm classification, DE postcode
  resolution — a config edit cannot silently un-verify them.
- Avoided `groupby.apply(include_groups=...)` (pandas ≥2.2 only; project
  floor is 2.0) — collapse/climatology are vectorised instead.

### Next checkpoint
Commit approval for Batch 1, then Batch 2.

---

## 2026-07-15 — Batch 1 committed; Batch 2 implemented (checkpoint 5)

**Status:** Batch 1 committed as 05debb1 (identity guard, station_id_regex
verification, seasons-optionality statement folded in). Batch 2 implemented
and green; diff shown, awaiting commit approval.

### Batch 1 additions at commit time
- Identity guard in collapse_pseudo_replicates: refuses to collapse
  station-time groups with divergent observations (a wrong station regex now
  fails loudly instead of silently mis-pooling).
- station_id_regex verified against the data: all 6,618 UK IDs match
  `^(.+)-\d+$`, producing exactly the 360 independent stations; greedy match
  strips only the final `-N`, so `E_ABRTW-1-1` → phase-station `E_ABRTW-1`;
  zero DK/DE IDs match the pattern, and the regex is only consulted when
  pseudo_replicated_rows is set.
- Stated: [seasons] is mandatory in every config even without a "season"
  time slice (transfer matching and RQ1/RQ2 diagnostics need it regardless).

### Batch 2 (implemented, uncommitted)
- CorrectionModel interface + registry (vwf/harness/corrections.py);
  AffineWindCorrection = pure delegation to the legacy fit/apply path,
  pinned by a golden bit-for-bit regression (exact frame equality on the
  synthetic-DK fixture).
- ERA5: longitude normalisation to [-180,180) before slicing (silent
  empty-subset hazard closed and demonstrated in a test); era5_dir param.
- **Deviation from the design's "unchanged files" list, flagged for review:**
  an optional `seasons` parameter (default None = exact legacy behaviour)
  threaded through time_utils.parse_time_slice /
  add_time_resolution_columns, correction.find_offset, and
  wind.correct_wind_speed / simulate_wind. Reason: the legacy apply AND
  offset-fit paths both hardcode NH seasons internally; without this seam a
  SH region cannot reuse the validated code, and duplicating those paths in
  the harness would invite numerical drift — the exact thing the golden test
  exists to prevent. Legacy default pinned by a dedicated test
  (explicit-NH == default, frame-equal).

### What was learned
- 180°E wraps to −180° under the standard normalisation (ERA5 0..359.75
  grids imply the half-open [-180, 180) convention) — my initial test
  expectation was wrong, the arithmetic was right.
- The NH hardcoding was deeper than the Phase 0 audit stated: not just the
  two time_utils sites, but also inside find_offset's slice-month parsing —
  a third silent-inversion path that the seasons seam now closes.

### Next checkpoint
Commit approval for Batch 2, then Batch 3 (AEMO skeleton + driver/transfer).

---

## 2026-07-15 — Batch 2 committed (split); Batch 3 implemented (checkpoint 6)

**Status:** Batch 2 landed as three independently-revertable commits
(d5ad651 seasons seam, 4d70a3c ERA5, b842a73 corrections interface), with
the longitude no-op pin strengthened to object identity. Batch 3
implemented and green (211 tests); diff shown, awaiting commit approval.

### Questions answered at Batch 2 approval
- Country-level fit: delivered in Batch 3 (this one), and it stayed a
  delegation wrapper as predicted — cluster_train_set →
  find_offsets_country_level → format_bc_factors, ~45 lines, no new maths.
- Longitude no-op pin: confirmed (assert_identical) and strengthened
  (`ds is original`).

### Batch 3 contents
- vwf/sources/aemo.py: AEMONemSource skeleton + pure transforms
  (aest_to_utc, scada_to_monthly_cf) implementing the decided conventions:
  UTC monthly bins, capacity in kW vs SCADA in MW, coverage floor (0.9),
  commissioning mask (first fully-post-commissioning month is first valid).
  Real data acquisition remains Phase 2; file-missing errors say so.
- vwf/harness/driver.py + scripts/validate_region.py: train / evaluate /
  transfer. Transfer implements §7 exactly: capacity-weighted collapse per
  slice (refuses unknown clusters), uniform application, season-NAME
  matching via the target's definitions, AU↔Europe pair enforcement.
- Country-level fit in AffineWindCorrection (delegation, above); fourth NH
  site closed (find_offsets_country_level's slice parsing).
- data.py train_set/val_set gained era5_dir/bbox pass-through (additive,
  default None = legacy) so the driver can point regions at their own ERA5.
- Tests: mirrored-hemisphere transfer fixture (normative §8 test 8 — modes
  must differ, name-matching exact, month-matching WORSE than uncorrected);
  weighted-collapse must-distinguish (§8 test 9); AEMO UTC-binning fixture
  proving July-AEST-morning generation lands in June-UTC; driver end-to-end
  train→evaluate on synthetic DK with correction beating uncorrected.

### What was learned / flagged
- The mirrored-hemisphere fixture quantifies the stake: month-matched
  transfer is worse than no correction at all (it corrects the wrong months
  AND leaves the biased ones alone). Worth a sentence in the eventual paper.
- One legacy test (test_sources) stubbed prep_era5 with a fixed-arity
  lambda; updated to **kwargs (test-only change).
- Driver country-level evaluate/transfer targets and entsoe-country
  file-backed loading intentionally raise NotImplementedError until the
  Phase 2 Europe re-runs wire them; turbine-level is fully driveable now.

### Next checkpoint
Commit approval for Batch 3, then Batch 4 (legacy manifest write, own
commit) — which completes the Phase 1 implementation.

---

## 2026-07-15 — PHASE 1 COMPLETE (checkpoint 7, phase boundary)

**Status:** All four batches committed on `multi-region-validation`
(8 commits, 53c4330..dc857a6). Suite: 213 passed (141 inherited + 72 new),
ruff and mypy clean throughout, verified in a clean venv per CI's install
path. Nothing pushed; no real data or curves touched anything committed.

### What the harness can now do
- A region is one TOML config; 13 shipped (DK/DE/UK turbine-level, 9
  ENTSO-E countries, AU-NEM) with explicit season months and verified
  granularity fields.
- `scripts/validate_region.py train|evaluate|transfer` runs turbine-level
  regions end to end: factors, metrics (capacity-weighted MBE/MAE/RMSE, r,
  EMD, seasonal-cycle RMSE, honest n_units), and a provenance manifest in
  every run dir. Legacy PyVWF runs write the same manifest.
- Correction variants sit behind CorrectionModel; the affine baseline is a
  pure delegate pinned bit-for-bit to the legacy pipeline.
- All four NH-season hardcoding sites are closed behind an explicit
  seasons seam (default = legacy); ERA5 longitude convention is normalised
  on load; transfer implements the §7 semantics with pair enforcement.

### Known seams left for Phase 2 (intentional NotImplementedErrors)
- entsoe-country file-backed source loading; country-level driver
  evaluation; country-level transfer targets.
- Real AEMO + ERA5-AU acquisition (user-executed/approved, local only).

### Phase 2 framing agreed at Batch 3/4 approval
Demo-first: this branch's purpose is a demonstrable AU-NEM credibility
artifact run locally against real curves and real DK/DE/UK/ENTSO-E data.
Bounded named deliverables (repo notebook, gridded NetCDF, short write-up);
RQ1–RQ7 subordinate to producing those. Synthetic/CI separation unchanged:
real data and curves never enter committed state or CI.

### Next checkpoint
Phase 2 plan sign-off (deliverables list + data-acquisition approvals).

---

## 2026-07-15 — D1 redefined as regression validation; reference plan (checkpoint 8)

**Status:** D1 scope revised at Phase 1 review: prove the refactored harness
reproduces known-good legacy results on DK (turbine, true coords), DE
(turbine, postcode path), and one ENTSO-E country-level region (joint
L-BFGS-B path), against real curves + real data, locally. Reference plan
below; awaiting sign-off before wiring or running.

### Reference hierarchy (per region)
- **R1 (the pass/fail gate): fresh legacy reference generated from main.**
  A git worktree of main (53c4330) runs the legacy path (PyVWF.train with
  dask_n_workers=0 for determinism, + simulate_cf) in the same local env.
  Diff targets are FACTORS and CORRECTED-CF FRAMES, not metric numbers:
  legacy metrics.py and harness skill.py use different formulas by design,
  so the gate sits upstream of metric choices (both metric sets tabulated
  side by side for the report, informationally).
- **R2 (sanity anchor, not a gate): the saved Feb-2026 outputs** under
  output/runs/{turbine_dk_research, turbine_grid}. Their code vintage
  predates several main-side fixes and their curve-library identity is
  unrecorded (no manifests then — the exact gap provenance now closes), so
  deltas are reported and traced to named commits, never gated numerically.
- **R3 (DK only): Energy-paper numbers** — need the user to point at the
  table/run; loose anchor with explanations.

### Proposed bounded configs
- DK: onshore, train 2015–2019, test 2020, clusters {1, 10} × {fixed,
  season} (overlaps the saved sweep for R2; keeps sequential offset fits
  tractable).
- DE: onshore, train 2015–2018, test 2019, clusters {10} × {fixed}
  (matches the saved turbine_grid config).
- NL (proposed country-level choice): clusters {5} × {fixed}, train
  2015–2021, test 2023, static grid. Rationale: canonical worked example,
  cleanest file set, single zone — and it carries a DOCUMENTED known
  result: the 2023 fleet-composition pathology (R²≈−0.88, mean_sim 0.029
  vs mean_obs 0.143, docs/TURBINE_GRID_EVALUATION_ANALYSIS.md). The
  harness must reproduce the same pathological numbers from the same
  inputs — reproducing known-bad is as falsifiable as known-good.
  (FR is the runner-up if a well-behaved region is preferred.)

### Proposed tolerances (real pass/fail)
- DK/DE factors and corrected-CF frames vs R1: exact; pass at atol 1e-12
  (CSV round-trip guard only). Pure delegation + seeded KMeans +
  deterministic offset fit in one env ⇒ any larger diff is a FAIL to
  investigate, not a tolerance to widen.
- NL factors vs R1: expected exact (same scipy, deterministic L-BFGS-B);
  pass atol 1e-9; 1e-9–1e-6 requires a written explanation before passing;
  >1e-6 fails.
- R2/R3: deltas reported and explained, no numeric gate.

### Mechanics (real curves never touch committed state)
A staging input dir OUTSIDE the repo (real curves as models.csv /
power_curves.csv, links to data dirs); PYVWF_INPUT points there for both
the main-worktree reference runs and the branch harness runs. References
live under gitignored output/. Committed artifact: docs/findings/
d1_regression.md (pass/fail table + explanations) and the regression
scripts — no real-derived outputs.

### D1 implementation to wire (after sign-off)
EntsoeFileSource ("entsoe-country") over the country_level_data file
conventions; country-level driver evaluation (capacity-weighted country
aggregate); corrected-CF saving in driver evaluate; scripts/d1_regression.py
producing the diff table.

### Next checkpoint
D1 plan sign-off (region choice, configs, tolerances, R3 pointer), then
wiring batch (diff before commit), then the reference + harness runs.

---

## 2026-07-15 — D1 EXECUTED: harness reproduces legacy bit-for-bit (checkpoint 9)

**Status:** D1 regression run and PASSED on all four regions. Wiring
implemented and green; diff shown, awaiting commit approval. Findings in
docs/findings/d1_regression.md.

### Win condition MET — zero discrepancies
| Region | Level | Path | atol | worst diff | verdict |
|---|---|---|---|---|---|
| DK | turbine | true coords | 1e-12 | 0.000e+00 | PASS |
| DE | turbine | postcode | 1e-12 | 0.000e+00 | PASS |
| NL | country | joint L-BFGS-B | 1e-9 | 0.000e+00 | PASS |
| FR | country | joint L-BFGS-B | 1e-9 | 0.000e+00 | PASS |

Every factors + cor_cf frame identical to machine zero. The harness is
behaviour-preserving; the "pure delegation" claim is now proven on real data,
not just the synthetic golden unit test.

### Methodology preconditions (both passed before trusting the diff)
- PYVWF_INPUT honoured by BOTH the main-worktree and the branch; both load
  the real 160-model curve library, no silent synthetic fallback.
- main deterministic against itself: DK bit-identical across two runs
  (1e-12); NL country-level (L-BFGS-B) also bit-identical twice. So any
  main-vs-harness delta would be a real refactor effect, not noise.

### What was learned / surprised
- The delegation is EXACT, not merely close — 0.000e+00 everywhere, including
  the country-level L-BFGS-B path and the postcode-geolocation DE path. The
  seasons seam (default None) and ERA5 longitude normalisation (no-op for the
  in-range EU data) are provably invisible on the European configs.
- NL reproduces the DOCUMENTED failure mode
  (docs/TURBINE_GRID_EVALUATION_ANALYSIS.md): uncorrected over-predicts
  (MBE +0.188), correction overshoots to under-predict (MBE -0.136). Same
  pathology; magnitude differs only because of training window (2015-2019 vs
  the doc's 2015-2021). Reproducing known-bad is as falsifiable as known-good.
- DK MAE reduction ~47% at 10/season — same order as the paper's loose ~43%
  anchor (different config, kept loose, not over-fitted).

### D1 wiring implemented (uncommitted, in the shown diff)
- vwf/sources/entsoe_files.py: EntsoeFileSource ("entsoe-country"),
  file-backed country loader per split.
- driver: resolve_source(split); country-level run_evaluate (grid clusters
  reused, capacity-weighted country aggregate metric); corrected-CF frame
  saving for both levels.
- scripts/d1_regression.py: committed frame comparator.
- Tests: entsoe source, country-level evaluate, d1 comparator. Suite green,
  ruff + mypy clean.

### Mechanics note
Real curves staged OUTSIDE the repo (PYVWF_INPUT -> staging dir with
*.real.csv copied to working names, data symlinked); references under
gitignored scratch. Nothing real entered committed state or CI. main
worktree at 53c4330 in scratch (git worktree; remove when D1 closes).

### Next checkpoint
Commit approval for the D1 wiring batch (+ findings doc), then D2 (AU-NEM
ingest) gated on data-acquisition approvals.

---

## 2026-07-15 — D1 committed and CLOSED (checkpoint 10)

**Status:** D1 done. Three commits on the branch (f26e124 wiring, 8646c88
comparator, af99418 findings), sole author. main worktree KEPT in scratch as
the reference env for the whole regression method (remove only when the
branch's real-data work is fully done). D1 win condition met: harness == legacy
bit-for-bit on DK/DE/NL/FR.

### CRITICAL carry-forward into D2 — validation cannot reuse D1's method
D1 validated the PLUMBING: the harness faithfully reproduces the legacy path,
and the new seams (seasons, longitude wrap) are provable no-ops **on EU data**.
It did NOT exercise:
- the SH-season code on the case it exists for (EU is all NH),
- the longitude wrap on the case it exists for (EU lon is in-range),
- any AU/NEM numbers at all.

Australia is where the SH-season and longitude-wrap code first does real work,
and **there is no legacy AU path to bit-diff against**. So D2's confidence
cannot come from a bit-identity gate. D2 must design AU validation deliberately
— candidate strategies to weigh in the D2 plan:
- **Property/invariant tests** on real AU data (CF in [0,1]; corrected error <
  uncorrected on held-out; seasonal cycle peaks in the SH winter months, i.e.
  JJA, not DJF — a direct SH-season correctness check the data itself judges).
- **Synthetic SH ground-truth**: plant a known SH-seasonal bias in a fabricated
  AU-shaped dataset and confirm the harness recovers it (extends the existing
  mirrored-hemisphere unit test to a full pipeline run).
- **Cross-check against an independent AU CF reference** if one exists
  (published NEM capacity factors / Renewables.Ninja AU) — external anchor, not
  self-consistency.
- **Longitude-wrap real exercise**: confirm the AU ERA5 subset (129–154°E)
  slices correctly and that a deliberately 0–360-encoded copy yields identical
  results (the normalisation doing real work, verified).
This is a design task for the D2 plan, not an assumption that D1's method
carries over.

### D2 data-acquisition (BLOCKED on user approval, nothing acquired)
- AEMO: bring exact file list + sizes before any download.
- ERA5-AU subset: user confirming CDS credentials/quota.
- Nothing acquired until explicit approval.

### Next checkpoint
D2 plan for sign-off: (1) AEMO file list + sizes, (2) ERA5-AU subset spec,
(3) the AU validation-design strategy above — presented before any download.

---

## 2026-07-15 — D2 plan presented (checkpoint 11)

**Status:** D2 acquisition spec + AU validation design presented for sign-off.
NOTHING acquired. Awaiting (a) approval of AEMO file list, (b) user CDS
credential/quota confirmation, (c) sign-off on the validation design.

### AEMO acquisition (public NEMWeb, HTTPS confirmed live July 2026)
Base: https://nemweb.com.au/Data_Archive/Wholesale_Electricity/MMSDM/<YYYY>/
MMSDM_<YYYY>_<MM>/MMSDM_Historical_Data_SQLLoader/DATA/ — zipped proprietary
CSV (C/I/D/F row format; the AEMO NEMWeb HTTP endpoint decommission was April
2026 but HTTPS at nemweb.com.au verified working now).
- PUBLIC_DVD_DISPATCH_UNIT_SCADA_<YYYYMM>010000.zip — ~20.5 MB/month, per-DUID
  5-min SCADAVALUE (MW). 72 months (2018-2023) ≈ 1.48 GB zipped.
- PUBLIC_DVD_DUDETAILSUMMARY_<YYYYMM>010000.zip — ~0.17 MB/month, per-DUID
  REGISTEREDCAPACITY (effective-dated → handles capacity changes, which the
  AEMONemSource commissioning/capacity mask already anticipates). ≈ 12 MB.
- Generation Information (xlsx, single file, few MB) + user's public turbine
  compilation → DUID fuel-type=Wind filter and lat/lon. Coordinates come from
  the user's compilation, NOT AEMO (preserves the metadata/curve provenance
  split).
Total ≈ 1.5 GB zipped; processed month-by-month, wind DUIDs only.

### ERA5-AU (CDS, BLOCKED on user credentials/quota)
reanalysis-era5-single-levels, hourly, vars 100m u/v + 10m u/v (roughness),
area N-10/W129/S-44/E154 (au_nem.toml bbox), 2018-2023, 0.25°. Est. ~1-2
GB/year → ~6-12 GB. User confirming CDS quota.

### AU validation design (no legacy reference — D1's bit-diff DOES NOT carry)
Four pillars, data/ground-truth as oracle:
- A. Data-judged invariants on real AU held-out 2023: CF∈[0,1]; corrected
  error < uncorrected; observed AU seasonal cycle peaks in JJA (austral
  winter) and the SH-keyed correction must track it — mislabeled seasons
  would worsen RMSE (the SH code exercised with data as judge).
- B. Synthetic SH ground-truth, pipeline-level: plant a known SH-seasonal
  bias in an AU-shaped dataset, confirm the FULL harness recovers it and NH
  seasons fail (extends the mirrored-hemisphere unit test to a real run).
- C. Longitude-wrap real exercise: real AU ERA5 (129-154°E) slices correctly
  AND a deliberately 0-360-re-encoded copy yields identical corrected CF —
  a bit-identity check constructable for the wrap code WITHOUT a legacy path.
- D. External anchor (optional/stretch): cross-check AU CF vs an independent
  published reference (Renewables.Ninja AU / OpenNEM) — direction+magnitude,
  loose, not a gate; the only pillar needing extra data.
Pillars A-C need only the AEMO+ERA5-AU above; D is optional.

### Next checkpoint
D2 sign-off (AEMO list, CDS confirmation, validation design), THEN acquire.

---

## 2026-07-16 — D2 sign-off recorded; pillar B built and PASSING (checkpoint 12)

**Status:** Four-pillar design approved (A/B/C in critical path, D deferred).
Pillar B implemented as the pre-flight and passing. AEMO window trimmed to
2020–2023. NOTHING acquired; CDS-first sequencing confirmed (user checks CDS
quota before any AEMO download).

### Decisions recorded (D2 sign-off)
- Pillar B FIRST as pre-flight (synthetic, clean pass/fail) — done, below.
- Pillar A's JJA seasonal-cycle check = HARD GATE when real data arrives.
- Pillar D deferred to optional stretch. Renewables.Ninja-AU is VWF-derived,
  so PyVWF-vs-RN-AU is partly CIRCULAR — it tests method fidelity, not
  correctness vs reality. If D happens, OpenNEM (actual metered generation)
  is the independent anchor. RN-AU is never to be treated as an oracle.
- AEMO window: 2020–2022 train / 2023 test (36 months). Config + source
  default updated accordingly.
- Sequencing: CDS credentials/quota confirmed by user FIRST, then AEMO SCADA.

### Pillar B — built, PASSING (tests/test_pillar_b_sh_pipeline.py, 3/3, ~13 s)
Synthetic AU on disk (AEST 5-min SCADA + biased ERA5), full real stack:
AEMONemSource → harness run_train/run_evaluate, SH config, planted 30% JJA
over-blow. Asserts: (1) the downward correction lands under the "winter"
LABEL = JJA; (2) held-out corrected < uncorrected; (3) must-distinguish —
SH-trained factors applied through the NH map are WORSE THAN NO CORRECTION
(rmse_nh > rmse_unc > 2×rmse_sh), the exact bug mode the seasons seam
eliminated, now demonstrated on real fitted factors end-to-end.

### Insight worth keeping (recorded in the test docstring)
Within-region training with meteorological seasons is PARTITION-INVARIANT:
NH and SH mappings group the same four month-sets; only names differ. The
fit maths cannot be hemisphere-wrong. The genuine SH risks are (a) factor
LABELS (interpretation + transfer) and (b) fit-time/apply-time label
consistency. Pillar B pins exactly those two; this also sharpens what
pillar A must judge on real data (the JJA gate).

### NEMWeb verification (user precondition for the 36-month download) — PASS
- Recent month resolves: MMSDM_2026_06 listed, updated 14 Jul 2026 — HTTPS
  archive alive and current after the Apr 2026 HTTP decommission.
- Oldest window edge resolves: MMSDM_2020_01 SCADA = 14.6 MB (fleet was
  smaller). Revised window estimate ~600–750 MB zipped for 36 months.

### Environment note
The previous session scratchpad was purged: venv, main worktree, staging
dir, and the scratch D1 runner scripts were lost (committed D1 artifacts
all safe; references regenerable by design — that was the method's point).
Rebuilt in the current scratchpad: worktree re-added at 53c4330 (stale
registration pruned), staging dir with real curves re-staged, venv
reinstalled. OPEN QUESTION for user: commit the two small D1 runner scripts
(run_legacy.py / run_harness.py) under scripts/ so D1 re-runs survive
scratch purges, or accept regenerating them from the findings doc?

### Next checkpoint
User CDS confirmation → acquisition (AEMO after CDS clears), then pillars
C and A on real data.

---

## 2026-07-16 — CDS confirmed; acquisition scripts ready for user review (checkpoint 13)

**Status:** Pillar B + window trim committed (e94febe); D1 runners committed
under scripts/ (120f5aa) closing the scratch-purge gap. ERA5-AU request
script and AEMO fetch script WRITTEN and shown, NOT committed, NOT run —
user executes both downloads. Nothing acquired.

### CORRECTION (flagged to user): 48 months, not 36
2020–2022 train + 2023 test = 4 years = 48 monthly archives, not the 36 I
wrote at checkpoint 12 (and the user echoed). Revised AEMO estimate at
14.6–22 MB/month: **~0.9 GB zipped** (previously stated 600–750 MB for
"36"). ERA5-AU likewise 48 monthly CDS requests.

### Decisions recorded
- CDS confirmed user-side; skip the tiny test request. Known caveat: a
  pre-migration token may throw an auth error on the first request → user
  regenerates token; not a blocker (hint baked into the script's error path).
- User executes BOTH downloads (CDS submission through their credentials is
  theirs, like every push/merge). Sequence: ERA5 submitted first (slow
  queue), AEMO pulled in parallel after.
- Partition-invariance framing adopted: pillar A's JJA gate is THE critical
  check, not one of four — to stay front and centre at pillar A.

### Artifacts awaiting user review (uncommitted)
- scripts/fetch_era5_au.py — reanalysis-era5-single-levels, hourly, 100m+10m
  u/v, area N-10/W129/S-44/E154, 0.25°, month-by-month (48 requests; a
  monolithic multi-year request would exceed the per-request cap),
  netcdf/unarchived, resumable (.part + skip-existing), --dry-run verified
  (48 planned, correct paths input/era5/AU/era5_au_YYYY_MM.nc — the layout
  au_nem.toml expects; roughness derives from 10m/100m shear at load, no
  preprocessing step).
- scripts/fetch_aemo_au.sh — 48× DISPATCH_UNIT_SCADA + 48× DUDETAILSUMMARY
  from nemweb.com.au MMSDM archives into $PYVWF_INPUT/aemo_raw/, curl -C -
  resumable; Generation Information workbook fetched manually (quarterly
  URL). bash -n clean.

### Next checkpoint
User reviews + runs both scripts; meanwhile (no data needed) build the
SCADA→AEMONemSource-layout processing step (aemo_raw → au_nem_md.csv +
au_nem_scada.csv) against synthetic fixtures, so ingest is ready the moment
data lands. Then pillar C, then pillar A (hard gate).

---

## 2026-07-16 — AU ingest processing built; fleet join smoked on real workbooks (checkpoint 14)

**Status:** Downloads running user-side (AEMO 43/48 at last check; ERA5
queueing, 2/48 landed). Processing batch built, all-green, diff shown,
awaiting commit approval. Generation Information workbook received and
copied to input/aemo_raw/.

### Design decisions in the batch
- **Partial/finalise split** of scada_to_monthly_cf (behaviour-preserving,
  existing tests untouched and green): AEMO archives are cut on MARKET-time
  month boundaries and straddle UTC months (first 10 AEST hours of each
  market month belong to the previous UTC month), so archives reduce to
  per-(DUID, UTC month) energy partials that are SUMMED across archives
  before finalisation. Chunk-boundary test proves the straddle is real on
  the fixture (must-distinguish, not vacuous).
- **Fast path**: AEMONemSource prefers au_nem_scada_monthly_partials.csv
  (~5k rows) over raw SCADA (~45M rows/1.7 GB for 48 months). No-drift
  guarantee: finalisation (capacity, coverage floor, commissioning mask)
  still runs through the same audited finalise_monthly_cf at load time;
  test pins fast == slow frame-identically AND that corrupting the partials
  changes the answer (the fast path is really being read).
- vwf/datasets/aemo_au.py: MMS C/I/D/F parser (validated against a real
  downloaded archive), Gen-Info wind-fleet extraction (capacity SUMS across
  unit-group rows — Boco Rock 9x1.60+58x1.70; "In Service/In Commissioning"
  status values; NEM-region filter), GWPT capacity-weighted phase centroids,
  conservative name join with manual alias overrides (no fuzzy matching: a
  wrong merge is unrecoverable, a miss is an alias entry).
- cdsapi added to the data extra.

### Real-workbook join smoke (metadata only, output in scratch)
64/105 DUIDs matched exactly; 41 unmatched (systematic naming divergences:
"Bango 973 Wind Farm" vs GWPT "Bango wind farm", trailing "- VIC" tags,
stage numbers); 10 matches capacity-suspicious (some will be staged
capacity, some may be wrong joins). ~2/3 of the fleet flows through with
zero manual work; the tail needs an alias file + user review of the report.
NOTE: Bango 973/999 are two DUIDs under one GWPT project — many-to-one
joins need explicit handling in the alias review, not a looser normaliser.

### Known limitations recorded in code + report (decisions pending)
- Hub height: uniform default (100 m), height_source="default-uniform" —
  GWPT has no heights. Vintage-aware assignment = named follow-up (RQ7).
- Power-curve model: uniform default, user picks a real-library key for
  real runs. Per-farm models unavailable without turbine-model data.
- Static nameplate capacity: staged commissioning inside the window (e.g.
  Coopers Gap) biases early months; DUDETAILSUMMARY capacity histories are
  the named follow-up BEFORE pillar A gates.

### Next checkpoint
Commit approval for the processing batch (+ the two fetch scripts +
pyproject); alias-file draft for the 41 unmatched DUIDs; full processing
run once the last SCADA archives land.

---

## 2026-07-16 — Curve/alias/commissioning evidence pack (checkpoint 15)

**Status:** Processing batch committed (ca0beba, 50c3de8, 0d069fe). AEMO
downloads complete (48/48 SCADA + DUDETAILSUMMARY); ERA5 landing. Alias
evidence pack written for per-line user approval; curve-matching gap
quantified; commissioning-gate correction made. Awaiting user rulings.

### Decisions recorded (from batch review)
- Uniform-curve default REJECTED for AU: it is a different method than the
  D1-validated add_models path (manufacturer → nearest power density) and
  would make the correction absorb curve-mismatch error, muddying pillar A.
  Real matching where data exists; per-farm default fallback only.
- Alias work reprioritised as coverage-raising (headline-fleet fraction),
  per-line user approval, suspicious-10 reviewed before the 41 misses.
- Many-to-one ruling: GI capacity + GWPT coordinates only.
- DUDETAILSUMMARY commissioning/capacity handling escalated to a GATE
  before pillar A (staged commissioning could inject a fake seasonal
  signal into the exact JJA cycle pillar A judges).

### Field coverage (verified against both workbooks)
Per matched farm: capacity 105/105 (GI, authoritative); per-turbine rated
capacity 127/127 unit-group rows (GI Unit Capacity × Unit Count);
coordinates 64/105 matched (GWPT); start year (GWPT). NOT present in
either source: rotor diameter 0, turbine model/manufacturer 0 (GI
"Technology Detail" is just "Onshore"; GWPT has no turbine-spec columns),
hub height 0. FCUD only 6/105 → commissioning dates are almost entirely
the Jan-1-of-start-year fallback; 28/64 matched farms commissioned INSIDE
the window — staged commissioning is half the fleet, not a corner case.

### CORRECTION: DUDETAILSUMMARY has NO capacity column
Checkpoint 11 claimed REGISTEREDCAPACITY lives in DUDETAILSUMMARY — wrong.
It lives in the sibling DUDETAIL table (PUBLIC_DVD_DUDETAIL_*.zip, ~48×0.1
MB, not yet downloaded). fetch_aemo_au.sh amended (third file type,
resumable re-run picks up only the new files). The capacity-history gate
builds on DUDETAIL once fetched.

### Fleet join evidence (input/turbine_level_data/AU_NEM/alias_review.md)
- Suspicious 10 → group sums resolve 9 as exact stage-shares (Dundonnell
  336.0/336, Hornsdale 316.8/316, ...). ONE real wrong match caught:
  MUWAWF2 was inheriting Murra Warra 1's coordinates; GWPT has a separate
  Murra Warra 2 project 3 km away. The capacity check did exactly its job.
- 41 misses drafted: 17 project-level aliases with near-exact capacity
  sums; 11 phase-level targets (Lal Lal Elaine/Yendon, Portland's capes
  incl. Yambuk/Codrington, Hallett stages with distinct coords, Woolnorth
  spanning two GWPT projects, Moorabool North+South centroid); 5 from the
  Below-Threshold sheet; 1 unresolved (Yawong, 7.2 MW).
- If approved: 97/105 DUIDs (~92%), from 61%. Requires a phase-aware alias
  extension (aliases target project|phase or BT rows; one-to-many =
  capacity-weighted centroid) — implementation after per-line approval.

### Curve-matching proposal (pending approval)
Close the model gap via a public-source pass: GWPT carries a Wiki URL per
project (GEM wiki pages typically state turbine make/model, e.g. "Vestas
V112-3.3"); Wikipedia's AU wind-farm lists as cross-check. Produce
au_turbine_models.csv (farm, manufacturer, model, rotor diameter from the
designation, count, provenance URL) for user review; diameter + GI
per-turbine capacity → the SAME add_models manufacturer/p_density path as
DK/DE/UK. Farms the pass cannot resolve get a per-farm fallback curve
marked model_source="default-fallback" — never a blanket uniform curve.

### Next checkpoint
User rules on: alias lines (esp. B10/B11 Hallett capacity conflicts, D
Yawong), curve-pass approval, DUDETAIL re-fetch. Then: phase-aware alias
implementation + capacity mask (gate) + curve table → pillar C → pillar A.

---

## 2026-07-16 — Aliases live (104/105); sensitivity verdict; observed cycles flip the story (checkpoint 16)

**Status:** Rulings executed. Fetch script amended+committed (704e278,
user re-runs for DUDETAIL). Phase-aware aliases implemented and committed
(337a637); full processing run done: 104/105 DUIDs (14.26 GW), monthly
partials from all 48 archives (3,510 DUID-months, 2m10s). Curve
sensitivity answered. Observed seasonal cycles computed from real SCADA.

### B10/B11 — INCLUDED on coordinates (user condition met)
Wikipedia independently confirms Hallett 4 = North Brown Hill (132 MW —
GI's 132.3 ✓, GWPT's 53 is wrong) and Hallett 5 = The Bluff (52.5 exact).
Geometry: phase 4 is 1.9 km north of Brown Hill; phase 5 in the Porcupine
Range 8 km from Hallett town. Distinct, right region. Capacity
discrepancy NOTED not resolved (GWPT capacity unused). Bonus: Wikipedia
carries turbine models (45× Suzlon S88 2.1 MW at Brown Hill) — and GEM
wiki 403s scripted fetches, so the curve pass will lean on Wikipedia.

### Curve sensitivity (5 farms × 5 curves, ERA5-AU 2020, sim-side)
- Among REAL curves: normalized cycles near-identical (r ≥ 0.9989, JJA
  share spread ≤ 0.02, peak month never moves). VERDICT: curve choice is
  a LEVEL effect → the model pass is defensibility/level accuracy, not
  pillar-A pass/fail. Effort calibrated accordingly.
- The SYNTHETIC curve is an outlier: amplitude inflated +0.15–0.46,
  r down to 0.974 — the uniform-synthetic rejection was load-bearing.

### Aliases implemented (337a637)
resolve_duid_aliases: DUID-keyed (site names not unique), targets =
project (any status; alias IS the approval) | project|phase | BT:row;
multi-target → capacity-weighted centroid; typos fail loudly. Semantics
fix found in testing: aliases must OVERRIDE name matches — MUWAWF2 had
name-matched the WRONG project and the resolver initially skipped it;
now re-sited to Murra Warra 2 (verified in output). Coverage 61%→99%.

### Observed seasonal cycles (real SCADA, pre-2020 farms only, n=58)
Fleet is predominantly WINTER-peaking: NSW peak Aug (JJA 1.09), SA peak
Jun (1.11), VIC peak Jul (1.14), QLD JJA 1.21 (n=1), TAS flat (n=2);
37/48 farms peak Jun–Sep, 10 peak DJF. BUT the sim-side preview (ERA5
through curves) had VIC/SA/TAS below-average JJA with Sep peaks — i.e.
**ERA5's seasonal cycle disagrees with the observed one in the southern
NEM**. That gap is exactly a seasonal reanalysis bias — the thing the
seasonal correction exists to fix and what pillar A measures. Caveats:
sim preview = 1 year/5 farms/nearest-cell; obs = no capacity-history mask
yet (pre-2020 subset mitigates).

### Pillar A gate, re-specified (proposal)
Not "cycle peaks in JJA" (hardcoded expectation; wrong for TAS and for
DJF-peaking farms). Instead, data-as-judge: (1) corrected seasonal cycle
must track the OBSERVED cycle better than uncorrected (per region,
capacity-weighted RMSE of normalized cycles); (2) the improvement must
hold specifically in the winter-labelled months; (3) pillar B's label pin
already guarantees "winter"=JJA in the factors. Hard gate = (1).

### Next checkpoint
User: run amended fetch (DUDETAIL), rule on pillar-A re-spec, calibrate
curve-pass effort given the sensitivity verdict. Me: curve pass at agreed
effort → locked CSV review → mask numbers → pillar C → pillar A.

---

## 2026-07-16 — Mask committed; ERA5 integrity 48/48; PILLAR C PASS (checkpoint 17)

**Status:** Registered-capacity mask committed (c16ee23): 42/3,455
farm-months (1.2%) masked (28 pre-registration, 13 below-final-build, 1
mid-month change). Overlap check: ZERO SCADA-bearing DUIDs lack DUDETAIL
histories (the "~20" was a counting artifact) — no un-maskable ramp-up
residual exists. ERA5-AU complete and integrity-checked. Pillar C run and
PASSED. Curve pass in flight (two research agents); pillar A HELD for
curve-CSV sign-off per ruling.

### ERA5-AU integrity pre-flight — 48/48 PASS
Every monthly file: opens, carries u100/v100/u10/v10, exact hourly
timestep count for its month (no CDS truncation), covers the au_nem bbox,
physical values (grid-mean 100m wind 5.62–8.53 m/s across months, >99.9%
finite, p99.9 < 60 m/s).

### PILLAR C — PASS, with an honest re-scope
Discovery: the planned 0–360 bit-identity check is PROVABLY VACUOUS for
Australia — AU longitudes (129–154°E, positive, <180°) are numerically
identical in both conventions; the wrap cannot corrupt AU data even in
principle. The transform that CAN corrupt an AU slice is LATITUDE
ORDERING (_slice_bbox has a descending-lat branch; CDS routes differ).
Pillar C as run:
- C1 normalisation no-op on real data: PASS (object identity).
- C2 slice correctness: grid 137×101 covers bbox; lon ascending; all 104
  farms STRICTLY inside the sliced grid (interpolation, never
  extrapolation): PASS.
- C3 0–360 re-encoded copy (labelled invariance proof, vacuous by
  construction): bit-identical, max|Δ|=0.0 on grid and on wind
  interpolated to all 104 farms.
- C4 lat-flipped copy (the NON-vacuous equivalence): bit-identical,
  max|Δ|=0.0 on grid and all 104 farm wind speeds. PASS.
Verdict: pillar A's AU inputs are trustworthy. (Scope: equivalence copies
over the full 2023 year; C1/C2 over the full period.)

### Curve pass, first half returned (top-20 manufacturer confirmations)
16/17 top-capacity farms MANUFACTURER-CONFIRMED from OEM/developer
domains (Nordex N163/5.X at MacIntyre, Goldwind GW140/GW155, GE 3.6-137 +
3.8-130 at Coopers Gap, Vestas V162 EnVentus at Rye Park/Golden Plains,
Senvion 3.7M144 at Murra Warra 1, GE Cypress 5.5-158 at MW2/Goyder...).
Port Augusta REP is the one secondary-source row (developer confirms 50 ×
4.2 MW but not the OEM; the matching Vestas PR is anonymised). Two
database-error catches by going to OEM sources: Collector is V117-4.2
(not V150), Ryan Corner V136-4.2 (not V150). Wikipedia fleet sweep still
running; au_turbine_models.csv assembles when it lands, then LOCKS for
user review.

### Next checkpoint
Assemble + lock curve CSV → user review → add_models wiring (per-turbine
capacity × OEM rotor diameter → p_density match into the real library) →
pillar A run with the approved data-as-judge gate.

---

## 2026-07-16 — Curve CSV assembled and LOCKED for review (checkpoint 18)

**Status:** Both research agents returned; configs/au_turbine_models.csv
assembled (uncommitted until user signs). Pillar A remains HELD.

### Tier summary (the numbers the write-up rests on)
| tier | farms | MW | % of fleet capacity |
|---|---|---|---|
| manufacturer-confirmed (incl. 1 propagated) | 20 | 6,293 | 44.1% |
| secondary-source (51 full + 15 manufacturer-only) | 66 | 6,757 | 47.4% |
| default-fallback | 18 | 1,210 | 8.5% |

### p_density cross-check (the Collector/Ryan-Corner error detector)
71/104 farms have computable p_density (GI per-turbine MW × source rotor):
ALL within the physical band — 191–538 W/m², median 304. One flag, caught
and resolved: WAMBOWF1's GI "Unit Capacity" is the whole farm (registered
as one 252 MW unit group, the only such farm); unit_mw_eff derived as
capacity/OEM-count (42 × 6.0), documented in the flags column.

### Review-worthy discrepancies (in the note column)
- GULLRWF2: GI says 31 × 3.57 MW; Wikipedia says Gullen Range is
  17 × GW82-1.5 + 56 × GW100-2.5 — GI's unit data looks wrong for this
  farm; mixed fleet, no single p_density.
- CROOKWF2: GI 28 × 3.43 vs Wikipedia Crookwell 2 = 46 units/92 MW —
  conflicting; kept FALLBACK rather than forced.
- PORTWF/YAMBUKWF: Vestas-vs-Senvion source conflict (farm infobox vs
  state list) — kept FALLBACK rather than guessing.
- Upstream catches (already in the CSV): Collector V117-4.2 not V150;
  Ryan Corner V136-4.2 not V150; Port Augusta REP stays secondary
  (Vestas PR is anonymised — not to be promoted without a real OEM
  source, per ruling).

### Largest fallbacks (stated limitation if unresolved)
Berrybank 2 DUIDs (290 MW), Mortlake South (157), Portland complex
(182 across PORTWF/YAMBUKWF), Flyers Creek (145), Hawkesdale (97),
Crookwell 2+3 (154). Fallback = 8.5% of capacity.

### Next checkpoint
User signs the CSV → commit → add_models wiring (manufacturer +
p_density into the real library; manufacturer-only rows match within
manufacturer by nearest unit capacity; fallback rows get per-farm
default marked model_source) → PILLAR A.

---

## 2026-07-16 — AU CSV signed+committed; open-library coverage assessed (checkpoint 19)

**Status:** au_turbine_models.csv approved and committed (f19e5fd). User
sourced a redistributable open curve library (NatLabRockies/turbine-models
BSD-3, VWF-Gaussian-smoothed, schema-matched; 69 real machines + 7
composites). Coverage assessed; recommendation below. PILLAR A HELD.

### Open-library AU coverage (capacity-weighted, fleet 14,260 MW)
| metric | REAL library | OPEN library |
|---|---|---|
| manufacturer match | 91.5% capacity (86 farms) | 52.1% (48 farms) — but see caveat |
| p_density within ±10% | 76.3% | 55.7% |
| p_density within ±20% | 77.9% | 71.3% |
| p_density within ±30% | 78.7% | 75.6% |
| (p_density computable) | 81.2% | 81.2% |

**Caveat that changes the matching strategy:** the open library's only
Vestas machines are V27/V29/V47/V82 (225 kW–1.65 MW) — its "manufacturer
matches" for V112–V162 farms are vintage mismatches (e.g. Dundonnell
V150-4.2, p_density ~237, would match V82 at 312 within-manufacturer,
while the library's Market-Average-2.4MW composite sits at 227).
Manufacturer-first matching is actively harmful on the open library; the
right strategy there is p_density-class matching into its 20 MW-scale
onshore references/composites (150–350 W/m²), with the IEC Class 1/2/3
normalized composites as the generic fallback for the ~19% of capacity
without a farm p_density.

### Recommendation (pending user ruling)
- REAL library stays PRIMARY: method-consistent manufacturer+p_density
  add_models matching at 91.5% capacity coverage.
- OPEN library is VIABLE as the parallel open stack via p_density-class
  matching (71% capacity within ±20%): run pillar A on BOTH. Given the
  level-not-shape sensitivity verdict, the seasonal gate should be robust
  across libraries — if both pass, the demo is validated on real curves
  AND reproducible on a fully-open stack. Present real as primary, open
  as the no-proprietary-dependency reproduction, with the class-proxy
  caveat stated.

### Follow-up flagged, NOT actioned (post-AU, per ruling)
The open library is a candidate replacement for the SYNTHETIC placeholder
curves in the JOSS package (real physics, redistributable, schema-matched)
— would upgrade the public repo from "runs on invented curves" to "runs on
real open curves". Parked until AU completes.

### Next checkpoint
User rules on the dual-library pillar A → add_models wiring (real primary;
open via p_density-class) → PILLAR A.

---

## 2026-07-16 — PILLAR A: GATE PASS ON BOTH LIBRARIES, no divergence (checkpoint 20)

**Status:** Dual-library pillar A run per rulings (real first as the gate,
open second as robustness). Infrastructure committed (fde882c): curve
assignment (real via add_models; open via p_density-class), daily ERA5-AU
pre-combine (~8-16 GB hourly → 29 MB/yr, mean-of-speed order preserved,
prep_era5 additive tweak pinned by must-distinguish test).

### The gate (capacity-weighted normalized-cycle RMSE vs observed, fleet,
76 farms with full 2023 obs; corrections trained 2020-2022, SH seasons,
5 clusters)
| library | uncorrected | corrected-season | verdict |
|---|---|---|---|
| REAL (headline) | 0.0724 | 0.0645 (−10.9%) | **PASS** |
| OPEN (robustness rider) | 0.0672 | 0.0614 (−8.7%) | **PASS** |
JJA |error|: real 0.0506→0.0443 (improved); open 0.0401→0.0400 (neutral).
NO DIVERGENCE — same verdict, same direction, comparable magnitude, and
the same per-region pattern; hard-stop not triggered.

### Per-region honesty (both libraries agree on the pattern)
- SA1 (24 farms): the big winner — 0.1107→0.0972 (real), 0.1059→0.0942
  (open). SA drives the fleet pass; consistent with checkpoint 16's
  finding that ERA5's seasonal bias is largest in the southern NEM.
- NSW1 (17): slightly WORSE cycle tracking (0.0886→0.0927 real).
- VIC1 (28): slightly worse (0.0669→0.0704 real).
- QLD1 (3 farms): worse (0.114→0.125) — small n.
- TAS1 (4 farms): improved on real, worsened on open — the one
  library-divergent region; n=4 and the flattest observed cycle, treated
  as small-n noise, stated not hidden. Gate-level divergence did NOT occur.
Write-up must say: fleet-level pass driven primarily by SA; NSW/VIC
marginally negative on this metric; QLD/TAS too small to weigh.

### Fitted seasonal factors (real, per cluster) — physically coherent
Two clusters scaled DOWN (over-blown ERA5 regions: scalars 0.78-0.90,
strongest reduction in winter), three scaled up; seasonal variation
present in all (e.g. cluster 1 winter 0.849 vs summer 0.985).

### Standing caveats attached to any headline
- Curtailment confound (SCADA = curtailed output): stated; a
  hemisphere-wide systematic seasonal mismatch is not a curtailment
  signature, but the residual contribution is unquantified.
- Open-stack results = p_density-class proxy matching, NOT per-OEM curves
  (caveat travels with every open result).
- 2023 = one held-out year; 76/104 farms had full-year obs.

### Next checkpoint
Findings write-up (docs/findings/pillar_a_au.md) + the remaining demo
deliverables (D3 notebook, D4 NetCDF, D5 transfer + synthesis).

---

## 2026-07-16 — Framing signed; exclusion check run; findings doc written (checkpoint 21)

**Status:** Pillar A framing signed with amendments (SA-led headline;
selectivity foregrounded as evidence; over-amplification language;
fixed-vs-season JJA sentence stated before readers find it; curtailment
paragraph reordered to lead with the tracking claim and end on what the
result IS). docs/findings/pillar_a_au.md written to the signed framing.

### Cluster-3 exclusion check (the cheap overfitting close-out)
Excluding the three far-north sites (lat > −20°: Mount Emerald, Windy Hill,
Kaban; the n=1 cluster was Mount Emerald):
- Gate STRENGTHENS: 0.0780 → 0.0637 (−18.3%), 74 farms.
- SA preserved and slightly better: 0.1107 → 0.0930.
- NSW FLIPS to mild improvement (0.0886 → 0.0847) — the outlier had been
  distorting the NSW clustering too.
- VIC remains mildly negative (0.0669 → 0.0729).
Doc line earned with evidence: "QLD degradation traces to an n=1 cluster;
excluding it the picture is unchanged where it matters."

### Raw-table readings folded into the doc (user's three)
1. ERA5 OVER-AMPLIFIES SA's real cycle (shape right, amplitude exaggerated
   both directions; June 1.617 vs 1.394 obs, Jan 0.929 vs 1.090 obs).
2. Real-library fixed beats season on JJA specifically (0.0403 vs 0.0443)
   while season wins the full cycle — SA's gain is mostly non-winter months.
3. n=1 cluster named AND checked (above).

### Next checkpoint
D3 (reproducible notebook), D4 (gridded NetCDF), D5 (AU↔EU transfer +
synthesis doc).

---

## 2026-07-16 — D3/D4 committed; D5 transfers run; NEW native-skill finding (checkpoint 22)

**Status:** D3 committed (8da53a6: notebook + 156 KB attributed derived
data; pre-commit checks passed — zero baked outputs, no local paths,
stranger-proof cell 1, aggregation irreversibility confirmed). D4 committed
(0106889: exporter regenerated from noc3 factors per ruling — max corrected
wind 57.7→36.0 m/s; EXTRAPOLATION_CAVEAT + FACTORS_PROVENANCE attributes).
D5 transfer runs complete; synthesis doc HELD for framing ruling.

### Transfer results (real curves; §7 semantics: capacity-weighted
collapse, uniform application, season-NAME matching)
| run | RMSE | MBE | vs uncorrected |
|---|---|---|---|
| DK uncorrected | 0.160 | +0.121 | — |
| DK native corrected (10, season) | 0.099 | +0.041 | −38% RMSE |
| DK ← AU transfer (season) | 0.166 | +0.129 | +4% (no benefit) |
| AU uncorrected | 0.104 | −0.024 | — |
| AU native corrected (5, season) | 0.120 | −0.041 | **+16% (worse)** |
| AU ← DK transfer (season) | 0.137 | −0.093 | +32% (harmful) |
- AU→DK: the capacity-weighted collapse of AU's OPPOSING clusters lands
  near identity (scalars 0.98–1.10) — cancellation; transfer applies a
  weak wrong-direction nudge. Graceful, useless.
- DK→AU: collapsed DK factors (scalars 0.72–0.84, DK's pull-down) push
  AU's already-slightly-negative bias further negative. Harmful, bounded,
  sign-consistent — degradation is graceful, not catastrophic.

### NEW FINDING (needs user framing before synthesis): AU native
correction improves SHAPE but worsens absolute farm-level skill
The pillar A gate (normalized-cycle RMSE, fleet) passes — but the standard
skill table shows AU native corrected RMSE 0.120 vs uncorrected 0.104
(MBE −0.024→−0.041). ERA5 is nearly LEVEL-unbiased in AU (MBE −0.024 vs
DK's +0.121), so per-cluster level adjustments add farm-level noise while
compressing the fleet seasonal cycle. In DK (large level bias) the same
correction wins everything. Generalisation result: the affine correction's
value tracks the bias structure — big level bias → big win (DK); small
level bias + seasonal-shape error (AU) → shape improves, absolute skill
does not. pillar_a_au.md currently reports the gate only; an expert would
ask about absolute skill — likely warrants an amendment. USER TO RULE.

### Next checkpoint
User framing ruling on the native-skill finding + transfer table →
findings-doc amendment (if ruled) → D5 synthesis doc → D2 close-out.

---

## 2026-07-16 — BRANCH COMPLETE: amendment + synthesis committed (checkpoint 23, final)

**Status:** pillar_a_au.md amended per the signed steers (diagnosis-led
headline — "corrected" nowhere implies "improved"; absolute-skill table IN
FULL beside the pre-specified gate, with the pre-specification statement
and the bias-structure/DK-contrast explanation; practical-guidance close:
corrected for seasonal profiles, uncorrected for absolute farm-level CF in
this region). d2_synthesis.md written, organised around the central
result: correction value tracks the STRUCTURE of the reanalysis bias;
transfer is the same fact from the other direction (gracefully useless /
bounded-harmful); one sentence of future work (shape-only variant, named
not evaluated); checkpoint-8 open-questions triage done once, in writing.

### The branch's honest ledger
- Europe re-validated bit-for-bit against the legacy method (D1, four
  regions, 0.000e+00, preconditions first).
- Australia validated with a sharper finding than planned: ERA5's NEM bias
  is near-zero on level and over-amplified in SA's seasonal shape; the
  correction compresses the shape (pre-specified gate PASS, both curve
  libraries) and cannot improve the level (absolute RMSE +16%, reported
  in full) — the method behaving correctly given the diagnosed structure.
- Transfer answered negatively-but-gracefully: corrections are regional
  bias fingerprints, not portable parameters.
- Open questions triaged once at the boundary, as mandated at checkpoint 8.
- Real curves and raw market/reanalysis data never entered committed
  state or CI. All commits sole-authored by the user. Nothing pushed —
  remote actions remain the user's.

### Named next task (SEPARATE scope, user to bring):
JOSS open-curve upgrade on main — replace the synthetic placeholders with
the open library; seasons seam + longitude normalisation as correctness
candidates to port — before submission. Not a continuation of this branch.
