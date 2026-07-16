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
