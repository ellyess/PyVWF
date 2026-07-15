# D1 — Regression validation of the refactored correction path

**Question.** The harness refactored the correction path (the `CorrectionModel`
delegate, the `seasons` seam, ERA5 longitude normalisation, file-backed
country loading). Does it still reproduce known-good results on regions where
the answer is already known, run against the real curve library and real data?

**Verdict: PASS.** On all four regions the harness reproduces the legacy
`main` pipeline **bit-for-bit** — correction factors and corrected capacity-factor
frames identical to machine precision (max abs diff `0.000e+00`). There were
no discrepancies to explain.

## Method

Reference-generated-fresh-from-main, diffed at frame level. Metric numbers are
NOT the gate: legacy `vwf.metrics` and harness `vwf.harness.skill` compute
skill with different (intentional) formulas, so the gate sits upstream on the
`factors_*.csv` and `cor_cf_*.csv` frames. If those match, the refactor
preserved the method.

- **Reference (R1, the gate):** a git worktree of `main` at `53c4330` runs the
  legacy `PyVWF.train(dask_n_workers=0)` + `simulate_cf` path.
- **Harness:** the branch runs the same config through
  `vwf.harness.driver` (`run_train` + `run_evaluate`).
- Both are driven by `PYVWF_INPUT` pointed at a staging directory holding the
  **real** curve library (`power_curves.real.csv` / `models.real.csv` copied to
  the working names) and symlinks to the real data. Real curves and data never
  enter committed state or CI.
- Diff: `scripts/d1_regression.py`, cell by cell over numeric columns.

### Methodology preconditions (both PASS, checked before trusting any diff)

1. **`PYVWF_INPUT` honoured by both trees.** Confirmed that both the
   main-worktree and the branch resolve `INPUT_ROOT`/`ERA5_DATA` to the staging
   dir and load the real 160-model curve library, with no silent fallback to
   the bundled synthetic curves. (A silent fallback on one side would compare
   real-vs-synthetic and fail everything for the wrong reason.)
2. **`main` is deterministic against itself.** `main` run twice on the DK
   config produced bit-identical output (`0.000e+00` at atol `1e-12`). The
   country-level path (joint L-BFGS-B offsets) was likewise checked twice on NL
   and is deterministic. So any main-vs-harness difference would be a real
   refactor effect, not run noise.

## Results

| Region | Level | Path exercised | Config | atol | Worst diff | Verdict |
|---|---|---|---|---|---|---|
| DK | turbine | true coords, per-turbine | onshore, 2015–2019→2020, {1,10}×{fixed,season} | 1e-12 | 0.000e+00 | **PASS** |
| DE | turbine | postcode geolocation | onshore, 2015–2018→2019, {10}×{fixed} | 1e-12 | 0.000e+00 | **PASS** |
| NL | country | joint L-BFGS-B offsets | 2015–2019→2023, {5}×{fixed} | 1e-9 | 0.000e+00 | **PASS** |
| FR | country | joint L-BFGS-B offsets | 2015–2019→2023, {5}×{fixed} | 1e-9 | 0.000e+00 | **PASS** |

Every frame (8 for DK, 2 for DE, 2 each for NL/FR) matched at `0.000e+00`.
The DK/DE turbine result is exact to `1e-12`; the country result is exact too,
well inside the looser `1e-9` allowed for the L-BFGS-B path.

## Metric anchors (informational, not gates)

Harness skill on the held-out year, real curves:

| Region | Variant | MBE | MAE | RMSE |
|---|---|---|---|---|
| DK | uncorrected | +0.121 | 0.124 | 0.160 |
| DK | affine (10, season) | +0.041 | 0.066 | 0.099 |
| DE | uncorrected | +0.039 | 0.057 | 0.083 |
| DE | affine (10, fixed) | +0.001 | 0.044 | 0.061 |
| NL | uncorrected | +0.188 | 0.188 | 0.204 |
| NL | affine (5, fixed) | −0.136 | 0.136 | 0.142 |
| FR | uncorrected | −0.013 | 0.015 | 0.018 |
| FR | affine (5, fixed) | −0.007 | 0.012 | 0.019 |

- **NL reproduces a documented failure (R2 anchor).**
  `docs/TURBINE_GRID_EVALUATION_ANALYSIS.md` records the NL 2023 static-grid run
  as pathological: the uncorrected simulation over-predicts and the correction
  overshoots into large under-prediction (fleet grew +155% between training and
  test). The harness reproduces exactly that shape — uncorrected MBE +0.188,
  corrected MBE −0.136. The magnitude differs from the doc's MAE 0.114 because
  this run trains on 2015–2019 where the doc used 2015–2021; the direction and
  pathology match. Reproducing a known-bad result is as good a regression check
  as a known-good one.
- **DK error reduction (R3 anchor, loose).** The Energy paper reports ~43%
  error reduction at its best configuration (700 clusters, bimonthly) over the
  `PyVWF(1;fixed)` baseline. This run does not use that configuration, so it is
  not a direct check; at the configs run here the DK correction reduces MAE by
  ~47% (0.124→0.066 at 10/season), the same order of magnitude. Kept loose on
  purpose — the paper's preprocessing and fleet may differ.

## Reproduction

The committed pieces are `scripts/d1_regression.py` (the frame comparator) and
the wiring it exercised (`EntsoeFileSource`, country-level `run_evaluate`,
corrected-CF saving). The run orchestration (main worktree + staging dir) is
environment-specific and lives outside the repo; the staging dir is
`PYVWF_INPUT` pointed at a directory with the real curve files copied to the
working names.

## Conclusion

The refactor is behaviour-preserving on the validated European regions, at both
observation levels and across the true-coordinate, postcode, and joint-offset
paths. D1's win condition is met with zero discrepancies. This clears the way
for D2 (AU-NEM ingest), still gated on data-acquisition approvals.
