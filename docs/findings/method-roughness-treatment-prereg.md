# Roughness temporal treatment: registered question, conditions and gates

**Date:** 2026-09-12 (drafted; registered on commit, before any run)
**Scope:** which temporal treatment of the derived surface roughness PyVWF
should use. Terms follow `CONTEXT.md`; the background is
`docs/design/roughness-temporal-treatment.md`.

**Everything below is fixed before any run.** The outcome columns are filled in
afterwards. A condition added later is labelled post hoc and cannot pass a gate.

## The question

Every region derives the roughness length by inverting the log wind profile
between the 10 m and 100 m winds. Eleven of the seventeen scorecard rows then
apply a single annual mean of that quantity, and six apply it hour by hour,
averaged to daily with the winds. The difference is not the formula, it is
whether the result varies in time.

**Does the treatment change held-out skill materially?** Nobody has established
which is better. The hourly derivation follows conditions and is what the
published method describes. The annual mean may be the more stable estimator,
because the hourly value is noisy and undefined outright for some hours and,
in complex terrain, for whole months
(`docs/design/undefined-roughness-in-complex-terrain.md`).

The answer decides which treatment the extended ERA5 download generates, so the
download waits on it.

## Rows

Two rows, because the two pipeline branches fit differently:

- **DK**, `configs/regions/scorecard/dk_k100.toml`: turbine-level, k=100,
  `season`, trained 2015 to 2019, tested on 2020.
- **FR**, `configs/regions/scorecard/fr_country.toml`: country-level, N=10,
  `fixed`, trained 2015 to 2021, tested on 2023. The country path fits one
  national series with joint offsets, so a change in the input enters the fit
  differently there. FR lies wholly inside the loaded ERA5 extent and is not
  suspended.

Everything else is held at each row's scorecard configuration: same fleet, same
years, same cluster count, same time slice, same curve library, same ERA5
files.

## Conditions

| Condition | Roughness applied |
|---|---|
| R0, as reported | the stored annual mean in the European files, which is what these rows run on today |
| R1, hourly | derived from the same files' 10 m and 100 m winds per timestep and averaged to daily, which is what `prep_era5` does for every region with no stored field |

No new data. Both conditions read the same European files; R1 ignores their
stored `z0` and derives it instead. The hourly components are in those files.

## Metrics

Per row and condition, from `metrics.csv` and the run records: uncorrected and
corrected RMSE, MBE, MAE and correlation; the correction gain; fit quality and
the fit diagnostics (zero-crossing speed, the share of training steps sent
below 0 m/s or above the curve); the off-curve and missing-value counts; and
the extrapolated share, which is zero for both rows.

Both conditions are scored on their common rows, as the harness now does. R1
may lose steps that R0 never lost, because an hourly roughness is undefined
under vanishing or inverted shear while an annual mean always has a value.
That is a property of the treatment, is reported as a result in its own right,
and does not excuse a comparison on different rows.

## The floor

Procedure B of the curve library study, unchanged: paired bootstrap, 1,000
draws, fixed seed, resampling units for DK and months for FR. The primary test
is whether the paired interval for the R1 minus R0 corrected RMSE difference
excludes zero.

The 0.002 turbine-level screen stands on its original basis and nothing else:
ten null clustering A/B pairs with a maximum absolute difference of 0.0018 in
MAE, never established as a limit, with a second test year and seed repeats
named as prerequisites and never done.

**A coincidence was checked and rejected.** The published method's offset
search stops when its step falls below 0.002, described there as the power
curve resolution. That 0.002 is a step in the offset, which is a wind speed in
m/s (`_find_offset_iterative`, whose initial step is 10.0 m/s). The screen's
0.002 is a difference in capacity-factor error. Same digits, different
quantities, so the published tolerance is not evidence for the screen and is
not used here.

At country level there is no absolute floor, as in the curve library study.

## Gates

| Gate | Requirement | Outcome |
|---|---|---|
| **G1** (DK) | The paired interval for the R1 minus R0 corrected RMSE difference. **Indistinguishable** if it includes zero and its width is at most 0.004. **Resolved** if it excludes zero; the sign names the better treatment. **Indeterminate** if it includes zero and is wider than 0.004. | |
| **G2** (FR) | The same paired interval at country level, with no absolute floor. Reported as consistent or inconsistent with zero, with its width beside it, since a wide interval consistent with zero establishes nothing. | |
| **G3** (both) | A method change needs G1 and G2 to resolve and agree in direction. Two ways of failing that are different results, and are separated below. | |

**0.004 is a judgement, not a measurement.** It states how much imprecision is
accepted before a comparison counts as resolved, and it is fixed here before
any result. It is informed by DK's measured marginal corrected-RMSE interval
width of 0.006 (`output/curve_library_study_2026-09-11/`), and a paired
interval is narrower than a marginal one, but that reasoning gives a direction
and not a number: it does not select 0.004 over 0.003 or 0.005. Nothing in the
data picks the value, and it should not be read as derived.

**G3 has two failure modes, and they are not the same result.**

- **Underpowered:** one or both rows indeterminate. Nothing is concluded, and
  the question stays open on its own evidence.
- **Genuinely divergent:** both rows resolve, with opposite signs. That is a
  result about the two pipeline branches, not a null: the turbine-level and
  country-level paths would want different treatments, because they fit
  different things (per-cluster ratios against one national series with joint
  offsets). It would mean generating for a split method, one treatment per
  branch, which the European files cannot express today, since one file set
  serves both. The download would then produce the hourly components for both
  and the stored field for whichever branch keeps it, and the split would be
  stated in the scorecard as a property of the method rather than of a region.

## Registered predictions

| # | Prediction | Outcome |
|---|---|---|
| P1 | DK's R1 minus R0 corrected RMSE difference is indistinguishable under G1. | |
| P2 | FR's interval includes zero. | |
| P3 | R1 loses steps that R0 does not, in both rows, because the hourly roughness is undefined in some hours. | |
| P4 | The uncorrected difference between conditions is larger than the corrected one, because the fit absorbs part of the change. | |

## Consequences, stated in advance

- **Indistinguishable:** the annual mean is kept, and recorded as a validated
  simplification rather than an accident. The extended download reproduces it,
  and the re-run changes extent only.
- **R1 better:** the hourly treatment becomes the method. The download
  produces it, the European rows are re-run on it, and the scorecard says which
  rows changed and by how much. The thesis disclosure stands as history.
- **R0 better:** the interesting outcome. It becomes a finding and a
  recommendation, and the roughness design note is rewritten around it.
- **Indeterminate:** no method change, and the question stays open with its
  measured intervals recorded.
- **Divergent:** no single method. The split is the finding, and the
  consequence for the download is in G3 above.

**What the download does in each case, fixed now.** The extended ERA5 download
is waiting on this record and cannot wait on a later decision. If anything
resolves, the download generates the treatment that wins. **If nothing
resolves, the download generates the stored annual mean**, exactly as the
European files carry it today, so that the extent change stays isolated and the
roughness question reopens later on its own evidence. That is fixed here so it
is not decided under pressure once results are in.

## Committed in advance

- Both rows are reported whichever way each falls, and neither is dropped.
- Reproducing the published Denmark results is out of scope here. It needs
  onshore-only mode, a sweep to 3,300 clusters and three metric scales, and is
  logged as candidate work instead.
- No figure elsewhere changes on the strength of this comparison alone.
- A deviation from this design is recorded, with its date and whether it came
  before or after a result was seen.

## Tooling this needs

- A way to ask for the derived roughness when a stored one is present:
  `[era5] roughness = "stored"` (default) or `"derived"`, carried on
  `RegionSpec` and passed to `prep_era5`. The two study configurations are then
  ordinary configs, and the same switch is what a method change would flip.
  Written after this record is committed, with tests.
- A driver that runs train and evaluate for each row and condition under
  `output/roughness_treatment_2026-09-12/`, one region per process, and the
  existing bootstrap, off-curve and audit scripts over the results.

Estimated compute: under an hour in total.
