# Roughness temporal treatment: registered question, conditions and gates

**Date:** 2026-09-12 (drafted; registered on commit, before any run)
**Scope:** which temporal treatment of the derived surface roughness PyVWF
should use. Terms follow `CONTEXT.md`; the background is
`docs/design/roughness-temporal-treatment.md`.

**Everything below is fixed before any run.** The outcome columns are filled in
afterwards. A condition added later is labelled post hoc and cannot pass a gate.

**Two deviations were recorded on 2026-09-12, before any DK run of this study
existed: G3 is withdrawn, and DK opts in to extrapolation in both conditions.
They are in the Deviations section at the end, which says what changed, why,
and whether a result had been seen. The registered text above them is left as
written.**

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
| **G1** (DK) | The paired interval for the R1 minus R0 corrected RMSE difference. **Indistinguishable** if it includes zero and its width is at most 0.004. **Resolved** if it excludes zero; the sign names the better treatment. **Indeterminate** if it includes zero and is wider than 0.004. | **Resolved: R1 better.** -0.00019, interval -0.00031 to -0.00007, width 0.00024. |
| **G2** (FR) | The same paired interval at country level, with no absolute floor. Reported as consistent or inconsistent with zero, with its width beside it, since a wide interval consistent with zero establishes nothing. | **Consistent with zero:** 0.0000, interval -0.0000 to 0.0001, width 0.0001. Narrow, and uninformative for the reason in D1. |
| **G3** (both) | A method change needs G1 and G2 to resolve and agree in direction. Two ways of failing that are different results, and are separated below. | Withdrawn on 2026-09-12, before any DK result: see deviation D1. |

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
| P1 | DK's R1 minus R0 corrected RMSE difference is indistinguishable under G1. | **Refuted.** The interval excludes zero, so G1 resolved rather than returning indistinguishable. |
| P2 | FR's interval includes zero. | **Held.** |
| P3 | R1 loses steps that R0 does not, in both rows, because the hourly roughness is undefined in some hours. | **Held for DK for the wrong reason, failed for FR.** DK: 49 unit-months partly scored under R1 against 47 under R0, all of them in the uncorrected variant, with the corrected variant losing nothing. FR: identical counts in both conditions, and its losses are in the corrected variant instead, from the affine pair rather than from an undefined roughness. The number came out right and the mechanism did not. None wholly missing, no rows excluded anywhere. |
| P4 | The uncorrected difference between conditions is larger than the corrected one, because the fit absorbs part of the change. | **Held, and the sign flips.** DK uncorrected +0.00149, corrected -0.00019: eight times larger, and in the opposite direction. The stated reason is wrong; see the finding. |

## Outcome, 2026-09-12

G1 resolved and named R1, the per-timestep treatment, the better one. **R1 is
adopted**, which is the consequence registered in advance for that result.

The reasons recorded for adopting it are wider than the gate, and the finding
states them rather than the gate: method fidelity, the inability of most rows
to show the effect at all, and the removal of a split that confounds every
comparison between regions. The measured accuracy effect is 0.0002 in
corrected RMSE, resolved by a paired design on 64,090 rows and far too small
to carry a method change on its own. A reader must not take "resolved" for
"material", and the finding says so in those words.

Full results, the P4 mechanism and the consequences for the download and the
European rows: `method-roughness-treatment.md`. Runs in
`output/roughness_treatment_2026-09-12/`.

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

## Deviations

Both were recorded on 2026-09-12, before any DK run of this study existed. FR
had already reported when they were written; DK had not.

### D1, 2026-09-12: G3 is withdrawn, and the country branch is untestable

**What changed.** G3 required G1 and G2 to resolve and agree in direction
before the method could change. It is withdrawn. DK alone decides the method,
under G1. FR is still run and still reported under G2, and its result is
reported as uninformative about the treatment rather than as evidence about it.

**Why.** The roughness reaches a simulated capacity factor only through the
hub-height profile,

    w(h) = w100 ln(h / z0) / ln(100 / z0)

At h = 100 m the factor is exactly 1 for every z0, so the roughness cancels
there. The treatment can move a speed only in proportion to the distance
between the hub height and the 100 m reference:

| Hub height | Factor at z0 = 0.01 m | at z0 = 0.25 m | Spread |
|---|---|---|---|
| 30 m | 0.869 | 0.799 | 0.070 |
| 45 m | 0.913 | 0.867 | 0.047 |
| 60 m | 0.945 | 0.915 | 0.030 |
| 80 m | 0.976 | 0.963 | 0.013 |
| 90 m | 0.989 | 0.982 | 0.006 |
| 100 m | 1.000 | 1.000 | 0.000 |

FR's 176 grid points all carry one hub height, 90 m, where the whole plausible
range of z0 moves a speed by 0.6%. This is not particular to FR. Every
country-level region gives its grid points a single uniform height, and every
one of those heights lies between 80 and 100 m: BE 100, SE 100, FR 90, ES 90,
IE 85, IT 80, NO 80, PT 80. At BE and SE the factor is identically 1 and the
treatment is exactly inert. No ENTSO-E grid in this repository can test the
treatment, so **the country branch is untestable as the grids stand**, which is
a result about the design and not about France.

**When, relative to a result.** After FR reported and before DK ran. The
hub-height check was made while reading FR's null. It does not rescue FR's
number or reinterpret it; it lowers what that number can support, from "the
treatment does not matter at country level" to "the treatment cannot be seen at
90 m". A deviation that weakens the study's own evidence is recorded on the
same terms as one that strengthens it, and this one is registered while the
informative row is still unrun.

**What it costs.** G3's divergent branch is now unreachable, because it needed
both rows to resolve. Whether the two pipeline branches want different
treatments stays open on no evidence either way, and the consequence stated for
a divergent result stands unused rather than refuted. Testing the country
branch needs grid points well below 100 m, which is a change to how
country-level grids are built and not a re-run.

**What is unchanged.** G1, G2, the floor, the metrics, the predictions P1 to
P4, and every branch of the download, including the indeterminate one. P2 is
still scored against FR's interval.

### D2, 2026-09-12: DK opts in to extrapolation, in both conditions

**What changed.** Both DK conditions set `[era5] allow_extrapolation = true`.
R0 runs from `configs/regions/study/dk_k100_stored.toml`, which is the
scorecard configuration plus the opt-in, and R1 from
`configs/regions/study/dk_k100_derived.toml`, which adds the derived treatment
to it. The scorecard configuration itself is untouched.

**Why.** DK's bounding box stops at 13.5°E and Bornholm lies near 14.9°E, so
the extent guard refuses the run. The runs record it: 15 of the 3,707 units in
the training fleet, 0.515% of its capacity, up to 1.55° beyond the loaded
extent (lon 7.5 to 13.5, lat 54.0 to 58.0), and 47 of the 5,446 in the test
fleet, 0.597%, up to 1.64°. Both conditions record exactly the same figures.

*[Correction, 2026-09-12, made before the DK results were read: this paragraph
first said 38 of 5,122 training units at 0.6%. That is the fleet metadata
before the join to the training years' observations, which is what a read-only
check of `train_set` returns; the fleet the run fits is the 3,707 above. The
test-fleet figures, and the 0.6% the scorecard's DK row carries, are
unaffected.]*

**Why not the alternatives.** Widening the box changes the input and the
treatment in one step, and would leave no run that isolates the treatment.
Substituting DE or UK loses the hub heights that make DK the informative row:
DK's units run from 12 to 140 m with a median of 45 m, against FR's uniform
90 m, and D1 above is exactly the reason that matters.

**What it does to the comparison.** The same units fall outside in both
conditions, and both extrapolate from the same edge winds, because the extent,
the box and the fleet are identical and only the treatment differs. The
hub-height speed at those units still differs between the conditions, exactly
as it does at every other unit, since that is the quantity under test. The
paired difference the study measures is therefore not confounded by the
extrapolation; each condition's absolute figures are.

**What it does not do.** It does not make the extrapolated winds sound. Under
the § rule a row with a non-zero extrapolated share carries the marker whether
or not the region opted in, and the scorecard's DK row is corrected in the same
sequence, before these runs, to carry § and its 0.6%. That correction rests on
the extent check, not on this comparison. These study runs are not scorecard rows.

**Logged, not started.** Widening DK's box to include Bornholm is separate
work: the hourly data is already in the European files, so it is a bounding-box
error and not a missing download. It produces a different DK row, so it needs
its own configuration, its own scorecard row and a statement of what moved. It
waits until this study reports.
