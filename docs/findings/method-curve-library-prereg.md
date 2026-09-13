# Curve library study: registered questions, conditions and gates

**Date:** 2026-09-11 (drafted); **revised 2026-09-13**, before any run of the
study. The revision is described in "What this revision changed" at the end,
and no result of this study existed when it was made.
**Scope:** the design of `method-curve-library.md`: what the power curve
library contributes to PyVWF's validation results, at country level and at
turbine level. Terms follow `CONTEXT.md`.

**Everything below is fixed before any run of the study.** The outcome columns
are filled in afterwards. A condition added later is labelled post hoc and
cannot pass a gate.

## Questions

- **Q1, country level.** How much of the country-level correction absorbs the
  specific-power mismatch of the fallback curve, rather than ERA5 bias? Every
  country-level unit is simulated on `2019COE_DW100_100kW_27.6` (167 W/m2),
  while the grids name Vestas V80-2.0 (398 W/m2), V90-2.0 (314 W/m2) and
  V90-3.0 (472 W/m2). The country rows have never run on the licensed library,
  so that part of the matrix is new data, not a re-run. The answer is a finding
  whichever way it falls.
- **Q2, turbine level.** Is specific power a sufficient statistic for power
  curve shape? Every turbine-level row except BR is assigned curves by
  specific power, and a large share of capacity runs on other-brand curves:
  DE 40.0%, US 48.3%, UK 21.8%, DK 11.6%. If held-out skill survives
  other-brand matching, the design choice is vindicated. If not, curve
  assignment is a larger error source than the ERA5 bias the correction exists
  to remove.

## Germany cannot answer Q2's own-machine half: a finding, measured in advance

T1 needs the register's own designation for a unit. **The German register has
none.** `DE_md.csv` carries 82 columns: `Manufacturer`, `kW`, `Rotor..m.`,
`Tower..m.`, `StartDate`, and 76 monthly generation columns. Its commonest
entries are `Enercon`, `Vestas`, `Nordex` and `Tacke`, which are brands, not
machines. Measured coverage for DE is **0.0% of capacity, 0 of 11,433 units**.

This changes what the German row means. DE's other-brand share has stood as
the study's concrete case, 13.4% of its capacity being Vestas turbines on
Gamesa curves, and read as the clearest example of specific-power matching
going wrong. **It is not that.** There was nothing better to match against:
`add_models` reads a manufacturer, a rating and a rotor diameter and nothing
else, so with no designation recorded, the nearest specific power within a
fuzzy manufacturer match is the only assignment the data permits. Germany is a
data-availability limit, not a method limit, and the two carry different
consequences: a method limit would argue for changing the matching, and this
argues for a better register.

**It is not a limit on the other-brand question.** T2 needs no designation, so
G2b reaches all four regions including Germany. It is the own-machine question
alone that Germany cannot answer.

The scorecard's interpretation of the curve-match audit says otherwise today
and is corrected separately, with this measurement as its evidence.

## Reported, not gated

Three facts belong in the write-up whatever the gates do, and none of them is
a condition:

- **DK has 23.0% of capacity with no recorded manufacturer**, so its curve
  match cannot be verified in either direction for that share.
- **BR records no manufacturers at all.** It is excluded from Q2's conditions
  for that reason, not for its result.
- **CL is the extreme case of the thing this study is about and is not in it.**
  91.6% of its capacity runs on a reference curve, the highest share in the
  scorecard, and its row carries both markers: † for a degenerate fit and ‡ for
  a gain that cannot be distinguished from zero when its units are resampled.
  It is not gated because it runs on the bundled open library and there is no
  licensed condition to compare it against: the licensed library has no Chilean
  coverage to add. Leaving it out silently would look like selection, so it is
  named here, with its markers, and reported in the write-up.

## Libraries

- **Open library:** the library shipped with PyVWF, 76 curves. Results on it are
  third-party reproducible and are reported first.
- **Licensed library:** 160 curves, not redistributable, identified by sha256.
  Results on it are reported alongside, marked as not third-party
  reproducible.
- **Combined library:** the disjoint union of the two, 236 curves. Checked on
  2026-09-11: no model key is in both, and all 76 open curves in the combined
  library are byte-identical to the open library. So no precedence rule is
  needed for the current data. If a key ever appears in both, the combined
  runs are invalid for it until a rule is fixed.

## Baselines

**C0 and T0 are the rows standing in the scorecard today, not the figures they
superseded.** For the eleven European rows that means the re-runs of
2026-09-13 on `era5/EU_2026-09` with the per-timestep roughness
(`method-eu-rerun.md`); for the US, the only turbine-level row in this study
that is not European, it means its existing run.

The reason is not tidiness. The superseded rows carry a documented input
defect: three were simulated from winds extrapolated up to five degrees past
the ERA5 data, and the rest carried a roughness treatment the project has
since replaced. Measuring what a curve library contributes against a baseline
the repository no longer stands behind would make this study's own numbers
uncitable.

## Conditions

Every condition uses the row's scorecard configuration
(`configs/regions/scorecard/`), its training years and its test year. Runs go
under `output/curve_library_study_<date>/`, one region per process.

**Three things are held identical across the conditions of a row**, and a
condition that changes any of them is a different experiment:

1. **The variant set.** The `cluster_list` and `time_slices` of the row's
   configuration, unchanged. Every variant of a run is scored on the rows
   common to all of them, so adding or removing one moves the score of the
   variant the row reports: in the AR row, the reported `fixed_10` score moved
   because an unreported `season_10` variant lacked two rows. The variant set
   is part of the design, not incidental to it.
2. **The ERA5 input.** The path, the box and the roughness treatment of the
   row's configuration. DK carries `allow_extrapolation = true`, as its
   scorecard row does, because its box stops short of Bornholm.
3. **The fleet, the observations, the training years and the test year.**

**Q1: the eight country-level rows** (BE, ES, FR, IE, IT, NO, PT, SE):

| Condition | Curve library | Grid model keys |
|---|---|---|
| C0, as reported | open | as in the grids; every unit substituted to the fallback curve |
| C1, licensed | combined | as in the grids; each resolves to its own Vestas curve |
| C2, open best match | open | each Vestas key replaced by the nearest-specific-power open model within the 0.5 to 2 times rating band (`assign_curves_from_library`'s rule) |

C2 is limited by the open library: its nearest in-band match for V90-3.0 is
`BAR_HighSP_5.0MW_134.9` at 350 W/m2, against 472 W/m2.

**Q2: the four turbine-level rows that depend on the licensed library** (DE,
DK, UK, US):

| Condition | Model keys |
|---|---|
| T0, as reported | the current curve assignment |
| T1, brand-and-spec match | a unit whose own model string maps exactly to a licensed-library key gets that key; every other unit keeps T0 |
| T2, other brand | a unit whose T0 curve is same brand gets the nearest-specific-power model of a different brand in the same rating band; every other unit keeps T0 |

T1's coverage, the share of capacity it reassigns, was measured before this
record was fixed, since it decides which regions the gate can reach and needs
no simulation (`scripts/analysis/curve_library_match.py`, whose rules are
fixed and unit-tested):

| Region | T1 coverage, by capacity | Units matched | Library keys reached | G2a |
|---|---|---|---|---|
| UK | 53.3% | 4,076 of 6,618 | 62 | gated |
| DK | 50.2% | 2,446 of 6,296 | 36 | gated |
| US | 32.0% | 309 of 1,091 | 34 | reported, ungated by rule |
| DE | **0.0%** | 0 of 11,433 | 0 | **untestable: no designation to match** |

**G2a therefore covers DK and UK.** Unverifiable units cannot enter T1 or T2
and keep T0.

*[Added 2026-09-13, before any run: what a G2a result would mean. `add_models`
reads a manufacturer, a capacity, a rotor diameter and a hub height, and no
model designation, for any region. Specific-power matching is therefore not a
fallback the code reaches when designations are missing; it is the only
assignment implemented. **T1 is not a variant of the method so much as the
first use of a column the data already carries**, for the Danish and British
registers that record one. So if T1 beats T0 on DK or UK, the remedy is not a
better heuristic: it is reading a field that was already there. That changes
what a G2a result implies without changing the gate, which is why it is
recorded here and dated rather than written into the gate's text.]*

## Scoring

**Every comparison is on common rows, twice over.** Within a run, the harness
already scores every variant on the rows all of its variants can score. Across
conditions, the study restricts to the rows common to all conditions of a row
before any difference is taken, as the roughness study and the European re-run
comparison do. A condition that loses rows the others keep is reported as
having lost them; it is never compared on its own surviving subset.

**Per row and condition, from `metrics.csv` and the run records:** uncorrected
RMSE and MBE; corrected RMSE and MBE for the row's reported configuration; the
correction gain; the scalar and offset distributions and `fit_quality`; the
fit diagnostics, which say what share of a cluster's training steps its fitted
pair sends below 0 m/s or above the curve; the off-curve and missing-value
counts; and the curve resolution with the curve-match audit classes.

**A condition that turns a clean fit degenerate is a result, not an artefact.**
If swapping the curve library pushes a fit outside the scalar bounds, or makes
an offset fail, that is evidence about how much the library carries: the
correction was absorbing curve error, and removing the error left the fit
without a stable solution. It is reported per condition as an outcome of the
study, with the cluster and the values, and the row that produced it keeps its
dagger in the write-up.

## The floor

**The primary test is the paired interval, not a threshold.** Procedure B of
this study: 1,000 paired draws, seed 20260911, 95% percentile intervals,
resampling units for turbine-level rows and months for country-level ones,
with the same draws for every condition of a row. A difference is resolved if
its interval excludes zero.

**There is no country-level floor.** Nothing has established one.

**The 0.002 turbine-level screen stands on its original basis and nothing
else:** ten null clustering A/B pairs with a maximum absolute difference of
0.0018 in MAE, with a second test year and seed repeats named as prerequisites
and never done. It is a screen for whether a resolved turbine-level difference
is worth acting on, and it is not a gate.

**The 3% figure is withdrawn.** The draft of 2026-09-11 made every gate a
comparison against "about 3%", citing `region-us-br.md`. That figure was never
established as a limit, it was relative where the gates needed absolute, and
using it would have made this study's conclusions rest on a number nobody
measured.

## Gates

Every gate has an indeterminate outcome, and indeterminate is a result: it
means the design could not answer the question, and the question stays open on
its own evidence rather than being settled by a narrow reading.

| Gate | Requirement | Outcome |
|---|---|---|
| **G1** (Q1) | For each country, the absorbed share is A = (uncorrected RMSE in C0 minus uncorrected RMSE in C1) divided by the correction gain in C0. **The scoreable set is fixed here, from C0, at seven:** BE, ES, FR, IE, IT, PT and SE. NO is excluded because its C0 correction gain is not positive, so there is no denominator to divide by. The fallback curve is a material part of the country-level correction if A is at least 0.5 in at least **4 of those 7**. **Indeterminate** if the interval on the C0-minus-C1 uncorrected RMSE difference includes zero in more than 3 of the 7, since A is then built on differences the design cannot resolve. | |
| **G2a** (Q2, T1) | **DK and UK only.** Specific power is sufficient for a region if the paired interval for corrected RMSE in T1 minus T0 includes zero, or excludes it by less than the 0.002 screen. It is insufficient if T1 beats T0 by more than the screen. **Precondition, measured before this record was fixed:** the gate applies only where T1 reassigns at least **10% of capacity**. DK (50.2%) and UK (53.3%) pass it; **DE is untestable at 0.0%**, for the reason above, and is reported as such rather than as a failure; **the US is ungated regardless of its coverage** (32.0%), which is reported as a finding about the licensed library's reach into the US fleet. **A T1 materially worse than T0 suspends this gate** pending a diagnosis of the matcher: a matcher that assigns a worse curve than specific power did is more likely to be wrong than specific power is to be right. | |
| **G2b** (Q2, T2) | **All four regions, DE included.** The same test for T2 minus T0. Specific power survives other-brand matching in a region if the interval includes zero or excludes it by less than the screen. T2 reassigns by rating band and specific power and needs no register designation, so Germany's missing designations do not reach it. | |
| **G3** (Q2) | Curve assignment is a larger error source than ERA5 bias for a region if the spread of uncorrected RMSE across T0, T1 and T2 exceeds that region's correction gain in T0. **Indeterminate** for a region whose T1 is suspended under G2a. | |

**The scoreable set is determined from C0 alone, and does not move.** C0 exists
today; C1 does not. Deciding scoreability from C1 would let the denominator set
shrink after the results are seen, and a threshold of four would stop meaning
what it meant when it was fixed.

**If a C1 run turns a country's correction gain negative, that is a result and
not an adjustment to this gate.** It would mean that giving the fleet its own
curves left the correction with nothing to repair, or something to make worse,
which is evidence about the curve library. It goes in the findings with its
figures. The gate's arithmetic stays on the seven named above.

A negative A, where the licensed curve raises the uncorrected error, is
reported as such.

## Registered predictions

| # | Prediction | Outcome |
|---|---|---|
| P1 | In C1, uncorrected MBE falls in at least 6 of the 8 countries, because a 167 W/m2 curve overproduces at moderate wind speeds. | |
| P2 | In at least 6 of the 8 countries, the paired interval for corrected RMSE in C1 minus C0 includes zero: the correction absorbs the curve error either way. | |
| P3 | G1 passes. | |
| P4 | In T2, the paired interval for corrected RMSE minus T0 includes zero in DE, DK and UK: at monthly resolution the correction absorbs other-brand curve shape. T2 reaches all four regions, so this is a prediction about three of them and the US is reported beside it. | |
| P5 | The paired interval for uncorrected RMSE excludes zero between at least two of T0, T1 and T2, in at least 2 of the 3 regions where all three conditions exist (DK, UK, US). Germany has T0 and T2 only. | |
| P6 | No condition turns a clean fit degenerate. | |

## Consequences, stated in advance

- **G1 passes:** the country-level scorecard rows and `method-country-level.md`
  say that the correction there largely repairs a curve mismatch. The
  correction notice stops saying the magnitude is unquantified.
- **G1 fails:** the curve substitution is recorded as a defect with a small
  effect on the country results. The notice gives the measured share.
- **G1 indeterminate:** the notice says the magnitude is still unquantified and
  states the intervals that failed to resolve it.
- **G2a or G2b fails:** specific-power matching becomes a leading caveat on the
  turbine-level scorecard rows.
- **Both pass:** the design choice is recorded as supported, for the regions
  that met the coverage precondition, at monthly resolution only.
- **G2a suspended:** the matcher is diagnosed before anything is concluded, and
  the suspension is reported whether or not the diagnosis succeeds.
- **G3 passes for a region:** that region's scorecard row carries it.
- **P6 fails, and a condition turns a clean fit degenerate:** this is the
  study's central claim arriving by another route, not a nuisance to note. A
  fit that was stable on a 167 W/m2 fallback and is unstable on the fleet's own
  curves was stable because it was carrying curve error: the scalar and offset
  had a mismatch to absorb, and removing the mismatch left them without one.
  The write-up says that in those terms, names the clusters and the values, and
  the row keeps its dagger. It would corroborate G1 independently of A, since
  it is a statement about what the fit was doing rather than about how much
  error it removed.

## Committed in advance

- No region is dropped after its results are seen. All eight country rows and
  all four turbine rows are reported.
- Open-library results lead. Licensed-library results follow, marked as not
  third-party reproducible.
- A deviation from this design is recorded, with its date and whether it came
  before or after a result was seen.
- No scorecard figure changes on this study alone. A condition is not a new
  row.

## Tooling this needs

This is new code, written after this record is committed and before any run:

- a study driver, `scripts/analysis/curve_library_study.py`. It builds a
  variant input root per condition under `output/`, with only the changed
  files copied and everything else linked, then runs train and evaluate for
  each row, one region per process. The link set now includes
  `era5/EU_2026-09`, and the driver checks that each condition's manifest
  records the same ERA5 path and roughness treatment as the row's scorecard
  configuration, so a condition cannot change the input by accident;
- an own-model matcher for T1. It normalises the registers' model strings
  (DK, DE, UK) and USWTDB's (US) to licensed-library keys, by exact normalised
  match only. The rules are fixed in the driver before the T1 runs, and unit
  tests check them on sample strings;
- the other-brand assignment for T2, reusing the rating band and nearest
  specific power;
- a comparison driver reusing the paired bootstrap and the common-row
  restriction already written for the roughness study and the European re-run,
  rather than a third implementation of the same arithmetic.

Estimated compute: about 90 minutes in total.

## What this revision changed, 2026-09-13

The draft was written on 2026-09-11 and predates eight changes to the project.
No run of this study existed when it was revised, so nothing here is a
post-hoc adjustment to a result; every change is dated and listed.

1. **The 3% noise floor is withdrawn**, and the paired interval of procedure B
   replaces it as the primary test. The draft had no resampling at all: its
   gates compared point estimates against a figure never established as a
   limit. G2, G3, P2, P4 and P5 all referenced it and are rewritten.
2. **Common-row scoring is now a stated condition**, within a run and across
   conditions. It did not exist when the draft was written.
3. **The variant set is registered** as part of each condition, after the AR
   row showed an unreported variant moving a reported score.
4. **C0 and T0 are the current rows**, not the superseded ones, for the reason
   given under Baselines.
5. **The extent guard and the roughness treatment** are held as part of each
   condition, including DK's opt-in. Neither existed in the draft.
6. **Degeneracy under a condition is registered as a result**, with the fit
   diagnostics and off-curve counts added to the recorded metrics.
7. **Chile is named in reported-not-gated**, with both its markers and the
   reason it is not a condition.
8. **T1 coverage was measured before the record was fixed**, which the draft
   left to the run. It decides which regions G2a can reach, and needs no
   simulation. G2a is now DK and UK; DE is untestable at 0.0% coverage,
   recorded as a finding about the German register rather than as a failed
   condition; the US keeps its coverage reported and its gate off by rule.
9. **The rulings taken after the draft are written in:** T1 renamed the
   brand-and-spec match; the 10% coverage precondition with the US ungated
   regardless; G1 over scoreable countries, 4 of 7 rather than 5 of 8, since
   NO has no positive gain to divide by; G2 split into G2a and G2b; a T1
   materially worse than T0 suspending its gate; an indeterminate branch on
   every gate; P2 restated as an interval; and P6 added for the degeneracy
   outcome.
