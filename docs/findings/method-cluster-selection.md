# Selecting a cluster count without leakage: what it cost and what it could not reach

**Reproduction record, added 2026-09-18.** Drivers:
`scripts/studies/method-cluster-selection/cluster_selection_study.py`,
`scripts/studies/method-cluster-selection/cluster_selection_gaps.py`,
`scripts/studies/method-cluster-selection/cluster_sweep_cost.py`. Until
2026-09-18 they were in `scripts/analysis/`, the path any command below uses;
`scripts/studies/README.md` maps each old path to its new one. Numbers: the run
manifests under `output/cluster_selection_2026-09-15/` record commits
`40317ee`, `cc9fc14` and `1ae99a3`, and those under
`output/cluster_sweep_cost_2026-09-15/` record `4af1ba4`, all clean trees.

**Date:** 2026-09-16
**Scope:** the five turbine-level configurations that contribute control points
to the pool: DE onshore, DK onshore, DK offshore, UK onshore, UK offshore.
Pre-registered in `method-cluster-selection-prereg.md`, whose gates and
predictions were fixed before any sweep and whose amendments are dated. Terms
follow `CONTEXT.md`.

**A protocol that removes the leakage selects cluster counts worse than the
chapter's in three of five rows, and the failure has three different causes.**
The conservative machinery explains Denmark offshore's shortfall entirely and
Denmark onshore's not at all. In two of the three failing rows the chapter's
count was outside the grid the protocol was allowed to search. **Neither of the
two obvious readings holds across the board**: it is not that the chapter's
counts were simply well chosen, and it is not that the machinery costs more
than the leakage it removes.

## What was run

Forward chaining inside the training years, one fold per training year after
the first, each trained on a contiguous prefix and validated on the next year,
so no fold trains on a year after the one it validates. For each cluster count
the mean fold score and its standard error; the selection is the smallest count
within one standard error of the best mean. Applied to RMSE and MAE separately,
with the smaller count taken where they disagree. Then a refit on all training
years and a single untouched test year.

Held constant: the `fixed` time slice, `input/era5/EU_2026-09` with the
per-timestep roughness, each configuration's shipped bounding box with
Denmark's study-scoped 15.4 east, and each row's curve library.

Runs: `output/cluster_selection_2026-09-15/`. Grids: 1, 10, 25, 50, 100, 200,
500, 1000 onshore, and 1, 2, 3, 5, 10, 25, 50, 100 offshore.

## The full table

Test-year MAE. B1 is the chapter's own count and B2 a fixed `k=100`.

| Row | folds | selected | RMSE chose | MAE chose | uncorrected | selected | B1 | B2 |
|---|---|---|---|---|---|---|---|---|
| DK offshore | 4 | 1 | 1 | 1 | 0.08457 | 0.04264 | 0.04073 (k=2) | 0.06555 |
| UK offshore | 3 | 25 | 25 | 25 | 0.15942 | 0.11224 | 0.13295 (k=10) | 0.11224 |
| DE onshore | 3 | 500 | 500 | 500 | 0.05952 | 0.03776 | 0.03776 (k=500) | 0.04030 |
| UK onshore | 3 | 50 | 50 | 200 | 0.08149 | 0.05141 | 0.04733 (k=300) | 0.04938 |
| DK onshore | 4 | 200 | 200 | 500 | 0.12506 | 0.05798 | 0.05691 (k=884) | 0.06019 |

Every row is corrected well below uncorrected. The question here is only which
count, not whether to correct.

## C-G3 decomposes into three causes

Three rows lose to the chapter's count. They lose for different reasons, and
separating them is worth more than the gate.

| Row | worse than chapter by | the rule's own cost | share explained | chapter's count in the grid? |
|---|---|---|---|---|
| DK offshore | 0.00191 | 0.00191 | **100%** | yes, `k=2` |
| UK onshore | 0.00408 | 0.00333 | **82%** | **no, `k=300`** |
| DK onshore | 0.00107 | 0.00003 | **3%** | **no, `k=884`** |

**Denmark offshore is entirely the machinery.** Its minimising count is the
chapter's count, and the one-standard-error rule stepped off it to `k=1`.

**Denmark onshore is entirely the grid.** The chapter's `k=884` scores 0.05691,
better than every count the grid contains, whose best is 0.05730 at `k=1000`.
No protocol searching this grid could have selected it.

**United Kingdom onshore is 82% machinery, on a grid that also excluded the
comparator.** Its `k=300` is not in the grid either, and the smaller-k tiebreak
took `k=50` over MAE's `k=200`.

**So C-G3 asked whether the protocol beats the chapter while forbidding it from
considering the chapter's answer.** That is a limitation of the grid registered
in advance, and it is the study's most useful finding about its own design. It
is recorded rather than repaired: **the grid is not widened and the gate is not
re-run**, because widening a grid after seeing which counts it excluded is
selection on the outcome, and the limitation is worth more than a repaired
gate.

## The one-standard-error rule's own cost

The gap between the selection and the minimising count, on the test year. The
registration fixed in advance that a gap above 0.002 is a finding about the
rule rather than about the row.

| Row | selected | its MAE | minimising | its MAE | gap |
|---|---|---|---|---|---|
| DK offshore | 1 | 0.04264 | 2 | 0.04073 | +0.00191 |
| UK offshore | 25 | 0.11224 | 50 | 0.11224 | 0.00000 |
| DE onshore | 500 | 0.03776 | 1000 | 0.03764 | +0.00012 |
| **UK onshore** | 50 | 0.05141 | 200 | 0.04808 | **+0.00333** |
| DK onshore | 200 | 0.05798 | 500 | 0.05795 | +0.00003 |

**One row exceeds the screen, and it is a finding about the rule.** United
Kingdom onshore costs 0.00333 against the RMSE minimiser and **0.00562 against
MAE's choice of `k=500`**, whose test MAE is 0.04579. In the other four the
rule is nearly free, the next largest cost being 0.00191.

**The smaller-k tiebreak fired twice and its two firings bracket the study.**
It is the same mechanism in both, and it cost almost nothing once and more than
anything else once:

| Row | RMSE chose | MAE chose | taken | cost of taking the smaller |
|---|---|---|---|---|
| DK onshore | 200 | 500 | 200 | +0.00003 |
| UK onshore | 50 | 200 | 50 | **+0.00333** |

Two conservatisms compound here by design, forward chaining giving early folds
less data and the one-standard-error rule preferring the simpler model, and
both push toward fewer clusters. **The selection is below the minimising count
in all five rows.** A selection at the bottom of its grid is not evidence that
the bottom is best.

## Curve shape, reported per row

From the mean fold score across the grid. Spread is the worst mean over the
best; ties counts how many are within 1% of the best; the plateau is the
largest group within 1% of its own floor.

| Row | best (RMSE) | spread | ties | plateau | label |
|---|---|---|---|---|---|
| DK offshore | 2 | 20.8% | 1 | 4 of 8 | plateaued |
| UK offshore | 25 | 75.5% | 3 | 3 of 8 | structured |
| DE onshore | 1000 | 32.1% | 2 | 2 of 8 | structured |
| UK onshore | 200 | 52.1% | 2 | 2 of 8 | structured |
| DK onshore | 500 | 24.9% | 1 | 1 of 8 | structured |

**No row is flat.** That matters because the flat curve was this protocol's
motivation: `method-cluster-count-dk.md` reports Denmark onshore spreading 0.7%
across a sixteenfold range, and the same configuration here spreads 24.9%. The
two differ in fleet, in grid, and in whether the score comes from
forward-chained training folds or one scored test year. **Nothing is yet shown
to be wrong with the published curve**, and the discrepancy is logged as its
own question in [`STATUS.md`](https://github.com/ellyess/PyVWF/blob/3dbb83b54355290b8a25727847fcca0640909706/STATUS.md) rather than reconciled here.

## The plateau has a mechanism, predicted before the curve existed

Grouping each training fleet into spatial components linked at 5 km, before
United Kingdom offshore's curve was known:

| | units | groups | sizes |
|---|---|---|---|
| DK offshore | 318 | 9 | 162, 111, then 10, 10, 8, 7, 5, 3, 2 |
| UK offshore | 981 | 22 | 118, 116, 100, 80, 75, 60, 50, 43, 40, 36, and 12 more |

Denmark's nine groups include two holding **86% of its units**. The registered
prediction was that if a cluster count stops paying once clusters correspond to
nothing, United Kingdom offshore should stay useful well past `k=3` and plateau
near its own group count.

**It held.** Denmark offshore's useful region is `k=1` and `k=2`, with a
four-count plateau from `k=3`. The United Kingdom improves steadily to `k=10`
and then jumps to a plateau at `k=25`, 50 and 100, all three scoring 0.08906.

**Stated at its real resolution:** the grid holds no value between 10 and 25,
so the break is located only within `(10, 25]`. The group count of 22 lies in
that interval, and **22 and 25 are indistinguishable here**. The claim
supported is that Denmark breaks at 3 with two dominant groups while the United
Kingdom does not break until above 10 with 22 even ones, not that the group
count was hit.

**The onshore rows show no break at all**, which the same instrument predicts:
on Denmark's onshore fleet, 5 km linking merges 16% of the turbines into one
component and 1 km fragments it into 1,719 groups with 756 singletons. There is
no threshold at which a group is a farm onshore, so the grouping measures
settlement density and the mechanism is not tested there.

## The selections are not adopted for the pool, 2026-09-16

**This study selected cluster counts and the control-point pool does not use
them.** A pool is a training set for the transfer problem, not an accuracy
target, and this study selected for out-of-sample skill on one held-out year,
which is a different objective. The rebuilt pool takes each configuration's
count as an inherited property, not as this study's answer, and
`method-why-corrections-do-not-transfer.md` is why: sample count is not the
binding constraint on transfer, so selecting a count cannot be expected to help
the thing the pool exists for.

*[Parked 2026-09-16: no rebuilt pool is being built. "The rebuilt pool" above
describes a plan. The one pool is the chapter's 1,729, the country tier cannot
be selected while the national study is blocked on training windows, and how
to build a defensible pool is an open question. See the dated note under D0 in
[`manuscript-chapters-45.md`](https://github.com/ellyess/PyVWF/blob/ce20ead0716ffcbe2ad132d616f14c288cbc679d/docs/design/manuscript-chapters-45.md).]*

Two of the selections would also have been poor choices for a pool on their own
terms. **Denmark onshore at 200 against the chapter's 884 drops 684 control
points**, and **United Kingdom onshore at 50 against 300 drops 243**, the
latter being the row whose one-standard-error gap exceeded the screen at
0.00333. Together they would have more than halved the pool, from 1,729 points
to 814.

The selections stand as this study's result. They are not the pool's.

## Caveats

- **Every result rests on a single held-out test year** and is
  screening-level, not an accredited yield assessment.
- **The grids are the limitation this study found in itself.** Two of the three
  rows that lose to the chapter lose to a count the grid does not contain.
- **Three folds for DE onshore, UK onshore and UK offshore**, so their standard
  errors come from three numbers and the interval is wide. That biases toward
  smaller counts, in the same direction as everything else here.
- **The fold scores are not comparable across folds within a row.** Denmark
  offshore's uncorrected RMSE nearly triples from 2016 to 2019 while its
  training fleet is fixed at 318 units, so the later folds are harder years
  rather than worse fits. The ranking within each fold is what the rule reads,
  and `k=2` ranks first in all four of Denmark offshore's.
- **Two rows were re-run** after a defect in the study runner let two fleet
  modes share a run directory; both reproduced their original selections on a
  clean grid. See the registration.
