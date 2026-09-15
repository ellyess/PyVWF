# The Netherlands end-to-end holdout: registered design

**Date:** 2026-09-13, committed before the run.
**Scope:** whether a country's gridded wind correction can be predicted from its
neighbours' observations alone, measured where the chapter measures it, in
capacity-factor error at observation locations. Terms follow `CONTEXT.md`.

**Everything below is fixed before any number exists.**

## Why this exists

Thesis chapter 4's Netherlands result carries the cross-border case: a gridded
kriging correction reaches a capacity-factor MAE of 0.0563 where the country's
own five-cluster correction reaches 0.1162. **That surface contains the
Netherlands' own five control points**
(`method-grid-validation-in-sample.md`), so it never tested prediction without
local data, and the country-holdout study that did test it, in correction-
parameter space, found a negative R-squared.

The two studies measure different quantities. This one closes that gap by
running the chapter's own end-to-end evaluation with the Netherlands withheld
from the surface.

## The conditions

| Condition | Surface built from |
|---|---|
| **H0** | All 1,729 control points. Reproduces the chapter. |
| **H1** | The 1,724 points that are not Dutch. The Netherlands predicted from its neighbours alone. |

Held constant: the domain split by declared `cluster_mode`, the kriging
configuration the chapter adopted, the 0.25 degree grid, the extraction at
observation locations, the test year 2023, the observations, and the fleet.

Both conditions are also run for IDW, because the chapter ships IDW as its
default and its Dutch IDW number, 0.1744, is worse than its kriging one.

## What is scored

Capacity-factor MAE at Dutch observation locations for 2023, under H0 and H1,
for kriging and IDW, against two baselines the chapter already reports:
**uncorrected ERA5 at 0.2654** and **the five-cluster country correction at
0.1162**.

RMSE and mean bias are reported beside MAE. The share of Dutch grid cells
beyond the 5 degree mask is reported under both conditions, since H1 moves the
nearest control point further away and the mask is what the product would do
about it.

## The coverage defect travels with every number

The Netherlands is excluded from this project's scorecard because an ENTSO-E
coverage defect caps its observed capacity factor at 0.57, which no rescaling
fixes (`CLAUDE.md`). **Every figure this study produces carries that statement**,
including in any table that quotes it. The defect does not invalidate a
comparison between corrections on the same observations, which is what this is,
and it does bound what the numbers mean about the Netherlands.

## Gates

| Gate | Requirement | Outcome |
|---|---|---|
| **H-G1** | H1's best gridded MAE is below the uncorrected 0.2654. This is the minimum for cross-border prediction to be worth anything: below it, a correction learned entirely from other countries is worse than no correction. | |
| **H-G2** | H1's best gridded MAE is below the five-cluster country correction's 0.1162. This is the chapter's claim as the manuscript needs it: neighbours alone beat a sparse local fit. | |
| **H-G3** | **Amended 2026-09-15: on the uniform grid only.** H0 reproduces the chapter's published 0.0563 for kriging and 0.1744 for IDW, to three decimals, when run on the uniform grid the chapter used. **If it does not, the study is void** and the discrepancy is diagnosed before anything else is read, because H1 is only interpretable against an H0 that reproduces. | |
| **H-G4** | **Added 2026-09-15.** Every condition is also run on the maintained fleet-weighted grids, and the difference between the two grids is reported per condition as a measured quantity. It gates nothing. This study is the one place the workstream runs both grids, because its purpose is testing the chapter's cross-border claim and that claim lives on the uniform grid; running both is how the manuscript states what the grid change did rather than asserting it. | |

## Amendment, 2026-09-15: both grids, and why only here

The workstream's founding assumption (`../design/manuscript-chapters-45.md`,
D0) is that country-level results are computed on the maintained fleet-weighted
grids and the chapter is a historical baseline. That would ordinarily retire a
reproduction gate. It does not here: this study exists to test the chapter's
headline cross-border claim, and the claim was made on the uniform grid, so a
run that cannot reproduce it cannot test it either. Both grids are run, H-G3
gates the uniform arm, and the maintained arm is reported beside it under H-G4.

## A gate stated against another pipeline's number, 2026-09-13

**Registered before any number of this study exists.** Several gates here are
stated against figures published by thesis chapter 4, which were produced by a
different pipeline: a different roughness treatment, no extent guard, and a
codebase 315 commits behind. Re-running today cannot reproduce those conditions
and is not trying to.

The rule, which applies to every gate in this document and to any later one:

**A gate stated against a figure from another pipeline is either verified
against today's equivalent or restated against it. It is never read across
pipelines silently.**

In practice: today's equivalent of the published figure is computed first and
reported beside it. If the two agree closely, the gate reads as registered and
the agreement is the evidence that it may. If they diverge, the gate is read
against today's figure and **the substitution is recorded as a dated
deviation**, with both numbers, because a gate's substance is a comparison
between two things measured the same way.

The direction this protects against is specific: a gate like "the correction
beats no correction" is meaningless if the correction is measured in one
pipeline and the baseline quoted from another, since the difference then
carries every change between the two.

## Registered predictions

| # | Prediction | Outcome |
|---|---|---|
| H-P1 | H-G1 passes. Even neighbours-only, a correction beats uncorrected ERA5, because the Dutch bias is large and in the same direction as its neighbours'. | |
| H-P2 | **H-G2 fails.** Withdrawing five local control points costs more than the neighbours supply, and the country-holdout study already found the parameters do not transfer. | |
| H-P3 | The gap between H0 and H1 is larger for kriging than for IDW, because kriging's weights are fitted globally and the Dutch points are the ones anchoring that region. | |

## Fixed in advance: what each outcome means

**H-G2 passes:** the cross-border claim survives in the quantity that matters,
and the chapter's reading is right for the wrong reason. The manuscript keeps
the Netherlands as its existence proof and states that the chapter's own
version of it was in-sample.

**H-G2 fails and H-G1 passes:** a correction from neighbours alone helps but
does not match local data. The manuscript reports that cross-border transfer is
real and weak, drops the Netherlands as an existence proof, and the merged
paper's answer is the one the country-holdout study points to.

**Both fail:** a correction learned from neighbours is worse than no correction
at that location, and the manuscript says that plainly.

**No outcome restores the chapter's Netherlands sentence as written**, since
that sentence describes a holdout that was not performed. This is fixed here so
that a passing gate is not read as a vindication of the original claim.

## What it needs from the port, and what it already has

**The Dutch inputs exist**, under
`output/runs/turbine_grid/NL-all-obs_country-corrected-calc_z0/`: the grid
points (`NL_2023_turb_info.csv`), the observed capacity factors
(`NL_2023_obs_cf.csv`), the uncorrected simulation (`NL_2023_unc_cf.csv`), the
corrected one, and the five-cluster factors the chapter's cluster baseline
comes from. So no region config has to be rebuilt and no observations have to
be reacquired.

**They are chapter-era and carry no manifests**, so nothing about how they were
produced is attributable to a code state. That is recorded rather than repaired:
this study compares surfaces against the same fixed observations, so a
provenance gap in the inputs is shared by every condition and cannot favour one.
H-G3 is the guard, since an H0 that reproduces the chapter's published number
is evidence that the inputs are the ones the chapter used.

Still blocked on the port:

- **a masked surface built from a given control-point set**, which is
  `atlite_export.py`, the next port phase. It is the only genuinely missing
  piece;
- **extraction at observation locations and simulation of corrected capacity
  factors**, which is `evaluate_grid_corrections.py` in the driver phase after
  it. Its scoring is a plain capacity-weighted monthly MAE against
  `NL_2023_obs_cf.csv` and could be written directly, but it should not be:
  a second implementation of the chapter's scoring would be a second definition,
  and H-G3 compares against the chapter's published number.

So the run is two port phases away, and nothing about it needs data that does
not exist.
