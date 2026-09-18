# Which points are offshore: registered design

**Date:** 2026-09-13, committed before either pool is interpolated.
**Scope:** whether chapter 4's one documented interpolation failure, Denmark
offshore, is a property of the correction surface or of how the offshore
control points were defined. Terms follow `CONTEXT.md`.

**Everything below is fixed before any number exists.** The outcome column is
filled afterwards.

## VOID, 2026-09-13, before any gate was read

**This study's premise is false and it is withdrawn.** It assumed the chapter's
Tables 6 and 7 were produced from surfaces split by the declared
`cluster_mode`, so that P0 would reproduce the chapter and P1 would test an
alternative. The chapter's script splits the pool and then concatenates it
straight back together before interpolating, under a comment saying so, so
**the tables use no split at all**. P0 therefore reproduced nothing: it gives
0.0496 for Denmark offshore against the chapter's 0.1113, and misses the other
three rows too. P1 against P0 compares two constructions the chapter never
used.

The run happened and its numbers are kept, in
`output/offshore_pool_2026-09-13/`. What they showed, noted and not written up:
P1 is worse than P0 in all four rows, against registered prediction O-P1. It is
internally valid, both arms sharing every choice but the pool, and it is
uninterpretable while neither arm reproduces the chapter.

Nothing here was read as a gate. The replacement question is registered in
`method-domain-split-prereg.md`, and it is larger and better founded than this
one.

The text below is left unedited, as the record of what was asked.

## Why this exists

Chapter 4 reports that gridded kriging makes Denmark offshore worse than no
correction at all, an MAE of 0.111 against an uncorrected 0.082, where IDW
reaches 0.051. It gives a cause: "With only 2 offshore control points, the
kriging surface produces unreliable corrections", and draws a recommendation
from it, that kriging needs roughly 5 to 10 spatially distributed control
points per domain.

The two points come from the **declared** pool. The chapter's
`prepare_control_points` splits on the run configuration's mode:

```python
onshore = df[df['cluster_mode'].isin(['onshore', 'all'])].copy()
offshore = df[df['cluster_mode'] == 'offshore'].copy()
```

Classifying the same 1,729 points against the project's own region shapes
instead gives a different split: 32 offshore rather than 12, because 19
clusters declared onshore fall inside offshore shapes, 11 of them Danish.
**Denmark's offshore pool would be 13 points rather than 2**, which is inside
the range the chapter's own recommendation calls sufficient.

So the chapter's stated cause is testable, and the test decides whether its
recommendation rests on anything.

## The two conditions

| Condition | Offshore pool |
|---|---|
| **P0** | Declared: `cluster_mode == "offshore"`, 12 points. Reproduces the chapter. |
| **P1** | Shape-classified: strictly inside `offshore_shapes.geojson`, 32 points. |

**A study-scoped bounding box, decided before the run.** The chapter-era Danish
onshore fleet reaches 15.139 degrees east and today's `configs/regions/dk.toml`
stops at 13.5, so 47 of its 4,888 units, **0.82% of capacity**, would fall
outside the loaded extent. `era5/EU` holds data to 22.0 east, so this is a box
to widen and not an extrapolation to allow: the box is widened to 15.4 east
**for this study only**, `dk.toml` is untouched, and no unit is simulated from
extrapolated winds. Both conditions load the same extent, so the comparison is
unaffected either way. Bornholm remains its own open item in [`STATUS.md`](https://github.com/ellyess/PyVWF/blob/3dbb83b54355290b8a25727847fcca0640909706/STATUS.md), and
the chapter-era fleet has proportionally more capacity out there than today's
row does: 0.82% against 0.6%.

Everything else is held: the same 1,729 control points, the same interpolation
functions ported from the chapter's own script, the same variogram
configuration the chapter adopted (ordinary kriging, exponential, geographic),
the same evaluation at the same observation locations, the same test years.

The onshore pool is whatever is left in each condition, so the two conditions
partition the same pool and no point is dropped or duplicated.

## What is scored

**Denmark offshore is the primary case**, because it is the failure the
chapter explains. **Every affected row is reported**: any country and mode
whose control-point membership differs between P0 and P1, which is Denmark
onshore and offshore and the United Kingdom onshore and offshore, plus any row
whose evaluated MAE moves by more than 0.001 for any other reason. Rows that
cannot move are stated as such rather than omitted.

Reported per row: MAE for uncorrected, cluster-based, grid IDW and grid
kriging, under both pools, with the offshore control-point count beside each.

## Fixed in advance: what each outcome means

**If Denmark offshore still fails under P1**, at 13 control points, the
chapter's result stands and its explanation is wrong. The failure is a
property of the kriging surface over this control-point geometry rather than
of the point count, and the chapter's recommendation of 5 to 10 points per
domain is not supported by the case it was drawn from. The manuscript reports
the failure, drops the count-based recommendation, and says what the diagnosis
is not.

**If Denmark offshore stops failing under P1**, the failure was an artefact of
the pool definition. The chapter's number stands as something that happened,
its explanation is wrong in the other direction, and the recommendation is
unsupported for a different reason. The manuscript reports that a conclusion
about an interpolation method turned on a metadata field, which is a finding
about the study design rather than about kriging.

**Both outcomes are publishable and neither rescues the recommendation.** That
is fixed here so that whichever way it falls, the recommendation does not
survive by default.

**If the two pools give the same offshore membership for Denmark**, the design
is void and is reported as such: the shape classification was checked on
2026-09-13 and gave 13, so this is a guard against a shape file changing under
the study rather than an expected outcome.

## What the port must not decide

The ported `export_pyvwf_grid` uses the **declared** pool, P0, because a port
reproduces rather than decides. The 30 points where the declared mode and the
shape classification disagree are reported as a diagnostic beside it, not
acted on. This study is where the choice is examined.

## Gates

| Gate | Requirement | Outcome |
|---|---|---|
| **O1** | Denmark offshore's grid kriging MAE under P1 is below its uncorrected MAE of 0.0822. That is the minimum for the chapter's explanation to be the right one, since it says the failure is caused by having too few points. | |
| **O2** | No row that the pool change does not touch moves by more than 0.001 in MAE. If untouched rows move, the two conditions differ in something other than the pool and the comparison is void. | |

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
| O-P1 | O1 passes: Denmark offshore stops failing at 13 control points. The 11 added Danish clusters are coastal and near the offshore sites, so the surface over them is constrained by data rather than extrapolated. | |
| O-P2 | The United Kingdom offshore, which gains 8 points, moves less than Denmark, because it already had 10 and was not in the sparse regime. | |
| O-P3 | IDW moves less than kriging under the pool change in every affected row, since its weights are local and bounded and it was never the method the sparse pool broke. | |
