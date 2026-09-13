# Which points are offshore: registered design

**Date:** 2026-09-13, committed before either pool is interpolated.
**Scope:** whether chapter 4's one documented interpolation failure, Denmark
offshore, is a property of the correction surface or of how the offshore
control points were defined. Terms follow `CONTEXT.md`.

**Everything below is fixed before any number exists.** The outcome column is
filled afterwards.

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

## Registered predictions

| # | Prediction | Outcome |
|---|---|---|
| O-P1 | O1 passes: Denmark offshore stops failing at 13 control points. The 11 added Danish clusters are coastal and near the offshore sites, so the surface over them is constrained by data rather than extrapolated. | |
| O-P2 | The United Kingdom offshore, which gains 8 points, moves less than Denmark, because it already had 10 and was not in the sparse regime. | |
| O-P3 | IDW moves less than kriging under the pool change in every affected row, since its weights are local and bounded and it was never the method the sparse pool broke. | |
