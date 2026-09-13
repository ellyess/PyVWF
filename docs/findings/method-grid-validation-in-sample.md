# Thesis chapter 4's end-to-end grid validation is in-sample, in all fourteen configurations

**Date:** 2026-09-13
**Scope:** the validation behind Tables 6 and 7 of the gridded-interpolation
chapter, and the Netherlands result the merged manuscript's case rests on. The
chapter is accepted and outside this repository; nothing in it is edited. Terms
follow `CONTEXT.md`.

**The chapter's end-to-end validation extracts gridded corrections at each
configuration's own observation locations from a surface interpolated from a
control-point pool that contains that configuration's own control points. There
is no holdout, no fold and no exclusion anywhere in that path.** Every one of
the fourteen country and mode configurations is therefore scoring a correction
that is, in the majority, its own fitted answer fed back to it.

**The Netherlands is the least in-sample of the fourteen and is still majority
self-determined.** Its headline result, the one that carries the cross-border
case, has 52.6% of its interpolation weight coming from its own five clusters.

## What the code does

Two scripts and no fold between them.

`compare_unified_corrections_to_grid.py` splits the pool by declared
`cluster_mode` into an onshore set of 1,717 points and an offshore set of 12,
interpolates each onto the 0.25 degree European grid, and writes
`europe_corrections_<method>.nc`. The interpolation call takes
`control_points` whole. There is a `spatial_cv_split` in the same file, and it
is used only for the cross-validation of Table 4; **it is not used when the
exported surfaces are built.**

`evaluate_grid_corrections.py` then reads `europe_corrections_<method>.nc` for
every country, extracts scalar and offset at that country's observation
locations, applies them to ERA5 winds, simulates capacity factors and scores
them. It takes no fold argument, and nothing in it removes a country's own
control points from the surface it reads.

So the quantity in Tables 6 and 7 is: how well does a correction surface
reproduce the observations at the locations whose corrections were fitted from
those same observations and then interpolated.

## How in-sample each configuration is, measured

For each configuration, the share of the inverse-distance weight at its own
footprint's grid cells that comes from its own control points, under the same
domain split the tables used:

| Configuration | Pool | Control points | Median self-weight | Minimum |
|---|---|---|---|---|
| UK offshore | offshore | 10 | **1.000** | 1.000 |
| DK offshore | offshore | 2 | **0.999** | 0.998 |
| DK onshore | onshore | 884 | 0.984 | 0.576 |
| UK onshore | onshore | 293 | 0.971 | 0.855 |
| PT | onshore | 3 | 0.962 | 0.961 |
| ES | onshore | 4 | 0.944 | 0.917 |
| SE | onshore | 4 | 0.913 | 0.752 |
| IT | onshore | 3 | 0.894 | 0.787 |
| DE onshore | onshore | 500 | 0.890 | 0.319 |
| IE | onshore | 3 | 0.870 | 0.785 |
| NO | onshore | 5 | 0.841 | 0.315 |
| FR | onshore | 10 | 0.838 | 0.720 |
| BE | onshore | 3 | 0.654 | 0.480 |
| **NL** | onshore | 5 | **0.526** | 0.193 |

**Fourteen of fourteen above 0.5. Seven above 0.9.** The two offshore
configurations are effectively total, because the offshore pool holds twelve
points across the whole domain and each configuration's own are the nearest by
a wide margin.

## The Netherlands, specifically

`grid_evaluation_metrics.csv` records, for NL in 2023: uncorrected MAE 0.2654,
cluster-based at five clusters 0.1162, grid IDW 0.1744, **grid kriging 0.0563**.

The chapter reads the last of those as cross-border borrowing: "the grid
benefits from surrounding high-density control points in Germany and Belgium,
providing better spatial information than the 5 national clusters alone".

**The five national clusters are in the surface.** So the comparison is not
*NL predicted from its neighbours* against *NL from its own data*. It is *NL's
own five clusters applied as a cluster correction* against *NL's own five
clusters plus its neighbours, interpolated*. The second is a real and useful
result, and it is a different claim: adding neighbouring information to a
sparse country fit helps. **It is not evidence that a country can be predicted
without its own observations**, which is what the merged manuscript's question
asks and what the phrase cross-border borrowing is normally taken to mean.

## What this does and does not overturn

**It does not make the numbers wrong.** Tables 6 and 7 measure what they
measure, and an in-sample correction surface is a legitimate object: it is what
a user with observations in their country would actually deploy. The defect is
in what the numbers are taken to show.

**It does overturn the cross-border reading of them**, which is the reading the
merged manuscript was going to be built on, and it removes the Netherlands as
an existence proof for prediction without local data.

**It is consistent with the country-holdout study run the same day**
(`method-loco-interpolation-prereg.md`). Removing a country's own control
points and predicting its correction parameters from the rest gives a negative
R-squared in 41 of 48 fold-by-method cells, and the Netherlands fold scores a
scalar MAE of 0.523 against its own points' spread. The two results say the
same thing from opposite directions: what looked like successful cross-border
transfer was largely a country predicting itself.

**It does not settle the end-to-end question**, because the holdout study
measures correction parameters and the chapter measures capacity factors at
observation locations. Those are different quantities. The run that settles it
is registered in `method-grid-nl-holdout-prereg.md` and has not been performed.

## Evidence

- `development:scripts/pyvwf_to_grid/compare_unified_corrections_to_grid.py`,
  `prepare_control_points` and the export path at lines 397 to 406 and 935.
- `development:scripts/pyvwf_to_grid/evaluate_grid_corrections.py`, line 374,
  which resolves the surface by method name only.
- `output/pyvwf_to_grid/grid_evaluation/grid_evaluation_metrics.csv`, the NL
  rows quoted above, which match the chapter's Table 7 to three decimals.
- `output/pyvwf_to_grid/all_corrections_centroids.csv`, the pool, including the
  five NL control points.
- The self-weight table above, computed from that pool and the chapter's own
  0.25 degree grid definition.

## Caveats

- The self-weight measure is inverse-distance with the chapter's exponent, at
  the grid cells nearest each configuration's control points. It describes the
  IDW surface exactly and the kriging surface only approximately, since kriging
  weights come from a fitted variogram rather than from distance alone. The
  direction is not in doubt: kriging weights also decay with distance.
- Observation locations are approximated by each configuration's own control
  points' nearest grid cells. For the turbine-level rows the fleet is spread
  more widely than its cluster centroids, which would lower those rows' figures
  somewhat and cannot raise them enough to matter at 0.98.
- This is a statement about the validation design, not about the corrections.
  Whether the corrections themselves are good is what the chapter's
  cross-validation, Table 4, addresses, and that one does hold out.
