# Correcting a unit from control points of a different kind: registered design

**Reproduction record, added 2026-09-18.** Driver:
`scripts/studies/method-domain-split/domain_split_study.py`. Until 2026-09-18
it was in `scripts/analysis/`, the path any command below uses;
`scripts/studies/README.md` maps each old path to its new one. Numbers: the
output under `output/domain_split_2026-09-15/` records no commit; the driver's
last commit before it was written is `f305b28`. The study is void, as this
registration records; the S1 and S2 figures are cited in
`docs/design/manuscript-chapters-45.md`.

**Date:** 2026-09-13. Registered before the study runs, with the four figures
already in hand declared below.
**Scope:** whether thesis chapter 4's one documented interpolation failure is
caused by correcting offshore units from onshore control points, and whether
the domain split the chapter's own code comment says it intended fixes it.
Terms follow `CONTEXT.md`. Replaces `method-offshore-pool-prereg.md`, which is
void.

## The study is void, and what re-founding it would take, 2026-09-15

**S1 and S2 were run and have not been read, and the study stays void.** S-G1
failed on Denmark offshore by 0.00175 against a 0.001 gate, and the diagnosis
is complete: the reference condition on the chapter's own archive reproduced
all nine covered rows to within 0.00005, so the miss is entirely the roughness
treatment. Restating a gate after seeing which way the miss went is the error
this registration was written to prevent, so it was not restated.

**It is now void for a second and larger reason.** The workstream's founding
assumption (`../design/manuscript-chapters-45.md`, D0) is that country-level
results are computed on the maintained fleet-weighted grids, while every figure
in this study, and the control-point pool it interpolates, is on the uniform
grids. **If this study is ever re-founded it is on the maintained grids and
with no gate against a chapter figure**, which means S-G1 does not carry over
in any form. S-G2 and S-G3 are internal and would.

The registration below is kept intact rather than edited, so that what was
asked can be read against what happened.

## Amendment, 2026-09-15: real winds everywhere, and what that does to S-G1

**Dated before the run it governs.** The study now reads
`input/era5/EU_2026-09` rather than `input/era5/EU`, and no row extrapolates.
Reproducing the chapter matters for auditing the chapter; it does not matter
for the manuscript, which needs results that are correct. Winds that do not
exist are not a condition worth preserving.

**Five rows, not four, rest on extrapolated winds in the chapter.** The
chapter's archive spans 12 west to 22 east and 42 to 72 north. Measured on the
chapter-era fleets, against each row's own shipped bounding box sliced from
that archive:

| Row | Units beyond the chapter's archive | Capacity share |
|---|---|---|
| IT all | 55 of 110 | 89.35% |
| PT all | 54 of 60 | 84.92% |
| ES all | 60 of 84 | 42.80% |
| NO all | 117 of 338 | 35.68% |
| SE all | 9 of 72 | 5.46% |

Portugal is the extreme case: sliced from the chapter's archive its box yields
a single row of grid cells at 42 north, so the whole country was interpolated
from one latitude and extrapolated south of it. The other nine configurations
have no unit outside, and are unaffected by the change of archive in this
respect.

**The bounding boxes are not what clipped those rows.** Every shipped box
already covers its own fleet; all five reach beyond the data the chapter's
archive holds, Spain, Italy and Portugal to the south and Norway and Sweden to
the east. The boxes are therefore unchanged, and Denmark
keeps the study-scoped 15.4 east registered above. Under `era5/EU_2026-09`,
which spans 12 west to 31.5 east and 36 to 72 north, all fourteen
configurations have **zero units outside the loaded extent**, so
`allow_extrapolation` stays at its default and the extent guard is left to
refuse anything the audit missed.

**The archives differ in one other respect, and it reaches all fourteen rows.**
Where they overlap their wind components are bit-identical, verified over
Denmark for January 2020 across `u10`, `v10`, `u100` and `v100`. But the older
files carry a stored annual-mean roughness and the newer ones carry none, so
the newer route derives roughness per timestep. That is the treatment this
project adopted on 2026-09-12 (`method-roughness-treatment.md`), where its
measured accuracy effect was 0.0002 in corrected RMSE at Denmark and smaller at
France. It is nonetheless a second change, and S-G1 cannot be read as though
only the five rows moved.

### S-G1, restated

The gate asked whether S0 reproduces the chapter's published grid kriging MAE
for all fourteen configurations. Nine of the fourteen survive it intact, and
five are being asked a question this study has decided not to ask.

- **The nine unaffected rows still gate**, to within 0.001, and the study is
  still void if one of them misses. Their comparison now carries one change,
  the roughness treatment, and a fourth condition separates it: **S0 computed
  on the chapter's own archive with its stored roughness**, which those nine
  rows can run and the other five cannot. The difference between the two is the
  treatment, measured on nine configurations rather than assumed from two.
- **The five extrapolated rows are reported as a difference, not as a pass or a
  fail**, with the reason stated beside each: real winds against extrapolated
  ones. How far each moves is a result in its own right, and it is the same
  measurement the country-level EU re-run made (`method-eu-rerun.md`).
- The four figures declared below as already known were produced on the
  chapter's archive. They are unchanged as declarations, and the values this
  study reports for those rows come from the new archive.

Nothing else in the design changes: the conditions, the fourteen
configurations, the predictions and gates S-G2 and S-G3 stand as registered.

## The question

> Does interpolating a correction from control points of a different kind cause
> the failure, and does separating them fix it?

It has a mechanism, a documented failure and a candidate fix, which is what
makes it worth more than the question it replaces.

**The mechanism.** The chapter's surfaces are built from the whole pool at
once: `all_points = pd.concat([onshore, offshore], ignore_index=True)`. So an
offshore unit takes a correction dominated by whatever control points are
nearest, which for Denmark's offshore sites is 884 Danish onshore clusters.
Measured as the share of inverse-distance weight at each configuration's own
grid cells coming from its own control points, Denmark offshore is **0.045**:
95.5% of its correction is other configurations' answers. The other thirteen
configurations are 0.523 to 0.982.

**The failure.** Denmark offshore is the only configuration of the fourteen
that the chapter reports as failing, at a grid kriging MAE of 0.1113 against an
uncorrected 0.0822, and it is the only one that is not majority
self-determined. The in-sample finding and the failure finding are the same
thing seen from two sides (`method-grid-validation-in-sample.md`).

## What is already known, declared

**Four numbers exist before this registration and cannot be treated as blind.**
They were produced by the void study and by the diagnosis that voided it:

| Row | Unsplit, reproduces the chapter | Split, with area-of-interest masking |
|---|---|---|
| DK offshore | 0.1113 | **0.0496** |
| DK onshore | 0.0678 | 0.0782 |
| UK offshore | 0.1338 | 0.1421 |
| UK onshore | 0.0616 | 0.0623 |

So the split helps the failing row by 0.0617 and harms the other three by
0.0007 to 0.0104. **Any threshold chosen now for those four rows would be
chosen knowing the answer**, which is the error recorded under D5 of the
manuscript decisions document. The design below handles that in two ways: the
ten remaining configurations are genuinely blind, and the four known rows are
reported separately from them and never pooled into a single pass or fail.

## The conditions, and the confound they separate

The split as tested in the void study changed **two** things at once: the
control-point pool, and the masking. The chapter masks by distance, neutralising
any cell more than 5 degrees from a control point. The ported surface masks by
area of interest, assigning each cell to the onshore or offshore shapes and
neutralising cells in neither. So the four figures above cannot say which change
did the work.

| Condition | Pool | Mask |
|---|---|---|
| **S0** | undivided, all 1,729 | distance, 5 degrees |
| **S1** | split by declared `cluster_mode` | distance, 5 degrees |
| **S2** | split by declared `cluster_mode` | area of interest |

S0 reproduces the chapter and is the baseline. **S1 isolates the pool**, which
is the question. S2 is what the port does and is reported so that the port's own
behaviour is on the record, not because it answers anything on its own.

Held constant: the 1,729 control points, ordinary kriging with an exponential
variogram in geographic coordinates, the 0.25 degree grid, the chapter-era
fleets and observations, the study-scoped Danish bounding box of 15.4 east,
`input/era5/EU_2026-09` as the wind archive for every row, and the evaluation
at observation locations.

## What is scored

All **fourteen** configurations, not the four that differ in membership. A
configuration whose pool does not change between S0 and S1 should not move, and
checking that is how the comparison is verified rather than assumed.

Per configuration and condition: capacity-factor MAE, RMSE and bias at
observation locations; the neutral-fill share, reported before any metric; the
off-curve share; and the self-weight, which is the mechanism's own variable.

The uncorrected and cluster-based baselines are reported for every row, and
today's uncorrected is verified against the chapter's published figure first,
per the cross-pipeline rule below.

## A gate stated against another pipeline's number

Carried over unchanged from the void registration, because it still applies: a
gate stated against a figure from another pipeline is either verified against
today's equivalent or restated against it, never read across pipelines
silently. Today's uncorrected reproduced the chapter's published figures to
within 0.0005 in all four rows tested on the chapter's archive, so the gates
below may be read as registered unless a further row diverges. Those four
checks do not carry over to the new archive, and today's uncorrected is
compared with the chapter's published figure again for every row.

## Fixed in advance: what a split that helps one row and hurts three means

**A split is not adopted or rejected on an aggregate.** Averaging across
configurations would let one large improvement buy several small harms, or the
reverse, and neither is the question. The question is mechanistic, so the
outcome is read mechanistically:

**The mechanism is supported if the benefit tracks self-weight.** A
configuration whose correction is mostly other configurations' answers should
gain from separation; one that is mostly its own should be nearly unchanged,
because separation removes points that were contributing little. So the
prediction is a relationship, not a direction: **improvement under S1 should be
larger where self-weight is lower**, across all fourteen.

**The mechanism is refuted if harm appears where self-weight is high**, because
that is separation damaging a configuration whose own points already dominated,
which the mechanism gives no reason for. Denmark onshore at a self-weight of
0.982 being harmed by 0.0104 is exactly that shape, and it is the single most
important thing for S1 to explain: if it survives the mask being held constant,
the mechanism is incomplete.

**Adoption is a separate question from the mechanism and is not decided here.**
Whether a gridded product should split by domain depends on what it is for, and
this study can establish the mechanism without settling that. A finding of
"separation fixes the failure and costs a little elsewhere" is a result; it is
not an instruction.

## Gates

| Gate | Requirement | Outcome |
|---|---|---|
| **S-G1** | S0 reproduces the chapter's published grid kriging MAE to within 0.001 for the **nine configurations the chapter's archive covers**. **If it does not, the study is void** and the discrepancy is diagnosed first, because S1 is only interpretable against an S0 that reproduces. The other five are reported as a difference and gate nothing; see the 2026-09-15 amendment. | |
| **S-G2** | Under S1, Denmark offshore's MAE falls below its uncorrected 0.0822. That is the failure being fixed, stated in the chapter's own terms. | |
| **S-G3** | Under S1, no configuration whose self-weight exceeds 0.9 moves by more than 0.002 in MAE. That is the mechanism's own prediction for rows separation should barely touch, and the threshold is the screen this project already uses for a difference that is negligible in a capacity factor. | |

## Registered predictions

| # | Prediction | Outcome |
|---|---|---|
| S-P1 | S-G2 passes: Denmark offshore falls below uncorrected under S1. Already indicated by the S2 figure of 0.0496, so this is weakly held. | |
| S-P2 | **S-G3 fails at Denmark onshore.** Its self-weight is 0.982 and it was harmed by 0.0104 under S2, and I do not have a mechanism for that, so I expect the harm to survive the mask being held constant and the mechanism to come out incomplete. | |
| S-P3 | Across the ten blind configurations, improvement under S1 correlates negatively with self-weight: the less of its own answer a configuration gets, the more separation helps it. | |
| S-P4 | The two offshore configurations are the only ones that move by more than 0.005 under S1, since they are the only ones whose pool changes from 1,729 points to 12. | |

## Committed in advance

- All fourteen configurations are reported, including the four whose figures
  are already known, and the two groups are reported separately.
- S2 is reported but is not used to decide anything, being confounded.
- A configuration is not dropped after its result is seen.
- The study is void if S-G1 fails on any of the nine gated rows, and the void
  is reported rather than repaired by adjusting S0.
- The five extrapolated rows are reported whatever they show, and their
  distance from the chapter's figures is not read as a defect in either.
