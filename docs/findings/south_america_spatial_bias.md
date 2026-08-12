# The South American spatial bias: Chile and Argentina

**Date:** 2026-07-24
**Branch:** multi-region-validation
**Scope:** why the affine correction removes the mean bias but not the skill for
CL and AR, evaluated on 2024.

## Summary

Both South American regions show the same failure: the correction drives the
mean bias to near zero but does not improve RMSE or correlation. The cause is a
spatial bias in which **ERA5 exaggerates the wind gradient**. It makes the
northern deserts too calm and Patagonia too windy, while the real fleet achieves
a nearly uniform capacity factor across both. Two different mechanisms act in
opposite directions, so a single wind-speed correction per cluster cannot fix
both, and for Chile it fails even in-sample.

This is not a data-plumbing bug. It is a limit of coarse reanalysis plus an
idealised power curve in complex terrain, and it exposes an assumption in the
affine method: that the bias is affine in wind speed. In Patagonia part of the
"bias" is generation loss, which is not.

## The observation

Per plant, over the 2024 test year, uncorrected. Zones are latitude bands.

| region | zone | plants | ERA5 100 m wind | obs CF | sim CF | obs/sim |
|---|---|---|---|---|---|---|
| CL | north (Atacama) | 14 | 3.9 m/s | 0.208 | 0.118 | **1.76** |
| CL | central | 17 | 4.9 m/s | 0.210 | 0.213 | 0.99 |
| CL | south | 26 | 5.9 m/s | 0.255 | 0.298 | 0.86 |
| AR | north | 12 | 4.7 m/s | 0.353 | 0.223 | **1.58** |
| AR | N Patagonia | 23 | 7.3 m/s | 0.334 | 0.466 | 0.72 |
| AR | S Patagonia | 25 | 8.2 m/s | 0.349 | 0.566 | **0.62** |

Read down the `obs CF` column and then the `ERA5 wind` column. **Observed
capacity factor is nearly flat** (CL 0.21 to 0.26, AR 0.33 to 0.35) while **ERA5
wind is steep** (CL 3.9 to 5.9, AR 4.7 to 8.2). The simulation faithfully
converts ERA5's gradient into a capacity-factor gradient the real fleet does not
have.

Anchoring each observed CF to the steady wind speed that would produce it on the
market-average curve (a rough anchor, since a real fleet mean comes from a wind
distribution, but the relative picture is robust):

| zone | ERA5 wind | steady wind implied by obs CF |
|---|---|---|
| CL Atacama | 3.9 | 5.6 |
| CL central | 4.9 | 5.6 |
| CL south | 5.9 | 6.0 |
| AR north | 4.7 | 6.7 |
| AR N Patagonia | 7.3 | 6.5 |
| AR S Patagonia | 8.2 | 6.6 |

The fleet behaves as though it sits in a **roughly uniform 5.6 to 6.7 m/s wind
everywhere**. ERA5 disagrees by up to a factor of two on the gradient.

## The two mechanisms

They are different, which is the crux.

**North (deserts), ERA5 under-predicts.** In the Atacama ERA5 gives 3.9 m/s,
which the curve turns into 12% CF, but the real farms achieve 21%. You cannot
get 21% from a 3.9 m/s mean with any onshore turbine. ERA5 at 0.25 degrees
(about 28 km) cannot resolve the coastal and terrain-driven wind acceleration
that real turbines are deliberately sited in: ridgelines, coastal jets, thermal
channelling off the Andes. The grid-cell mean is far below the hub-height wind
at the chosen microsites. This is a genuine wind-speed bias, and it is the kind
the affine correction is designed for: a positive scalar fixes it, if it is
stationary.

**South (Patagonia), ERA5 over-predicts.** In Santa Cruz ERA5 gives 8.2 m/s,
which the curve turns into 57% CF, but the real farms achieve 35% (38% once
partial builds are excluded, below). Getting only 35% from an 8.2 m/s mean
implies about 20 percentage points of loss: wakes in large clustered farms,
availability, electrical losses, and above all curtailment, because Patagonia is
far from the Buenos Aires load and transmission-constrained. Some of the gap may
also be ERA5 over-stating the wind. Crucially, **losses are not a wind-speed
bias**: a farm curtailed 15% of the time has its output cut regardless of wind,
which no affine transform of wind speed represents.

Because the two zones fail in opposite directions and for different reasons, a
single scalar and offset per cluster cannot correct both, and the aggregate mean
bias hides it: the north's under-prediction and the south's over-prediction
partly cancel.

## The correction removes the mean, not the structure

Scored on a training year (2023), so this is in-sample and cannot be blamed on
generalisation:

| region | variant | MBE | RMSE | r |
|---|---|---|---|---|
| CL | uncorrected | -0.029 | 0.1208 | 0.408 |
| CL | affine, 10, fixed | -0.005 | **0.1209** | 0.376 |
| AR | uncorrected | 0.138 | 0.2469 | 0.259 |
| AR | affine, 10, fixed | 0.017 | 0.1703 | 0.289 |

**Chile is the sharp case: the correction does not reduce RMSE even on its own
training data** (0.1208 to 0.1209), and correlation falls. The affine form
removes the mean and can do nothing with the residual, because the residual is
not affine in wind. This is structural, not overfitting.

Argentina's correction does help in-sample and out (RMSE about 0.25 to 0.17 both
ways, MBE to near zero), but absolute skill stays low (r about 0.28) because ten
clusters cannot give sixty plants their own level, and the residual per-plant
scatter dominates.

For contrast, per-plant temporal correlation is high in both (median r 0.86):
the reanalysis tracks each plant's month-to-month variation well. The failure is
entirely in the cross-sectional level, which is exactly what a spatial bias is.

## Partial builds contaminate the southern observations

A real operating wind farm sustains an annual CF well above 0.12. Plants below
that are partial builds or mostly-off farms whose full nameplate is in the
capacity denominator (the AR config's stated, unscreened caveat; capacity is
GWPT-joined because CAMMESA carries none).

| region | plants with obs CF < 0.12 | their capacity | of which in the south |
|---|---|---|---|
| CL | 2 of 57 | 441 MW (8%) | 1 plant, 11 MW |
| AR | 8 of 60 | 682 MW (14%) | 6 plants, 482 MW |

Examples in AR: Villalonga II (obs CF 0.031), Pomona II (0.051), La Castellana
II (0.052), all phase-2 plants. Excluding them raises AR's southern observed CF
from 0.342 to 0.381, which shrinks the Patagonian over-prediction but does not
remove it (sim still 0.52). So partial builds explain roughly a third of the AR
southern gap and little of Chile's; the rest is the genuine simulation bias
above.

## What this means for the method

- The affine-in-wind assumption holds in the north (a resolution-driven
  wind-speed bias) and breaks in the south (losses that are not a wind-speed
  bias). South America is the first region in the set where a large part of the
  "bias" is generation loss rather than reanalysis wind error.
- Reporting CL or AR as a headline corrected region would be unsupported. The
  mean bias is removed but the model has no more skill than uncorrected, and for
  CL that is true in-sample.
- The uniform 100 m hub height and single market-average curve for every plant
  (both `default-uniform` in the metadata) are not the cause of the gradient:
  they are constant across zones. But they blur the diagnosis, so both regions
  were rebuilt with real turbine curves and hub heights (next section).

## Matched curves: the diagnosis confirmed (step 2, done)

Every CL and AR plant was given a real power curve and hub height. Per-farm
turbine specs (manufacturer, model, count, rotor diameter, hub height) were
compiled from public sources (thewindpower.net, Wikipedia, operator and
manufacturer records) into `configs/curation/cl_turbine_specs.csv` and
`ar_turbine_specs.csv`, then matched to the open curve library by scale and
specific power through the same code path the US and NZ fleets use
(`scripts/region_tools/apply_turbine_specs.py`,
`vwf.datasets.eia_us.assign_curves_from_library`). CL: 57 of 60 plants matched
(96% of capacity), 11 distinct curves, real hub heights 80 to 148 m for 23
plants. AR: 65 of 65 matched, 12 curves, hub heights 45 to 130 m.

**The northern under-prediction got worse, not better, which settles it.**
Chile's Atacama plants carry higher specific power than the default (304 vs 226
W/m2: modern high-wind machines), so their correct curves produce *less* at low
wind. With matched curves the northern simulated CF fell from 0.118 to 0.099
against an observed 0.208. No correctly specified turbine makes 21% capacity
factor from ERA5's 3.9 m/s. **The northern bias is the reanalysis wind, full
stop**, and correct turbine metadata rules out the "wrong turbine" explanation
rather than supporting it.

The southern over-prediction eased slightly with correct curves (CL south
0.298 to 0.310, essentially unchanged; AR S Patagonia 0.566 to 0.516), so it is
still mostly losses, not curve error.

What matched curves did buy, and it is the practically important part:

| region | uncorrected RMSE | corrected RMSE (fixed) | corrected r |
|---|---|---|---|
| CL uniform | 0.121 | 0.130 (correction *hurts*) | 0.32 |
| **CL matched** | 0.123 | **0.104** (correction helps) | **0.45** |
| AR uniform | 0.238 | 0.172 | 0.265 |
| AR matched | 0.225 | 0.213 | 0.108 |
| AU-NEM uniform | 0.152 | 0.087 | 0.46 |
| AU-NEM matched | 0.115 | 0.109 | 0.47 |

Three different outcomes, each informative:

- **CL: matched curves make the correction work.** With the uniform curve the
  affine correction made Chile *worse* (0.121 to 0.130): it was fighting a curve
  error mixed into the wind bias. With correct curves the curve error is gone,
  the correction only fights the wind bias, and it helps (0.123 to 0.104,
  r 0.32 to 0.45). This is the concrete case of the general rule that a uniform
  curve lets the "bias correction" launder a curve error, which then does not
  generalise.
- **AU-NEM: matched curves fix the raw simulation.** Uncorrected RMSE 0.152 to
  0.115 and mean bias 0.096 to 0.009: the uniform curve had been systematically
  over-predicting, and the affine correction had been removing that as if it
  were reanalysis bias. Corrected RMSE is marginally worse (0.087 to 0.109)
  precisely because the correction no longer has a large curve-driven bias to
  remove. The matched result is the honest one; the uniform's better corrected
  score was the correction laundering a curve error.
- **AR: blocked by observations, not curves.** Matched curves improve AR's
  uncorrected simulation (0.238 to 0.225, r 0.22 to 0.27) and ease the southern
  over-prediction, but the corrected metric collapses (r 0.27 to 0.11). The
  cause is the capacity-placeholder problem: AR's capacity is GWPT-joined and
  duplicated across phases, so ~8 plants have physically impossible observed
  capacity factors (0.03 to 0.05). The per-cluster correction trains on those
  corrupted observations and learns garbage. **AR cannot be corrected until its
  capacity denominators are fixed**, which is a separate data task; matched
  curves are necessary but not sufficient.

Matched curves are now the default for all three regions (uniform metadata kept
as `<c>_md.uniform.bak.csv`). No region should use the uniform curve for a
delivered result: it is physically wrong per farm and it lets the correction
absorb curve error.

## AR capacity denominators fixed (done)

AR's blocker was the observation side: CAMMESA carries no capacity, so the CF
denominator was joined from GWPT, and the join took the full-farm capacity for
every phase code that matched a multi-phase farm (`nlargest(1, "cap")` in
`scripts/process/cammesa_ar.py`). A 3 MW phase-II code inherited its 52 MW
parent's nameplate, so its capacity factor read 0.03 when the real value was
0.50.

Fixed using the turbine research: real nameplate = turbine count x unit MW, from
`configs/curation/ar_turbine_specs.csv`, written as capacity overrides
(`ar_coord_overrides.csv`, 20 rows) where it diverged from GWPT by more than
15%. Six unrepresentative codes were hard-dropped: three with zero generation
across 2021-2024 (duplicate MEM codes whose output is reported elsewhere:
ARAUCO EOLICO, ARAUCO EOLICO 2, NECOCHEA EOLICO) and three with a steady CF near
0.07 (the state-owned Arauco farm's old IMPSA turbines in weak La Rioja wind,
and a La Castellana II anomaly). `EXCLUDE` now hard-drops from the fleet rather
than merely excusing an unmatched plant.

The observed capacity factors are now all physically plausible: 59 plants,
median 0.474, none below 0.12 or above 0.65 (was ~8 plants at 0.03 to 0.07).

The effect on the model, 2024, fixed slice, matched curves throughout:

| AR state | uncorrected RMSE / r | affine RMSE / r |
|---|---|---|
| original (uniform curve, GWPT caps) | 0.238 / 0.22 | 0.172 / 0.27 |
| matched curves, GWPT caps | 0.225 / 0.27 | 0.213 / 0.11 |
| **matched curves + fixed capacities** | **0.151 / 0.41** | **0.133 / 0.44** |

The middle row is the trap: matched curves made the affine correction look
*worse* because they exposed the capacity errors the uniform curve had masked.
With the capacities fixed, the affine correction helps (0.151 to 0.133) instead
of hurting, and the scaled-affine availability crutch is no longer needed. AR is
now a usable region.

One northern cluster (La Rioja / Cuyo) still fits an extreme wind scalar (about
15), the same ERA5 desert-wind under-resolution as Chile's Atacama, but on the
cleaned fleet it no longer dominates the fleet metric. That residual is the
genuine reanalysis-wind limit, shared by both South American regions, and its
fix is a higher-resolution wind product, not more data cleaning.

## Curtailment screen (step 1): no accessible series

Investigated with the live CEN SIP API (user-supplied key). No wind-curtailment
("vertimiento") endpoint exists under the SIP host: every candidate path
(`/vertimiento`, `/energia-vertida`, `/desacople`, `/reduccion-generacion`, and
the accented and versioned variants) returns 404, and the OpenAPI spec is not
served at any standard path. The one lead in the delivered-generation data,
`generacion-real`'s `factor_ernc` column, is flat at 1.0 for every wind row
across 2021, and `valor_ernc` simply equals `gen_real_mw`, so neither carries a
curtailment signal.

So the screen cannot be built from the SIP service. A vertimiento series, if
CEN exposes one, would be under the Operación or Mercados service (different
credentials and an endpoint map that is not published). This stays open, but it
is lower priority than it looked: the loss-term work already showed the southern
over-prediction is captured by the affine offset, not a missing curtailment
term, and the AR investigation showed its apparent "losses" were capacity
errors. CAMMESA (Argentina) has no known curtailment series either.
2. **AR capacity denominators.** Fix the GWPT-joined, phase-duplicated
   capacities and screen partial builds before AR is corrected at all. This is
   AR's real blocker, ahead of any wind question.
3. **A higher-resolution wind product** (ERA5-Land at 0.1 degrees, or NEWA-style
   downscaling) at the northern sites. Now the leading candidate for the
   northern bias, since matched curves ruled out the turbine explanation. If the
   finer wind removes the northern under-prediction, the 0.25 degree grid was
   the cause.
4. **A loss term** in the correction (step 4, done): an availability-and-
   curtailment factor that scales output independently of the wind scalar and
   offset. See the next section.

## Loss-term correction (step 4, done): the hypothesis did not hold

A `scaled-affine` correction model was added (`vwf.harness.corrections`): the
affine wind correction, plus one per-cluster availability factor `a in (0, 1]`
that multiplies the corrected capacity factor, fitted from the level the affine
part leaves. The motivation was that a curtailed or low-availability farm loses
output regardless of wind, which no wind-speed transform represents, so
Patagonia's over-prediction should need such a factor.

**It did not.** Fitted on the matched-curve fleets:

| region | affine RMSE | scaled-affine RMSE | affine r | scaled-affine r |
|---|---|---|---|---|
| CL | 0.1048 | 0.1044 | 0.433 | 0.435 |
| AR | 0.2126 | 0.1620 | 0.108 | 0.412 |

CL is unchanged: every cluster's availability came out at 1.0, so the loss term
is inert. That is correct, because Chile's problem is wind, not loss.

AR looks transformed, but the availability factors say otherwise. Nine of ten
AR clusters have `a` at 1.0, including all three Patagonian clusters. **The
Patagonian over-prediction is already handled by the affine offset** (which
shifts the level per cluster), so the clean curtailment-loss hypothesis is not
supported. The entire AR gain comes from one cluster, Arauco / La Rioja in the
north, where the affine fit had exploded to a wind scalar of 7.9 and an offset
of -5.8, and the availability factor of 0.27 pulls the resulting over-prediction
back down. Excluding that one cluster, affine and scaled-affine give an
identical pooled correlation of 0.356; the headline 0.108 to 0.412 jump is
entirely that cluster being rescued.

So the availability term is not measuring availability. It is a regulariser that
caps a runaway affine fit on a data-corrupted cluster. Arauco is exactly such a
cluster: six MEM codes sharing a site with duplicated placeholder capacities and
mixed-vintage turbines (the AR turbine research flagged it as the least reliable
in the fleet). The right fix there is the data, not a model term.

The useful product of this is diagnostic, not corrective: **a cluster that needs
`a` well below 1 to fit is a cluster with a data problem worth investigating.**
The model stays available (`correction.model = "scaled-affine"`) as that
diagnostic and as a robustness cap, but it is not wired in as a default, because
scaling a farm's output to hide a bad capacity denominator is the opposite of
what a per-farm report should do. Confirming genuine curtailment still needs the
CEN vertimiento series (step 1, blocked).

Net recommendation for the two regions:

- **CL** with matched curves and the plain affine correction: RMSE 0.104,
  r 0.45. Usable, with the standing caveat that the northern desert is
  under-resolved by ERA5 (a higher-resolution wind product is the next lever).
- **AR** is not production-ready. Matched curves plus affine give RMSE 0.213 /
  r 0.11; the scaled-affine 0.162 / 0.41 is masking the Arauco data problem
  rather than solving it. AR needs its capacity denominators rebuilt and its
  partial builds and duplicate MEM codes resolved before any correction is
  trustworthy.

## Caveats

- The steady-wind anchor understates true mean wind (a distributed wind gives a
  different CF than steady wind on the convex rising limb), so its absolute
  numbers are indicative; the relative flatness of observed effective wind
  against ERA5's gradient is the robust part.
- One test year each. The in-sample check uses one training year.
- Zone boundaries are latitude cuts, not climate polygons; a few central plants
  straddle them.
