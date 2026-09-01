# The South American spatial bias: Chile and Argentina

**Date:** 2026-07-24
**Scope:** why the affine correction removes the mean bias but not the skill for
CL and AR, evaluated on 2024.

Both regions fail the same way: the correction drives mean bias to near zero
without improving RMSE or correlation. The cause is a spatial bias in which
**ERA5 exaggerates the wind gradient**, making the northern deserts too calm and
Patagonia too windy while the real fleet achieves a nearly uniform capacity
factor across both. Two mechanisms act in opposite directions, so a single
wind-speed correction per cluster cannot fix both, and for Chile it fails even
in sample.

This is not a plumbing bug. It is a limit of coarse reanalysis plus an idealised
power curve in complex terrain, and it exposes an assumption in the affine
method: that the bias is affine in wind speed. In Patagonia part of the "bias"
is generation loss, which is not.

## The observation

Per plant over the 2024 test year, uncorrected. Zones are latitude bands.

| region | zone | plants | ERA5 100 m wind | obs CF | sim CF | obs/sim |
|---|---|---|---|---|---|---|
| CL | north (Atacama) | 14 | 3.9 m/s | 0.208 | 0.118 | **1.76** |
| CL | central | 17 | 4.9 m/s | 0.210 | 0.213 | 0.99 |
| CL | south | 26 | 5.9 m/s | 0.255 | 0.298 | 0.86 |
| AR | north | 12 | 4.7 m/s | 0.353 | 0.223 | **1.58** |
| AR | N Patagonia | 23 | 7.3 m/s | 0.334 | 0.466 | 0.72 |
| AR | S Patagonia | 25 | 8.2 m/s | 0.349 | 0.566 | **0.62** |

**Observed capacity factor is nearly flat** (CL 0.21 to 0.26, AR 0.33 to 0.35)
while **ERA5 wind is steep** (CL 3.9 to 5.9, AR 4.7 to 8.2). The simulation
faithfully converts ERA5's gradient into a capacity-factor gradient the real
fleet does not have.

Anchoring each observed CF to the steady wind speed that would produce it on the
market-average curve gives CL Atacama 5.6 against ERA5's 3.9, CL central 5.6
against 4.9, CL south 6.0 against 5.9, AR north 6.7 against 4.7, AR N Patagonia
6.5 against 7.3, and AR S Patagonia 6.6 against 8.2. The fleet behaves as though
it sits in a **roughly uniform 5.6 to 6.7 m/s wind everywhere**, and ERA5
disagrees by up to a factor of two on the gradient. The anchor is rough, since a
real fleet mean comes from a wind distribution rather than a steady wind, so its
absolute numbers are indicative and the relative flatness is the robust part.

## Two mechanisms, opposite directions

**North: ERA5 under-predicts.** In the Atacama ERA5 gives 3.9 m/s, which the
curve turns into 12% CF against a real 21%. No onshore turbine reaches 21% from
a 3.9 m/s mean. ERA5 at 0.25 degrees, about 28 km, cannot resolve the coastal
and terrain-driven acceleration real turbines are deliberately sited in:
ridgelines, coastal jets, thermal channelling off the Andes. This is a genuine
wind-speed bias and exactly what the affine correction is designed for.

**South: ERA5 over-predicts.** In Santa Cruz ERA5 gives 8.2 m/s, which the curve
turns into 57% CF against a real 35%, or 38% once partial builds are excluded.
That implies roughly 20 percentage points of loss: wakes in large clustered
farms, availability, electrical losses, and above all curtailment, Patagonia
being far from the Buenos Aires load and transmission-constrained. **Losses are
not a wind-speed bias**: a farm curtailed 15% of the time loses output
regardless of wind, which no affine transform of wind speed represents.

The two failing in opposite directions is what hides them: the north's
under-prediction and the south's over-prediction partly cancel in the aggregate
mean bias.

## The correction removes the mean, not the structure

Scored on a training year (2023), so in sample and not attributable to
generalisation:

| region | variant | MBE | RMSE | r |
|---|---|---|---|---|
| CL | uncorrected | -0.029 | 0.1208 | 0.408 |
| CL | affine, 10, fixed | -0.005 | **0.1209** | 0.376 |
| AR | uncorrected | 0.138 | 0.2469 | 0.259 |
| AR | affine, 10, fixed | 0.017 | 0.1703 | 0.289 |

**Chile is the sharp case: the correction does not reduce RMSE even on its own
training data**, and correlation falls. The affine form removes the mean and can
do nothing with the residual, because the residual is not affine in wind. That is
structural, not overfitting.

Argentina's correction helps in sample and out, but absolute skill stays low
(r about 0.28) because ten clusters cannot give sixty plants their own level.

For contrast, per-plant *temporal* correlation is high in both (median r 0.86):
the reanalysis tracks each plant's month-to-month variation well. The failure is
entirely cross-sectional, which is exactly what a spatial bias is.

## Partial builds contaminate the southern observations

A real operating farm sustains an annual CF well above 0.12. Plants below that
are partial builds or mostly-off farms whose full nameplate sits in the capacity
denominator.

| region | plants with obs CF < 0.12 | their capacity | of which in the south |
|---|---|---|---|
| CL | 2 of 57 | 441 MW (8%) | 1 plant, 11 MW |
| AR | 8 of 60 | 682 MW (14%) | 6 plants, 482 MW |

Excluding them raises AR's southern observed CF from 0.342 to 0.381, which
shrinks the Patagonian over-prediction without removing it (sim still 0.52). So
partial builds explain roughly a third of the AR southern gap and little of
Chile's.

## Matched curves confirm the diagnosis

Every CL and AR plant was given a real power curve and hub height. Per-farm
turbine specs were compiled from public sources into
`configs/curation/cl_turbine_specs.csv` and `ar_turbine_specs.csv`, then matched
to the open curve library by scale and specific power through the same path the
US and NZ fleets use. CL: 57 of 60 plants matched, 96% of capacity, 11 distinct
curves, real hub heights 80 to 148 m for 23 plants. AR: 65 of 65, 12 curves,
hub heights 45 to 130 m.

**The northern under-prediction got worse, not better, which settles it.**
Chile's Atacama plants carry higher specific power than the default (304 against
226 W/m2, modern high-wind machines), so their correct curves produce *less* at
low wind: northern simulated CF fell from 0.118 to 0.099 against an observed
0.208. **The northern bias is the reanalysis wind, full stop**, and correct
turbine metadata rules out the "wrong turbine" explanation rather than
supporting it. The southern over-prediction eased only slightly (CL south 0.298
to 0.310, AR S Patagonia 0.566 to 0.516), so it is still mostly losses.

What matched curves bought, across three regions:

| region | uncorrected RMSE | corrected RMSE (fixed) | corrected r |
|---|---|---|---|
| CL uniform | 0.121 | 0.130 (correction *hurts*) | 0.32 |
| **CL matched** | 0.123 | **0.104** | **0.45** |
| AR uniform | 0.238 | 0.172 | 0.265 |
| AR matched | 0.225 | 0.213 | 0.108 |
| AU-NEM uniform | 0.152 | 0.087 | 0.46 |
| AU-NEM matched | 0.115 | 0.109 | 0.47 |

Three different outcomes, each informative:

- **CL: matched curves make the correction work.** With the uniform curve the
  correction made Chile *worse*, because it was fighting a curve error mixed into
  the wind bias. With correct curves it only fights the wind bias, and it helps.
- **AU-NEM: matched curves fix the raw simulation.** Uncorrected RMSE 0.152 to
  0.115 and mean bias 0.096 to 0.009: the uniform curve had been systematically
  over-predicting and the correction had been removing that as if it were
  reanalysis bias. Corrected RMSE is marginally worse precisely because the
  correction no longer has a large curve-driven bias to remove. **The matched
  result is the honest one**; the uniform's better corrected score was the
  correction laundering a curve error.
- **AR: made worse, and the reason is the observations.** Matched curves improve
  AR's uncorrected simulation but the corrected metric collapses, because AR's
  capacity was GWPT-joined and duplicated across phases, so about eight plants
  carried physically impossible capacity factors that the correction then trained
  on. Matched curves exposed what the uniform curve had masked.

Matched curves are the default for all three regions. No region should use the
uniform curve for a delivered result: it is physically wrong per farm and it
lets the correction absorb curve error.

## AR capacity denominators, fixed

CAMMESA carries no capacity, so the CF denominator was joined from GWPT and the
join took the full-farm capacity for every phase code matching a multi-phase
farm. A 3 MW phase-II code inherited its 52 MW parent's nameplate, so its
capacity factor read 0.03 where the real value was 0.50.

Fixed from the turbine research: real nameplate as turbine count times unit MW,
written as capacity overrides (20 rows) where it diverged from GWPT by more than
15%. Six unrepresentative codes were hard-dropped: three with zero generation
across 2021-2024 (duplicate MEM codes reported elsewhere) and three with a
steady CF near 0.07 (the state-owned Arauco farm's old IMPSA turbines in weak
La Rioja wind, and a La Castellana II anomaly). Observed capacity factors are now
all plausible: 59 plants, median 0.474, none below 0.12 or above 0.65.

| AR state | uncorrected RMSE / r | affine RMSE / r |
|---|---|---|
| uniform curve, GWPT caps | 0.238 / 0.22 | 0.172 / 0.27 |
| matched curves, GWPT caps | 0.225 / 0.27 | 0.213 / 0.11 |
| **matched curves + fixed capacities** | **0.151 / 0.41** | **0.133 / 0.44** |

The middle row is the trap: matched curves made the correction look *worse*
because they exposed capacity errors the uniform curve had masked. With
capacities fixed the correction helps rather than hurts.

One northern cluster (La Rioja / Cuyo) still fits an extreme wind scalar near
15, the same ERA5 desert under-resolution as Chile's Atacama, but on the cleaned
fleet it no longer dominates the fleet metric. `scorecard.md` still flags AR's
k=10 row as a degenerate fit on that basis.

## A loss term: the hypothesis did not hold

A `scaled-affine` model adds one per-cluster availability factor `a` in (0, 1]
multiplying the corrected capacity factor, on the reasoning that a curtailed or
low-availability farm loses output regardless of wind. Fitted on the
matched-curve fleets, **before** the AR capacity fix above:

| region | affine RMSE | scaled-affine RMSE | affine r | scaled-affine r |
|---|---|---|---|---|
| CL | 0.1048 | 0.1044 | 0.433 | 0.435 |
| AR | 0.2126 | 0.1620 | 0.108 | 0.412 |

CL is unchanged: every cluster's availability came out at 1.0, so the term is
inert. That is correct, because Chile's problem is wind, not loss.

AR looks transformed, and the availability factors say why it should not be
believed. Nine of ten AR clusters have `a` at 1.0, **including all three
Patagonian clusters**, so the Patagonian over-prediction is already handled by
the affine offset and the clean curtailment-loss hypothesis is unsupported. The
entire gain comes from one cluster, Arauco / La Rioja, where the affine fit had
exploded to a wind scalar of 7.9 and an offset of -5.8 and the availability
factor of 0.27 pulls the over-prediction back. Excluding that cluster, affine and
scaled-affine give an identical pooled correlation of 0.356.

So the availability term was not measuring availability but capping a runaway fit
on a data-corrupted cluster, and **the capacity fix confirms it**: repairing
Arauco's denominators takes plain affine from 0.213 to 0.133 and r from 0.11 to
0.44, without any loss term. The right fix was the data.

The useful product is diagnostic: **a cluster that needs `a` well below 1 to fit
is a cluster with a data problem worth investigating.** The model stays available
as that diagnostic and as a robustness cap, but it is not a default, because
scaling a farm's output to hide a bad capacity denominator is the opposite of
what a per-farm report should do.

## Curtailment screen: no accessible series

Investigated with the live CEN SIP API. No wind-curtailment ("vertimiento")
endpoint exists under the SIP host: every candidate path returns 404 and the
OpenAPI spec is not served at any standard path. The one lead in the
delivered-generation data, `generacion-real`'s `factor_ernc` column, is flat at
1.0 for every wind row across 2021, and `valor_ernc` simply equals
`gen_real_mw`, so neither carries a curtailment signal. A series, if CEN exposes
one, would be under the Operación or Mercados service, with different
credentials and an unpublished endpoint map. CAMMESA has no known curtailment
series either.

This is lower priority than it looked: the loss-term work showed the southern
over-prediction is captured by the affine offset rather than a missing
curtailment term, and the AR investigation showed its apparent losses were
capacity errors.

## Where the two regions stand

- **CL**, matched curves and plain affine: RMSE 0.104, r 0.45. Usable, with the
  standing caveat that the northern desert is under-resolved by ERA5.
- **AR**, matched curves and fixed capacities: RMSE 0.151 to 0.133, r 0.44.
  Usable, having been unusable before the denominators were rebuilt. Its k=10
  row remains daggered in `scorecard.md` for the northern cluster's extreme
  scalar.

Neither is a headline win. The correction removes the mean bias and adds limited
skill, and the residual northern under-prediction is a reanalysis-resolution
artefact shared by both.

**The leading next lever is a higher-resolution wind product** (ERA5-Land at 0.1
degrees, or NEWA-style downscaling) at the northern sites. Matched curves ruled
out the turbine explanation, so if a finer wind removes the northern
under-prediction, the 0.25 degree grid was the cause.

## Caveats

- The steady-wind anchor understates true mean wind, since a distributed wind
  gives a different CF than a steady wind on the convex rising limb. Its
  absolute numbers are indicative.
- One test year each; the in-sample check uses one training year.
- Zone boundaries are latitude cuts, not climate polygons, and a few central
  plants straddle them.
