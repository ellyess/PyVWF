# A physics-informed correction for ERA5 wind

The affine correction `w' = a*w + b` is fitted per cluster per time slice
against observed generation. It works where it is fitted and does not transfer
(`method-ml-transfer.md`). This documents an alternative for the case where a
region has no observed generation to fit against: keep the physics that is
known, learn only the parts that are not, and require every learned quantity to
mean the same thing in every region.

Gates were fixed before any model was fitted; the record of what was registered
and how each prediction turned out is in
`method-physics-informed-prespecification.md`. Code: `src/vwf/pinn/`, drivers in
`scripts/pinn/`, 51 tests in `tests/test_pinn_physics.py`.

## 1. Why the incumbent does not transfer

Five diagnostics on artefacts already on disk. Two falsified the hypothesis they
were written to test, and both are reported.

**The fitted pair is rank-1.** `pearson(a, b)` is -0.999 in Germany and below
-0.99 in three more regions. Regressing offset on scalar recovers a pivot at
3.8 to 4.2 m/s, essentially the turbine cut-in speed: below cut-in no power is
made, so the objective cannot constrain the correction there and the fitted
family degenerates to a rotation about that point. Constraining the offset to
that ridge changes transfer skill by less than 0.03 in every region, so the
collinearity is not the defect.

**Part of the target is estimation noise.** Empirical variograms of the
per-cluster scalar: UK clusters 20 to 30 km apart differ by 59% of the region's
total scalar variance, against 6% in Brazil and 15% in Germany. At k=100 over
348 turbines the UK clusters hold about 3.5 turbines each.

**Feature scale was mis-specified.** Pooled importances put relief at 28 km at
0.19 and terrain spread at 84 km at 0.17, against 0.02 for the kilometre-scale
derivatives the earlier experiment used. Correcting the scale is not sufficient
on its own: a multi-scale set improves 3 of 5 regions and collapses the UK.

**Elevation excess fails; cell relief holds.** The registered prediction was
that the scalar rises with a site's elevation above its ERA5 cell mean. It does
not: pearson +0.66 (BR), -0.29 (DE, p=0.004), +0.13 (DK), +0.01 (UK), -0.06
(US). Region-inconsistent, wrong sign in Germany. The elevation *range* within
the cell does hold, at +0.76, +0.70, -0.08, +0.51, +0.44, pooled +0.55. Denmark
is null and Denmark is flat (cell relief 78 +/- 29 m against 320 to 362 m
elsewhere), so it has almost no relief variance to correlate with. The mechanism
is not that the turbine is high: the largest US scalar (4.25, central
Washington) sits 142 m *below* its cell mean inside a cell spanning 1,083 m of
relief. Developers site into the windy features within a cell, and relief
measures how much such structure exists.

**The published transfer metric answers an unanswerable question.**
`method-ml-transfer.md` scores transfer as R-squared against the holdout
region's own mean, which a practitioner arriving in an unseen region does not
have. Against the identity correction, the same targets, regressor and seeds
give 3 of 5 positive rather than 1 of 5. This does not overturn the published
finding, which is correct on its own terms; it means the headline is
metric-dependent and both numbers belong in any restatement of it.

Together these indict the **two-stage design** rather than the choice of
regressor: stage one injects estimation noise, depends on an arbitrary k-means
partition, produces a rank-1 pair whose second component is unidentified below
cut-in, and averages every turbine-month into one of ~100 cluster summaries.

## 2. The model

For unit i on day t, with h the hub height and w the ERA5 100 m wind:

    ln u_hub  =  ln w_100  +  (s + delta) * ln(h / 100)  +  gamma
    u_curve   =  u_hub * (rho / rho_0)^(1/3)
    cf_day    =  eta * E[ P_c( u_curve + sigma * Z ) ],   Z ~ N(0, 1)
    CF_month  =  mean over the days of the month

The reanalysis 100 m wind; a power-law profile to hub height whose exponent `s`
is measured hourly from the 10 m and 100 m fields and corrected by `delta`; a
terrain speed-up `gamma`; the IEC 61400-12 equivalent-speed correction under the
ISA density ratio `(1 - 0.0065z/288.15)^4.2559`, which has no free parameters;
the power curve integrated over the within-day wind distribution by five-node
Gauss-Hermite quadrature with `sigma = kappa * sd_day`; and the monthly mean the
observations report. The whole operator is differentiable, so observed
generation supervises the learned quantities directly and no intermediate target
is estimated.

Four quantities are learned, all bounded by a physical statement:

| symbol | meaning | bounds | function of |
|---|---|---|---|
| `gamma` | log terrain speed-up | 0.67x to 2.46x | terrain |
| `delta` | offset to the measured 10-100 m shear exponent | -0.15 to +0.25 | terrain |
| `eta` | conversion efficiency: wake, availability, electrical, curtailment | 0.55 to 1.0 | fleet |
| `kappa` | within-day spread the pre-smoothed curves have not absorbed | 0 to 1.5 | one global number |

**The speed-up is pinned to zero on flat ground, structurally.** `gamma` is an
amplitude times a saturating function of the ERA5-cell relief,
`gamma = A(x) * r/(1+r)` with `r = relief/L`, so a site with no sub-grid relief
receives exactly no terrain correction however `A` is fitted. A hard guarantee,
not a penalty, and it is tested. It matters because most training data is flat
and most of the world is flat: without it the term drifts there and takes the
extrapolation with it.

**No feature can name the region.** Terrain descriptors are elevation, cell land
fraction, and topographic position, spread and relief at 1, 5, 28 and 84 km:
micro-siting, the individual ridge, the ERA5 cell, the orographic blocking
scale. No longitude, latitude or country. Fleet descriptors are capacity density
in MW/km2 within 10 and 50 km, an offshore flag and hub height. Capacity density
is invariant to whether a row is a turbine, a farm, a plant or a complex, unlike
raw capacity, which separates Europe from the Americas perfectly (2.9 to 3.4
against 4.9 to 5.1 in log10 kW) and was removed for that reason.

**`kappa` exists because the curves are already smoothed.** The VWF method
convolves manufacturer curves with a Gaussian of `sigma = 0.6 + 0.2w` at hourly
resolution. How much of the within-day spread that already covers is a question
for the data. Retaining the daily standard deviation matters because the curve
is strongly non-linear across the range a day's wind spans: in Denmark the
within-day standard deviation is 21% of the daily mean.

With linear heads the model has **37 active parameters**, against roughly 800
free per-cluster factors in a k=100 four-season affine fit. What does the work is
the operator they sit inside, not the heads.

Fitting: full-batch Adam, lr 0.05, weight decay 1e-3, 60 epochs, capacity-weighted
MSE on monthly capacity factor, training regions weighted equally, standardisation
fitted on training regions only, five seeds.

## 3. Results

Leave-one-region-out: fit on the others, apply zero-shot to the holdout, score
capacity-factor RMSE on its held-out test year.

| holdout | uncorrected | RF transfer | **zero-shot** | ablation | in region | affine, in region |
|---|---|---|---|---|---|---|
| DK | 0.1464 | 0.1011 | **0.0989** | 0.2298 | 0.0823 | 0.0855 |
| DE | 0.0860 | 0.0702 | **0.0790** | 0.1692 | 0.0597 | 0.0572 |
| UK | 0.1451 | 0.1504 | **0.1429** | 0.3252 | 0.1267 | 0.1146 |
| US | 0.1098 | 0.1691 | **0.1072** | 0.3522 | 0.0909 | 0.0978 |
| BR | 0.1386 | 0.1316 | **0.1062** | 0.2624 | 0.0954 | 0.1054 |

Skill against uncorrected ERA5 (`1 - MSE/MSE_uncorrected`):

| holdout | RF transfer | **zero-shot** | ablation | in region |
|---|---|---|---|---|
| DK | +0.523 | **+0.544** | -1.464 | +0.684 |
| DE | +0.334 | **+0.156** | -2.876 | +0.518 |
| UK | -0.075 | **+0.031** | -4.023 | +0.238 |
| US | -1.373 | **+0.046** | -9.288 | +0.315 |
| BR | +0.098 | **+0.413** | -2.584 | +0.527 |

Seed spread is 0.0002 to 0.0011 constrained, 0.0297 to 0.1059 for the ablation.

- **P1 PASS.** Beats uncorrected ERA5 in 5 of 5 and degrades none.
- **P2 PASS.** Beats the incumbent RF transfer in 4 of 5. The exception is
  Germany, whose turbine locations are postcode centroids, 579 distinct points
  behind 4,288 rows, so a terrain-driven method reads settlement centres while a
  method leaning on latitude and longitude does not.
- **P3 PASS.** Removing the constraints at matched features and parameter count
  costs 0.09 to 0.24 RMSE in every region and destabilises the fit. The gain is
  the constraints, not the features.

**The incumbent transfer does harm; this does not.** Mean skill -0.099 against
+0.238. In the United States the incumbent degrades RMSE by 54%, from a
transferred mean scalar of 1.41 +/- 0.54 applied to a fleet whose bias does not
warrant it. For a method offered as general, never making things worse matters
more than the size of the average gain.

**Against in-region fits**, the affine reference is optimistic twice over: its
configuration is chosen on the test year, and `scorecard.md` flags the Brazilian
and American rows as degenerate, with scalars reaching 4.8 and 46.4. Read those
two as beating a fit known to be degenerate. Zero-shot Brazil matches Brazil's
own fitted correction to 0.8%; fitted in region the model wins 3 of 5.

**A second gate on four regions never used as holdouts.** New Zealand, Chile,
Argentina and the Australian NEM, chosen for difficulty: Chile and New Zealand
have the lowest physiographic coverage in the study at 33.9% and 33.3%, and New
Zealand has twelve farms.

| holdout | uncorrected | linear, 5 regions | MLP, 9 regions | affine, in region |
|---|---|---|---|---|
| AR | 0.1512 | **0.1294** | 0.1351 | 0.133 † |
| AU-NEM | 0.1153 | **0.0788** | 0.0801 | 0.094 |
| CL | 0.1225 | 0.1279 | **0.1181** | 0.105 † |
| NZ | 0.1567 | **0.0848** | 0.0851 | 0.106 |

Both configurations beat uncorrected ERA5 in 4 of 4. Zero-shot also beats the
in-region affine fit in 3 of 4, two of those affine rows being degenerate.

**The headline configuration is linear heads on five regions.** MLP heads on
nine regions reach a higher mean skill on the original five (+0.338 against
+0.238) but were chosen after seeing results on those same five, and on the four
fresh regions they win only 1 of 4. Per the consequence registered before that
run, the linear five-region default stands. Neither dominates: the MLP wins the
mean, 0.374 against 0.354, and harms no region, while the linear model harms
Chile by 4.4%.

**The fitted quantities are ordered as the physics would suggest.** Median
speed-up by cell relief, across 29,634 units: 1.025 (below 100 m), 1.064,
1.193, 1.286, 1.392 (above 1,000 m). Tehachapi Pass receives 1.74, the Columbia
Basin 1.47, and Altamont Pass, whose acceleration is channelling rather than
orographic, receives 1.02, the term correctly declining to explain it.

**They are not measurements.** Efficiency and speed-up both reduce output, so
they trade along a nearly flat direction: starting the efficiency at 0.70, 0.80,
0.90 and 0.98 gives fitted values spanning 0.097 at a training-loss spread of
0.0004, while held-out RMSE spans 0.004. Predictions are robust to where the fit
puts the split; the split is not. A fitted efficiency of 0.79 must not be
reported as "this fleet loses 21%".

## 4. Negative results

These are the substantive output of the workstream, and four of them refute
predictions made with numbers behind them before the refuting data existed.

**Two candidate predictors of which regions gain, both refuted.** Physiographic
coverage looked convincing on five regions (Denmark 93% and +0.54 skill, the US
50% and +0.05) and does not survive nine: New Zealand has the lowest coverage of
all (33.3%) and the highest skill (+0.707), Argentina among the highest (91.5%)
and less than half of it. Pooled pearson +0.224, p = 0.56. Level-dominated bias
was offered as the replacement and also fails, +0.377, p = 0.32, refuted
outright by Australia, which has 0.6% level share and gains +0.534. **No
predictor is established.** For a new region the honest advice is to run it and
look.

**Two physics terms, ranked by the size of the residual they targeted, both
rejected on transfer.** A hyperbolic deep-array wake term `1/(1+cD)` in local
capacity density cut the dense-bin residual span by 61% and cost transfer in
5 of 5 regions, mean skill +0.338 to +0.235. A log-law profile term, applying an
hourly roughness inverted in closed form from the measured shear, cut the
tall-turbine residual by 62%, improved Germany alone, and cost mean skill 0.0089
against a 0.005 tolerance. Each removed the systematic it aimed at and
redistributed the error onto other axes rather than removing it.

The profile result pins the mechanism. The median span each head can impose on
the profile ratio is 13.30% for a shear-exponent offset and 4.24% for a
log-roughness offset, and widening the latter's bounds saturates at 7.32%
because z0 hits its physical clamp. **The power law's extra leverage consists of
profile shapes no physical roughness can produce**, and the fit was using it.
The term is deliberately not promoted by widening a bound past the physical
range, since that freedom is what the result indicts.

The wake result produces a falsifiable data statement instead: the observation
unit differs by region, and so does how much wake loss the target already
contains. A plant-level capacity factor is an array average with internal wakes
inside it; a single-turbine one is not. A single global coefficient cannot be
right for both. Fitting `c` separately by observation unit, or harmonising the
targets, should recover the transfer.

**Constraining all four terms at once improves transfer.** If the model carries
compensating freedom, squeezing one term relocates error and squeezing all of
them should behave differently. A scalar shrinking every bounded quantity toward
its starting value gives mean skill +0.2347 at no shrinkage, **+0.2655 at 0.60**
and +0.2559 at 0.35: an interior optimum, so too little freedom leaves no room
for a correction and too much lets the terms absorb one another. Removing 40% of
the model's freedom improves zero-shot transfer by 13% relative. This is the
only change in the programme that improved transfer by taking capability away,
and it explains why the wake and profile terms failed.

The value, as opposed to the hypothesis, did not confirm. On the four fresh
regions the registered gate required 3 of 4 and got 2 of 4, so **0.60 is not
adopted**. Three things hold alongside that: mean skill improves on the fresh
four as well (+0.354 to +0.389); the gain is concentrated where the model was
doing damage, Chile going from -0.091 skill to +0.102 while the other three move
by less than 0.007; and across all nine regions it wins 6 of 9, lifts mean skill
+0.288 to +0.320, and harms no region where no shrinkage harms one. It behaves
like insurance rather than a uniform gain. The per-region optimum varies (DE, DK
and US prefer 0.35; BR and UK prefer 0.60) and is unexplained.

**Measured and rejected without being built.** A seasonal term: month explains
0.7% to 3.9% of residual variance in every region, at amplitudes of 0.02 to 0.04
capacity factor. Abstention outside the physiographic envelope: mean skill fell
from +0.238 to +0.235, because the model degrades no region relative to
uncorrected ERA5 and the rule was solving a problem that did not arise. Real
Brazilian hub heights: sweeping the assumed height across 80 to 120 m moves RMSE
by 0.0006 total, and randomising heights around 100 m with a modern fleet's
spread costs 0.0001 against a seed spread of 0.0028, so building a 193-complex
turbine-spec table would not repay the effort for this purpose.

## 5. Where the remaining error sits

Fitted in region on all five original regions, so this is the model's floor
rather than a transfer penalty. `span/sd` is the swing in mean residual across a
slice relative to the residual's own spread.

| axis | span/sd | what the residual does |
|---|---|---|
| terrain spread at 84 km | 1.14 | +0.011 flat, -0.066 above 400 m (n=157) |
| local capacity density | 0.70 | +0.051 in the densest bin (>0.75 MW/km2) |
| hub height | 0.55 | +0.024 below 40 m, +0.036 above 120 m |
| ERA5-cell relief | 0.52 | |
| land fraction | 0.46 | |
| elevation | 0.27 | smallest, the density term doing its job |

Offshore against onshore, RMSE 0.122 against 0.072 at almost identical mean bias
(+0.014 against +0.013): offshore error is spread, not level.

Ranking candidate terms by residual size has now failed twice, so the two
remaining leads it produced, directional terrain exposure and offshore
boundary-layer physics, are not recommended on its strength.

## 6. Limits

- **One test year per region**, as throughout this repository. Orderings of two
  close arms are not meaningful; the uncorrected-to-corrected drop is.
- **The fitted physical quantities are not measurements** (section 3).
- **The incumbent affine still wins in region** in Germany and the UK.
- **Curtailment is not separable.** `eta` absorbs it and calls it a loss. The
  US carries an unscreened ERCOT/SPP confound, the Australian NEM a similar one.
- **Hub heights are defaults outside Europe**: 100 m for every Brazilian unit,
  80 m for 62% of American ones.
- **Observation units differ by region.** Capacity density removes this from the
  features; it does not remove it from the target.
- **Geolocation quality varies**: Germany has 579 distinct locations behind
  4,288 rows, the UK 332 behind 5,621.
- **No stability variables were downloaded.** Surface heat flux, boundary-layer
  height and friction velocity would let the profile use Monin-Obukhov
  similarity directly rather than inferring stability from the 10-100 m shear.
- **Curve library.** These runs use the licensed 236-curve library, matching the
  canonical runs they are compared against.
- **Two open questions, both blocked on region count.** The per-region optimum
  of the joint constraint, and any predictor of regional gain. Nine regions
  cannot separate either from noise, which puts data acquisition chosen for
  regime diversity ahead of further modelling.
