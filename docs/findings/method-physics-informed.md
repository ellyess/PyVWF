# A physics-informed correction for ERA5 wind, from the data in this repository

What follows is the method: what it takes in, what it computes, what it learns,
and why each learned quantity is one that can survive being carried to a region
it was never fitted on. Results and gate outcomes are in `method-physics-informed-results.md`; the
evidence that motivated the design is in `method-physics-informed-diagnostics.md`; the gates were
fixed in advance in `method-physics-informed-prespecification.md`.

## The problem this is answering

`w' = a*w + b`, fitted per cluster per time slice against observed generation,
works well where it is fitted and does not transfer (`method-ml-transfer.md`).
The diagnostics locate the reason in the SHAPE of the method rather than in the
choice of regressor:

- the fitted pair is rank-1 (`pearson(a, b) = -0.999`), a rotation about the
  turbine cut-in speed, so it carries one number, not two;
- that number is estimated from as few as three turbines per cluster, and in the
  UK fleet two clusters 20-30 km apart already differ by 59% of the region's
  total scalar variance;
- it absorbs, into a single wind-speed multiplier, at least five physically
  distinct effects: unresolved terrain, hub-height extrapolation error, air
  density, the daily-averaging bias of a non-linear power curve, and conversion
  losses. Those five have different spatial signatures, so their sum has none.

A scalar of 1.37 means "Brazil". It cannot mean anything else, because nothing
in its definition distinguishes the parts.

## The idea

Keep the physics that is known, learn only the parts that are not, and make
every learned quantity one whose meaning does not change with the region.

    ln u_hub = ln w100  +  (shear + delta) * ln(h/100)  +  gamma
    u_curve  = u_hub * (rho/rho0)^(1/3)
    cf       = eta * E[ P(u_curve + sigma*Z) ],   Z ~ N(0,1)
    CF_month = mean over the days of the month

Reading left to right: the reanalysis 100 m wind; a power-law profile to hub
height whose exponent is MEASURED hourly from the 10 m and 100 m fields and
corrected by `delta`; a terrain speed-up `gamma`; the IEC air-density
correction, which has no free parameters; the power curve integrated over the
within-day wind distribution; and the monthly mean the observations report.

Four things are learned, all bounded, all physical:

| symbol | meaning | bounds | depends on |
|---|---|---|---|
| `gamma` | log terrain speed-up | 0.67x to 2.46x | physiography |
| `delta` | correction to the measured 10-100 m shear exponent | -0.15 to +0.25 | physiography |
| `eta` | conversion efficiency: wake, availability, electrical, curtailment | 0.55 to 1.0 | fleet |
| `kappa` | how much within-day wind spread the pre-smoothed curves have not already absorbed | 0 to 1.5 | one global number |

## What makes it transferable, concretely

**The speed-up is pinned to zero on flat ground, structurally.** `gamma` is not
a free network output. It is an amplitude times a saturating function of the
ERA5-cell relief,

    gamma = A(x) * r / (1 + r),      r = relief / relief_scale

so a site with no sub-grid relief gets exactly no terrain correction however
`A` has been fitted. This is a hard guarantee, not a penalty term, and it is
tested. It matters because most training data is flat and most of the world is
flat: without it the term is free to drift there and take the extrapolation with
it.

**Relief, not elevation.** D1 predicted the correction would rise with a site's
elevation ABOVE its ERA5 cell mean, and that prediction failed -- the sign is
region-inconsistent, and wrong in Germany. What does hold, in four of five
regions, is the elevation RANGE within the cell. The largest American scalar
(4.25, central Washington) sits 142 m BELOW its cell mean inside a cell spanning
1,083 m of relief. The mechanism is not that the turbine is high; it is that the
cell contains structure the reanalysis smooths away, and developers site into
it. Relief measures how much such structure exists; elevation excess does not.

**Terrain at the scale the error lives at.** The earlier experiment described
terrain on a 3-pixel window of a 30 arc-second grid -- about 90 m around a
centroid. Pooled importances put `relief_28km` at 0.19 and `std_84km` at 0.17
against 0.02 for the kilometre-scale features. Descriptors here are computed at
1, 5, 28 and 84 km: micro-siting, the individual ridge, the ERA5 cell, and the
orographic blocking scale.

**No feature can name the region.** No longitude, no latitude, no country. The
fleet descriptors are capacity density in MW/km2, which is invariant to whether
a data row is a turbine, a farm, a plant or a complex -- unlike raw capacity,
which separates Europe from the Americas perfectly and was removed for that
reason (addendum 2).

**The supervision is the observation.** There is no intermediate target, so the
estimation noise, k-means partition dependence and rank-1 degeneracy the
diagnostics found in the per-cluster factors never enter. Every turbine-month is
a training row rather than being averaged into one of a hundred cluster
summaries.

## What the data supports that the incumbent was discarding

**Within-day wind spread.** `prep_era5` averages the hourly reanalysis to daily
means and every published result is built on that field. The power curve is
strongly non-linear across the range a day's wind actually spans -- in Denmark
the within-day standard deviation is 21% of the daily mean -- so evaluating the
curve at the daily mean is systematically wrong, in a direction that depends on
where the day sits on the curve. The daily standard deviation is retained here
and the curve is integrated over it by Gauss-Hermite quadrature. `kappa` exists
because the shipped curves are already Gaussian-smoothed by the VWF method
(sigma = 0.6 + 0.2w) for turbulence and within-farm spread at hourly resolution;
how much of the within-day spread that already covers is a question for the
data, not for an assumption.

**A time-varying shear exponent.** The pre-combined European files ship a single
time-AVERAGED roughness field, while the raw American files ship none and the
pipeline inverts the log profile hour by hour. So Europe runs on a
climatological roughness and the Americas on a varying one: a difference in the
character of an input, across exactly the regions transfer is measured between.
The power-law exponent between 10 m and 100 m is computed identically in every
region and responds to atmospheric stability, which a static roughness cannot.

**Air density.** Curves are published at 1.225 kg/m3. A turbine at 1,200 m --
Tehachapi, the Bahia highlands, much of the interior American and Brazilian
fleets -- stands in 11% thinner air. Nothing in the incumbent represents this;
it is absorbed by the scalar. Here it is the ISA density from site elevation and
the IEC equivalent-speed correction, with nothing to fit.

## What it is honest to call this

The heads are small: linear by default, optionally a two-layer MLP. With linear
heads the whole model has **37 parameters**, against roughly 800 free
per-cluster factors in a k=100, four-season affine fit. Calling 37 bounded
coefficients a neural network is generous; what does the work is the operator
they sit inside, which is differentiable end to end so that the observations can
reach them. The MLP variant is run as a declared sensitivity, on the
expectation that extra capacity helps in-region and hurts transfer.

## What it cannot do

- **Curtailment is not separable.** In high-penetration markets a suppressed
  capacity factor is a market outcome, not a resource one. `eta` will absorb it
  and call it a loss. This needs semi-dispatch data, which the repository does
  not have for the US.
- **Hub heights are defaults outside Europe.** Every Brazilian unit carries a
  uniform 100 m and 62% of American ones carry 80 m, so the profile term is
  informed by real metadata only in Europe.
- **Observation units differ by region** -- turbine, farm, plant, complex -- and
  a plant-level capacity factor already contains intra-plant wake losses that a
  single-turbine one does not. Capacity density removes this from the FEATURES;
  it does not remove it from the TARGET.
- **No stability variables were downloaded.** Surface heat flux, boundary-layer
  height and friction velocity would let the profile use Monin-Obukhov
  similarity directly instead of inferring stability from the 10-100 m shear.
  That is one CDS request, and it is named here rather than done.
