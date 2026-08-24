# Evaluation: what the programme established, and what to do next

Closing document for the physics-informed correction workstream. Gates and
predictions were fixed in `method-physics-informed-prespecification.md` before each experiment;
the gated result is in `method-physics-informed-results.md`; the diagnosis that motivated the
design is in `method-physics-informed-diagnostics.md`; the method is in `method-physics-informed.md`.

## 1. The configuration ladder

Mean skill against uncorrected ERA5, across the five leave-one-region-out
holdouts. Every arm is zero-shot: the model never saw the region it is scored on.

| configuration | mean skill | note |
|---|---|---|
| incumbent RF transfer | **−0.099** | harmful on average |
| A: linear heads, 5 regions | +0.238 | **the gated model** (P1/P2/P3) |
| A + abstention | +0.235 | no better |
| A + air density | +0.247 | free physics, small gain |
| B: MLP heads, 5 regions | +0.311 | |
| C: linear heads, 9 regions | +0.318 | |
| **D: MLP heads, 9 regions** | **+0.338** | best |

Per region, RMSE on the held-out test year:

| holdout | uncorrected | RF transfer | **D** | affine, fitted IN region |
|---|---|---|---|---|
| BR | 0.1386 | 0.1316 | **0.1030** | 0.1054 |
| DE | 0.0860 | 0.0702 | **0.0678** | 0.0572 |
| DK | 0.1464 | 0.1011 | **0.0951** | 0.0855 |
| UK | 0.1451 | 0.1504 | **0.1363** | 0.1146 |
| US | 0.1098 | 0.1691 | **0.1001** | 0.0978 |

D beats the incumbent RF transfer in **5 of 5**, and beats the incumbent affine
correction *fitted on the region's own data* in 1 of 5 (Brazil, 0.1030 against
0.1054 -- a row `scorecard.md` flags as a degenerate fit, so read it as clearing
a low bar rather than a sound one). E8's pre-set bar -- beat both B and C in at least 3 of 5 -- is met
(3/5 and 4/5), so the two effects are additive rather than one carrying the other.

## 2. What the failed predictions taught

Seven registered predictions failed. Three matter:

**Envelope coverage is not the binding constraint.** D5 measured that adding
four regions moved the original five's physiographic coverage by about three
points, and I predicted transfer would barely move. It improved in 5 of 5, mean
skill +0.238 to +0.318. The mechanism is not that the new regions cover the
holdout's terrain; it is that the **shared global constants** -- the sub-daily
spread factor, the relief length scale, how efficiency depends on fleet density
-- are pinned better by more data, and those help everywhere. This is a
stronger argument for data acquisition than the coverage framing gave, and it
revises `method-ml-transfer.md`'s "regime coverage is the binding constraint"
rather than confirming it.

**Constrained capacity helps rather than hurts.** I predicted MLP heads would
fit within-region structure and fail to transfer. They transfer better in 5 of 5
(mean 0.1068 to 0.1022) while being no better in-region. Inside a constrained
form the extra flexibility cannot be spent on region identity -- the relief pin
and the bounds forbid it -- so it is spent on better shapes for the physical
functions. The linear default was wrong; D is the configuration to carry forward.

**Air density moves the speed-up UP, not down.** I predicted the fitted speed-up
above 800 m would fall once density was modelled. It rose, 1.462 to 1.537, and
the rise is the physically correct direction: without a density term the model
must SUPPRESS the speed-up to reproduce the lower observed output in thin air.
The movement is ten times larger above 800 m than below 200 m, so the term acts
exactly where it should.

## 3. Where the remaining error is

Residual anatomy (`d7`), fitted in-region on all five regions so what it
measures is the model's floor rather than a transfer penalty. `span/sd` is the
swing in mean residual across a slice relative to the residual's own spread:
large means a systematic term is missing, near zero means the axis is handled.

| axis | span/sd | what the residual does |
|---|---|---|
| terrain spread at 84 km | **1.14** | +0.011 flat → **−0.066** above 400 m (n=157) |
| local capacity density | **0.70** | **+0.051** in the densest bin (>0.75 MW/km²) |
| hub height | **0.55** | +0.024 below 40 m, **+0.036** above 120 m, −0.006 at 40-60 m |
| ERA5-cell relief | 0.52 | |
| land fraction | 0.46 | |
| elevation | 0.27 | smallest -- the density term is doing its job |

Offshore against onshore: RMSE **0.122 against 0.072**, at almost identical mean
bias (+0.014 against +0.013). Offshore error is spread, not level.

## 4. What to do next, in order

**1. A physically-shaped wake term.** The model over-predicts by **5 percentage
points of capacity factor** in the densest fleets. Efficiency currently depends
on capacity density through a smooth learned function; the residual says that
dependence is too weak at the top end. The fix is not more flexibility but the
right shape -- a Jensen-type array efficiency, `eta ~ exp(-c * density)`, with
`c` the only free number. Highest value per unit of effort, and it is physics
the model is currently approximating badly rather than physics it lacks.

**2. A profile with the right curvature.** The model over-predicts at BOTH hub
height extremes and is unbiased in the middle, which is the signature of a
power law fitted between 10 m and 100 m being extrapolated outside that range.
A log profile with a stability correction has the right curvature; failing that,
letting the shear correction depend on `ln(h/100)` would absorb most of it. This
matters most for Denmark's old low-hub fleet and for modern 120-150 m machines.

**3. Directional terrain exposure.** The named miss is Altamont Pass: fitted
speed-up 1.016 where the incumbent needs 3.43, because its ERA5-cell relief is
only 130 m. Altamont's resource is flow CHANNELLED through a gap in the coastal
hills -- a directional effect that scalar relief cannot see at any scale. ERA5
carries wind direction; terrain exposure computed per sector and weighted by the
daily direction is the WAsP-style term this needs.

**4. Offshore physics.** 71% higher error offshore at the same mean bias. The
marine boundary layer and large-array wakes are both unrepresented.

## 5. What NOT to do, measured rather than assumed

- **A seasonal term.** Month explains **0.7-3.9%** of residual variance in every
  region, at amplitudes of 0.02-0.04 CF. The residual lives in cross-unit
  spread, not the calendar. This looked like an obvious gain and is not one.
- **Abstention outside the physiographic envelope.** Mean skill FELL (+0.238 to
  +0.235). It helps the two lowest-coverage regions and costs Brazil more.
  Since the model degrades no region relative to uncorrected ERA5, the rule was
  solving a problem that did not arise.
- **More terrain scales.** Elevation already carries the least recoverable
  structure of any axis, and relief is mid-table. The 84 km signal is real, but
  the MLP already recovers part of it -- which is why capacity helped -- so the
  gain is in functional form, not in more descriptors.

## 6. Data, not modelling

- **German coordinates.** 579 distinct locations behind 4,288 rows (postcode
  centroids). Germany is the single region where the incumbent RF beats the
  gated model, and a terrain-driven method reading settlement centres is the
  obvious explanation. The repository has a public turbine register; real
  coordinates would test it directly.
- **Hub heights outside Europe.** Every Brazilian unit carries a 100 m default
  and 62% of American ones carry 80 m, so the profile term is informed by real
  metadata only in Europe -- and the profile is improvement (2) above.
- **ERA5 stability variables.** Surface heat flux, boundary-layer height and
  friction velocity are one CDS request, and would let the profile use
  Monin-Obukhov similarity directly instead of inferring stability from the
  10-100 m shear.
- **US curtailment.** Unchanged from the existing scorecard: without
  semi-dispatch data the efficiency term absorbs curtailment and calls it a loss.

## 7. Honest limits, restated

- The fitted physical quantities are **not measurements**: D6 showed efficiency
  and speed-up trade off along a nearly flat direction. Predictions are robust
  to where the fit puts the split; the split is not.
- **One test year per region.** Orderings of two close arms are not meaningful.
- **The incumbent still wins in-region** in Germany and the UK.
- Configuration D was selected after seeing E5 and E7, and is labelled
  **post-hoc** in the pre-specification. It has not been re-validated against a
  fresh pre-specified gate, and should be before it is used as a headline.
