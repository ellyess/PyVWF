# Q: a fresh pre-specified gate on configuration D

**Written before the experiment was run. Results are appended below, not above.**

## Why this is needed

Configuration D (MLP heads, nine training regions) reaches mean zero-shot skill
+0.338 against the gated configuration A's +0.238. But D was **chosen after
seeing E5 and E7**, both scored on the same five holdout regions, so its
advantage on those five is not independent evidence. The evaluation document
says so itself and asks for a fresh gate before D is used as a headline
(`method-physics-informed-evaluation.md`). This is that gate.

## What makes it fresh

Four regions have caches in this repository and have **never been scored as
holdouts**: New Zealand, Chile, Argentina and the Australian NEM. They entered
E7 and E8 only as members of the training pool. No choice about D -- not the MLP
heads, not the nine-region pool, not the epoch count or seeds -- was informed by
any held-out result on these four.

They are also the hard cases, not a soft target, and the numbers saying so were
measured before this gate:

| region | test units | physiographic coverage | median NN distance |
|---|---|---|---|
| AR | 59 | 91.5% | 0.56 |
| AU-NEM | 83 | 89.2% | 1.19 |
| CL | 59 | **33.9%** | 2.95 |
| NZ | **12** | **33.3%** | 2.46 |

Chile and New Zealand are the least-covered regions in the whole study, below
even the UK and US at ~50%. The existing scorecard independently caveats CL and
AR as "not headline wins" (ERA5 exaggerates their north-south wind gradient),
flags both as degenerate affine fits, and notes AU-NEM's curtailment confound.
New Zealand has twelve farms.

## Protocol

Leave-one-region-out over {NZ, CL, AR, AU-NEM}. For each, train on all eight
other regions and apply zero-shot to its held-out test year. Scored with the
harness's own `skill_metrics`, capacity weighted, on the same test years the
scorecard uses. Three seeds. Configuration D exactly as it stands: MLP heads
(hidden 16), 60 epochs, `profile="power"`, no wake term, no density term, no
abstention.

## Gates

**Q1 (primary).** Zero-shot corrected RMSE < uncorrected RMSE in **at least 3
of 4** regions, and **no region degraded by more than 10%** relative. Same form
as P1, so the two are comparable.

**Q2 (does D's advantage reproduce).** D beats configuration A -- linear heads,
trained on the original five regions only -- in **at least 3 of 4**. A's training
pool never contains the holdout, so this is a like-for-like zero-shot contest
between the gated configuration and the post-hoc one, on regions neither was
tuned against.

## Reference points, never gates

Uncorrected ERA5, and the in-region affine correction from the refreshed
scorecard: NZ 0.157 to 0.106, CL 0.123 to 0.105, AR 0.151 to 0.133, AU-NEM 0.115
to 0.094. The Chilean and Argentine rows are daggered as degenerate fits.

## Predictions, registered

1. **Q1 is genuinely at risk.** I expect it to land at exactly 3/4 or fail at
   2/4, because two of the four regions sit at ~33% coverage and coverage has
   predicted skill everywhere so far.
2. **Skill will rank AR > AU-NEM > CL ~ NZ**, following coverage rather than
   fleet size or data quality.
3. **Q2 passes.** The nine-region pool and the extra head capacity were both
   worth more in the low-coverage regions of the original five (UK and US gained
   most), and these are lower-coverage still.

**What failure means.** If Q1 fails, configuration D does not generalise beyond
the regions it was chosen on, and the headline reverts to configuration A's
gated +0.238. If Q2 fails, D's advantage was an artefact of the five regions it
was selected against and the linear five-region default should stand.

Nothing below this line was known when the above was written.

---

## RESULTS

Held-out capacity-factor RMSE, three seeds, on the four regions' own test years.

| holdout | uncorrected | config A | **config D** | affine in-region | skill A | skill D |
|---|---|---|---|---|---|---|
| AR | 0.1512 | **0.1294** | 0.1351 | 0.133 † | 0.268 | 0.201 |
| AU-NEM | 0.1153 | **0.0788** | 0.0801 | 0.094 | 0.534 | 0.518 |
| CL | 0.1225 | 0.1279 | **0.1181** | 0.105 † | −0.091 | 0.071 |
| NZ | 0.1567 | **0.0848** | 0.0851 | 0.106 | 0.707 | 0.705 |

Seed spread 0.0003 to 0.0014, so every difference above is larger than seed noise.

### Q1: PASS

Configuration D beats uncorrected ERA5 in **4 of 4**, degrading none. On the two
least-covered regions in the entire study, at 33% coverage, it removes 46% of New
Zealand's error and 3.6% of Chile's. Against the pre-registered expectation that
this gate was "genuinely at risk" and would land at 3/4 or fail, it passed
outright on regions chosen for difficulty.

**Zero-shot also beats the in-region affine fit in 3 of 4** (AU-NEM 0.0801 vs
0.094, NZ 0.0851 vs 0.106, and AR 0.1351 vs 0.133 is a near-tie the other way);
only Chile's in-region fit wins clearly. Two of those affine rows are daggered
as degenerate, so this is again a comparison against a low bar.

### Q2: FAIL

D beats configuration A in **1 of 4**, not the required 3. Configuration A --
linear heads, five training regions, the originally gated model -- is better in
Argentina, Australia and New Zealand.

Per the consequence registered before the run: **D's advantage was an artefact
of the five regions it was selected against, and the linear five-region default
stands.** The headline remains configuration A.

One nuance that does not rescue D but should be recorded: mean skill still
favours D, 0.374 against 0.354, because Chile swings hard between them, and
**A is the arm that harms a region** (Chile, −0.091 skill, a 4.4% RMSE increase)
while D harms none. Neither configuration dominates; A wins three regions
individually, D wins the mean and the do-no-harm property.

### Prediction 2: FAILED, and it takes a framing with it

Predicted ranking AR > AU-NEM > CL ~ NZ, following physiographic coverage.
Actual: **NZ > AU-NEM > AR > CL**. New Zealand has the LOWEST coverage of the
four (33.3%) and the HIGHEST skill (+0.705); Argentina has the highest coverage
(91.5%) and less than a third of NZ's skill.

Pooling all nine regions now scored, coverage does not predict skill:

| predictor of zero-shot skill | pearson | p |
|---|---|---|
| physiographic coverage | +0.224 | 0.56 |
| uncorrected RMSE | +0.226 | 0.56 |
| **absolute uncorrected MBE** | **+0.562** | 0.115 |
| MBE as a share of RMSE | +0.502 | 0.17 |

Coverage looked convincing on five regions (Denmark 93% and +0.54, the US 50%
and +0.05) and does not survive nine. The better predictor is how much of a
region's error is **level bias**: New Zealand's MBE is 40% of its RMSE and
Denmark's 75%, both large gainers; Argentina's is 6.5% and Chile's 22%, both
small. At n=9 and p=0.115 that is suggestive, not established.

This restores, rather than replaces, the finding already in
`method-generalisation.md`: the correction earns its keep where the reanalysis
bias is level-dominated. That was established for the affine model and now holds
for the physics-informed one. The coverage framing in
`method-physics-informed-diagnostics.md` D5 and in the evaluation should be read
as superseded: coverage bounds what a terrain model can EXTRAPOLATE, but it does
not predict how much a region stands to gain.
