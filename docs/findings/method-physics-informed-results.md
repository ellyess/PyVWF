# E1: does a physics-informed correction transfer to unseen regions?

Gates were fixed in `method-physics-informed-prespecification.md` before any model was fitted.
Method: `method-physics-informed.md`. Evidence that motivated the design: `method-physics-informed-diagnostics.md`.

Run: commit `bd1c721`, licensed combined curve library (236 curves), five seeds,
60 epochs, leave-one-region-out over {DK, DE, UK, US, BR}. Scored with the
harness's own `skill_metrics` on each region's held-out test year, capacity
weighted, pseudo-replicates collapsed. 180 minutes.

## The raw table

Capacity-factor RMSE on the held-out test year. Lower is better.

| holdout | uncorrected | rf-transfer | **pinn (zero-shot)** | pinn-ablation | pinn in-region | affine in-region |
|---|---|---|---|---|---|---|
| DK | 0.1464 | 0.1011 | **0.0989** | 0.2298 | 0.0823 | 0.0855 |
| DE | 0.0860 | 0.0702 | **0.0790** | 0.1692 | 0.0597 | 0.0572 |
| UK | 0.1451 | 0.1504 | **0.1429** | 0.3252 | 0.1267 | 0.1146 |
| US | 0.1098 | 0.1691 | **0.1072** | 0.3522 | 0.0909 | 0.0978 |
| BR | 0.1386 | 0.1316 | **0.1062** | 0.2624 | 0.0954 | 0.1054 |

Skill against uncorrected ERA5 (`1 - MSE/MSE_uncorrected`):

| holdout | rf-transfer | **pinn** | pinn-ablation | pinn in-region |
|---|---|---|---|---|
| DK | +0.523 | **+0.544** | −1.464 | +0.684 |
| DE | +0.334 | **+0.156** | −2.876 | +0.518 |
| UK | −0.075 | **+0.031** | −4.023 | +0.238 |
| US | −1.373 | **+0.046** | −9.288 | +0.315 |
| BR | +0.098 | **+0.413** | −2.584 | +0.527 |

Seed spread (sd of RMSE): pinn 0.0002–0.0011; ablation **0.0297–0.1059**.

`affine in-region` is the incumbent per-cluster affine correction at its
best-of-sweep configuration, read from `cluster_sweep_2026-07-24/*_metrics.csv`.
It is a reference point, not a transfer arm, and it is optimistic in two ways.

First, its configuration is chosen on the test year, which the physics-informed
arms never do. Second, and more sharply: `scorecard.md` marks the Brazilian and
American rows with a dagger, meaning `fit_quality` finds the fit behind them
**degenerate** against the calibrated bounds (scalar in 0.2 to 3.0) -- the US
row's factors reach a scalar of 46.4 and Brazil's 4.8. So in two of the five
regions the number this work is compared against comes from a fit the
repository's own quality check rejects. That does not change any figure in the
table, and the comparison is still the right one to make, because it is what
the method actually delivers today. It does mean the BR and US comparisons
should be read as "better than a fit known to be degenerate" rather than
"better than a sound one".

## Gates

**P1 PASS.** Zero-shot beats uncorrected ERA5 in **5 of 5** regions, and
degrades none: not by 10%, not at all.

**P2 PASS.** Beats the incumbent RF transfer in **4 of 5**. The exception is
Germany (0.0790 against 0.0702), and it is worth naming why: German turbine
locations are postcode centroids, 579 distinct points behind 4,288 rows, so the
terrain a physiography-driven model reads is sampled at settlement centres
rather than at turbines. A method that leans on terrain is handicapped there in
a way a method leaning on latitude and longitude is not.

**P3: the physics earns its place, decisively.** Removing the constraints while
holding features and parameter count fixed costs 0.09 to 0.24 RMSE in every
region, and turns the correction from useful into far worse than doing nothing
(skill −1.46 to −9.29). It also destabilises the fit: seed spread rises from
~0.0005 to 0.03–0.11. So the gain is not "better features"; it is the
constraints.

## What the comparison actually shows

**The incumbent ML does harm; this does not.** RandomForest transfer of the
per-cluster factors is negative in 2 of 5 regions, and in the United States it
is catastrophic: RMSE 0.1098 → 0.1691, a 54% degradation, from a fitted mean
scalar of 1.41 ± 0.54 applied to a fleet whose reanalysis bias does not warrant
it. The physics-informed correction is positive in 5 of 5. For a method offered
as general, never making things worse matters more than the size of the average
gain.

**Where the gains are large, and where they are not.** Denmark (+0.54) and
Brazil (+0.41) gain a great deal; Germany (+0.16) moderately; the UK (+0.03) and
the US (+0.05) barely. That ordering is not arbitrary; see coverage, below.

**Zero-shot Brazil matches Brazil's own fitted correction.** 0.1062 against
0.1054 for the incumbent affine fitted on Brazilian data at its best sweep
configuration: a difference of 0.8%. Everywhere else the in-region fit still
wins by 10–38%, as it should.

**As a replacement for the incumbent, fitted in-region, it wins 3 of 5**: DK
−3.7%, US −7.1%, BR −9.5% against the affine's best-of-sweep; DE +4.4% and UK
+10.5% against it. With 37 parameters rather than roughly 800 free per-cluster
factors, and no configuration chosen on the test year.

## Coverage predicts the ordering (SUPERSEDED, see below)

D5 measured, before any model ran, how much of each held-out region's
physiography the other four span:

| holdout | joint coverage | pinn skill |
|---|---|---|
| DK | 93.0% | +0.544 |
| DE | 89.4% | +0.156 |
| BR | 70.5% | +0.413 |
| UK | 50.5% | +0.031 |
| US | 49.8% | +0.046 |

On these five that reads cleanly: the two regions where the correction barely
helps are the two where half the units sit outside the training envelope.

**This does not survive the four regions added later.** New Zealand has the
lowest coverage in the whole study (33.3%) and the highest skill of any region
(+0.707); Argentina has among the highest coverage (91.5%) and less than half of
New Zealand's skill. Pooled over all nine, coverage and skill correlate at
pearson +0.224, p = 0.56 -- nothing. A replacement predictor, level-dominated
bias, was then offered and also fails (+0.377, p = 0.32).

The table above is left standing because it is what the five-region evidence
showed and the claim was made on it. It should be read as refuted:
`method-physics-informed-predictors.md` has both candidates and the evidence
against each. Coverage still bounds what a terrain model can EXTRAPOLATE; it
does not predict how much a region stands to gain.

## The pre-registered UK prediction: satisfied, but only trivially

Addendum 3 predicted, before the UK holdout ran, that the physics-informed
advantage over the RF baseline would be at least the median across regions,
because the UK's per-cluster factors are the noisiest (D4) and this method never
reads them.

| | BR | DE | DK | UK | US |
|---|---|---|---|---|---|
| rmse(rf) − rmse(pinn) | +0.0254 | −0.0088 | +0.0022 | **+0.0076** | +0.0619 |

The median is +0.0076 and the UK **is** the median region, so the prediction is
satisfied by identity rather than by evidence. It should not be counted as
support. The honest reading is that the UK's advantage is middling, and that D4's
target-noise explanation for the UK is neither confirmed nor refuted here.

## What must not be claimed

- **Not an in-region replacement everywhere.** The incumbent affine still wins
  in Germany and the UK when both are fitted in-region.
- **The fitted physical quantities are not measurements.** D6 showed the
  efficiency and speed-up trade off along a nearly flat direction; predictions
  are robust to where the fit puts the split, the split is not.
- **One test year per region**, as throughout this repository. Orderings of two
  close arms are not meaningful; the uncorrected-to-corrected drop is.
- **US carries an unscreened curtailment confound** (ERCOT/SPP), unchanged from
  the existing scorecard.
- **Curve library.** These runs use the licensed 236-curve library, matching the
  canonical runs they are compared against. The open library is not expected to
  change the verdict, but that has not been re-run here.
