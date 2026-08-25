# Pre-specification: physics-informed ERA5 wind bias correction

Written **before** any model in this workstream was fitted. Diagnostics D0-D2
(`method-physics-informed-diagnostics.md`) are measurements of existing artefacts and carry no
gate; everything from D3 onward involves choices that could be tuned to the
evaluation, so the gates are fixed here first.

## The question

The affine correction `w' = a*w + b` is fitted per cluster per region and does
not transfer to unseen regions (`method-ml-transfer.md`, 1/5 under its gate).
Can a correction whose free parameters are **physical quantities constrained by
boundary-layer theory**, learned as functions of local physiography, transfer
zero-shot to a region it was never fitted on?

## Metric (fixed here, applies to every gate below)

Transfer is scored as **skill against declining to correct**:

    skill = 1 - MSE(method) / MSE(uncorrected ERA5)

D0 established why this and not `r2_score` against the holdout's own mean: the
holdout mean is not available to a practitioner arriving in an unseen region,
so an R2 denominator built from it scores a question nobody can act on. Both
numbers are reported throughout for comparability with the published result.

## Gates

**P1 (primary).** Leave-one-region-out over {DK, DE, UK, US, BR}: fit on four,
apply zero-shot to the fifth, score capacity-factor RMSE on that region's
held-out test year at fleet scope, against observed generation.

- PASS iff (a) corrected RMSE < uncorrected RMSE in **>= 3 of 5** regions, and
  (b) **no region is degraded by more than 10%** relative RMSE.
- Clause (b) is part of the gate, not a footnote: a method offered as general
  must not badly damage the regions it fails on.

**P2 (comparative).** The same LORO protocol run for the incumbent statistical
baseline (RandomForest on the per-cluster scalar, published feature set SET_A,
RF(100/10/10), seeds [0,1,2,3,42]).

- The physics-informed model is reported as an improvement only if it beats
  P2's skill in **>= 3 of 5** regions. Beating "no correction" while losing to
  the incumbent is a negative result and will be reported as one.

**P3 (does the physics earn its place).** An ablation in which every physical
constraint is removed (same features, same capacity, free-form regression onto
the same target) must be run and reported alongside. If the unconstrained
ablation matches the constrained model, the physics is decorative and the
finding is that feature scale, not physics, was the binding constraint.

## Reference points (reported, never gates)

- Within-region fitted affine correction: the incumbent method's in-region
  performance, an upper bound for a zero-shot method, not a target.
- Uncorrected ERA5.

## Falsification conditions, stated in advance

- P1 fails -> the physics-informed parameterisation does not generalise either,
  and the honest headline is that the constraint is data coverage, not model
  form (consistent with `method-ml-transfer.md`).
- P1 passes but P2 fails -> the gain is from feature scale, not physics.
- P3's ablation matches -> same conclusion as above, stated plainly.

## Committed in advance

- Regions, their k, and their train/test years are those of the canonical
  post-fix runs already on disk; no region is dropped after seeing results.
- Seeds are [0,1,2,3,42] throughout, as in the published experiment.
- Any configuration explored beyond the pre-specified one is reported as a
  sensitivity, and cannot promote a failed gate.

---

## Addendum 1: air density (registered before E1's results were read)

Written while E1 was still running; the E1 output file at the time contained
only the uncorrected baselines. Recorded here rather than folded in silently,
because adding a model term after seeing a transfer score is tuning, and adding
one before is physics.

**The omission.** Power scales with air density, and neither the incumbent
pipeline nor the model as first committed represents it. Power curves are quoted
at the ISO standard 1.225 kg/m3. A turbine at 1,200 m -- Tehachapi, the Bahia
highlands, much of the interior US and Brazilian fleets -- sits in air about 11%
thinner, which is an 11% power reduction that nothing in either model accounts
for. In the incumbent it is absorbed by the affine scalar; in the model as
committed it is absorbed by the terrain speed-up, whose amplitude is driven by
relief -- and relief correlates with elevation. The speed-up term is therefore
currently conflating two different physical effects with opposite signs.

**The fix has no free parameters.** ISA density ratio from site elevation,

    rho/rho0 = (1 - 0.0065 z / 288.15) ** 4.2559

and the IEC 61400-12 equivalent-speed correction `v_eff = v * (rho/rho0)**(1/3)`
applied before the power curve. Elevation is already in the cache.

**Pre-registered predictions**, all three of which can fail:

1. Fitted speed-up at high-elevation sites (z > 800 m) FALLS when density is
   added, because it is currently absorbing the density deficit. If it does not
   move, the term is not doing what it is named for.
2. LORO transfer improves, or is unchanged, for the two regions with
   high-elevation fleets (US, BR).
3. LORO transfer for the near-sea-level fleets (DK, DE, UK) changes by less
   than 0.002 RMSE, since their density ratios are within 1% of unity.

**Scoring.** Density is a separate flag, so its contribution is measured on its
own (E3) rather than being absorbed into the headline. The P3 ablation excludes
it, keeping that gate a clean physics-versus-no-physics comparison. E1's
pre-specified gates are scored on the model as committed, without density; a
density-enabled run is a separate, clearly labelled arm and cannot promote a
failed gate.

---

## Addendum 2: fleet features (registered before any transfer result was read)

The first E1 launch was stopped ten minutes in and discarded. At that point the
run had printed only the uncorrected baselines -- no arm, no seed, no transfer
score had been produced -- so nothing about the change below was informed by a
result.

**The defect.** The conversion-efficiency head was given `log10(capacity)` and
`p_density` as fleet features. Neither is comparable across regions:

| | DK | DE | UK | US | BR |
|---|---|---|---|---|---|
| observation unit | turbine | turbine | farm | plant | complex |
| median capacity (kW) | 750 | 1,800 | 2,300 | **75,000** | **136,400** |
| `p_density` present | yes | yes | yes | **no** | **no** |

`log10(capacity)` runs 2.9-3.4 in Europe against 4.9-5.1 in the Americas: two
disjoint ranges, so the head could label the continent from a feature that is
supposed to describe a fleet. `p_density` is absent for US and BR entirely, and
was being filled with a constant, which is itself a region flag. A transfer
score from that model would measure the wrong thing.

**The replacement.** Installed capacity per unit area within 10 km and 50 km,
plus the offshore flag and hub height. Capacity density is invariant to how rows
are aggregated -- the megawatts inside a 10 km circle are the same number
whether they arrive as one plant row or forty turbine rows -- and it is the
quantity wake losses actually depend on. After the change no fleet feature
separates the continents: the 5th-95th percentile ranges of both densities
overlap across all five regions.

**Two limits this does not fix, stated rather than hidden.** Hub height is a
uniform 100 m default for every Brazilian unit and 80 m for 62% of American
ones, so it carries real information only in Europe. And capacity density is
computed from the OBSERVED fleet, so it understates the true density wherever
reporting coverage is partial.

**Also fixed here:** epochs cut from 80 to 60, on the measurement that the
training loss moves 0.5% between epoch 40 and epoch 79. Gauss-Hermite nodes
stay at 5, on the measurement that 3, 5 and 9 give an identical held-out RMSE to
four decimals.

---

## Addendum 3: the UK prediction (registered while the DK holdout was running)

E1 runs its holdouts in the order DK, DE, UK, US, BR, and at the time of
writing the run had produced no completed arm for any region. The United
Kingdom is third, roughly two hours away. What follows is therefore a genuine
forecast, and it is the sharpest available test of the claim that the
two-stage design, not the regressor, is what fails.

**The reasoning.** D4 measured the reliability of the per-cluster affine
factors and found the UK's are the worst of the five by a wide margin: two
clusters 20-30 km apart already differ by 59% of the region's entire scalar
variance, against 6% in Brazil and 15% in Germany. At k=100 over 348 turbines
the UK's clusters hold about three and a half turbines each, and the factors
are correspondingly noisy. Any method that learns FROM those factors -- the
published RF baseline included -- is fitting a target that is substantially
estimation noise, and D3 made the UK worse, not better, by describing terrain
more carefully: a model chasing noise it cannot distinguish from signal.

The physics-informed model never touches those factors. It is fitted to
observed generation directly, so the UK's thin clusters cost it nothing beyond
the observations they were built from.

**The prediction.** The physics-informed model's advantage over the RF transfer
baseline, measured as `rmse(rf-transfer) - rmse(pinn)` on the held-out year,
is at least as large for the UK as the median of that quantity across the five
regions.

**What would falsify it.** A UK advantage below the median would say the UK's
difficulty is something other than target noise -- most likely its
pseudo-replicated farm observations or its offshore fraction -- and the D4
explanation for the UK failure would have to be withdrawn.

**Not a gate.** P1, P2 and P3 stand as written. This is one prediction about
one region, recorded so it cannot be constructed after the fact.

---

## Addendum 4: two further arms, registered before any transfer result exists

E1 is running and has produced no completed arm. Both of the following are
committed to now, and both will be reported whatever E1 says -- including if
E1's gates pass and they turn out to be unnecessary.

### E4: abstention outside the physiographic envelope

D5 measured what the earlier work inferred: half the British and American test
units sit outside the training regions' terrain envelope, and the American fleet
reaches 22 standard deviations from its nearest training analogue. A correction
fitted on one envelope and applied far outside it is extrapolation whatever its
functional form, and `method-ml-transfer.md` named "predict only where training
data covers the local envelope, abstain elsewhere" as future work. D5 supplies
the measurement that makes it implementable.

**The rule, fixed here.** For each unit, `d` is the distance in standardised
terrain-feature space to the nearest TRAINING unit. `d0` is the 95th percentile
of the training fleet's own within-training nearest-neighbour distance -- a
scale the training set calibrates for itself, with nothing tuned on the holdout.
The terrain terms are damped by

    w = exp(-max(0, d - d0)^2 / d0^2),   gamma_eff = w*gamma,  delta_eff = w*delta

so `w = 1` anywhere inside the envelope and decays outside it. Only the terrain
terms are damped. Conversion efficiency, the sub-daily spread and air density
are not: losses and thin air do not stop existing because the terrain is
unfamiliar, and the fleet features are well covered everywhere.

**Predictions.** (1) The worst per-region degradation relative to uncorrected
ERA5 shrinks. (2) Regions with high coverage (DK 93%, DE 89%) move by less than
0.002 RMSE, because `w` is essentially 1 throughout. (3) The mean skill across
regions does not fall. Failing (2) would mean the gate is firing where it should
be dormant, and would invalidate the rule as specified.

### E5: capacity sensitivity, linear heads against an MLP

The heads are linear by default: 37 parameters in total. The same model with
two 16-wide hidden layers is a genuine neural network and a fair question --
whether the extra capacity buys anything.

**Prediction:** the MLP improves the in-region arm and is equal or WORSE on
zero-shot transfer, because added capacity is spent on within-region structure
that D0 already showed does not carry across regions. If the MLP transfers
better, the linear default is wrong and will be changed.

Reported as a sensitivity. It cannot promote or demote any gate in P1-P3, which
are scored on the linear model as committed.

### Addendum 4a: clarification to E4's threshold (before E4 was run)

`d0` was specified as "the 95th percentile of the training fleet's own
within-training nearest-neighbour distance". Implemented over raw rows that is
degenerate: German rows are postcode centroids and British rows are farm
generation split across turbines, so **52% of those distances are exactly
zero** and the 95th percentile comes out at 0.97 instead of the 2.38 that
distinct sites give. The correction would have been damped more than twice as
hard as the training data's real spacing warrants.

The reference set is therefore deduplicated before the threshold is taken. This
is the specification's evident intent -- the spacing between distinct sites --
not a change of rule, and it is fixed here before E4 has been run even once.

---

## Addendum 5: E7, nine regions instead of five (registered before running)

`method-ml-transfer.md` argued that the binding constraint is regime coverage
and that the remedy is data acquisition. Four more regions are already in this
repository -- New Zealand, Chile, Argentina and the Australian NEM -- and the
caches are now built for all of them. E7 asks whether training on eight regions
instead of four improves zero-shot transfer to the original five.

**Coverage first, measured before any fit.** Adding the four moves the
physiographic coverage of the original five only slightly:

| holdout | coverage, 5 regions | coverage, 9 | median NN, 5 | median NN, 9 |
|---|---|---|---|---|
| DK | 93.0% | 93.0% | 0.276 | 0.267 |
| DE | 89.4% | 89.8% | 0.246 | 0.242 |
| BR | 70.5% | 70.5% | 1.602 | 1.448 |
| UK | 50.5% | **53.4%** | 2.116 | 1.966 |
| US | 49.8% | **52.8%** | 1.439 | 1.306 |

Three points for the UK and the US, nothing for the rest. The four new regions
are themselves split: Argentina (91.5%) and Australia (89.2%) sit inside the
existing envelope, while New Zealand (33.3%) and Chile (33.9%) are further
outside it than any original region, Chile reaching a 90th-percentile
nearest-neighbour distance of 16.1.

**Prediction.** Zero-shot transfer to the original five improves in at most two
of them, and by less than 0.005 RMSE where it does. The reasoning: coverage is
what binds, coverage barely moved, and the two regions that add genuinely new
physiography are the two smallest fleets in the study -- New Zealand has 12
units and Chile 47.

**What would falsify it.** A larger or broader improvement would mean fleet
count and physics-sample count matter in their own right, beyond envelope
coverage, and D5's framing would need revising.

**Weighting stays as in E1**: regions weighted equally regardless of size. This
is now a stronger assumption -- New Zealand's 12 farms would carry the same
weight as Denmark's 3,692 turbines -- so a size-tempered variant is reported
alongside as a sensitivity. Neither can promote or demote P1-P3, which are
scored on the five-region model.

---

## Addendum 6: the seeds were doing nothing (found mid-run, fixed before E1 completed)

E1's first completed arm printed `pinn RMSE 0.0990 +- 0.0000` across five seeds.
The zero is not luck. The head weights were initialised to exactly zero with a
fixed bias, and the gradient is full-batch, so nothing in the fit was random:
every seed returned the identical model, and five seeds cost five times the
compute to re-measure one number.

This is a defect in the pre-specification, not only in the code. Seeds were
specified "as in the published experiment", where they varied a RandomForest;
carried over to a deterministic optimisation they measure nothing, and quoting a
spread of 0.0000 would have implied a robustness that had never been tested.

**The fix.** Head weights are now initialised from a small Gaussian
(std 0.02) with the bias still at the stated physical starting value, so the fit
still begins at the identity correction but from a different point each seed.
The spread across seeds now measures what a spread should: whether the optimum
is unique -- a live question, since D6 found flat directions in the loss.

Verified on Brazil in-region: RMSE 0.0955, 0.0951, 0.0954, 0.0952, 0.0957 across
the five seeds, sd 0.00023. Small, which is the reassuring answer, but now it is
an answer rather than an artefact.

E1 is rerun from the start under this change; no result from before it is
reported.

---

## Addendum 7: E8, combining the two things that worked (POST-HOC, labelled as such)

Registered after reading E5 and E7, and therefore **not** a pre-specified test.
It is recorded here so the record shows plainly which results were forecast and
which were not.

Two independently-established findings point the same way and neither was
predicted:

- **E7**: training on nine regions rather than five improved zero-shot transfer
  in 5 of 5 regions (mean skill +0.238 to +0.318). My registered prediction --
  at most two regions, under 0.005 RMSE -- failed, and its stated falsification
  condition was that D5's envelope-coverage framing would need revising.
- **E5**: MLP heads beat linear heads on zero-shot transfer in 5 of 5 (mean
  0.1068 to 0.1022) while being no better in-region. My registered prediction
  was the exact inverse, and its stated consequence was that the linear default
  is wrong and should change.

E8 runs both together. This is confirmation of two established effects in
combination, not a search over configurations: no other combination is being
tried, and the result cannot promote or demote P1-P3, which stand on the
five-region linear model as specified.

**What it would take to be believed.** E8 must beat BOTH E5 and E7 alone, in at
least 3 of 5 regions each. If it does not, the two effects are not additive and
the honest statement is that one of them is doing the work.

---

## Addendum 8: E9, a physically-shaped wake term (registered before implementing)

The residual anatomy (`d7`) found the model over-predicts by **+0.051 capacity
factor** in the densest bin of local capacity density (above 0.75 MW/km2),
against roughly +0.004 in the middle bins. Efficiency currently depends on
capacity density through a smooth learned function of standardised features, and
that dependence is evidently too weak at the top end. This is physics the model
approximates badly rather than physics it lacks, so the fix is the right SHAPE,
not more flexibility.

**Form.** Array efficiency multiplies the rest of the efficiency term:

    eta = eta_base(offshore, hub height, 50 km density) * 1 / (1 + c * D)

where `D` is local capacity density in MW/km2 within 10 km and `c >= 0` is a
single global coefficient -- the only free number the term adds.

**A deviation from what section 4 of `method-physics-informed-evaluation.md` proposed, and why.**
That said `exp(-c*D)`. Observed densities run to 9 MW/km2, where an exponential
with any `c` large enough to matter at the median (0.14) collapses the
efficiency to near zero. Deep-array physics saturates rather than collapsing --
an infinite wind farm reaches an asymptotic efficiency set by the momentum flux
it can draw down, which is the Frandsen picture -- and `1/(1 + cD)` has that
shape while `exp(-cD)` does not. Recorded here rather than changed quietly.

**Predictions, each able to fail:**

1. The mean residual in the densest capacity-density bin falls in magnitude,
   from +0.051 toward zero.
2. In-region RMSE pooled over the five regions improves or is unchanged
   (within 0.001).
3. Zero-shot transfer improves in **>= 3 of 5** regions against the same
   configuration without the term.

**What would falsify the diagnosis.** If prediction 1 fails, the dense-fleet
over-prediction is not a density effect and the residual anatomy has been
misread; the term should then be removed rather than retuned.

**Scored against configuration D** (MLP heads, nine training regions), which is
what would be carried forward, with three seeds. D is itself post-hoc, so this
is a comparison between two post-hoc configurations and cannot promote or demote
P1-P3.

### Addendum 8a: an implementation gap in E9, found from its own result

The first E9 run confirmed prediction 1 -- the density-slice residual span fell
from 0.0561 to 0.0219 -- but only scraped prediction 2, with pooled in-region
RMSE rising 0.07795 to 0.07870 against a 0.001 tolerance, and it pushed error
onto other axes (hub-height span 0.0455 to 0.0558, land-fraction 0.0386 to
0.0488) while raising the overall bias from +0.0108 to +0.0176.

The cause is a gap between what addendum 8 specified and what was built.
Addendum 8 says `eta = eta_base(offshore, hub height, 50 km density) * 1/(1+cD)`
-- the 10 km density withheld from the head. It was not withheld. The model
therefore had two routes to the same effect, a learned function of the
standardised density and a physical function of the raw one, and could fit the
physical term while quietly undoing it in the head.

The density is now withheld, as specified. Both E9 runs are repeated from
scratch under the corrected implementation and the first run's numbers are
reported only as the record of this error, never as the result.

---

## Addendum 9: E10, profile curvature (registered before implementing)

`method-physics-informed-evaluation.md` ranked this the second improvement lead
and described the evidence as "the model over-predicts at BOTH hub-height
extremes and is unbiased in the middle, which is the signature of a power law
fitted between 10 m and 100 m being extrapolated outside that range."

**That reasoning was wrong, and the check that shows it took ten minutes.** A
power law fitted on 10-100 m and extrapolated does not err in the same direction
at both ends. Against the log profile that generated it, it gives LESS wind
below 100 m and MORE above:

| z0 (m) | h=20 | h=30 | h=45 | h=80 | h=120 | h=150 |
|---|---|---|---|---|---|---|
| 0.0002 | −0.40% | −0.46% | −0.41% | −0.15% | +0.15% | +0.36% |
| 0.03 | −1.21% | −1.38% | −1.21% | −0.45% | +0.42% | +1.01% |
| 0.30 | −2.80% | −3.12% | −2.68% | −0.97% | +0.91% | +2.17% |

So curvature can explain the over-prediction above 120 m and **predicts the
opposite sign** below it. Whatever drives the low-hub-height residual, it is not
this.

**The apparent trend is also largely region composition.** Splitting the
residual by height AND region:

| height | share DE | share DK | share UK | DE | DK | UK |
|---|---|---|---|---|---|---|
| 0-40 m | 0% | 64% | 36% | -- | +0.014 | +0.034 |
| 40-60 m | 7% | 63% | 30% | +0.006 | −0.005 | −0.022 |
| 60-80 m | 31% | 11% | 54% | +0.012 | −0.032 | +0.033 |
| 80-100 m | 54% | 14% | 26% | +0.015 | −0.022 | +0.075 |
| 100-120 m | 89% | 1% | 10% | +0.005 | −0.008 | 0.000 |
| **120-200 m** | **91%** | 0% | 9% | **+0.037** | -- | +0.028 |

Within regions there is no monotone height trend: Denmark runs
+0.014, −0.005, −0.032, −0.022, −0.008 and Germany sits near +0.01 until its top
bin. The pooled "trend" is mostly which region occupies which bin. The one place
a genuine height signal survives is the 120-200 m bin, **91% German**, at
+0.037 -- and that is exactly where curvature has the right sign.

This is the E9 lesson applied before spending the day rather than after:
residual structure ranked this second by SIZE, and most of that size is
composition.

**The fix, which is still worth making.** The power law has the wrong curvature
in ln z; the log law has the right one by construction. The current model uses
the power law because its exponent is measured hourly and so responds to
stability, which a static roughness cannot. Both can be had: the measured
exponent inverts to an effective roughness in closed form,

    z0_eff = exp( ln(10) * (r - 2) / (r - 1) ),    r = w100/w10 = 10**shear

and the log law is then applied with that time-varying roughness. The learned
correction changes meaning from a shear-exponent offset to a **log-roughness
offset**, which is the more physical quantity of the two and stays equally
bounded.

**Predictions, judged on TRANSFER, not on the in-region residual:**

1. Zero-shot held-out RMSE improves for **Germany** by more than for any other
   region, Germany having by far the tallest fleet.
2. Mean zero-shot skill across the five regions improves or is unchanged
   (within 0.005).
3. The 120-200 m residual bin falls in magnitude from +0.037.

**Falsification.** If (1) fails, the tall-turbine residual is not curvature and
the change should be reverted rather than retuned. Given the effect is 1-2% of
wind speed over about 2% of rows, the honest prior is that this is **small**;
it is being done because it is correct and nearly free, not because it is
expected to move the headline.

Scored against configuration D (MLP heads, nine training regions), three seeds.

---

## Addendum 10: E11, joint constraint (registered before implementing)

Three results now point the same way. D6 found the efficiency and speed-up terms
trading off along a nearly flat direction. E9 constrained the efficiency's
density dependence: it fixed that systematic and pushed error onto other axes,
costing transfer in 5 of 5. E10 constrained the profile to physically realisable
shapes: same pattern, and it identified what was surrendered as profile shapes
no physical roughness can produce.

The common reading is that the model carries **compensating degrees of freedom**,
so constraining one term relocates the error rather than removing it. If that is
right, constraining SEVERAL at once should behave differently from constraining
one. If it is wrong, the freedom is being used productively and taking it away
will simply make the model worse, monotonically.

**The knob.** One scalar `lambda` shrinks every bounded quantity toward its
initial value: for a bound `(lo, hi)` with starting value `v0`,

    lo' = v0 + lambda * (lo - v0),    hi' = v0 + lambda * (hi - v0)

`lambda = 1` is the model as it stands. `lambda -> 0` collapses it to no speed-up,
no shear correction, a fixed 0.90 efficiency and a fixed 0.5 spread factor.
Applied to all four of gamma, delta, eta and kappa at once, which is what makes
this a JOINT constraint rather than a fourth single-term experiment. It is a
one-parameter family, so this is a sweep, not a search.

**Protocol.** Configuration A -- linear heads, five-region training pool -- since
that is the headline after the fresh gate. Sweep `lambda` in {1.0, 0.6, 0.35}
over the original five holdouts, two seeds (the seed spread is 0.0002-0.0014, so
two is ample). Then take whichever lambda wins the sweep and validate it on the
four fresh regions against `lambda = 1`, three seeds. Sweeping on the regions
already used and confirming on the unused ones is deliberate: it keeps the
confirmation independent of the choice.

**Predictions:**

1. Mean zero-shot skill over the five is **maximised at lambda < 1**. This is the
   hypothesis; it is what "the model has too much freedom" predicts.
2. The winning lambda beats `lambda = 1` on the fresh four in **at least 3 of 4**.
3. Skill falls again at the smallest lambda, since 0.35 leaves little room for a
   real correction. A monotone rise all the way down would mean the correction
   is barely earning its parameters at all, which is a different and more
   uncomfortable finding.

**Falsification.** If skill is maximised at `lambda = 1`, the compensating-freedom
reading of D6, E9 and E10 is wrong: the freedom is productive, those three
results need another explanation, and no amount of joint constraining will help.
That is the outcome I would bet against but it is the one the last two
experiments would have predicted, each having lost skill when freedom was
removed.
