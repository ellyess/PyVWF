# Pre-specification: physics-informed ERA5 wind bias correction

Written **before** any model in this workstream was fitted. Diagnostics D0-D2
(`pinn_diagnostics.md`) are measurements of existing artefacts and carry no
gate; everything from D3 onward involves choices that could be tuned to the
evaluation, so the gates are fixed here first.

## The question

The affine correction `w' = a*w + b` is fitted per cluster per region and does
not transfer to unseen regions (`ml_transfer_retest.md`, 1/5 under its gate).
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
  form (consistent with `ml_transfer_retest.md`).
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
functional form, and `ml_transfer_retest.md` named "predict only where training
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

`ml_transfer_retest.md` argued that the binding constraint is regime coverage
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
