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
