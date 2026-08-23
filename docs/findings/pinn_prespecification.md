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
