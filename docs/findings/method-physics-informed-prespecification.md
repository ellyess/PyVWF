# Physics-informed correction: registered gates and predictions

The audit record for `method-physics-informed.md`. Every gate and prediction
below was fixed before the run it governs, together with the consequence of its
failure. The outcome column was filled in afterwards.

## Metric

    skill = 1 - MSE(method) / MSE(uncorrected ERA5)

Skill against declining to correct, not R-squared against the holdout region's
own mean: that mean is unavailable to a practitioner arriving in an unseen
region, so an R-squared built from it scores a question nobody can act on. Both
are reported where the comparison with `method-ml-transfer.md` needs them.

## Gates

| gate | requirement | outcome |
|---|---|---|
| **P1** primary | Leave-one-region-out over {DK, DE, UK, US, BR}, zero-shot: corrected RMSE < uncorrected in >= 3 of 5, **and** no region degraded by more than 10% relative | **PASS**, 5 of 5, none degraded |
| **P2** comparative | Beats the incumbent RandomForest transfer of the per-cluster scalar in >= 3 of 5 | **PASS**, 4 of 5 |
| **P3** ablation | An unconstrained model at matched features and parameter count must be run and reported. If it matches, the physics is decorative | **PASS**, ablation is far worse than doing nothing in every region |
| **Q1** fresh | Same form as P1 on four regions never used as holdouts {NZ, CL, AR, AU-NEM} | **PASS**, 4 of 4 |
| **Q2** fresh | The post-hoc configuration (MLP heads, nine training regions) beats the gated one in >= 3 of 4 | **FAIL**, 1 of 4. Headline reverts to the gated linear five-region model |

P1's second clause is part of the gate, not a footnote: a method offered as
general must not badly damage the regions it fails on.

Falsification conditions, stated in advance: P1 fails means the parameterisation
does not generalise either and the honest headline is that the constraint is
data coverage; P1 passes but P2 fails means the gain is feature scale, not
physics; the ablation matching means the same.

Committed in advance: regions, cluster counts and train/test years are those of
the canonical runs already on disk, with no region dropped after seeing results;
any configuration explored beyond the pre-specified one is a sensitivity and
cannot promote a failed gate.

## Registered predictions

Each was recorded before the run that tested it. Seven of thirteen failed, and
the failures carried more information than the successes.

| # | prediction | outcome |
|---|---|---|
| 1 | The scalar rises with elevation above the ERA5 cell mean, consistently across regions | **FAIL**. Region-inconsistent, wrong sign in Germany. Cell relief holds instead |
| 2 | A multi-scale terrain set beats the published one in >= 3 of 5 **and** improves both failing regions | **FAIL** as a conjunction. 3 of 5, but the UK collapses |
| 3 | The UK advantage over the RF baseline is at least the median across regions | Satisfied by identity: the UK **is** the median region. Not counted as support |
| 4 | Air density lowers the fitted speed-up above 800 m, which is currently absorbing it | **FAIL**, and instructively: it rose 1.462 to 1.537, which is the physically correct direction, and moved ten times more above 800 m than below 200 m |
| 5 | Density changes near-sea-level fleets by < 0.002 RMSE | PASS |
| 6 | Abstention outside the terrain envelope shrinks the worst per-region degradation and does not lower mean skill | **FAIL**. Mean skill fell, +0.238 to +0.235 |
| 7 | MLP heads help in region and are equal or worse zero-shot | **FAIL**. Better zero-shot in 5 of 5, no better in region |
| 8 | Nine training regions improve transfer in at most 2 of 5, by < 0.005 RMSE | **FAIL**. 5 of 5, mean skill +0.238 to +0.318 |
| 9 | Combining the two (post-hoc, labelled as such) beats each alone in >= 3 of 5 | PASS, 3 of 5 and 4 of 5 |
| 10 | A wake term cuts the dense-fleet residual | PASS, span -61% |
| 11 | The wake term improves transfer in >= 3 of 5 | **FAIL, 0 of 5.** Term not adopted |
| 12 | Profile curvature helps Germany most, and holds mean transfer skill within 0.005 | Half PASS: Germany improves and is the only region that does; mean skill falls 0.0089. Term not adopted |
| 13 | Constraining all four terms jointly beats no constraint, and confirms on the fresh four in >= 3 of 4 | Sweep PASS with an interior optimum at 0.60; **confirmation FAIL, 2 of 4**. Not adopted |

Predictions 8 and 7 together revised the framing that motivated the work: more
regions help by pinning the model's shared global constants, not by covering any
particular holdout's terrain, and constrained extra capacity cannot be spent on
region identity because the flat-ground pin and the bounds forbid it.

## Deviations, and when they were caught

Three changes were made mid-programme. Each is recorded because a change made
after seeing a transfer score is tuning, and one made before is not.

- **Fleet features.** The efficiency head had been given `log10(capacity)`,
  which runs 2.9 to 3.4 in Europe and 4.9 to 5.1 in the Americas because the
  observation unit differs, so it could identify the continent from a feature
  meant to describe a fleet. A second feature was absent for the US and Brazil
  and was being filled with a constant. Replaced with aggregation-invariant
  capacity density. Caught with only the uncorrected baselines printed and no
  arm completed; that run was discarded.
- **Seeds were doing nothing.** Zero-initialised heads plus a full-batch
  gradient made every seed return an identical model, and a reported spread of
  0.0000 would have implied a robustness never tested. Heads are now initialised
  from a small Gaussian with the bias still at the physical starting value. The
  experiment was rerun from the start and no earlier result is reported.
- **The abstention threshold.** Specified as the 95th percentile of the training
  fleet's within-training nearest-neighbour distance. Over raw rows that is
  degenerate: 52% of those distances are exactly zero, because German rows are
  postcode centroids and British rows are farm generation split across turbines.
  The reference set is deduplicated before the threshold is taken, which is the
  specification's evident intent rather than a change of rule, and it was fixed
  before the rule had been run once.
