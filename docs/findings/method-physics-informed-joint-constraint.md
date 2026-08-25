# E11: constraining all four terms at once. The hypothesis holds, the value does not.

Registered in addendum 10 before implementing, with three predictions. Two
confirmed, the confirmation gate failed, and the result is genuinely mixed in a
way worth keeping separate from a clean win.

## Why the test existed

Three results had pointed the same way. D6 found efficiency and speed-up trading
along a nearly flat direction. E9 constrained the efficiency's density
dependence and E10 constrained the profile shape; each fixed the systematic it
aimed at (−61%, −62%) and each **cost transfer in 5 of 5 regions**, having
pushed the error onto other axes rather than removing it.

If that reading is right, the model carries compensating freedom and squeezing
one term relocates error instead of removing it. Squeezing them **all at once**
should then behave differently. One scalar `lambda` shrinks every bounded term
toward its starting value: `lambda = 1` is the model as it stands.

## The sweep: predictions 1 and 3 CONFIRMED

Five original regions, configuration A, two seeds.

| holdout | uncorrected | λ=1.00 | λ=0.60 | λ=0.35 |
|---|---|---|---|---|
| BR | 0.1386 | 0.1071 | **0.1069** | 0.1122 |
| DE | 0.0860 | 0.0791 | 0.0741 | **0.0705** |
| DK | 0.1464 | 0.0989 | 0.0948 | **0.0943** |
| UK | 0.1451 | 0.1432 | **0.1422** | 0.1499 |
| US | 0.1098 | 0.1071 | 0.1073 | **0.1047** |
| **mean skill** | -- | **+0.2347** | **+0.2655** | **+0.2559** |

Mean skill peaks at **λ = 0.60**, not at λ = 1 (prediction 1), and falls again at
λ = 0.35 (prediction 3). An interior optimum: too little freedom leaves no room
for a correction, too much and the terms absorb one another.

**Removing 40% of the model's freedom from all four terms improves zero-shot
transfer by 13% relative.** Constraining one term made things worse twice;
constraining all four makes them better. That is what compensating degrees of
freedom predict, and it is the resolution of the E9/E10 puzzle: those were not
failures of wake or curvature physics but symptoms of a model with too much
slack overall.

## The confirmation: prediction 2 FAILED

λ = 0.60 was chosen on those five regions, so its margin there is not
independent. Re-tested on the four regions never used for any choice:

| holdout | uncorrected | λ=1.00 | λ=0.60 | change |
|---|---|---|---|---|
| AR | 0.1512 | 0.1294 | **0.1281** | −0.0013 |
| AU-NEM | 0.1153 | **0.0788** | 0.0807 | +0.0019 |
| **CL** | 0.1225 | 0.1279 | **0.1161** | **−0.0118** |
| NZ | 0.1567 | **0.0848** | 0.0913 | +0.0065 |

**2 of 4, against the 3 of 4 required. Prediction 2 fails**, and λ = 0.60 is
therefore not established as uniformly better on data it was not chosen against.

## What survives the failed gate, stated carefully

The registered gate was a per-region count and it failed; that governs. But three
things are true alongside it and should not be buried:

- **Mean skill improves on the fresh four too**, +0.354 to +0.389. The direction
  replicates on data that had no part in choosing λ, even though the per-region
  count does not.
- **The gain is concentrated where the model was doing damage.** Chile is the
  only region in the whole study that λ = 1 makes *worse* than no correction
  (skill −0.091). At λ = 0.60 it becomes a gain (+0.102). The other three fresh
  regions move by less than 0.007 either way.
- **Across all nine regions**: λ = 0.60 wins 6 of 9, lifts mean skill +0.288 to
  +0.320, and **harms no region at all**, where λ = 1 harms one. The worst region
  goes from −0.091 to +0.039.

So the honest reading is that joint constraint behaves like **insurance rather
than a uniform gain**: it costs a little where the model was already doing well
(NZ −0.047 skill, AU-NEM −0.023) and rescues the case where unconstrained
freedom was letting it extrapolate badly.

## Status

- **The compensating-freedom hypothesis is supported.** Prediction 1 confirmed
  on the five, and the mean effect replicates on the fresh four. This is the
  first thing in the programme to make transfer better by taking capability
  away, and it explains why E9 and E10 failed.
- **λ = 0.60 is not adopted.** The registered confirmation gate failed at 2 of 4.
  Configuration A at λ = 1 remains the headline.
- **The per-region optimum varies and is unexplained**: DE, DK and US prefer
  0.35 in the sweep; BR and UK prefer 0.60. A single global λ is the wrong shape
  if that variation is real rather than noise, and nothing here distinguishes
  those.

## What would settle it

A sweep finer than three points, chosen and confirmed on disjoint region sets,
with enough regions that the per-region variation can be separated from noise.
Nine regions cannot do that. The cheaper intermediate question, not run: whether
the optimum tracks anything measurable about a region, given that two candidate
predictors of gain have already been refuted
(`method-physics-informed-predictors.md`).
