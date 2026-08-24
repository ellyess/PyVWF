# ML transfer of correction factors

**Date:** 2026-07-24
**Scope:** can a model learn correction factors in some regions and predict
them in an unseen one? Two pre-registered rounds are recorded here: five
regions, then eight after adding complex terrain.

**Headline: the negative transfer result stands, in both rounds.** Under the
pre-specified gate (leave-one-region-out scalar R² > 0 in at least 3 of 5
regions, primary configuration), round one scores **1/5**: only Brazil is
predictable from the other regions. But three specific things moved relative to the
development-branch experiment, and they narrow *why* transfer fails.

Script: `scripts/analysis/ml_transfer_retest.py`. Gates were written down before any
model run (pre-specification reproduced at the bottom).

## Round one: five regions

### Why re-test at all

The development-branch experiment (`scripts/pyvwf_ml/ML_RESULTS_SUMMARY.md` on
`development`) trained on 1,474 Europe-only centroids and found
leave-one-country-out R² of −0.10 to −1.91. Two things have changed since:

1. **Targets are cleaner.** That experiment's k-means targets used
   `init="random"`: the seed lottery measured since (DK k=500 MAE spread
   0.0514 to 0.0842 across seeds). Its DE/DK/UK centroids were partitions
   drawn from that lottery. The targets used here use k-means++.
2. **The training set now spans climates.** DK/DE/UK (temperate maritime) +
   US (mixed continental) + BR (tropical trade-wind), all through the same
   pipeline with the scalar-dilution and Hawaii-bbox fixes in place.

### Dataset

Per-cluster **fixed** factors from the canonical post-fix runs (all at commit
`8a032d6`, external combined curve library, k-means++, no capacity weighting,
no geographic distance):

| Region | Run | k | Centroids |
|---|---|---|---|
| DK | train-sweep3 | 100 | 100 |
| DE | train-sweep3 | 100 | 100 |
| UK | train-sweep3 | 100 | 100 |
| US | train-sweep2 | 100 | 100 |
| BR | train-sweep2 | 60 | 60 |

Centroid = mean lon/lat of cluster members. Features are computed exactly as
the prior experiment: ETOPO-derived elevation/slope/aspect/roughness/curvature
(here from `etopo_global.nc`, 30 arc-sec, same formulas) plus abs_lat and
normalised lon/lat (Set A). Model: RandomForest(100, max_depth=10,
min_samples_split=10), identical hyperparameters. 5 seeds; mean ± std.

### T1 (primary, gated): leave-one-region-out, Set A

| Holdout | scalar R² | offset R² |
|---|---|---|
| BR | **+0.187 ± 0.023** | **+0.194** |
| DE | −0.016 ± 0.031 | −0.019 |
| DK | −0.226 ± 0.052 | −0.191 |
| UK | −0.197 ± 0.080 | −0.072 |
| US | **−1.049 ± 0.219** | −0.910 |

**Gate: 1/5 → NEGATIVE result stands.** Two honest observations alongside:

- **BR is the first region ever to transfer positively** in this programme's
  ML experiments, and it is the climatically most distinct one. Its
  corrections (mean scalar 1.37, pull-up) sit at the extrapolation edge the
  prior Europe-only dataset never probed.
- **DE/DK/UK holdouts are near zero, not catastrophic** (−0.02 to −0.23,
  vs −0.10 to −1.91 previously). The failure mode has softened from "worse
  than useless" to "no better than the holdout mean".

### Why the US fails: unrepresented terrain regimes, not noise

US holdout is the one catastrophic case (−1.05). The US target distribution
explains it: scalar 95th percentile 2.22, max 4.25, where all other regions
together span roughly 0.2–2.4. Every extreme US cluster is a complex-terrain
site: Tehachapi (scalar 3.30), Altamont (3.43), San Gorgonio area (2.84),
central Washington (4.25), all mountain-pass regimes where ERA5 badly
underestimates ridge winds. The European training fleet contains no analogue,
and 1-km terrain derivatives evidently do not encode "mountain pass" well
enough to extrapolate to it. Transfer fails hardest exactly where the
training set has no coverage of the physical regime, consistent with the D2
"corrections are regional bias fingerprints" result, but now with a mechanism.

### T2/T3: feature-set variants (informational)

- **Terrain-only (Set B)** is uniformly worse than Set A here (e.g. UK −1.57,
  DE −0.36), the opposite of the prior finding that dropping spatial
  features helped. With multiple continents in training, lon/lat stop being
  pure country-identity encodings.
- **Fleet features (Set C: hub height, log capacity, cluster size)** lift DE
  to **+0.12** and BR to +0.34, leave UK ~0, don't rescue DK/US. Hub-height
  metadata carries genuine cross-region signal, relevant to the
  Australia/NEM synthesis point that height metadata should be first-class.

### T4: pooled random 5-fold CV (comparability anchor)

Scalar **R² = 0.43 ± 0.12**, offset **0.48 ± 0.13**, up from 0.35/0.41 in
the prior experiment. Not strictly comparable (different regions, k, library),
but consistent with cleaner targets being more learnable. Within-region
structure is real and learnable; cross-region extrapolation is what fails.

### T5: variance decomposition

Between-region share of variance: scalar 18%, offset 19%. Transfer failure is
NOT "each region has a different mean": 80%+ of variance is within-region.
The model learns within-region structure when it has local data (T4) and
cannot extrapolate it across regions (T1).

### Sensitivity

- **BR k=120 instead of 60** (n=519): same verdict (gate 2/5: BR +0.30,
  UK creeps to +0.03). No qualitative change.
- **High-k tier** (DK/DE 500, UK 500→332 ceiling, US 300, BR 120; n=1,751):
  gate would read 3/5 (BR +0.23, DE +0.13, US +0.05; DK −0.20, UK −2.07),
  but this tier is NOT trustworthy as a positive: US k=300 targets have
  scalar std 2.67 (extreme-outlier-dominated), the pooled random-CV scalar R²
  collapses to −0.09 ± 0.90 under those outliers, and UK at its
  unique-coordinate ceiling has near-single-site clusters with noisy factors.
  Recorded as a density hint, not a result: with more centroids per region,
  cross-region prediction improves for the regions whose regimes overlap.

### What this means

1. **Did cleaner targets + climate diversity flip the negative result?** No.
   The pre-specified gate fails. Headline unchanged: correction factors are
   not predictable in unseen regions from terrain + location + fleet
   features at useful accuracy.
2. **Would new data help?** The failure pattern says the binding constraint is
   **regime coverage, not sample count**: BR transfers once anything like it
   exists in training; US fails because nothing like Tehachapi exists in
   training. New regions chosen for *regime* diversity (complex terrain,
   monsoon, Mediterranean) attack exactly this. That is an argument for
   data acquisition first, ML re-evaluation second.



Round one above traced its one catastrophic case, the United States at
R² −1.05, to a mountain-pass regime with no analogue in the training pool.
Round two tested the obvious remedy by adding New Zealand, Chile and
Argentina, the same class of complex-terrain regime where ERA5 under-resolves
ridge wind. The hypothesis was that those analogues should rescue the US
holdout.

Gates were written before any model run (`ml_transfer_expanded.py`): G1, US
holdout scalar R² rises above −0.30; G2, R² > 0 in ≥ 5 of 8 regions; G3, NZ/CL/AR
each transfer positively.

**The hypothesis is refuted, and informatively.** Adding NZ/CL/AR did not rescue
US; it made US and BR catastrophically worse (US −1.05 → −94.6, BR +0.19 →
−431). The cause is direct: **the new regions' correction factors carry
extreme-scalar artifacts** -- CL has a cluster with scalar **39.3**, AR one with
**15.5** -- the Atacama and La Rioja northern-wind blow-ups, 10 to 40× the normal
0.4–2 range. Those artifacts poison the training pool.

Capping the scalar at a physical 3.0 (what a sane bias correction would do)
removes the numerical poisoning but not the conclusion:

| holdout | 5-region | 8-region | delta |
|---|---|---|---|
| US | −0.41 | −1.12 | −0.72 |
| BR | +0.25 | −0.85 | −1.09 |
| NZ | -- | **+0.37** | new |
| CL | -- | **+0.67** | new |
| AR | -- | **+0.42** | new |

Two things are now clear:

1. **The new regions are internally predictable** (NZ/CL/AR holdouts +0.37 to
   +0.67): within a pool that already contains them, they interpolate.
2. **But adding them still hurts US and BR**, even capped. "Complex terrain" is
   not one regime: US mountain-pass funnelling, Atacama coastal desert, NZ
   island funnelling and Andean-foothill Cuyo have *different* terrain→bias
   relationships, and 1-km ETOPO features do not encode which is which. More
   complex-terrain regions add *conflicting* mappings, not better coverage.

**Verdict: the negative transfer result stands and sharpens.** The wall is not
"too few regimes in training." It is that in exactly these regimes the
correction factor is a **reanalysis-resolution artifact, not a transferable
physical bias** (the scalar-39 clusters are where ERA5's 0.25° grid fails, not
where a learnable bias lives). ML cannot transfer through a target dominated by
where the reanalysis breaks.

The actionable read is the same as the South-America finding: the fix for these
regimes is at the **wind** level (a higher-resolution wind product that removes
the extreme scalars at source), not a learned transfer of the correction
factors. ML-as-interpolation (train on all regions, apply within the covered
space) is defensible; ML-as-transfer to an unseen complex-terrain regime is not.

Scripts: `scripts/analysis/ml_transfer_expanded.py` (round two),
`ml_transfer_retest.py` (round one).

## Named, not evaluated

- ERA5-climatology features at centroids (mean wind, seasonal amplitude,
  Weibull shape), the prior experiment's `era5` feature group, never run on
  the multi-region set.
- Robust targets (log-scalar, or winsorised) for outlier-heavy fleets.
- A "regime overlap" gate: predict only where training data covers the local
  terrain/climate envelope, abstain elsewhere (the 5° distance mask
  generalised to feature space).

## Pre-specification (verbatim gates)

Primary: T1 = LORO, Set A, scalar, RF(100/10/10), 5 seeds. Gate: positive iff
holdout scalar R² > 0 in ≥ 3/5 regions on the primary configuration
(DK/DE/UK/US k=100, BR k=60). Sensitivity variants and other feature sets are
supporting material only and cannot flip the headline. R² per sklearn
`r2_score` (denominator = holdout's own mean), matching the prior experiment.
