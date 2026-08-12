# Cluster sweep and ML transfer re-test

**Date:** 2026-07-24
**Scope:** (1) a cluster-count sweep over the nine turbine-level regions to find
the best spatial resolution per region; (2) a pre-registered re-test of ML
transfer with the new complex-terrain regions (NZ, CL, AR) added.

## Part 1: cluster sweep

Each region's affine correction was fit at a range of cluster counts (one train
run, all `k` at once) and scored on the test year. RMSE on the test fleet,
`fixed` slice:

| Region | uncorr | RMSE by k (fixed) | knee | best |
|---|---|---|---|---|
| NZ | 0.157 | 1:.143 4:.111 5:.109 6:.108 **7:.106** | 5 | 7 (ceiling) |
| CL | 0.123 | 1:.139 **3:.236 5:.236** 8:.106 12:.108 18:.145 25:.144 | 8 | 8 |
| AR | 0.151 | 1:.152 8:.134 12:.132 25:.131 **35:.118** | 12 | 35 (edge) |
| AU-NEM | 0.115 | 5:.109 8:.100 12:.099 **20:.097** 30:.098 45:.099 | 15 | 20 |
| BR | 0.139 | 5:.135 10:.116 20:.110 **60:.105** 90:.110 | 20 | 60 |
| US | 0.110 | 5:.104 10:.101 25:.101 100:.101 **250:.098** | 25 | 250 (marginal) |
| DK | 0.147 | 10:.093 50:.091 **100:.088** 200:.090 | 100 | 100 |
| UK | 0.145 | 10:.129 25:.124 **50:.115** 100:.123 200:.124 | 50 | 50 |
| DE | 0.086 | 10:.062 25:.059 50:.059 **100:.057** 200:.059 | 50 | 100 |

Reading the curves rather than a single winner (one test year makes the
minimum noisy):

- **Several regions were under-clustered.** UK's optimum is `k=50`, not the
  `k=100` used previously (100 overfits: 0.123 vs 0.115). US plateaus from
  `k=10`; the drop to `k=250` is 0.003 and courts the fake-plateau. BR's knee is
  `k=20`. The shipped configs mostly hardcoded `k=10`.
- **CL is pathologically unstable.** At `k=3` and `k=5` the RMSE *doubles* to
  0.236. That is the northern extreme-scalar trap: at low `k` the Atacama is
  lumped with central Chile and the shared wind scalar explodes to fit the
  desert's ERA5 under-prediction. Only `k=1` (one national cluster) or `k≥8`
  (the Atacama gets its own cluster) are safe. This is a red flag for CL as a
  finely-clustered region.
- **NZ hits its fake-plateau ceiling** at `k=7` (8 farms reach the trainer), so
  its "best" is one-farm-per-cluster; `k=5` is the honest choice.
- **AR keeps improving to the grid edge** (`k=35`), but that is overfitting a
  59-farm fleet; `k=12` is the stable knee.

**Recommended `cluster_list`** (national baseline + the knee), applied to the
configs: NZ [1, 5], CL [1, 8], AR [1, 12], AU-NEM [1, 15], BR [1, 20],
US [1, 25], DK [1, 100], UK [1, 50], DE [1, 50].

## Part 2: ML transfer re-test (pre-registered)

The 2026 re-test ([ml_transfer_retest.md](ml_transfer_retest.md)) found
leave-one-region-out (LORO) transfer negative (1/5) and traced the one
catastrophic case, US (R² −1.05), to the mountain-pass regime the Europe-only
pool had no analogue of. This session added NZ, CL and AR -- the same class of
complex-terrain, ERA5-under-resolves-the-ridge-wind regime. The hypothesis:
adding those analogues should rescue the US holdout.

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

Scripts: `scripts/analysis/ml_transfer_expanded.py` (this experiment),
`ml_transfer_retest.py` (the 5-region original).
