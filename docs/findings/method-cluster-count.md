# Cluster-count sweep across the turbine regions

**Date:** 2026-07-24
**Scope:** how test-year skill varies with the number of spatial clusters, for
each of the nine turbine-level regions. The Denmark grid is swept far more
finely, over four time resolutions, in
[method-cluster-count-dk.md](method-cluster-count-dk.md). The ML transfer
experiment that originally shared this document now lives in
[method-ml-transfer.md](method-ml-transfer.md).

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
  desert's ERA5 under-prediction. This is a red flag for CL as a
  finely-clustered region.

  **Correction (2026-08-24).** This document originally concluded that
  "only `k=1` or `k>=8` are safe". That is withdrawn. The fit-quality work
  ([method-scalar-bounds.md](method-scalar-bounds.md)) showed that CL at
  `k=10`, which the rule admits and which the scorecard reported, is itself
  degenerate: it carries a fitted wind scalar of 80.23 and one offset that
  never converged, while still scoring as a corrected win. Test-year RMSE is
  not a fit-health test, and cluster count does not predict fit health. Read
  `fit_quality`, which now travels with every corrected row in `metrics.csv`,
  rather than choosing `k` from this table alone.
- **NZ hits its fake-plateau ceiling** at `k=7` (8 farms reach the trainer), so
  its "best" is one-farm-per-cluster; `k=5` is the honest choice.
- **AR keeps improving to the grid edge** (`k=35`), but that is overfitting a
  59-farm fleet; `k=12` is the stable knee.

**Recommended `cluster_list`** (national baseline + the knee), applied to the
configs: NZ [1, 5], CL [1, 8], AR [1, 12], AU-NEM [1, 15], BR [1, 20],
US [1, 25], DK [1, 100], UK [1, 50], DE [1, 50].
