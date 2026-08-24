# Denmark onshore: cluster-count and time-resolution sweep

**Date:** 2026-07-24
**Scope:** the paper's research grid, reproduced on the harness for Denmark
**onshore only** with the combined (open + licensed) turbine-curve library and
real turbine matching (no uniform default curve).

- `n_clu` in {1, 2, 3, 5, 7, 10, 20, 50, 70, 100, 200, 500, 700, 800, 900,
  1000, 2000, 3000, 3300}
- `t_freq` in {month, bimonth, season, fixed}
- Fleet: 4866 onshore turbines reach the trainer (the 630 offshore turbines are
  dropped), so every `k` up to 3300 is a real cluster count, not a
  one-turbine-per-cluster plateau.
- Train 2015-2019, test 2020. Monthly capacity factor, `european-turbine`
  source restricted to onshore.

## Result

Uncorrected baseline (constant across `k`): RMSE 0.1583, MAE 0.1234,
MBE +0.1206, r 0.753.

RMSE on the 2020 test fleet, by cluster count and time resolution:

| k | month | bimonth | season | fixed |
|---|---|---|---|---|
| 1 | 0.0962 | 0.0957 | 0.0956 | 0.0981 |
| 2 | 0.0987 | 0.0981 | 0.0980 | 0.1007 |
| 3 | 0.0990 | 0.0982 | 0.0981 | 0.1008 |
| 5 | 0.0962 | 0.0955 | 0.0953 | 0.0982 |
| 7 | 0.0968 | 0.0960 | 0.0958 | 0.0987 |
| 10 | 0.0971 | 0.0963 | 0.0961 | 0.0990 |
| 20 | 0.0956 | 0.0947 | 0.0945 | 0.0975 |
| 50 | 0.0926 | 0.0917 | 0.0914 | 0.0944 |
| 70 | 0.0921 | 0.0912 | 0.0909 | 0.0939 |
| 100 | 0.0898 | 0.0888 | 0.0886 | 0.0916 |
| 200 | 0.0872 | 0.0861 | 0.0857 | 0.0886 |
| 500 | 0.0879 | 0.0866 | 0.0861 | 0.0888 |
| 700 | 0.0877 | 0.0864 | 0.0859 | 0.0884 |
| 800 | 0.0874 | 0.0860 | 0.0855 | 0.0881 |
| 900 | 0.0876 | 0.0860 | 0.0855 | 0.0881 |
| 1000 | 0.0878 | 0.0863 | 0.0857 | 0.0883 |
| 2000 | 0.0876 | 0.0859 | 0.0851 | 0.0877 |
| 3000 | 0.0876 | 0.0859 | 0.0851 | 0.0876 |
| 3300 | 0.0876 | 0.0859 | 0.0851 | 0.0876 |

Best: `k=3300`, `season`, RMSE 0.0851 (MAE 0.0533, MBE +0.0282, r 0.824).

## Reading

- **Cluster count is the dominant lever, and it saturates early.** RMSE falls
  from ~0.096 (`k=1`) to a floor at **`k=200`** (0.0857, season). Going from 200
  to 3300, a 16x increase in clusters, buys **0.0006 RMSE** (0.0857 -> 0.0851).
  For a working correction, `k` around 100-200 captures essentially all the
  skill; the rest of the grid is confirmation, not gain.
- **Time resolution is a minor lever.** `season` ~ `bimonth` < `month` < `fixed`
  at every `k`, but the whole spread is ~0.003 RMSE. `fixed` (one annual factor)
  is the weakest; `season` is best. `month` is slightly *worse* than `season`,
  i.e. the extra temporal freedom over-fits monthly noise rather than adding
  skill. This matches the paper's headline that `n_clu` dominates `t_freq`.
- **A low-`k` clustering artifact.** `k=2` and `k=3` (0.098) are slightly worse
  than `k=1` (0.096): splitting the onshore fleet into two or three clusters
  partitions it badly before finer clustering recovers from `k=5` on. The
  national single-cluster fit is a reasonable floor; 2-3 clusters are not.
- **What the correction does.** It roughly halves RMSE (0.158 -> 0.085), cuts
  the mean bias from +0.121 to +0.028, and lifts correlation from 0.753 to
  0.824. Denmark onshore is a well-behaved region: ERA5's bias here is close to
  a clusterable wind-speed offset, which is exactly what the affine model
  represents (unlike the Atacama/Patagonia regimes in
  [region-south-america.md](region-south-america.md)).

**Practical setting for DK onshore:** `cluster_list = [1, 200]`, `season` (or
`bimonth`). Beyond `k=200` the model is not wrong, just not better.

## Compute note

The harness offset fit was single-core (a deliberate bit-for-bit delegate to
`PyVWF.train(dask_n_workers=0)`). This run added the parallel branch from the
legacy `PyVWF.train` to the harness (`vwf.harness.corrections`), opt-in via
`PYVWF_OFFSET_WORKERS`, default sequential so the golden regression test is
untouched. At 4 workers the result is bit-identical to sequential (verified at
`k=50`, max abs metric diff 0.0) and the expensive tail runs ~4x faster
(`k=3300` in ~18 min; the sequential `k=200` run alone was still going past 10
min). Whole grid: ~1.5 h.

Scripts and data: per-`k` metrics under
`output/validation/dk_onshore_sweep_2026-07-24/` (`combined_metrics.csv` is the
merged table; the heavy per-`k` CF frames are pruned, factors and metrics kept).
