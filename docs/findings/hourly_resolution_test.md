# Does the correction survive at hourly resolution? (Chile, 2024)

**Date:** 2026-08-12
**Question:** every validated result in this repo scores monthly capacity
factors. Any use of the correction inside a forecasting system would score it
hourly. Does it hold at that timescale?

**Answer: no, and the first-order reason is not the one predicted.** All three
pre-registered gates failed. But the failure is dominated by three degenerate
clusters, not by a general property of the method, and removing them turns a
catastrophic result into a merely neutral one.

## The structural finding that came first

`prep_era5` unconditionally ran `ds.resample(time="1D").mean()`. **PyVWF had no
sub-daily path at all.** Every published result, across all nine regions, is
built on daily-mean wind; the hourly `era5/CL` and `era5/NZ` downloads were being
averaged away on load, and `BR_daily` / `US_daily` are merely pre-aggregated to
skip that cost.

This was found the hard way. The first run reported a plausible, prediction-
confirming result ("hourly RMSE 0.2912 -> 0.2800, correction helps a little"),
which was worthless: the simulation was daily, so the merge matched only
midnight observations and compared daily-mean simulated power against a single
hour's output. The monthly control caught it. Without that control the run would
have produced a fabricated confirmation of the hypothesis being tested.

`prep_era5` now takes `resample_daily`, defaulting to `True` so every existing
result and the golden regression path are unchanged.

## Method

Chile was chosen on a constraint fixed before any result was seen: the test needs
the correction to have been FITTED on hourly winds, or a failure could just be a
daily-to-hourly distribution mismatch. Only CL and NZ use hourly ERA5. Of those,
CL has 59 plants against NZ's 12, and CEN publishes a fixed UTC-4 with no DST,
whereas NZ's EMI data uses trading periods with 50-period DST days.

Trained correction: affine, k=10, fixed slice, matched curves
(`cl_matched_2026-07-24`), applied to held-out 2024 against the raw CEN hourly
series.

## Raw table

| arm | n | uncorr RMSE | corr RMSE | change | uncorr MBE | corr MBE | uncorr r | corr r |
|---|---|---|---|---|---|---|---|---|
| hourly (native) | 463,071 | 0.2263 | 0.2613 | **-15.5%** | +0.0361 | +0.0615 | 0.657 | 0.596 |
| hourly, degenerate clusters dropped | 431,213 | 0.2254 | 0.2298 | **-1.9%** | +0.0434 | +0.0443 | 0.670 | 0.658 |
| daily (from hourly power) | 20,474 | 0.1657 | 0.2048 | -23.6% | +0.0323 | +0.0735 | 0.721 | 0.626 |
| daily (published: daily wind) | 19,757 | 0.1790 | 0.1816 | -1.5% | +0.0094 | +0.0194 | 0.704 | 0.697 |
| monthly (from hourly power) | 672 | 0.1178 | 0.1637 | -38.9% | +0.0322 | +0.0738 | 0.555 | 0.286 |
| monthly (published: daily wind) | 654 | 0.1202 | 0.1105 | **+8.1%** | +0.0085 | +0.0184 | 0.508 | 0.507 |

Control: the published-method monthly row gives 0.1202 -> 0.1105 against the
canonical 0.1227 -> 0.1040. Exact agreement is not expected, because the
canonical run builds its observations through the processed `cl_obs.csv` path
(capacity overrides, commissioning-prefix stripping, exclusions) while this reads
raw CEN. Order of magnitude and sign agree, which is what the control is for.

## Gates

Pre-registered before running, with the prediction that G1 and G3 would pass and
G2 fail (the correction is a level correction; it should help a little at hourly
but not proportionally).

- **G1** (helps at all): 0.2613 vs 0.2263. **FAIL**
- **G2** (retains half the 15.3% monthly gain, so >= 7.6%): -15.5%. **FAIL**
- **G3** (bias reduced): |+0.0615| vs |+0.0361|. **FAIL**

**The prediction was wrong.** The correction does not merely fail to transfer to
hourly; applied as-is it makes the simulation worse on every metric.

## What actually causes it

Two explanations were separated by a pre-planned test. (a) A general effect: the
level correction acting through the convex power curve, which would hit every
plant. (b) A few degenerate clusters: this factors file contains scalars of
**80.2** (cluster 6), **28.7** (cluster 8) and 3.25 (cluster 2), the known
Atacama artifact. An 80x wind multiplier pins every hour above rated power.

Dropping those three clusters, 7% of rows, moves the hourly result from -15.5% to
**-1.9%**, and correlation from 0.596 back to 0.658 against an uncorrected 0.670.
So roughly **88% of the degradation comes from three degenerate clusters**, and
the general convexity effect is real but small.

The practical reading: at hourly resolution the correction is approximately
**neutral** once the degenerate fits are excluded, not catastrophic. It still
does not help, so the negative answer stands, but the cause is a fixable data and
QA problem rather than a fundamental limit of the method.

## The finding that matters beyond forecasting

**Monthly aggregation was masking degenerate factors.** A cluster fitted with a
wind scalar of 80 passes the monthly metric (CL is reported as a win, 0.123 ->
0.104) because averaging smears the damage. Nothing in the current pipeline
flags it.

That has an immediate consequence for the gridded export: `export_correction_field`
writes `scalar` per grid cell, so it would happily ship a cell with scalar 80 to
a downstream atlite pipeline. The existing confidence layers (`km_to_centroid`,
`cluster_n_train`) do not catch this, because the cluster can be well-sampled and
still degenerate. **A physical-plausibility check on export is warranted**, and a
plausibility warning at fit time would be better still.

## Secondary result: the raw physics improves at hourly

Uncorrected, the hourly simulation is a better model than the published daily one:
monthly RMSE 0.1178 against 0.1202, correlation 0.555 against 0.508. Daily
averaging suppresses simulated output (uncorrected monthly MBE +0.0085 daily-wind
against +0.0322 hourly), which happens to cancel part of ERA5's over-prediction.
The correction was then fitted to remove whatever residual that cancellation left.
This is why the correction is entangled with its own resolution: it is not a
standalone physical wind-speed correction, it is the residual of one particular
pipeline configuration.

## Caveats

- One region, one test year. Chile also carries the standing north-south gradient
  caveat, and its degenerate clusters may be worse than other regions'. **This
  should be replicated on New Zealand** (the only other hourly region) before the
  conclusion is generalised.
- The scalar > 3.0 degeneracy threshold is the same physical-plausibility cap
  used in the ML transfer retest; it was not tuned here.
- The observation construction differs from the canonical monthly path, so
  absolute values carry some slack. The resolution effects (-15% to -39%) are far
  larger than that slack.

Script: `scripts/analysis/hourly_resolution_test.py`.
Data: `output/hourly_test/cl_2024_by_aggregation.csv`.
