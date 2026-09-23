# Wake coefficient by observation unit, stage 0: registered gates

**Date:** 2026-09-23 (registered on commit, before the driver exists and before
any model is fitted)
**Scope:** gate K0, the first and cheapest stage of a test of whether the
deep-array wake term rejected in `method-physics-informed.md` would transfer if
its coefficient were fitted per unit rather than globally. Terms follow
`CONTEXT.md`; "unit" is the `obs_unit` sense defined there. Stages 1 and 2 are
not registered here, and are not registered at all unless K0 passes.

**K0 asks whether in-region residuals already show the unit contrast that the
per-unit hypothesis needs.** If they do not, no per-unit coefficient has
anything to fit, and the idea stops here without any change to `vwf.pinn`.

**Everything below is fixed before any residual is computed.** A gate or
condition added later is labelled post hoc and cannot pass.

## The hypothesis and the competing one

The wake term multiplies conversion efficiency by `1/(1 + cD)`, where `D` is
installed capacity within 10 km in MW/km2 and `c >= 0` is one global coefficient
(`src/vwf/pinn/model.py`, `wake_coefficient()`). In E9 it cut the dense-bin
residual span by 61% and worsened zero-shot transfer in 5 of 5 holdouts, mean
skill +0.338 to +0.235, against the post-hoc nine-region configuration
(`git show c2eb17e:docs/findings/pinn_wake_term.md`; prediction 11 of
`method-physics-informed-prespecification.md`). E9's figures are not
attributable to a commit and were not re-run in
`method-physics-informed-rerun-prereg.md`.

- **H-unit**, stated in `method-physics-informed.md`: an aggregate target
  (farm, plant or complex) contains its internal wake losses and a
  single-turbine target does not, so one `c` cannot serve both.
- **H-equal**, the competing reading: the capacity-weighted mean of the turbine
  CFs in an array is the array's CF, so turbine and aggregate targets contain
  the same wake loss in expectation. Under it, a density residual, where one
  exists, does not depend on the unit.

K0 is built to separate the two before anything is implemented.

## Facts about the targets, measured before registration

Measured read-only from the E1 rerun cache, `output/pinn_rerun_2026-09-16/cache/`
(manifest `git_commit` `9faa192`), on the test split. Capacity density is
`vwf.pinn.train._capacity_density(meta, 10.0)`.

| Region | Unit | Test year | Units | Capacity (MW) | D median | D max | Capacity share with D > 0.747 |
|---|---|---|---|---|---|---|---|
| DK | turbine | 2020 | 5,446 | 6,167 | 0.125 | 1.76 | 17.8% |
| DE | turbine | 2019 | 4,814 | 8,975 | 0.105 | 0.59 | 0.0% |
| UK | farm | 2019 | 5,998 rows, 339 locations | 13,726 | 0.560 | 3.12 | 49.1% |
| US | plant | 2022 | 1,276 | 141,120 | 0.603 | 9.33 | 60.8% |
| BR | complex | 2024 | 173 | 35,878 | 1.194 | 5.11 | 79.9% |

0.747 MW/km2 is the lower edge of E9's densest bin.

Three facts shape the gates:

1. **About a third of Denmark's "turbine" capacity carries a shared target.**
   On the test split, 36.2% of DK capacity belongs to units whose non-missing
   monthly CF exactly equals another unit's in the same month in more than half
   of their observed months. That covers 542 of 558 offshore units (1,659 MW)
   and 823 onshore units (576 MW), in groups of up to 111. DK is not flagged
   `pseudo_replicated_rows`. The same measure gives 0.0% for DE. **DK's dense
   capacity is therefore mostly offshore parks whose target is an aggregate.**
2. **Germany has no capacity in E9's densest bin.** Its maximum D is 0.59, and
   its 4,814 units sit at 622 distinct coordinates (postcode centroids). A DE
   slope is estimated over a narrow range. A rebuild on MaStR locations moves
   these figures; see the deviation of 2026-09-23 below.
3. **The efficiency head keeps the 50 km density.** Its correlation with the
   10 km density on the log scale is 0.48 to 0.65 by region, so part of any
   density residual is absorbed before K0 sees it. This applies in every
   region, and is the same in the wake arm the stage would feed.

## The condition

**Model.** The `pinn-in-region` arm of `scripts/pinn/e1_loro.py` with the E1
settings (`--epochs 60 --hidden 0`, profile `power`, density off, bound scale
1.0), wake off, and the efficiency head's inputs set to
`("log_capdens_50km", "is_offshore", "log_height")`. This is the "clean
control": the wake arm's inputs with `c = 0`. Seeds 0, 1, 2, 3 and 42, as E1.
Each region is fitted on its own training years and predicted on its own test
year:

| Region | Training years | Test year |
|---|---|---|
| DK | 2015 to 2019 | 2020 |
| DE | 2015 to 2018 | 2019 |
| UK | 2015 to 2018 | 2019 |
| US | 2019 to 2021 | 2022 |
| BR | 2021 to 2023 | 2024 |

**Residual.** Per test-year unit-month with an observed CF, `r = cf_sim -
cf_obs`, where `cf_sim` is the mean of the five seeds' predictions. A positive
`r` is over-prediction, which is what an unmodelled array loss produces.

**Slope.** Per region, the capacity-weighted least-squares slope `b` of `r` on
`D`, in CF per MW/km2, with an intercept. D is linear rather than logged
because the term is linear in D for small cD. Rows are not collapsed. Shared
targets are handled by the resampling instead.

**Interval.** A pigeonhole bootstrap with 1,000 draws and generator seed 0:
blocks of units and months of the test year are resampled independently with
replacement, and each draw keeps the cross product. A block is a connected
component of units linked by an exactly equal, non-missing, non-zero observed
CF in any test-year month. Every other unit is its own block. On the test split
this gives DK 4,254 blocks, DE 4,630, UK 332, US 1,271 and BR 173. [2026-09-23:
those counts are over every cache unit; the rule applies to the units observed
in the test year, which gives the counts in the note on the observed fleet
below.] The 95%
interval is the 2.5th to 97.5th percentile. Gates that combine regions use the
same draws in every region.

**Shared target.** A unit whose observed CF exactly equals another unit's in
the same month in more than half of its observed test-year months.

**What the condition must change, checked before any slope is read:** the
manifest records the efficiency head's inputs, and they exclude
`log_capdens_10km`. The clean control's in-region RMSE is reported beside the
E1 rerun's `pinn-in-region` RMSE for each region
(`output/pinn_rerun_2026-09-16/e1/e1_primary_raw.csv`). If the two are
identical to five decimals in every region, the input change did not reach
the model and K0 is not read.

## Gates

Read in order. A gate is not read until the one before it has been.

| Gate | Requirement |
|---|---|
| **V** estimator | Run the estimator both ways on each region's clean-control frame. First, replace `r` with `r + 0.05 (D - mean D)`: the interval of the recovered slope minus the true slope must contain 0.05. Second, permute D across blocks with generator seed 1: the interval must contain 0. **If V fails in any region, K0 is not read** until the estimator is fixed, recorded as a deviation. |
| **K0a** aggregate slopes | `b > 0` with the 95% interval excluding 0, in each of UK, US and BR. |
| **K0b** unit contrast | `Delta = mean(b_UK, b_US, b_BR) - mean(b_DK, b_DE) > 0`, with the interval excluding 0. Regions are weighted equally, as in training. |
| **K0c** within-region contrast | Among DK onshore units only, the slope for shared-target units minus the slope for the other units, estimated jointly with an indicator and its interaction with D, is > 0 with the interval excluding 0. It compares unit type inside one region, on one geography, with offshore removed. |

**K0 passes only if K0a, K0b and K0c all pass.**

K0b groups by `obs_unit`, as the hypothesis is stated. Because of fact 1,
`Delta` is also reported with DK restricted to units without a shared target.
That figure is reported, not gated.

### What each outcome does

- **K0a fails.** At least one aggregate region shows no density residual for a
  wake term to take up. The per-unit prediction stops at stage 0.
  `method-physics-informed.md` gets a dated note that the falsifiable statement
  in its section 4 was tested and failed, citing this document.
- **K0a passes, K0b fails.** A density residual exists but does not depend on
  the unit. This supports H-equal. The same note is added, with the reading.
- **K0a and K0b pass, K0c fails.** The contrast between units does not appear
  inside Denmark, so it is attributed to region rather than to unit. The same
  note is added.
- **K0 passes.** Stage 1 may be registered. A pass says nothing about
  transfer: it only shows the residual structure a per-unit coefficient would
  need.

Every outcome is reported with the full table below, before any
interpretation.

### Deviation, 2026-09-23: a German sensitivity on MaStR locations

**Recorded after this document's first commit, `6f856e5`, and before the
driver exists or any model is fitted.** The gates above do not change.

Branch `terrain-study-run` (not merged) rebuilt the DE caches with MaStR
coordinates and hub heights for the turbine-only study
(`method-physics-informed-turbine-prereg.md`, deviation of 2026-09-20), into
`output/pinn_turbine_2026-09-18/cache_mastr/`. Its manifest records commit
`14722be`, `git_dirty: false`, and the same licensed library hashes as the E1
rerun cache. Compared read-only with the DE caches K0 uses: the unit IDs and
the observations are identical in both splits, with a largest difference of
0.0 and no row missing on one side only. 3,114 of 4,814 test units moved,
60.2% of test capacity, and 638 changed hub height. The train split moved
3,074 of 4,288 units, 70.7% of capacity, and 592 changed hub height. The
unmoved units stay on postcode centroids, so the rebuilt fleet mixes two
location resolutions.

The move changes the regressor K0b depends on:

| DE cache | Split | Distinct locations | D median | D max | Capacity share with D > 0.747 |
|---|---|---|---|---|---|
| centroids (gated) | train | 579 | 0.092 | 0.63 | 0.0% |
| MaStR | train | 3,297 | 0.107 | 1.06 | 1.8% |
| centroids (gated) | test | 622 | 0.105 | 0.59 | 0.0% |
| MaStR | test | 3,410 | 0.126 | 0.62 | 0.0% |

**The gated DE input stays the centroid cache.** It is the cache E1 and every
other K0 region were built from, at one commit. The rebuilt cache was built at
another commit and moves the ERA5 winds and the terrain features with the
locations, so it changes more than D.

**Reported, not gated:** the DE clean control is fitted a second time on
`cache_mastr`, with the same settings, seeds, block rule and bootstrap draws.
It reports `b_DE` with its interval, and `Delta` recomputed with that `b_DE`.
It cannot change V, K0a, K0b, K0c or the outcome of K0. If K0b's outcome
would differ under this figure, the report says so beside the gated result.

### Note, 2026-09-23: the observed fleet, and one unit the MaStR cache drops

**Recorded while writing the driver, before any registered fit.** A smoke run
of the driver with one seed, one epoch and 20 draws, written outside the
repository, checked that every output is produced. Its slope values were
not read. No gate, prediction or threshold changes.

**The facts above were measured over every cache unit, and the residuals
exist only for units observed in the test year.** The two differ most in the
US, where 520 of 1,276 plants have an observation in 2022. Restated over the
units the model simulates and the test year observes:

| Region | Units observed | Capacity (MW) | D median | D max | Capacity share with D > 0.747 | Shared-target capacity | Blocks |
|---|---|---|---|---|---|---|---|
| DK | 5,365 | 6,127 | 0.127 | 1.76 | 17.9% | 36.5% | 4,168 |
| DE | 4,814 | 8,975 | 0.105 | 0.59 | 0.0% | 0.0% | 4,630 |
| UK | 5,998 rows | 13,726 | 0.560 | 3.12 | 49.1% | 99.7% | 327 |
| US | 520 | 94,669 | 0.719 | 9.30 | 67.7% | 0.0% | 515 |
| BR | 151 | 28,000 | 1.089 | 5.11 | 76.0% | 0.0% | 151 |

D is `RegionTensors.capdens`, the density the model sees, which counts the
capacity of every simulated unit whether observed or not. DK simulates 5,399
of its 5,446 units: 47 have no wind, as in the E1 rerun. Blocks are formed
over the observed units, since a unit with no residual row would only dilute
the resampling. The block counts are those the driver writes.

**The MaStR cache drops one German unit for having no wind** (4,813 simulated
against 4,814), so its block partition cannot equal the gated one. The
sensitivity gives every unit the block it has in the gated cache, so the
registered draws apply unchanged, and records the dropped unit with its
capacity share, 0.007%.

## Registered predictions

| # | Prediction |
|---|---|
| 1 | V passes in all five regions. |
| 2 | K0a passes: UK, US and BR each have a positive slope with an interval above 0. |
| 3 | **`b_DK` is positive with an interval above 0**, because DK's dense capacity is offshore parks with shared targets (fact 1). K0b therefore fails. |
| 4 | K0c's interval contains 0. |
| 5 | DE has the widest slope interval of the five (fact 2). |
| 6 | The DK-restricted `Delta`, reported and not gated, is larger than the gated `Delta`. |
| 7 | Added with the deviation of 2026-09-23. The MaStR `b_DE` interval overlaps the centroid `b_DE` interval, and K0b's outcome is the same under both. |

The prior is stated in advance: **K0 most likely fails**, on K0b or K0c.

## Output to report

In this order, from the run directory:

1. Per region: rows, blocks, capacity, the clean control's in-region RMSE
   against the E1 `pinn-in-region` RMSE, and `b` with its interval. The
   slope of each seed's residual is also given, to show the seed spread.
2. V, per region.
3. K0a, K0b (gated and DK-restricted) and K0c, with intervals.
4. The German sensitivity on MaStR locations: `b_DE` and `Delta`, beside the
   gated values.
5. The E9 six-bin residual means, recomputed on these residuals, as a
   description only.

## Commands

The driver, `scripts/studies/method-wake-unit/k0_density_slopes.py`, is
committed after this document and before the run. It fits the condition
through `vwf.pinn.train.fit(..., fleet_columns=...)` and writes the per-row
frame (ID, year, month, `cf_sim` per seed, `cf_obs`, capacity, D, type, block,
shared flag), the slope table and a manifest with `registration` set to this
file. `--sensitivity DE=PATH` fits a region a second time from another cache
directory, writes its outputs under `sensitivity/<CODE>/`, and records the
path and that directory's manifest commit. It changes no gate, and no module
under `src/vwf`. A test pins its recorded command line.

Run from the repository root, on a clean tree:

```bash
python scripts/dev/run_locked.py -- env PYVWF_INPUT=input/combined PYTHONPATH=src \
  /opt/anaconda3/bin/python -u scripts/studies/method-wake-unit/k0_density_slopes.py \
  --cache output/pinn_rerun_2026-09-16/cache \
  --config DK=configs/regions/scorecard/dk_k100.toml \
  --config DE=configs/regions/scorecard/de_k100.toml \
  --config UK=configs/regions/scorecard/uk_k50.toml \
  --config US=configs/regions/scorecard/us_k250.toml \
  --config BR=configs/regions/scorecard/br_k60.toml \
  --sensitivity DE=output/pinn_turbine_2026-09-18/cache_mastr \
  --seeds 0 1 2 3 42 --epochs 60 --hidden 0 --draws 1000 \
  --out output/method_wake_unit_<run date>/k0 \
  --registration docs/findings/method-wake-unit-prereg.md
```

The cache and the curve library are those of the E1 rerun: the licensed
library, `power_curves_sha256` `689cfee7...`, `models_sha256` `eefec036...`,
as recorded in `output/pinn_rerun_2026-09-16/cache/run_manifest.json`. The
result is therefore not third-party reproducible. The German sensitivity reads
`cache_mastr`, built at `14722be` on the same library. Five in-region fits per
region, and five more for the sensitivity: about an hour on the E1 timing.

## What this does not settle

- **Transfer.** K0 reads in-region residuals only. Stages 1 (a one-seed arm
  with `c` fixed at 0 for turbine units) and 2 (leave-one-region-out with a
  per-unit `c`, target harmonisation and a random-cluster control) were
  reviewed in draft but are not registered.
- **Whether DK should be flagged `pseudo_replicated_rows`.** Fact 1 bears on
  every DK scoring that treats those units as independent. That is a separate
  question and is not decided here.
- **One test year per region**, as throughout this repository.

## Committed in advance

- Every gate is reported whichever way it goes.
- A deviation from this plan is recorded with its date, and with whether it
  came before or after a residual was seen.
- A run that fails part-way is repeated in full from a clean tree, not
  resumed.
