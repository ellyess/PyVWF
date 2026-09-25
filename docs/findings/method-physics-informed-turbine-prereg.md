# Does a terrain correction learned from turbine and farm data transfer between regions? Registered design and gates

**Date:** 2026-09-18 (registered on commit, before the code it needs is
committed and before any model is fitted)
**Scope:** whether the physics-informed model's terrain speed-up, trained only on
per-unit observations, reduces the unit-to-unit part of the capacity-factor
error in a region it has never seen, separately from the fleet-wide level that
Studies A and B scored. Terms follow `CONTEXT.md`. The model is
`method-physics-informed.md`; the preceding studies are
`method-physics-informed-loco.md`.

**The two earlier studies scored a national series, and a national series
cannot see terrain.** Averaging units into one national figure keeps the
fleet-wide level and removes every unit-to-unit difference, which is the only
part of the error terrain could explain. Studies A and B therefore tested
whether a level transfers. This study tests whether terrain does. It trains on
turbine, farm and plant data only. Its primary metric is the spatial part of
the error, and one arm controls for the terrain term.

**A learned terrain term also has to beat an atlas that needs no learning.**
The closest physical model chain in the literature, Nayak et al. (2025,
*Applied Energy* 402, 126882), scales ERA5 at each plant by the ratio of the
Global Wind Atlas (GWA) mean speed to the ERA5 mean speed. It needs no
observations, and it has never been tested on the error between onshore units.
One arm applies that ratio, so the learned term is judged against the
strongest comparison that learns nothing.

**Everything below is fixed before any run exists.** A condition or a gate
added later is labelled post hoc and cannot pass.

## What was seen before this was written

**One exploratory analysis of training years informed this design, and it is
disclosed in full.** On 2026-09-18 the uncorrected per-unit error of every
turbine-level region's training years was split into three parts:
- **level**: the capacity-weighted fleet error each month;
- **spatial**: each unit's mean deviation from that level;
- **the rest**: month-to-month noise.

Each unit's spatial error was then regressed on its ERA5-cell relief. No test
year was touched, and no model was fitted.

| Region | Rows | Distinct sites | Level share | Spatial share | Gradient per tenfold relief (95% CI) |
|---|---|---|---|---|---|
| DK | 3,692 | 3,689 | 0.60 | 0.23 | -0.001 (-0.006 to 0.004) |
| DE | 4,288 | 579 | 0.36 | 0.42 | -0.065 (-0.069 to -0.060) |
| UK | 5,621 | 330 | 0.24 | 0.59 | -0.132 (-0.139 to -0.125) |
| US | 1,091 | 1,071 | 0.03 | 0.65 | -0.099 (-0.121 to -0.076) |
| BR | 125 | 119 | 0.18 | 0.67 | -0.269 (-0.377 to -0.182) |
| AR | 59 | 37 | 0.02 | 0.86 | -0.151 (-0.294 to -0.037) |
| AU-NEM | 67 | 59 | 0.05 | 0.77 | -0.145 (-0.296 to -0.073) |
| CL | 47 | 43 | 0.12 | 0.70 | -0.099 (-0.187 to -0.021) |
| NZ | 8 | 7 | 0.08 | 0.87 | not estimable |

**These figures motivated the predictions below, so the predictions are not
blind.** The test is what they do not show: whether a relationship fitted in
eight regions predicts the ninth on its own test year.

**The GWA ratios were also computed once before registration, as a check of the
script.** They went into a session scratchpad, not `output/`, and they used no
observation and fitted no model. Across the 18 caches:
- **Median ratio:** 1.03 to 1.59, highest in NZ.
- **Clipping:** under 1% of capacity is clipped, except AR (5%) and CL (3%).
- **Neutral value:** only DK's units with no ERA5 wind get it, and they are
  dropped anyway.

## Germany's locations: a change already expected

The DE caches carry postcode centroids: 579 distinct points behind 4,288
turbines. A per-turbine source, the Marktstammdatenregister (MaStR), was being
downloaded when this document was committed. It carries coordinates and hub
heights for every turbine above 30 kW.

If a match of MaStR to the DE fleet succeeds, the DE caches are rebuilt with its
coordinates and hub heights. That rebuild, and the match rate, are recorded
here as a dated deviation before any fit.

If the match fails, the DE fold runs on the caches as they stand, and the
postcode resolution is a caveat on the DE result. Either way the gates, arms and
predictions do not change.

## Folds

Nine turbine-level regions, one held out at a time. Each is trained on its
training years, scored on its single test year, and read from the caches
already built for the leave-one-country-out study,
`output/pinn_loco_2026-09-16/cache/`. Those caches were built at `5960efd` from
a clean tree, with per-timestep roughness and the licensed library, and no unit
was substituted.

| Region | Training years | Test year | Config | Role |
|---|---|---|---|---|
| DE | 2015 to 2018 | 2019 | `scorecard/de_k100.toml` | gated |
| UK | 2015 to 2018 | 2019 | `scorecard/uk_k50.toml` | gated |
| US | 2019 to 2021 | 2022 | `scorecard/us_k250.toml` | gated |
| BR | 2021 to 2023 | 2024 | `scorecard/br_k60.toml` | gated |
| AR | 2021 to 2023 | 2024 | `scorecard/ar_k10.toml` | gated |
| AU-NEM | 2020 to 2022 | 2023 | `scorecard/au_nem_k45.toml` | gated |
| CL | 2021 to 2023 | 2024 | `scorecard/cl_k10.toml` | gated |
| DK | 2015 to 2019 | 2020 | `scorecard/dk_k100.toml` | negative control |
| NZ | 2019 to 2023 | 2024 | `scorecard/nz_k7.toml` | reported only |

- **The pool:** every fold trains on the other eight regions, DK and NZ
  included.
- **DK is the negative control.** It is flat, a median 71 m of ERA5-cell
  relief, and its spatial error has no relationship with relief. A terrain
  term cannot help it. A gain there means something other than terrain is
  doing the work.
- **NZ is scored and reported, and gated nowhere.** Its 12 test units sit at
  about 7 distinct sites, too few to estimate a spatial error.

## Arms

Every arm uses:
- seeds 0, 1, 2, 3 and 42, and 60 epochs;
- linear heads and the power-law profile on the measured shear;
- density off, wake off and bound scale 1.0;
- the licensed library.

The fleet features are the three Studies A and B used: capacity density within
50 km, the offshore flag, and hub height. That keeps the fleet side identical
across arms and lets the secondary scoring apply the same model to country
grids.

| Arm | Trained on | Terrain speed-up | Other terrain effects | Fleet head |
|---|---|---|---|---|
| `uncorrected` | nothing | none | none | none |
| `no-terrain` | the pool | fixed at zero | none: the shear offset is one global constant | features |
| `relief-only` | the pool | the relief term with one global strength | none: the shear offset is one global constant | features |
| `full` | the pool | the relief term with a strength predicted from terrain features | the shear offset is predicted from terrain features | features |
| `gwa-ratio` | the pool | fixed, not learned: the GWA ratio below | none: the shear offset is one global constant | features |
| `in-region` | the region's own training years | as `full` | as `full` | features |

**The three pooled arms differ in terrain alone.**
- `no-terrain` against `relief-only` measures the physical terrain term.
- `relief-only` against `full` measures what the terrain features add beyond
  relief.
- **The fleet head is identical in all three,** so a difference between them
  cannot come from the fleet side.
- **`relief-only` is not Studies A and B's `features-off` arm.** That arm also
  removed the fleet features.
- **`gwa-ratio` is `no-terrain` with the atlas added.** The shear offset and
  fleet head are the same, so `gwa-ratio` against `no-terrain` measures what
  the atlas adds. `full` against `gwa-ratio` measures whether learning beats
  it.

### The GWA ratio

The atlas is Global Wind Atlas version 4 (June 2025, CC BY 4.0), with mean
wind speed at 100 m on a grid of about 250 m. Its underlying WRF climatology
covers 2008 to 2017. The files are in `input/raw/gwa4/`, one per region.

For each unit and split:

    R_i = GWA_i / E_i

- **`GWA_i`** is the atlas's 100 m mean speed averaged over the grid cells
  within 2.5 km of the unit. The 2.5 km radius follows Nayak et al.
- **`E_i`** is the unit's mean daily ERA5 100 m speed over the days of its
  split, from the cache. ERA5 is a model input, not an observation, so using
  the test year's ERA5 for test units leaks no observation.

The arm fixes the speed-up at ln R_i and applies it to the 100 m wind before
the profile, as the learned speed-up is applied.
- **Clipping:** R_i is clipped to the speed-up's own bounds, 0.67 to 2.46.
- **Missing values:** a unit with no atlas value within 2.5 km, such as an
  offshore unit beyond the country file, gets R_i = 1.
- **Recorded:** the share of capacity clipped and the share given the
  neutral value, per region and split.

**Two differences from Nayak et al., declared:**
- **Height:** they take the ratio at hub height, and here it is taken at
  100 m.
- **Period:** the atlas covers 2008 to 2017 and `E_i` covers the split's own
  years. Both are recorded as caveats, not corrected.

## Metrics

Each fold is scored on its test year, per unit, after the UK's pseudo-replicate
collapse, on the unit-months every arm can score. Let `e` be simulated minus
observed monthly capacity factor.

- **Level**, `L_m`: the capacity-weighted mean of `e` over the units observed in
  month `m`.
- **Spatial error of unit `i`**, `s_i`: the mean over its observed months of
  `e - L_m`.
- **Spatial RMSE:** the square root of the mean of `s_i` squared, each unit
  weighted by capacity times its number of observed months. **This is the
  primary metric.**
- **Level RMSE:** the square root of the capacity-weighted mean of `L_m`
  squared over unit-months. It is reported for every arm.
- **Per-unit RMSE:** as in the rerun and Studies A and B.

For each arm, the figure is the mean over seeds. The **noise margin** between
two arms is max(0.002, twice the square root of the mean of the two arms' seed
variances), the same rule Studies A and B used.

## Gates

Read in order. A gate is not read until the one before it has been. The gated
regions are the seven marked above.

| Gate | Requirement | Outcome |
|---|---|---|
| **T0** records | The study manifest records `git_dirty: false` at a commit containing this document and the code below. The cache manifest records `5960efd` and `git_dirty: false`. The library hash is `689cfee71dc9e1aa5408cff4e2dbf5c205e30bd5f0b416ba8ee99691391ee00d`. No cache has a substituted unit, and roughness is `derived` in every cache. **A failure means no metric is read until it is explained.** | |
| **T1** terrain term | `relief-only` spatial RMSE is below `no-terrain` by more than the margin in at least 5 of 7 gated regions. | |
| **T2** features | `full` spatial RMSE is below `relief-only` by more than the margin in at least 5 of 7 gated regions. | |
| **T3** negative control | In DK, `full` and `relief-only` are each within the margin of `no-terrain` on spatial RMSE. | |
| **T4** floor | `full` per-unit RMSE is below `uncorrected` in at least 5 of 7 gated regions, and above it by more than 10% relative in none. | |
| **T5** learned against atlas | `full` spatial RMSE is below `gwa-ratio` by more than the margin in at least 5 of 7 gated regions. | |

### What each outcome means

- **T1 passes:** a terrain speed-up learned in eight regions reduces the
  unit-to-unit error in a ninth, the first evidence in this programme that
  terrain transfers across borders. It is still screening-level and one test
  year per region.
- **T1 fails:** the consistent sign in the training years does not become
  transferable skill. The relief relationship differs too much between regions
  to be learned once.
- **T2 passing on top of T1:** the terrain features carry information beyond
  relief. **T2 failing on top of T1:** relief is enough, and the features add
  parameters rather than skill.
- **T3 fails:** a flat region gains from a terrain arm, so the arm is fitting
  something else. **T1 and T2 are then not read as evidence about terrain**,
  whatever their counts.
- **T4 fails:** the model harms per-unit accuracy in regions it has not seen,
  whatever its spatial gain, and cannot be used there as a per-unit correction.
- **T5 passes:** learning from generation in other regions beats an atlas
  applied without learning, which is the practical case for training on
  generation at all. **T5 fails:** a free atlas does at least as well, and the
  learned terrain term is not worth its data where an atlas exists.

## Secondary scoring, reported and gated nowhere

**Does a level learned from per-unit data transfer to national series?**
`gwa-ratio` takes no part: a country grid point sits about 50 km from its
neighbours and carries every farm nearest to it, so an atlas value at the
point describes no farm.
`no-terrain`, `relief-only` and `full` are each fitted once per seed on all nine
turbine-level regions. They are applied to the test years of the nine
country-level folds of Studies A and B: FR, BE, ES, IE, IT and NO, and the
flagged SE, PT and NL. No turbine-level model has seen any of them. They are
scored as national monthly RMSE, as in Studies A and B, and reported beside
Study A's transfer arm. DK, DE and UK are not scored here, because they are
training regions in this study.

**Prediction:** the median national RMSE across the six gated country-level
folds (FR, BE, ES, IE, IT, NO) is worse than Study A's transfer arm on the same
six. A national level depends on things terrain cannot see:
the capacity denominator, each grid's one curve and hub height, and
availability. Per-unit data teaches none of them.

## Registered predictions

| # | Prediction | Outcome |
|---|---|---|
| 1 | T1 passes. The training years show the same sign in all seven gated regions and gradients of -0.065 to -0.269. | |
| 2 | T2 fails. Features beyond relief add less than the margin in at least 3 of 7. | |
| 3 | T3 passes: DK shows no terrain gain. | |
| 4 | BR shows the largest spatial gain of the `relief-only` arm over `no-terrain`, having the steepest training-year gradient. | |
| 5 | The secondary national scoring is worse than Study A's transfer arm in the median over the six gated country-level folds. | |
| 6 | T5 passes: `full` beats `gwa-ratio` in at least 5 of 7 gated regions. | |
| 7 | `gwa-ratio` beats `no-terrain` on spatial RMSE by more than the margin in fewer than 4 of 7 gated regions. Gruber et al. (2022) found GWA correction of reanalysis gave little or negative improvement, and Nayak et al. (2025) found raw ERA5 beat GWA scaling in several countries. | |

## Code this needs

Committed immediately after this document, before any fit, with tests:
- **`pyvwf.pinn.train`:**
  - `fit(terrain_off=...)` hands the terrain heads zeros while leaving the
    fleet head on.
  - `fit(relief_off=...)` passes zero relief to the speed-up, fixing it at
    zero.
  - Both default off, so every earlier result reproduces unchanged.
- **`pyvwf.pinn.runs`:** the level and spatial decomposition above, with a test
  against a hand computation.
- **`pyvwf.pinn.train`:** a fixed speed-up per unit, `fit(fixed_speedup=True)`.
  The speed-up is read from the tensors, not the model, and the relief term is
  off.
- **A GWA preprocessing script:** it computes R_i per unit and split from the
  rasters and caches, and writes it, with the clipped and neutral shares, beside
  each cache. Its test checks a synthetic raster against a hand computation.
- **`scripts/pinn/turbine_loro.py`:** the folds, arms, metrics, gates,
  manifest, clean-tree guard and the secondary scoring.

## Commands

From the repository root, on a clean tree, one process at a time, detached.
The caches are those of the leave-one-country-out study unless a deviation
names others (see Germany's locations above).

```bash
PYTHONPATH=src /opt/anaconda3/bin/python -u scripts/pinn/gwa_ratio.py \
  --cache output/pinn_loco_2026-09-16/cache --gwa input/raw/gwa4 \
  --out output/pinn_turbine_2026-09-18/gwa
```

```bash
PYVWF_INPUT=input/combined PYTHONPATH=src /opt/anaconda3/bin/python -u \
  scripts/pinn/turbine_loro.py --tag primary \
  --cache output/pinn_loco_2026-09-16/cache \
  --gwa output/pinn_turbine_2026-09-18/gwa \
  --out output/pinn_turbine_2026-09-18/study \
  --config DK=configs/regions/scorecard/dk_k100.toml \
  --config DE=configs/regions/scorecard/de_k100.toml \
  --config UK=configs/regions/scorecard/uk_k50.toml \
  --config US=configs/regions/scorecard/us_k250.toml \
  --config BR=configs/regions/scorecard/br_k60.toml \
  --config AR=configs/regions/scorecard/ar_k10.toml \
  --config AU-NEM=configs/regions/scorecard/au_nem_k45.toml \
  --config CL=configs/regions/scorecard/cl_k10.toml \
  --config NZ=configs/regions/scorecard/nz_k7.toml \
  --config FR=configs/regions/scorecard/fr_country.toml \
  --config BE=configs/regions/scorecard/be_country.toml \
  --config ES=configs/regions/scorecard/es_country.toml \
  --config IE=configs/regions/scorecard/ie_country.toml \
  --config IT=configs/regions/scorecard/it_country.toml \
  --config NO=configs/regions/scorecard/no_country.toml \
  --config SE=configs/regions/scorecard/se_country.toml \
  --config PT=configs/regions/scorecard/pt_country.toml \
  --config NL=configs/regions/pinn_loco/nl_country.toml \
  --registration docs/findings/method-physics-informed-turbine-prereg.md
```

## Cost

- **Fits:** 9 folds times 5 fitted arms times 5 seeds is 225, plus 15 for the
  secondary scoring.
- **Time:** at the measured 3 to 4 minutes per pooled fit, about twelve hours,
  run overnight.
- **Download:** the GWA files, 2.4 GB, were downloaded on 2026-09-18, before
  this document was committed.

## Committed in advance

- Every gate and prediction is reported whichever way it goes.
- The full per-region table comes first, before any reading: spatial, level and
  per-unit RMSE for every arm, DK and NZ included.
- A run that stops part-way is repeated in full from a clean tree, not resumed.
- A deviation is recorded with its date, and with whether it came before or
  after a result was seen.
