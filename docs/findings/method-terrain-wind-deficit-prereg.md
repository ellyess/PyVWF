# Terrain and the simulated-capacity-factor deficit: registered question, measures and gates

**Date:** 2026-09-19 (registered on commit, before any driver exists or runs)
**Scope:** whether the shortfall of uncorrected simulated capacity factor
against observed capacity factor at a US plant grows with the terrain that the
reanalysis grid cannot resolve, and what the package should do about it.
Terms follow `docs/CONTEXT.md`. The question comes from issue #26.

**Everything below is fixed before any run.** Outcomes are filled in
afterwards. A measure or subgroup added later is labelled post hoc and cannot
pass a gate.

## The question

On the US k=250 scorecard configuration, eight clusters fitted offsets whose
roots lay beyond -10 m/s under the old unbounded search, with scalars of 3 to
6: their uncorrected simulation gives a third to a sixth of the observed
capacity factor (CF). The bracketed search refuses those roots, so the
clusters lose their factors. A first look (issue #26) found ordinary observed
CFs, and the same power curves simulating normal CFs at other US plants, and
noticed that the clusters sit at gap, pass and ridge sites.

**Does the plant-level CF deficit grow with sub-grid terrain, and are those
clusters the tail of that relation rather than a separate population explained
by the curve or its assignment?** The alternative explanations are the power
curve, the curve assignment and the observations.

## Seen before registration

Stated so the confirmation can be read with it in mind:

- The 2022 (test year) observed and uncorrected simulated CF of the 33 plants in
  clusters 15, 27, 38, 102, 109, 156, 183, 232 and 236, and the fleet's 2022
  capacity-weighted means (issue #26). No training-year CF was computed for any
  plant.
- For the same plants, the 2022 simulated CF of the same model key at other US
  plants.
- The site names of those plants, which is where the terrain hypothesis came
  from.
- **Training-year information, at cluster level.** Every cluster's fixed-slice
  scalar was in the factors tables read for issues #26 and #27, and a scalar is
  the cluster's capacity-weighted observed CF over simulated CF in 2019 to
  2021, close to `Y` at cluster level. The nine clusters' per-year scalars (2.4
  to 6.0, and 46 for cluster 38) were examined; the other clusters' scalars
  were in view but not examined against anything.
- **Not seen:** any terrain measure for any plant, any plant-level `Y` in the
  training years, and any fleet-wide relation between terrain and the deficit,
  in any year.

## Fleet and years

- Configuration: `configs/regions/scorecard/us_k250.toml`, input root
  `input/combined`, ERA5 `era5/US_daily` (0.25 degree grid).
- Plants: the training fleet of
  `output/validation/bracketed_2026-09-19/US/train-bracketed/train_turb_info_250.csv`,
  with its cluster labels. A plant enters if it has at least 12 plant-months
  with both an observed and a simulated CF in the window used.
- **Training years 2019 to 2021** for gates G1 to G4. **Test year 2022** only for
  G5, read once, after G1 to G4.

## Measures

**Outcome, per plant.** `Y = ln(sum of monthly observed CF / sum of monthly
uncorrected simulated CF)`, over the plant's months with both values, in the
window. Positive `Y` means the simulation falls short. The simulated CF is the
harness's uncorrected path on the configuration above (the frames a train run
does not save, so the driver simulates them); observations come from
`pyvwf.harness.driver.load_obs_and_fleet`.

**Primary covariate, per plant.** Sub-grid relief `R`: the standard deviation
of ETOPO 2022 30 arc-second elevation over the ERA5 grid cell whose centre is
nearest the plant, taken as the 0.25 by 0.25 degree box centred on that grid
point (30 by 30 ETOPO points), from `input/reference/terrain/etopo_global.nc`.

**Secondary covariate.** Site elevation above the cell `H`: the ETOPO elevation
at the plant's coordinates minus the mean over the same box. Reported beside
`R` for every gate, and gates nothing.

**Curve-match class, per plant.** Same brand, other brand, reference curve or
unverifiable, computed with `classify`, `own_manufacturer` and
`curve_side_manufacturer` from `scripts/analysis/curve_match_audit.py` on the
training fleet above. That script writes only aggregates, so the driver
reconstructs the per-plant classes with its functions. **Validation before use:**
the reconstruction's other-brand share of capacity must equal the scorecard's
US figure, 48.3%, to one decimal place, or G4 is not assessable.

All correlations are Spearman's rho across plants. Intervals are 95% percentile
intervals from 1,000 draws of plants with replacement, seed 20260919.

## Gates

| Gate | Requirement | Outcome |
|---|---|---|
| G1 relation | rho(R, Y) at least 0.20 over 2019 to 2021, with an interval that excludes zero | **PASS**, 0.488 [0.393, 0.576] |
| G2 tail | the median fleet percentile of `R` over the plants of clusters 15, 27, 38, 102, 109, 156, 183, 232 and 236 is at least 75 | **PASS**, 95.1 |
| G3 curve control | among model keys with at least 3 plants on each side of the fleet median of `R`, the median `Y` of the high-`R` plants exceeds that of the low-`R` plants for at least two thirds of the keys; at least 5 such keys must exist, or G3 is not assessable | **PASS**, 16 of 21 keys |
| G4 matching control | G1 holds with every plant classed other brand or unverifiable removed | **PASS**, 0.437 [0.304, ...]; read once as not assessable on a miscomputed precondition, see the findings document's Deviations |
| G5 confirmation | on 2022 alone, rho(R, Y) is positive with an interval that excludes zero | **PASS**, 0.424 [0.324, 0.519] |

Read on 2026-09-20 from `output/terrain_wind_deficit_2026-09-20_corrected/`;
`method-terrain-wind-deficit.md` carries the result, the deviations and the
recommendation.

G2 names its clusters here, before `R` exists, so the case cannot be chosen by
its answer. They are the eight clusters of issue #26 whose roots lay beyond
-10 m/s, and cluster 38, the fourth fixed-slice factor refused since the rerun,
whose 2020 fit was not a root. G3 compares one curve at different terrain, which separates wind
from curve directly: if `Y` follows the model key rather than `R`, G3 fails. A
gate that is not assessable does not pass.

## Registered predictions

G1, G2, G3 and G4 pass, and G5 passes. The strongest competing prediction, that
the deficit is a curve or assignment effect, predicts G3 or G4 fails.

## Consequences for the package, stated in advance

What the study would recommend under each outcome. It recommends; each change
is its own piece of work, with its own pins moved under the change-detector
policy.

- **A terrain flag on clusters.** If G1, G2, G3 and G4 pass, but rho(R, Y)
  is below 0.40 or the plants in the top decile of `R` have a median `Y` below
  ln 2: terrain explains the deficit, but not so strongly that one wind
  correction cannot serve it. The package would compute `R` per cluster at
  training, carry it in the factors table and the manifest, and flag clusters
  above a threshold in `fit_quality`, so that a refused or degenerate factor
  at a high-relief site says why. The threshold would come from this study's
  data and be registered before use.
- **A separate correction regime.** If G1 to G4 pass with rho(R, Y) at
  least 0.40 and the top-decile median `Y` at least ln 2 (the simulation
  gives at most half the observed CF): the affine correction on 0.25 degree
  winds is asked to do more than it can, which is what the roots beyond
  -10 m/s show. High-relief clusters would get their own regime. Candidates
  are a finer wind product for them, or a correction fitted in a
  multiplicative form that does not need offsets beyond the bounds; the
  choice would be its own study. The flag above comes first either way.
- **No change to terrain handling.** If G1 fails, or G3 or G4 fails: terrain
  does not explain the deficit, or cannot be separated from the curve or its
  assignment. The package changes nothing about terrain, and the next step is
  a curve-assignment study on the same plants: re-simulating each mismatched
  plant on its USWTDB model's curve.
- **G5 fails with G1 to G4 passing:** the relation does not hold out of the
  training years. The recommendation drops to the flag at most, stated as
  unconfirmed.

## Tooling this needs

- A driver, `scripts/studies/method-terrain-wind-deficit/terrain_deficit.py`,
  with `cli(argv)` and its recorded command line pinned in
  `tests/test_script_command_lines.py`, committed before it runs.
- Nothing new in `src/pyvwf`. The relief computation stays in the driver unless
  the flag outcome moves it into the package.
- Runs under `output/terrain_wind_deficit_<date>/`, from the repository root on
  a clean tree.

## Committed in advance

This file, before the driver. The commit that adds it is older than any
output of the study.
