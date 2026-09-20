# Terrain and the simulated capacity-factor deficit in the United States

**Reproduction record, added 2026-09-20.** Driver:
`scripts/studies/method-terrain-wind-deficit/terrain_deficit.py`, run with no
arguments as its docstring records. Numbers: every figure below comes from
`output/terrain_wind_deficit_2026-09-20_corrected/`, produced at commit
`1e0e16b` from a clean tree. The gates were fixed in
`method-terrain-wind-deficit-prereg.md`, committed at `b5928d3` on 2026-09-19
(merged as `f8f7371`), before the driver existed. This study writes no
`metrics.csv`: it measures the uncorrected simulation against observations and
fits nothing, so its outputs are the four tables named below.

**Date:** 2026-09-20
**Scope:** whether the shortfall of uncorrected simulated capacity factor
against observed capacity factor at a US plant grows with the terrain the
reanalysis grid cannot resolve, and what the package should do about it. Terms
follow `docs/CONTEXT.md`. The question comes from issue #26.

**Sub-grid terrain predicts the deficit, and the clusters whose offset roots
lay beyond the search bounds are the tail of that relation rather than a
separate population.** All five registered gates pass, including both controls
that could have shown the deficit to be a power curve or a curve-assignment
effect.

## What was measured

- **Fleet:** the training fleet of
  `output/validation/bracketed_2026-09-19/US/train-bracketed`
  (`configs/regions/scorecard/us_k250.toml`, commit `201c62e`), 1,091 plants
  with its cluster labels.
- **Training years 2019 to 2021** for G1 to G4; the **single test year 2022**
  only for G5, read once, after the others.
- **Curve library:** the combined library under `input/combined`. The fleet's
  `curve_resolution.csv` records 92 model keys, all resolved and none
  substituted, of which **71.0% of capacity simulates on curves of `external`
  origin** and 29.0% on curves the open library also contains. So **this study
  is not third-party reproducible**. It uses no fitted factors: the deficit is
  measured against the uncorrected simulation.
- **Outcome** `Y = ln(sum observed CF / sum uncorrected simulated CF)` per
  plant, over the plant's months with both values, at least 12 of them.
- **Primary covariate** `R`, the standard deviation of ETOPO 2022 30
  arc-second elevation over the plant's 0.25 degree ERA5 cell. **Secondary**
  `H`, the site elevation above that cell's mean, which gates nothing.
- **356 of the 1,091 plants have an outcome.** The rest do not report monthly
  in the training years. This was not stated in the pre-registration; see
  Deviations.

## The gates, as they read

Source: `us_gates.csv`. Intervals are 95% percentile intervals from 1,000
draws of plants with replacement, seed 20260919.

| Gate | Registered threshold | Value | Interval | Outcome |
|---|---|---|---|---|
| G1 relation | Spearman rho(R, Y) at least 0.20, interval excludes zero | **0.488** | 0.393 to 0.576 | **PASS** |
| G2 tail | the nine named clusters' median relief percentile at least 75 | **95.1** | | **PASS** |
| G3 curve control | high-relief plants fall further for at least two thirds of at least five model keys | **0.762** (16 of 21 keys) | | **PASS** |
| G4 matching control | G1 holds without other-brand and unverifiable plants | **0.437** | 0.304 to | **PASS** |
| G5 confirmation | on 2022 alone, positive, interval excludes zero | **0.424** | 0.324 to 0.519 | **PASS** |

Supporting figures, from `us_plant_deficit.csv` unless stated. The relief and
deficit distributions are printed by the driver and recomputable from that
file:

| Quantity | Value |
|---|---|
| Plants with an outcome | 356 of 1,091 |
| `R`, median / mean / maximum | 25.7 / 47.6 / 354.8 m |
| `Y`, minimum / median / maximum | -1.22 / 0.023 / 3.83 |
| Secondary covariate `H`, rho against `Y` | 0.238 [0.120, 0.350] |
| Top decile of `R` | at and above 130.1 m |
| Median `Y` in that decile | 0.746 (ln 2 = 0.693) |
| The nine named clusters | 14 plants, median `R` 158.1 m, median `Y` 1.177 |
| Curve-match classes | 165 other brand, 120 same brand, 68 reference curve, 3 unverifiable |
| G4's population | 188 plants |
| Other-brand share of capacity, fitted fleet | 0.4831, against the scorecard's 0.483 (printed by the driver as G4's precondition; G4's pass in `us_gates.csv` records that it cleared) |
| Confirmation year plants | 354 (`us_confirmation_2022.csv`) |

The four model keys where high-relief plants fall furthest short, from
`us_within_key_control.csv`, with the median `Y` on each side:

| Model key | High relief | Low relief | Median `Y` high | Median `Y` low |
|---|---|---|---|---|
| Alstom.Eco.80 | 6 | 3 | 0.784 | 0.281 |
| Suzlon.S88.2100 | 9 | 3 | 0.456 | -0.243 |
| GE.1.5xle | 3 | 5 | 0.432 | 0.038 |
| Vestas.V80.1800 | 5 | 3 | 0.407 | -0.328 |

### The reported coverage check

Not a gate. Source: `us_relief_coverage.csv`.

| Population | Plants | Median `R` | Mean `R` | 90th percentile `R` |
|---|---|---|---|---|
| Fitted fleet | 1,091 | 28.1 | 61.4 | 164.6 |
| With an outcome | 356 | 25.7 | 47.6 | 130.1 |

**The plants with an outcome sit on gentler ground than the fleet**, on every
one of the three measures. The relation is therefore measured on a population
whose terrain is less extreme than the fleet's, which is the direction that
understates a terrain effect rather than manufacturing one.

## Reading

The simulation's shortfall grows with terrain the 0.25 degree grid cannot
resolve, and the effect is large where the terrain is: in the top decile of
sub-grid relief the median plant's observed capacity factor is 2.1 times its
simulated one.

The two controls are what make this more than a correlation.

- **G3 separates wind from curve.** Within a single model key, plants on
  rougher ground fall further short than plants on gentler ground, for 16 of
  the 21 keys that have at least three plants on each side. The same curve
  cannot explain a deficit that depends on where it stands.
- **G4 separates wind from curve assignment.** Dropping every plant whose
  assigned curve is another manufacturer's, and every plant whose manufacturer
  cannot be identified, leaves 188 plants and a rho of 0.437 with an interval
  that excludes zero.

**The nine clusters of issue #26 are the tail, not a separate population.**
Their median relief percentile is 95.1, and their median `Y` of 1.177 means
their observed output is about 3.2 times their simulated output. They are the
extreme of the relation the other 342 plants trace, which is what G2
registered as a prediction before `R` was computed for any plant.

**The secondary covariate is weaker.** Site elevation above the cell mean
correlates with the deficit at 0.238, against relief's 0.488. Height above the
cell says where in the cell a plant sits; relief says how much the cell hides.

## What this recommends for the package

The pre-registration fixed the mapping before the run. With G1 to G4 passing,
rho of 0.488 at or above 0.40, and a top-decile median `Y` of 0.746 at or
above ln 2, the outcome maps to **a separate correction regime for
high-relief clusters, with a terrain flag first**:

1. **A terrain flag on clusters.** Compute `R` per cluster at training, carry
   it in the factors table and the manifest, and flag clusters above a
   threshold in `fit_quality`, so a refused or degenerate factor at a
   high-relief site says why. The threshold comes from this study's data and
   is registered before use.
2. **A separate regime for the flagged clusters.** The affine correction on
   0.25 degree winds is being asked for more than it can give: that is what
   the offset roots beyond -10 m/s were. Candidates are a finer wind product
   for those clusters, or a correction in a form that does not need offsets
   outside the bounds. Which one is its own study, registered separately;
   nothing is implemented from this document.

G5 passing means the relation holds in a year no part of the fit or the
covariate saw.

## Deviations

- **G4 was read once on a miscomputed precondition.** The pre-registration
  requires the rebuilt per-plant curve-match classes to reproduce the
  scorecard's US other-brand share of capacity, 48.3%, before G4 counts. The
  first run (2026-09-20, `output/terrain_wind_deficit_2026-09-20/`, commit
  `d68b542`) took that share over the 356 plants with an outcome, which reads
  0.4755, and reported G4 as not assessable. The share is taken over the
  fitted fleet, as `scripts/analysis/curve_match_audit.py` takes it, where the
  same classes read 0.4831. **The classification and the statistic G4 tests
  were unchanged by the correction**: G4's rho was 0.437 with a lower bound of
  0.304 in both runs. Only the precondition moved, and with it G4's status
  from not assessable to pass. The driver was corrected in `1e0e16b`, before
  the re-run, and the first run's directory is left in place.
- **The outcome population is 356 of the fitted fleet's 1,091 plants**, those
  that report monthly in the training years. The pre-registration set a
  12-month minimum per plant but did not say what share of the fleet would
  clear it, and it registered no check on how that population differs from the
  fleet. The check above was added with the correction, is reported rather
  than gated, and finds the scored plants on gentler ground than the fleet on
  median, mean and 90th percentile relief.
- **No other deviation.** The gates, their thresholds, the nine clusters, the
  draw count and the seed are as registered.

## Caveats

- **One region and one test year.** The relation is measured on the US fleet
  and confirmed on 2022 alone. Nothing here says the same slope holds
  elsewhere, and the other regions' terrain has not been measured.
- **Not third-party reproducible.** The fleet simulates on the combined curve
  library, whose curves are not redistributable.
- **Reporting is not random.** The 356 plants are those that report monthly to
  EIA-923. They sit on gentler ground than the fleet, so a fleet-wide relation
  could be stronger than the one measured, but nothing here establishes that
  the rest of the fleet behaves the same way at all.
- **The deficit is not only terrain.** The US row carries an unscreened
  ERCOT and SPP curtailment confound (`scorecard.md`), which depresses
  observed output and therefore works against the relation found here rather
  than for it. Relief explains part of the spread, not all of it: rho of 0.488
  leaves most of the variance unexplained.
- **`R` is a proxy.** Standard deviation of elevation in the cell is one
  measure of what a grid cell hides. It says nothing about channelling
  direction, stability or roughness length, all of which matter at the gap and
  ridge sites this flags.
