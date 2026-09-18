# Leaving one country out: the physics-informed model corrects national capacity factor in most European countries, and its answer depends on the training pool

**Date:** 2026-09-17, Study B added 2026-09-18
**Scope:** whether the physics-informed model simulates accurate monthly
capacity factors in European countries it was never trained on, scored as the
national monthly capacity factor on each country's single test year. Registered
design and gates: `method-physics-informed-loco-prereg.md`. The model is
`method-physics-informed.md`. Terms follow `CONTEXT.md`.

**Trained on eight European countries and applied to the ninth, the model cuts
national capacity-factor RMSE in eight of nine gated countries, to a median of
29% of the uncorrected error, and recovers a median 94% of what fitting inside
the country achieves.** It fails the registered floor gate, because in France,
the one country whose uncorrected series was already unbiased, it imposes a
bias and makes the error 78% worse.

**Adding six non-European regions to the training pool does not help.** It
beats the Europe-only pool in one of nine countries and is worse, beyond noise,
in four (Study B, gate W1 fails, as predicted). It moves the answer in both
directions: it repairs France, so no country is made worse, and it corrects
Italy and Norway, the two most biased, less. Italy's error is 0.043 with the
European pool and 0.076 with the world pool. For a country with no
observations, the choice of training pool moves the national error by up to
0.049 (Norway).

Terrain and fleet features beat a constants-only control in seven of nine
countries in both studies, against the registered prediction. One of those
features is a data convention that can identify a country, and no finer wind
product has been tried, so it is not claimed as a transferable property of
place.

## Results, Study A (Europe pool)

National monthly capacity-factor RMSE on each fold's test year: 12 months,
mean over five seeds (0, 1, 2, 3, 42), with the seed standard deviation where it
is not zero. Each fold is scored on the months every arm can score, and none was
excluded. Curve library: `input/combined` (licensed), `power_curves_sha256`
`689cfee7...`, no substituted unit in any cache. Roughness: per timestep
everywhere. Run at `5960efd`, clean tree.

Data: `output/pinn_loco_2026-09-16/europe/loco_europe_raw.csv`, sha256
`86ae858f4eb9120c33c9b7b867038eb275fc652a50a4fced10e3650edc9d59b9`.

| Fold | Training years | Test year | uncorrected | transfer | features-off | in-country |
|---|---|---|---|---|---|---|
| DK | 2015 to 2019 | 2020 | 0.1175 | 0.0198 ± 0.0016 | 0.0350 | 0.0298 ± 0.0002 |
| DE | 2015 to 2018 | 2019 | 0.0504 | 0.0145 ± 0.0009 | 0.0504 | 0.0073 ± 0.0002 |
| UK | 2015 to 2018 | 2019 | 0.0429 | 0.0375 ± 0.0363 | 0.0326 | 0.0307 ± 0.0004 |
| FR | 2015 to 2021 | 2023 | 0.0204 | 0.0362 ± 0.0004 | 0.0200 | 0.0115 ± 0.0002 |
| BE | 2015 to 2021 | 2023 | 0.1226 | 0.0343 ± 0.0054 | 0.0842 | 0.0163 ± 0.0023 |
| ES | 2015 to 2021 | 2023 | 0.0710 | 0.0188 ± 0.0026 | 0.0375 | 0.0173 ± 0.0002 |
| IE | 2017 to 2021 | 2023 | 0.0452 | 0.0190 ± 0.0019 | 0.0506 | 0.0191 ± 0.0010 |
| IT | 2015 to 2021 | 2023 | 0.1544 | 0.0432 ± 0.0025 | 0.1238 | 0.0188 ± 0.0042 |
| NO | 2015 to 2021 | 2023 | 0.1286 | 0.0513 ± 0.0021 | 0.1137 | 0.0794 ± 0.0010 |
| SE, flagged | 2015 to 2021 | 2023 | 0.0984 | 0.1132 ± 0.0008 | 0.0567 | 0.0182 ± 0.0006 |
| PT, flagged | 2015 to 2021 | 2023 | 0.1808 | 0.1112 ± 0.0028 | 0.1336 | 0.0271 ± 0.0006 |
| NL, flagged | 2015 to 2021 | 2023 | 0.2140 | 0.0918 ± 0.0025 | 0.1010 | 0.0182 ± 0.0000 |

National monthly MBE, same runs:

| Fold | uncorrected | transfer | features-off | in-country |
|---|---|---|---|---|
| DK | +0.1107 | -0.0086 | +0.0243 | +0.0242 |
| DE | +0.0426 | +0.0125 | +0.0487 | -0.0028 |
| UK | +0.0370 | +0.0075 | +0.0295 | -0.0288 |
| FR | -0.0015 | -0.0343 | +0.0109 | +0.0053 |
| BE | +0.1164 | +0.0309 | +0.0818 | +0.0066 |
| ES | -0.0698 | -0.0138 | -0.0351 | +0.0110 |
| IE | +0.0353 | +0.0103 | +0.0419 | +0.0100 |
| IT | -0.1512 | -0.0419 | -0.1206 | -0.0102 |
| NO | -0.1245 | -0.0422 | -0.1049 | -0.0709 |
| SE, flagged | -0.0967 | -0.1104 | -0.0530 | -0.0124 |
| PT, flagged | -0.1771 | -0.1075 | -0.1288 | +0.0145 |
| NL, flagged | +0.2017 | +0.0857 | +0.0984 | +0.0049 |

Per-unit RMSE for the turbine folds, on common unit-months, gated nowhere:

| Fold | uncorrected | transfer | features-off | in-country |
|---|---|---|---|---|
| DK | 0.1475 | 0.0939 | 0.1044 | 0.0823 |
| DE | 0.0861 | 0.0718 | 0.0804 | 0.0598 |
| UK | 0.1456 | 0.1735 | 0.1336 | 0.1280 |

### Gates and predictions

| Gate | Requirement | Outcome |
|---|---|---|
| G0 records | clean tree, one commit, library, no substitution, derived roughness, exact per-year grids | **Pass** |
| G1 floor | transfer below uncorrected in at least 7 of 9, none worse by more than 10% | **Fail.** 8 of 9 below; FR worse by 78% |
| G2 recovery | recovery ratio at least 0.5 in most defined folds | **Pass.** 7 of 9 defined folds |
| G3 place | transfer below features-off by more than the margin in at least 5 of 9 | **Pass.** 7 of 9; not FR, not UK |

Recovery ratios, gated folds: DK 1.04, DE 0.94, UK -0.70, FR -3.16, BE 0.94,
ES 0.99, IE 1.00, IT 0.94, NO 1.36.

| # | Prediction | Outcome |
|---|---|---|
| 1 | G1 passes | **Failed**, on France alone |
| 2 | G2 fails | **Failed**: G2 passed |
| 3 | G3 fails | **Failed**: G3 passed |
| 4 | Transfer's gain rises with uncorrected bias | **Held**, Spearman +0.92 |
| 5 | Flagged folds reported with their defects | **Held** |

Three of five predictions failed, and all three in the direction of the model
doing better than predicted. They are reported as failed predictions.

## What Study A shows

**The error being corrected is a level, not a shape.** Uncorrected ERA5 already
tracks each country's months: Pearson r is 0.97 to 0.996 in every gated fold.
What is wrong is the level, with national MBE from -0.15 to +0.12. Transfer
leaves r where it was, 0.965 to 0.996, and removes most of the bias. That is
why prediction 4 holds so strongly: the countries with the most bias have the
most to gain from a correction that mostly moves the level.

**France shows what a transferred correction costs where none is needed.**
France's uncorrected national series has an MBE of -0.0015, the only gated fold
already unbiased. The model trained on the other eight moves it to -0.0343, the
pool's average correction applied to a country that did not want one. The
floor gate exists for this case and it fails. A method that makes accurate
series worse cannot be applied blind, and nothing in the uncorrected series
tells a user in advance which kind of country they are in.

**Constants carry much of the gain, and features carry more.** The
features-off control, one global efficiency, spread and speed-up amplitude,
beats uncorrected in eight of nine gated folds on its own. With terrain and
fleet features, transfer beats that control by more than the margin in seven
of nine. The registered reading of a G3 pass is that something tied to place
transfers across borders, which the coefficient-space studies never found.
Study B repeats it, 7 of 9 on a different pool (not UK, not NO). Two things
still stop it being claimed:
- **The registration requires a finer wind product first,** and none has been
  tried.
- **Hub height identifies a country on the country tier.** Every grid point in
  a country carries the same height: 80 m in IT, NO and PT, 85 m in IE, 90 m in
  FR and ES, and 100 m in BE, NL and SE. The efficiency head sees log hub height.
  A value shared by a few countries can carry their level into a country left
  out without any physics behind it. This confound was not controlled in
  the design, and it is the first thing a follow-up should remove.

**National accuracy is not unit accuracy.** In the UK, transfer beats
uncorrected nationally in four of five seeds, but is worse per unit in all
five, 0.1735 against 0.1456 on average. Errors of opposite sign across farms
cancel in the national mean. A user after a farm's capacity factor gets a worse
answer from transfer than from ERA5 alone in the UK.

**One UK seed found a different optimum.** Seeds 0, 2, 3 and 42 score 0.012 to
0.030 nationally. Seed 1 scores 0.101, with a mean speed-up of 1.54 against 0.99
to 1.12 for the others, at a training loss within the others' range
(0.0028 against 0.0021 to 0.0029). That seed alone gives the UK its seed spread
of 0.036 and a G3 margin of 0.051. The training data does not pin the solution
when the UK's own terrain is held out.

**Where in-country is worse than transfer.** In Norway, in-country scores 0.079
and transfer 0.051, a recovery ratio of 1.36. Norway's own affine scorecard row
is also worse than uncorrected (`scorecard.md`), and its training register has
known defects (`method-country-level.md`). Fitting on Norway's own data teaches
the model Norway's defects. Denmark's ratio of 1.04 is inside seed noise.

## Flagged folds

These are never trained on and sit outside every gate.

- **SE, derived capacity register:** transfer is worse than uncorrected, 0.113
  against 0.098, with a bias of -0.110. Sweden's denominator is constructed to
  peak near 0.9, so its observed level is not a measurement.
- **PT, frozen training register:** transfer halves the error, 0.181 to 0.111,
  and in-country reaches 0.027. Uncorrected r is 0.85, far below every other
  fold.
- **NL, observations capped at 0.57:** transfer cuts the error from 0.214 to
  0.092, and in-country reaches 0.018. The in-country fit learns the cap.

## How this sits with the coefficient-space result

`method-why-corrections-do-not-transfer.md` finds that fitted scalars and
offsets do not transfer between countries. This finding does not contradict it.
The quantity differs: a national capacity factor, not a coefficient pair at a
cluster centroid. The model differs too. It transfers a physical operator with
bounded, mostly global quantities, and no unidentified pair is ever fitted. A
correction that is mostly a level shift can transfer as a level shift, while
the local split between scalar and offset stays unpredictable. Whether the
feature-borne part is physical is the question G3 raised and did not settle.

**Study B agrees with the coefficient-space result where they overlap.**
`method-why-corrections-do-not-transfer.md` eliminated regime coverage: more
diverse training regions do not predict better transfer. Here, six regions from
three other continents made European transfer worse in four countries and better
in one.

## Deviation, recorded 2026-09-17, after Study A ran

**The features-off arm is not exactly identical across seeds.** The
registration said its seed spread would be zero by construction. The largest
spread across folds is 6.2e-6. The seed also orders the turbine regions' unit
minibatches, so floating-point sums differ in order. It changes no gate: G3's
margin has a floor of 0.002.

## Results, Study B (world pool)

The same twelve folds and arms, with US, BR, AR, AU-NEM, CL and NZ added to every
fold's pool. Run at `9d1767c`, clean tree, same caches as Study A. Data:
`output/pinn_loco_2026-09-16/world/loco_world_raw.csv`, sha256
`ffc67e4ffcf106ee400edfef2959d263f6e904091ffc1ded56ba74e304e94974`. The
`uncorrected` and `in-country` arms do not use the pool and match Study A
exactly.

National monthly RMSE, mean over five seeds:

| Fold | uncorrected | transfer, Europe (A) | transfer, world (B) | W1 margin | features-off (B) | in-country |
|---|---|---|---|---|---|---|
| DK | 0.1175 | 0.0198 ± 0.0016 | 0.0185 ± 0.0007 | 0.0025 | 0.0465 | 0.0298 |
| DE | 0.0504 | 0.0145 ± 0.0009 | 0.0269 ± 0.0007 | 0.0020 | 0.0510 | 0.0073 |
| UK | 0.0429 | 0.0375 ± 0.0363 | 0.0373 ± 0.0087 | 0.0528 | 0.0365 | 0.0307 |
| FR | 0.0204 | 0.0362 ± 0.0004 | 0.0131 ± 0.0012 | 0.0020 | 0.0217 | 0.0115 |
| BE | 0.1226 | 0.0343 ± 0.0054 | 0.0423 ± 0.0042 | 0.0097 | 0.0830 | 0.0163 |
| ES | 0.0710 | 0.0188 ± 0.0026 | 0.0152 ± 0.0029 | 0.0055 | 0.0294 | 0.0173 |
| IE | 0.0452 | 0.0190 ± 0.0019 | 0.0326 ± 0.0032 | 0.0052 | 0.0495 | 0.0191 |
| IT | 0.1544 | 0.0432 ± 0.0025 | 0.0764 ± 0.0028 | 0.0053 | 0.1136 | 0.0188 |
| NO | 0.1286 | 0.0513 ± 0.0021 | 0.0996 ± 0.0049 | 0.0075 | 0.1050 | 0.0794 |
| SE, flagged | 0.0984 | 0.1132 ± 0.0008 | 0.0782 ± 0.0032 | 0.0047 | 0.0562 | 0.0182 |
| PT, flagged | 0.1808 | 0.1112 ± 0.0028 | 0.0999 ± 0.0012 | 0.0043 | 0.1311 | 0.0271 |
| NL, flagged | 0.2140 | 0.0918 ± 0.0025 | 0.0811 ± 0.0034 | 0.0059 | 0.1100 | 0.0182 |

National monthly MBE of the transfer arm:

| Fold | uncorrected | transfer, Europe (A) | transfer, world (B) |
|---|---|---|---|
| DK | +0.1107 | -0.0086 | +0.0017 |
| DE | +0.0426 | +0.0125 | +0.0255 |
| UK | +0.0370 | +0.0075 | +0.0362 |
| FR | -0.0015 | -0.0343 | -0.0047 |
| BE | +0.1164 | +0.0309 | +0.0394 |
| ES | -0.0698 | -0.0138 | +0.0079 |
| IE | +0.0353 | +0.0103 | +0.0269 |
| IT | -0.1512 | -0.0419 | -0.0745 |
| NO | -0.1245 | -0.0422 | -0.0929 |
| SE, flagged | -0.0967 | -0.1104 | -0.0754 |
| PT, flagged | -0.1771 | -0.1075 | -0.0958 |
| NL, flagged | +0.2017 | +0.0857 | +0.0767 |

Per-unit RMSE of the transfer arm for the turbine folds, world pool: DK 0.0944,
DE 0.0709, UK 0.1328, against uncorrected 0.1475, 0.0860 and 0.1456.

### Gate and prediction

| Gate | Requirement | Outcome |
|---|---|---|
| W1 more climates | B's transfer below A's by more than the margin in at least 5 of 9 | **Fail.** 1 of 9 (FR); B worse beyond the margin in 4 (DE, IE, IT, NO) |

Prediction 5, that W1 fails, **held**.

Study B's own G1 to G3 are reported, not gated:
- **G1:** transfer below uncorrected in 9 of 9, none worse by more than 10%.
- **G2:** recovery at least 0.5 in 8 of 9, median 0.77, against A's 0.94.
- **G3:** transfer beats features-off in 7 of 9, not the UK or Norway.

### What Study B shows

**Other climates do not improve European transfer.** Across the gated folds the
median transfer error is 53% of uncorrected with the world pool, against 29%
with Europe alone. B is worse in DE, IE, IT and NO. Italy and Norway are the
two most negatively biased folds, and the world pool removes less of their
bias: 0.077 of Italy's -0.151 against 0.109 with Europe alone, and 0.032 of
Norway's -0.125 against 0.082. Germany and Ireland are biased by about +0.04,
and the world pool leaves two to three times as much of it in place.

**The pool sets the level.** Where the Europe pool over-corrected, the world pool
lands closer: France's bias goes from -0.034 to -0.005, and the three flagged
folds improve. Where the Europe pool corrected well, the world pool
under-corrects. That is what a transferred correction that is mostly a pooled
level would do when the pool changes. The features-off control moves with the
pool too, DK from 0.035 to 0.047 and ES from 0.037 to 0.029, so the pool shifts
even the global constants.

**Neither pool is safe everywhere, and the uncorrected series cannot choose
between them.** The European pool damages France, and the world pool leaves
Italy and Norway with nearly twice the error. No gated fold is made worse by the world
pool, and B's G1 passes where A's failed. That is a gain in safety bought with
accuracy, not a better model.

**The UK became stable.** The five seeds span 0.026 to 0.046, against 0.012 to
0.101 in Study A. Per unit, transfer now beats uncorrected in the UK, 0.133
against 0.146, where in Study A it was worse in every seed.

## Caveats

- **One test year per fold,** 2019 or 2020 for turbine folds and 2023 for
  country folds. Twelve months decide each fold's figure, and the seed spread
  does not measure that.
- **Country grids are coarse.** They carry one curve and one hub height per
  country and are typed all onshore. BE's national series includes offshore
  generation its grid does not represent.
- **The loss mixes two statistics.** Regions are weighted equally, as
  published, but turbine regions contribute unit-month errors and country
  regions national-month errors.
- **The fleet features changed from the reproduced configuration** before the
  run: the 10 km capacity density was dropped (registered).
- **Screening-level.** No figure here is a yield assessment.
