# What the power curve library contributes: C1, the country-level rows

**Date:** 2026-09-13
**Scope:** the curve library study of `method-curve-library-prereg.md`. **This
document is in progress**: condition C1 has run and is reported here. C2, T1
and T2 have not. Terms follow `CONTEXT.md`.

**Giving each country grid point its own power curve lowers its simulated
output everywhere, by 0.09 to 0.22 in mean bias, without exception. Whether
that helps depends entirely on where the row started.** Three rows were
over-producing by +0.165 or more and are repaired by it; five sat at +0.085 or
below and are pushed further wrong by the same change. G1 fails, at three of
the seven scoreable countries against the four it required.

## What ran

Every country-level row runs on one key: `Vestas.V80.2000` (FR, IT, PT),
`Vestas.V90.2000` (ES, IE) or `Vestas.V90.3000` (BE, NO, SE). None is in the
open library, so under C0 every unit is simulated on the first column of
`power_curves.csv`, a 100 kW distributed-wind machine at 167 W/m2.

| Condition | Library | Result |
|---|---|---|
| C0, as reported | open | substituted share 1.00: every unit on the fallback curve |
| C1 | combined | substituted share **0.00**: every unit on its own Vestas curve |

C0 is the row standing in the scorecard today, which for these eight is the
re-run of 2026-09-13 on the per-timestep roughness and the wider ERA5 box.
Runs: `output/curve_library_study_2026-09-13/C1/`, all eight at
`git_dirty: false`, library `689cfee7…` verified by sha256 after each run.
Everything else is held: fleet, observations, training years, test year,
cluster count, time slice, ERA5 input.

## The mechanism: the real curves produce less, in every country

| Row | Uncorrected MBE, C0 | C1 | Change |
|---|---|---|---|
| BE | +0.3367 | +0.1164 | **-0.220** |
| IE | +0.1680 | +0.0361 | **-0.132** |
| FR | +0.1648 | -0.0015 | **-0.166** |
| SE | +0.0844 | -0.0967 | **-0.181** |
| NO | +0.0271 | -0.1245 | **-0.152** |
| ES | +0.0128 | -0.0697 | **-0.083** |
| IT | -0.0692 | -0.1509 | **-0.082** |
| PT | -0.0847 | -0.1771 | **-0.092** |

Monotone and universal: eight countries, eight falls, between 0.082 and 0.220.
This is the result, and it does not depend on the ratio the gate is built from.

It is what the curves' specific powers predict. The fallback rates 167 W/m2
and the machines the grids name rate 314, 398 and 472, so the fallback reaches
rated output at a far lower wind speed and over-produces at moderate speeds by
construction. Replacing it removes that over-production everywhere.

## The split is that change landing on different starting points

| Row | Uncorrected RMSE, C0 | C1 | Helped |
|---|---|---|---|
| FR | 0.1711 | **0.0204** | yes |
| BE | 0.3399 | **0.1226** | yes |
| IE | 0.1721 | **0.0453** | yes |
| SE | 0.0876 | 0.0984 | no |
| NO | 0.0350 | 0.1286 | no |
| ES | 0.0281 | 0.0709 | no |
| IT | 0.0703 | 0.1541 | no |
| PT | 0.0893 | 0.1808 | no |

The three rows the change helps are exactly the three whose uncorrected MBE was
**+0.165 or more**; the five it hurts are exactly those at **+0.085 or below**,
two of them already negative. Nothing else separates the groups: the Vestas key
does not (each of the three keys appears on both sides), and the hub height
does not (100 m and 90 m appear on both sides).

**The candidate explanation, and it is not settled here.** Five countries sat
near zero uncorrected bias *with* a curve that over-produces by construction.
If the curve was wrong and the total was right, something else in those five
was biased the other way and the wrong curve was cancelling it. That reading is
uncomfortable and it is the obvious one, so it is stated rather than left to be
inferred. What would settle it, none of which is done here:

- whether the five share a wind-speed bias in the reanalysis that the three do
  not, which the uncorrected wind distributions would show directly;
- whether their ENTSO-E capacity denominators are overstated, which would
  depress observed capacity factors and mimic a simulation that over-produces;
- whether the grids' capacity is placed where the fleet is, since a grid point
  standing in for a region can carry a siting bias in either direction;
- whether the choice of Vestas key per country was itself fitted to the
  observations at some point, which would make the starting bias circular.

## The paired comparison

Procedure B: 1,000 paired draws, seed 20260911, months resampled, all four
frames scored on the rows common to both conditions. Twelve rows per country,
none excluded, and each side reproduces its own `metrics.csv` to 1e-12.

| Row | Corrected RMSE, C1 minus C0 | 95% interval | Excludes zero |
|---|---|---|---|
| BE | -0.00149 | -0.0084 to +0.0055 | no |
| NO | +0.00100 | -0.0008 to +0.0031 | no |
| IE | +0.00267 | -0.0011 to +0.0092 | no |
| PT | +0.00480 | -0.0092 to +0.0185 | no |
| SE | +0.00512 | -0.0041 to +0.0139 | no |
| ES | +0.00717 | +0.0044 to +0.0096 | yes |
| FR | +0.01807 | +0.0108 to +0.0245 | yes |
| IT | +0.03678 | +0.0171 to +0.0549 | yes |

**The correction absorbs most of it.** Uncorrected RMSE moves by up to 0.217;
corrected RMSE moves by at most 0.037, and in five of eight the interval covers
zero. Where it does not, C1 is worse, never better.

## G1 fails, and the reason to print its denominator

A is the absorbed share: C0's uncorrected RMSE minus C1's, over C0's correction
gain. The scoreable set was fixed from C0 at seven, excluding Norway for a
non-positive gain.

| Row | A | Denominator, C0 gain |
|---|---|---|
| FR | 0.95 | 0.1589 |
| IE | 0.84 | 0.1508 |
| BE | 0.68 | 0.3198 |
| SE | -0.19 | 0.0578 |
| PT | -1.48 | 0.0619 |
| IT | -1.57 | 0.0535 |
| ES | **-22.38** | **0.0019** |

**A is at least 0.5 in three of seven. G1 required four. It fails.**

**Registered limitation, found in the result rather than in advance.** G1
divides by C0's correction gain, and the rule anticipated a gain that is not
positive, which is why Norway is excluded. It did not anticipate one that is
positive and negligible. Spain's gain of 0.0019 makes A a ratio to noise whose
sign and size are arbitrary, and Italy's and Portugal's gains of about 0.05
have the same weakness in milder form. The denominator is printed beside every
A so a reader can discount those rows themselves. No floor was introduced:
any floor chosen now would be chosen knowing which countries it admits.

## Spain's correction was repairing fabricated winds

Spain is not an instance of a pattern, and the check that shows it is the other
two returned rows.

| Row | Correction gain, superseded row | Current |
|---|---|---|
| **ES** | **0.109** | **0.0019** |
| IT | 0.032 | 0.0535 |
| PT | 0.036 | 0.0619 |

Italy's and Portugal's gains rose when their winds became real. Spain's fell by
98%. Its published correction was almost entirely repairing winds that had been
extrapolated up to five degrees past the ERA5 data
(`method-eu-rerun.md`), and with sound input there is very little left for it to
repair. Spain's row now reports an uncorrected RMSE of 0.028 against a
corrected 0.026.

## The two halves of this study are not measuring the same thing

Q1's conditions replace a fallback curve that is in the fleet by accident. Q2's
T1 and T2 swap curves between real machines. If C1's mechanism is that real
curves produce less than a 167 W/m2 stand-in, that mechanism has no counterpart
in Q2, where every condition is already on a plausible machine. **A null in Q2
would not contradict C1's result, and the two must not be read as one finding
about "the curve library".**

## Caveats

- One held-out test year per row, 2023, and everything is screening-level.
- C1 changes the library for the whole fleet at once. It cannot separate the
  curve's shape from its rating.
- The country rows fit under-determined offsets against one national series per
  month, which is a standing caveat on every figure here.
- C2, T1 and T2 have not run. Nothing here is the study's conclusion.
