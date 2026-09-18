# What the power curve library contributes

**Reproduction record, added 2026-09-18.** Drivers:
`scripts/studies/method-curve-library/curve_library_assign.py`,
`scripts/studies/method-curve-library/curve_library_match.py`,
`scripts/studies/method-curve-library/curve_library_study.py`,
`scripts/studies/method-curve-library/curve_library_tables.py`. Until
2026-09-18 they were in `scripts/analysis/`, the path any command below uses;
`scripts/studies/README.md` maps each old path to its new one. Numbers: the run
manifests under `output/curve_library_study_2026-09-13/` record commits
`2dff287`, `1196398`, `b00ec1f`, `207657b` and `6dd7af2`, each with a clean
tree; the override tables and the paired comparisons record none.

**Date:** 2026-09-13
**Scope:** the curve library study of `method-curve-library-prereg.md`. All four
conditions have run: C1 and C2 at country level, T1 and T2 at turbine level.
Terms follow `CONTEXT.md`.

**[Correction notice, 2026-09-13. C2's corrected-RMSE results are withdrawn.
The corrected values are not yet known.]**

**What was claimed.** This document's headline, "curve error worth 20 to 28% of
the fallback's effect on mean bias is indistinguishable from ERA5 bias to a
correction fitted per cluster and season", and the section "C2 per tier, and
P2b refuted", including P2b's refutation and every paired interval for
corrected RMSE under C2.

**What actually ran.** C2 overrides each unit's model key. The study driver
applied that override to the fleet returned by `vwf.data.train_set`, and
`train_set` simulates the fleet before it returns: it computes `gen_cf["sim"]`
by `wind.simulate_wind(reanalysis, turb_info, power_curves)` and hands back the
result. The correction's wind scalar is then fitted as `obs / sim` from that
frame, so **the scalar was fitted on the unmodified curve assignment**, while
the offset fit and the evaluation saw the overridden one. The C2 training fits
are hybrids of two conditions and are not the condition that was registered.

**How it was found, and the evidence.** The fitted scalar is bit-identical
between C0 and C2 at all eight country rows, every digit, while offsets move by
1.14 to 2.03 m/s. `calculate_scalar` computes `scalar = obs / sim`, so a scalar
that does not move is a `sim` that did not move. C1, which changes the library
rather than the keys, moves its scalars by 0.30 to 4.48, because
`load_power_curves` is called inside `train_set` and is therefore ahead of the
simulation.

**What stands, and why.** The whole C1 half. C1's fit is sound for the reason
just given, and the C1 results reported here are uncorrected mean-bias
quantities or comparisons between C0 and C1: the mechanism table, the split
across starting points, the C1 paired comparison, G1's failure at three of the
seven scoreable countries, Spain's collapsed correction gain, and the
open-library adequacy statement. C2's **uncorrected** figures also stand, since
they use no factors; that includes the ratio table of what C2 recovers of the
licensed library's effect on mean bias, 72 to 101%.

**What follows.** The override is being moved ahead of the simulation, and C2's
eight rows and the seven T conditions re-run. Corrected figures will replace
the withdrawn ones here, with their own date. Until then this document states
no corrected-RMSE result for C2, and none should be quoted from its history.

**[Closed the same day, 2026-09-13.** The override is applied in
`vwf.data.prep_country`, ahead of every simulation, and the fifteen affected
conditions were re-run. The corrected figures are in place below, and the
withdrawn ones are named where they stood. **The corrected result differs in
kind, not only in value:** C2's corrected RMSE now resolves away from C0 in
four of the eight rows where the withdrawn figures said six of eight were
indistinguishable. The uncorrected figures are unchanged to four decimals,
which is the check that the fix touched what it was meant to: every C2 mean-bias
change reproduces its withdrawn value exactly, because no factor enters it.**]

**Giving each country grid point its own power curve lowers its simulated
output everywhere, by 0.09 to 0.22 in mean bias, without exception. Whether
that helps depends entirely on where the row started.** Three rows were
over-producing by +0.165 or more and are repaired by it; five sat at +0.085 or
below and are pushed further wrong by the same change. G1 fails, at three of
the seven scoreable countries against the four it required.

**At turbine level the curve assignment barely matters, and ERA5 bias is 8 to
43 times the larger error source.** Reading the machine designation the Danish,
British and American registers already record, or moving every unit onto
another manufacturer's curve at the same scale, changes corrected RMSE by at
most 0.0012 on any row, against correction gains of 0.012 to 0.063. G2a and G2b
both pass; G3 fails everywhere.

**The strongest result in the study is in the fits rather than the metrics.**
Giving a country fleet its real curves turns stable scalars degenerate in four
of the eight rows, Italy from none to eight, its worst cluster from 2.649 to
7.133. The scalar had been absorbing a systematic power-curve mismatch, so a
scalar inside the plausible band was evidence of absorbed curve error rather
than of a good fit. P6 is refuted and that is the finding.

## What an affine correction absorbs, measured at country level

*[Replaced 2026-09-13. This section previously led with "curve error worth 20 to
28% of the fallback's effect on mean bias is indistinguishable from ERA5 bias to
a correction fitted per cluster and season", and reported the corrected
differences as ordering by nothing, with Norway smallest at 0.00018. Those
figures came from the hybrid fits described in the correction notice and are
withdrawn. The corrected result below says something weaker and more useful.]*

**The correction absorbs 81 to 98% of what changing the curve does to error,
and the remainder is resolvable in half the rows.** C2 replaced each grid's
Vestas key with the nearest open-library model, 1.9, 19.4 or 70.6 W/m2 away
depending on the country. Uncorrected RMSE moves by 0.026 to 0.154; corrected
RMSE moves by 0.0009 to 0.014, between 1.7% and 19.1% of it. But the paired
interval excludes zero in four of the eight rows, and C2 is worse in all four.

| Row | Uncorrected RMSE change | Corrected RMSE change | Share surviving | Resolves |
|---|---|---|---|---|
| NO | +0.0566 | +0.00095 | 1.7% | no |
| BE | -0.1537 | -0.00297 | 1.9% | no |
| IE | -0.1236 | +0.00547 | 4.4% | **yes** |
| PT | +0.0798 | +0.00376 | 4.7% | no |
| FR | -0.1441 | +0.01368 | 9.5% | **yes** |
| SE | -0.0257 | +0.00337 | 13.1% | no |
| ES | +0.0442 | +0.00710 | 16.1% | **yes** |
| IT | +0.0732 | +0.01398 | 19.1% | **yes** |

So a curve error of this size is **not** invisible to a scorecard row, which is
what the withdrawn figures said. It is heavily damped, by roughly an order of
magnitude, and what survives the damping is large enough to resolve in half of
these rows against 1,000 paired draws. The honest statement is that the
correction hides most of a curve error and not all of it, and that how much it
hides varies by a factor of ten across eight rows of the same design.

| Tier | Gap | Row | ΔMBE under C1 | ΔMBE under C2 | C2 as a share of C1 |
|---|---|---|---|---|---|
| near | 1.9 W/m2 | ES | -0.0825 | -0.0837 | 1.014 |
| near | 1.9 | IE | -0.1318 | -0.1309 | 0.993 |
| moderate | 19.4 | FR | -0.1663 | -0.1498 | 0.901 |
| moderate | 19.4 | IT | -0.0817 | -0.0711 | 0.871 |
| moderate | 19.4 | PT | -0.0924 | -0.0806 | 0.872 |
| far | 70.6 | NO | -0.1516 | -0.1148 | 0.757 |
| far | 70.6 | SE | -0.1812 | -0.1450 | 0.800 |
| far | 70.6 | BE | -0.2203 | -0.1582 | 0.718 |

The two conditions' bias changes correlate at 0.942 across the eight.

**What this supports, and what it does not.** Three gap values, two or three
countries each, eight points: that supports **monotone across three levels with
no overlap between tiers**, since the ratios are 0.99 to 1.01, then 0.87 to
0.90, then 0.72 to 0.80. It does not support a dose-response curve, and none is
claimed. Establishing that would need more gap values, or the same eight
countries run against a range of substitutes rather than one each.

## The Q1 answer, and what it rests on

**The country-level result depends on the fallback being wrong, not on the
replacement being right.** Any curve in the plausible range removes most of
what the 167 W/m2 fallback was doing: even the worst substitute in the study
recovers 72% of the effect, and the nearest recovers all of it.

**For these rows the open library is adequate for the bias and not quite for
the skill.** C2 uses only curves a third party has. It recovers 72 to 101% of
the licensed library's effect on mean bias. Corrected skill is another matter:
it is indistinguishable from C0 in four of the eight countries and resolvably
worse in the other four, by 0.005 to 0.014 in RMSE. Anyone running these rows
without the licensed library reproduces the mean-bias result and loses up to
0.014 of corrected RMSE, which is small beside the correction's own gain of
0.05 to 0.32 but is not nothing.

*[Corrected 2026-09-13: this paragraph said "leaves corrected skill
indistinguishable from C0 in six of the eight countries", from the withdrawn
hybrid fits. The corrected runs give four of eight.]*

## What ran at country level

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

*[Revised 2026-09-13, after C2 ran: this section was written when C1 was the
only condition, and framed the mechanism as the real curve's fidelity. C2 does
not support that framing. A substitute 70.6 W/m2 away from the machine the grid
names still produces 72 to 80% of the same change, so the mechanism is that the
fallback is extreme rather than that the replacement is accurate. The claim
this document leads with is weakened accordingly, from "the real curve produces
less" to "a curve that is not the fallback produces less, and how much less
depends on how close it is". The section's own numbers are unchanged and its
reasoning from specific power still holds: it is the size of the fallback's
error that the reasoning explains, not the precision of its replacement.]*

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

## The C1 paired comparison

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

## C2 per tier, and P2b refuted

*[Replaced 2026-09-13: the corrected-RMSE column and every interval below come
from the re-run. The withdrawn version reported ES -0.00477, IE -0.00301,
FR +0.00436, IT -0.00408, PT -0.00042, BE +0.00734, NO +0.00018, SE +0.00211,
with two rows resolving and disagreeing in direction. The uncorrected columns
and the tier distances are unchanged.]*

Every condition clean at `git_dirty: false`, open library `56314f39…` verified
by sha256, substituted share 0.00: the open substitutes resolve, so no unit
fell back. Runs: `output/curve_library_study_2026-09-13/C2/`.

| Tier | Row | Uncorr RMSE C0 to C2 | Corr RMSE C0 to C2 | Corrected difference | 95% interval | Excludes zero |
|---|---|---|---|---|---|---|
| near | ES | 0.0281 to 0.0723 | 0.0262 to 0.0333 | +0.00710 | +0.0048 to +0.0103 | yes |
| near | IE | 0.1721 to 0.0485 | 0.0212 to 0.0267 | +0.00547 | +0.0015 to +0.0112 | yes |
| moderate | FR | 0.1711 to 0.0270 | 0.0122 to 0.0259 | +0.01368 | +0.0074 to +0.0189 | yes |
| moderate | IT | 0.0703 to 0.1435 | 0.0168 to 0.0308 | +0.01398 | +0.0070 to +0.0199 | yes |
| moderate | PT | 0.0893 to 0.1691 | 0.0274 to 0.0312 | +0.00376 | -0.0099 to +0.0168 | no |
| far | BE | 0.3399 to 0.1862 | 0.0201 to 0.0171 | -0.00297 | -0.0081 to +0.0017 | no |
| far | NO | 0.0350 to 0.0915 | 0.0363 to 0.0372 | +0.00095 | -0.0013 to +0.0032 | no |
| far | SE | 0.0876 to 0.0619 | 0.0298 to 0.0332 | +0.00337 | -0.0052 to +0.0112 | no |

Four rows resolve and all four agree in direction: C2 is worse than C0. Belgium
is the only row where the substitute improves corrected skill, and its interval
covers zero.

**P2b is refuted, and more clearly than the withdrawn figures said.** It
predicted the far tier would move further from C0 than the near tier in
corrected RMSE. The tiers order the other way and are not even monotone:

| Tier | Gap | Absolute corrected differences | Mean |
|---|---|---|---|
| near | 1.9 W/m2 | 0.0071, 0.0055 | 0.0063 |
| moderate | 19.4 | 0.0137, 0.0140, 0.0038 | 0.0105 |
| far | 70.6 | 0.0030, 0.0009, 0.0034 | **0.0024** |

The far tier, whose substitutes sit 70.6 W/m2 from the machine the grid names,
moves least; the moderate tier moves most; Norway, with the study's worst
substitute, still moves least of all eight at 0.00095. Every row that resolves
is in the near or moderate tier.

The prediction's mechanism was real and it named the wrong quantity. The
distance between a substitute and the machine it replaces does order the
countries, monotonically across the three tiers, **in the uncorrected mean
bias**. It does not survive into the corrected RMSE. Written about the bias,
P2b would have held.

**The tiers earned their registration by being refuted.** Pooled across the
eight countries this would have read as an unremarkable average, and neither
the ordering in the bias nor its inversion in the corrected skill would have
been visible.

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

## Q2: what ran at turbine level

T0 is each row as it stands in the scorecard. T1 gives every unit whose
register names a machine the licensed library's curve for that machine. T2
moves every unit that is on its own maker's curve to another maker's, at the
same rating band and the nearest specific power.

| Condition | Test-fleet capacity it reassigns | Rows it cannot reach |
|---|---|---|
| T1 | DK 55.8%, UK 49.2%, US 27.5% | DE: the register records no designation for any unit |
| T2 | DE 44.4%, DK 56.3%, UK 71.3%, US 24.1% | none, since T2 needs no designation |

Runs: `output/curve_library_study_2026-09-13/T1/` and `T2/`, all seven at
`git_dirty: false`, combined library `689cfee7…` verified by sha256 after each.
Everything else is held: fleet, observations, training years, test year, cluster
count, time slice, ERA5 input and roughness treatment. The US baseline is a
regeneration of the scorecard row's evaluation, for the reason in "The US
baseline had to be rebuilt" below.

**Corrected RMSE, condition minus T0.** Procedure B: 1,000 paired draws, seed
20260911, units resampled, both conditions scored on the rows common to them.

| Row | T1 minus T0 | 95% interval | T2 minus T0 | 95% interval |
|---|---|---|---|---|
| DE | untestable | | **+0.00112** | +0.00086 to +0.00138 |
| DK | **+0.00119** | +0.00081 to +0.00158 | **-0.00064** | -0.00097 to -0.00031 |
| UK | **-0.00052** | -0.00127 to -0.00002 | -0.00008 | -0.00085 to +0.00087 |
| US | +0.00011 | -0.00122 to +0.00135 | -0.00049 | -0.00174 to +0.00085 |

Bold excludes zero. **Every difference in the table is smaller than 0.0012, in
either direction, against correction gains of 0.012 to 0.063 on the same rows.**

## G2a and G2b pass; G3 fails in all four rows

**G2a: specific power is sufficient, in Denmark and the United Kingdom.** The
gate asks whether T1's corrected RMSE differs from T0 by less than the 0.002
screen. Denmark differs by +0.00119 and the United Kingdom by -0.00052, both
resolvable and both inside the screen. Denmark's T1 is the *worse* of the two,
which raises the suspension clause registered for a T1 materially worse than
T0; 0.00119 is not material against a screen of 0.002, so the gate is not
suspended and it passes.

**G2b: the same, for all four rows including Germany.** T2 differs from T0 by
+0.00112 (DE), -0.00064 (DK), -0.00008 (UK) and -0.00049 (US), every one inside
the screen. Two resolve and two do not, and the two that resolve point in
opposite directions.

Both gates passing means the same thing: **at monthly resolution, over one test
year, assigning a unit by specific power rather than by its own designation, or
even moving it to another manufacturer's machine of the same class, costs
nothing a scorecard row would show.** The consequence registered for this is
that the design choice is recorded as supported, for these regions and at this
resolution only.

**G3 fails in all four rows, by a factor of 8 to 43.** The gate asks whether
the spread of uncorrected RMSE across T0, T1 and T2 exceeds that row's
correction gain, which would make curve assignment the larger error source.

| Row | Spread across the conditions | T0 correction gain | Ratio |
|---|---|---|---|
| DE | 0.00068 | 0.02897 | 0.023 |
| DK | 0.00164 | 0.06294 | 0.026 |
| UK | 0.00122 | 0.03114 | 0.039 |
| US | 0.00153 | 0.01247 | 0.123 |

Curve assignment, varied as far as this study varies it, is between 2.3% and
12.3% of what the bias correction is worth on the same row. **ERA5 bias is the larger
error source everywhere, and it is not close.**

## P4 and P5 refuted

**P4** predicted that T2's corrected interval would cover zero in Germany,
Denmark and the United Kingdom. It covers zero in the United Kingdom only.
Germany and Denmark both resolve, in opposite directions: Germany is 0.00112
worse under T2, Denmark 0.00064 better. The United States, reported beside it
by registration rather than gated, covers zero. So the prediction holds in one
of the three rows it named.

The prediction's substance survives its arithmetic. It said the correction
absorbs other-brand curve shape at monthly resolution, and the differences are
0.0006 to 0.0011 where the correction is worth 0.029 to 0.063. What it got
wrong was expecting that to be indistinguishable from zero with 1,000 draws
over thousands of units. **An effect can be negligible and resolvable at the
same time, and P4 conflated the two.** That is the same error P2b made in the
other direction.

**P5** predicted that uncorrected RMSE would resolve between at least two of
T0, T1 and T2 in at least two of Denmark, the United Kingdom and the United
States. It does so in Denmark only: T1 and T2 each differ from T0 (-0.00152,
-0.00164) but not from each other. In the United Kingdom no pair resolves, and
in the United States none does either. One of three, against the two it needed.

The reading is that **the three curve assignments produce simulated output that
is barely distinguishable even before any correction is applied**, which is a
stronger statement than the corrected nulls and makes them unsurprising.

## Denmark, the United Kingdom and the United States passed by three different accidents

The override tables were first built from the training fleet alone. A row fits
the training fleet and is scored on the test fleet, and the design says a
condition reassigns the units the runs use, so the table was the wrong object.
It was caught by the driver refusing T2 for Germany: 223 of the 2,211 units it
asked to move were not in the German test fleet.

The other three rows had passed, and the fleet shapes say why each one did:

| Row | Training fleet | Test fleet | In training only | In test only | Why it passed |
|---|---|---|---|---|---|
| DE | 4,288 | 4,814 | 333 | 859 | it did not |
| DK | 3,707 | 5,446 | **0** | 1,739 | strict subset |
| UK | 5,621 | 5,998 | **0** | 377 | strict subset |
| US | 1,091 | 1,276 | **2** | 187 | not a subset; neither of the two was a mover |

Denmark's and the United Kingdom's training fleets are strict subsets of their
test fleets, so a training-fleet table applied in full to both by a property of
those registers. The United States is not a subset at all and passed on a margin
of two units, which the rule happened not to move. Germany's fleet shrinks as
well as grows, and it was refused. **Three rows, three different accidents, and
none of them a check.** Had Germany run first the defect would have been visible
on the study's first turbine condition.

The tables now cover the union of both fleets and declare which fleet holds each
unit; the driver refuses an undeclared absence, a declared absence that is
present, and a phase the table reaches nothing in. Rebuilding added only
test-only units and changed no key: T1 DK +412, UK +259, US +30; T2 DE +253,
DK +432, UK +281, US +23.

**It was not cosmetic.** Under the old construction Denmark's T1 was +0.00054
away from where it is now and its T2 -0.00064, both resolving, which is the same
order as the differences G2a and G2b decide. Denmark would have answered both
gates differently, in opposite directions. That a tooling defect can silently
reverse a registered gate is the clearest argument in this study for a check
that refuses rather than records.

**United Kingdom T1 is identical under both constructions, to six decimals, and
the mechanism is worth stating.** All 259 units the new table adds are assigned
**the key they already carried**, so the override changes nothing for any of
them. The comparable counts are 193 of 409 added units in Denmark, 2 of 28 in
the United States, and zero of all of them under every T2, which is what an
other-brand rule should give. That is also a small finding about T1: for every
added British unit and for half the added Danish ones, reading the register's
own machine designation agrees with what specific power had already chosen.

## The US baseline had to be rebuilt, and a byte test could not verify it

The US scorecard row's evaluation directory holds no per-unit capacity-factor
frames; they are regenerable intermediates and had been pruned. The paired
bootstrap needs them, so the evaluation was re-run from the same training
directory and the result checked against the published `metrics.csv`.

**It did not match byte for byte, and the byte test could not say why.** The
regenerated file differs in two ways that have nothing to do with the
simulation: `metrics.csv` gained twelve diagnostic columns, and common-row
scoring, added after the 2026-08-24 refresh, scores all three variants on the
6,067 rows they share instead of each on its own 6,078, 6,068 and 6,069. Scored
the way v0.4.0 scored it, the regenerated run reproduces the published RMSE to
1e-12 in all three variants.

So the frames are the ones the published row used, and the byte comparison was
the right test to run and the wrong test to conclude from: **it cannot
distinguish a frame that differs from a convention that differs.** A
regeneration check needs a quantity that is invariant to the scoring
convention, which the 1e-12 rebuild is and the file hash is not. The US
baseline used here is the regenerated run, so that T0, T1 and T2 are all scored
on the same 6,067 rows; mixing conventions inside a paired interval is what
common-row scoring exists to prevent.

## What the training-side fit was worth, measured

The defect in the correction notice was repaired by re-running every affected
condition, which leaves both versions on disk and turns the repair into a
measurement nobody would have designed: **how much does it matter that the
correction's scalar is fitted on the condition's own simulation rather than the
baseline's?** The hybrid runs answer it directly, since they differ from the
corrected ones in that and nothing else.

| Condition | Row | Corrected RMSE, corrected fit minus hybrid | Excludes zero |
|---|---|---|---|
| C2 | IT | +0.01806 | yes |
| C2 | ES | +0.01186 | yes |
| C2 | BE | -0.01032 | yes |
| C2 | FR | +0.00932 | no |
| C2 | IE | +0.00848 | yes |
| C2 | PT | +0.00418 | no |
| C2 | SE | +0.00127 | no |
| C2 | NO | +0.00077 | no |
| T1 | US | +0.00057 | yes |
| T2 | US | -0.00057 | yes |
| T2 | DE | +0.00010 | yes |
| T2 | DK | -0.00008 | yes |
| T1 | DK | +0.00005 | yes |
| T1 | UK | +0.00001 | no |
| T2 | UK | -0.00000 | no |

**Up to 0.018 at country level, five of eight resolving; at most 0.0006 at
turbine level.** That is a clean statement of when the training-side fit
matters: it matters when a whole fleet changes curve at once, and it does not
when half a fleet moves between machines of roughly the same class. The country
rows change every unit's curve from a 167 W/m2 stand-in to something between
314 and 472, so the simulated capacity factor the scalar divides into moves
across the board; the turbine rows move 24 to 71% of capacity between real
machines whose curves are similar in shape, so the fleet aggregate barely
shifts.

The uncorrected side is identical between the two versions, to the digit, in
all fifteen. That is not a coincidence but the check on the diagnosis: no
factor enters an uncorrected figure, so a defect in the fit cannot reach one.

## P6 is refuted: the scalar was absorbing the curve error

**Prediction P6 said no condition would turn a clean fit degenerate. Six of
them do.** The threshold is the project's own: a scalar outside 0.2 to 3.0, or
an offset that did not converge.

| Condition | Row | Scalars outside 0.2 to 3.0 | The clusters, and what they became |
|---|---|---|---|
| C1 | IT | 0 → **8** | cluster 0 in all four seasons (1.741→3.668, 1.890→4.040, **2.649→7.133**, 1.868→3.739), cluster 1 summer (1.619→3.709), cluster 2 in autumn, spring and summer (1.544→3.212, 1.477→3.048, 1.742→4.179) |
| C1 | PT | 0 → 2 | cluster 0 autumn (1.655→3.392) and summer (1.485→3.088) |
| C1 | FR | 0 → 1 | cluster 7, fixed (2.027→4.606) |
| C1 | NO | 0 → 1 | cluster 0, fixed (1.315→3.048) |
| C2 | IT | 0 → **6** | cluster 0 in all four seasons, cluster 1 summer, cluster 2 summer |
| C2 | FR | 0 → 1 | cluster 7, fixed (2.027→3.839) |
| T1 | US | 5 → 6 | a row already daggered; its worst cluster, 38, moves 46.394→17.987 |

**The mechanism, stated precisely.** A wrong power curve is close to a
multiplicative bias on output: a 167 W/m2 machine reaches rated power at a far
lower wind speed than a 314 to 472 W/m2 one, so at any given wind it produces a
systematically higher capacity factor. The affine correction's wind scalar is
fitted as observed over simulated capacity factor, and a multiplicative bias on
output is exactly what a wind-speed scalar can compensate for, through the
curve's own steep response. So on the fallback the scalar had a real job: it was
cancelling a systematic power-curve mismatch, and it took a plausible value
doing so.

Give the fleet its own curves and that job disappears. The scalar has nothing
left to correct that the winds do not already carry, and it runs to values that
no longer describe a wind-speed correction at all: a scalar of 7.133 is not a
statement about Italian winds in summer.

**This is what the dagger has been marking.** The scorecard marks a degenerate
fit as a warning about that row's numbers. These conditions say the reverse is
also true: **a stable scalar was evidence that curve error was being absorbed,
not evidence that the fit was good.** Italy's four seasonal scalars sat between
1.74 and 2.65 on the fallback, inside the band and unremarkable, precisely
because they were carrying a curve mismatch of a size the band could
accommodate. The daggered rows and the country-level curve result are one
phenomenon seen from two sides: where the mismatch is large enough the scalar
leaves the band and gets a dagger, and where it is not the scalar stays in the
band and the mismatch is invisible.

That is the study's Q1 claim arriving through the fit rather than through an
error metric, and it is stronger evidence than A, because it is a statement
about what the correction was doing rather than about how much error it
removed. G1 fails on the ratio; this does not depend on the ratio.

**What it does not say.** It does not say the daggered scorecard rows are
carrying curve error specifically; each has its own diagnosis and the US row's
confounds are recorded elsewhere. It says a scalar inside the band is not
evidence that nothing is being absorbed.

## C1 was never re-run, and its results stand

C1 changes the curve library rather than the model keys, and
`vwf.data.load_power_curves` is called inside `train_set`, ahead of the
simulation the scalar is fitted from. So C1's training fit was always under the
condition, which the numbers confirm: C1's scalars move by 0.30 to 4.48 across
the eight rows, where every key-override condition moved its scalars by exactly
zero. No C1 run was re-executed and none of its figures changed.

## Every registered prediction but one is refuted

| # | Prediction | Outcome |
|---|---|---|
| P1 | C1 lowers uncorrected MBE in at least 6 of 8 countries | **Held**, in 8 of 8 |
| P2 | C1's corrected interval covers zero in at least 6 of 8 | **Refuted**: 5 of 8 |
| P2b | C2 moves further from C0 in the far tier than the near tier | **Refuted**: the far tier moves least |
| P3 | G1 passes | **Refuted**: G1 fails at 3 of 7 |
| P4 | T2's corrected interval covers zero in DE, DK and UK | **Refuted**: 1 of 3 |
| P5 | Uncorrected RMSE resolves between two of T0, T1, T2 in 2 of 3 rows | **Refuted**: 1 of 3 |
| P6 | No condition turns a clean fit degenerate | **Refuted**: 6 conditions do |

Seven predictions, one held. Four of the six refutations are of the same shape:
a prediction that a difference would be indistinguishable from zero, against a
difference that is negligible in size and resolvable against 1,000 draws over
thousands of units. That is a lesson about how the predictions were written
rather than about the curve library, and it is recorded in the pre-registration
so the next study's predictions name a size rather than a hypothesis test.

## The two halves of this study are not measuring the same thing

Q1's conditions replace a fallback curve that is in the fleet by accident. Q2's
T1 and T2 swap curves between real machines. C1's mechanism is that a 167 W/m2
stand-in over-produces at moderate wind speeds, and that mechanism has no
counterpart in Q2, where every condition is already on a plausible machine.

Q2 did return nulls, and **they do not contradict C1's result**. The two halves
answer different questions: Q1 asks what a curve that is wrong by a factor of
two in specific power does to a row, and Q2 asks what a curve that is wrong by
a brand does. The first is worth 0.08 to 0.22 in mean bias; the second is worth
at most 0.0012 in corrected RMSE. **Nothing here supports a single sentence
about "how much the curve library matters".** It matters in proportion to how
wrong the curve is, and the two halves of this study sit at opposite ends of
that range.

## Caveats

- One held-out test year per row: 2023 for the country rows, 2019 for Germany
  and the United Kingdom, 2020 for Denmark, 2022 for the United States.
  Everything is screening-level.
- C1 changes the library for the whole fleet at once. It cannot separate the
  curve's shape from its rating.
- The country rows fit under-determined offsets against one national series per
  month, which is a standing caveat on every figure here.
- Q2's nulls are at monthly resolution. A curve difference that cancels in a
  monthly mean need not cancel in an hourly one, and nothing here tests that.
- T1 reaches only the units whose register names a machine: 55.8% of Danish
  test-fleet capacity, 49.2% British, 27.5% American, and none in Germany. A
  null for T1 is a null about those units.
- T2 moves units within a rating band and to the nearest specific power, so it
  tests brand rather than class. It does not test what happens when a unit is
  put on a curve of the wrong size, which is the error the fallback made and
  which Q1 measures instead.
- The US row carries an unscreened ERCOT and SPP curtailment confound and five
  degenerate clusters in every condition, so its intervals are the weakest here.
- The P6 result is about what the scalar was doing, on eight country rows with
  one time slice each. It is not a general claim that every stable scalar in
  the scorecard is absorbing curve error.
- What the correction absorbs is measured on eight country-level rows with one
  test year each, at monthly resolution, and with offsets fitted against one
  national series. It is not a general statement about affine corrections.
