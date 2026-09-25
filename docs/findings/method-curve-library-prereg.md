# Curve library study: registered questions, conditions and gates

**Date:** 2026-09-11 (drafted); **revised 2026-09-13**, before any run of the
study. The revision is described in "What this revision changed" at the end,
and no result of this study existed when it was made.
**Scope:** the design of `method-curve-library.md`: what the power curve
library contributes to PyVWF's validation results, at country level and at
turbine level. Terms follow `CONTEXT.md`.

**Correction notice, 2026-09-25: the country-level fits with more than one
cluster used the per-cluster solver, not the joint national fit.**
`country_obs_is_per_cluster` read the rounding differences between clusters'
capacity-weighted means of one national observation as distinct zonal
observations, so every national fit with N greater than 1 fitted each cluster's
offset alone against the national series (fixed in `ac26f6a`;
`docs/findings/scorecard.md`, notice of the same date, has the details and the
joint-fit figures of the eight scorecard rows). Scalars, uncorrected figures and
N=1 fits are unaffected. The outcome entries recorded here for G1, P2, P2b and
P3 on the country rows rest on those fits and were not re-measured; the design
itself is unaffected.

**Everything below is fixed before any run of the study.** The outcome columns
are filled in afterwards. A condition added later is labelled post hoc and
cannot pass a gate.

## Questions

- **Q1, country level.** How much of the country-level correction absorbs the
  specific-power mismatch of the fallback curve, rather than ERA5 bias? Every
  country-level unit is simulated on `2019COE_DW100_100kW_27.6` (167 W/m2),
  while the grids name Vestas V80-2.0 (398 W/m2), V90-2.0 (314 W/m2) and
  V90-3.0 (472 W/m2). The country rows have never run on the licensed library,
  so that part of the matrix is new data, not a re-run. The answer is a finding
  whichever way it falls.
- **Q2, turbine level.** Is specific power a sufficient statistic for power
  curve shape? Every turbine-level row except BR is assigned curves by
  specific power, and a large share of capacity runs on other-brand curves:
  DE 40.0%, US 48.3%, UK 21.8%, DK 15.0% (corrected from 11.6%; see the
  reported-not-gated section). If held-out skill survives
  other-brand matching, the design choice is vindicated. If not, curve
  assignment is a larger error source than the ERA5 bias the correction exists
  to remove.

## Germany cannot answer Q2's own-machine half: a finding, measured in advance

T1 needs the register's own designation for a unit, and the three registers
that record one keep it in three different shapes: Denmark in a `model` column,
the United Kingdom packed into the manufacturer field ("Vestas V90 3000"), the
United States in `uswtdb_model`. **The German register has none.** `DE_md.csv` carries 82 columns: `Manufacturer`, `kW`, `Rotor..m.`,
`Tower..m.`, `StartDate`, and 76 monthly generation columns. Its commonest
entries are `Enercon`, `Vestas`, `Nordex` and `Tacke`, which are brands, not
machines. Measured coverage for DE is **0.0% of capacity, 0 of 11,433 units**.

This changes what the German row means. DE's other-brand share has stood as
the study's concrete case, 13.4% of its capacity being Vestas turbines on
Gamesa curves, and read as the clearest example of specific-power matching
going wrong. **It is not that.** There was nothing better to match against:
`add_models` reads a manufacturer, a rating and a rotor diameter and nothing
else, so with no designation recorded, the nearest specific power within a
fuzzy manufacturer match is the only assignment the data permits.

**Denmark's designation is dropped a layer earlier still, and Denmark's alone.**
`load_turbine_metadata`'s column allowlist for DK keeps ID, manufacturer,
capacity, diameter, height, lon, lat and location_type, and drops `model`. So
the Danish designation does not merely go unread by the matching: **it never
enters the pipeline**, and nothing downstream could use it if it wanted to. T1
reads `dk_md.csv` directly to recover it.

Checked for the other two, since the claim is about a route rather than a
register:

| Register | Designation | Route | Reaches the fleet |
|---|---|---|---|
| DK | `model` column | `load_turbine_metadata`, whose DK allowlist omits it | **no** |
| UK | packed into `manufacturer` | `load_turbine_metadata`, which keeps every column for UK | yes, inside the manufacturer field |
| US | `uswtdb_model` | the plant metadata, which does not use that loader | yes, in its own column |
| DE | none recorded | n/a | n/a |

So the United Kingdom's and the United States' designations arrive intact and
are simply never read; Denmark's is discarded on the way in. Three registers,
three states, and one common consequence: no designation reaches curve
assignment anywhere in PyVWF.

**One more thing the DK route does.** The same branch truncates the
manufacturer to its first word: "Vestas Wind Systems A/S" becomes "Vestas",
"NEG Micon" becomes "NEG", "Solid Wind Power A/S" becomes "Solid". The
curve-match audit and T2 read that truncated string, and for the brand
comparison it is harmless, since a library brand word still matches. It is
recorded because it is a silent alteration of source data on a path the study
depends on, not because it changes a figure here. Germany is a
data-availability limit, not a method limit, and the two carry different
consequences: a method limit would argue for changing the matching, and this
argues for a better register.

**It is not a limit on the other-brand question.** T2 needs no designation, so
G2b reaches all four regions including Germany. It is the own-machine question
alone that Germany cannot answer.

The scorecard's interpretation of the curve-match audit says otherwise today
and is corrected separately, with this measurement as its evidence.

## Reported, not gated

Three facts belong in the write-up whatever the gates do, and none of them is
a condition:

- **DK has 3.1% of capacity with no recorded manufacturer**, so its curve
  match cannot be verified in either direction for that share. *[Corrected
  2026-09-13: this said 23.0%, the figure the scorecard published. That figure
  counted every NEG Micon unit as unverifiable, because the loader truncates
  the Danish manufacturer to its first word and "neg" is a stop word in the
  audit. Denmark's manufacturers are recorded; part of each is discarded on the
  way in. The scorecard's DK row is corrected with it.]*
- **BR records no manufacturers at all.** It is excluded from Q2's conditions
  for that reason, not for its result.
- **CL is the extreme case of the thing this study is about and is not in it.**
  91.6% of its capacity runs on a reference curve, the highest share in the
  scorecard, and its row carries both markers: † for a degenerate fit and ‡ for
  a gain that cannot be distinguished from zero when its units are resampled.
  It is not gated because it runs on the bundled open library and there is no
  licensed condition to compare it against: the licensed library has no Chilean
  coverage to add. Leaving it out silently would look like selection, so it is
  named here, with its markers, and reported in the write-up.

## Libraries

- **Open library:** the library shipped with PyVWF, 76 curves. Results on it are
  third-party reproducible and are reported first.
- **Licensed library:** 160 curves, not redistributable, identified by sha256.
  Results on it are reported alongside, marked as not third-party
  reproducible.
- **Combined library:** the disjoint union of the two, 236 curves. Checked on
  2026-09-11: no model key is in both, and all 76 open curves in the combined
  library are byte-identical to the open library. So no precedence rule is
  needed for the current data. If a key ever appears in both, the combined
  runs are invalid for it until a rule is fixed.

## Baselines

**C0 and T0 are the rows standing in the scorecard today, not the figures they
superseded.** For the eleven European rows that means the re-runs of
2026-09-13 on `era5/EU_2026-09` with the per-timestep roughness
(`method-eu-rerun.md`); for the US, the only turbine-level row in this study
that is not European, it means its existing run.

The reason is not tidiness. The superseded rows carry a documented input
defect: three were simulated from winds extrapolated up to five degrees past
the ERA5 data, and the rest carried a roughness treatment the project has
since replaced. Measuring what a curve library contributes against a baseline
the repository no longer stands behind would make this study's own numbers
uncitable.

## Conditions

Every condition uses the row's scorecard configuration
(`configs/regions/scorecard/`), its training years and its test year. Runs go
under `output/curve_library_study_<date>/`, one region per process.

**Three things are held identical across the conditions of a row**, and a
condition that changes any of them is a different experiment:

1. **The variant set.** The `cluster_list` and `time_slices` of the row's
   configuration, unchanged. Every variant of a run is scored on the rows
   common to all of them, so adding or removing one moves the score of the
   variant the row reports: in the AR row, the reported `fixed_10` score moved
   because an unreported `season_10` variant lacked two rows. The variant set
   is part of the design, not incidental to it.
2. **The ERA5 input.** The path, the box and the roughness treatment of the
   row's configuration. DK carries `allow_extrapolation = true`, as its
   scorecard row does, because its box stops short of Bornholm.
3. **The fleet, the observations, the training years and the test year.**

**Q1: the eight country-level rows** (BE, ES, FR, IE, IT, NO, PT, SE):

| Condition | Curve library | Grid model keys |
|---|---|---|
| C0, as reported | open | as in the grids; every unit substituted to the fallback curve |
| C1, licensed | combined | as in the grids; each resolves to its own Vestas curve |
| C2, open best match | open | each Vestas key replaced by the nearest-specific-power open model within the 0.5 to 2 times rating band (`assign_curves_from_library`'s rule), **reported in three tiers, never pooled** |

### C2 is three experiments, not one, and the tiers are fixed here

Each country grid names exactly one key, and the open library's nearest in-band
match for those three keys is not equally near. The substitutes and their
distances were computed before any run
(`scripts/analysis/curve_library_tables.py`, tables in
`output/curve_library_study_2026-09-13/tables/`):

| Tier | Rows | Key | Substitute | Distance in specific power | Distance in rating |
|---|---|---|---|---|---|
| **near** | ES, IE | `Vestas.V90.2000`, 314 W/m2 | `VestasV82_1.65MW_82` | -1.9 W/m2 | -350 kW |
| **moderate** | FR, IT, PT | `Vestas.V80.2000`, 398 W/m2 | `EWT_DW58_1MW_58` | -19.4 W/m2 | -1,000 kW |
| **far** | BE, NO, SE | `Vestas.V90.3000`, 472 W/m2 | `NREL_Reference_5MW_126` | -70.6 W/m2 | +2,000 kW |

**C2 is reported per tier, and any gate touching C2 is evaluated per tier and
never across all eight countries.** Pooling them would average a swap of
1.9 W/m2 against a 5 MW reference design standing in for a 3 MW machine, and
the spread that produced would read as a finding about the open library when it
is a finding about which substitute each country happened to land on. The tiers
are fixed here, from the distances above, before any C2 run exists.

*[Correction, 2026-09-13: this section predicted the nearest in-band open match
for the V90-3.0 would be `BAR_HighSP_5.0MW_134.9` at 350 W/m2 against 472. The
curve is wrong and the direction is right: the nearest is
`NREL_Reference_5MW_126` at 401 W/m2, so the gap is 70.6 W/m2 rather than the
122 predicted. The point the prediction was making, that the open library has
no close match for the highest-specific-power key, stands with a smaller
number.]*

**Q2: the four turbine-level rows that depend on the licensed library** (DE,
DK, UK, US):

| Condition | Model keys |
|---|---|
| T0, as reported | the current curve assignment |
| T1, brand-and-spec match | a unit whose own model string maps exactly to a licensed-library key gets that key; every other unit keeps T0 |
| T2, other brand | a unit whose T0 curve is same brand gets the nearest-specific-power model of a different brand in the same rating band; every other unit keeps T0 |

T1's coverage, the share of capacity it reassigns, was measured before this
record was fixed, since it decides which regions the gate can reach and needs
no simulation (`scripts/analysis/curve_library_match.py`, whose rules are
fixed and unit-tested):

| Region | T1 coverage, by capacity | Units matched | Library keys reached | Designation read from | G2a |
|---|---|---|---|---|---|
| DK | 65.9% | 2,003 of 3,707 | 31 | the register's `model` column | gated |
| UK | 47.1% | 3,288 of 5,621 | 60 | packed into the manufacturer field | gated |
| US | 32.0% | 309 of 1,091 | 34 | `uswtdb_model` | reported, ungated by rule |
| DE | **0.0%** | 0 of 4,288 | 0 | nothing: none is recorded | **untestable** |

*[Corrected twice on 2026-09-13. First, before any run: an earlier version of
this table gave DK 50.2% and UK 53.3%, measured over each register in full. The figures above
are over the training fleet each row actually fits, which is what a condition
reaches and therefore what the precondition should be read against. Both still
clear 10% and G2a's scope is unchanged. Germany's denominator changes with
them, from the register's 11,433 units to the fleet's 4,288, and its coverage
stays zero. Those stale figures survived in G2a's own text until the second
correction below, which is worth saying plainly: the table was fixed and the
gate that reads it was not, so the precondition was stated against numbers the
document had already withdrawn. Neither reading changes which regions G2a
covers.

Second, after the seven T conditions were re-run on tables covering both
fleets (see the construction fix at the end of this document), the figures
above are joined by the test-fleet ones, since the gate scores the test fleet:

| Condition | Region | Training fleet | Test fleet |
|---|---|---|---|
| T1 | DK | 65.9%, 2,003 of 3,707 | 55.8%, 2,415 of 5,446 |
| T1 | UK | 47.1%, 3,288 of 5,621 | 49.2%, 3,547 of 5,998 |
| T1 | US | 32.0%, 309 of 1,091 | 27.5%, 339 of 1,276 |
| T2 | DE | 51.1%, 2,211 of 4,288 | 44.4%, 2,241 of 4,814 |
| T2 | DK | 64.7%, 1,844 of 3,707 | 56.3%, 2,276 of 5,446 |
| T2 | UK | 70.7%, 3,736 of 5,621 | 71.3%, 4,017 of 5,998 |
| T2 | US | 28.6%, 280 of 1,091 | 24.1%, 303 of 1,276 |

Every row still clears the 10% precondition on both fleets, so no gate's scope
moves.]*

**G2a therefore covers DK and UK.** Unverifiable units cannot enter T1 or T2
and keep T0.

*[Added 2026-09-13, before any run: what a G2a result would mean. `add_models`
reads a manufacturer, a capacity, a rotor diameter and a hub height, and no
model designation, for any region. Specific-power matching is therefore not a
fallback the code reaches when designations are missing; it is the only
assignment implemented. **T1 is not a variant of the method so much as the
first use of a column the data already carries**, for the Danish and British
registers that record one. So if T1 beats T0 on DK or UK, the remedy is not a
better heuristic: it is reading a field that was already there.

"Already there" is bounded, and the bound was checked rather than assumed. The
Danish designation is a field of the register itself: the processing renames a
`Model` column straight out of it and maps the Danish "Ukendt" to "Unknown".
But PyVWF's own fleet metadata has carried it only since 2026-02-13, when the
column entered `process_dk_raw_data.py` (`git log -S`, against a history
reaching back to 2022-06-22). So it was in the data throughout and in this
pipeline for seven months. The published method's description of the Danish
metadata, which lists manufacturer, capacity, rotor diameter and hub height
and no designation, was accurate when it was written. That changes
what a G2a result implies without changing the gate, which is why it is
recorded here and dated rather than written into the gate's text.]*

## Scoring

**Every comparison is on common rows, twice over.** Within a run, the harness
already scores every variant on the rows all of its variants can score. Across
conditions, the study restricts to the rows common to all conditions of a row
before any difference is taken, as the roughness study and the European re-run
comparison do. A condition that loses rows the others keep is reported as
having lost them; it is never compared on its own surviving subset.

**Per row and condition, from `metrics.csv` and the run records:** uncorrected
RMSE and MBE; corrected RMSE and MBE for the row's reported configuration; the
correction gain; the scalar and offset distributions and `fit_quality`; the
fit diagnostics, which say what share of a cluster's training steps its fitted
pair sends below 0 m/s or above the curve; the off-curve and missing-value
counts; and the curve resolution with the curve-match audit classes.

**A condition that turns a clean fit degenerate is a result, not an artefact.**
If swapping the curve library pushes a fit outside the scalar bounds, or makes
an offset fail, that is evidence about how much the library carries: the
correction was absorbing curve error, and removing the error left the fit
without a stable solution. It is reported per condition as an outcome of the
study, with the cluster and the values, and the row that produced it keeps its
dagger in the write-up.

## The floor

**The primary test is the paired interval, not a threshold.** Procedure B of
this study: 1,000 paired draws, seed 20260911, 95% percentile intervals,
resampling units for turbine-level rows and months for country-level ones,
with the same draws for every condition of a row. A difference is resolved if
its interval excludes zero.

**There is no country-level floor.** Nothing has established one.

**The 0.002 turbine-level screen stands on its original basis and nothing
else:** ten null clustering A/B pairs with a maximum absolute difference of
0.0018 in MAE, with a second test year and seed repeats named as prerequisites
and never done. It is a screen for whether a resolved turbine-level difference
is worth acting on, and it is not a gate.

**The 3% figure is withdrawn.** The draft of 2026-09-11 made every gate a
comparison against "about 3%", citing `region-us-br.md`. That figure was never
established as a limit, it was relative where the gates needed absolute, and
using it would have made this study's conclusions rest on a number nobody
measured.

## Gates

Every gate has an indeterminate outcome, and indeterminate is a result: it
means the design could not answer the question, and the question stays open on
its own evidence rather than being settled by a narrow reading.

| Gate | Requirement | Outcome |
|---|---|---|
| **G1** (Q1) | Uses C0 and C1 only, which are not tiered: C1 gives every country its own curve, so the countries differ in fleet rather than in how near a substitute fell. For each country, the absorbed share is A = (uncorrected RMSE in C0 minus uncorrected RMSE in C1) divided by the correction gain in C0. **The scoreable set is fixed here, from C0, at seven:** BE, ES, FR, IE, IT, PT and SE. NO is excluded because its C0 correction gain is not positive, so there is no denominator to divide by. The fallback curve is a material part of the country-level correction if A is at least 0.5 in at least **4 of those 7**. **Indeterminate** if the interval on the C0-minus-C1 uncorrected RMSE difference includes zero in more than 3 of the 7, since A is then built on differences the design cannot resolve. | **Fails.** A is at least 0.5 in 3 of the 7: FR 0.95, IE 0.84, BE 0.68, then SE -0.19, PT -1.48, IT -1.57, ES -22.38. Not indeterminate: every uncorrected difference resolves. |
| **G2a** (Q2, T1) | **DK and UK only.** Specific power is sufficient for a region if the paired interval for corrected RMSE in T1 minus T0 includes zero, or excludes it by less than the 0.002 screen. It is insufficient if T1 beats T0 by more than the screen. **Precondition, measured before this record was fixed:** the gate applies only where T1 reassigns at least **10% of capacity**. DK (65.9% of the training fleet, 55.8% of the test fleet) and UK (47.1% and 49.2%) pass it; **DE is untestable at 0.0%**, for the reason above, and is reported as such rather than as a failure; **the US is ungated regardless of its coverage** (32.0% and 27.5%), which is reported as a finding about the licensed library's reach into the US fleet. **A T1 materially worse than T0 suspends this gate** pending a diagnosis of the matcher: a matcher that assigns a worse curve than specific power did is more likely to be wrong than specific power is to be right. | |
| **G2b** (Q2, T2) | **All four regions, DE included.** The same test for T2 minus T0. Specific power survives other-brand matching in a region if the interval includes zero or excludes it by less than the screen. T2 reassigns by rating band and specific power and needs no register designation, so Germany's missing designations do not reach it. | **Passes, all four rows.** T2 minus T0: DE +0.00112, DK -0.00064, UK -0.00008, US -0.00049, every one inside the screen. DE and DK resolve, in opposite directions. |
| **G3** (Q2) | Curve assignment is a larger error source than ERA5 bias for a region if the spread of uncorrected RMSE across T0, T1 and T2 exceeds that region's correction gain in T0. **Indeterminate** for a region whose T1 is suspended under G2a. | **Fails, all four rows.** Spread of uncorrected RMSE across the conditions against the T0 correction gain: DE 0.00068 vs 0.02897, DK 0.00164 vs 0.06294, UK 0.00122 vs 0.03114, US 0.00153 vs 0.01247. Curve assignment is 2.3% to 12.3% of the correction's worth. No region is indeterminate; G2a was not suspended. |

*[Limitation, recorded 2026-09-13 after C1 ran and before any other condition:
G1 divides by C0's correction gain. The rule anticipated a gain that is not
positive, which is why Norway is excluded; it did not anticipate one that is
positive and negligible. Spain's is 0.0019, which makes its A a ratio to noise
of arbitrary sign and size, and Italy's and Portugal's of about 0.05 carry the
same weakness in milder form. **No floor is introduced**, since any floor
chosen now is chosen knowing which countries it admits. A is reported with its
denominator beside it instead, so a reader discounts those rows themselves
rather than having them discounted invisibly. G1's arithmetic is unchanged and
it fails at three of seven.]*

**The scoreable set is determined from C0 alone, and does not move.** C0 exists
today; C1 does not. Deciding scoreability from C1 would let the denominator set
shrink after the results are seen, and a threshold of four would stop meaning
what it meant when it was fixed.

**If a C1 run turns a country's correction gain negative, that is a result and
not an adjustment to this gate.** It would mean that giving the fleet its own
curves left the correction with nothing to repair, or something to make worse,
which is evidence about the curve library. It goes in the findings with its
figures. The gate's arithmetic stays on the seven named above.

A negative A, where the licensed curve raises the uncorrected error, is
reported as such.

## Registered predictions

| # | Prediction | Outcome |
|---|---|---|
| P1 | In C1, uncorrected MBE falls in at least 6 of the 8 countries, because a 167 W/m2 curve overproduces at moderate wind speeds. | **Held**, 8 of 8, by 0.082 to 0.220. |
| P2 | In at least 6 of the 8 countries, the paired interval for corrected RMSE in C1 minus C0 includes zero: the correction absorbs the curve error either way. | **Refuted**, 5 of 8. ES, FR and IT resolve, all worse under C1. |
| P2b | C2's corrected RMSE moves further from C0 in the far tier than in the near tier. The near tier's substitute sits 1.9 W/m2 away and the far tier's 70.6 W/m2, so a C2 result that does not order that way would say the distance in specific power is not what drives the difference. | **Refuted**, and inverted: mean absolute difference is 0.0024 far, 0.0063 near, 0.0105 moderate. The distance does order the tiers in the uncorrected mean bias and does not reach the corrected RMSE. |
| P3 | G1 passes. | **Refuted**: G1 fails at 3 of 7. |
| P4 | In T2, the paired interval for corrected RMSE minus T0 includes zero in DE, DK and UK: at monthly resolution the correction absorbs other-brand curve shape. T2 reaches all four regions, so this is a prediction about three of them and the US is reported beside it. | **Refuted**, 1 of 3: UK covers zero, DE and DK resolve in opposite directions. US, reported beside it, covers zero. The differences are 0.0006 to 0.0011 against correction gains of 0.029 to 0.063, so the prediction's substance held and its arithmetic did not. |
| P5 | The paired interval for uncorrected RMSE excludes zero between at least two of T0, T1 and T2, in at least 2 of the 3 regions where all three conditions exist (DK, UK, US). Germany has T0 and T2 only. | **Refuted**, 1 of 3: DK only, where T1 and T2 each differ from T0 but not from each other. |
| P6 | No condition turns a clean fit degenerate. | **Refuted**: 6 conditions do. C1 at IT (0 to 8), PT (0 to 2), FR and NO (0 to 1); C2 at IT (0 to 6) and FR (0 to 1); T1 US 5 to 6. The registered consequence applies and is written up as the study's strongest result. |

## Consequences, stated in advance

- **G1 passes:** the country-level scorecard rows and `method-country-level.md`
  say that the correction there largely repairs a curve mismatch. The
  correction notice stops saying the magnitude is unquantified.
- **G1 fails:** the curve substitution is recorded as a defect with a small
  effect on the country results. The notice gives the measured share.
- **G1 indeterminate:** the notice says the magnitude is still unquantified and
  states the intervals that failed to resolve it.
- **G2a or G2b fails:** specific-power matching becomes a leading caveat on the
  turbine-level scorecard rows.
- **Both pass:** the design choice is recorded as supported, for the regions
  that met the coverage precondition, at monthly resolution only.
- **G2a suspended:** the matcher is diagnosed before anything is concluded, and
  the suspension is reported whether or not the diagnosis succeeds.
- **G3 passes for a region:** that region's scorecard row carries it.
- **P6 fails, and a condition turns a clean fit degenerate:** this is the
  study's central claim arriving by another route, not a nuisance to note. A
  fit that was stable on a 167 W/m2 fallback and is unstable on the fleet's own
  curves was stable because it was carrying curve error: the scalar and offset
  had a mismatch to absorb, and removing the mismatch left them without one.
  The write-up says that in those terms, names the clusters and the values, and
  the row keeps its dagger. It would corroborate G1 independently of A, since
  it is a statement about what the fit was doing rather than about how much
  error it removed.

## Committed in advance

- No region is dropped after its results are seen. All eight country rows and
  all four turbine rows are reported.
- Open-library results lead. Licensed-library results follow, marked as not
  third-party reproducible.
- A deviation from this design is recorded, with its date and whether it came
  before or after a result was seen.
- No scorecard figure changes on this study alone. A condition is not a new
  row.

## Tooling this needs

This is new code, written after this record is committed and before any run:

- a study driver, `scripts/analysis/curve_library_study.py`. It builds a
  variant input root per condition under `output/`, with only the changed
  files copied and everything else linked, then runs train and evaluate for
  each row, one region per process. The link set now includes
  `era5/EU_2026-09`, and the driver checks that each condition's manifest
  records the same ERA5 path and roughness treatment as the row's scorecard
  configuration, so a condition cannot change the input by accident;
- an own-model matcher for T1. It normalises the registers' model strings
  (DK, DE, UK) and USWTDB's (US) to licensed-library keys, by exact normalised
  match only. The rules are fixed in the driver before the T1 runs, and unit
  tests check them on sample strings;
- the other-brand assignment for T2, reusing the rating band and nearest
  specific power;
- a comparison driver reusing the paired bootstrap and the common-row
  restriction already written for the roughness study and the European re-run,
  rather than a third implementation of the same arithmetic.

Estimated compute: about 90 minutes in total.

## What this revision changed, 2026-09-13

The draft was written on 2026-09-11 and predates eight changes to the project.
No run of this study existed when it was revised, so nothing here is a
post-hoc adjustment to a result; every change is dated and listed.

1. **The 3% noise floor is withdrawn**, and the paired interval of procedure B
   replaces it as the primary test. The draft had no resampling at all: its
   gates compared point estimates against a figure never established as a
   limit. G2, G3, P2, P4 and P5 all referenced it and are rewritten.
2. **Common-row scoring is now a stated condition**, within a run and across
   conditions. It did not exist when the draft was written.
3. **The variant set is registered** as part of each condition, after the AR
   row showed an unreported variant moving a reported score.
4. **C0 and T0 are the current rows**, not the superseded ones, for the reason
   given under Baselines.
5. **The extent guard and the roughness treatment** are held as part of each
   condition, including DK's opt-in. Neither existed in the draft.
6. **Degeneracy under a condition is registered as a result**, with the fit
   diagnostics and off-curve counts added to the recorded metrics.
7. **Chile is named in reported-not-gated**, with both its markers and the
   reason it is not a condition.
8. **T1 coverage was measured before the record was fixed**, which the draft
   left to the run. It decides which regions G2a can reach, and needs no
   simulation. G2a is now DK and UK; DE is untestable at 0.0% coverage,
   recorded as a finding about the German register rather than as a failed
   condition; the US keeps its coverage reported and its gate off by rule.
9. **The rulings taken after the draft are written in:** T1 renamed the
   brand-and-spec match; the 10% coverage precondition with the US ungated
   regardless; G1 over scoreable countries, 4 of 7 rather than 5 of 8, since
   NO has no positive gain to divide by; G2 split into G2a and G2b; a T1
   materially worse than T0 suspending its gate; an indeterminate branch on
   every gate; P2 restated as an interval; and P6 added for the degeneracy
   outcome.

## Construction fix, 2026-09-13: an override table covers both fleets

This is a fix to the tooling, not a change to the design. The design says a
condition reassigns the units a row's runs use, and a row uses two fleets: it
fits the training fleet and is scored on the test fleet. The table builder
read only the training fleet, so at evaluation every unit installed after the
training window kept its T0 key. That is not the condition registered above,
and it was the builder's defect rather than the design's.

It surfaced because the driver's override check refused T2 DE: 223 of the
2,211 units the table asked to move were not in the German test fleet. The
refusal is the only reason this was found before the results were written; a
check that recorded the mismatch instead would have left a condition reaching
90% of its units looking like an ordinary result.

**Why the other three rows passed, measured rather than reasoned.** Both
fleets of all four turbine rows, taken from the run standing in the scorecard
and from `val_set`'s own route:

| Row | Training fleet | Test fleet | In training only | In test only | Train-only units the old table asked to move |
|---|---|---|---|---|---|
| DE | 4,288 | 4,814 | 333 | 859 | 223 |
| DK | 3,707 | 5,446 | 0 | 1,739 | 0 |
| UK | 5,621 | 5,998 | 0 | 377 | 0 |
| US | 1,091 | 1,276 | 2 | 187 | 0 |

Denmark's and the United Kingdom's training fleets are strict subsets of their
test fleets, so a table built from the training fleet applied in full to both.
That is a property of those two registers, not a property of the construction:
their passes were structural luck. The United States is not even a subset, and
passed on a margin of two units, neither of which the rule moved. Germany's
fleet shrinks as well as grows, and it was refused. Had Germany run first, the
defect would have been visible on the first condition.

**The same-unit-assigned-differently case does not occur here, and is refused
if it ever does.** A unit in both fleets could in principle carry different
values on the two sides, and the rule would then give it two keys, one per
phase. Measured across all four turbine rows and all eight country rows on
2026-09-13: no unit in both fleets differs in `model`, `capacity`, `diameter`,
`height`, `n_turbines`, `uswtdb_model`, `lon` or `lat`, and no fleet holds a
duplicate ID. So no tie-break rule exists, and none is needed. The builder
refuses rather than picking: `load_fleets` compares the two fleets on the
fields the condition's own rule reads and raises before any table is written.
The field list is per condition because the rules differ, and that distinction
is load-bearing.

**The Belgian capacity case, recorded because it will matter later.** A country
grid point records capacity per year, so the same point is a different object
in the two fleets: 14 Belgian points hold 0 MW in the training fleet and 11 to
18 MW in the test one. The union takes the training row. C2 maps one model key
to a substitute and reads no capacity, so this never reaches C2, and the union
construction is therefore safe at country level **by the shape of this
condition rather than by anything the grids guarantee**. A later country-level
condition that reads capacity, a rating band or a specific power will be
refused by `load_fleets`, and that refusal is the correct behaviour: the two
fleets genuinely disagree about the object, and whoever writes that condition
has to say which year's capacity it means instead of inheriting the training
year's by silence. Nothing is fixed here in anticipation, because there is no
way to know now which answer such a condition would want. The same difference
in a turbine row would be a conflict for T2, which picks a rating band, and
would stop it.

**What the fix is.** The table now covers the union of the two fleets and
carries `in_train` and `in_test` per unit. The driver applies the rows the
phase's fleet holds, and the check refuses on any unit missing that was not
declared absent, on any unit declared absent that is in fact present, and on a
phase the table reaches nothing in. The check is therefore no weaker than the
one that caught Germany: a table keyed on the wrong type still misses every
unit, and every one of those misses is undeclared.

**Registered before the rerun.** The new tables differ from the old ones by
addition only: no key changed and no unit was dropped, and every row added is
a test-only unit (T1 DK +412, UK +259, US +30; T2 DE +253, DK +432, UK +281,
US +23). So:

- the **training** side of DK, UK and US is expected to be unchanged, to the
  byte, in `factors_*.csv` and in the training fleet: the units the table
  reaches there are exactly those it reached before;
- the **evaluation** side of DK, UK and US is expected to move, because the
  test fleet now carries the condition on the units that joined it. The
  direction is not predicted. A result that did not move would mean the added
  units are inert, which is a claim needing its own evidence;
- **T2 DE becomes runnable** for the first time, with 2,464 movers, 2,241 of
  them in the test fleet.

**The country rows are untouched.** All eight have identical training and test
fleets, unit for unit and key for key, so the C2 tables rebuild byte-identical
in their mapping and every unit is flagged present in both. C1 and C2 stand as
run; neither is re-run.

**Outcome of the rerun, 2026-09-13.** All seven T conditions ran, T2 DE for the
first time. Every training side is byte-identical to the run it replaced:
`factors_*.csv`, `fit_diagnostics_*.csv`, `train_turb_info_*.csv` and
`curve_resolution.csv` compare equal for all seven, 32 files in total, against
the preserved single-fleet runs under `T1_train_fleet_only/` and
`T2_train_fleet_only/`. The prediction that the training side would not move
therefore held, and the evaluation-side results are reported in the finding.

## How these predictions were written, 2026-09-13, after all of them resolved

Seven predictions, one held. Four of the six refutations share a shape, and it
is a shape in the predictions rather than in the world.

P2, P2b, P4 and P5 each predicted that a difference would be **indistinguishable
from zero**. The differences they were about are 0.0006 to 0.014, against
corrections worth 0.012 to 0.32 on the same rows: negligible by any standard a
reader would apply. They are also resolvable, because the paired bootstrap draws
1,000 times over thousands of units and can separate 0.0006 from zero without
difficulty. **Negligible and indistinguishable are different claims, and every
one of these four predicted the second while meaning the first.**

P2b has its own version of the same error, already recorded under the
conditions rules: it named corrected RMSE when its mechanism lived in the
uncorrected mean bias.

For the next study, a prediction about an effect being small states a size and
a direction, not a hypothesis test: "the difference is under 0.002 in corrected
RMSE", which the 0.002 screen in G2a and G2b did, and which is why both gates
gave a usable answer where four predictions about the same numbers did not. A
prediction that an interval covers zero is only worth registering where zero is
genuinely the interesting value, which for a correction's residual it is not.
