# Q: a fresh pre-specified gate on configuration D

**Written before the experiment was run. Results are appended below, not above.**

## Why this is needed

Configuration D (MLP heads, nine training regions) reaches mean zero-shot skill
+0.338 against the gated configuration A's +0.238. But D was **chosen after
seeing E5 and E7**, both scored on the same five holdout regions, so its
advantage on those five is not independent evidence. `method-physics-informed-
evaluation.md` says so and asks for a fresh gate before D is used as a headline.
This is that gate.

## What makes it fresh

Four regions have caches in this repository and have **never been scored as
holdouts**: New Zealand, Chile, Argentina and the Australian NEM. They entered
E7 and E8 only as members of the training pool. No choice about D -- not the MLP
heads, not the nine-region pool, not the epoch count or seeds -- was informed by
any held-out result on these four.

They are also the hard cases, not a soft target, and the numbers saying so were
measured before this gate:

| region | test units | physiographic coverage | median NN distance |
|---|---|---|---|
| AR | 59 | 91.5% | 0.56 |
| AU-NEM | 83 | 89.2% | 1.19 |
| CL | 59 | **33.9%** | 2.95 |
| NZ | **12** | **33.3%** | 2.46 |

Chile and New Zealand are the least-covered regions in the whole study, below
even the UK and US at ~50%. The existing scorecard independently caveats CL and
AR as "not headline wins" (ERA5 exaggerates their north-south wind gradient),
flags both as degenerate affine fits, and notes AU-NEM's curtailment confound.
New Zealand has twelve farms.

## Protocol

Leave-one-region-out over {NZ, CL, AR, AU-NEM}. For each, train on all eight
other regions and apply zero-shot to its held-out test year. Scored with the
harness's own `skill_metrics`, capacity weighted, on the same test years the
scorecard uses. Three seeds. Configuration D exactly as it stands: MLP heads
(hidden 16), 60 epochs, `profile="power"`, no wake term, no density term, no
abstention.

## Gates

**Q1 (primary).** Zero-shot corrected RMSE < uncorrected RMSE in **at least 3
of 4** regions, and **no region degraded by more than 10%** relative. Same form
as P1, so the two are comparable.

**Q2 (does D's advantage reproduce).** D beats configuration A -- linear heads,
trained on the original five regions only -- in **at least 3 of 4**. A's training
pool never contains the holdout, so this is a like-for-like zero-shot contest
between the gated configuration and the post-hoc one, on regions neither was
tuned against.

## Reference points, never gates

Uncorrected ERA5, and the in-region affine correction from the refreshed
scorecard: NZ 0.157 to 0.106, CL 0.123 to 0.105, AR 0.151 to 0.133, AU-NEM 0.115
to 0.094. The Chilean and Argentine rows are daggered as degenerate fits.

## Predictions, registered

1. **Q1 is genuinely at risk.** I expect it to land at exactly 3/4 or fail at
   2/4, because two of the four regions sit at ~33% coverage and coverage has
   predicted skill everywhere so far.
2. **Skill will rank AR > AU-NEM > CL ~ NZ**, following coverage rather than
   fleet size or data quality.
3. **Q2 passes.** The nine-region pool and the extra head capacity were both
   worth more in the low-coverage regions of the original five (UK and US gained
   most), and these are lower-coverage still.

**What failure means.** If Q1 fails, configuration D does not generalise beyond
the regions it was chosen on, and the headline reverts to configuration A's
gated +0.238. If Q2 fails, D's advantage was an artefact of the five regions it
was selected against and the linear five-region default should stand.

Nothing below this line was known when the above was written.

---

## RESULTS

*(appended after the run)*
