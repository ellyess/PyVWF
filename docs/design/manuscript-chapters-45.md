# Merging thesis chapters 4 and 5: what has to be decided

**Date:** 2026-09-13
**Status:** open. Nothing here is decided, and nothing has been ported or run.
**Scope:** the manuscript merging the gridded-interpolation chapter and the
machine-learning chapter, co-authored, drafted against results produced from
this repository as it stands. The chapters themselves are accepted, on Spiral,
read-only and outside this repository. Terms follow `CONTEXT.md`.

This document records the decisions the merge needs and the evidence each one
rests on, so that they are made once and on purpose. Phase 0 inventoried what
the chapters did; phase 1a examined the control points they rest on; phase 1b
surveyed what porting their code would take.

## The shape of the problem

**The chapters' code is not in this repository.** It was removed from `main` on
2026-07-06 (commit `e8208e1`, "Scope reduction (scripts): move grid/ml driver
scripts to development") and survives on the local `development` branch, whose
head `f2ed688` of 2026-06-17 is 315 commits behind the research branch. What
this branch implements of chapter 4 is nearest-centroid assignment, which is
step 2 of the chapter's eight; `src/vwf/harness/export.py` says so in its own
docstring, naming IDW as "a future refinement, not a claim this file makes".

So this is a port, not a re-run, and the manuscript's first decision follows
from that.

## D1. Does the manuscript reproduce the chapters or supersede them?

Reproducing means porting about 3,000 lines of library code and 5,200 lines of
driver scripts to run against today's pipeline, and accepting that the numbers
will move because the pipeline has changed underneath them. Superseding means
taking the chapters' questions and answering them on today's rows, with the
chapters cited for what they claimed.

Evidence that bears on it is in D5 and D6: the turbine-level half of the
control-point pool is stable to almost everything that has changed, and the
country-level half is not comparable without more work.

## D2. The Netherlands

Chapter 4 includes NL as one of nine country-level configurations, and NL is
one of its two headline demonstrations of cross-border borrowing: kriging MAE
0.056 against a cluster-based 0.116. **This project excludes the Netherlands.**
`CLAUDE.md` records an ENTSO-E coverage defect that caps the Dutch capacity
factor at 0.57, which no rescaling fixes.

Either the manuscript restores NL and states the defect beside the number, or
it loses the cross-border example and needs another. Restoring it and stating
it is the better option if the number survives the defect being named, which
needs the numbers rather than a preference.

## D3. Cluster counts

The chapter's control points are DK onshore 884, DE onshore 500, UK onshore
293, and its validation tables quote best counts of DK 700, DE 500, UK 300.
Today's scorecard rows are DK k=100, DE k=100, UK k=50. **Reproducing the
chapter and using today's rows are different papers**, and the difference is
not cosmetic: the number of clusters sets the spatial density of the control
points, which is the independent variable the whole interpolation argument
rests on.

## D4. Per-farm against per-turbine

Chapter 4 says in its text that Germany and the United Kingdom are per-farm and
Denmark per-turbine, and its Table 1 labels all three "Per-turbine capacity
factors". This is an internal contradiction in an accepted chapter. It should
be settled from the data and stated once in the manuscript, whatever else is
decided, because it changes what a "control point" is in two of the three
turbine-level rows.

## D5. The country-level tier is not comparable without more work

Forty of the 1,729 control points are country-level, and they cover the largest
areas. The concern was that they are fallback-curve artefacts: the curve
library study established that a country row whose grid names a curve the
library lacks has every unit simulated on a 167 W/m2 fallback, and that the
fitted wind scalar then absorbs the mismatch while staying inside the plausible
band, so no degeneracy rule catches it.

**The evidence available today points away from that.** Placing the chapter's
scalars on the axis between today's fallback fit (C0) and today's real-curve
fit (C1), as `t = (chapter - C0) / (C1 - C0)`, gives a median `t` of 0.81 by
means and 0.93 by medians across the eight rows. The chapter's country tier
sits nearer the real-curve fit than the fallback fit, and in four of eight rows
it falls outside the interval altogether. The likely explanation is dating:
country-level runs have used the default input root only since July 2026, and
the chapter's runs predate that, so their grid keys would have resolved in the
licensed library.

That is an inference, not a record. **The chapter-era outputs carry no
manifests**, so the library cannot be read from provenance. Four things differ
at once between those runs and today's: cluster count (BE 3 against 12, IT 3
against 12), the ERA5 extent, the training years, and the curve library era.
What would separate them is a single row re-run at the chapter's cluster count
on today's code, which is a phase 2 decision and not free.

## D6. The turbine tier reruns cleanly, and that is a finding

| Row | Chapter mean scalar | Today's mean scalar | Chapter clusters | Today's k |
|---|---|---|---|---|
| DK onshore | 0.789 | 0.780 | 884 | 100 |
| DE onshore | 0.916 | 0.900 | 500 | 100 |
| UK onshore | 0.974 | 0.882 | 293 | 50 |

**Across an eight-fold change in cluster count, three years, a rebuilt
pipeline, a changed roughness treatment and a re-run ERA5 archive, the mean
fitted scalar of each turbine-level row moves by 0.009, 0.016 and 0.092.** The
turbine-level tier is 1,689 of the 1,729 control points and carries one
degenerate fit between them.

This is the strongest evidence the manuscript has that the turbine half of
these chapters can be reproduced on today's code and give the same answer. It
also bounds what a re-run can be expected to change: not the corrections
themselves, but what is built on top of them.

## Open discrepancies that block specific claims

- **The IDW-against-kriging contest is untested against the degenerate control
  points.** `pykrige` is not installed here, so only IDW and nearest neighbour
  were re-scored when the five degenerate points were dropped. IDW moved 1.8%
  and the chapter decides the contest on a 3% gap in scalar MAE. Untested, not
  unaffected.
- **A reimplementation of the chapter's spatial cross-validation matches it on
  offset and not on scalar**: offset MAE 0.632 against the published 0.641,
  scalar MAE 0.180 against 0.161. The likely cause is the onshore and offshore
  separation the chapter's script performs and the reimplementation does not.
  **If the manuscript reproduces any chapter 4 figure, this has to be resolved
  first**, because it is the difference between reproducing a number and
  producing a similar one.
