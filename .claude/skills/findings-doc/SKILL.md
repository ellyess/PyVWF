---
name: findings-doc
description: Write, revise or correct a PyVWF findings document under docs/findings/, or a scorecard row, so that every number is traceable, the full metrics table comes before interpretation, negative results are kept, and corrections are dated notices. Use whenever a docs/findings/ file is created or edited, or a result is added to the scorecard.
paths: "docs/findings/**"
---

# findings-doc

Findings documents are argumentative records for readers who know the project.
Their shape and naming are set in `docs/README.md`; read it first. Terms follow
`CONTEXT.md`: define each once, by citing it, and use it consistently.

## Stops

1. Before drafting, you need the run directory or `Data:` path behind every
   number. Without it, stop and ask for it.
2. After the full metrics table, stop. The human frames the result; you do not.
3. Before committing, show the diff and wait for approval.
4. **When a decision changes, re-read the whole document, not the edited
   part.** A document written under the old decision carries sentences that
   still assume it, often far from the edit: a consequence, a caveat, a
   pointer, a section that names the open question as open. Twice now a stale
   sentence has survived an approved change and contradicted a decision made
   two bullets above it. Read the document end to end and fix every sentence
   written under the old answer, then show the diff.

## Shape

- Name the file `<type>-<subject>.md` per `docs/README.md`. One question per
  document. Revise in place as the answer changes; git holds the history.
- Open with the H1, then `**Date:**` and `**Scope:**` lines, then a bold lead
  that states the claim.
- State the claim before the caveat, in the lead and in every section.
- Keep reasoning about why the code is shaped as it is out of the document.
  Move it to `docs/design/`. Rewrite it for a reader who knows the domain but
  has no history with the project. Leave out references to conversations, "as
  we decided", and any assumed context.
- End with the caveats. A negative or unflattering result is reported in the
  body and the lead, never moved to an appendix.

## Evidence

Each check below is mechanical. If one fails, stop and name the check. Do not
write around it.

1. **Every number cites its source:** a run directory, a `metrics.csv`, or a
   `Data:` path.
2. **The full metrics table comes first:** every row of the cited `metrics.csv`,
   with the uncorrected row first, before any interpretation. An omitted row
   needs a stated reason.
3. **Training years and the single test year are stated** for every result.
4. **The curve library is stated for every result.**
   - For runs from commit `5c11310` on, read it from the run's
     `curve_resolution.csv`. A run with the licensed library as its input root,
     but only curves of `open` origin, is third-party reproducible; say which
     applies.
   - For earlier runs, read it from the manifest's `curve_library` block.
   - A non-zero substituted share is reported, with the keys that were missing
     and the fallback curve they used.
5. **A degenerate fit is never quoted without its fit quality:** maximum
   scalar, implausible scalars, failed offsets.
6. **A gated claim cites its pre-registration,** and the commit that fixed the
   gates is older than the commit with the results. Check with `git log`.
7. **Every change made after a result was seen is labelled:** what changed,
   when, why, and its measured effect on the numbers. A change that moved
   nothing says so, with the evidence.

Review question, not a mechanical check: does any number mislead when read
alone, with its real meaning left to a footnote? If so, restructure the table.
A true statement that conceals is still wrong.

## Conditions

A condition is a rule applied to a fleet, and a rule is written against the
unit its author has in mind.

- **Before a condition runs, report what its rule does to the smallest and the
  largest unit in the fleet.** Not a sample: those two, by capacity and by
  rating. Twice now that check would have caught a condition testing something
  other than its claim, and both times at the small end: an other-brand
  condition moved 100 kW machines onto a distributed reference curve, because
  nothing in its rule said a reference design is not a brand; and an off-curve
  count treated a missing capacity factor as a zero, which only bites where the
  wind is too low to produce.
- The diagnosis is worth keeping with the rule. Conditions get written against
  the typical unit, and the rule does something different where the typical
  case's logic stops holding. The extremes are where that shows, and they are
  two rows to print.
- **Distrust the reasoning, not the evidence that happens to support it.** A
  measurement said one configuration of fourteen was the only one interpolated
  from other data, and it was also the only one that failed. That was too tidy,
  so it was withdrawn, and the withdrawal was wrong: **the discipline of
  distrusting a tidy story was applied to the evidence instead of to the
  reasoning.** The measurement was right and the story was true. When a result
  looks too neat, the check goes on the inference that produced it, not on the
  number: recompute the reasoning, name the step that would have to be wrong,
  and test that step. Discarding the measurement feels like rigour and costs
  the finding.

  Its concrete form, from the same two incidents: **choose the case to test
  before seeing which case favours the hypothesis, or test them all.** A single
  row was picked to decide whether a whole tier of control points was an
  artefact, and it was picked because its statistic was the most artefact-like
  of the eight. It confirmed the hypothesis, as the most confirming case will,
  and running the same test on every comparable row reversed the conclusion:
  the one row said artefact at a ratio of 0.09, the four rows with no other
  known defect said the opposite at 1.02. Same data, opposite answers, decided
  by which case was tested. All six were free there, which is the only reason
  it was recoverable; when they are not free, the case is named in the
  pre-registration before the statistic that would choose it is computed. An
  explanation for cases set aside has to be independent of the hypothesis.
- **Verify that a condition changed what it was supposed to change, not only
  that it was applied.** The counterpart to the coverage rule below: that one
  checks the condition reached the fleet, this one checks it reached the
  result. A curve
  study applied its overrides to the fleet a loader returned, one step after
  that loader had already simulated it, so every fitted wind scalar came out
  bit-identical to the baseline's while the offsets moved. The guard verified
  the fleet and passed. The evidence sat in published results for a day: eight
  country rows, a column of exact zeroes where a changed curve must change a
  scalar fitted as observed over simulated output. Name, before the run, the
  fitted quantity the condition must move, and compare it against the baseline
  afterwards. The check is cheap, it is independent of how the condition was
  built, and a condition that changed nothing is indistinguishable from a
  condition that worked and had no effect, which is what half these studies
  predict.
- **Ask whether a condition can differ analytically before spending a run on
  it.** Two bases of one target were compared by a twelve-fold study, and the
  answer followed in three lines from the estimator being a weighted sum with
  coordinate-only weights: any such predictor commutes with a linear
  reparameterisation, so three of the four methods could not have differed and
  did not, to the floating-point digit. The run was still worth having for the
  fourth method and for the raw table, but the invariance should have been
  derived first and the study framed around the one case that could move.
- **Whatever a study varies must appear in the run directory path.** A run
  directory is keyed on the region code and the run name, and on nothing else,
  so two configurations of one region collide silently. Every earlier study in
  this repository is clean because it put its varying dimension in the path:
  the roughness study used `train-R0` and `train-R1`, the curve-library and
  EU re-run studies used a condition directory above the region. A cluster
  study that varied the fleet mode, which the path does not carry, wrote
  onshore and offshore factors into one directory and scored an eleven-count
  grid where eight were registered.

  Two things made that dangerous rather than merely untidy. **`run_evaluate`
  scores every `factors_*.csv` it finds in the training directory**, so a path
  collision becomes a contaminated candidate grid rather than an error. And
  **a per-process loop satisfies the memory isolation rule without preventing a
  path collision**: the rule is about processes and the collision is in the
  path, so obeying one says nothing about the other. Check the path, not the
  process.

- **Verify that a condition reaches what it claims, by a route independent of
  how it was built, and report its coverage as a share of capacity before its
  gate is fixed.** A condition reaching little cannot test much, and a gate
  written without knowing that is a gate on nothing. Three defects in one study
  would each have produced a clean null, which is what several of its predictions expect, and none was
  caught by the checks built into the condition itself: a matcher comparing a
  manufacturer against itself, an assignment moving small units onto a
  reference curve, and a coverage figure measured over the register rather
  than the fleet the run fits. An independent route is one whose failure mode
  differs: the extremes above are one, and comparing a figure against the same
  figure measured a different way is another, which is what caught a coverage
  of 0.0% against 50.2% measured directly.

## Corrections

- A published claim that was wrong gets a **correction notice**, dated and
  placed after the `**Scope:**` line. It states what actually ran and what was
  claimed. It names the figures it affects and the ones that stand.
- A correction notice is never softened into a limitation.
- **A notice is published as soon as the claim is known to be wrong, before the
  corrected figure exists.** It says so: which values are withdrawn, that the
  corrected ones are not yet known, and what will produce them. Waiting for the
  right number is how a wrong one stays published for weeks. The withdrawn text
  stays in place, marked, so that what was claimed can be read against what
  replaces it.
- A false claim in dated history, such as a CHANGELOG entry, is corrected in
  place with a dated bracket, not deleted.
- A correction of a correction says so.
- A claim that was wrong in the project's favour is corrected the same way.

## Scorecard rows

When a result enters `docs/findings/scorecard.md`:

- Commit the exact configuration behind it, byte-identical, to
  `configs/regions/scorecard/` (`<stem>_k<N>.toml`, or `<stem>_country.toml`).
- For a turbine-level row, fill the other brand, reference curve and
  unverifiable columns from `scripts/analysis/curve_match_audit.py`. For a
  country-level row, fill the substituted column.
- Mark a degenerate fit with a dagger, and add its fit quality to the table
  under the scorecard.
- Mark a row whose `extrapolated_capacity_share` is above zero with §, and
  state the share beside it. Read the value from the row's `metrics.csv`. The
  rule holds even when the region opted in (`[era5] allow_extrapolation =
  true`): opting in lets the run finish, and does not make the number sound.
  The markers and their rules are in `docs/README.md`.

## CHANGELOG

An entry for findings work goes under `[Unreleased]`. It carries no shares or
metrics; those live in the scorecard and the run outputs, where they are
regenerated.
