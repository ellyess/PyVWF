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
- **Report a condition's coverage before its gate is fixed**, as a share of
  capacity: how much of the fleet the rule actually reaches. A condition
  reaching little cannot test much, and a gate written without knowing that is
  a gate on nothing.
- **A prediction names the quantity it is about, and the mechanism has to show
  up in that quantity.** A curve library condition predicted that a worse
  substitute would move corrected RMSE further. The mechanism was real, and it
  lived in the uncorrected mean bias: the correction absorbed it before it
  reached corrected RMSE, so the prediction failed while being right about the
  world. Before registering one, ask which side of the correction the effect
  sits on, and name that side.
- **Verify that a condition reaches what it claims, by a route independent of
  how it was built.** Three defects in one study would each have produced a
  clean null, which is what several of its predictions expect, and none was
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
