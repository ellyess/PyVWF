---
name: findings-doc
description: Write, revise or correct a PyVWF findings document under docs/findings/, or a scorecard row, so that every number is traceable, the full metrics table comes before interpretation, negative results are kept, and corrections are dated notices. Use whenever a docs/findings/ file is created or edited, or a result is added to the scorecard. Do not use for guides, runbooks or design pages, which follow the writing rules in docs/README.md.
paths:
  - "docs/findings/**"
  - "docs/scorecard.md"
---

# findings-doc

Findings documents are argumentative records for readers who know the project.
Their shape and naming are set in `docs/README.md`; read it first. Terms follow
`docs/CONTEXT.md`: define each once, by citing it, and use it consistently.

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

A condition is a rule applied to a fleet, written against the unit its author
has in mind. Each rule below records a defect that would otherwise have
produced a clean null or a wrong result. The full account of each incident is
in this section as it stood at tag `archive/research-2026-10-27`, in
`.claude/skills/findings-doc/SKILL.md`.

- **Before a condition runs, report what its rule does to the smallest and the
  largest unit in the fleet,** by capacity and by rating. Rules break where the
  typical case's logic stops holding: an other-brand condition moved 100 kW
  machines onto a reference curve, and an off-curve count treated a missing
  capacity factor as zero.
- **Distrust the reasoning, not the evidence that happens to support it.** When
  a result looks too neat, recompute the inference and test the step that would
  have to be wrong; do not discard the measurement. **Choose the case to test
  before seeing which case favours the hypothesis, or test them all:** one row
  picked for being the most artefact-like said artefact at 0.09, and the four
  clean rows said the opposite at 1.02. When testing every case is not free,
  name the case in the pre-registration before computing the statistic that
  would choose it. An explanation for cases set aside has to be independent of
  the hypothesis.
- **Verify that a condition changed what it was supposed to change, not only
  that it was applied.** Name,
  before the run, the fitted quantity the condition must move, and compare it
  against the baseline afterwards. A curve study once left every wind scalar
  bit-identical because its overrides reached the fleet after it was
  simulated.
- **Verify that a condition reaches what it claims, by an independent route,
  and report its coverage as a share of capacity before its gate is fixed.** An
  independent route is one whose failure mode differs: the extremes above, or
  the same figure measured another way. A
  matcher comparing a manufacturer against itself, and a coverage measured over
  the register rather than the fitted fleet (0.0% against 50.2% measured
  directly), each passed the condition's own checks.
- **Before registering a gate against an external comparator, confirm it
  exists on the same units, folds and grid; if not, the gate is not
  registrable.** Name the comparator's files, not a label for it, and name the
  statistic of every threshold, or the choice is made with the scores in view.
  When the comparator is another pipeline's output, check the values it
  carried, not how it was described: files named `.uniform.bak.csv` held a
  different weighting from the uniform grid they were taken for.
- **A study is not finished until its own gates are read.** Close a
  registration before its numbers are quoted anywhere else, even in a document
  that cites it properly.
- **Ask whether a condition can differ analytically before spending a run on
  it.** A weighted-sum estimator with coordinate-only weights commutes with a
  linear reparameterisation, so three of four methods could not have differed.
- **Whatever a study varies must appear in the run directory path.** A run
  directory is keyed on the region code and run name only, and `run_evaluate`
  scores every `factors_*.csv` it finds, so a collision contaminates the
  candidate grid silently. A per-process loop isolates memory, not paths.

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

When a result enters `docs/scorecard.md`:

- Commit the configuration its run was made from, unchanged, to
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
- Add the row's run, commit, roughness fill and curve library to the
  scorecard's "Reproducing a row" table.
- A correction to a row is a dated line in the scorecard's corrections log,
  linking the notice that holds the figures before and after.

## CHANGELOG

An entry for findings work goes under `[Unreleased]`. It carries no shares or
metrics; those live in the scorecard and the run outputs, where they are
regenerated.
