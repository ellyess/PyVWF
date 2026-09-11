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

## Corrections

- A published claim that was wrong gets a **correction notice**, dated and
  placed after the `**Scope:**` line. It states what actually ran and what was
  claimed. It names the figures it affects and the ones that stand.
- A correction notice is never softened into a limitation.
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
