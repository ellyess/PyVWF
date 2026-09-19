---
name: provenance-guard
description: Check PyVWF's provenance before any release-shaped action (a version bump, a CHANGELOG promotion, CITATION.cff, a tag, a release, a Zenodo archive, a merge to main). Runs the committed-files and packaging tests, checks the scorecard's reproducibility claims against the run records, and walks the licence checklist. Read-only; it ends with a report and never performs the release.
---

# provenance-guard

Run these checks before any release-shaped action. Report every result. Then
stop.

**Never perform the action.** Merging to `main`, pushing, tagging and
publishing a release belong to the maintainer. So does anything that mints a
DOI. This holds even when asked in passing. Hand over the exact commands
instead.

Terms follow `docs/CONTEXT.md`.

## 1. Tests

Run each file in its own process:

- `tests/test_committed_files.py`: no tracked file is excluded by `.gitignore`,
  apart from a stated allowlist; no licensed-library file is tracked; and
  both copies of the open library match their recorded sha256.
- `tests/test_curve_library.py`: the open library is internally consistent,
  its two copies are byte-identical, and every curve has a provenance row.
- `tests/test_packaging.py`: `vwf.__version__`, the newest CHANGELOG release,
  its compare links and `CITATION.cff` agree.
- `tests/test_scorecard_configs.py`: every configuration in
  `configs/regions/scorecard/` loads, every scorecard row has one carrying its
  region code, none is orphaned, and the configuration each row reports in its
  Best cfg column is one its committed configuration can produce. This is what
  makes "reproducible against a commit" a checked claim rather than a stated
  one.

A failure stops the release. Report it; do not work around it.

## 1a. Guards

**A guard that has never been seen to fire is an assumption.** For each check
that exists to catch a failure rather than to compute a number, confirm it has
been proved to catch that failure, and record when it was. Proving one is
cheap: point it at input you know is wrong and watch it refuse.

Three are in play, and their state as of 2026-09-13:

| Guard | What it refuses | Proved to fire |
|---|---|---|
| The ERA5 extent guard (`vwf.wind.ExtrapolationError`) | a unit outside the loaded extent | not deliberately; it fired on Denmark in the course of work |
| The curve library study's override refusal | a fleet that is not the one a condition asked for | no |
| The same study's library check | a run that resolved the wrong curve library | **yes, 2026-09-13**, by naming the open library's hash for a run on the combined one |

A guard in the "no" column is not evidence that the failure it names has not
happened. It is evidence that nobody has checked.

## 2. Scorecard reproducibility claims

This check needs the local run tree under `output/`, so it cannot run in CI.

1. For each scorecard row, find its curve resolution record. The scorecard
   names where the records are.
   - A run from commit `5c11310` on carries one: `curve_resolution` in its
     `run_manifest.json`, and its `curve_resolution.csv`.
   - An older row's record is in the re-evaluation that the scorecard cites
     for it. A row with no record at all is reported as not checked.
2. Read the `open` and `external` shares of capacity from that record.
3. A row with any `external` share of capacity is not third-party
   reproducible. Whether that curve library is the licensed one is a separate
   question, answered by the manifest's `curve_library` identity.
4. Compare with the scorecard's statement of which rows are not third-party
   reproducible. Report any row where the claim and the record disagree, in
   either direction.
5. Report any non-zero substituted share, with the scorecard's statement of it.

## 3. Licence checklist

A reader does these checks; no search replaces them.

- **Curated tables.** List the files under `configs/curation/` that are new or
  changed since the last release tag (`git diff --stat <tag>..HEAD --
  configs/curation`). For each, confirm:
  - it has a per-row source column with no blank cells;
  - each cited source's terms allow redistribution, or the maintainer has
    recorded a decision about it.
- **Runbooks.** Every region in the scorecard has a runbook with a licence
  line.
- **Data sources.** Every region's data source has a row in
  `docs/guides/data-sources.md` with a licence key.
- **Findings.** A findings document that quotes licensed-library results
  states that they are not third-party reproducible.

Existing tables that predate the per-row source rule are not failures. List
them as known gaps.

## 4. Report

- One line per check: pass, fail, or not run, with the reason.
- The known gaps, unchanged since the last report or new.
- The commands the maintainer would run next. Do not run them.
