# docs-critic against a no-agent baseline

**Date:** 2026-10-29.

**Scope:** the run registered in
[`agent-critics-docs-prereg.md`](agent-critics-docs-prereg.md) (committed
`9efbf16`, 2026-10-28, before the harness change that runs it and before any
review): whether the `docs-critic` definition catches more of five defects in
published-page changes, or raises fewer false alarms on their fixed controls,
than a generic reviewer with the same model, tools and Bash hook. No training
years and no test year are involved; the units are reviews of page changes.

**No measurable difference.** `docs-critic` scored 1 (3 defects caught, 2
false alarms) and the baseline 1 (1 caught, no false alarm), where the
registered gate needs a margin of 2 to read "value shown". `docs-critic` missed
two of the five defects, an overclaimed reproduction and a dropped caveat, and
neither reviewer meets the first registration's bar. By the registration, this
outcome is the finding. The baseline's figures rest on 5 of its 10 reviews,
because the other 5 were excluded by a scoring rule that misfired (below).

**Data:** `scripts/dev/agent_evals/results-2026-10-29-docs.md`, written by
`evals.py score` from the run directory
`scripts/dev/agent_evals/runs/2026-10-29-docs/` (local; run folders are not
committed). Harness at `929ebe4`. The judge is `claude-sonnet-5-5`, and its
calibration passed on all ten fixtures.

## Every review

| Pair | Arm | Reviewer | Verdict | Judge | Quote in review | Files read | Outcome |
|---|---|---|---|---|---|---|---|
| cited-test | defect | docs-critic | FIX | YES | yes | inside | hit |
| cited-test | defect | baseline-docs | REVISE | YES | yes | inside | hit |
| cited-test | control | docs-critic | FIX | YES | yes | inside | false alarm |
| cited-test | control | baseline-docs | REVISE | NO | no | inside | clear |
| dropped-caveat | defect | docs-critic | FIX | NO | no | inside | miss |
| dropped-caveat | defect | baseline-docs | REVISE | NO | no | outside snapshot | excluded |
| dropped-caveat | control | docs-critic | FIX | YES | yes | inside | false alarm |
| dropped-caveat | control | baseline-docs | REVISE | NO | no | inside | clear |
| file-count | defect | docs-critic | FIX | YES | yes | inside | hit |
| file-count | defect | baseline-docs | REVISE | YES | yes | outside snapshot | excluded |
| file-count | control | docs-critic | FIX | NO | no | inside | clear |
| file-count | control | baseline-docs | REVISE | NO | no | inside | clear |
| pinned-halves | defect | docs-critic | FIX | YES | yes | inside | hit |
| pinned-halves | defect | baseline-docs | REVISE | NO | no | outside snapshot | excluded |
| pinned-halves | control | docs-critic | FIX | NO | no | outside snapshot | excluded |
| pinned-halves | control | baseline-docs | REVISE | NO | no | outside snapshot | excluded |
| reproduction-claim | defect | docs-critic | FIX | NO | no | inside | miss |
| reproduction-claim | defect | baseline-docs | REVISE | NO | no | inside | miss |
| reproduction-claim | control | docs-critic | FIX | NO | no | inside | clear |
| reproduction-claim | control | baseline-docs | REVISE | NO | no | outside snapshot | excluded |

## Per reviewer, and the gate

| Reviewer | Model | Hits | Noted | Misses | False alarms | Clear | Excluded |
|---|---|---|---|---|---|---|---|
| docs-critic | opus | 3 | 0 | 2 | 2 | 2 | 1 |
| baseline-docs | opus | 1 | 0 | 1 | 0 | 3 | 5 |

| Readout | docs-critic | baseline-docs |
|---|---|---|
| Defects caught | 3 of 5 | 1 of 5 |
| Controls with a false alarm | 2 of 5 | 0 of 5 |
| Score, caught minus false alarms | 1 | 1 |
| First registration's bar (at least 4 caught, at most 1 false alarm) | not passed | not passed |

**Gate: D = 1, B = 1, |D - B| = 0: no measurable difference.**

**Each pair against the check it targets** (the check numbers are
`docs-critic`'s):

| Pair | Check | docs-critic, defect / control | baseline, defect / control |
|---|---|---|---|
| reproduction-claim | 2, claims against evidence | miss / clear | miss / excluded |
| dropped-caveat | 5, negatives | miss / false alarm | excluded / clear |
| file-count | 1, counts | hit / clear | excluded / clear |
| cited-test | 3, checks that could fire | hit / false alarm | hit / clear |
| pinned-halves | 1 and 3 | hit / excluded | excluded / excluded |

`docs-critic` caught the three defects about counts and tests, and missed the
two about what a page claims and what it leaves out. Every verdict, on either
arm, was FIX (`docs-critic`) or REVISE (the baseline).

## How it ran, and where it departs from the registration

- **The reviews** ran in one Claude Code session started from the empty
  directory `prepare` printed, outside any git repository, with the run's
  `--agents` JSON, as registered. So that the session needed no access to the
  run directory, whose `run.json` names each case and arm, the prompts and a
  list of tokens with their reviewers were copied into that directory, and the
  replies were copied back; the copies are byte-identical (`diff -rq`). The
  session was given the other run's instruction by mistake, found that path
  absent and followed the instruction in its own directory; it sent only this
  run's 20 prompts, to this run's two reviewers. The sha256 of both
  definition files in `run.json` match the files at `929ebe4`.
- **The judge** ran outside that session: one `claude -p --model
  claude-sonnet-5-5` call per judge prompt, with no tools, no settings and an
  empty working directory.
- **The working tree was dirty** (`harness_dirty: true`): only a local
  `.claude/launch.json`.

## The exclusion rule misfired; found after the results

*[Found on 2026-10-29, after scoring, and not part of the registration.]* A
review is excluded when it lists a file, or a command naming a path, outside
its snapshot. None of the six excluded reviews lists a read outside its
snapshot; the rule can only see what a review lists, so a read left off the
list would go undetected either way. The parser takes every line after
"Files read:" as a path, so it counted notes of
files the reviewer tried to open and found absent, all inside the snapshot
(four reviews), and a closing "Verdict: REVISE" (two). One of those two also
lists a command with `2>/dev/null`, which the Bash hook blocked before it ran.

Scoring the six by the judge alone, as if none were excluded, raises the
baseline to 2 (file-count becomes a hit) and leaves `docs-critic` at 1, so
D - B is -1 and the outcome is still no measurable difference. That check was
run after the results and is not registered. The parser fix is a separate
change, and the registered numbers above stand as scored.

## Caveats

- **Five cases, one draw each**: a smoke test, not a detection rate, and a tie
  says nothing about defects of other kinds.
- **The baseline is thinly measured.** Five of its ten reviews were excluded,
  so its registered score rests on half its arms.
- **Three cases are reframed** (registration, Limits): the defects of
  file-count, cited-test and pinned-halves sat in a commit message, a test name
  and a docstring, and are presented as CHANGELOG lines.
- **The cases were written by an agent that knew the answers.**
- **`docs-critic` blocks everything.** It gave FIX on every control, so its
  verdict carries no information here; only the judged defect does.
- **Not comparable with the first evaluation**: these reviewers have Bash, and
  those did not.
