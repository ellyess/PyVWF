# The review agents against a no-agent baseline

**Date:** 2026-10-29.

**Scope:** the run registered in
[`agent-critics-baseline-prereg.md`](agent-critics-baseline-prereg.md)
(committed `6d5914b` and `7eab0c2`, 2026-10-27, before any review): whether
the three critic definitions, `method-critic`, `implementation-critic` and
`red-team`, catch more of five seeded defects, or raise fewer false alarms on
their fixed controls, than a generic reviewer on the same models. No training
years and no test year are involved; the units are reviews of proposals.

**No measurable difference.** The critic panel scored 3 and the baseline panel
2, a margin of 1 where the registered gate needs 2 to read "value shown". Both
panels caught all five defects; the critics raised a false alarm on 2 of the
5 controls and the baselines on 3. Neither panel meets the original bar of at
most one false alarm. By the registration, this outcome is the finding.

**Data:** `scripts/dev/agent_evals/results-2026-10-29.md`, written by
`evals.py score` from the run directory `scripts/dev/agent_evals/runs/2026-10-29/`
(local; run folders are not committed). Harness at `929ebe4`. The judge is
`claude-sonnet-5-5`, and its calibration passed: it found the defect in all
five positive fixtures and in none of the five negative ones.

## Every review

| Pair | Arm | Reviewer | Verdict | Judge | Quote in review | Files read | Outcome |
|---|---|---|---|---|---|---|---|
| ftol-stall | defect | method-critic | REJECT | YES | yes | inside | hit |
| ftol-stall | defect | implementation-critic | REVISE | YES | yes | inside | hit |
| ftol-stall | defect | red-team | REVISE | YES | yes | inside | hit |
| ftol-stall | defect | baseline-1 | REVISE | YES | yes | outside snapshot | excluded |
| ftol-stall | defect | baseline-2 | REVISE | YES | yes | inside | hit |
| ftol-stall | defect | baseline-3 | REVISE | YES | yes | inside | hit |
| ftol-stall | control | method-critic | REJECT | NO | no | inside | clear |
| ftol-stall | control | implementation-critic | REVISE | NO | no | inside | clear |
| ftol-stall | control | red-team | REVISE | NO | no | inside | clear |
| ftol-stall | control | baseline-1 | REVISE | YES | yes | inside | false alarm |
| ftol-stall | control | baseline-2 | REVISE | NO | no | inside | clear |
| ftol-stall | control | baseline-3 | REVISE | NO | no | inside | clear |
| manifest-key | defect | method-critic | REVISE | YES | yes | outside snapshot | excluded |
| manifest-key | defect | implementation-critic | REVISE | YES | yes | inside | hit |
| manifest-key | defect | red-team | REVISE | YES | yes | inside | hit |
| manifest-key | defect | baseline-1 | REVISE | YES | yes | inside | hit |
| manifest-key | defect | baseline-2 | REVISE | YES | yes | inside | hit |
| manifest-key | defect | baseline-3 | REVISE | YES | yes | inside | hit |
| manifest-key | control | method-critic | REVISE | NO | no | inside | clear |
| manifest-key | control | implementation-critic | REVISE | NO | no | inside | clear |
| manifest-key | control | red-team | REVISE | NO | no | inside | clear |
| manifest-key | control | baseline-1 | REVISE | NO | no | inside | clear |
| manifest-key | control | baseline-2 | REVISE | NO | no | inside | clear |
| manifest-key | control | baseline-3 | REVISE | NO | no | inside | clear |
| obs-routing | defect | method-critic | REJECT | YES | yes | inside | hit |
| obs-routing | defect | implementation-critic | REVISE | YES | yes | inside | hit |
| obs-routing | defect | red-team | REJECT | NO | no | inside | miss |
| obs-routing | defect | baseline-1 | REVISE | YES | yes | inside | hit |
| obs-routing | defect | baseline-2 | REVISE | YES | yes | inside | hit |
| obs-routing | defect | baseline-3 | REVISE | YES | yes | inside | hit |
| obs-routing | control | method-critic | REJECT | NO | no | inside | clear |
| obs-routing | control | implementation-critic | REVISE | YES | yes | inside | false alarm |
| obs-routing | control | red-team | REVISE | NO | no | inside | clear |
| obs-routing | control | baseline-1 | REVISE | YES | yes | inside | false alarm |
| obs-routing | control | baseline-2 | REVISE | NO | no | inside | clear |
| obs-routing | control | baseline-3 | REVISE | YES | yes | outside snapshot | excluded |
| sorted-glob | defect | method-critic | REJECT | YES | yes | outside snapshot | excluded |
| sorted-glob | defect | implementation-critic | REVISE | YES | yes | inside | hit |
| sorted-glob | defect | red-team | REVISE | YES | no | inside | miss |
| sorted-glob | defect | baseline-1 | REVISE | YES | yes | inside | hit |
| sorted-glob | defect | baseline-2 | REVISE | YES | yes | inside | hit |
| sorted-glob | defect | baseline-3 | REVISE | YES | yes | inside | hit |
| sorted-glob | control | method-critic | REVISE | NO | no | inside | clear |
| sorted-glob | control | implementation-critic | REVISE | NO | no | inside | clear |
| sorted-glob | control | red-team | REVISE | NO | no | inside | clear |
| sorted-glob | control | baseline-1 | REVISE | NO | no | inside | clear |
| sorted-glob | control | baseline-2 | REVISE | NO | no | inside | clear |
| sorted-glob | control | baseline-3 | REVISE | NO | no | inside | clear |
| stale-affine | defect | method-critic | REJECT | YES | yes | inside | hit |
| stale-affine | defect | implementation-critic | REVISE | YES | yes | inside | hit |
| stale-affine | defect | red-team | REJECT | YES | yes | outside snapshot | excluded |
| stale-affine | defect | baseline-1 | REVISE | YES | yes | inside | hit |
| stale-affine | defect | baseline-2 | REVISE | YES | yes | outside snapshot | excluded |
| stale-affine | defect | baseline-3 | REVISE | YES | yes | outside snapshot | excluded |
| stale-affine | control | method-critic | REJECT | YES | yes | inside | false alarm |
| stale-affine | control | implementation-critic | REVISE | YES | yes | inside | false alarm |
| stale-affine | control | red-team | REJECT | YES | yes | inside | false alarm |
| stale-affine | control | baseline-1 | REVISE | NO | no | outside snapshot | excluded |
| stale-affine | control | baseline-2 | REVISE | NO | no | inside | clear |
| stale-affine | control | baseline-3 | REVISE | YES | yes | inside | false alarm |

## Per reviewer

| Reviewer | Model | Hits | Noted | Misses | False alarms | Clear | Excluded |
|---|---|---|---|---|---|---|---|
| method-critic | opus | 3 | 0 | 0 | 1 | 4 | 2 |
| implementation-critic | sonnet | 5 | 0 | 0 | 2 | 3 | 0 |
| red-team | opus | 2 | 0 | 2 | 1 | 4 | 1 |
| baseline-1 | opus | 4 | 0 | 0 | 2 | 2 | 2 |
| baseline-2 | sonnet | 4 | 0 | 0 | 0 | 5 | 1 |
| baseline-3 | opus | 4 | 0 | 0 | 1 | 3 | 2 |

## The gate and the registered readouts

| Readout | Critic panel | Baseline panel |
|---|---|---|
| Defects caught (any reviewer a hit) | 5 of 5 | 5 of 5 |
| Controls with a false alarm | 2 of 5 (obs-routing, stale-affine) | 3 of 5 (ftol-stall, obs-routing, stale-affine) |
| Score, caught minus false alarms | 3 | 2 |
| Original bar (at least 4 caught, at most 1 false alarm) | not passed | not passed |
| Clean pairs only (sorted-glob, stale-affine) | 2 caught, 1 false alarm | 2 caught, 1 false alarm |
| Blocking verdicts, defect arm / control arm, per reviewer | 5 / 5 for each | 5 / 5 for each |

**Gate: C = 3, B = 2, |C - B| = 1: no measurable difference.**

Each critic against its model-matched baseline:

| Pair of reviewers | Hits | False alarms |
|---|---|---|
| method-critic / baseline-1 (opus) | 3 / 4 | 1 / 2 |
| implementation-critic / baseline-2 (sonnet) | 5 / 4 | 2 / 0 |
| red-team / baseline-3 (opus) | 2 / 4 | 1 / 1 |

No review on either arm gave PROCEED: every verdict was REVISE or REJECT,
fixed proposal or not, as in the run of 2026-10-16. `red-team` missed two
defects: on obs-routing the judge found no defect, and on sorted-glob it said
YES with a quote the review does not contain, which the scoring rule counts
as a miss. No baseline missed any.

**The expected catchers against the outcome.** Each case names the critics
expected to catch it. Every expected catcher either scored a hit or was
excluded with the judge finding the defect:

| Case | Expected | Outcome |
|---|---|---|
| ftol-stall | method-critic, implementation-critic | both hit |
| manifest-key | implementation-critic, method-critic | hit; excluded, judge YES |
| obs-routing | implementation-critic | hit |
| sorted-glob | implementation-critic | hit |
| stale-affine | method-critic, red-team | hit; excluded, judge YES |

## How it ran, and where it departs from the registration

- **The reviews** ran in one Claude Code session started with the run's
  `--agents` JSON, as registered. The session sent each prompt, unchanged, to
  the reviewer `run.json` names and saved each reply as returned. The sha256 of
  the four definition files in `run.json` (three critics and the baseline)
  match the files at `929ebe4`.
- **The judge** ran outside that session: one `claude -p --model
  claude-sonnet-5-5` call per judge prompt, with no tools, no settings and an
  empty working directory, so it saw only the prompt. The registration names
  the judge model and its blinding, not the process that calls it.
- **The working tree was dirty** (`harness_dirty: true` in `run.json`). The
  only uncommitted path was a local `.claude/launch.json`; no harness, case or
  definition file differed from `929ebe4`.

## The exclusion rule misfired; found after the results

*[Found on 2026-10-29, after scoring, and not part of the registration.]* A
review is excluded when it lists a file outside its snapshot. All eight
exclusions here are the rule misreading the review's list. The parser takes every line after "Files read:" as a
path, so it counted a closing "Verdict: REVISE" (three reviews), a note that
`docs/CONTEXT.md` did not exist at the snapshot (two), a note that files were
grepped rather than opened (one), and a note listing snapshot files seen in
search output (two). None lists a read outside its snapshot. The rule can
only see what a review lists, so a read left off the list would go
undetected either way.

Scoring the eight by the judge alone, as if none were excluded, does not
change the outcome: the critic panel still catches 5 and raises 2 false
alarms, the baseline panel 5 and 3, and C - B stays 1. That check was run
after the results and is not registered. The parser fix is a separate change,
and the registered numbers above stand as scored.

## Caveats

- **One draw per review**, five pairs: a smoke test, not a detection rate. One
  pair moves a panel's score by 1, which is why the margin is 2.
- **The cases favour the critics** (registration, Limits): two of the five
  defects were first caught by these critics, and `implementation-critic`'s
  known-positive line was added after manifest-key was fixed. The baselines
  matched them anyway.
- **The cases are defects this repository already caught.** A generic reviewer
  may find them easy for that reason, so the tie says nothing about defects
  not yet seen.
- **The reviewers do not read `AGENTS.md` into context** (`omitClaudeMd`), so
  this measures the definitions, not the critics as deployed with the rules
  loaded.
- **Every reviewer blocks everything.** No review gave PROCEED on a fixed
  proposal, so a REVISE verdict on its own carries no information here; only
  the judged defect does.
