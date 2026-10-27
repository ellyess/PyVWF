# Pre-registration: the review agents against a no-agent baseline

**Date:** 2026-10-27. Committed before the harness change that runs it and
before any review is made. No result exists.

**Scope:** whether the three critic definitions (`method-critic`,
`implementation-critic`, `red-team`) catch more of five seeded defects, or
raise fewer false alarms on their fixed controls, than a generic reviewer on
the same models. It is the second run of `agent-critics-prereg.md`, whose run
of 2026-10-16 had no baseline and did not pass its bar.

No training years are involved, and no test year is involved: nothing is
fitted, and the units are reviews of proposals.

## What is measured

The critic definitions on their own, not the critics as deployed. Every
reviewer runs with `omitClaudeMd: true`, so none reads `AGENTS.md`. A probe on
2026-10-27 showed the difference: a subagent without that field quoted all ten
`AGENTS.md` headings and the glob rule verbatim, and one with it answered "not
in context". `AGENTS.md` carries two rules that describe cases here ("never
sort a glob and take the last entry", and "show it fires on a known
positive"), so a run with it loaded would measure those rules as well as the
definitions.

## Design

- **Cases:** the five pairs in `scripts/dev/agent_evals/cases/`, unchanged:
  ftol-stall, manifest-key, obs-routing, sorted-glob and stale-affine. Each
  has a defect arm and a control arm.
- **Reviewers, six, all with Read, Grep and Glob only:**

| Reviewer | Definition | Model |
|---|---|---|
| `method-critic` | `.claude/agents/method-critic.md` | opus |
| `implementation-critic` | `.claude/agents/implementation-critic.md` | sonnet |
| `red-team` | `.claude/agents/red-team.md` | opus |
| `baseline-1` | `scripts/dev/agent_evals/baseline-reviewer.md` | opus |
| `baseline-2` | the same | sonnet |
| `baseline-3` | the same | opus |

- **The baseline definition** is a generic instruction and nothing else:
  review the proposed change before it is made, say whether it should go ahead
  and what is wrong or missing, only read, and end with a verdict. The
  baseline models match the critics', so a difference is the definition's.
- **How they run:**
  - `evals.py prepare` writes one `--agents` JSON for the six reviewers, with
    each definition's body as its prompt and `omitClaudeMd: true`, and records
    every definition's sha256 in `run.json`.
  - The session that makes the calls starts with that JSON. Its definitions
    take precedence over `.claude/agents/`, so nothing on disk changes.
  - Every reviewer gets the same prompt: the proposal, its snapshot path and
    the "Files read" instruction.
- **Size:** 60 reviews (6 reviewers × 5 pairs × 2 arms) and 70 judge calls
  (60, plus the 10 calibration fixtures). The judge is `claude-sonnet-5-5`,
  blind to the reviewer and the arm, as before.

## Scoring

The scoring rule of `agent-critics-prereg.md` applies unchanged: verdict,
found, hit, noted, miss, false alarm, clear, excluded, and judge calibration.
A baseline review is scored exactly as a critic's.

## Gates

**The comparison (gated).** A panel catches a defect when any of its three
reviewers scores a hit, and raises a false alarm on a control when any of its
three does. Its score is defects caught minus controls with a false alarm,
from -5 to +5. C is the critic panel's score and B the baseline panel's, from
the same run.

| Outcome | Rule | Reading |
|---|---|---|
| Value shown | C ≥ B + 2 | The critic definitions add something on these cases |
| No measurable difference | \|C - B\| ≤ 1 | Nothing measurable over a generic reviewer on the same models |
| Worse than baseline | C ≤ B - 2 | The definitions cost more than they add on these cases |

If the judge's calibration fails, no outcome is read, and the table is
reported as unscored by a calibrated judge.

**Reported with the gate, without a bar of their own:**

- each panel's defects caught and controls with a false alarm, separately;
- the original absolute bar (at least 4 of 5 caught, at most 1 false alarm),
  for each panel;
- each critic against its model-matched baseline: method-critic against
  baseline-1, implementation-critic against baseline-2, red-team against
  baseline-3. Hits, false alarms, and blocking verdicts on each arm;
- the expected catchers of each case, against the outcome;
- **a secondary readout on the two clean pairs**, stale-affine and
  sorted-glob, chosen now, before any review: the same panel counts, restricted
  to those two pairs.

A margin of 2 is used because one pair moves a score by 1. This is a smoke
test, not a detection rate. A "no measurable difference" outcome is the
finding, and stands unless a later registered run meets the bar.

## Limits

- **The critics may have been shaped by manifest-key.** `implementation-
  critic` asks for "a known-positive case for any new check". That line was
  added in `ea43c5e` on 2026-09-23, eight days after the defect was fixed
  (`cc9fc14`, 2026-09-15), and the case's judge counts naming the missing
  known-positive check as finding it.
- **Two of the five cases were first caught by these critics.** The mistake
  log in `docs/design/agent-guards.md` records ftol-stall (C071) and
  obs-routing (C046) as caught by a critic agent. The cases favour the
  critics.
- **stale-affine is clean by date, not by wording.** `method-critic`'s
  Comparison and Arms checks describe its defect closely, but were added three
  weeks before it was found.
- **PR 6b's held-out document cases are the cleaner test.** They are chosen
  from mistakes the docs-critic's text did not draw on, and registered before
  it reviews anything.
- **The cases are defects this repository already caught.** A generic reviewer
  may find them easy for that reason, so a tie here says nothing about defects
  not yet seen.

## Commands

```bash
python scripts/dev/agent_evals/evals.py prepare --label <date> \
    --judge-model claude-sonnet-5-5
claude --agents "$(cat scripts/dev/agent_evals/runs/<date>/agents.json)"
python scripts/dev/agent_evals/evals.py judge --label <date>
python scripts/dev/agent_evals/evals.py score --label <date>
```

In the `claude` session, each `prompts/<token>.md` goes to the reviewer
`run.json` names for it. Each reply is saved, unedited, as
`responses/<token>.md`. Run directories are local (`runs/` is ignored), and
the results summary is committed with the findings document.
