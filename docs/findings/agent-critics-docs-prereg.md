# Pre-registration: docs-critic against a no-agent baseline, on held-out cases

**Date:** 2026-10-28. Committed before the cases, before the harness change
that runs them, and before any review. No result exists.

**Scope:** whether the `docs-critic` definition catches defects in changes to
published pages, and stays clear on their fixed versions, better than a
generic reviewer on the same model with the same tools. It extends
`agent-critics-baseline-prereg.md` to the fourth review agent.

No training years are involved, and no test year is involved: nothing is
fitted, and the units are reviews of page changes.

## Held-out status

The five cases come from mistakes that the definition's text does not draw on.
Since `f065bfe` its examples are generic, and its words match none of the 98
rows in the mistake log of `docs/design/agent-guards.md`. The mistakes its
first version drew on (C011, C012, C063, C064 and C065) are development
examples only and are not cases. The generic definition has reviewed none of
these cases. The reviews of two published-page changes on 2026-10-27 ran with
the earlier file on disk and reviewed none of them either.

## Cases

Five pairs, in `scripts/dev/agent_evals/cases/docs/`. Each defect arm is a
page change carrying the defect; each control arm is the same change with the
defect fixed. A reviewer sees the change and a checkout of the repository at
the snapshot, as it stood before the fix.

| Pair | Mistake | Snapshot | Check | Defect arm | Control arm |
|---|---|---|---|---|---|
| `reproduction-claim` | C018 | `3dd8498^` | 2, claims against evidence | The Denmark sweep page as first committed, calling the run a reproduction of the published paper and saying it matches the paper's headline | The same page calling it a re-run of the paper's grid, with the configuration differences stated |
| `dropped-caveat` | C060 | `643499a^` | 5, negatives | The scorecard's country table republished with Sweden's row moved and no word on its denominator | The same change with a caveat: Sweden's 2015-19 capacity is derived from its own generation, as `method-country-level.md` already records at the snapshot |
| `file-count` | C084 | `5194a21^` | 1, counts | A CHANGELOG line saying all 43 test files pass, one process each | The same line saying 52 |
| `cited-test` | C172 | `c3859a0^` | 3, checks that could fire | A CHANGELOG line citing `test_the_clipped_count_reaches_the_manifest_and_the_metrics` as evidence that the clipped count reaches the run manifest and `metrics.csv` | The same line saying the test checks the in-memory record and a JSON round trip, and that the driver's writes are untested |
| `pinned-halves` | C173 | `cc9fc14` | 1 and 3 | A CHANGELOG line saying two tests pin both halves of the run-directory defect | The same line saying one test pins the run name and the per-row output files are untested |

C094 was a candidate and is not a case: its defect was a claim about mypy's
output, which a reviewer limited to read-only git and `wc` cannot check from
the repository.

## Design

- **Reviewers, two, both on opus:**

| Reviewer | Definition | Tools |
|---|---|---|
| `docs-critic` | `.claude/agents/docs-critic.md` | Read, Grep, Glob, Bash under `guard_docs_critic.py` |
| `baseline-docs` | `scripts/dev/agent_evals/baseline-reviewer.md` | the same tools and the same hook |

  The baseline takes the critic's model, tools and hook, so a difference is
  the definition's.
- **How they run:** one `--agents` JSON with `omitClaudeMd: true`, every
  definition's sha256 recorded in `run.json`, as in the baseline registration.
  A probe on 2026-10-28 confirmed that a `hooks` field in that JSON fires: a
  hook that blocks Bash blocked it and left a marker file, and the same agent
  without the hook ran Bash.
- **Where the session runs:** from an empty directory outside any git
  repository, so that `git log` and `git show` cannot read the history in
  which the fixes sit. The hook is called by absolute path.
- **The prompt** gives the change, its one-line summary, and the snapshot path.
  It asks for the files read and the commands run.
- **Exclusion:** a review is excluded if it lists a file, or a command naming a
  path, outside the snapshot.
- **Size:** 20 reviews (2 reviewers × 5 pairs × 2 arms) and 30 judge calls (20,
  plus 10 calibration fixtures), the judge `claude-sonnet-5-5`, blind to the
  reviewer and the arm.

## Scoring

The scoring rule of `agent-critics-prereg.md` applies unchanged: verdict,
found, hit, noted, miss, false alarm, clear, excluded, and judge calibration.
docs-critic's verdict line reads PASS or FIX; FIX counts as blocking and PASS
as not, exactly as REVISE and PROCEED do for the other reviewers.

## Gates

**The comparison (gated).** Each reviewer's score is defects caught (hits)
minus controls with a false alarm, from -5 to +5. D is docs-critic's score and
B the baseline's, from the same run.

| Outcome | Rule | Reading |
|---|---|---|
| Value shown | D ≥ B + 2 | The definition adds something on these cases |
| No measurable difference | \|D - B\| ≤ 1 | Nothing measurable over a generic reviewer with the same model and tools |
| Worse than baseline | D ≤ B - 2 | The definition costs more than it adds on these cases |

If the judge's calibration fails, no outcome is read.

**Reported with the gate, without a bar of their own:** each reviewer's hits
and false alarms; the absolute bar of the first registration (at least 4 of 5
caught, at most 1 false alarm) for each; and each pair's outcome against the
check it targets.

A "no measurable difference" outcome is the finding, and stands unless a later
registered run meets the bar. This is a smoke test of five cases, one draw
each, not a detection rate.

## Limits

- **Three cases are reframed.** C084, C172 and C173 sat in a commit message, a
  test name and a docstring. They are presented as CHANGELOG lines, because
  docs-critic reviews published pages, not code. The defect in each is the real
  one; the page it sits on is not.
- **The cases are written by an agent who knows the answers,** as in the first
  evaluation.
- **The reviewer and the baseline share Bash**, which the first evaluation's
  reviewers did not have. The two registrations' figures are not comparable.
- **Five cases, one draw each.** A tie says nothing about defects of other
  kinds.

## Commands

```bash
python scripts/dev/agent_evals/evals.py prepare --suite docs --label <date> \
    --judge-model claude-sonnet-5-5
# start the session from the empty directory prepare prints and records
claude --agents "$(cat <repo>/scripts/dev/agent_evals/runs/<date>/agents.json)"
python scripts/dev/agent_evals/evals.py judge --label <date>
python scripts/dev/agent_evals/evals.py score --label <date>
```
