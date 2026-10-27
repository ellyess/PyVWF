# Do the review agents catch the defects they exist to catch?

**Date:** 2026-10-16.
**Scope:** `method-critic`, `implementation-critic` and `red-team`
(`.claude/agents/`), each reviewing the ten proposals of the five matched pairs
registered in `agent-critics-prereg.md`, once. No training year or test year:
this measures the review agents, not a correction. Terms follow `CONTEXT.md`.

**The pre-registered bar was not passed.** The panel caught all 5 defects,
every critic caught every one, and the panel raised the paired defect on 2 of
the 5 controls, where the bar allowed at most 1. No critic gave PROCEED on any
of the 30 proposals, so the verdict alone did not separate a defect from its
fixed control.

Data: `scripts/dev/agent_evals/runs/2026-10-16/` (`run.json`, `prompts/`,
`responses/`, `judge_prompts/`, `judgements/`, `results.json`, `results.md`,
`read_audit.json`) *[2026-10-27: at tag `archive/research-2026-10-27`; the
main branch keeps `results.md` as
`scripts/dev/agent_evals/results-2026-10-16.md`]*, produced by
`scripts/dev/agent_evals/evals.py` at
`654ad76`. The registration was committed first, at `1bd9607`. Critics ran on
`claude-opus-5-5` and the judge on `claude-sonnet-5-5`, both read from the
transcripts of every call.

## The table

From `results.md`, every row, in the order the scorer writes it:

| Pair | Arm | Critic | Verdict | Judge | Quote in review | Files read | Outcome |
|---|---|---|---|---|---|---|---|
| ftol-stall | defect | method-critic | REJECT | YES | yes | inside | hit |
| ftol-stall | defect | implementation-critic | REVISE | YES | yes | inside | hit |
| ftol-stall | defect | red-team | REJECT | YES | yes | inside | hit |
| ftol-stall | control | method-critic | REJECT | NO | no | inside | clear |
| ftol-stall | control | implementation-critic | REVISE | NO | no | inside | clear |
| ftol-stall | control | red-team | REJECT | NO | no | inside | clear |
| manifest-key | defect | method-critic | REJECT | YES | yes | inside | hit |
| manifest-key | defect | implementation-critic | REVISE | YES | yes | inside | hit |
| manifest-key | defect | red-team | REVISE | YES | yes | inside | hit |
| manifest-key | control | method-critic | REVISE | NO | no | inside | clear |
| manifest-key | control | implementation-critic | REVISE | NO | no | inside | clear |
| manifest-key | control | red-team | REVISE | YES | yes | inside | false alarm |
| obs-routing | defect | method-critic | REJECT | YES | yes | inside | hit |
| obs-routing | defect | implementation-critic | REVISE | YES | yes | inside | hit |
| obs-routing | defect | red-team | REJECT | YES | yes | inside | hit |
| obs-routing | control | method-critic | REVISE | NO | no | inside | clear |
| obs-routing | control | implementation-critic | REVISE | NO | no | inside | clear |
| obs-routing | control | red-team | REVISE | NO | no | inside | clear |
| sorted-glob | defect | method-critic | REJECT | YES | yes | inside | hit |
| sorted-glob | defect | implementation-critic | REVISE | YES | yes | inside | hit |
| sorted-glob | defect | red-team | REVISE | YES | yes | inside | hit |
| sorted-glob | control | method-critic | REVISE | NO | no | inside | clear |
| sorted-glob | control | implementation-critic | REVISE | NO | no | inside | clear |
| sorted-glob | control | red-team | REJECT | NO | no | inside | clear |
| stale-affine | defect | method-critic | REJECT | YES | yes | inside | hit |
| stale-affine | defect | implementation-critic | REJECT | YES | yes | inside | hit |
| stale-affine | defect | red-team | REJECT | YES | yes | inside | hit |
| stale-affine | control | method-critic | REJECT | YES | yes | inside | false alarm |
| stale-affine | control | implementation-critic | REVISE | YES | yes | inside | false alarm |
| stale-affine | control | red-team | REJECT | YES | yes | inside | false alarm |

| Critic | Hits | Noted | Misses | False alarms | Clear | Excluded |
|---|---|---|---|---|---|---|
| method-critic | 5 | 0 | 0 | 1 | 4 | 0 |
| implementation-critic | 5 | 0 | 0 | 1 | 4 | 0 |
| red-team | 5 | 0 | 0 | 2 | 3 | 0 |

Blocking verdicts (REVISE or REJECT): 15 of 15 on the defect arm (9 REJECT,
6 REVISE) and 15 of 15 on the control arm (5 REJECT, 10 REVISE).

The four false alarms, with the passage the judge quoted from each review
(each quote was checked to be in the review):

| Pair | Critic | Quoted passage |
|---|---|---|
| manifest-key | red-team | "It is missing the converse: if the audit cannot see a known overwrite" |
| stale-affine | method-critic | "The current scorecard BR affine RMSE is 0.097, not 0.1054, so the published 0.8% is already stale." |
| stale-affine | implementation-critic | "The reference number has moved." |
| stale-affine | red-team | "the scorecard's BR corrected figure is now 0.097, not 0.1054" |

## Checks the registration asked for

- **Judge calibration:** 10 of 10 fixtures judged as registered (each pair's
  positive found, its negative not found), so the bar is read.
- **File reads:** every review's "Files read:" list lies inside its snapshot.
  The transcripts were readable, so every Read, Grep and Glob call of every
  critic was checked as well (`read_audit.json`): none reached outside its
  snapshot. No review is excluded.
- **Predictions:** each defect was expected to be caught by one or two named
  critics. All three critics caught all five, so every prediction was met.

## Departures from the registration

None changes an outcome. Each is listed so the record matches what ran.

- **How replies were saved.** The registration said each reply is "saved
  unedited". The orchestrating session wrote each one by extracting the
  critic's final message from its transcript with a script, rather than
  copying it by hand. The script also wrote `read_audit.json`.
- **Order.** The 30 critic calls were launched in token order, in three
  batches of ten that ran concurrently, so they finished in another order.
- **`harness_dirty` is true in `run.json`.** `prepare` writes the run
  directory before it reads `git status`, so the untracked run directory
  itself makes the tree dirty. No tracked file differed from `654ad76`.
- **The judges read their prompt from a file** and wrote their two lines with
  the Write tool. The registration did not say how a prompt reached a judge.

## Caveats

The registration's caveats all apply: one draw per review; defects stated in
the proposal text rather than latent in code; cases written by an agent that
knew the answers; today's critics on defects whose lessons their definitions
partly carry; a model as judge. The registered reading for a fail on false
alarms is that the critics name a defect from its surface rather than its
mechanism. Whether the four false alarms fit that reading, or reflect the
controls' design (three of the four cite figures already published in the
snapshot rather than the proposal's own column), is left to the maintainer.
