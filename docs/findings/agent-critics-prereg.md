# Do the review agents catch the defects they exist to catch?

**Date:** 2026-10-16. Registered before any review agent has seen a case.
**Scope:** the three review subagents in `.claude/agents/`, `method-critic`,
`implementation-critic` and `red-team`, as defined at the commit of this
registration, on five matched pairs of proposals. No training year or test
year: this measures the review agents, not a correction. The one case built on
a physics-informed result states its regions' training and test years inside
the proposal. Terms follow `CONTEXT.md`.

**The run asks whether the three critics, given a proposal that carries a
defect this repository has already found, name that defect and block on it,
and whether they leave the same proposal alone once the defect is removed.**
It decides nothing about the critics' definitions. What to change afterwards
is the maintainer's decision after reading the table.

## Why

`AGENTS.md` asks that a check be shown to fire on a known positive and to
return a negative before either answer is trusted. The critics are checks, and
`../design/ai-native-development.md` records that nothing measures them: no
record says how often a critic's blocker was right or what the critics miss.
This registration applies the rule to them.

## The cases

Each pair is one TOML file in `scripts/dev/agent_evals/cases/`. Its defect arm
is a proposal whose design contains a defect the repository found and fixed;
its control arm is the same proposal with the defect-bearing sentences replaced
by the fix the repository adopted. Every other sentence is identical, and the
`Claim` paragraph is identical by test.

| Pair | The defect | Found and fixed in | Recorded in | Critic reads the tree at | Expected to catch |
|---|---|---|---|---|---|
| `ftol-stall` | The joint national fit's `ftol` of 1e-6 is absolute on an objective near 1e-4, so a fit stops after one step with offsets at zero and reports success; the proposal reads near-zero offsets as physics. | `26e0b67` | `scorecard.md`, the note on the joint fit's stopping tolerance | `26e0b67^` | method-critic, implementation-critic |
| `obs-routing` | The router counts distinct floating-point cluster means with `nunique`, so every national fit with more than one cluster goes to the per-cluster solver, not the joint fit. | `ac26f6a` | `CHANGELOG.md`, 2026-09-25 | `ac26f6a^` | implementation-critic |
| `sorted-glob` | A region's grid file is found by sorting a glob and taking the last entry, not by the loader's exact names, so a backup file can be reported as the grid in use. | `9b0d26e` | `../design/agent-guards.md`, "Resolving files by listing" | `9b0d26e^` | implementation-critic |
| `manifest-key` | An audit reads `cluster_list` at the manifest's top level, where it is not, skips every run as undeclared, and flags nothing; it is never run on the known contaminated runs. | `cc9fc14` | `../design/agent-guards.md`, "Detectors trusted in one direction" | `cc9fc14^` | implementation-critic, method-critic |
| `stale-affine` | The zero-shot model is compared with an affine column read from a superseded sweep and scored on the harness's common rows, which leave out units in refused clusters, while the model is scored on every row. | `8dd2c2c`, `46a039f` | `method-physics-informed.md`, correction notice of 2026-10-16 | `8dd2c2c^` | method-critic, red-team |

The `obs-routing` defect was first found by `implementation-critic` reviewing a
proposal (the message of `ac26f6a` says so). That proposal is not in the
repository, so this pair is written afresh like the others.

**Blinding.** A critic sees only the proposal text and the path of a
checkout. The proposal carries no commit hash, no document path, and none of
the words that would mark it as a test (`blinding_problems` in
`scripts/dev/agent_evals/evals.py`, which `tests/test_agent_evals.py` shows
firing on a leaking text and passing every case). The checkout is the tracked
tree at the parent of the fixing commit, written by `git archive` outside the
repository, with no git metadata, under a directory named by a hash. Each
snapshot was checked before this registration to hold neither its own fix nor
its write-up: `JOINT_FIT_FTOL` is absent at `26e0b67^`, `OBS_SAME_TOLERANCE` at
`ac26f6a^`, the `zonemixed` account at `9b0d26e^`, the `cluster_list` account at
`cc9fc14^`, and the correction notice at `8dd2c2c^`; each is present in a later
snapshot, so the search can return a positive. Both arms of a pair are
reviewed against the same snapshot.

## Procedure

1. `python scripts/dev/agent_evals/evals.py prepare --label 2026-10-16
   --critic-model claude-opus-5-5 --judge-model claude-sonnet-5-5` writes 30
   prompts (5 pairs, 2 arms, 3 critics) and `run.json`, which records the
   models, the harness commit, the SHA-256 of each critic definition, and each
   snapshot's commit.
2. A Claude Code session sends each prompt to the subagent `run.json` names,
   once, in the order of the prompts' tokens, which interleaves pairs, arms and
   critics. The critics run on `claude-opus-5-5`, the session's model, since
   their definitions say `model: inherit`. A critic receives the prompt alone:
   no scout brief, no other critic's output. Its reply is saved unedited. A
   call that returns no review is repeated once, and the repeat is recorded.
3. `evals.py judge` writes one judge prompt per review and two calibration
   prompts per pair. Each goes to a fresh subagent on `claude-sonnet-5-5`,
   which answers in two lines, `FOUND: YES|NO` and `QUOTE:`. The judge sees the
   pair's defect description and the review, without the critic's name, the
   arm, or the list of files read. The description is the same for both arms,
   so the judge cannot tell a control's review from a defect's except by what
   it says.
4. `evals.py score` writes `results.json` and `results.md`.

The run's directory, `scripts/dev/agent_evals/runs/2026-10-16/`, holds every
prompt, review and judgement, and is committed with the results.

## Scoring rule

- **Verdict:** the first `Verdict:` the review states, PROCEED, REVISE or
  REJECT. A review stating none has no verdict.
- **Found:** the judge answers YES and quotes at least 15 characters that the
  review contains, ignoring whitespace, case, asterisks and backticks. A YES
  whose quote is not in the review does not count.
- **Defect arm:** a **hit** is found with a REVISE or REJECT verdict. Found
  with PROCEED or no verdict is **noted**, and counts as a miss for the bar.
  Not found is a **miss**.
- **Control arm:** a **false alarm** is found: the review asserts the paired
  defect in a proposal that carries the fix. Otherwise **clear**. A control's
  verdict does not decide a false alarm, because a critic may block a control
  for a reason of its own, and that reason may be right. The blocking verdicts
  on each arm are counted and reported.
- **Excluded:** a review whose closing "Files read:" list names a path outside
  its snapshot. It counts in neither direction and is reported. A review with
  no list is scored and reported as "not listed". The limit on reading is an
  instruction, not a sandbox: the session's working directory holds the cases.
  If the subagents' transcripts can be read, their file reads are checked too,
  and any read outside the snapshot is reported beside the table with the
  counts recomputed both ways.
- **Judge calibration:** each pair's positive fixture, a short review naming
  the defect, must be found; its negative fixture, a review raising something
  else, must not. If any of the ten fails, the bar is not read, and the table
  is reported as unscored by a calibrated judge.

## The bar

**Pass:** the panel, any of the three critics, catches at least **4 of the 5**
defects, and raises the paired defect on at most **1 of the 5** controls.
Per-critic counts are reported without a bar. The expected catchers in the
table are predictions, reported against the outcome; they are not gates.

The panel is the unit because `idea-review` runs the three together and acts
on any blocker. With five pairs one defect moves the panel by 20 points, so
the bar is a smoke test: a pass says the critics, once each, named defects of
these kinds; it is not a detection rate.

## Readings, fixed now

- **Pass:** under these conditions the critics catch defects of the kinds this
  repository has met, stated in a proposal in front of them, and do not cry
  wolf on the fixed version. The results document says so and no more.
- **Fail on hits:** the pairs missed, and whether the critic expected to catch
  each saw it at all, are the result. A miss by every critic on a pair is
  reported as the critics' blind spot for that kind of defect.
- **Fail on false alarms:** the critics name a defect from its surface, the
  code or setting, rather than its mechanism.
- Either result goes in `../design/ai-native-development.md` beside the
  workflow it measures, negative or not.

## Caveats

- **One draw per review.** The agents are not deterministic and take no seed;
  a second run could differ. This run is one sample.
- **The defects are in the text.** Each defect arm states its defective design
  in the proposal. The run tests whether a critic recognises a defect put in
  front of it, not whether it finds one latent in code it was not pointed at,
  which is how `obs-routing` was first found.
- **The cases were written knowing the answers,** by an agent session that
  then runs them. The proposals are not the originals, which the repository
  does not keep.
- **Today's critics on yesterday's defects.** The critic definitions date from
  2026-09-23 and later, and carry some lessons of these incidents in general
  form: `implementation-critic` asks for "a known-positive case for any new
  check", which bears on `manifest-key`, and `method-critic` asks for a
  comparison "on the same fleet, years", which bears on `stale-affine`. This
  favours the critics. The snapshots' own `AGENTS.md` carries the
  file-resolution rule in every snapshot after `9b0d26e^`.
- **The judge is a model.** The calibration catches a judge that says yes, or
  no, to everything; it does not catch subtler misgrading. Every review and
  every judgement is committed, so the table can be re-scored by hand.
