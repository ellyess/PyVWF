# Development with coding agents

PyVWF began as the code of the maintainer's PhD, written by hand from 2022
through the thesis (`docs/publications.md`). The method, the affine correction
and its first validation were built that way, before any coding agent. Of the
844 commits on `main` at `1e6e7e1`, 81 predate July 2026. Since mid-2026 it has
been developed with Claude Code, a coding agent, working under written rules
and machine-checked guards; the agent instructions and skills were first
committed on 2026-09-11 and the guard hooks on 2026-09-23. The maintainer owns
the method, the gates and every scientific decision. Agents implement changes,
write tests and drivers, draft documents, run studies and critique proposals.
`AGENTS.md` requires each diff to be shown to the maintainer before it is
committed.

Commits carry no AI co-author line, by the maintainer's choice, and every
commit is made under the maintainer's name. The git history therefore does not
separate a change an agent wrote from one written by hand, and this page states
how agents are used instead. Terms follow [`CONTEXT.md`](../CONTEXT.md): a gate,
a pre-registration, a correction notice, a run, a manifest, a driver, the
training years and the test year each have the sense defined there.

## Who decides what

The maintainer decides:

- which question a study asks, and the method that answers it;
- the gates, which are fixed in a pre-registration before the run they judge;
- what a result means, and how it is reported, including every correction
  notice;
- every merge to `main`, tag, release and DOI. The deny rules in
  `.claude/settings.json` refuse those commands to an agent.

Agents, in sessions the maintainer directs:

- write code, tests and study drivers, and run the studies through the run
  lock;
- draft findings documents and correction notices, through the `findings-doc`
  skill;
- gather prior work and critique a proposal, as the subagents described below;
- check provenance before a release-shaped action, read-only, and stop.

## The workflow

Each piece is a committed file, so a reader can open it.

| Piece | Where | What it does |
|---|---|---|
| Instructions | `AGENTS.md`, `src/pyvwf/AGENTS.md` | The root file holds the maintainer's decisions and the rules for AI-assisted contributions and for acting on state. The second loads for package code. Each `CLAUDE.md` imports its `AGENTS.md`. |
| Skills | `skills/`, linked as `.claude/skills` | `new-region` adds a region in gated phases and stops for sign-off at each. `findings-doc` keeps every number traceable, keeps negative results and dates every correction. `provenance-guard` checks a release, read-only, and never performs it. |
| Subagents | `.claude/agents/` | `method-critic`, `implementation-critic` and `red-team` review a proposal independently, with read-only tools. `docs-critic` reviews a change to a published page, with Bash limited by a hook to read-only git and `wc`. A scout, a size critic and the skill that chained them with the critics are kept on the maintainer's machine rather than in the repository: none was evaluated. |
| Guards | `.claude/settings.json`, `.claude/hooks/`, `.pre-commit-config.yaml` | Deny rules for release-shaped commands; a hook that blocks edits while a run holds the lock; a hook that refuses a commit of code a local-only suite covers without a passing stamp; a pre-commit hook against em dashes. |
| Guard tests | `tests/test_agent_guards.py` | Every guard is shown to fire on a known positive and to pass a known negative. |
| Pre-registrations | `docs/findings/*-prereg.md` | Gates and predictions, committed before the run they judge. |
| Run lock | `scripts/dev/run_locked.py` | Refuses to start a run on a dirty tree, and holds `output/.run-lock` while it runs, so every manifest records a clean tree. |
| Local-suite stamps | `scripts/dev/stamp.py` | Runs the real-data pins or the physics-informed tests in full, which CI cannot, and records a fingerprint of the code they covered. |
| Manifests | each run's `run_manifest.json` | The package version, the git state, the config and the curve library behind a run's numbers. |

## Guards, and what they miss

A written rule alone did not stop a tag and a release that were the
maintainer's to make. Since then, a rule a machine can check becomes a guard,
and the written rule stays for what the guard misses.
[`agent-guards.md`](agent-guards.md) lists each guard with what it misses, and
keeps the incident behind each rule in `AGENTS.md`.

Taken together, the guards have two limits. They see only the tool calls Claude
Code makes, so a commit made outside it, or by a script it cannot read, is not
checked. They match command text, so they are not a security boundary: a
denied command routed through another text would pass, and `AGENTS.md`
forbids that route rather than preventing it.

## What this record does not show

- **The critics are evaluated once, on five cases.** The run below says
  whether they named five known defects and left the fixed versions alone. It
  is one draw per review, not a rate, and it does not record how often a
  critic's blocker on a new proposal was right.
- **Authorship is not recorded per change.** No commit says whether an agent
  wrote it, so no count below separates agent work from the maintainer's.
- **The counts measure volume, not correctness.** A correction notice records
  an error that was found. An error not yet found leaves no record.
- **The rules before 2026-09-11 are not in the repository.** `AGENTS.md` was
  first committed on that date (`7a3bfd5`). Before it, agent instructions were
  kept in a local `CLAUDE.md`, which `.gitignore` has excluded since `1b10581`
  (2026-08-12).

## Evaluating the critics

A critic is a check, so `AGENTS.md` asks the same of it as of any check: show
it fires on a known positive and stays quiet on a known negative. Five
defects this repository found and fixed were each written into a proposal,
and each proposal was paired with a copy carrying the fix instead. Each critic
reviewed all ten, blind, against a checkout of the repository from before the
fix. A second model judged whether each review named the defect, quoting it.
The cases, the scoring rule and the pass bar were committed before any critic
saw a case: `docs/findings/agent-critics-prereg.md`. The driver and the cases
are in `scripts/dev/agent_evals/`, and `tests/test_agent_evals.py` checks the
scoring both ways.

The one run, on 2026-10-16, did not pass its bar. All three critics named all
five defects and blocked on them. The panel also raised the paired defect on
2 of the 5 fixed proposals, where the bar allowed 1, and no critic gave
PROCEED on any of the 30 proposals, fixed or not. The table, the quoted
passages and the departures from the registration are in
`docs/findings/agent-critics.md` at tag `archive/research-2026-10-27`, and
summarised in `scripts/dev/agent_evals/results-2026-10-16.md`. No training year or test year applies: the
run measures the review agents, not a correction.

## Counts

`scripts/dev/repo_stats.py` produced this table from git and the tracked tree
at one commit; it and its test are at tag `archive/research-2026-10-27`.

Counts at `219567a` (2026-10-16), from
`python scripts/dev/repo_stats.py --rev 219567a`.

| Count | Value | What it counts |
|---|---|---|
| commits | 837 | reachable from the commit, merges included |
| pull requests merged | 83 | distinct numbers in 'Merge pull request #N' messages; a squashed or rebased PR is not counted |
| test files | 103 | tests/**/test_*.py |
| test functions | 945 | def test_ in those files, before parametrisation |
| agent guard tests | 18 | test functions in tests/test_agent_guards.py |
| findings documents | 45 | docs/findings/*.md, the scorecard included |
| pre-registrations | 17 | docs/findings/*-prereg.md and the one *-prespecification.md |
| correction notices in findings | 34 | '**Correction notice, <date>' in docs/findings |
| correction notices in the CHANGELOG | 2 | '*[Correction, <date>' in CHANGELOG.md |
| AGENTS.md files | 3 | agent instruction files |
| skills | 4 | .claude/skills/*/SKILL.md |
| subagents | 4 | .claude/agents/*.md |
| hooks | 2 | .claude/hooks/*.py |

The first commit is from 2022-06-22, so the commit count covers the project's
whole history, including the years before any agent instruction entered the
repository.

## Case study: a stale comparison in the physics-informed study

The physics-informed study compares a model applied zero-shot, to a region it
was not trained on, with the affine correction fitted in that region. Its
findings document is `docs/findings/method-physics-informed.md`, at tag
`archive/research-2026-10-27`. The error and
its first explanation are recorded there, in the correction notice of
2026-10-16, and in two commits: `8dd2c2c`, which adds the driver that rescored
the comparison, and `46a039f`, which adds the correction notice. The
comparison and its first explanation were an agent's work; no commit records
that, and the maintainer confirmed it on 2026-10-16.

**Before.** Since 2026-08-24 (`ed6c3b6`, in
`method-physics-informed-fresh-gate.md`, folded into the findings document on
2026-09-01), the study said that the model applied zero-shot beat the
in-region affine correction in 3 of 4 regions never used as holdouts. Since
2026-09-01 (`8808cc8`), the findings document also said that zero-shot Brazil
matched Brazil's own fitted correction to 0.8%. Both rested on an
"affine, in region" column with two faults:

- `scripts/studies/physics-informed/e1_report.py` (now at tag
  `archive/research-2026-10-27`) read it from the cluster
  sweep of 2026-07-24 (its `SWEEP` constant). The scorecard had replaced that
  sweep four times since.
- It was scored on the harness's common rows, which leave out every unit in a
  cluster whose fit was refused. The model was scored on every row. The
  left-out units are harder: in Chile, leaving them out lowers the uncorrected
  RMSE from 0.1226 to 0.1100.

**The first explanation was wrong, and was tested before it reached the
findings document.** The first account of the gap was that the model's simulation chain and
the harness's differ. On the model's own rows, the harness's uncorrected RMSE
matches the model's uncorrected column to within 0.0008 in all nine regions,
so the chains are not the difference. The explanation was dropped, and the
commit message of `46a039f` records it as tested and wrong.

**The rescoring was checked against a known answer first.** The driver
`scripts/studies/physics-informed/affine_on_pinn_rows.py` (at the same tag) scores the current
scorecard rows on the model's rows, the way the model's drivers score. Before
its result was used, a dry run reproduced the scorecard's affine RMSE where the
affine rows cover all the model's rows (DK 0.0853, NZ 0.1046). The run was made
at `8dd2c2c` from a clean tree, under the run lock, into
`output/pinn_affine_rescore_2026-10-16/`.

**After.** On the like-for-like column, which takes the uncorrected value
where the affine fit refused a unit:

- Zero-shot is lower than the affine correction in 2 of the 4 fresh regions,
  not 3: Argentina (0.1294 against 0.1351) and New Zealand (0.0848 against
  0.1046). It is level in the Australian NEM (0.0788 against 0.0787) and higher
  in Chile (0.1279 against 0.1178). The claim is withdrawn.
- Zero-shot Brazil is 1.8% above the affine correction (0.1062 against 0.1043),
  not within 0.8%. The claim is withdrawn.
- Fitted in region, the model is lower than the affine correction in 3 of 5
  holdouts, as the document said. That claim stands.
- The gates P1 to P3 and Q1 to Q2 do not read the affine column, and are
  unaffected.

Training years and test years: Denmark 2015 to 2019, tested on 2020; Germany
and the UK 2015 to 2018, tested on 2019; the US 2019 to 2021, tested on 2022;
Brazil 2021 to 2023, Argentina and Chile 2021 to 2023, each tested on 2024;
the Australian NEM 2020 to 2022, tested on 2023; New Zealand 2019 to 2023,
tested on 2024. One test year per region. The model's figures for the four
fresh regions are still not tied to a commit.

**On the record.** The withdrawn sentences stay in place with dated brackets,
the correction notice carries the full rescored table, and `CHANGELOG.md`
records the change.

**What it shows, and what it does not.** No guard could have caught this
error. The fault was not in the model's code but in which file a report read
and which rows it scored, the shape of two written rules in `AGENTS.md`:
resolve a file the way the code does, and read the state now, not the state
you expect. What caught it was a rescoring checked against a known answer, and
a first explanation tested on identical rows before it was recorded. It was
caught 53 days after the claim was first made.
