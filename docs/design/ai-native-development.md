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
and the written rule stays for what the guard misses. Every guard is tested in
both directions, firing on a known positive and passing a known negative
(`tests/test_agent_guards.py`), because a guard never seen to fire is an
assumption.

| Guard | Enforces | Misses |
|---|---|---|
| Deny rules, `.claude/settings.json` | `git tag`, `gh release`, `gh pr merge`, a bare `git push`, a push naming `main`, `HEAD`, tags or a version, and `git diff`, `show` or `log` with `--output` | Anything that reaches git by another text: `git -C . push`, a full path, `sh -c`, a script. A branch whose name contains `main` or `HEAD` is blocked too. |
| `.claude/hooks/guard_run_lock.py` | While `scripts/dev/run_locked.py` holds `output/.run-lock`: no file edit on a path git does not ignore, no tree-changing git command, and no shell write to such a path by a redirect, `tee` or `sed -i` | A run not started through the wrapper; a write by any other program (`cp`, `mv`, `python -c`, a script), to a target held in a variable, or after a `cd`; a reused process id. The liveness check assumes a POSIX system. |
| `.claude/hooks/guard_local_suites.py` | A commit that stages code a local-only suite in `scripts/dev/stamp.py` covers (`realdata`, the real-data pins; `pinn`, the physics-informed tests) needs a passing stamp whose fingerprint matches the staged content. The fingerprint ignores comments and docstrings. `git commit -a`, and staging covered code in the same command as the commit, are refused | A commit made outside Claude Code or by a script it cannot see into; a commit hidden in `sh -c` or an alias gets the staged-code check, by a plain text match, but not the `-a` refusal; a same-command `git add` after a `cd` into a subdirectory, or with `-C`, is judged against the wrong base; a `realdata` run where every pin skipped still passes, which is why the skips are read and stated. Changes to the physics-informed study drivers are not covered, because no test exercises them. |
| `.claude/hooks/guard_docs_critic.py`, registered in `.claude/agents/docs-critic.md` | The docs-critic agent's Bash: only `git diff`, `git show`, `git log` and `wc`, with no git option before the subcommand, no `--output` and no redirect. It fails closed | Nothing outside that agent. |
| `no-em-dash` hooks, `.pre-commit-config.yaml` | No em dash in a committed file or a commit message | Nothing, once `pre-commit install` has run for both hook types. |

The run-lock and stamp hooks fail open: an unexpected error lets the tool call
through and shows the error, so a broken guard cannot block all work and is
visible rather than silent.

Taken together, the guards have two limits. They see only the tool calls Claude
Code makes, so a commit made outside it, or by a script it cannot read, is not
checked. They match command text, so they are not a security boundary: a
denied command routed through another text would pass, and `AGENTS.md`
forbids that route rather than preventing it. A denied command means stop, and
the command goes to the maintainer.

## The rules, and the incidents behind them

Each rule in `AGENTS.md` came from a failure. The accounts, briefly:

- **Create nothing in the tree while a run is in flight.** A run stamps
  `git_dirty` from `git status --porcelain`, which counts an untracked file, so
  one file created mid-run marks every manifest of the run dirty, and the only
  repair is to repeat the run.
- **Read the state now, not the state you expect.** In one day a CI run
  dispatched with the push tested the commit before it, `git add -A` swept an
  unrelated file into a commit, and a message stated a test-file count from
  memory. A grep of a batch's output also kept an exception and dropped the
  line that named its cause.
- **Resolve a file the way the code does.** Globbing `no_grid_points_20*.csv`
  and taking the last entry returned a `.zonemixed.bak.csv` backup, which
  sorts after `.csv`; the loader resolves by exact name. Two retractions came
  from that glob.
- **Test a check both ways.** Three clean answers in two days came from checks
  that never ran, among them an audit reading `cluster_list` at the wrong level
  of the manifest, which flagged none of 224 runs when nine known positives
  were available. The next day a reproduction check flagged everything, by
  comparing full precision against values recorded to five decimals.
- **List a path before overwriting or deleting it.** Clearing two
  contaminated regions deleted two clean rows' manifests with them, and the
  reproduction meant to restore them overwrote the originals it was to be
  checked against.
- **Run the real-data pins when you touch the code they cover, the way their
  tests run them.** CI cannot run them, so no pull request check shows a pin
  moving. Until 2026-09-24 the covered list was `harness/`, `metrics.py`,
  `correction.py` and `data.py`, so a change to `wind.py` that moved two
  published rows (AU-NEM and NZ) needed no stamp; the list is now every module
  under `src/pyvwf` except `pinn/`, `viz/` and `cli/`. A UK pin was once reported as moving by 0.0014 in MBE when the row had
  been run on the wrong input root; the movement was 1e-16. The stamp
  fingerprints each covered file's syntax tree without its docstrings, so a
  comment or docstring edit keeps a stamp valid; this is safe because no
  covered code reads its own source or a docstring to compute anything. The
  realdata suite may run in two or three processes:

  | Workers | Wall time | Passed / failed / skipped | Largest process |
  |---|---|---|---|
  | 1 | 728 s | 43 / 0 / 0 | not measured |
  | 2 | 404 s | 43 / 0 / 0 | 9.6 GB |
  | 3 | 340 s | 43 / 0 / 0 | 8.8 GB |

  Measured on 2026-10-16 on a 16 GB machine (parallel runs at `8dd2c2c`, the
  serial stamp at `83442de`). Three workers left under 1 GB of swap free, so
  two is the count to use there; the script refuses more than three. The
  physics-informed suite accepts only one worker.
- **Run the checks, read them, then write the message.** A message that says a
  check passed is a claim about output already seen.

## The critics' models

`method-critic` and `red-team` run on `opus` and `implementation-critic` on
`sonnet`, so two of the three share a model. Until 2026-10-16 all three ran on
the session's own model, so three verdicts were one model prompted three ways;
from 2026-10-16 to 2026-10-22 `red-team` ran on `fable`, so the three used three
models. Different models should make errors less correlated, which has not been
measured, but not uncorrelated: all three are Claude models, reading the same
proposal and the same repository. A blocker all three raise is weaker evidence
than three independent referees would give. The aliases resolve to whichever
version Claude Code maps them to, and nothing records which model produced a
critique, or how often a critic's blocker was right.

## The mistake log

An audit of the history, from 2026-05-20 to 2026-10-27, confirmed 98 mistakes.
It counted inverse diffs, fix chains (a commit after merge changing lines
another made in the previous 14 days, found with `git blame`), correction
notices and re-recorded pins, and the incidents above. Of the 188 candidate
records, 80 were not mistakes (refinements, plan changes, refactors and the
like), and the other 108 were 98 distinct mistakes. Who made each was
read from the Claude Code transcripts, which cover the whole period except the
session behind the commits of 2026-05-20, 12:52 to 13:21; rows marked "git
only" rest on git history alone. **The full log, every row with its
commits, class and catch, is at tag `archive/research-2026-10-27`:
[`docs/design/agent-guards.md`](https://github.com/ellyess/PyVWF/blob/archive/research-2026-10-27/docs/design/agent-guards.md).** This page keeps its
summary and the mistakes that reached publication.

Who made the 98: agents 85, agents at the maintainer's request 4, the
maintainer 4 (all in code from before agents), and cannot tell 5.

| Stage reached | Mistakes | Agent itself | Audit | CI | Maintainer | Test | Critic agent |
|---|---|---|---|---|---|---|---|
| After publication | 31 | 10 | 15 | 0 | 3 | 1 | 2 |
| After merge | 13 | 4 | 6 | 3 | 0 | 0 | 0 |
| Before merge | 44 | 34 | 2 | 3 | 2 | 2 | 1 |
| Before commit | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| Total | 98 | 58 | 23 | 6 | 5 | 3 | 3 |

"Audit" is an audit the maintainer commissioned, by an agent, or in three cases
by a session in another project. 31 mistakes reached publication; agents made
25 of them, 19 of those in the period before written rules. Audits, not
standing guards, found 15 of the 31. Agents caught 58 of their own mistakes, 44
before merge. No hook caught a live mistake.

Per 100 commits on main, counting only fix chains and inverse diffs, which were
recorded the same way throughout:

| Period | Commits on main | Agent-made | Per 100 commits | All confirmed | Per 100 commits |
|---|---|---|---|---|---|
| P1 | 228 | 18 | 7.9 | 20 | 8.8 |
| P2 | 319 | 10 | 3.1 | 10 | 3.1 |
| P3 | 216 | 3 | 1.4 | 3 | 1.4 |

P1 is 2026-05-20 to 09-10 (no written rules), P2 09-11 to 09-22 (rules and
skills), P3 09-23 onwards (rules, skills and hooks). Correction notices and
incident accounts are left out of the rates because they begin on 2026-09-11,
with the written rules, and so undercount P1 by construction. The fall
coincides with the rules and the hooks, and **does not show that they caused
it**:

- **Dating.** Dated by fix instead of introduction, the agent-made counts are
  15, 5 and 11: P3 is then above P2.
- **Eligibility.** Fix chains count only fixes made after merge. P1 committed
  straight to main, so every fix was eligible; P2 worked on long branches, so
  most of its fixes came before merge and are not counted; P3 used 84 small
  pull requests.
- **Time to be found.** P3 work was at most five weeks old when counted; P1
  work had been searched by the audits of 2026-09-11 and 2026-09-24.
- **Kind of work and models.** P1 was mostly new code, P2 studies and
  pre-registrations, P3 refactors, guards and the observation rebuild, and the
  models changed between periods.
- **Small counts and the window.** P3 has three. Fix chains look back 14 days,
  so most published mistakes, which lived longer, appear only through notices,
  which cannot give a rate.

### Mistakes that reached publication (31)

Publication means a findings document, the scorecard, the CHANGELOG, the README
or the paper on main, or a release, tag or Zenodo version. In date order:

| ID | Date | Mistake | Introduced | Fixed | Period | Who | Class | Caught by | Time to catch | Guard or eval case today |
|---|---|---|---|---|---|---|---|---|---|---|
| C008 | 2026-02-13 | Metrics hard-coded to a 2015 to 2019 window | `a95e134` | `3775bb5`, `9680157` | before P1 | cannot tell (git only) | `untested-path` | test | >=151 d | none (path retired) |
| C017 | 2026-02-13 | Usage document said roughness comes from terrain data | `a95e134` | `9f26e32` | before P1 | cannot tell (git only) | `doc-drift` | agent itself | 211 d | none |
| C071 | 2026-02-13 | Joint-fit tolerance stopped 47 of 540 fits at their starting point | `a95e134` | `26e0b67` | before P1 | maintainer (git only) | `numerical-method` | critic agent | 224 d | machine: `test_country_joint_fit_tolerance.py`; eval `ftol-stall` |
| C153 | 2026-02-13 | Country offset fit accepted non-converged and on-bound results | `a95e134` | `1a19b52` | before P1 | maintainer (git only) | `numerical-method` | audit agent | 223 d | machine: `test_country_offset_refusal.py` |
| C186 | 2026-02-13 | An hour missing one wind type reported as the whole fleet | `a95e134` | `84dd1a9` | before P1 | maintainer (git only) | `numerical-method` | agent itself | 252 d | machine: `test_entsoe_fetch_consistency.py` |
| C097 | 2026-05-20 | Quantile mapping added unasked, and written into the README and paper | `4e416b1` | `5ebc741` | P1 | cannot tell (git only) | `scope-creep` | maintainer | 8 d | rule: Propose, then stop |
| C014 | 2026-07-06 | Country rows simulated on the 100 kW fallback curve | `f4b80b5` | `bcf4dfa` | P1 | agent | `silent-fallback` | audit agent | 67 d | machine: country runs refuse a missing curve |
| C121 | 2026-07-14 | Landing-page math rendered as raw text; shipped in v0.2.0 | `4b1bbe4` | `be4c0c5` | P1 | agent | `untested-path` | maintainer | 2 d | none (`sphinx-build -W` passes) |
| C162 | 2026-07-14 | Test fixture set INPUT_ROOT for only some of its users | `3775bb5` | `1a5548a` | P1 | agent | `env-divergence` | audit agent | 73 d | none |
| C050 | 2026-07-15 | A finding cited a document that was never committed | `af99418` | `4f32990` | P1 | agent | `data-provenance` | audit agent | 65 d | rule: findings-doc |
| C012 | 2026-07-23 | UK and NZ presented as corrected wins | `7a2f010` | `4253985` | P1 | agent | `overclaiming` | agent itself | 50 d | rule: findings-doc |
| C013 | 2026-08-12 | Corrected and uncorrected output scored on different units | `3dd8498` | `513f57c`, `37bba11` | P1 | agent | `numerical-method` | agent itself | 30 d | machine: `test_common_row_scoring.py` |
| C015 | 2026-08-12 | Italy, Portugal and Spain winds extrapolated beyond the loaded extent | `7d634dc` | `d5fd5a5`, `c480f46` | P1 | agent | `silent-fallback` | agent itself | 30 d | machine: `ExtrapolationError` |
| C016 | 2026-08-12 | Rows on different roughness treatments compared without saying so | `7d634dc` | `9f26e32` | P1 | agent | `data-provenance` | agent itself | 31 d | rule: findings-doc |
| C018 | 2026-08-12 | A Denmark study presented as a reproduction of the published paper | `3dd8498` | `cef131d` | P1 | agent | `overclaiming` | agent itself | 31 d | rule: findings-doc |
| C036 | 2026-08-12 | Sweden's zonal capacity-factor denominators back-derived | `dd87232` | `40317ee` | P1 | agent | `data-provenance` | agent itself | 34 d | machine: fetcher refuses derived capacity |
| C046 | 2026-08-12 | National multi-cluster fits routed to the per-cluster solver | `56ab195` | `ac26f6a`, `dc14687` | P1 | agent | `numerical-method` | critic agent | 44 d | machine: `test_country_zonal.py`; eval `obs-routing` |
| C052 | 2026-08-12 | Offshore capacity summed onto land grid points | `dd87232` | `643499a`, `fc8c7f5` | P1 | agent | `numerical-method` | audit agent | 44 d | machine: `test_weight_country_grid_points.py` |
| C055-A | 2026-08-12 | Belgium's observations left onshore-only while its offshore capacity joined the fleet | `dd87232` | `63bb2e2`, `7fc0e16` | P1 | agent | `data-provenance` | audit (session in another project) | 71 d | partly machine: fetch default is now all types; fleet coverage is rule only |
| C055-B | 2026-08-12 | Gates that cannot see a partial-fleet series reported as validating Belgium and France | `dd87232` | `b567df4` | P1 | agent | `one-direction-check` | audit (session in another project) | 71 d | rule: Test a check both ways |
| C059-D | 2026-08-12 | Scalar floor set so that it stayed quiet on Belgium | `569a00a` | `232ee52` | P1 | agent | `one-direction-check` | audit (session in another project) | 71 d | rule: Test a check both ways |
| C064 | 2026-08-12 | "57" Netherlands scalars flagged where the count is 29 | `569a00a` | `a087b92` | P1 | agent | `miscount` | audit agent | 71 d | rule: Count what you assert |
| C074 | 2026-08-12 | GWPT filter built today's fleet for every year | `dd87232` | `acfbbfa` | P1 | agent | `data-provenance` | audit agent | 44 d | machine: `test_gwpt.py` |
| C053 | 2026-08-23 | Physics-informed scores taken on a mismatched row basis | `cb91ac9` | `46a039f` | P1 | agent | `numerical-method` | agent itself | 53 d | rule: findings-doc; eval `stale-affine` |
| C009 | 2026-08-24 | CHANGELOG said the rows reproduce without the licensed library | `2795bb0` | `c96090f` | P1 | agent | `overclaiming` | audit agent | 18 d | rule: provenance-guard |
| C010 | 2026-09-11 | "Seven rows" shipped in the v0.5.1 tag and Zenodo version; four was right | `c96090f` | `ef0d79a` | P2 | agent | `overclaiming` | agent itself | 51 min | rule: provenance-guard |
| C095 | 2026-09-11 | Tag, release and DOI for v0.5.1 that were the maintainer's to make | not committed | `7a3bfd5` | P2 | agent, at request | `authority` | maintainer | 1 h | machine: deny rules |
| C060 | 2026-09-15 | Sweden's row republished without its denominator caveat | `40317ee` | `ec35db7`, `293b01c` | P2 | agent | `overclaiming` | audit agent | 37 d | rule: findings-doc |
| C011 | 2026-09-16 | "No fit affected" inferred from a golden test that cannot fire on the changed branch | `724ab1b` | `027608b` | P2 | agent | `one-direction-check` | audit agent | 2.6 d | partly machine: offset pins |
| C063 | 2026-09-16 | "Five" small folds where the fold table shows eight | `2f300ad` | `68874fb` | P2 | agent | `miscount` | audit agent | 39 d | rule: Count what you assert |
| C065 | 2026-09-16 | Inverse distance weighting disagreement stated as 0.0 exactly; it is 2.7e-15 | `020199e` | `4ea8bfe` | P2 | agent | `overclaiming` | audit agent | 39 d | rule: Read a command's whole output |

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
summarised in `scripts/dev/agent_evals/results-2026-10-16.md`. No training
year or test year applies: the run measures the review agents, not a
correction.

Two further runs are registered and not yet run: the three critics against a
generic reviewer on the same models
(`docs/findings/agent-critics-baseline-prereg.md`), and `docs-critic` against
one on five held-out page changes (`docs/findings/agent-critics-docs-prereg.md`).
Each is gated on the critic beating the baseline by a margin fixed before any
review, and a "no measurable difference" outcome stands as the finding.

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
