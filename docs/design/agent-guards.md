# Guards for coding agents

Why the rules in `AGENTS.md` read as they do, which of them a machine enforces,
and what each guard misses. The rules themselves are short instructions; the
failures that produced them are recorded here, so the rules stay readable and
the reasons are not lost.

## Rules and guards

A written rule alone did not stop a tag and a release that were the
maintainer's to make. The response was a deny list, and the principle since is
that a rule a machine can check becomes a guard, and the written rule stays for
what the guard misses. Every guard is tested in both directions, firing on a
known positive and passing a known negative (`tests/test_agent_guards.py`),
because a guard that has never been seen to fire is an assumption.

| Guard | Enforces | Misses |
|---|---|---|
| Deny rules, `.claude/settings.json` | `git tag`, `gh release`, `gh pr merge`, a bare `git push`, and a push naming `main`, `HEAD`, tags or a version | Anything that reaches git by another text: `git -C . push`, a full path, `sh -c`, a script. A branch whose name contains `main` or `HEAD` is blocked too. |
| `.claude/hooks/guard_run_lock.py` | While `scripts/dev/run_locked.py` holds `output/.run-lock`: no Write, Edit, MultiEdit or NotebookEdit on a path git does not ignore, no tree-changing git command, and no shell write to such a path by a `>`, `>>` or `&>` redirect, `tee` or `sed -i` | A run not started through the wrapper; a shell write by any other program (`cp`, `mv`, `python -c`, a script), to a target holding a variable, or after a `cd`; a lock whose process id was reused. The liveness check assumes a POSIX system. |
| `.claude/hooks/guard_local_suites.py` | For each local-only suite in `scripts/dev/stamp.py`, `realdata` (the real-data pins) and `pinn` (the physics-informed tests, which need an extra only CI's extras job installs, so the stamp is the check before a push): a commit that stages the code the suite covers needs a stamp whose fingerprint matches the staged content and whose run passed. The fingerprint reads a covered `.py` file's syntax tree without its docstrings, so an edit to comments or docstrings alone keeps the stamp valid; every other file, and a `.py` file that does not parse, is read byte for byte. `git commit -a` is refused, and so is anything else that stages covered code after the hook has read the index: `git commit <paths>`, and a `git add`, `stage`, `rm` or `mv` in the same command as the commit, whenever its paths can reach covered code (a glob, `.`, `-A`, `-u` or a directory above covered code all can). | A commit made outside Claude Code or by a script it cannot see into. The `-a` refusal reads only a `git commit` it can parse in command position, skipping option values and heredoc bodies, so a message quoting `git commit -a` does not trip it; a commit hidden in `sh -c` or an alias still gets the staged-code check, through a plain text match, but not the `-a` refusal. A `realdata` run where every pin skipped still passes, which is why the skips are read and stated; the `pinn` suite instead refuses to run without its extra. Changes to `scripts/studies/physics-informed/` are not covered, because no test exercises the drivers. `tests/test_agent_guards.py` checks that every `test_pinn_*.py` file is in the `pinn` suite and that the realdata paths agree with `CONTRIBUTING.md` and `src/pyvwf/AGENTS.md`. Relative paths in a same-command `git add` are read against the session's working directory, so an `add` after a `cd` into a subdirectory is judged against the wrong base; `-C` on the `add` is not followed either. |
| `no-em-dash` hooks, `.pre-commit-config.yaml` | No em dash in a committed file or a commit message | Nothing, once `pre-commit install` has run for both hook types. |

Both hooks fail open: an unexpected error exits 1, which lets the tool call
through and shows the error to the user, so a broken guard cannot block all
work. A guard that errors is therefore visible, not silent, and is fixed rather
than worked around.

## The critics' models

The three critics judge a proposal independently.
Until 2026-10-16 all three ran on the session's own model (`model: inherit`),
so three verdicts were one model prompted three ways. Each now names its own:
`method-critic` and `red-team` run on `opus`, `implementation-critic` on
`sonnet`. From 2026-10-16 to 2026-10-22 `red-team` ran on `fable`, so the three
used three different models; since then two share one, so their errors are
likely more correlated than in that week. Different models should make errors
less correlated, which has not been measured, but not uncorrelated: all three
are Claude models from one developer, and they read the same proposal and the
same repository. A blocker
all three raise is weaker evidence than three independent referees would give.
The aliases resolve to whichever version Claude Code maps them to, and nothing
records which model produced a critique, or how often a critic's blocker was
right ([`ai-native-development.md`](ai-native-development.md)).

## The incidents

Each section keeps the account that produced a rule in `AGENTS.md`.

### A run in flight

**Create nothing in the tree while a run that writes a manifest is in
flight.** Not a tracked file,
not an untracked one, not the analysis script you intend to use on the
results. A run stamps `git_commit` and `git_dirty` into every manifest it
writes, and `git_dirty` comes from `git status --porcelain`, which counts an
untracked file too. One file created mid-run makes every manifest of that
run say `git_dirty: true`, and the only repair is to delete the runs and
repeat them. Write it outside the repository, or wait for the run to finish.
The rule is scoped to its mechanism: work alongside something that records no
git state, such as a download or a read-only audit, is not covered.

### Acting on the state you expect

**Check the state you are about to act on, not the one you expect.** Three
failures in one day had this shape: a CI run dispatched in the same command
as the push tested the commit before it, `git add -A` swept an unrelated
untracked file into a commit, and a commit message stated a test-file count
from memory. Concretely: read `git status` before `git add`, name the paths
you mean; after pushing, confirm the remote tip with `git ls-remote` and a
dispatched run's `headSha` before reading its result; and count what you are
about to assert rather than recalling it. Read a command's own output rather
than a filter of it: grepping a batch of runs for `DONE|Error|Traceback`
kept the exception and dropped the line that named its cause, and the
diagnosis then cost a rerun.

### Resolving files by listing

**Resolve a file the way the code under test resolves it, never by listing
and picking.** Sorting a glob and taking the last entry is how a backup file
becomes live. Asking which grid a region uses by globbing
`no_grid_points_20*.csv` and taking `[-1]` returned
`no_grid_points_2024.zonemixed.bak.csv`, because `.zonemixed.bak.csv` sorts
after `.csv`; the loader resolves by exact name and never sees it. Two
retractions came from that one glob: a claim that a backup was in production
use, and a wrong point count for another region. Call the resolver, or
construct the exact name the resolver constructs.

### Detectors trusted in one direction

**Before trusting a detector's negative, run it against a case you know is
positive, and before trusting its positive, show it can return negative.**
Three clean answers in two days were artefacts of a check that never ran: a
glob that resolved a backup file, a column named for degrees holding
kilometres, and an audit reading `cluster_list` at the manifest's top level
when it lives under `correction`, which reported 224 runs as declaring
nothing and therefore flagged none. The last had a known positive available,
nine already-diagnosed contaminated runs, and validating against them first
would have failed in one line. The mirror cost the same day: a reproduction
check reported every value as differing, because it compared full-precision
reruns against originals recorded to five decimal places at a tolerance of
5e-7. Both directions, same sentence.

### Overwriting what was to be checked

**Before a destructive or overwriting operation, list what the path holds,
and copy aside anything the operation is meant to verify against.** Two
losses in one study had that shape. Clearing two contaminated regions deleted
`output/.../DK/` and `UK/` whole, which also held two clean rows' manifests.
Then the reproduction meant to restore them wrote `final_<row>.csv` over the
full-precision originals it was going to be checked against, leaving only a
log rounded to five decimals, so bit-identity became unverifiable. Both are
the same error: acting on a path without checking what else is under it.

### Real-data pins and the input root

**Run the real-data pins when you touch the code they cover, and read
their output.** The `realdata` tests skip where their inputs are absent,
which includes CI, so no pull request check will tell you that a pin moved.
Before committing a change to code the pins reach, run `pytest -m realdata`
and state the counts in the pull request. The covered paths are listed in
`scripts/dev/stamp.py`: every module under `src/pyvwf` except `pinn/`,
`viz/` and `cli/`. *[Note, 2026-09-24: until this date the list was
`harness/`, `metrics.py`, `correction.py` and `data.py`, so a change to
`wind.py` that moved two published rows (AU-NEM and NZ, see
`docs/findings/scorecard.md`) needed no stamp.]* A pin that moves is re-recorded in the same
commit, with the size of the movement in the message. Rerun a row the way
its test runs it: `tests/test_pin_bootstrap_reproduction.py` sets
`PYVWF_INPUT` per row, and rerunning a combined-library row under the
default input root produces a difference that looks exactly like a code
change. That mistake has been made: a UK pin was reported as moving by
0.0014 in MBE when the row had simply been run on the wrong curve library,
and the real movement was 1e-16.

**A stamp survives an edit to comments or docstrings, and nothing else.**
*[Added 2026-10-16, when two of the day's stamps, about eleven minutes each,
were spent on a comment and a `# type: ignore`.]* `scripts/dev/stamp.py`
fingerprints each covered `.py` file by its parsed syntax tree with the
module, class and function docstrings removed. Comments never reach the tree,
and neither do line numbers, so blank lines and moved comments do not count
either. Any edit the interpreter would run differently changes the tree, and
the fingerprint with it. Files that are not `.py`, and a `.py` file that does
not parse, are fingerprinted byte for byte. This is safe for one reason: no
covered code reads its own source or a docstring to compute anything. The one
`__doc__` read under `src/pyvwf` outside `cli/` is the argparse epilog of
`pyvwf.datasets.generate_country_level_training_data`, which changes help
text, not a number. Code that starts reading docstrings, or its own source,
would end that; search for `__doc__`, `inspect.getdoc` and `inspect.getsource`
before relying on the rule. The tree is written out by the script's own
serializer, not `ast.dump`: the hook runs under the first `python3` on
`PATH` and the stamp under the project's environment, and `ast.dump` writes
the same tree differently on 3.12 and 3.13. A test pins the digest of one
snippet on every Python in the test matrix.

**A stamp may run the realdata suite in two or three processes.**
*[Added 2026-10-16.]* `python scripts/dev/stamp.py realdata --workers N`
runs the suite under pytest-xdist with `--dist loadfile`, one test file per
process, and records `workers` in the stamp. Serial stays the default. Each
pin already trains in its own subprocess, but the suite had never been run
in parallel, so it was measured first, from the main checkout with no
`PYVWF_INPUT` set and nothing else heavy running, on a 16 GB machine. The
parallel runs were at `8dd2c2c` and the serial stamp at `83442de`:

| Workers | Wall time | Passed / failed / skipped | Largest process | Swap used, before and after |
|---|---|---|---|---|
| 1 (that day's stamp) | 728 s | 43 / 0 / 0 | not measured | not measured |
| 2 | 404 s | 43 / 0 / 0 | 9.6 GB | 1.8 GB, 2.5 GB |
| 3 | 340 s | 43 / 0 / 0 | 8.8 GB | 2.5 GB, 3.2 GB of 4.0 GB |

The largest process is `/usr/bin/time -l`'s maximum resident set, which is
one process, not the sum. Three workers left under 1 GB of swap free, so two
is the count to use on that machine; the script refuses more than three,
the most measured. The wall time cannot fall much further with `loadfile`:
`tests/test_pin_offset_fits.py` runs in one process and takes about 330 s on
its own. The physics-informed suite has not been measured in parallel and
accepts only one worker.

### Commit messages written before the checks

**Read the checks before the commit command, not after.** Run the
pre-commit hooks (ruff check and format among them), the test files the
change touches and, for `src/pyvwf`, mypy with `pandas-stubs` and
`lint-imports`, and read the output; then write the message. A message that says a check passed
is a claim about output already seen. Amending works only while the commit
is still local.

### The deny rules

The reason: a written rule alone did not stop a tag and a release that were
the maintainer's to make. The deny rules match command text, so they miss
`git -C . push`, a full path to git, `sh -c`, or a script. They are not a
security boundary, and must not be treated as a puzzle.

A denied command means stop. Do not route around it with another command, a
script or a different tool. Hand the command to the maintainer instead.

The match is by text, so a branch whose name contains `main` or `HEAD` is also
blocked. Push such a branch by hand.

### Log of confirmed mistakes, 2026-05-20 to 2026-10-27

The sections above keep the accounts behind the rules. This log records every
mistake an audit of the history confirmed, in one fixed format, so that a
later count does not depend on which mistakes happened to produce a rule.
Terms follow `docs/CONTEXT.md`.

**How it was built.** Four signals were counted separately:

- inverse diffs (reverts);
- fix chains: a commit, made after merge, that changed code, test or script
  lines another commit had introduced in the previous 14 days, found with
  `git blame` rather than message keywords;
- correction notices, withdrawals and re-recorded pins in `docs/` and
  `CHANGELOG.md`;
- the incidents in the sections above.

207 candidates merged to 184 where they shared a fixing commit. The Belgium
and France candidates were then split into their agent-era parts, giving 188
records. 80 were not mistakes: refinement 26, plan change 22, refactor 16,
change of data or source 7, refuted hypothesis 2, upstream change 1, other 6.
The other 108 records are 98 distinct mistakes.

Who made each was read from the Claude Code transcripts, which cover the
whole period except the session behind the commits of 2026-05-20, 12:52 to
13:21. Rows marked "git only" rest on git history alone. Read-only agents
classified each candidate from its diff and transcript, and four published
cases were checked against their commits.

**Periods.** P1 is 2026-05-20 to 09-10 (agents, no written rules), P2 is
09-11 to 09-22 (rules and skills, no hooks), and P3 is 09-23 onwards (rules,
skills and hooks). A mistake's period is the date it was introduced. Six were
introduced before P1, in code dated 2026-02-13 that predates agents, and were
found during the audit window.

**Belgium and France.** The ENTSO-E fetch default `--psr-type onshore` and the
original Belgian files predate agents, and are not counted. The agent-era
parts are:

- C055-A: fleet coverage not checked;
- C055-B: gates trusted in one direction;
- C059-D: a floor calibrated on the defective case.

**The incidents above.** Each is in the log as one of C081 to C095, with two
exceptions:

- The `wind.py` change in "Real-data pins and the input root" is not an
  agent-era mistake. The defect is the maintainer's code from before agents,
  and the narrow stamp list came from the maintainer's own patch.
- The two stamps spent on a comment are a cost, not a mistake. The edits they
  were spent on are C093.

#### Summary

Who made the 98 mistakes: agents 85, agents at the maintainer's request 4,
the maintainer 4 (all in code from before agents), and cannot tell 5.

| Stage reached | Mistakes | Agent itself | Audit | CI | Maintainer | Test | Critic agent |
|---|---|---|---|---|---|---|---|
| After publication | 31 | 10 | 15 | 0 | 3 | 1 | 2 |
| After merge | 13 | 4 | 6 | 3 | 0 | 0 | 0 |
| Before merge | 44 | 34 | 2 | 3 | 2 | 2 | 1 |
| Before commit | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| Total | 98 | 58 | 23 | 6 | 5 | 3 | 3 |

"Audit" is an audit the maintainer commissioned, by an agent, or, in three
cases, by a session in another project.

What the counts show:

- 31 mistakes reached publication. Agents made 25 of them, and 19 of those
  25 were introduced in P1.
- Audits found 15 of the 31 published mistakes: 12 by audit agents, 3 by the
  session in another project. An audit is not a standing guard.
- Agents caught 58 of their own mistakes, 44 of them before merge.
- No hook caught a live mistake.

| Class | Mistakes | Of which published |
|---|---|---|
| `overclaiming` | 18 | 6 |
| `numerical-method` | 12 | 7 |
| `one-direction-check` | 11 | 3 |
| `data-provenance` | 10 | 5 |
| `stale-state` | 8 | 0 |
| `env-divergence` | 7 | 1 |
| `file-resolution` | 5 | 0 |
| `untested-path` | 3 | 2 |
| `incomplete-sweep` | 3 | 0 |
| `miscount` | 2 | 2 |
| `silent-fallback` | 2 | 2 |
| `infeasible-registration` | 2 | 0 |
| `overwrote-evidence` | 2 | 0 |
| `population-mismatch` | 2 | 0 |
| `authority` | 1 | 1 |
| `doc-drift` | 1 | 1 |
| `scope-creep` | 1 | 1 |
| `data-alignment` | 1 | 0 |
| `filtered-output` | 1 | 0 |
| `ignored-instruction` | 1 | 0 |
| `merge-skew` | 1 | 0 |
| `pin-precision` | 1 | 0 |
| `spec-divergence` | 1 | 0 |
| `untestable-gate` | 1 | 0 |
| `wrong-environment` | 1 | 0 |

#### Rates per 100 commits

The rates use only the fix chains and inverse diffs, which were recorded the
same way in every period. Correction notices and incident accounts are not
used for rates: they begin on 2026-09-11, with the written rules, so they
undercount P1 by construction. They inform the classes and the catches above.

| Period | Commits on main | Agent-made | Per 100 commits | All confirmed | Per 100 commits |
|---|---|---|---|---|---|
| P1 | 228 | 18 | 7.9 | 20 | 8.8 |
| P2 | 319 | 10 | 3.1 | 10 | 3.1 |
| P3 | 216 | 3 | 1.4 | 3 | 1.4 |

Five more mistakes found by these two signals were introduced before P1, and
are left out.

The fall coincides with the written rules (P2) and the hooks (P3). These
figures do not show that the rules or hooks caused it, for these reasons:

- **Eligibility.** Fix chains count only fixes made after merge. P1 work was
  committed straight to main, with 7 pull requests in the period, so every
  fix was eligible. P2 work sat on long branches (`manuscript-chapters-45`
  until pull request #20), so most of its fixes came before merge and are not
  counted. P3 used 84 small pull requests.
- **Time to be found.** P3 work was at most five weeks old when counted. P1
  work had been searched by the audits of 2026-09-11 and 2026-09-24.
- **Kind of work.** P1 was mostly new code: the JOSS packaging, the harness,
  new regions, the country-level rework and the physics-informed study. P2
  was studies, pre-registrations and findings, 319 commits in 12 days. P3 was
  refactors, guards, a tech-debt stack and the observation rebuild.
- **Models.** P1 ran on Opus 4.7, Opus 4.8, Fable 5 and Opus 5, P2 on
  Opus 5, and P3 on Opus 5.5.
- **Small counts.** P3 has three.
- **The window.** Fix chains look back 14 days. Mistakes that lived longer,
  which include most published ones, appear only through notices, which
  cannot give a rate.
- **Dating.** Dated by fix instead of introduction, the agent-made counts
  are 15, 5 and 11.

#### Entry format

Each mistake has one row with these fields:

| Field | Meaning |
|---|---|
| ID | The audit's identifier. |
| Date | When the mistake was introduced: the introducing commit's author date, or the session date if it was never committed. |
| Mistake | What was wrong, in one line. |
| Introduced | The commit that introduced it, or "not committed" if it existed only in a session. |
| Fixed | The commit or commits that fixed it. |
| Period | P1, P2, P3, or "before P1". |
| Who | "agent"; "agent, at request" (the maintainer asked for that action); "maintainer"; or "cannot tell". "(git only)" marks a row with no transcript. |
| Class | One of the six classes in the sections above (`stale-state`, `file-resolution`, `one-direction-check`, `overwrote-evidence`, `overclaiming`, `numerical-method`), or a class the audit named. |
| Caught by | Agent itself, audit agent, audit by another session, CI, maintainer, test, or critic agent. |
| Time to catch | From introduction to the fix, or to the moment the transcript shows it was noticed. |
| Guard or eval case today | "machine" if a check now blocks it; "rule" if only a written rule covers it; "none" otherwise. |

Rows are grouped by how far the mistake got before it was caught, the
furthest first. Publication means a findings document, the scorecard, the
CHANGELOG, the README or the paper on main, or a release, tag or Zenodo
version. Within a group, rows are in date order.

#### Caught after publication (31)

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

#### Caught after merge (13)

| ID | Date | Mistake | Introduced | Fixed | Period | Who | Class | Caught by | Time to catch | Guard or eval case today |
|---|---|---|---|---|---|---|---|---|---|---|
| C073 | 2026-02-13 | Country config curve keys were not pinned | `a95e134` | `bf97f4a` | before P1 | maintainer (git only) | `data-provenance` | agent itself | 224 d | machine: `test_curve_resolution.py` |
| C110 | 2026-05-20 | Unconditional shapely import broke CI collection | `4e416b1` | `1fcda52` | P1 | cannot tell (git only) | `env-divergence` | CI | 18 min | machine: CI |
| C111 | 2026-05-28 | Orphan import left behind by the quantile-mapping removal | `5ebc741` | `464a435` | P1 | agent | `incomplete-sweep` | CI | 1 h | machine: pre-commit ruff |
| C113 | 2026-07-14 | Broke the Python 3.10 CI job | `01bd73b` | `f44c503` | P1 | agent | `env-divergence` | CI | 1.2 h | machine: CI |
| C156 | 2026-09-11 | Fleet file picked as the last entry of a sorted glob | `1166ece` | `423abfa` | P2 | agent | `file-resolution` | audit agent | 13 d | rule: Resolve a file the way the code does; eval `sorted-glob` |
| C167 | 2026-09-13 | The same sorted-glob pattern for the training directory | `2f26208` | `a587e4e` | P2 | agent | `file-resolution` | audit agent | 12 d | rule: Resolve a file the way the code does; eval `sorted-glob` |
| C172 | 2026-09-15 | Test names claimed more than the tests assert | `b578bd2` | `c3859a0` | P2 | agent | `overclaiming` | audit agent | 10 d | rule: Test a check both ways |
| C173 | 2026-09-15 | A count in a docstring was wrong | `cc9fc14` | `69cd22f` | P2 | agent | `overclaiming` | audit agent | 10 d | none |
| C154 | 2026-09-18 | A broken import reported as a skip | `9c6cb42` | `458fe85` | P2 | agent | `one-direction-check` | audit agent | 6 d | machine: `test_script_command_lines.py` |
| C171 | 2026-09-19 | A pin skipped on a library version, read as a pass | `086f254` | `3d2ffeb` | P2 | agent, at request | `one-direction-check` | audit agent | 6 d | rule: Test a check both ways (the stamp accepts skips) |
| C092 | 2026-09-20 | UK pin rerun on the wrong input root and reported as moving | not committed | `508addc` | P2 | agent | `wrong-environment` | agent itself | 51 min | rule: Rerun a row the way its test runs it |
| C168 | 2026-09-23 | Stamp hook read the index before a same-command `git add` | `99ff19e` | `47c716b` | P3 | agent | `one-direction-check` | agent itself | 39 h | machine: `test_agent_guards.py` |
| C183 | 2026-10-16 | Two green pull requests combined into a red main | `1a02a2e` | `9e26147` | P3 | agent, at request | `merge-skew` | agent itself | 1 min | machine for agent merges only: deny rules |

#### Caught before merge (44)

| ID | Date | Mistake | Introduced | Fixed | Period | Who | Class | Caught by | Time to catch | Guard or eval case today |
|---|---|---|---|---|---|---|---|---|---|---|
| C116 | 2026-07-16 | Docstrings still called the curves invented after the curve swap | `f50e0a7` | `e76f00c` | P1 | agent | `incomplete-sweep` | agent itself | 20 min | none |
| C120 | 2026-07-16 | Em dashes written after the instruction not to | `f50e0a7` | `363a344` | P1 | agent | `ignored-instruction` | maintainer | 1.2 h | machine: pre-commit `no-em-dash` |
| C001 | 2026-07-22 | Denmark "peaks near k=100" read from one random initialisation | `b6b2303` | `893add8` | P1 | agent | `overclaiming` | maintainer | 1 h | rule: seed spread (`src/pyvwf/AGENTS.md`) |
| C003 | 2026-07-22 | Turkey per-plant data claimed available through a free API | `0f71e30` | `bba7bcf` | P1 | cannot tell (git only) | `overclaiming` | agent itself | 2.9 h | rule: Test a check both ways |
| C005 | 2026-08-12 | Chile "k=1 or k>=8 are safe" rule drawn from test-year RMSE alone | `3dd8498` | `41462e9`, `6effe0e` | P1 | agent | `overclaiming` | agent itself | 12 d | rule: findings-doc |
| C126 | 2026-08-23 | Region-identity features in a model meant to transfer | `c12a208` | `6e47323` | P1 | agent | `numerical-method` | agent itself | 14 min | rule: method-critic, when invoked |
| C130 | 2026-08-23 | Join over duplicated locations | `e53ee09` | `ba14074` | P1 | agent | `data-alignment` | agent itself | 6 min | none |
| C131 | 2026-08-23 | Physics-informed outputs unbounded | `c12a208` | `ce9740e` | P1 | agent | `numerical-method` | test | 42 min | machine: pinn stamp hook |
| C134 | 2026-08-23 | Threshold calibrated over duplicated rows | `59e60ce` | `3f6cec3` | P1 | agent | `numerical-method` | agent itself | 3 min | none |
| C137 | 2026-08-23 | Seeds that varied nothing | `c12a208` | `a53595c` | P1 | agent | `one-direction-check` | agent itself | 1.2 h | machine: pinn test |
| C138 | 2026-08-23 | Output file renamed without updating its reader | `e9ccc64` | `b0bc57d` | P1 | agent | `incomplete-sweep` | agent itself | 3.4 h | none |
| C139 | 2026-08-23 | Summary step assumed the pinn arm after arms became selectable | `f33fb8c` | `8104b00` | P1 | agent | `untested-path` | agent itself | 12 h | none |
| C145 | 2026-08-23 | Branch pushed with no pull request, so CI never ran | `c12a208` | `09d6d25` | P1 | agent | `env-divergence` | agent itself | 8.8 d | machine: CI on pull requests; deny rules |
| C006 | 2026-08-24 | A physical mechanism asserted with the wrong sign | `55f6a50` | `9a811d0` | P1 | agent | `overclaiming` | agent itself | 13 h | rule: findings-doc |
| C019 | 2026-09-11 | Denmark extent resolved from the file, not from the run | `d5fd5a5` | `3eb7a88` | P2 | agent | `file-resolution` | test | 14 h | machine: `ExtrapolationError` |
| C023 | 2026-09-11 | Germany's 13.4% read as the matcher's limit, not the register's | `1166ece` | `577bf7b` | P2 | agent, at request | `overclaiming` | agent itself | 44 h | rule: findings-doc |
| C026 | 2026-09-11 | Denmark curve-match audit run on truncated manufacturer names | `1166ece` | `7c01fe1` | P2 | agent | `data-provenance` | agent itself | 46 h | rule: findings-doc (the truncation is still in the loader) |
| C094 | 2026-09-11 | Committed before reading mypy's output; message claimed five errors | `6e7c9d2` | `631f0bc` | P2 | agent | `overclaiming` | agent itself | 4.5 min | rule: Run the checks, read them, then write the commit message |
| C020 | 2026-09-12 | Wrong fleet count | `2a6da49` | `b3cef71` | P2 | agent | `population-mismatch` | agent itself | 18 min | rule: findings-doc |
| C021 | 2026-09-12 | A gate that could not be tested as registered | `6cb9582` | `2a6da49` | P2 | agent | `untestable-gate` | agent itself | 41 min | rule: findings-doc |
| C024 | 2026-09-13 | Coverage measured over the register, not the fleet the run fits | `958aa31` | `2aaf0f7`, `9aaa9d5` | P2 | agent | `population-mismatch` | agent itself | 26 min | rule: findings-doc |
| C025 | 2026-09-13 | Predicted nearest curve was wrong | `958aa31` | `a4f7baf` | P2 | agent | `overclaiming` | agent itself | 43 min | rule: findings-doc |
| C028 | 2026-09-13 | A late override never reached the simulation | `b00ec1f` | `d05fbb3`, `03769c5` | P2 | agent | `one-direction-check` | agent itself | 2.6 h | machine: `test_curve_library_study.py` |
| C029 | 2026-09-13 | Study run on the wrong pool | `080013a` | `7a05bb0` | P2 | agent | `stale-state` | agent itself | 6 h | rule: findings-doc |
| C037 | 2026-09-13 | Gate registered against a comparator that did not exist | `9d040a5` | `167afad` | P2 | agent | `infeasible-registration` | agent itself | 2.7 d | rule: findings-doc |
| C082 | 2026-09-13 | CI dispatched in the same command as the push | `fdbf21a` | `e1744c3` | P2 | agent | `stale-state` | agent itself | 3.5 min | rule: confirm the remote tip and headSha |
| C083 | 2026-09-13 | `git add -A` swept an unrelated file into a commit | not committed | `ef5d499` | P2 | agent | `stale-state` | agent itself | 16 s | machine: stamp hook |
| C084 | 2026-09-13 | "43 files" stated from memory; there were 52 | `5194a21` | `31b1b30` | P2 | agent | `overclaiming` | agent itself | 12 s | rule: Count what you assert |
| C031 | 2026-09-15 | Overstated reading of the distance mask | `1a9263f` | `be1e4ae` | P2 | agent | `overclaiming` | agent itself | 12 min | rule: findings-doc |
| C032 | 2026-09-15 | Registered grid could not run | `c98213a` | `61fe4c1`, `56fdcf4` | P2 | agent | `infeasible-registration` | agent itself | 6 min | rule: idea-review |
| C033 | 2026-09-15 | Capacity denominator derived as max/0.9 | `56fdcf4` | `b4ecb65` | P2 | agent | `data-provenance` | agent itself | 25 min | machine: fetcher refuses derived capacity |
| C034 | 2026-09-15 | Grid file resolved by globbing | `61fe4c1` | `629bda2` | P2 | agent | `file-resolution` | agent itself | 1 h | rule: Resolve a file the way the code does; eval `sorted-glob` |
| C038 | 2026-09-15 | Grids called uniform that were not | `629bda2` | `bff4467` | P2 | agent | `one-direction-check` | audit agent | 17 h | rule: findings-doc |
| C087 | 2026-09-15 | Column named for degrees held kilometres | `882e616` | `1bc281f` | P2 | agent | `numerical-method` | agent itself | 1.5 min | rule: Test a check both ways |
| C066 | 2026-09-18 | Machine-specific hashes pinned; message claimed CI ran them | `7149875` | `3c77435` | P2 | agent | `env-divergence` | CI | 2.5 h | machine: CI |
| C067 | 2026-09-18 | Pin recorded at the wrong precision | `0783d96` | `d76220a` | P2 | agent | `pin-precision` | agent itself | 3.7 h | machine: this pin only |
| C149 | 2026-09-20 | Share computed wrongly | `f2e48d7` | `1e0e16b` | P2 | agent | `numerical-method` | agent itself | 3.5 h | none |
| C178 | 2026-09-20 | Cache override applied without its ratios | `ea69497` | `bb548e9` | P2 | agent | `stale-state` | agent itself | 11 min | machine: `test_pinn_cache_override.py` |
| C044 | 2026-09-24 | A sentence in a correction notice was wrong | `f0d6fac` | `564dfa3` | P3 | agent | `stale-state` | agent itself | 8 min | rule: Read the state now |
| C158 | 2026-09-24 | KMeans partitions depended on the thread count | `1246e71` | `037ab65` | P3 | agent | `env-divergence` | CI | 18 min | machine: `test_clustering_threads.py`; CI |
| C047 | 2026-09-25 | "3 to 8 iterations" claimed for the joint fit | `dc14687` | `26e0b67`, `cfc2b73` | P3 | agent | `overclaiming` | critic agent | 30 min | machine: `test_country_joint_fit_tolerance.py`; eval `ftol-stall` |
| C093 | 2026-10-16 | A docstring broke the `-W` docs build; a `type: ignore` was unused without entsoe | `82605df` | `05f0bde`, `223004b` | P3 | agent | `env-divergence` | CI | 2 h | machine: CI |
| C054 | 2026-10-22 | France called not degenerate while the scorecard daggered it | `81c9011` | `6466ea5` | P3 | agent | `stale-state` | agent itself | 35 min | rule: Read the state now |
| C056 | 2026-10-22 | 1.11 TWh quoted for a year; it was June to December | `63bb2e2` | `7fc0e16` | P3 | agent | `data-provenance` | audit agent | 29 h | rule: findings-doc |

#### Caught before commit (10)

| ID | Date | Mistake | Introduced | Fixed | Period | Who | Class | Caught by | Time to catch | Guard or eval case today |
|---|---|---|---|---|---|---|---|---|---|---|
| C002 | 2026-07-22 | Draft comparison mixed the open and licensed libraries | not committed | `8a032d6` | P1 | agent | `data-provenance` | agent itself | 9 min | rule: findings-doc |
| C004 | 2026-07-23 | Two bidding-zone pairings stated and retracted | not committed | `dd87232` | P1 | agent | `stale-state` | agent itself | 1.3 h | rule: Read the state now |
| C140 | 2026-08-24 | Wake term diverged from its specification | not committed | `b418713` | P1 | agent | `spec-divergence` | agent itself | 78 min | rule: method-critic, when invoked |
| C081 | 2026-09-12 | File created in the tree while a manifest-writing run was in flight | not committed | `1ff1d54` | P2 | agent | `stale-state` | agent itself | 5 min | rule: create or edit nothing while a run holds the lock (the lock hook misses `cat >`) |
| C085 | 2026-09-13 | A grep filter hid the line naming the cause | not committed | `207657b` | P2 | agent | `filtered-output` | agent itself | 2 min | rule: Read a command's whole output |
| C086 | 2026-09-15 | Backup grid picked by a sorted glob; two claims retracted | not committed | `9b0d26e` | P2 | agent | `file-resolution` | agent itself | 3 min | rule: Resolve a file the way the code does; eval `sorted-glob` |
| C088 | 2026-09-15 | Audit read `cluster_list` at the manifest's top level | not committed | `cc9fc14` | P2 | agent | `one-direction-check` | agent itself | 20 s | rule: Test a check both ways; eval `manifest-key` |
| C090 | 2026-09-15 | Clean rows' manifests deleted with contaminated ones | not committed | `1ae99a3` | P2 | agent | `overwrote-evidence` | agent itself | 40 s | rule: Before overwriting or deleting, list the path |
| C089 | 2026-09-16 | Tolerance 5e-7 against originals recorded to five decimals | not committed | `216be94` | P2 | agent | `one-direction-check` | agent itself | 30 s | rule: Test a check both ways |
| C091 | 2026-09-16 | Reproduction overwrote the originals it was checked against | not committed | `216be94` | P2 | agent | `overwrote-evidence` | agent itself | 10 to 25 min | rule: copy aside what is to be checked |
