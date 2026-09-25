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
| `.claude/hooks/guard_run_lock.py` | While `scripts/dev/run_locked.py` holds `output/.run-lock`: no Write, Edit, MultiEdit or NotebookEdit on a path git does not ignore, and no tree-changing git command | A run not started through the wrapper; a shell write that is not a git command; a lock whose process id was reused. The liveness check assumes a POSIX system. |
| `.claude/hooks/guard_local_suites.py` | For each local-only suite in `scripts/dev/stamp.py`, `realdata` (the real-data pins) and `pinn` (the physics-informed tests, which need an extra only CI's extras job installs, so the stamp is the check before a push): a commit that stages the code the suite covers needs a stamp whose fingerprint matches the staged content and whose run passed. `git commit -a` is refused, and so is anything else that stages covered code after the hook has read the index: `git commit <paths>`, and a `git add`, `stage`, `rm` or `mv` in the same command as the commit, whenever its paths can reach covered code (a glob, `.`, `-A`, `-u` or a directory above covered code all can). | A commit made outside Claude Code or by a script it cannot see into. The `-a` refusal reads only a `git commit` it can parse in command position, skipping option values and heredoc bodies, so a message quoting `git commit -a` does not trip it; a commit hidden in `sh -c` or an alias still gets the staged-code check, through a plain text match, but not the `-a` refusal. A `realdata` run where every pin skipped still passes, which is why the skips are read and stated; the `pinn` suite instead refuses to run without its extra. Changes to `scripts/pinn/` are not covered, because no test exercises the drivers. `tests/test_agent_guards.py` checks that every `test_pinn_*.py` file is in the `pinn` suite and that the realdata paths agree with `CONTRIBUTING.md` and `src/pyvwf/AGENTS.md`. Relative paths in a same-command `git add` are read against the session's working directory, so an `add` after a `cd` into a subdirectory is judged against the wrong base; `-C` on the `add` is not followed either. |
| `no-em-dash` hooks, `.pre-commit-config.yaml` | No em dash in a committed file or a commit message | Nothing, once `pre-commit install` has run for both hook types. |

Both hooks fail open: an unexpected error exits 1, which lets the tool call
through and shows the error to the user, so a broken guard cannot block all
work. A guard that errors is therefore visible, not silent, and is fixed rather
than worked around.

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
