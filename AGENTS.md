# Instructions for coding agents

These rules apply to any agent working in PyVWF, and to anyone using one.
`src/pyvwf/AGENTS.md` adds the rules for package code. Why each rule exists,
and what each guard misses, is in `docs/design/agent-guards.md`.

## Commands

```bash
python scripts/dev/ci_local.py                 # CI's checks in CI's 3.12 env
pytest -m "not realdata"                       # the fast set, what CI runs
python scripts/dev/stamp.py realdata           # the real-data pins, stamped
python scripts/dev/run_locked.py -- <command>  # any run that writes manifests
```

The rest are in `CONTRIBUTING.md`.

## The maintainer's decisions

- **Never merge to `main`, push `main`, tag, release, or mint a DOI.** Hand over
  the commands instead. This holds when asked in passing.
- **A denied command means stop.** Do not route around it with `git -C`, a full
  path, `sh -c` or a script.
- **Propose, then stop,** before anything irreversible or outward-facing.
- **Show the diff before each commit.** One concern per commit.

## AI-assisted contributions

- **A human is accountable for every change.** No AI co-author trailer.
- **Mark AI-drafted text** in a pull request, issue or comment with a
  `> [!NOTE]` alert. The human writes the replies.
- **Agents do not act on GitHub alone.** Open, comment on or close an issue or
  pull request only when asked.

## Data and credentials

- Commit no input data, licensed curve library or confidential source.
  `tests/test_committed_files.py` checks the tracked tree.
- Write no credential into a file. Pass it in the environment for one command.

## Runs

- Start runs from the repository root, into `output/`, on a clean tree,
  through `scripts/dev/run_locked.py`.
- While a run holds the lock, create or edit nothing in the tree. A hook blocks
  file edits, tree-changing git commands, and redirects, `tee` and `sed -i`
  into the tree; other shell writes are on you.

## Acting on state

- **Read the state now, not the state you expect.** `git status` before
  `git add`; name the paths. After a push, confirm the remote tip and the CI
  run's `headSha`. Count what you assert.
- **Read a command's whole output,** not a grep of it.
- **Resolve a file the way the code does;** never sort a glob and take the last
  entry.
- **Test a check both ways:** show it fires on a known positive and can return
  a negative.
- **Run the checks, read them, then write the message.** Claim only output
  already seen.

## Results

- **Every result states its training years and its single test year,**
  wherever it is reported.
- **Keep negative results,** as prominently as positive ones.
- **Correct a wrong claim in place, with a date:** a correction notice in a
  findings document, a dated bracket in dated history such as the CHANGELOG.

## Writing

- No em dashes, in code, documents, commit messages or pull requests.
- One home per fact: `docs/README.md` says where each lives. Link to it.
- Use the terms in `docs/CONTEXT.md`.
- CHANGELOG entries go under `[Unreleased]`, with no numbers.

## Code size

For the lines a change adds or edits:

- **No function or class with one caller,** unless public, a registry entry, or
  a split of a function `CONTRIBUTING.md` lists as too long.
- **No constant repeating a default** set elsewhere; read it from there.
- **No check the types or caller already guarantee;** no `try`/`except`
  around code that cannot fail.
- **Comments say why.** Public names keep a Google-style docstring.
- **Edit the existing path;** no `_v2`, no flag keeping an old branch alive.
- **Report an unrelated problem as a follow-up,** not in the same change.

## Skills and agents

- Skills (`.claude/skills/`): `new-region` adds a region in gated phases;
  `findings-doc` writes a findings document or scorecard row;
  `provenance-guard` checks before any release-shaped action.
- Agents (`.claude/agents/`): `method-critic`, `implementation-critic` and
  `red-team` judge a proposal independently, evaluated in
  `scripts/dev/agent_evals/`.
