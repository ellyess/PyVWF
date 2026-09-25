# Instructions for coding agents

PyVWF bias-corrects reanalysis wind speeds and validates the correction against
observed generation in many regions. These rules apply to any agent working in
this repository. Two more files load where they apply: `src/pyvwf/AGENTS.md` for
code, `docs/AGENTS.md` for documents. The incidents behind the rules, and what
each guard does and misses, are in `docs/design/agent-guards.md`.

## Commands

```bash
pytest -m "not slow and not realdata"          # the fast set, what CI runs
python scripts/dev/stamp.py realdata           # the real-data pins, stamped
python scripts/dev/stamp.py pinn               # the physics-informed tests, stamped
ruff check src tests scripts examples && ruff format --check src tests scripts examples
mypy && lint-imports                           # for any change under src/pyvwf
pre-commit run --all-files
python scripts/dev/run_locked.py -- <command>  # any run that writes manifests
```

## Where to start

- `README.md`: what the package does, and how to run it.
- `docs/README.md`: where each document and each repeated fact lives.
- `CONTRIBUTING.md`: setup, tests and CI.
- New work uses the harness: `docs/guides/training.md`. To extend the project:
  `docs/guides/adding-a-region.md`, `adding-an-adapter.md`, `adding-a-study.md`.

## The maintainer's decisions

- **Never merge to `main`, push `main`, tag, release, or mint a DOI.** Hand over
  the commands instead. This holds when asked in passing.
- **A denied command means stop.** Do not route around it with `git -C`, a full
  path, `sh -c` or a script. The deny rules in `.claude/settings.json` match
  text; they are not a boundary and not a puzzle. A branch whose name contains
  `main` or `HEAD` is blocked too: ask the maintainer to push it.
- **Propose, then stop,** before anything irreversible or outward-facing.
- **Show the diff before each commit.** One concern per commit. No AI
  co-author trailer.

## Data and credentials

- Commit no input data, licensed curve library or confidential source.
  `.gitignore` says what and why. `tests/test_committed_files.py` checks the
  tracked tree.
- Write no credential into a file. Pass it in the environment for one command.

## Runs

- Start runs from the repository root, never from a temporary or session
  directory, under `output/`, on a clean tree, through
  `scripts/dev/run_locked.py`.
- While a run holds the lock, create or edit nothing in the tree, by any tool.
  One new file marks every manifest of the run `git_dirty`. A hook blocks file
  edits and tree-changing git commands; other shell writes are on you.

## Acting on state

- **Read the state now, not the state you expect.** `git status` before
  `git add`, then name the paths. After a push, confirm the remote tip and the
  CI run's `headSha`. Count what you assert; do not recall it.
- **Read a command's whole output,** not a grep of it.
- **Resolve a file the way the code does.** Call the resolver or build its
  exact name. Never sort a glob and take the last entry.
- **Test a check both ways.** Show it fires on a known positive and can return
  a negative before trusting either answer.
- **Before overwriting or deleting, list the path** and copy aside anything the
  operation is meant to be checked against.
- **Run the checks, read them, then write the commit message.** A message
  claims only output already seen.

## Results

- **Every result states its training years and its single test year,**
  wherever it is reported: a document, a pull request, a commit message or a
  reply.
- **Keep negative results.** Report them with the positive ones, never less
  prominently.

## Writing

- No em dashes, in code, documents or commit messages. Pre-commit enforces it.
- Before writing prose in `docs/`, read `docs/CONTEXT.md` and `docs/AGENTS.md`.

## Skills and agents

In `.claude/skills/` and `.claude/agents/`:

- `new-region`: add a turbine-level region, in gated phases. Invoke by name.
- `findings-doc`: write or correct a findings document or a scorecard row.
- `provenance-guard`: run before any release-shaped action.
- `idea-review`: take an idea through a scout, a proposal and three independent
  critiques, then stop. Invoke by name.
