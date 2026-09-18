# Instructions for coding agents

PyVWF is a Python package for the bias correction of reanalysis wind speeds,
validated against observed generation in many regions. These rules apply to
any agent working in this repository.

## Where to start

- `README.md`: what the package does, and how to install and run it.
- `docs/README.md`: how the documentation is organised, how findings documents
  are named, and the writing rules for procedural documents.
- `CONTRIBUTING.md`: development setup, tests and continuous integration.
- `docs/CONTEXT.md`: the controlled vocabulary. Use its approved terms.
- For new work, use the harness: `scripts/analysis/validate_region.py` and
  `docs/guides/training.md`. The older batch path is the last section of that
  guide.

## Skills

- `new-region`: add a turbine-level region, in gated phases. Invoke it by name.
- `findings-doc`: write or correct a findings document or a scorecard row.
- `provenance-guard`: run before any release-shaped action.

They live in `.claude/skills/`.

## Standing rules

- **Commit no input data.** Data under `input/`, licensed curve libraries and
  confidential sources stay local. The comment blocks in `.gitignore` say what
  and why. `tests/test_committed_files.py` checks the tracked tree.
- **Write no credential into a file.** Pass an API key in the environment for a
  single command.
- **Propose, then stop,** before anything irreversible or outward-facing.
- **Leave releases to the maintainer.** Merging to `main`, pushing, tagging and
  publishing a release are the maintainer's. So is anything that mints a DOI.
  This holds even when asked in passing. Hand over the commands instead.
- **Show the diff before each commit.** Keep one concern per commit. Add no AI
  co-author trailer; the human who commits is the author.
- **Run under `output/`.** Start runs from the repository root, on a clean
  tree. Never run from a temporary or session directory.
- **Create nothing in the tree while a run that writes a manifest is in
  flight.** Not a tracked file,
  not an untracked one, not the analysis script you intend to use on the
  results. A run stamps `git_commit` and `git_dirty` into every manifest it
  writes, and `git_dirty` comes from `git status --porcelain`, which counts an
  untracked file too. One file created mid-run makes every manifest of that
  run say `git_dirty: true`, and the only repair is to delete the runs and
  repeat them. Write it outside the repository, or wait for the run to finish.
  The rule is scoped to its mechanism: work alongside something that records no
  git state, such as a download or a read-only audit, is not covered.
- **Check the state you are about to act on, not the one you expect.** Three
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
- **Resolve a file the way the code under test resolves it, never by listing
  and picking.** Sorting a glob and taking the last entry is how a backup file
  becomes live. Asking which grid a region uses by globbing
  `no_grid_points_20*.csv` and taking `[-1]` returned
  `no_grid_points_2024.zonemixed.bak.csv`, because `.zonemixed.bak.csv` sorts
  after `.csv`; the loader resolves by exact name and never sees it. Two
  retractions came from that one glob: a claim that a backup was in production
  use, and a wrong point count for another region. Call the resolver, or
  construct the exact name the resolver constructs.
- **Before trusting a detector's negative, run it against a case you know is
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
- **Before a destructive or overwriting operation, list what the path holds,
  and copy aside anything the operation is meant to verify against.** Two
  losses in one study had that shape. Clearing two contaminated regions deleted
  `output/.../DK/` and `UK/` whole, which also held two clean rows' manifests.
  Then the reproduction meant to restore them wrote `final_<row>.csv` over the
  full-precision originals it was going to be checked against, leaving only a
  log rounded to five decimals, so bit-identity became unverifiable. Both are
  the same error: acting on a path without checking what else is under it.
- **Read the checks before the commit command, not after.** Run ruff, the test
  files the change touches and, for `src/vwf`, mypy with `pandas-stubs`, and
  read the output; then write the message. A message that says a check passed
  is a claim about output already seen. Amending works only while the commit
  is still local.
- **Keep negative results.** Every result states its training years and its
  single test year.
- **Correct in place, with a date.** A wrong published claim gets a dated
  correction notice. A false claim in dated history gets a dated bracket. The
  `findings-doc` skill has the details.
- **Keep numbers out of the CHANGELOG.** Entries go under `[Unreleased]`, with
  no shares or metrics.
- **Use no em dashes,** in code, documents or commit messages.

## Blocked commands

`.claude/settings.json` denies these commands to Claude Code:

- `git tag`, `gh release`, and `gh pr merge`, which merges to `main` by another
  route;
- a bare `git push`, which pushes the current branch, and that may be `main`;
- a push whose command names `main`, `HEAD`, `--tags`, `--follow-tags`,
  `--mirror`, `--all`, `refs/tags/`, `tag`, or a version-shaped name such as
  `v1.2.0`.

A push that names a feature branch stays allowed.

The reason: a written rule alone did not stop a tag and a release that were
the maintainer's to make. The deny rules match command text, so they miss
`git -C . push`, a full path to git, `sh -c`, or a script. They are not a
security boundary, and must not be treated as a puzzle.

A denied command means stop. Do not route around it with another command, a
script or a different tool. Hand the command to the maintainer instead.

The match is by text, so a branch whose name contains `main` or `HEAD` is also
blocked. Push such a branch by hand.

## Vocabulary

@docs/CONTEXT.md
