# Instructions for coding agents

PyVWF is a Python package for the bias correction of reanalysis wind speeds,
validated against observed generation in many regions. These rules apply to
any agent working in this repository.

## Where to start

- `README.md`: what the package does, and how to install and run it.
- `docs/README.md`: how the documentation is organised, how findings documents
  are named, and the writing rules for procedural documents.
- `CONTRIBUTING.md`: development setup, tests and continuous integration.
- `CONTEXT.md`: the controlled vocabulary. Use its approved terms.
- `PIPELINE.md` describes the older batch path. For new work, use the harness:
  `scripts/analysis/validate_region.py` and `docs/guides/training.md`.

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
  tree. Never run from a temporary or session directory. Do not edit tracked
  files while a run writes its manifest.
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

@CONTEXT.md
