# Instructions for agents working in docs

The root `AGENTS.md` applies too. These rules cover documents.

@CONTEXT.md

- **Vocabulary.** `CONTEXT.md` lists the approved terms. Procedural documents
  (`guides/`, `runbooks/`) use only those. Findings and design documents define
  a term once, by citing it, then use it consistently.
- **One home per fact.** `README.md` in this folder says where each repeated
  fact lives. State it there and link to it elsewhere.
- **Findings go through the `findings-doc` skill.** Every result states its
  training years and its single test year. Negative results stay in the body.
- **Correct in place, with a date.** A wrong published claim gets a dated
  correction notice; a false claim in dated history gets a dated bracket.
- **No numbers in the CHANGELOG.** Entries go under `[Unreleased]`.
- **A new published page needs a toctree entry in `index.md`.** The docs build
  runs with `-W`, so an orphan page fails CI. Pages that are not for the site
  go in `exclude_patterns` in `conf.py`.
