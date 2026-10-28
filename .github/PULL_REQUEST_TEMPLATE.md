## What this changes, and why

<!-- The reasoning and the numbers, not a restatement of the diff. -->

## Checks

- [ ] `pytest -m "not realdata"` passes locally
- [ ] `ruff check` and `ruff format --check` over `src tests scripts examples`
- [ ] For a change under `src/pyvwf`: `mypy` (with `pandas-stubs`) and `lint-imports`

## Real-data pins

Tick one.

- [ ] This changes none of the code the real-data pins reach (the paths are
      listed in `CONTRIBUTING.md`).
- [ ] I ran `python scripts/dev/stamp.py realdata` and the result is below.

<!--
CI cannot run these. Give the counts, name any pin that skipped for want of
its inputs, and give the size of each movement. Resolve each row's input root
the way its test does.
-->

```
passed / skipped / failed:
moved:
skipped for want of inputs:
```

## Pinned outputs

- [ ] No pinned output moved.
- [ ] A pinned output moved. It is re-recorded in the same commit, the
      CHANGELOG says what moved and why under `[Unreleased]`, and the size of
      each movement is above.

## AI assistance

- [ ] No coding agent was used.
- [ ] A coding agent was used. I am accountable for the change, AI-drafted
      text here is marked with a `> [!NOTE]` alert, I write the replies, and
      there is no AI co-author line.
