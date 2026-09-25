## What this changes, and why

<!-- The reasoning and the numbers, not a restatement of the diff. -->

## Checks

- [ ] `pytest -m "not slow and not realdata"` passes locally
- [ ] `ruff check` and `ruff format --check` over `src tests scripts examples`
- [ ] `mypy` passes, from an environment that has `pandas-stubs`

## Real-data pins

Tick the first box or the second, not both.

- [ ] This pull request changes nothing under `pyvwf/harness/`,
      `pyvwf/metrics.py`, `pyvwf/correction.py` or `pyvwf/data.py`, so the
      real-data pins cannot be affected.
- [ ] I ran `pytest -m realdata` locally and the result is below.

<!--
CI never runs these: they read local inputs and skip where those are absent,
so only a local run can tell you a pin moved. Give the counts, name any pin
that skipped for want of its inputs, and for each pin that moved, give the
size of the movement. Resolve each row's input root the way its test does:
a row that runs on input/combined and is rerun under the default input/
answers differently, because the curve library differs.
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
