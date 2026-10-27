# Instructions for agents working in src/pyvwf

The root `AGENTS.md` applies too. These rules cover the package code.

## The affine path

- **The real-data pins reach every module except `pinn/`, `viz/` and `cli/`,
  and CI cannot run them.** After touching one, run
  `python scripts/dev/stamp.py realdata` from the root and state the counts and
  skips in the pull request. Re-record a pin that moves in the same commit,
  with the size of the movement in the message. A hook refuses the commit
  without a matching passing stamp. The exact list is `SUITES` in
  `scripts/dev/stamp.py`.
- **Rerun a row the way its test runs it.** The tests set `PYVWF_INPUT` per
  row; the wrong input root looks exactly like a code change.
- **New affine-path work goes through the harness,** and a new correction model
  or source through its registry.
- **Provenance never aborts a run.** Record any new config key in the manifest,
  with a default that reproduces current results.

## The physics-informed correction, `pinn/`

- **Research code with no stable API, not wired into the harness.** Do not wire
  it in, register it or export it from `pyvwf` unless the task says so.
- **Its tests run only in CI's extras job.** Before committing, run
  `python scripts/dev/stamp.py pinn`; a hook refuses the commit otherwise.
- **Gated results use the seeds their preregistration specifies.** A difference
  smaller than the seed spread is not an effect.

## Every change

- Respect the layers in `.importlinter`. Scripts use only public `pyvwf` names.
- Before committing: `mypy` (with `pandas-stubs`), `lint-imports`, and the test
  files the change touches.
