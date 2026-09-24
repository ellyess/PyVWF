# Instructions for agents working in src/vwf

The root `AGENTS.md` applies too. These rules cover the package code.

## The affine path

- **Touching the code the real-data pins reach moves them, and CI cannot see
  them.** That is `harness/`, `sources/`, `datasets/`, `extensions/`,
  `loaders/`, `metrics.py`, `correction.py`, `data.py`, `wind.py`,
  `curves.py`, `clustering.py`, `config.py`, `time_utils.py`,
  `geospatial.py`, `utils.py` and `provenance.py`: everything except
  `vwf.py`, `pinn/`, `viz/` and `cli/`. They skip where inputs are absent.
  Run `python scripts/dev/stamp.py realdata` from the root, read the output,
  and state the counts and the skips in the pull request. A pin that moves is
  re-recorded in the same commit, with the size of the movement in the
  message. A hook refuses the commit unless the staged code matches a passing
  stamp.
- **Rerun a row the way its test runs it.** `tests/test_pin_bootstrap_reproduction.py`
  sets `PYVWF_INPUT` per row. The wrong input root looks exactly like a code
  change.
- **New affine-path work goes through the harness,** and a new correction model
  or source through its registry, not a branch in calling code.
- **Provenance never aborts a run.** Keep `write_manifest_safe`'s contract, and
  record any new config key in the manifest with a default that reproduces
  current results.

## The physics-informed correction, `pinn/`

- **What `pinn/` is for.** The affine correction needs observed generation in
  the region it corrects, so it cannot serve regions without data. `pinn/`
  tests whether a correction learned from spatial inputs such as terrain can
  produce capacity-factor series in regions it never saw.
- **It is research code with no stable API, and not wired into the harness
  yet.** Its evaluation is cross-region and zero-shot, which the harness's
  train and evaluate verbs do not fit. Do not wire it in, register it as a
  correction model, or export its names from `vwf` unless the task says so in
  those words. The harness and registry rules above do not apply to it.
- **CI never runs its tests.** They need the `pinn` extra (torch, rasterio),
  which CI does not install. Run `python scripts/dev/stamp.py pinn`, which
  refuses to run without the extra rather than stamping a run of skips. A hook
  refuses a commit touching `pinn/` without a matching passing stamp.
- **Drivers live in `scripts/pinn/`,** which is exempt from the public-name
  test. Launch a programme through `scripts/dev/run_locked.py`, the whole
  programme under one lock: `run_overnight.sh` launches each stage fresh, so a
  file written in the evening makes the night's later stages refuse or run
  dirty.
- **Gated results use the seeds their preregistration specifies.** A difference
  smaller than the known seed spread is not an effect.

## Every change

- Respect the layers in `.importlinter`. A module imports only from layers
  below it. Scripts use only public `vwf` names
  (`tests/test_public_analysis_api.py`), `scripts/pinn/` excepted.
- Before committing: `mypy` (with `pandas-stubs`) and `lint-imports`, and the
  test files the change touches.
