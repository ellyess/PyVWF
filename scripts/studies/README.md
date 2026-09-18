# Study drivers

A study is one research question with its own findings document in
`docs/findings/`, usually a pre-registration, and the scripts that produce
its numbers. This folder holds those scripts, one directory per study.
Reusable tools stay in `scripts/analysis/`; logic more than one study needs
belongs in `src/vwf/`.

## The layout

- **One directory per findings document.** It is named by the document's stem.
  For `docs/findings/method-roughness-treatment.md`, the drivers are in
  `scripts/studies/method-roughness-treatment/`. For a study with only a
  pre-registration so far, drop the `-prereg`: the drivers of
  `method-offshore-pool-prereg.md` are in `scripts/studies/method-offshore-pool/`.
- **A header line in the findings document** names each driver and the commit
  that produced the document's numbers. When the output records no commit, the
  header says so, and names the driver's last commit before the output was
  written. So the link runs both ways, and an audit is mechanical: every file
  here is named by the header of `docs/findings/<stem>.md`, or of
  `<stem>-prereg.md` when that is the only document.
- **Registered constants stay in code.** Seeds, draw counts, gates, cluster
  grids and fold definitions that a pre-registration fixes are module
  constants, not command-line flags. A flag would let a run differ from its
  record without trace. Paths are arguments, with the recorded path as the
  default.
- **Output** goes to `output/<study>_<date>/`, and a driver that runs the
  harness records a manifest there.
- **Tests** load a driver by path with `importlib.util.spec_from_file_location`,
  because the directory names are not importable module names.

## Adding a study

1. Write the pre-registration in `docs/findings/<stem>-prereg.md` and commit it
   before any code that could produce a result.
2. Add the driver to `scripts/studies/<stem>/`, and commit it before it runs.
3. Run it from the repository root, on a clean tree, into `output/<study>_<date>/`.
4. Write `docs/findings/<stem>.md`, with the header line naming the driver and
   the commit the run records.

`AGENTS.md`, `CONTEXT.md` ("gate", "pre-registration") and the `findings-doc`
skill hold the rules for the document itself.

## Deferred

`scripts/pinn/` moves here as one unit, to `scripts/studies/physics-informed/`,
once the turbine-only study has run on its registered paths (issue #12).
`scripts/analysis/ml_transfer_retest.py` and `ml_transfer_expanded.py` move
with it, because the `scripts/pinn/` drivers import the first by that path.

## Path map

Every driver that has moved here, with its old path and the last commit that
touched it there. A command in a dated findings document that names an old path
refers to this file; `git show <commit>:<old path>` recovers it as it stood,
and the document's header names the commit its numbers came from.

| old path | new path | last commit at old path |
|---|---|---|
| `scripts/analysis/curve_library_assign.py` | `scripts/studies/method-curve-library/curve_library_assign.py` | `34d9c3a` |
| `scripts/analysis/curve_library_match.py` | `scripts/studies/method-curve-library/curve_library_match.py` | `b7ed5fd` |
| `scripts/analysis/curve_library_study.py` | `scripts/studies/method-curve-library/curve_library_study.py` | `6dd7af2` |
| `scripts/analysis/curve_library_tables.py` | `scripts/studies/method-curve-library/curve_library_tables.py` | `9aaa9d5` |
| `scripts/analysis/roughness_treatment_study.py` | `scripts/studies/method-roughness-treatment/roughness_treatment_study.py` | `1ff1d54` |
