# Adding a study

A study is one research question. It has its own findings document, usually a
pre-registration, its drivers and its run directories. `CONTEXT.md` defines
the terms. This guide says where each piece goes, and in what order.

## Where each piece goes

| Piece | Path |
|---|---|
| Pre-registration | `docs/findings/<stem>-prereg.md` |
| Findings document | `docs/findings/<stem>.md` |
| Drivers | `scripts/studies/<stem>/`, named by the findings document's stem |
| Region configs the study alone uses | `configs/regions/study/` |
| Run directories | `output/<name>_<date>/`, one per study run |
| Test of each driver's command line | `tests/test_script_command_lines.py` |

A study that has only a pre-registration so far drops the `-prereg` from its
directory name. The drivers of `method-offshore-pool-prereg.md` are in
`scripts/studies/method-offshore-pool/`.

`<stem>` follows the findings naming in [`docs/README.md`](../README.md), for
example `method-roughness-treatment`.

## The order

1. Write the pre-registration.
   - Fix every gate and prediction in it.
   - Name each comparator by its files, not by a label.
   - State the training years and the single test year.
2. Commit the pre-registration before any code that could produce a result.
3. Write the driver in `scripts/studies/<stem>/`.
4. Add the driver's recorded command line to `RECORDED` in
   `tests/test_script_command_lines.py`.
5. Commit the driver and its test before the driver runs.
6. Start the run from the repository root, on a clean tree.
7. Write its output to `output/<name>_<date>/`.
8. Read the gates against the output before you quote a number anywhere else.
9. Write the findings document.

Step 2 matters. A gate counts only if a commit older than the results fixes it.

<!-- keeps: the skill name and the file it applies to -->
The `findings-doc` skill, in `.claude/skills/`, holds the rules for the
findings document. It covers the metrics table, the curve library, fit
quality and correction notices.

## The driver

A driver is a thin entry point over `vwf`.

- **Put reusable logic in `src/vwf/`.** Logic that two studies need belongs
  there. Logic that only this study needs can stay in the driver.
- **Parse the command line in a `cli(argv)` function.** Build the parser with
  `vwf.cli.common.make_parser(__doc__)`. Have `cli` call `main` with the
  parsed values.
- **Record the command line in the module docstring.** Write it as a usage
  line, run from the repository root.
- **Make each input or output path a flag.** Its default is the path the
  recorded run used, written literally. A re-run can then read other inputs,
  or write beside its record rather than over it.
- **Keep registered constants in code.** Seeds, draw counts, gates, cluster
  grids and fold definitions stay module constants. A flag would let a run
  differ from its record without trace.
- **Use public `vwf` names only.** `tests/test_public_analysis_api.py` fails
  on a script that imports a private name.

The directory names under `scripts/studies/` are not valid module names. A
test therefore loads a driver by path, with
`importlib.util.spec_from_file_location`.

## The run

A driver that runs the harness writes a manifest into each run directory. The
manifest records the git commit and whether the tree was dirty.

- **Create nothing in the tree while such a run is in flight.** One new file,
  tracked or not, marks every manifest of the run as dirty.
- **Put whatever the study varies in the run directory path.** A run directory
  is keyed on the region code and the run name only. Two configurations that
  share both collide without an error.
- **Never run from a session or temporary directory.** A run that matters
  writes under `output/`.

## The findings document

The findings document opens with a header line. It names each driver and the
commit that produced the document's numbers. When the output records no
commit, the header says so. It then names the driver's last commit before the
output was written.

The link then runs both ways. Every driver is named by the header of its
findings document, so an audit of the two directories is mechanical.

A driver that moves keeps a row in the path map of
[`scripts/studies/README.md`](../../scripts/studies/README.md). A dated
command that names the old path then still resolves.
