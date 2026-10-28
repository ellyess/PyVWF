# Contributing to PyVWF

Contributions of all kinds are welcome: bug reports, documentation, new
bias-correction methods, validation case studies and performance work.

## Getting help

- **Questions:** open a [GitHub Discussion] or an issue with the `question`
  label.
- **Bugs:** open a [GitHub Issue] with what you ran and what you expected, the
  full traceback, your OS, Python version and `pip show pyvwf` output, and a
  minimal example where you can.
- **Features:** open an issue with the use case and, ideally, a sketch of the
  API. Open one before a large change.

[GitHub Issue]: https://github.com/ellyess/PyVWF/issues
[GitHub Discussion]: https://github.com/ellyess/PyVWF/discussions

## Development setup

```bash
git clone https://github.com/ellyess/PyVWF.git
cd PyVWF
pip install -e ".[dev,docs]"   # the install CI's test jobs use, plus the docs build
pre-commit install             # ruff, the file checks, nbstripout and the em-dash hooks on every commit
git config blame.ignoreRevsFile .git-blame-ignore-revs   # keeps the one formatting commit out of blame
```

The conda route and what each of the six extras adds are in
[docs/guides/installation.md](docs/guides/installation.md).

## Tests and checks

The test suite uses synthetic data and needs no downloads or API access.

```bash
pytest -m "not realdata"                        # the fast set: what CI runs on a push or pull request
pytest -n 4 --dist loadfile -m "not realdata"   # the same, in four processes
pytest                                          # every test: what CI runs on a manual dispatch
ruff check src tests scripts examples && ruff format --check src tests scripts examples
mypy && lint-imports                            # for any change under src/pyvwf; mypy needs pandas-stubs
python scripts/dev/ci_local.py                  # the lint job's checks but pre-commit, the fast set and the docs build, in CI's 3.12 environment
pre-commit run --all-files                      # every commit hook over the whole tree, as the lint job runs it
pytest --cov=pyvwf -m "not realdata"            # with coverage, which the test jobs gate
sphinx-build -b html docs docs/_build/html -W   # the docs, failing on any warning
```

`scripts/dev/ci_local.py` builds Python 3.12 environments from the CI
constraints under `output/ci-local/`, without the `data` extra, so it sees
pandas 3 and a missing `entsoe` as CI does; `--skip <check>` leaves one out.
Refresh the constraints with `python scripts/dev/lock.py` (needs `uv`), and
check them with `--check`.

### The real-data pins

`realdata` tests read git-ignored data under `input/` or `output/`, such as
the pins that record what the code produces on real inputs. They skip where the
data is absent, which includes CI, so **only a local run can tell you a pin
moved.** A pull request that changes code the pins reach runs them and states
the result in its description. That is `pyvwf/harness/`, `pyvwf/sources/`,
`pyvwf/datasets/`, `pyvwf/loaders/`, `pyvwf/metrics.py`, `pyvwf/correction.py`,
`pyvwf/data.py`, `pyvwf/wind.py`, `pyvwf/curves.py`, `pyvwf/clustering.py`,
`pyvwf/era5.py`, `pyvwf/country_level.py`, `pyvwf/sampling.py`,
`pyvwf/config.py`, `pyvwf/time_utils.py`, `pyvwf/utils.py` and
`pyvwf/provenance.py`:

```bash
python scripts/dev/stamp.py realdata               # the pins, stamped for the commit guard
python scripts/dev/stamp.py realdata --workers 2   # the same in two processes, in about half the time
```

Run `ci_local.py` first, so a commit CI would fail does not cost a stamp. A
stamp survives a later edit to comments or docstrings only.

- **State what you saw.** Give the counts, the size of each movement, and
  which pins skipped for want of inputs. "No pin moved" is a claim about output
  you have seen, not an expectation.
- **Resolve each row's input root the way its test does.** A row on
  `input/combined` rerun under the default `input/` answers differently,
  because the curve library differs, and the difference looks exactly like a
  code change. The pin tests (`tests/test_pin_offset_fits.py`,
  `tests/test_pin_bootstrap_reproduction.py`) set `PYVWF_INPUT` per row; read
  what they pass.
- **Pins are change detectors, not guards.** The repository aims at the most
  accurate results it can produce; [published results](docs/publications.md)
  are legacy, and a recorded output is superseded by a better one. When a
  deliberate improvement moves a pin, re-record it in the same commit, say in
  the CHANGELOG under `[Unreleased]` what moved and why, and give the size of
  each movement in the commit message or the pull request. A pin that moves
  without a deliberate change is a regression until shown otherwise.

### What continuous integration runs

On every pull request and push to `main`, `.github/workflows/ci.yml` runs:

- **lint:** `ruff check` and `ruff format --check`, `mypy` (the package ships
  `py.typed`), `lint-imports` for the layering in `.importlinter`, `deptry`,
  `vulture`, and every pre-commit hook;
- **tests:** the fast set on Python 3.10 to 3.13 with the `dev` extra only,
  installed from `pyproject.toml`, with coverage gated. 3.10 to 3.12 install
  through `constraints/`, so their result depends on the repository rather
  than that day's releases (3.10 resolves pandas 2, 3.11 and 3.12 pandas 3);
  3.13 is unpinned;
- **extras:** the same fast set on 3.12 with `pinn`, `data` and `parallel`
  installed, failing if any test still skips for a missing import;
- **examples:** every file in `examples/`, `run_minimal.py` included. The
  tracked tree must be unchanged afterwards, except the regenerated example
  data, which must equal the committed data to a relative 1e-12 (the last bit
  differs between platforms);
- **docs:** a Sphinx build with `-W`, installed with the `docs` extra;
- **package:** an sdist and wheel, `twine` validation, and a clean install and
  import of the wheel;
- **docker:** the image, its default command and its runtime checks
  ([installation guide](docs/guides/installation.md#docker)).

So the suite and the example must pass with only the `dev` extra: a test that
needs `torch` or `rasterio` skips itself where it is missing.

## Submitting a pull request

1. Fork the repository and branch from `main`.
2. Keep the change to one concern, with Google-style docstrings (`Args:`,
   `Returns:`) on anything public.
3. Add or update tests, preferring synthetic fixtures (`tests/conftest.py`).
4. Run the fast set and the lint locally; for code the pins reach, the stamp
   too.
5. Open the pull request with the template, which asks for the checks and the
   pins.

## AI-assisted contributions

PyVWF is itself developed with a coding agent
([docs/design/ai-native-development.md](docs/design/ai-native-development.md)).
Contributions written with one are welcome on the same terms:

- **A human is accountable for every change.** Do not add an AI co-author
  line.
- **Mark AI-drafted text** in a pull request, issue or comment with a
  `> [!NOTE]` alert. You write the replies.
- **Agents do not act on GitHub alone.** An agent opens, comments on or closes
  an issue or pull request only when you ask it to.

`AGENTS.md` holds the rules an agent working in this repository follows.

## Coding conventions

- Target Python 3.10+, `ruff`-clean, with type hints where they help.
- Keep a heavy or optional dependency behind a guarded import, as the optional
  visualisation import in `pyvwf/__init__.py` is.
- Split a long function on the pinned path only alongside a change that opens
  it, never as a refactor of its own: a split costs a stamp and can move a pin.
  The longest, as of 2026-09-29 (#91): `sampling.cluster_with_geometries`,
  `sampling.create_sampling_points`, `harness.driver.run_evaluate`,
  `data.train_set`, `correction.find_offsets_country_level`,
  `harness.driver.run_transfer`, `harness.regions.load_region`,
  `harness.export.export_correction_field` and `era5.prep_era5`.
- The version lives in `pyvwf.__version__`; `tests/test_packaging.py` keeps it
  in step with `CITATION.cff`.

## Scientific contributions

New bias-correction methods are especially welcome. Include a short validation
(for example on the Denmark case, with RMSE, MAE and MBE on its training years
and single test year) and the diagnostic plots from `pyvwf.viz`. A study goes
through [docs/guides/adding-a-study.md](docs/guides/adding-a-study.md):
pre-registered gates, with a failed gate reported as a failure.

## Code of conduct

By participating you agree to uphold the [Code of Conduct](CODE_OF_CONDUCT.md):
a welcoming, harassment-free environment for everyone, and honest
representation of what the software does. Report unacceptable behaviour to
benmoufok.ellyess@gmail.com.
