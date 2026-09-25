# Contributing to PyVWF

Thank you for your interest in PyVWF! Contributions of all kinds are welcome:
bug reports, documentation, new bias-correction methods, validation case
studies, and performance improvements.

## Getting help and support

- **Questions / usage help:** open a [GitHub Discussion] or an issue with the
  `question` label.
- **Bug reports:** open a [GitHub Issue]. Please include:
  - what you ran (command or minimal code snippet) and what you expected,
  - the full error traceback,
  - your OS, Python version, and `pip show pyvwf` / `conda list` output,
  - a minimal reproducible example where possible.
- **Feature requests:** open an issue describing the use case and, ideally, a
  sketch of the proposed API.

[GitHub Issue]: https://github.com/ellyess/PyVWF/issues
[GitHub Discussion]: https://github.com/ellyess/PyVWF/discussions

## Development setup

```bash
git clone https://github.com/ellyess/PyVWF.git
cd PyVWF

# Option A: pip, the same install CI uses
pip install -e ".[dev,docs]"

# Option B: conda, the conda-forge stack with the data extra
conda env create -f environment.yaml
conda activate pyvwf
```

Both routes, and what each of the six extras adds, are in
[docs/guides/installation.md](docs/guides/installation.md).

## Running the tests and linter

The test suite uses synthetic data and needs no ERA5 downloads or API access.
Two markers split it:

- `realdata` tests read git-ignored data under `input/` or `output/`, such as
  the pins that record what the code produces on real inputs. They skip where
  the data is absent, which includes CI.
- `slow` tests are pins that take seconds per case.

**Real-data pins are change detectors, not guards.** The repository aims at
the most accurate results it can produce, not at reproducing earlier ones.
[Published results](docs/publications.md) are legacy, and a recorded output
is superseded by a better one rather than preserved. A pin records what the
code produces now, so that a change to it is seen and measured, not so that
the change is prevented. When a deliberate improvement changes a pinned
output:

- re-record the fixture in the same commit as the change;
- say in the CHANGELOG, under `[Unreleased]`, what moved and why;
- give the size of each movement in the commit message or the pull request,
  since the CHANGELOG carries no metrics.

A pinned output that moves without a deliberate change is a regression until
shown otherwise, which is the case the detector exists for.

**Run the real-data set when you touch the code the pins cover.** The pins
that read real inputs carry the `realdata` marker, and they skip where the
inputs are absent, which includes CI. So continuous integration cannot tell
you that a pin moved: only a local run can. A pull request that changes
code the pins reach runs them and states the result in its description.
That is `vwf/harness/`, `vwf/sources/`, `vwf/datasets/`, `vwf/extensions/`,
`vwf/loaders/`, `vwf/metrics.py`, `vwf/correction.py`, `vwf/data.py`,
`vwf/wind.py`, `vwf/curves.py`, `vwf/clustering.py`, `vwf/config.py`,
`vwf/time_utils.py`, `vwf/geospatial.py`, `vwf/utils.py` and
`vwf/provenance.py`:

```bash
pytest -m realdata                     # every pin that reads local inputs
python scripts/dev/stamp.py realdata   # the same, stamped for the commit guard
```

State the counts, and for each pin that moved, the size of the movement.
"No pin moved" is a claim about output you have seen, not an expectation.
Where the inputs for a pin are absent, say which ones skipped, so a reader
knows what was not covered.

**Resolve each row's input root the way its test does.** A row that runs on
`input/combined` and is rerun under the default `input/` produces a different
answer, because the curve library differs, and the difference looks exactly
like a code change. `tests/test_pin_bootstrap_reproduction.py` sets
`PYVWF_INPUT` per row for this reason. Read what the test passes rather than
assuming the default.

Install the commit hooks once, after the dev extra. They run `ruff check`,
`ruff format`, the whitespace and file checks and `nbstripout` on each commit:

```bash
pre-commit install
pre-commit run --all-files   # the same checks over the whole tree
```

The tree was reformatted once with `ruff format`. To keep that commit out of
`git blame`, run `git config blame.ignoreRevsFile .git-blame-ignore-revs`.

```bash
pytest -m "not slow and not realdata"   # the fast set: what CI runs on a push or pull request
pytest                                  # every test: what CI runs on a manual dispatch
pytest --cov=vwf           # with coverage
ruff check src tests scripts examples   # lint, as CI does
mypy                       # type check; needs pandas-stubs, from the dev extra
```

Continuous integration (`.github/workflows/ci.yml`) runs, for every pull request
and every push to `main`:

- `ruff check` and `ruff format --check` over `src tests scripts examples`,
  then `mypy` (the package ships `py.typed`, so type information reaches
  downstream users), `lint-imports` for the layering in `.importlinter`,
  `deptry src` for declared dependencies, `vulture` for dead code, and
  every pre-commit hook over the tracked tree;
- the suite plus `examples/run_minimal.py` on Python 3.10 to 3.13, installed
  from `pyproject.toml` so the declared dependencies are exercised as a fresh
  `pip install` would get them, with coverage gated;
- every file in `examples/`, which then must leave the tracked tree unchanged,
  with the regenerated example data equal to the committed data to a relative
  1e-12 (the last bit differs between platforms);
- a Sphinx build of the docs with `-W`, so a broken docstring or an orphaned
  page fails rather than quietly degrading the site;
- an sdist and wheel build, `twine` metadata validation, then a clean-environment
  install and import of the wheel with no repository on `sys.path`;
- a Docker build, which runs the image's default command on bundled data and
  checks that the example corrected something, that the console script and
  the curve library resolve, and that the image does not run as root
  ([docs/guides/docker.md](docs/guides/docker.md)).

CI installs neither the `data` extra nor the other optional extras, so the
suite and the example must pass without them. Tests that need `torch`,
`pykrige` or `rasterio` skip themselves where those are missing.

To build the docs locally:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html -W
```

The version lives in one place, `vwf.__version__`, from which `pyproject.toml`
reads it dynamically; `tests/test_packaging.py` asserts it is valid semantic
versioning and stays in step with `CITATION.cff`.

## Submitting a pull request

1. Fork the repository and create a feature branch from `main`.
2. Make your change, keeping it focused and well documented (Google-style
   docstrings, with `Args:` and `Returns:` sections, as used throughout
   `vwf/`).
3. **Add or update tests.** New scientific functionality should come with tests;
   prefer synthetic fixtures (see `tests/conftest.py`) so the suite stays fast
   and dependency-light.
4. Ensure `pytest -m "not slow and not realdata"` and `ruff check src tests
   scripts examples` pass locally. That is the set CI runs on a pull request.
5. Open a pull request describing the change and its motivation. Link any
   related issue.

## Coding conventions

- Target Python 3.10+.
- Follow the existing module style: small, documented functions with type hints
  where helpful, Google-style docstrings, and `ruff`-clean code (`E`, `F` rules;
  see `pyproject.toml`).
- Keep new heavy/optional dependencies behind `try/except` imports, mirroring the
  optional visualisation import in `vwf/__init__.py`.

## Scientific contributions

New bias-correction methods are especially welcome. Where possible, include a
short validation (e.g. against the Denmark case study, reporting RMSE/MAE/MBE)
and the diagnostic distribution / QQ plots from `vwf.viz`.

## Code of conduct

By participating in this project you agree to uphold our
[Code of Conduct](CODE_OF_CONDUCT.md): a welcoming, harassment-free environment
for everyone, and honest representation of what the software actually does.
Report unacceptable behaviour to benmoufok.ellyess@gmail.com.
