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

# Option A: conda (pinned, reproducible)
conda env create -f environment.yaml
conda activate pyvwf

# Option B: pip
pip install -e ".[dev]"
```

## Running the tests and linter

The test suite uses synthetic data and needs no ERA5 downloads or API access:

```bash
pytest                 # run all tests
pytest --cov=vwf       # with coverage
ruff check src/vwf tests   # lint
```

Continuous integration (`.github/workflows/ci.yml`) runs the same checks on
Python 3.10–3.12 for every pull request.

## Submitting a pull request

1. Fork the repository and create a feature branch from `main`.
2. Make your change, keeping it focused and well documented (NumPy-style
   docstrings, as used throughout `vwf/`).
3. **Add or update tests.** New scientific functionality should come with tests;
   prefer synthetic fixtures (see `tests/conftest.py`) so the suite stays fast
   and dependency-light.
4. Ensure `pytest` and `ruff check src/vwf tests` pass locally.
5. Open a pull request describing the change and its motivation. Link any
   related issue.

## Coding conventions

- Target Python 3.10+.
- Follow the existing module style: small, documented functions with type hints
  where helpful, NumPy-style docstrings, and `ruff`-clean code (`E`, `F` rules;
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
