# Installation

PyVWF needs Python 3.10 or later. It is tested on 3.10, 3.11 and 3.12.

## Two routes

**conda.** This route installs the conda-forge scientific stack, the `data`
extra and an editable install of PyVWF. It pins versions that
`pyproject.toml` leaves as ranges:

```bash
git clone https://github.com/ellyess/PyVWF.git
cd PyVWF
conda env create -f environment.yaml
conda activate pyvwf
```

**pip.** This route installs PyVWF into any environment on Python 3.10 or
later:

```bash
pip install -e .
```

The base install simulates, bias-corrects, evaluates and plots. It is
everything the correction itself needs.

A third route is the [Docker image](docker.md), which carries the scientific
stack ready built.

## Extras

Each extra adds one capability. Install the ones a task needs, and no more:

| Extra | Adds | Needed for |
|---|---|---|
| `data` | `entsoe-py`, `openpyxl`, `pyarrow`, `cdsapi` | Downloading observations and ERA5. Nothing in the simulation path imports these. |
| `dev` | `pytest`, `ruff`, `mypy`, `pandas-stubs`, `pre-commit`, `import-linter`, `deptry`, `vulture` | Running the tests and the other checks. See [CONTRIBUTING.md](https://github.com/ellyess/PyVWF/blob/main/CONTRIBUTING.md). |
| `docs` | `sphinx`, `furo`, `myst-parser` | Building the documentation site. |
| `grid` | `pykrige`, `rasterio` | The gridded correction surfaces (`vwf.extensions.grid`). |
| `pinn` | `torch` | The physics-informed correction (`vwf.pinn`). It adds close to a gigabyte. |
| `touchdesigner` | `mapbox-earcut` | Exporting cluster maps to TouchDesigner. |

Install one:

```bash
pip install -e ".[data]"
```

Install several:

```bash
pip install -e ".[dev,docs]"
```

Continuous integration installs `dev` and `docs` only. The test suite and
`examples/run_minimal.py` must therefore pass without the other four. A test
that needs `torch`, `pykrige` or `rasterio` skips itself where the library is
absent.

## What PyVWF ships, and what it does not

PyVWF bundles the open library of power curves: 69 real machines and 7
composites, derived from the NREL turbine-models archive under BSD-3-Clause,
and smoothed as VWF smooths them. So an installed PyVWF runs on real curve
physics with no further download. It matches a unit to a curve by specific
power.

PyVWF ships no turbine metadata and no observed generation. Such datasets are
usually proprietary. See [data sources](data-sources.md) for the input layout
and for each source's terms.

PyVWF reads every input from one directory, the input root. See
[choose the input root](training.md#choose-the-input-root).

## Check the installation

Run the synthetic example. It needs no data and takes a few seconds:

```bash
python examples/run_minimal.py
```

It prints the uncorrected and the corrected capacity factors of two clusters,
and the error it removed.
