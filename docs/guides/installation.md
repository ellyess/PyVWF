# Installation

PyVWF needs Python 3.10 or later. It is tested on 3.10, 3.11, 3.12 and 3.13.

## Three routes

**pip.** Install PyVWF into any environment on Python 3.10 or later:

```bash
git clone https://github.com/ellyess/PyVWF.git
cd PyVWF
pip install -e .
```

The base install simulates, bias-corrects, evaluates and plots. It is
everything the correction itself needs.

**conda.** `environment.yaml` installs the conda-forge scientific stack,
`entsoe-py`, `openpyxl` and `pyarrow` from the `data` extra, pytest and ruff,
and an editable install of PyVWF. It takes the lower bounds `pyproject.toml`
declares and pins nothing else but ruff:

```bash
conda env create -f environment.yaml
conda activate pyvwf
```

**Docker.** The image carries the scientific stack under the correction
(geopandas, pyproj, netCDF4), which is awkward to build reproducibly across
machines. See [Docker](#docker) below.

## Extras

Each extra adds one capability. Install the ones a task needs, and no more:

| Extra | Adds | Needed for |
|---|---|---|
| `data` | `entsoe-py`, `openpyxl`, `pyarrow`, `cdsapi` | Downloading observations and ERA5. Nothing in the simulation path imports these. |
| `dev` | `pytest`, `ruff`, `mypy`, `pandas-stubs`, `pre-commit`, `import-linter`, `deptry`, `vulture` | Running the tests and the other checks. See [CONTRIBUTING.md](https://github.com/ellyess/PyVWF/blob/main/CONTRIBUTING.md). |
| `docs` | `sphinx`, `furo`, `myst-parser` | Building the documentation site. |
| `parallel` | `dask[distributed]` | Fanning the offset fit across workers (`PYVWF_OFFSET_WORKERS`). |
| `pinn` | `torch`, `rasterio` | The physics-informed correction (`pyvwf.pinn`). It adds close to a gigabyte. |
| `touchdesigner` | `mapbox-earcut` | Exporting cluster maps to TouchDesigner. |

```bash
pip install -e ".[data]"
pip install -e ".[dev,docs]"
```

Continuous integration's test matrix installs `dev` only, so the test suite and
`examples/run_minimal.py` must pass without the other extras; a test that needs
`torch` or `rasterio` skips itself where the library is absent. A
separate job installs `dev`, `pinn`, `data` and `parallel` and runs the
same fast set with them, and the docs job installs `docs`.

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

## Docker

Build the image from the repository root and run the bundled example, which is
its default command:

```bash
docker build -t pyvwf .
docker run --rm pyvwf
```

Any other command overrides the default; the image declares no entrypoint:

```bash
docker run --rm pyvwf pyvwf-validate --help
```

### Run against your own data

Inputs and outputs are mounted, not baked in. `docker-compose.yml` binds
`./input` to `/data/input` read only, binds `./output` to `/data/output`, and
sets `PYVWF_INPUT` to `/data/input`. A run must never write back to the input
root, which is why that mount is read only.

```bash
docker compose run --rm pyvwf \
    pyvwf-validate train --region configs/regions/nz.toml \
    --out /data/output/validation
```

- **Pass `--out` under `/data/output`.** The working directory is `/app`, so a
  default `output/` path lands inside the container, and `--rm` deletes it.
- **Mount a curve library under the input root.** The image carries the open
  library only; see [choose the input root](training.md#choose-the-input-root).
- **Match the container user to your own.** The image runs as a non-root user,
  `pyvwf`, with uid 1000, and a bind mount keeps host ownership, so writes to
  `./output` are refused when the two differ. Where your uid differs, run `UID=$(id -u) GID=$(id -g) docker compose run --rm pyvwf`;
  `docker-compose.yml` reads those two variables and falls back to 1000.

### Build arguments

| Argument | Default | Effect |
|---|---|---|
| `EXTRAS` | empty | `"[pinn]"` adds torch, for the physics-informed correction. It adds close to a gigabyte, and the affine correction does not need it. |
| `PYTHON_VERSION` | `3.12` | `3.10` builds against the oldest supported interpreter. |
| `CONSTRAINTS` | empty | A constraints file to install against, such as `constraints/py312.txt`. Continuous integration builds with that one. |

```bash
docker build -t pyvwf --build-arg EXTRAS="[pinn]" .
docker build -t pyvwf --build-arg PYTHON_VERSION=3.10 .
```

The build resolves dependencies from `pyproject.toml`, constrained only by
`CONSTRAINTS` when it is set, rather than from a list pinned in the
Dockerfile, which would hide a dependency the package declares but does not
install.

### What continuous integration checks

The `docker` job in `.github/workflows/ci.yml` builds the image on every pull
request and every push to `main`, from a clean checkout, without a warm cache, so a broken dependency layer cannot hide. It
then checks that the default command runs end to end and reports a reduced
error, that `pyvwf-validate` resolves on `PATH`, that the bundled open library
resolves with no input root mounted, and that the container runs as uid 1000,
not root.
