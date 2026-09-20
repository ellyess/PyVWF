# Docker

The image carries the scientific stack under the correction: geopandas, pyproj
and netCDF4. That stack is awkward to install reproducibly across machines.
The image is the way to run PyVWF without building that stack yourself.

## Build and run the bundled example

Build the image from the repository root:

```bash
docker build -t pyvwf .
```

Run it with no arguments:

```bash
docker run --rm pyvwf
```

The default command is `examples/run_minimal.py`, the synthetic end-to-end
example. It needs no data and no ERA5 download. It prints the uncorrected and
corrected capacity factors of two clusters.

Any other command overrides the default. The image declares no entrypoint, so
nothing has to be worked around:

```bash
docker run --rm pyvwf pyvwf-train --help
```

## Run against your own data

Inputs and outputs are mounted, not baked in. A real run reads tens of
gigabytes of your own data, which does not belong in an image.

`docker-compose.yml` wires the two mounts. It binds `./input` to `/data/input`
read only, and `./output` to `/data/output`. It sets `PYVWF_INPUT` to
`/data/input`, so the container resolves the input root at the mount point. A
run must never write back to the input root, which is why that mount is read
only.

```bash
docker compose run --rm pyvwf \
    pyvwf-validate train --region configs/regions/nz.toml \
    --out /data/output/validation
```

Follow these rules:

- **Pass `--out` under `/data/output`.** The working directory is `/app`, so a
  default `output/` path lands inside the container. `--rm` then deletes it
  with the container.
- **Pass `--outdir` under `/data/output` too.** The legacy `pyvwf-train`
  command takes `--outdir` rather than `--out`.
- **Mount a curve library under the input root.** The image carries the open
  library only. See [choose the input root](training.md#choose-the-input-root).

## Match the container user to your own

The image runs as a non-root user, `pyvwf`, with uid 1000. A bind mount keeps
host ownership, so the container user must match the host user. Otherwise
writes to `./output` are refused.

uid 1000 is the common case on Linux. Where your uid differs, run as yourself:

```bash
UID=$(id -u) GID=$(id -g) docker compose run --rm pyvwf
```

`docker-compose.yml` reads those two variables and falls back to 1000.

## Build arguments

Two build arguments change what the image contains:

| Argument | Default | Effect |
|---|---|---|
| `EXTRAS` | empty | `"[pinn]"` adds torch, for the physics-informed correction. It adds close to a gigabyte, and the affine correction does not need it. |
| `PYTHON_VERSION` | `3.12` | `3.10` builds against the oldest supported interpreter. |

```bash
docker build -t pyvwf --build-arg EXTRAS="[pinn]" .
docker build -t pyvwf --build-arg PYTHON_VERSION=3.10 .
```

The build resolves dependencies from `pyproject.toml` rather than from a list
pinned in the Dockerfile. A pinned list hides a dependency the package declares
but does not install.

## What continuous integration checks

The `docker` job of `.github/workflows/ci.yml` builds the image from a clean
checkout on every pull request and every push to `main`. It builds without a
warm cache, so a broken dependency layer cannot hide. It then checks five
things:

- the default command runs end to end on bundled data;
- its output says the correction reduced the error, so an image that exits zero
  and prints nothing fails;
- `pyvwf-train` resolves on `PATH`;
- the bundled open library resolves inside the image, with no input root
  mounted;
- the container does not run as root, and its uid is 1000.

The image therefore cannot rot unnoticed. The other continuous integration
jobs are described in
[CONTRIBUTING.md](https://github.com/ellyess/PyVWF/blob/main/CONTRIBUTING.md#running-the-tests-and-linter).
