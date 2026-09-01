# syntax=docker/dockerfile:1

# PyVWF in a container.
#
# Two reasons this exists. The scientific stack underneath the correction
# (geopandas, pyproj, netCDF4) is awkward to install reproducibly across
# machines, and a real run is driven by large user-supplied inputs that belong
# on a mounted volume rather than baked into an image.
#
#   docker build -t pyvwf .
#   docker run --rm pyvwf                      # the bundled synthetic demo
#   docker run --rm pyvwf pyvwf-train --help   # the CLI
#
# Running against your own data means mounting it; see the README.

ARG PYTHON_VERSION=3.12

# --------------------------------------------------------------- builder ---
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /src

# Only what the build backend reads, so an edit to a doc or an example does
# not invalidate the expensive dependency layer below.
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Optional extras: --build-arg EXTRAS="[pinn]" adds torch for the
# physics-informed correction. Empty by default, because torch is close to a
# gigabyte and nothing in the affine pipeline imports it.
ARG EXTRAS=""

# Built into a virtualenv so the runtime stage copies one self-contained tree
# and leaves the toolchain behind. Dependencies are resolved from
# pyproject.toml rather than a list pinned here: a hand-curated Dockerfile
# dependency list silently papers over missing declared dependencies, which is
# how a missing `bottleneck` once reached main. The cache mount keeps pip's
# downloads between builds without that trade-off, and never enters the image.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install ".${EXTRAS}"

# --------------------------------------------------------------- runtime ---
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="PyVWF" \
      org.opencontainers.image.description="Bias-corrected wind power simulation from ERA5 reanalysis" \
      org.opencontainers.image.source="https://github.com/ellyess/PyVWF" \
      org.opencontainers.image.licenses="BSD-3-Clause"

# MPLBACKEND: there is no display in a container, so matplotlib must not reach
# for one. MPLCONFIGDIR: keeps its cache off $HOME, so the image still works
# when `--user` overrides the built-in account to match a host uid.
# PYVWF_INPUT: where a mounted input tree is expected; vwf.config reads it, and
# nothing under it ships in the image.
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYVWF_INPUT=/data/input

COPY --from=builder /opt/venv /opt/venv

# The wheel carries the package and its bundled curve library, but not the
# example scripts, region configs or analysis drivers, which are what make the
# image runnable rather than merely importable.
WORKDIR /app
COPY examples/ examples/
COPY configs/ configs/
COPY scripts/ scripts/

# Non-root. The mount points are created up front and owned by that user, so a
# volume declared by compose does not arrive root-owned and unwritable.
RUN useradd --create-home --uid 1000 pyvwf \
 && mkdir -p /data/input /data/output \
 && chown -R pyvwf:pyvwf /data /app
USER pyvwf

# The default command is the synthetic end-to-end example: `docker run pyvwf`
# does something meaningful with no data and no arguments. Any other command
# overrides it, since there is no ENTRYPOINT to work around.
CMD ["python", "examples/run_minimal.py"]
