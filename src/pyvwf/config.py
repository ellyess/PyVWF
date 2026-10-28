"""Central configuration for PyVWF paths, constants, and settings.

This module provides a single source of truth for:
- Data directory paths
- Regional shape files

The ERA5 directory and bounding box are not defaults here: each comes from the
region config's ``[era5]`` table.

Paths are resolved relative to :attr:`PyVWFPaths.INPUT_ROOT`, which defaults to
``input/`` in the working directory (the layout of a repository checkout) and
can be pointed anywhere with the ``PYVWF_INPUT`` environment variable, so an
installed copy of PyVWF works outside a checkout::

    export PYVWF_INPUT=/data/pyvwf-inputs
"""

from __future__ import annotations

import os
import warnings
from importlib import resources
from pathlib import Path


#: Suffix of a national series summed over bidding zones (NO, SE).
AGGREGATED_SUFFIX = "_aggregated"


def country_obs_stem(code: str, split: str, years) -> str:
    """The file stem of a country-level observation series, without ``.csv``.

    The writer (``pyvwf.datasets.entsoe_country_obs``) and the reader
    (``pyvwf.sources.entsoe_files``) both build names here, so the two cannot
    drift (#84). A zone-summed national series adds ``AGGREGATED_SUFFIX``.

    Args:
        code: Region or bidding-zone code, any case.
        split: ``"train"`` or ``"test"``.
        years: ``(first, last)`` for a training split, the year for a test split.
    """
    c = code.lower()
    if split == "train":
        first, last = years
        return f"{c}_train_{int(first)}_{int(last)}"
    if split == "test":
        return f"{c}_test_{int(years)}"
    raise ValueError(f"split must be 'train' or 'test', got {split!r}")


class PyVWFPaths:
    """Central configuration for data directories and file paths."""

    # Root directories. PYVWF_INPUT lets an installed copy point at input data
    # living anywhere; the default matches a repository checkout.
    INPUT_ROOT = Path(os.environ.get("PYVWF_INPUT", "input"))
    OUTPUT_ROOT = Path(os.environ.get("PYVWF_OUTPUT", "output"))

    # Data directories. input/ is organised by pipeline stage:
    #   raw/<source>/         upstream downloads (provenance)
    #   observations/{turbine,country}/  processed CF the adapters read
    #   era5/                 reanalysis, per region
    #   reference/            static shared lookups (curves, models, shapes, terrain)
    TURBINE_DATA = INPUT_ROOT / "observations" / "turbine"
    COUNTRY_LEVEL_DATA = INPUT_ROOT / "observations" / "country"
    REGIONS = INPUT_ROOT / "reference" / "shapes"

    # Regional shapes
    COUNTRY_SHAPES = REGIONS / "country_shapes.geojson"
    OFFSHORE_SHAPES = REGIONS / "offshore_shapes.geojson"

    # Coastline linework, used to rebuild islands the country shapes omit
    # (see pyvwf.clustering.repair_region_shape).
    COASTLINES = INPUT_ROOT / "reference" / "terrain" / "coastlines.geojson"

    @classmethod
    def reference_file(cls, filename: str) -> Path:
        """Locate a small static reference table (power curves, turbine models).

        Resolution order:

        1. ``INPUT_ROOT/reference/<filename>``: your own copy. Manufacturer-
           specific curve libraries are licensed and not redistributable, so a
           local file always wins.
        2. The open curve library bundled in ``pyvwf.resources``, so an
           installed PyVWF runs outside a repository checkout.

        Falling back to (2) emits a ``UserWarning``: the bundled curves are
        real, but a fleet is matched to them by specific power rather than
        machine identity, and the user has to know that.

        Args:
            filename: File to locate, e.g. ``"power_curves.csv"``.

        Returns:
            Path to the resolved file.

        Raises:
            FileNotFoundError: If the file is neither in ``INPUT_ROOT`` nor
                bundled with the package.
        """
        local = cls.INPUT_ROOT / "reference" / filename
        if local.is_file():
            return local

        packaged = Path(str(resources.files("pyvwf.resources") / filename))
        if not packaged.is_file():
            raise FileNotFoundError(
                f"{filename} not found at {local} and not bundled with PyVWF. "
                "Set PYVWF_INPUT (or PyVWFPaths.INPUT_ROOT) to the directory "
                "holding your input data."
            )

        warnings.warn(
            f"Using the BUNDLED open-library {filename} shipped with PyVWF "
            f"({packaged}) because none was found at {local}. These are real, "
            "redistributable curves (NREL/turbine-models, VWF-"
            "smoothed), but your fleet is matched to them by specific power, "
            "not by actual machine identity. For production results, supply "
            "your own power-curve library and point PYVWF_INPUT at it.",
            UserWarning,
            stacklevel=2,
        )
        return packaged
