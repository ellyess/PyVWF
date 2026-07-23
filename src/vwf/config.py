"""Central configuration for PyVWF paths, constants, and settings.

This module provides a single source of truth for:
- Data directory paths
- Country bounding boxes for ERA5 subsetting
- Regional shape files

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
    COUNTRY_DATA = INPUT_ROOT / "country-data"
    TURBINE_DATA = INPUT_ROOT / "observations" / "turbine"
    COUNTRY_LEVEL_DATA = INPUT_ROOT / "observations" / "country"
    REGIONS = INPUT_ROOT / "reference" / "shapes"
    ERA5_DATA = INPUT_ROOT / "era5" / "EU"

    # Static reference files
    POWER_CURVES = INPUT_ROOT / "reference" / "power_curves.csv"
    TURBINE_MODELS = INPUT_ROOT / "reference" / "models.csv"

    # Regional shapes
    COUNTRY_SHAPES = REGIONS / "country_shapes.geojson"
    OFFSHORE_SHAPES = REGIONS / "offshore_shapes.geojson"

    @classmethod
    def reference_file(cls, filename: str) -> Path:
        """Locate a small static reference table (power curves, turbine models).

        Resolution order:

        1. ``INPUT_ROOT/reference/<filename>``: your own copy. Manufacturer-
           specific curve libraries are licensed and not redistributable, so a
           local file always wins.
        2. The open curve library bundled in ``vwf.resources``, so an
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

        packaged = Path(str(resources.files("vwf.resources") / filename))
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

    @classmethod
    def get_country_shapes(cls, offshore: bool = False) -> Path:
        """Get path to country/offshore shape file.

        Args:
            offshore: If True, return offshore shapes path.

        Returns:
            Path to appropriate shapefile.
        """
        return cls.OFFSHORE_SHAPES if offshore else cls.COUNTRY_SHAPES

    @classmethod
    def get_turbine_metadata(cls, country: str) -> Path:
        """Get path to turbine metadata file for a country.

        Args:
            country: Country code (e.g., 'DK', 'UK', 'DE').

        Returns:
            Path to turbine metadata CSV file.
        """
        country_lower = country.lower()
        return cls.TURBINE_DATA / country.upper() / f"{country_lower}_md.csv"

    @classmethod
    def get_country_level_grid_points(cls, country: str) -> Path:
        """Get path to country-level grid points file.

        Args:
            country: Country code (e.g., 'NL', 'FR', 'BE').

        Returns:
            Path to grid points CSV file.
        """
        country_lower = country.lower()
        return cls.COUNTRY_LEVEL_DATA / "grid_points" / country_lower / f"{country_lower}_grid_points.csv"

    @classmethod
    def get_country_level_observations(cls, country: str, train: bool = True) -> Path:
        """Get path to country-level observation file.

        Args:
            country: Country code (e.g., 'NL', 'FR', 'BE').
            train: If True, return training observations path.

        Returns:
            Path to observations CSV file.
        """
        country_lower = country.lower()
        obs_dir = cls.COUNTRY_LEVEL_DATA / "observations" / country_lower
        # This is a template - actual filenames vary by year range
        return obs_dir


class BoundingBoxes:
    """Country bounding boxes for ERA5 subsetting.

    Format: (lon_min, lon_max, lat_min, lat_max)
    """

    DEFINITIONS = {
        "UK": (-11, 3, 49, 61),
        "NL": (2.5, 7.5, 50.5, 53.8),
        "BE": (2.2, 6.6, 49.3, 51.8),
        "DE": (5.0, 16.0, 47.0, 55.2),
        "DK": (7.5, 13.5, 54.0, 58.2),
        "NO": (4.0, 31.5, 57.5, 71.5),
        "FR": (-6.0, 10.5, 41.0, 51.5),
        # Phase 1 Countries
        "ES": (-9.5, 3.5, 36.0, 43.8),    # Spain
        "SE": (11.0, 24.2, 55.3, 69.0),   # Sweden
        "IT": (6.6, 18.5, 36.6, 47.1),    # Italy
        "PT": (-9.5, -6.2, 37.0, 42.2),   # Portugal
        "IE": (-10.5, -5.4, 51.4, 55.4),  # Ireland
        # Multi-region expansion: contiguous US (CONUS). Alaska and Hawaii are
        # excluded: a separate, antimeridian-crossing box would be needed for
        # AK, and the EIA-923 wind fleet is overwhelmingly CONUS.
        "US": (-125.0, -66.0, 24.0, 50.0),
        # Brazil: whole country (crosses the equator). The wind fleet is
        # concentrated in the Nordeste (~ -18..-2 lat) and the far south (Rio
        # Grande do Sul), both inside this box.
        "BR": (-74.0, -34.0, -34.0, 6.0),
    }

    @classmethod
    def get(cls, country: str) -> tuple[float, float, float, float]:
        """Get bounding box for a country.

        Args:
            country: Country code (case-insensitive).

        Returns:
            Tuple of (lon_min, lon_max, lat_min, lat_max).

        Raises:
            KeyError: If country not found.
        """
        return cls.DEFINITIONS[country.upper()]

    @classmethod
    def has_bbox(cls, country: str) -> bool:
        """Check if bounding box exists for country.

        Args:
            country: Country code (case-insensitive).

        Returns:
            True if bounding box is defined.
        """
        return country.upper() in cls.DEFINITIONS
