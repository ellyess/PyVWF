"""Central configuration for PyVWF paths, constants, and settings.

This module provides a single source of truth for:
- Data directory paths
- Country bounding boxes for ERA5 subsetting
- Regional shape files
"""
from pathlib import Path


class PyVWFPaths:
    """Central configuration for data directories and file paths."""

    # Root directories
    INPUT_ROOT = Path("input")
    OUTPUT_ROOT = Path("output")

    # Data directories
    COUNTRY_DATA = INPUT_ROOT / "country-data"
    TURBINE_DATA = INPUT_ROOT / "turbine_level_data"
    COUNTRY_LEVEL_DATA = INPUT_ROOT / "country_level_data"
    REGIONS = INPUT_ROOT / "regions"
    ERA5_DATA = INPUT_ROOT / "era5" / "EU"

    # Static reference files
    POWER_CURVES = INPUT_ROOT / "power_curves.csv"
    TURBINE_MODELS = INPUT_ROOT / "models.csv"

    # Regional shapes
    COUNTRY_SHAPES = REGIONS / "country_shapes.geojson"
    OFFSHORE_SHAPES = REGIONS / "offshore_shapes.geojson"

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
