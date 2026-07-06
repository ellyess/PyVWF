"""PyVWF - Python Virtual Wind Farm model.

PyVWF is a Python package for simulating wind farm generation using
reanalysis data and applying bias corrections.

Core functionality (always available):
- PyVWF: Main model class for wind simulation
- train_set, val_set: Data preparation functions
- Loaders: Functions for loading turbine and country-level data
- Configuration: Path and bounding box configuration

Optional functionality (requires additional dependencies):
- Visualisation: Distributional diagnostics via ``vwf.viz`` (requires matplotlib).
"""

__version__ = "0.1.0"

# ============================================================================
# CORE FUNCTIONALITY (Always available)
# ============================================================================

from vwf.vwf import PyVWF
from vwf.data import train_set, val_set
from vwf.loaders import (
    load_turbine_metadata,
    load_turbine_observations,
    load_year_specific_grid_points,
)
from vwf.config import PyVWFPaths, BoundingBoxes

# ============================================================================
# OPTIONAL FUNCTIONALITY
# ============================================================================

# Visualisation (requires: matplotlib — already a core dependency)
try:
    from vwf.viz import (
        Results,
        load_results,
        plot_cf_distribution,
        plot_qq,
    )
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    Results = None
    load_results = None
    plot_cf_distribution = None
    plot_qq = None

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Core
    "PyVWF",
    "train_set",
    "val_set",
    # Loaders
    "load_turbine_metadata",
    "load_turbine_observations",
    "load_year_specific_grid_points",
    # Configuration
    "PyVWFPaths",
    "BoundingBoxes",
    # Optional: Viz
    "Results",
    "load_results",
    "plot_cf_distribution",
    "plot_qq",
    "HAS_VIZ",
]