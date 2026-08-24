"""PyVWF - Python Virtual Wind Farm model.

PyVWF is a Python package for simulating wind farm generation using
reanalysis data and applying bias corrections.

Core functionality (always available):
- PyVWF: Main model class for wind simulation
- train_set, val_set: Data preparation functions
- Loaders: Functions for loading turbine and country-level data
- ObservationSource: Pluggable adapters supplying observed generation. See
  docs/guides/ADDING_AN_OBSERVATION_SOURCE.md to add a new region.
- Configuration: Path and bounding box configuration

Optional functionality (requires additional dependencies):
- Visualisation: Distributional diagnostics via ``vwf.viz`` (requires matplotlib).
"""

__version__ = "0.4.0"

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
from vwf.sources import (
    EuropeanTurbineSource,
    InMemoryCountrySource,
    ObservationSource,
    available_sources,
    get_source,
    register,
    resolve,
)
from vwf.config import PyVWFPaths, BoundingBoxes

# ============================================================================
# VISUALISATION
# ============================================================================

# matplotlib is a core dependency, so vwf.viz always imports. (It was once
# guarded by a try/except that rebound these names to None on ImportError,
# which quietly turned a missing dependency into an AttributeError deep in
# user code instead of an honest ImportError here.)
from vwf.viz import (
    Results,
    load_results,
    plot_cf_distribution,
    plot_correction_factor_map,
    plot_error_vs_clusters,
    plot_factor_joint,
    plot_qq,
    plot_sim_vs_obs,
)

#: Retained for backwards compatibility; always True now that the
#: visualisation layer's dependencies are core.
HAS_VIZ = True

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
    # Observation sources
    "ObservationSource",
    "EuropeanTurbineSource",
    "InMemoryCountrySource",
    "available_sources",
    "get_source",
    "register",
    "resolve",
    # Configuration
    "PyVWFPaths",
    "BoundingBoxes",
    # Visualisation
    "Results",
    "load_results",
    "plot_cf_distribution",
    "plot_correction_factor_map",
    "plot_error_vs_clusters",
    "plot_factor_joint",
    "plot_qq",
    "plot_sim_vs_obs",
    "HAS_VIZ",
]