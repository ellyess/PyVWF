"""PyVWF - Python Virtual Wind Farm model.

PyVWF is a Python package for simulating wind farm generation using
reanalysis data and applying bias corrections.

Core functionality (always available):
- load_region, run_train, run_evaluate, run_transfer: the harness, the one
  path for training and evaluating a region (docs/guides/training.md)
- train_set, val_set: Data preparation functions
- Loaders: Functions for loading turbine data
- ObservationSource: Pluggable adapters supplying observed generation. See
  docs/guides/adding-a-region.md to add a new region.
- Configuration: Path and bounding box configuration

Optional functionality (requires additional dependencies):
- Visualisation: Distributional diagnostics via ``vwf.viz`` (requires matplotlib).
"""

from vwf._version import __version__ as __version__  # re-exported

# ============================================================================
# CORE FUNCTIONALITY (Always available)
# ============================================================================

from vwf.data import train_set, val_set
from vwf.harness.driver import run_evaluate, run_train, run_transfer
from vwf.harness.regions import load_region
from vwf.loaders import (
    load_turbine_metadata,
    load_turbine_observations,
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

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Core
    "load_region",
    "run_train",
    "run_evaluate",
    "run_transfer",
    "train_set",
    "val_set",
    # Loaders
    "load_turbine_metadata",
    "load_turbine_observations",
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
]
