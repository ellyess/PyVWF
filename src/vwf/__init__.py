"""PyVWF - Python Virtual Wind Farm model.

PyVWF is a Python package for simulating wind farm generation using
reanalysis data and applying bias corrections.

Core functionality (always available):
- PyVWF: Main model class for wind simulation
- train_set, val_set: Data preparation functions
- Loaders: Functions for loading turbine and country-level data
- ObservationSource: Pluggable adapters supplying observed generation. See
  docs/ADDING_AN_OBSERVATION_SOURCE.md to add a new region.
- Configuration: Path and bounding box configuration

Optional functionality (requires additional dependencies):
- Geospatial features: Requires geopandas, shapely
- Grid extension (vwf.extensions.grid): Spatial interpolation onto an atlite
  cutout. Requires pykrige (core dep today; will move to ``pyvwf[grid]`` extra).
- ML extension (vwf.extensions.ml): Learned correction models. Optional model
  backends via ``pip install pyvwf[ml]`` (xgboost, lightgbm).
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
# OPTIONAL FUNCTIONALITY
# ============================================================================

# Grid extension (requires: pykrige)
try:
    from vwf.extensions.grid import export_pyvwf_grid
    HAS_GRID = True
    # Backwards-compatible alias for the pre-extensions-split flag name.
    HAS_ATLITE = HAS_GRID
except ImportError:
    HAS_GRID = False
    HAS_ATLITE = False
    export_pyvwf_grid = None

# Geospatial utilities (requires: geopandas, shapely)
try:
    from vwf.geospatial import (
        add_domain_column,
        categorize_points_by_region,
        categorize_points_spatial_join,
        filter_by_domain,
    )
    HAS_GEOSPATIAL = True
except ImportError:
    HAS_GEOSPATIAL = False
    add_domain_column = None
    categorize_points_by_region = None
    categorize_points_spatial_join = None
    filter_by_domain = None

# ML extension (requires: scikit-learn; optional: xgboost, lightgbm)
try:
    from vwf.extensions.ml import (
        train_correction_model,
        predict_correction_grid,
        create_feature_matrix,
        export_ml_correction_grid,
        compare_interpolation_methods,
    )
    HAS_ML = True
except ImportError:
    HAS_ML = False
    train_correction_model = None
    predict_correction_grid = None
    create_feature_matrix = None
    export_ml_correction_grid = None
    compare_interpolation_methods = None

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
    # Optional: Grid extension
    "export_pyvwf_grid",
    "HAS_GRID",
    "HAS_ATLITE",  # alias of HAS_GRID; retained for back-compat
    # Optional: Geospatial
    "add_domain_column",
    "categorize_points_by_region",
    "categorize_points_spatial_join",
    "filter_by_domain",
    "HAS_GEOSPATIAL",
    # Optional: ML
    "train_correction_model",
    "predict_correction_grid",
    "create_feature_matrix",
    "export_ml_correction_grid",
    "compare_interpolation_methods",
    "HAS_ML",
    # Optional: Viz
    "Results",
    "load_results",
    "plot_cf_distribution",
    "plot_qq",
    "HAS_VIZ",
]