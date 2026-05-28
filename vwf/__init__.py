"""PyVWF - Python Virtual Wind Farm model.

PyVWF is a Python package for simulating wind farm generation using
reanalysis data and applying bias corrections.

Core functionality (always available):
- PyVWF: Main model class for wind simulation
- train_set, val_set: Data preparation functions
- Loaders: Functions for loading turbine and country-level data
- Configuration: Path and bounding box configuration

Optional functionality (requires additional dependencies):
- Geospatial features: Requires geopandas, shapely
- Atlite export: Requires atlite
- ML corrections: Requires scikit-learn, xgboost
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

# Atlite export (requires: atlite)
try:
    from vwf.atlite_export import export_pyvwf_grid
    HAS_ATLITE = True
except ImportError:
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

# ML corrections (requires: scikit-learn, xgboost)
try:
    from vwf.ml_correction import (
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
    # Configuration
    "PyVWFPaths",
    "BoundingBoxes",
    # Optional: Atlite
    "export_pyvwf_grid",
    "HAS_ATLITE",
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