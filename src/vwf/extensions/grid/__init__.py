"""PyVWF Grid extension: interpolate point corrections onto a spatial grid.

Public API:
- ``export_pyvwf_grid``: kriging-based export of correction fields onto an
  atlite cutout grid.
- ``mask_from_geojson_fast``: rasterised onshore/offshore mask helper used by
  the ML extension as well.

Drivers in ``scripts/pyvwf_to_grid/`` orchestrate the full grid pipeline
(create unified correction dataframe, compare interpolation methods, evaluate,
and emit production-ready NetCDFs).
"""

from vwf.extensions.grid.atlite_export import (
    export_pyvwf_grid,
    mask_from_geojson_fast,
)

__all__ = [
    "export_pyvwf_grid",
    "mask_from_geojson_fast",
]
