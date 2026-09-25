"""Interpolate point corrections onto a spatial grid.

Ported from the `development` branch for the chapters 4 and 5 manuscript. The
interpolators are the chapter's own, and they are the single definition the two
registered studies use: ``docs/findings/method-loco-interpolation-prereg.md``
and ``docs/findings/method-offshore-pool-prereg.md``.

Needs the ``grid`` extra for kriging: ``pip install -e '.[grid]'``.
"""

from pyvwf.extensions.grid.evaluate import (
    corrected_capacity_factors,
    corrections_at,
    country_skill,
    skill,
    turbine_skill,
)
from pyvwf.extensions.grid.geodataframes import (
    correction_geodataframe,
    country_correction_geodataframes,
)
from pyvwf.extensions.grid.interpolation import (
    IDW_POWER,
    KRIGING_COORDINATES,
    KRIGING_VARIOGRAM,
    MAX_DISTANCE_DEG,
    RBF_KERNEL,
    degree_distances,
    distance_to_nearest,
    idw_at,
    kriging_at,
    nearest_at,
    rbf_at,
    to_grid,
)
from pyvwf.extensions.grid.surface import (
    NEUTRAL_OFFSET,
    NEUTRAL_SCALAR,
    area_mask,
    correction_surface,
    cutout_lonlat,
    declared_domains,
    domain_disagreement,
    export_correction_surface,
    spatial_bin_average,
)

__all__ = [
    "IDW_POWER",
    "NEUTRAL_OFFSET",
    "NEUTRAL_SCALAR",
    "KRIGING_COORDINATES",
    "KRIGING_VARIOGRAM",
    "MAX_DISTANCE_DEG",
    "RBF_KERNEL",
    "degree_distances",
    "distance_to_nearest",
    "idw_at",
    "kriging_at",
    "nearest_at",
    "rbf_at",
    "area_mask",
    "corrected_capacity_factors",
    "correction_geodataframe",
    "corrections_at",
    "correction_surface",
    "country_correction_geodataframes",
    "country_skill",
    "cutout_lonlat",
    "declared_domains",
    "domain_disagreement",
    "export_correction_surface",
    "skill",
    "spatial_bin_average",
    "to_grid",
    "turbine_skill",
]
