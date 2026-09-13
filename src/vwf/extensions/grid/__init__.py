"""Interpolate point corrections onto a spatial grid.

Ported from the `development` branch for the chapters 4 and 5 manuscript. The
interpolators are the chapter's own, and they are the single definition the two
registered studies use: ``docs/findings/method-loco-interpolation-prereg.md``
and ``docs/findings/method-offshore-pool-prereg.md``.

Needs the ``grid`` extra for kriging: ``pip install -e '.[grid]'``.
"""
from vwf.extensions.grid.interpolation import (
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

__all__ = [
    "IDW_POWER",
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
    "to_grid",
]
