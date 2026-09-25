"""Spatial interpolation of correction factors, as thesis chapter 4 defined it.

Ported on 2026-09-13 from
``development:scripts/pyvwf_to_grid/compare_unified_corrections_to_grid.py``,
where these lived inside a driver script. They are library code here because
two registered studies depend on them and a second implementation would be a
second definition of the same number
(``docs/findings/method-loco-interpolation-prereg.md``,
``docs/findings/method-offshore-pool-prereg.md``).

**Distance is Euclidean in degrees, not great-circle.** That is what the
chapter did, and it is how its published figures reproduce: IDW over the 1,729
control points in five longitude-sorted folds gives a scalar MAE of 0.1607
against a published 0.1610 and an offset MAE of 0.6408 against 0.6410. Degrees
of longitude are shorter than degrees of latitude everywhere except the
equator, by a factor of about two at 60 degrees north, so this weighting is
anisotropic in kilometres and stretches east to west. **It is recorded as a
reproduction constraint and not as a recommendation.** The chapter's own
kriging configuration search found geographic coordinates better for the offset
target, which is the same criticism arrived at from the other side.

One defect from the original is fixed and pinned by a test: its grid-wise IDW
corrected exact control-point matches only in the first batch of 10,000 target
points, because it sliced a per-batch mask with the global offset. The effect
was small, since a distance of zero already dominates the weighting, but it
made the correction silently conditional on grid size.

``pykrige`` is an optional dependency, in the ``grid`` extra, and is imported
where it is used so that this module imports without it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from pyvwf.metrics import weighted_mean

#: The chapter's IDW exponent.
IDW_POWER = 2.0

#: The chapter's RBF kernel.
RBF_KERNEL = "thin_plate_spline"

#: The chapter's variogram and coordinate system for ordinary kriging, adopted
#: in its Table 5 as the balanced choice across both targets.
KRIGING_VARIOGRAM = "exponential"
KRIGING_COORDINATES = "geographic"

#: Beyond this distance in degrees from any control point, the IDW product
#: masks a cell to neutral values. About 500 km at mid latitudes, and it
#: neutralises roughly 35% of the European domain.
MAX_DISTANCE_DEG = 5.0

#: Columns every control-point frame must carry.
REQUIRED_COLUMNS = ("lon", "lat", "scalar", "offset")

VALUE_COLUMNS = ("scalar", "offset")


def _check(control_points: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in control_points.columns]
    if missing:
        raise ValueError(f"control points are missing {missing}; need {list(REQUIRED_COLUMNS)}")
    if control_points.empty:
        raise ValueError("control points are empty, so every target would be undefined")


def _targets(lons, lats) -> np.ndarray:
    return np.column_stack(
        [np.asarray(lons, dtype=float).ravel(), np.asarray(lats, dtype=float).ravel()]
    )


def _grid_targets(grid_lons, grid_lats) -> tuple[np.ndarray, tuple[int, int]]:
    lon_grid, lat_grid = np.meshgrid(
        np.asarray(grid_lons, dtype=float), np.asarray(grid_lats, dtype=float)
    )
    return _targets(lon_grid, lat_grid), lon_grid.shape


#: Mean Earth radius in km, for the great-circle option.
EARTH_RADIUS_KM = 6371.0

#: The chapter's metric. Changing the default would silently restate every
#: figure this module reproduces.
DEFAULT_METRIC = "degrees"


def degree_distances(
    targets: np.ndarray, coords: np.ndarray, metric: str = DEFAULT_METRIC
) -> np.ndarray:
    """Distance between every target and every control point.

    Named rather than inlined so that the one place the metric is decided is
    visible, and so a study that wants great-circle distance says so.

    Args:
        targets: (n, 2) array of lon, lat.
        coords: (m, 2) array of lon, lat.
        metric: ``degrees`` for Euclidean in degrees, which is the chapter's
            and the default, or ``great_circle`` for haversine in km.

    Returns:
        An (n, m) array. **The two metrics are not in the same units**, so a
        distance threshold such as :data:`MAX_DISTANCE_DEG` belongs to
        ``degrees`` and means nothing under ``great_circle``.
    """
    if metric == "degrees":
        diff = targets[:, None, :] - coords[None, :, :]
        return np.sqrt((diff**2).sum(axis=-1))
    if metric == "great_circle":
        lon1, lat1 = np.radians(targets[:, 0]), np.radians(targets[:, 1])
        lon2, lat2 = np.radians(coords[:, 0]), np.radians(coords[:, 1])
        dlat = lat2[None, :] - lat1[:, None]
        dlon = lon2[None, :] - lon1[:, None]
        h = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1)[:, None] * np.cos(lat2)[None, :] * np.sin(dlon / 2) ** 2
        )
        return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    raise ValueError(f"unknown metric {metric!r}; use 'degrees' or 'great_circle'")


def idw_at(
    control_points: pd.DataFrame,
    lons,
    lats,
    *,
    power: float = IDW_POWER,
    k: int | None = None,
    metric: str = DEFAULT_METRIC,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse distance weighting at arbitrary points.

    This is the single definition: the grid-wise and point-wise entry points
    both call it, so a grid cell and a held-out control point at the same
    coordinates get the same number by construction rather than by agreement
    between two implementations.

    Args:
        control_points: frame with ``lon``, ``lat``, ``scalar`` and ``offset``.
        lons: target longitudes.
        lats: target latitudes.
        power: the exponent on distance. The chapter uses 2.
        k: use only the k nearest control points, or all of them when None.
        metric: see :func:`degree_distances`. The chapter's is ``degrees``.

    Returns:
        Interpolated ``scalar`` and ``offset``, one per target.
    """
    _check(control_points)
    coords = control_points[["lon", "lat"]].to_numpy(dtype=float)
    values = {c: control_points[c].to_numpy(dtype=float) for c in VALUE_COLUMNS}
    targets = _targets(lons, lats)
    out = {c: np.empty(len(targets)) for c in VALUE_COLUMNS}

    # Batched so a continental grid does not allocate targets by control points
    # in one array. The batch size changes nothing about the result; the
    # original's exact-match correction was conditional on it, which is the
    # defect this port fixes.
    batch = 10_000
    for start in range(0, len(targets), batch):
        chunk = targets[start : start + batch]
        dist = degree_distances(chunk, coords, metric)
        if k is not None and k < dist.shape[1]:
            nearest = np.argsort(dist, axis=1)[:, :k]
            dist_k = np.take_along_axis(dist, nearest, axis=1)
            take = {c: values[c][nearest] for c in VALUE_COLUMNS}
        else:
            dist_k = dist
            take = {c: np.broadcast_to(values[c], dist.shape) for c in VALUE_COLUMNS}

        with np.errstate(divide="ignore"):
            weights = 1.0 / dist_k**power
        exact = ~np.isfinite(weights).all(axis=1)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        for c in VALUE_COLUMNS:
            # A control point with no value for this column weighs on neither
            # side (weighted_mean), where it used to carry the whole cell to NaN.
            out[c][start : start + len(chunk)] = weighted_mean(take[c], weights, axis=1)

        # A target sitting exactly on a control point takes that point's value
        # rather than a weighted average of an infinity. Applied to every
        # batch, which is the fix.
        if exact.any():
            rows = np.flatnonzero(exact)
            hit = np.argmin(dist_k[rows], axis=1)
            for c in VALUE_COLUMNS:
                out[c][start + rows] = take[c][rows, hit]

    return out["scalar"], out["offset"]


def nearest_at(
    control_points: pd.DataFrame, lons, lats, *, metric: str = DEFAULT_METRIC
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour assignment, equivalent to Voronoi cell membership."""
    _check(control_points)
    coords = control_points[["lon", "lat"]].to_numpy(dtype=float)
    dist = degree_distances(_targets(lons, lats), coords, metric)
    nearest = np.argmin(dist, axis=1)
    return (
        control_points["scalar"].to_numpy(dtype=float)[nearest],
        control_points["offset"].to_numpy(dtype=float)[nearest],
    )


def rbf_at(
    control_points: pd.DataFrame, lons, lats, *, kernel: str = RBF_KERNEL
) -> tuple[np.ndarray, np.ndarray]:
    """Radial basis function interpolation, thin-plate spline by default.

    Every control point influences every target, so this overshoots where the
    control-point geometry is irregular. The chapter reports it worst of the
    four on every metric and it is ported to keep the comparison complete.
    """
    _check(control_points)
    coords = control_points[["lon", "lat"]].to_numpy(dtype=float)
    targets = _targets(lons, lats)
    return tuple(
        RBFInterpolator(coords, control_points[c].to_numpy(dtype=float), kernel=kernel)(targets)
        for c in VALUE_COLUMNS
    )


def kriging_at(
    control_points: pd.DataFrame,
    lons,
    lats,
    *,
    variogram_model: str = KRIGING_VARIOGRAM,
    coordinates_type: str = KRIGING_COORDINATES,
    nlags: int = 6,
    n_closest_points: int | None = None,
    with_variance: bool = False,
):
    """Ordinary kriging at arbitrary points, one fit per target field.

    Args:
        control_points: frame with ``lon``, ``lat``, ``scalar`` and ``offset``.
        lons: target longitudes.
        lats: target latitudes.
        variogram_model: the chapter adopts ``exponential``.
        coordinates_type: the chapter adopts ``geographic``, which is
            great-circle. **Note that this differs from the distance IDW and
            nearest neighbour use here**, which is Euclidean in degrees; the
            difference is the chapter's and is reproduced rather than resolved.
        nlags: lags used to fit the variogram.
        n_closest_points: fit each target from only its nearest control points,
            a moving window. The gridded export uses this to avoid the global
            cost at 23,989 cells, and it changes the answer: a local window is
            a different estimator from global ordinary kriging, not an
            approximation to it.
        with_variance: also return the kriging variance per target, which the
            variance mask needs.

    Raises:
        ImportError: if ``pykrige`` is absent. It is in the ``grid`` extra.
    """
    _check(control_points)
    try:
        from pykrige.ok import OrdinaryKriging
    except ImportError as error:  # pragma: no cover - exercised by the extra
        raise ImportError(
            "kriging needs pykrige, which is in the 'grid' extra: pip install -e '.[grid]'"
        ) from error

    lon = np.asarray(lons, dtype=float).ravel()
    lat = np.asarray(lats, dtype=float).ravel()
    predictions, variances = [], []
    for column in VALUE_COLUMNS:
        model = OrdinaryKriging(
            control_points["lon"].to_numpy(dtype=float),
            control_points["lat"].to_numpy(dtype=float),
            control_points[column].to_numpy(dtype=float),
            variogram_model=variogram_model,
            coordinates_type=coordinates_type,
            nlags=nlags,
            verbose=False,
            enable_plotting=False,
        )
        # A moving window is unsupported by the vectorised backend, so pykrige
        # needs the loop one. Stated rather than caught, since a silent backend
        # change is a silent change of estimator.
        #
        # A window wider than the pool is global kriging, and pykrige does not
        # say so: asked for the 80 nearest of 12 points it indexes past the end
        # and raises IndexError. The chapter's export carried exactly that
        # combination, 80 for an offshore pool of 12, and never ran it. Capping
        # to the pool is not an approximation: the two are the same estimator
        # once the window covers everything, and dropping the window then also
        # restores the vectorised backend.
        window = None if n_closest_points is None else int(n_closest_points)
        if window is not None and window >= len(control_points):
            window = None
        extra = {"n_closest_points": window, "backend": "loop"} if window is not None else {}
        z, var = model.execute("points", lon, lat, **extra)
        predictions.append(np.asarray(z).ravel())
        variances.append(np.asarray(var).ravel())
    if with_variance:
        return tuple(predictions), tuple(variances)
    return tuple(predictions)


def to_grid(method, control_points: pd.DataFrame, grid_lons, grid_lats, **kwargs):
    """Run one of the interpolators over a lon by lat grid.

    Args:
        method: ``nearest_at``, ``idw_at``, ``rbf_at`` or ``kriging_at``.
        control_points: frame with the four required columns.
        grid_lons: 1D grid longitudes.
        grid_lats: 1D grid latitudes.
        **kwargs: forwarded to the interpolator.

    Returns:
        ``scalar`` and ``offset`` as 2D arrays shaped (lat, lon). With
        ``with_variance=True``, which only :func:`kriging_at` accepts, four
        arrays: the two fields then their two kriging variances.
    """
    targets, shape = _grid_targets(grid_lons, grid_lats)
    result = method(control_points, targets[:, 0], targets[:, 1], **kwargs)
    if kwargs.get("with_variance"):
        (scalar, offset), (scalar_var, offset_var) = result
        return (
            np.asarray(scalar).reshape(shape),
            np.asarray(offset).reshape(shape),
            np.asarray(scalar_var).reshape(shape),
            np.asarray(offset_var).reshape(shape),
        )
    scalar, offset = result
    return np.asarray(scalar).reshape(shape), np.asarray(offset).reshape(shape)


def distance_to_nearest(
    control_points: pd.DataFrame, lons, lats, *, metric: str = DEFAULT_METRIC
) -> np.ndarray:
    """Degrees from each target to its nearest control point.

    The IDW product's mask reads this, and both registered studies report it
    per held-out fold, because a prediction far from any control point is a
    different kind of number from one made between two of them.
    """
    _check(control_points)
    coords = control_points[["lon", "lat"]].to_numpy(dtype=float)
    return degree_distances(_targets(lons, lats), coords, metric).min(axis=1)
