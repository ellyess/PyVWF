"""Build a gridded correction surface from a set of control points.

Ported on 2026-09-13 from ``development:src/vwf/extensions/grid/atlite_export.py``.
This is the piece both registered studies were blocked on
(``docs/findings/method-offshore-pool-prereg.md``,
``docs/findings/method-grid-nl-holdout-prereg.md``), because both need a surface
built from a control-point set that is not the whole pool.

**The set is a parameter, not a variant.** :func:`correction_surface` takes the
control points it is to interpolate, and the whole pool is an ordinary call.
There is no separate holdout path, because a study running through a route the
product does not use measures something the product does not do. Same reasoning
as the one implementation of inverse distance weighting in
:mod:`vwf.extensions.grid.interpolation`, which this calls rather than
reimplementing.

**The domain split is by the declared mode**, the ``cluster_mode`` the run
configuration set, which is what the chapter's tables used. Classifying the
same points against the region shapes disagrees on 30 of 1,729, and that
disagreement is **reported and not acted on**: whether it should be is a
registered question and not a porting decision. See
:func:`domain_disagreement`.

Differences from the original, all deliberate:

- the control points are an argument rather than a CSV path, which is the
  capability above;
- the four interpolations are computed once. The original computed them, then
  recomputed the same four lines verbatim before combining;
- ``workers`` is gone. The original submitted a local closure to a
  ``ProcessPoolExecutor``, which cannot pickle one, so the parallel path raised
  for any ``workers > 1`` and only the sequential path ever ran;
- kriging goes through :func:`vwf.extensions.grid.interpolation.kriging_at`, so
  the export and the cross-validation share one definition. The original had
  its own, defaulting to a spherical variogram in Euclidean degrees while the
  cross-validation used exponential in great-circle;
- the output says the correction applies to **wind speed**. The original's
  attributes said "applied to wind power output", which is wrong and is the
  kind of wrong a user acts on: the whole method depends on correcting speed
  before the power curve.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from vwf.extensions.grid import interpolation as interp
from vwf.geospatial import categorize_points_spatial_join, union_geometries
from vwf.harness.corrections import PLAUSIBLE_SCALAR

#: Values a domain column may carry, and what each means.
DOMAIN_ALIASES = {
    "onshore": "onshore", "land": "onshore", "inland": "onshore",
    "false": "onshore", "0": "onshore",
    "offshore": "offshore", "sea": "offshore", "ocean": "offshore",
    "true": "offshore", "1": "offshore",
}

#: Country-level control points carry this mode and belong with the onshore
#: pool, which is what the chapter's own ``prepare_control_points`` did.
COUNTRY_MODE = "all"

#: What a cell holds where the surface is asked to decline to answer. Since
#: 2026-09-15 it declines nowhere by default: see ``neutral_outside_areas``.
NEUTRAL_SCALAR, NEUTRAL_OFFSET = 1.0, 0.0

#: Beyond this many degrees from any control point, a kriged value has reverted
#: to the pool mean and carries no information about the place it is applied to
#: (``docs/findings/method-distance-mask.md``,
#: ``docs/findings/method-loco-interpolation-prereg.md``). It is a statement
#: about provenance, not about safety: the far cells hold the tamest values on
#: the grid, and every unusable one measured sat inside this horizon.
INFORMATION_HORIZON_DEG = interp.MAX_DISTANCE_DEG

#: A correction whose offset is negative sends every speed below
#: ``-offset / scalar`` to a negative corrected speed, which has no value on the
#: power curve. Above this crossing the pair is refusing ordinary winds rather
#: than correcting them. Same quantity as ``vwf.wind.fit_diagnostics``.
MAX_ZERO_CROSSING_SPEED = 4.0


def normalise_domain(series: pd.Series) -> pd.Series:
    """Map a domain column's many encodings onto onshore and offshore."""
    if series.dtype == bool:
        return series.map({True: "offshore", False: "onshore"})
    if pd.api.types.is_numeric_dtype(series):
        return series.map(lambda x: "offshore" if int(x) == 1 else "onshore")
    text = series.astype(str).str.lower().str.strip()
    return text.map(lambda x: DOMAIN_ALIASES.get(x, x))


def declared_domains(points: pd.DataFrame, *, domain_col: str = "cluster_mode") -> pd.Series:
    """The onshore and offshore split the chapter's tables used.

    Country-level points, whose mode is ``all``, go to onshore. That is the
    chapter's rule and it is reproduced rather than revisited.
    """
    if domain_col not in points.columns:
        raise ValueError(
            f"no {domain_col!r} column; the declared split needs one. Pass "
            "domain_col, or use domain_disagreement to see what the shapes say.")
    mode = points[domain_col].astype(str).str.lower().str.strip()
    return pd.Series(np.where(mode == "offshore", "offshore", "onshore"),
                     index=points.index, name="domain")


def domain_disagreement(points: pd.DataFrame, *, onshore_geojson, offshore_geojson,
                        domain_col: str = "cluster_mode") -> pd.DataFrame:
    """Where the declared mode and the region shapes disagree.

    Reported, never acted on. On the chapter's 1,729 control points the two
    differ on 30: nineteen declared onshore that fall inside offshore shapes,
    one declared offshore that falls onshore, and ten outside both files. Which
    split a surface should use is registered as a study, not decided here.

    Returns:
        The disagreeing points, with ``declared`` and ``by_shapes`` columns.
    """
    declared = declared_domains(points, domain_col=domain_col)
    by_shapes = categorize_points_spatial_join(
        points, onshore_geojson=onshore_geojson, offshore_geojson=offshore_geojson)
    differ = declared.to_numpy() != by_shapes.to_numpy()
    return points.loc[differ].assign(declared=declared[differ].to_numpy(),
                                     by_shapes=by_shapes[differ].to_numpy())


def cutout_lonlat(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """The 1D longitude and latitude axes of an atlite cutout.

    An atlite cutout names its axes ``x`` and ``y`` and also carries 2D ``lat``
    and ``lon`` variables. The original dropped ``lat``, ``lon`` and ``height``
    unconditionally and failed on a cutout without them.
    """
    ds = ds.drop_vars([v for v in ("lat", "lon", "height") if v in ds.variables])
    renames = {old: new for old, new in (("x", "lon"), ("y", "lat")) if old in ds.dims}
    ds = ds.rename(renames)
    missing = [c for c in ("lon", "lat") if c not in ds.coords]
    if missing:
        raise KeyError(f"cutout has no {missing} coordinate; looked for x and y too")
    return ds["lon"].values, ds["lat"].values


def area_mask(lon: np.ndarray, lat: np.ndarray, geojson_path, *, name: str) -> xr.DataArray:
    """Which grid cells fall inside a set of region shapes.

    The shapes are unioned into one geometry before the join, so a cell inside
    two overlapping polygons matches once. That is what keeps this free of the
    duplicate-row defect fixed in :mod:`vwf.geospatial`, and it is worth saying
    because `offshore_shapes.geojson` holds 44 overlapping pairs.
    """
    geometry = union_geometries(geojson_path)
    polygon = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    cells = gpd.GeoDataFrame(
        {"position": np.arange(lon_grid.size)},
        geometry=gpd.points_from_xy(lon_grid.ravel(), lat_grid.ravel()), crs="EPSG:4326")
    inside = np.zeros(lon_grid.size, dtype=bool)
    joined = gpd.sjoin(cells, polygon, predicate="within", how="inner")
    if len(joined):
        inside[joined["position"].to_numpy(dtype=int)] = True
    return xr.DataArray(inside.reshape(lat_grid.shape),
                        coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name=name)


def spatial_bin_average(points: pd.DataFrame, *, ddeg: float,
                        value_cols: tuple[str, ...] = ("scalar", "offset")) -> pd.DataFrame:
    """Average points within coarse lon and lat bins, to thin a dense set.

    Thinning changes the answer and the caller is told how much by the returned
    row count. It is a cost control, not a neutral step.
    """
    kept = points[["lon", "lat", *value_cols]].dropna().copy()
    kept["lon_bin"] = np.floor(kept["lon"] / ddeg).astype(int)
    kept["lat_bin"] = np.floor(kept["lat"] / ddeg).astype(int)
    return (kept.groupby(["lon_bin", "lat_bin"], as_index=False)
            [["lon", "lat", *value_cols]].mean(numeric_only=True))


def _on_grid(values, lon: np.ndarray, lat: np.ndarray) -> xr.DataArray:
    """Wrap a (lat, lon) array as a DataArray on this grid.

    A named helper rather than a dict of keyword arguments splatted into
    ``xr.DataArray``: the splat form defeats the overloads in xarray's stubs,
    so mypy cannot tell ``coords`` from ``dims`` and reports every call five
    times.
    """
    return xr.DataArray(values, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


def control_support(control_points: pd.DataFrame, domain: pd.Series,
                    lon: np.ndarray, lat: np.ndarray, *,
                    horizon: float = INFORMATION_HORIZON_DEG) -> dict[str, xr.DataArray]:
    """How much data each grid cell's correction rests on.

    The chapter shipped a surface with no way to ask this of a cell, and its
    distance mask answered the question by deleting the cells rather than
    labelling them. These are the labels.

    Args:
        control_points: the interpolated set.
        domain: each control point's domain, from :func:`declared_domains`.
        lon: 1D grid longitudes.
        lat: 1D grid latitudes.
        horizon: the radius the support count is taken over, in degrees.

    Returns:
        ``distance_to_control_deg`` and ``distance_to_control_km``, which are
        the same geometry in the two metrics and **are not interchangeable**;
        ``n_control_within_horizon``, which separates a cell between two
        clusters from one trailing off a single cluster; and
        ``nearest_is_onshore``, which is how a cell outside every area of
        interest chooses the domain surface it takes.
    """
    lon_grid, lat_grid = np.meshgrid(np.asarray(lon, float), np.asarray(lat, float))
    targets = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    coords = control_points[["lon", "lat"]].to_numpy(float)
    degrees = interp.degree_distances(targets, coords, "degrees")
    nearest = degrees.argmin(axis=1)
    onshore = (domain.to_numpy() == "onshore")[nearest]
    shape = lon_grid.shape
    x, y = np.asarray(lon, float), np.asarray(lat, float)
    kilometres = interp.degree_distances(targets, coords, "great_circle")
    return {
        "distance_to_control_deg": _on_grid(degrees.min(axis=1).reshape(shape), x, y),
        "distance_to_control_km": _on_grid(kilometres.min(axis=1).reshape(shape), x, y),
        "n_control_within_horizon": _on_grid(
            (degrees <= horizon).sum(axis=1).reshape(shape), x, y),
        "nearest_is_onshore": _on_grid(onshore.reshape(shape), x, y),
    }


def correction_surface(
    control_points: pd.DataFrame,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    onshore_geojson,
    offshore_geojson,
    domain_col: str = "cluster_mode",
    method: str = "kriging",
    variogram_model: str = interp.KRIGING_VARIOGRAM,
    coordinates_type: str = interp.KRIGING_COORDINATES,
    n_closest_onshore: int | None = 50,
    n_closest_offshore: int | None = 80,
    thin_onshore_above: int = 15_000,
    thin_bin_ddeg: float = 0.05,
    neutral_outside_areas: bool = False,
    information_horizon_deg: float = INFORMATION_HORIZON_DEG,
    scalar_bounds: tuple[float, float] = PLAUSIBLE_SCALAR,
    max_zero_crossing_speed: float = MAX_ZERO_CROSSING_SPEED,
) -> xr.Dataset:
    """A gridded correction field from the control points it is given.

    Args:
        control_points: the set to interpolate, with ``lon``, ``lat``,
            ``scalar``, ``offset`` and a domain column. **The whole pool is an
            ordinary argument**; a holdout is the same call with rows removed.
        lon: 1D grid longitudes.
        lat: 1D grid latitudes.
        onshore_geojson: shapes bounding where an onshore correction applies.
        offshore_geojson: the same, offshore.
        domain_col: the declared mode column. See :func:`declared_domains`.
        method: ``kriging`` or ``idw``.
        variogram_model: kriging only.
        coordinates_type: kriging only; ``geographic`` is great-circle.
        n_closest_onshore: moving-window size for the onshore kriging.
        n_closest_offshore: the same, offshore.
        thin_onshore_above: bin-average the onshore points above this count.
        thin_bin_ddeg: bin size for that thinning.
        neutral_outside_areas: leave cells in neither area at scalar 1 and
            offset 0. **Default False, which corrects every cell**, because a
            cell filled with unity is indistinguishable in the file from a
            cell whose correction happens to be the identity. A cell outside
            both areas takes the domain surface of its nearest control point.
        information_horizon_deg: the provenance horizon recorded in the
            attributes and used for the support count.
        scalar_bounds: inclusive range outside which a cell's scalar is a
            degenerate fit by the project's own definition (``CONTEXT.md``).
        max_zero_crossing_speed: a cell whose correction sends speeds below
            this to a negative corrected speed is flagged implausible.

    Returns:
        A dataset of ``scalar`` and ``offset`` on the grid, with the two area
        masks, the per-domain surfaces before combination, the support
        variables from :func:`control_support`, the kriging variances where the
        method supplies them, ``zero_crossing_speed`` and ``plausible``, and
        attributes recording every choice above.

        **``plausible`` is the guard, not the distance.** Measurement says
        geometry does not select the cells holding unusable corrections: they
        sit near the control points, not far from them, and a variance
        threshold picks the same wrong cells
        (``docs/findings/method-distance-mask.md``). A flag on the correction's
        own behaviour picks the right ones.

    Raises:
        ValueError: if either domain has fewer than five points, which is too
            few to fit a variogram and is where the chapter's Denmark offshore
            failure sits.
    """
    required = {"lon", "lat", "scalar", "offset"}
    missing = sorted(required - set(control_points.columns))
    if missing:
        raise ValueError(f"control points are missing {missing}")

    domain = declared_domains(control_points, domain_col=domain_col)
    pools = {"onshore": control_points[domain == "onshore"].copy(),
             "offshore": control_points[domain == "offshore"].copy()}
    for name, pool in pools.items():
        if len(pool) < 5:
            raise ValueError(
                f"the {name} pool has {len(pool)} control points and needs at least 5 to "
                "fit a variogram. Denmark offshore's documented failure is what two "
                "points does; refusing is better than producing a surface from it.")

    thinned = {}
    if len(pools["onshore"]) > thin_onshore_above:
        before = len(pools["onshore"])
        pools["onshore"] = spatial_bin_average(pools["onshore"], ddeg=thin_bin_ddeg)
        thinned = {"onshore_thinned_from": before, "onshore_thin_bin_ddeg": thin_bin_ddeg}

    masks = {"onshore": area_mask(lon, lat, onshore_geojson, name="is_onshore_area"),
             "offshore": area_mask(lon, lat, offshore_geojson, name="is_offshore_area")}
    windows = {"onshore": n_closest_onshore, "offshore": n_closest_offshore}

    raw, fields, variances = {}, {}, {}
    for name, pool in pools.items():
        if method == "kriging":
            scalar, offset, scalar_var, offset_var = interp.to_grid(
                interp.kriging_at, pool, lon, lat, variogram_model=variogram_model,
                coordinates_type=coordinates_type, n_closest_points=windows[name],
                with_variance=True)
            variances[name] = {"scalar_variance": scalar_var,
                               "offset_variance": offset_var}
        elif method == "idw":
            scalar, offset = interp.to_grid(interp.idw_at, pool, lon, lat)
        else:
            raise ValueError(f"unknown method {method!r}; use 'kriging' or 'idw'")
        for label, values in (("scalar", scalar), ("offset", offset)):
            raw[f"{label}_{name}"] = _on_grid(values, lon, lat)
            fields[f"{label}_{name}"] = raw[f"{label}_{name}"].where(masks[name])

    support = control_support(control_points, domain, lon, lat,
                              horizon=information_horizon_deg)
    inside = masks["onshore"] | masks["offshore"]

    combined = {}
    for label, neutral in (("scalar", NEUTRAL_SCALAR), ("offset", NEUTRAL_OFFSET)):
        # Inside an area the cell takes that area's surface. Outside both it
        # takes the surface of whichever domain its nearest control point
        # belongs to, which is a choice: the alternative, filling with unity,
        # is what this replaces.
        outside = xr.where(support["nearest_is_onshore"],
                           raw[f"{label}_onshore"], raw[f"{label}_offshore"])
        joined = xr.where(masks["onshore"], raw[f"{label}_onshore"],
                          xr.where(masks["offshore"], raw[f"{label}_offshore"], outside))
        if neutral_outside_areas:
            joined = joined.where(inside, other=neutral)
        combined[label] = joined.rename(label)

    low, high = scalar_bounds
    crossing = xr.where(
        (combined["offset"] < 0) & (combined["scalar"] > 0),
        -combined["offset"] / combined["scalar"], np.nan).rename("zero_crossing_speed")
    plausible = ((combined["scalar"] >= low) & (combined["scalar"] <= high)
                 & (crossing.isnull() | (crossing <= max_zero_crossing_speed))
                 ).rename("plausible")

    variance_fields = {}
    for name, pair in variances.items():
        for label, values in pair.items():
            variance_fields[f"{label}_{name}"] = _on_grid(values, lon, lat)
    if variances:
        for label in ("scalar_variance", "offset_variance"):
            variance_fields[label] = xr.where(
                support["nearest_is_onshore"], variance_fields[f"{label}_onshore"],
                variance_fields[f"{label}_offshore"]).rename(label)

    out = xr.Dataset({f"is_{name}_area": mask for name, mask in masks.items()}
                     | fields | combined | support | variance_fields
                     | {"zero_crossing_speed": crossing, "plausible": plausible})
    out["scalar"].attrs.update(
        long_name="PyVWF scalar correction",
        description="Multiplicative correction applied to WIND SPEED, before the power "
                    "curve. Corrected speed = scalar * speed + offset.")
    out["offset"].attrs.update(
        long_name="PyVWF offset correction",
        units="m s-1",
        description="Additive correction applied to WIND SPEED, before the power curve.")
    out["distance_to_control_deg"].attrs.update(
        long_name="Euclidean distance to the nearest control point",
        units="degree",
        description="The metric the information horizon is stated in. NOT "
                    "interchangeable with distance_to_control_km.")
    out["distance_to_control_km"].attrs.update(
        long_name="Great-circle distance to the nearest control point",
        units="km",
        description="Provided because degrees of longitude shorten toward the pole, "
                    "so the two metrics order cells differently.")
    out["n_control_within_horizon"].attrs.update(
        long_name="Control points within the information horizon",
        description="Separates a cell between several clusters from one trailing off "
                    "a single cluster at the same distance.")
    out["zero_crossing_speed"].attrs.update(
        long_name="Speed below which the correction returns a negative speed",
        units="m s-1",
        description="-offset / scalar where the offset is negative, else missing. "
                    "Below it the corrected speed has no value on the power curve.")
    out["plausible"].attrs.update(
        long_name="The correction at this cell is usable",
        description=f"False where the scalar leaves [{low}, {high}], the project's "
                    f"definition of a degenerate fit, or the zero crossing exceeds "
                    f"{max_zero_crossing_speed} m/s. This is the guard: distance and "
                    "kriging variance do not select the cells holding unusable "
                    "corrections (docs/findings/method-distance-mask.md).")
    out.attrs.update(
        title="PyVWF gridded bias correction field",
        usage="v_corrected = v_ERA5 * scalar + offset, applied before the power curve",
        method=method,
        variogram_model=variogram_model if method == "kriging" else "",
        coordinates_type=coordinates_type if method == "kriging" else "",
        n_control_points=int(len(control_points)),
        n_control_points_onshore=int(len(pools["onshore"])),
        n_control_points_offshore=int(len(pools["offshore"])),
        domain_split="declared cluster_mode; country-level points to onshore",
        outside_areas=("neutral" if neutral_outside_areas
                       else "corrected from the nearest control point's domain"),
        neutral_scalar=NEUTRAL_SCALAR,
        neutral_offset=NEUTRAL_OFFSET,
        information_horizon_deg=float(information_horizon_deg),
        information_horizon_meaning=(
            "Beyond this distance the interpolated value has reverted to the pool mean "
            "and carries no information about the place it is applied to. This is a "
            "statement about provenance, not about safety: the far cells hold the "
            "tamest values on the grid. See docs/findings/method-distance-mask.md and "
            "the leave-one-country-out result it cites."),
        recommended_filter=(
            "Filter on is_onshore_area or is_offshore_area for cells a fleet could "
            "occupy, and on plausible for cells whose correction is usable. Do not "
            "filter on distance_to_control_deg for safety; use it to say how much "
            "local information a value carries."),
        n_closest_onshore=str(n_closest_onshore),
        n_closest_offshore=str(n_closest_offshore),
        **{k: str(v) for k, v in thinned.items()},
    )
    return out


def export_correction_surface(out_nc, dataset: xr.Dataset, *, for_atlite: bool = True) -> Path:
    """Write a surface, optionally with the axis names atlite expects.

    Args:
        out_nc: destination.
        dataset: from :func:`correction_surface`.
        for_atlite: rename ``lon`` and ``lat`` to ``x`` and ``y`` and order the
            dimensions ``(y, x)``, which is what atlite reads.

    Returns:
        The path written.
    """
    out = dataset
    if for_atlite:
        out = out.rename({"lon": "x", "lat": "y"})
        for name in out.data_vars:
            if set(out[name].dims) == {"y", "x"}:
                out[name] = out[name].transpose("y", "x")
    path = Path(out_nc)
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out.to_netcdf(path)
    return path
