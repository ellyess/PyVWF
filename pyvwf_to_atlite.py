from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from pykrige.ok import OrdinaryKriging

import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep


def _coerce_finite_lonlat(
    df: pd.DataFrame,
    *,
    lon_col: str = "lon",
    lat_col: str = "lat",
    name: str = "points",
) -> pd.DataFrame:
    """Coerce lon/lat to numeric and drop rows with NaN/Inf coordinates."""
    d = df.copy()
    d[lon_col] = pd.to_numeric(d[lon_col], errors="coerce")
    d[lat_col] = pd.to_numeric(d[lat_col], errors="coerce")
    m = np.isfinite(d[lon_col].to_numpy()) & np.isfinite(d[lat_col].to_numpy())
    dropped = int((~m).sum())
    if dropped:
        print(f"[pyvwf_to_atlite] dropping {dropped} {name} rows with non-finite lon/lat")
    return d.loc[m].copy()


def _normalise_domain_series(s: pd.Series) -> pd.Series:
    """Normalise various encodings to 'onshore'/'offshore'."""
    if s.dtype == bool:
        return s.map(lambda x: "offshore" if x else "onshore")

    if pd.api.types.is_numeric_dtype(s):
        return s.map(lambda x: "offshore" if int(x) == 1 else "onshore")

    ss = s.astype(str).str.lower().str.strip()
    mapping = {
        "onshore": "onshore",
        "offshore": "offshore",
        "land": "onshore",
        "sea": "offshore",
        "ocean": "offshore",
        "inland": "onshore",
        "true": "offshore",
        "false": "onshore",
        "1": "offshore",
        "0": "onshore",
    }
    return ss.map(lambda x: mapping.get(x, x))


def cutout_lonlat(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return 1D lon and lat coordinates from an atlite cutout dataset."""
    ds = ds.drop_vars(['lat','lon','height']).rename({'x':'lon', 'y':'lat'})
    if "lon" in ds.coords and "lat" in ds.coords:
        return ds["lon"].values, ds["lat"].values
    raise KeyError("Could not find lon/lat coordinates in cutout dataset (expected coords 'lon' and 'lat').")


def _union_geom(path: Path):
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"No geometries found in {path}")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    # GeoPandas >= 0.14: unary_union deprecated -> use union_all()
    return gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union


def mask_from_geojson_fast(
    lon: np.ndarray,
    lat: np.ndarray,
    geojson_path: Path,
    *,
    name: str,
) -> xr.DataArray:
    """
    Vectorised mask on (lat, lon) for points inside GeoJSON union.

    Uses GeoPandas spatial join if a spatial index (rtree/pygeos) is available.
    Falls back to a slower Point-in-Polygon loop otherwise.
    """
    geom = _union_geom(geojson_path)
    poly = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

    LON, LAT = np.meshgrid(lon, lat)
    pts = gpd.GeoDataFrame(
        {"idx": np.arange(LON.size, dtype=np.int64)},
        geometry=gpd.points_from_xy(LON.ravel(), LAT.ravel()),
        crs="EPSG:4326",
    )

    # Fast path: spatial join
    try:
        hit = gpd.sjoin(pts, poly, predicate="within", how="left")
        inside = hit["index_right"].notna().to_numpy().reshape(LAT.shape)
        return xr.DataArray(inside, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name=name)
    except Exception as e:
        warnings.warn(
            f"[pyvwf_to_atlite] Falling back to slow point-in-polygon mask for {geojson_path}. "
            f"Install rtree or pygeos for a big speedup. ({type(e).__name__}: {e})"
        )

    # Slow fallback
    geom_p = prep(geom)
    coords = np.column_stack([LON.ravel(), LAT.ravel()])
    inside = np.fromiter((geom_p.contains(Point(x, y)) for x, y in coords), dtype=bool, count=coords.shape[0])
    inside = inside.reshape(LAT.shape)
    return xr.DataArray(inside, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name=name)


def spatial_bin_average(
    df: pd.DataFrame,
    *,
    ddeg: float,
    lon_col: str = "lon",
    lat_col: str = "lat",
    value_cols: tuple[str, ...] = ("scalar", "offset"),
) -> pd.DataFrame:
    """Bin points onto a coarse lon/lat grid and average within bins."""
    d = df[[lon_col, lat_col, *value_cols]].dropna().copy()
    d["lon_bin"] = np.floor(d[lon_col] / ddeg).astype(int)
    d["lat_bin"] = np.floor(d[lat_col] / ddeg).astype(int)
    out = (
        d.groupby(["lon_bin", "lat_bin"], as_index=False)[[lon_col, lat_col, *value_cols]]
        .mean(numeric_only=True)
    )
    return out


def krige_to_grid(
    df: pd.DataFrame,
    *,
    value_col: str,
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    variogram_model: str = "spherical",
    nlags: int = 6,
    n_closest_points: int | None = None,
) -> xr.DataArray:
    """Ordinary kriging from scattered points to a lon/lat grid."""
    d = df[["lon", "lat", value_col]].dropna()
    d["lon"] = pd.to_numeric(d["lon"], errors="coerce")
    d["lat"] = pd.to_numeric(d["lat"], errors="coerce")
    m = np.isfinite(d["lon"].to_numpy()) & np.isfinite(d["lat"].to_numpy()) & np.isfinite(d[value_col].to_numpy())
    d = d.loc[m]
    if len(d) < 5:
        raise ValueError(f"Not enough points to krige {value_col}: {len(d)} points")

    ok = OrdinaryKriging(
        d["lon"].values,
        d["lat"].values,
        d[value_col].values,
        variogram_model=variogram_model,
        nlags=nlags,
        verbose=False,
        enable_plotting=False,
    )

    kwargs = {}
    if n_closest_points is not None:
        kwargs["n_closest_points"] = int(n_closest_points)

    # Moving window (n_closest_points) is NOT supported with backend="vectorized"
    backend = "vectorized"
    if n_closest_points is not None:
        # moving window requires loop or C
        backend = "C"  # try compiled first
        try:
            z, _ss = ok.execute("grid", lon_grid, lat_grid, backend=backend, **kwargs)
        except Exception:
            backend = "loop"
            z, _ss = ok.execute("grid", lon_grid, lat_grid, backend=backend, **kwargs)
    else:
        z, _ss = ok.execute("grid", lon_grid, lat_grid, backend=backend, **kwargs)
    return xr.DataArray(np.asarray(z), coords={"lat": lat_grid, "lon": lon_grid}, dims=("lat", "lon"), name=value_col)


def export_pyvwf_grid(
    *,
    cutout_nc: Path,
    points_csv: Path,
    out_nc: Path,
    onshore_geojson: str | Path,
    offshore_geojson: str | Path,
    domain_col: str | None = "type",
    scalar_col: str = "scalar",
    offset_col: str = "offset",
    variogram_model: str = "spherical",
    # performance controls
    workers: int = 1,
    onshore_thin_if_gt: int = 15000,
    onshore_bin_ddeg: float = 0.05,
    n_closest_onshore: int = 50,
    n_closest_offshore: int = 80,
) -> Path:
    """
    Export PyVWF correction fields onto an atlite cutout grid.

    Designed to be feasible for large point sets:
        - builds onshore/offshore AOI masks from user-provided GeoJSONs (fast sjoin path)
        - (optionally) spatially thins large onshore point sets via bin-averaging
        - uses local kriging (n_closest_points) to avoid global O(N^2) scaling
    """
    ds_cut = xr.open_dataset(cutout_nc)
    lon, lat = cutout_lonlat(ds_cut)
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError("Cutout lon/lat contain NaN/Inf. Check cutout coordinate extraction.")

    # AOI masks on cutout grid (use the provided paths!)
    onshore_geojson = Path(onshore_geojson)
    offshore_geojson = Path(offshore_geojson)

    onshore_aoi = mask_from_geojson_fast(lon, lat, onshore_geojson, name="is_onshore_aoi")
    offshore_aoi = mask_from_geojson_fast(lon, lat, offshore_geojson, name="is_offshore_aoi")
    offshore_aoi = offshore_aoi & (~onshore_aoi)  # enforce exclusivity (prefer onshore)

    

    # Load correction points
    pts = pd.read_csv(points_csv)
    pts = _coerce_finite_lonlat(pts, name="raw points")

    # Ensure lon/lat exist (common alt names handled)
    if "lon" not in pts.columns and "longitude" in pts.columns:
        pts = pts.rename(columns={"longitude": "lon"})
    if "lat" not in pts.columns and "latitude" in pts.columns:
        pts = pts.rename(columns={"latitude": "lat"})
    for req in ["lon", "lat"]:
        if req not in pts.columns:
            raise ValueError(f"points file must contain '{req}' column (or longitude/latitude)")

    # Domain split
    domain_series = None
    if domain_col is not None and domain_col in pts.columns:
        domain_series = _normalise_domain_series(pts[domain_col])

    if domain_series is None:
        # try common alternatives
        for c in ["domain", "type", "onshore_offshore", "site_type", "location_type", "is_offshore"]:
            if c in pts.columns:
                domain_series = _normalise_domain_series(pts[c])
                break

    if domain_series is None:
        # fall back to AOI membership of the point (centre)
        on_prep = prep(_union_geom(onshore_geojson))
        off_prep = prep(_union_geom(offshore_geojson))

        def classify(lon_, lat_):
            p = Point(float(lon_), float(lat_))
            if on_prep.contains(p):
                return "onshore"
            if off_prep.contains(p):
                return "offshore"
            return "unknown"

        domain_series = pts.apply(lambda r: classify(r["lon"], r["lat"]), axis=1)

    pts["domain_final"] = domain_series.astype(str).str.lower().str.strip()
    unknown = (pts["domain_final"] == "unknown").sum()
    if unknown:
        print(f"[pyvwf_to_atlite] Dropping {unknown} points outside onshore/offshore AOIs.")
    pts = pts.loc[pts["domain_final"].isin(["onshore", "offshore"])].copy()

    pts_on = pts.loc[pts["domain_final"].eq("onshore")].copy()
    pts_on = _coerce_finite_lonlat(pts_on, name="onshore points")
    pts_off = pts.loc[pts["domain_final"].eq("offshore")].copy()
    pts_off = _coerce_finite_lonlat(pts_off, name="offshore points")

    print(f"[pyvwf_to_atlite] onshore points: {len(pts_on)}")
    print(f"[pyvwf_to_atlite] offshore points: {len(pts_off)}")
    print(f"[pyvwf_to_atlite] grid size: {len(lon) * len(lat)}")

    if len(pts_on) < 5 or len(pts_off) < 5:
        raise ValueError(f"Insufficient points after domain split: onshore={len(pts_on)}, offshore={len(pts_off)}")

    # Thin large onshore set (huge speed/memory win)
    if len(pts_on) > int(onshore_thin_if_gt):
        before = len(pts_on)
        pts_on = spatial_bin_average(pts_on, ddeg=float(onshore_bin_ddeg), value_cols=(scalar_col, offset_col))
        print(f"[pyvwf_to_atlite] onshore thinning: {before} -> {len(pts_on)} (ddeg={onshore_bin_ddeg})")

    # Kriging tasks
    def _do(kind: str) -> tuple[str, xr.DataArray]:
        if kind == "scalar_on":
            return kind, krige_to_grid(
                pts_on, value_col=scalar_col, lon_grid=lon, lat_grid=lat,
                variogram_model=variogram_model, n_closest_points=n_closest_onshore
            ).rename("scalar_onshore")
        if kind == "offset_on":
            return kind, krige_to_grid(
                pts_on, value_col=offset_col, lon_grid=lon, lat_grid=lat,
                variogram_model=variogram_model, n_closest_points=n_closest_onshore
            ).rename("offset_onshore")
        if kind == "scalar_off":
            return kind, krige_to_grid(
                pts_off, value_col=scalar_col, lon_grid=lon, lat_grid=lat,
                variogram_model=variogram_model, n_closest_points=n_closest_offshore
            ).rename("scalar_offshore")
        if kind == "offset_off":
            return kind, krige_to_grid(
                pts_off, value_col=offset_col, lon_grid=lon, lat_grid=lat,
                variogram_model=variogram_model, n_closest_points=n_closest_offshore
            ).rename("offset_offshore")
        raise ValueError(kind)

    kinds = ["scalar_on", "offset_on", "scalar_off", "offset_off"]
    workers = int(workers)
    if workers <= 1:
        results = dict(_do(k) for k in kinds)
    else:
        # Cap workers to avoid macOS swap storms; local kriging helps, but still be sensible.
        workers = min(workers, max(1, os.cpu_count() or 1), 4)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_do, k) for k in kinds]
            results = dict(f.result() for f in futs)

    scalar_on = results["scalar_on"].where(onshore_aoi)
    offset_on = results["offset_on"].where(onshore_aoi)

    scalar_off = results["scalar_off"].where(offshore_aoi)
    offset_off = results["offset_off"].where(offshore_aoi)
    
    #  AOI masks (already computed earlier) 

    inside = (onshore_aoi | offshore_aoi)

    #  Apply masks to the interpolated surfaces (optional but recommended) 
    scalar_on = results["scalar_on"].where(onshore_aoi)
    offset_on = results["offset_on"].where(onshore_aoi)

    scalar_off = results["scalar_off"].where(offshore_aoi)
    offset_off = results["offset_off"].where(offshore_aoi)

    #  Combine into a single correction field (NaN outside both AOIs for now) 
    scalar = xr.where(onshore_aoi, scalar_on,
            xr.where(offshore_aoi, scalar_off, np.nan)).rename("scalar")

    offset = xr.where(onshore_aoi, offset_on,
            xr.where(offshore_aoi, offset_off, np.nan)).rename("offset")

    #  (Optional) clip/winsorise BEFORE identity fill 
    # scalar = scalar.clip(0.5, 1.5)
    # offset = offset.clip(-1.0, 1.0)

    #  Identity outside AOIs (THIS IS THE PART YOU ASKED FOR) 
    scalar = scalar.where(inside, other=1.0)
    offset = offset.where(inside, other=0.0)

    ds_out = xr.Dataset(
        {
            "is_onshore_aoi": onshore_aoi,
            "is_offshore_aoi": offshore_aoi,
            "scalar": scalar,
            "offset": offset,
            "scalar_onshore": scalar_on,
            "offset_onshore": offset_on,
            "scalar_offshore": scalar_off,
            "offset_offshore": offset_off,
        }
    )

    ds_out["scalar"].attrs.update(
        dict(long_name="PyVWF scalar correction", description="Multiplicative correction applied to wind power output")
    )
    ds_out["offset"].attrs.update(
        dict(long_name="PyVWF offset correction", description="Additive correction applied to wind power output")
    )
    ds_out.attrs.update(
        dict(
            title="PyVWF gridded bias correction fields on atlite cutout grid",
            interpolation=f"ordinary kriging ({variogram_model}); local neighbourhood",
            source_points=str(points_csv),
            cutout=str(cutout_nc),
            onshore_aoi=str(onshore_geojson),
            offshore_aoi=str(offshore_geojson),
            n_closest_onshore=int(n_closest_onshore),
            n_closest_offshore=int(n_closest_offshore),
            onshore_bin_ddeg=float(onshore_bin_ddeg) if len(pts_on) > int(onshore_thin_if_gt) else 0.0,
        )
    )

    out_nc = Path(out_nc)
    out_nc.parent.mkdir(parents=True, exist_ok=True)
    
    # 
    # Final rename for atlite compatibility
    # Atlite expects grid dimensions to be named 'x' and 'y'
    # 

    ds_out = ds_out.rename({"lon": "x", "lat": "y"})

    # Ensure ordering is (y, x), which atlite uses internally
    for v in ds_out.data_vars:
        if set(ds_out[v].dims) == {"y", "x"}:
            ds_out[v] = ds_out[v].transpose("y", "x")
            
    ds_out.to_netcdf(out_nc)
    return out_nc


if __name__ == "__main__":
    # Example usage (edit paths for your machine)
    CUTOUT_PATH = Path(
        "/Users/ellyess/Library/CloudStorage/OneDrive-ImperialCollegeLondon/PhD/pypsa-eur-wind/cutouts/europe-2023-sarah3-era5.nc"
    )
    out = export_pyvwf_grid(
        cutout_nc=CUTOUT_PATH,
        points_csv=Path("out/correction_points.csv"),
        out_nc=Path("out/pyvwf_bias_grid.nc"),
        onshore_geojson=Path("input/regions/country_shapes.geojson"),
        offshore_geojson=Path("input/regions/north_sea_shape.geojson"),
        domain_col="type",   # correction_points.csv uses 'type'
        variogram_model="spherical",
        workers=1,           # start with 1 on mac; increase cautiously after verifying no swap
        onshore_thin_if_gt=15000,
        onshore_bin_ddeg=0.05,
        n_closest_onshore=50,
        n_closest_offshore=80,
    )
    print("Wrote:", out)
