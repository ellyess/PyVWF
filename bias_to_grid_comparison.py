from __future__ import annotations

"""
bias_to_grid_comparison.py (from-scratch)

Purpose
-------
Compare several interpolation methods for bias-correction factors (e.g. scalar/offset)
from scattered control points to a lon/lat grid, using user-provided onshore/offshore
GeoJSON polygons. Outputs:
  - NetCDF surfaces (merged onshore+offshore into ONE grid per method/value)
  - Optional PNG maps (single map shows BOTH onshore and offshore on one plot)

Design goals
------------
- Simple, robust, macOS-friendly plotting (no Cartopy required).
- GeoJSON-only (no NaturalEarth/shapereader fallbacks).
- Clear orchestration style inspired by pyvwf_to_atlite.py:
    * small, explicit helpers
    * one top-level runner: run_bias_to_grid_comparison(...)

Inputs
------
Controls CSV (required) with columns:
  lon, lat, mode ('onshore'|'offshore'), scalar, offset
Grid NetCDF (required) with 1D coords:
  lon, lat
GeoJSON (required):
  onshore polygon(s), offshore polygon(s)

Notes
-----
- Ordinary Kriging requires pykrige.
- RBF uses SciPy's RBFInterpolator (SciPy >= 1.7). Local neighbors make it practical.
"""

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

import geopandas as gpd
from shapely.prepared import prep

# Optional deps
try:
    from pykrige.ok import OrdinaryKriging
    _HAS_PYKRIGE = True
except Exception:
    _HAS_PYKRIGE = False

try:
    from scipy.interpolate import RBFInterpolator
    _HAS_SCIPY_RBFI = True
except Exception:
    _HAS_SCIPY_RBFI = False


# ----------------------------
# Basic helpers (pyvwf style)
# ----------------------------

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
    out = d.loc[m].copy()
    dropped = len(d) - len(out)
    if dropped:
        warnings.warn(f"Dropped {dropped} {name} rows with non-finite lon/lat.")
    return out


def _union_geom(path: Path):
    gdf = gpd.read_file(path)
    
    if path.stem == 'offshore_shapes':
        north_sea = Path('input/regions/north_sea_shape.geojson')
        gdf_ns = gpd.read_file(north_sea)
        gdf = gdf.clip(gdf_ns)
        print("Offshore clipped to North Sea.")
    if gdf.empty:
        raise ValueError(f"GeoJSON is empty: {path}")
    geom = (gdf.geometry.union_all() if hasattr(gdf.geometry, 'union_all') else gdf.geometry.unary_union)
    return geom


def mask_from_geojson_fast(
    *,
    geojson: Path,
    lon: np.ndarray,
    lat: np.ndarray,
) -> xr.DataArray:
    """
    Rasterise a polygon GeoJSON to a (lat, lon) boolean mask.

    Implementation:
      - build point grid
      - prepared unary_union contains checks

    This is intentionally simple and dependency-light.
    """
    geom = prep(_union_geom(geojson))

    lon2, lat2 = np.meshgrid(lon, lat)
    # shapely contains isn't vectorized everywhere; list-comp is robust.
    flat = [geom.contains(gpd.points_from_xy([x], [y])[0]) for x, y in zip(lon2.ravel(), lat2.ravel())]
    mask = np.asarray(flat, dtype=bool).reshape(lon2.shape)

    return xr.DataArray(mask, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name=geojson.stem)


def _sym_vmin_vmax(values: np.ndarray, *, center: float) -> tuple[float, float]:
    a = np.asarray(values)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return center - 1.0, center + 1.0
    m = float(np.nanmax(np.abs(a - center)))
    if m == 0.0:
        m = 1.0
    return center - m, center + m


# ----------------------------
# Fast NN / IDW (BallTree)
# ----------------------------

def _balltree_haversine(lon: np.ndarray, lat: np.ndarray):
    from sklearn.neighbors import BallTree
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    pts_rad = np.deg2rad(np.c_[lat, lon])  # (lat, lon)
    tree = BallTree(pts_rad, metric="haversine")
    return tree


def _predict_nn(
    *,
    train_lon: np.ndarray,
    train_lat: np.ndarray,
    train_val: np.ndarray,
    query_lon: np.ndarray,
    query_lat: np.ndarray,
    max_dist_km: float,
) -> np.ndarray:
    tree = _balltree_haversine(train_lon, train_lat)
    q_rad = np.deg2rad(np.c_[query_lat.astype(float), query_lon.astype(float)])
    dist_rad, ind = tree.query(q_rad, k=1)
    dist_km = dist_rad[:, 0] * 6371.0
    out = np.full(len(query_lon), np.nan, dtype=float)
    ok = dist_km <= float(max_dist_km)
    out[ok] = np.asarray(train_val, dtype=float)[ind[ok, 0]]
    return out


def _predict_idw(
    *,
    train_lon: np.ndarray,
    train_lat: np.ndarray,
    train_val: np.ndarray,
    query_lon: np.ndarray,
    query_lat: np.ndarray,
    k: int,
    power: float,
    max_dist_km: float,
) -> np.ndarray:
    tree = _balltree_haversine(train_lon, train_lat)
    q_rad = np.deg2rad(np.c_[query_lat.astype(float), query_lon.astype(float)])
    k = int(max(1, k))
    dist_rad, ind = tree.query(q_rad, k=k)
    dist_km = dist_rad * 6371.0
    vals = np.asarray(train_val, dtype=float)[ind]  # (nq, k)
    within = dist_km <= float(max_dist_km)

    out = np.full(len(query_lon), np.nan, dtype=float)
    zero = within & (dist_km == 0.0)
    has_zero = zero.any(axis=1)
    if has_zero.any():
        out[has_zero] = vals[has_zero][zero[has_zero]].reshape(-1)

    rest = ~has_zero
    if rest.any():
        d = dist_km[rest]
        v = vals[rest]
        w = np.zeros_like(d, dtype=float)
        w[within[rest]] = 1.0 / np.maximum(d[within[rest]], 1e-12) ** float(power)
        wsum = w.sum(axis=1)
        ok = wsum > 0
        out_rest = np.full(d.shape[0], np.nan, dtype=float)
        out_rest[ok] = (w[ok] * v[ok]).sum(axis=1) / wsum[ok]
        out[rest] = out_rest

    return out

def _predict_ok(
    *,
    train_lon: np.ndarray,
    train_lat: np.ndarray,
    train_val: np.ndarray,
    query_lon: np.ndarray,
    query_lat: np.ndarray,
    variogram_model: str = "spherical",
    nlags: int = 6,
    n_closest_points: int | None = None,
) -> np.ndarray:
    """
    Ordinary Kriging prediction.
    Falls back gracefully to NaNs if kriging fails.
    """
    out = np.full(len(query_lon), np.nan, dtype=float)

    if len(train_val) < 5:
        return out

    try:
        ok = OrdinaryKriging(
            train_lon.astype(float),
            train_lat.astype(float),
            train_val.astype(float),
            variogram_model=variogram_model,
            nlags=int(nlags),
            verbose=False,
            enable_plotting=False,
        )

        kwargs = {}
        if n_closest_points is not None:
            kwargs["n_closest_points"] = min(int(n_closest_points), len(train_val))

        try:
            z, _ = ok.execute(
                "points",
                query_lon.astype(float),
                query_lat.astype(float),
                backend="C",
                **kwargs,
            )
        except Exception:
            z, _ = ok.execute(
                "points",
                query_lon.astype(float),
                query_lat.astype(float),
                backend="loop",
                **kwargs,
            )

        out[:] = np.asarray(z, dtype=float)
        return out

    except Exception as e:
        warnings.warn(
            f"OK failed ({variogram_model}, nlags={nlags}, "
            f"n_closest_points={n_closest_points}): {type(e).__name__}: {e}"
        )
        return out

def _predict_rbf(
    train: np.ndarray,
    train_val: np.ndarray,
    query: np.ndarray,
    kernel: str = "linear",
    **kwargs,
) -> np.ndarray:
    """
    Radial Basis Function interpolation.
    Returns NaN for queries too far from training data (if max_dist_km is set).
    """
    out = np.full(len(query), np.nan, dtype=float)

    if len(train_val) < 3:
        return out

    rbf = RBFInterpolator(
        train.astype(float),
        train_val.astype(float),
        kernel=kernel,
        **kwargs,
    )

    pred = rbf(query)
    pred = np.asarray(pred, dtype=float)

    out[:] = pred
    return out


def _grid_lonlat(grid_ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return 1D lon and lat coordinates from an atlite cutout dataset."""
    grid_ds = grid_ds.drop_vars(['lat','lon','height']).rename({'x':'lon', 'y':'lat'})
    if "lon" in grid_ds.coords and "lat" in grid_ds.coords:
        return grid_ds["lon"].values, grid_ds["lat"].values
    raise KeyError("Could not find lon/lat coordinates in cutout dataset (expected coords 'lon' and 'lat').")



def _masked_query_points(lon: np.ndarray, lat: np.ndarray, mask: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lon2, lat2 = np.meshgrid(lon, lat)
    m = np.asarray(mask.values, dtype=bool)
    idx = np.flatnonzero(m.ravel())
    qlon = lon2.ravel()[idx]
    qlat = lat2.ravel()[idx]
    return idx, qlon, qlat


def _scatter_to_grid(z: np.ndarray, idx: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.full(shape[0] * shape[1], np.nan, dtype=float)
    out[idx] = z
    return out.reshape(shape)


# =============================================================================
# INTERPOLATORS (point-wise; used for speed)
# =============================================================================
def interpolate_nn_masked(
    controls: pd.DataFrame,
    grid_ds: xr.Dataset,
    mask: xr.DataArray,
    *,
    value_col: str,
    max_dist_km: float,
) -> xr.DataArray:
    df = _coerce_finite_lonlat(controls, name="controls")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if len(df) < 1:
        raise ValueError(f"No valid controls for {value_col}.")

    lon, lat = _grid_lonlat(grid_ds)
    idx, qlon, qlat = _masked_query_points(lon, lat, mask)

    pred = _predict_nn(
        train_lon=df["lon"].to_numpy(),
        train_lat=df["lat"].to_numpy(),
        train_val=df[value_col].to_numpy(dtype=float),
        query_lon=qlon,
        query_lat=qlat,
        max_dist_km=max_dist_km,
    )
    grid = _scatter_to_grid(pred, idx, (len(lat), len(lon)))
    return xr.DataArray(grid, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name=f"{value_col}_nn")


def interpolate_idw_masked(
    controls: pd.DataFrame,
    grid_ds: xr.Dataset,
    mask: xr.DataArray,
    *,
    value_col: str,
    k: int,
    power: float,
    max_dist_km: float,
) -> xr.DataArray:
    df = _coerce_finite_lonlat(controls, name="controls")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col])
    if len(df) < 2:
        raise ValueError(f"Not enough valid controls for {value_col}: {len(df)}")

    lon, lat = _grid_lonlat(grid_ds)
    idx, qlon, qlat = _masked_query_points(lon, lat, mask)

    pred = _predict_idw(
        train_lon=df["lon"].to_numpy(),
        train_lat=df["lat"].to_numpy(),
        train_val=df[value_col].to_numpy(dtype=float),
        query_lon=qlon,
        query_lat=qlat,
        k=k,
        power=power,
        max_dist_km=max_dist_km,
    )
    grid = _scatter_to_grid(pred, idx, (len(lat), len(lon)))
    return xr.DataArray(grid, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name=f"{value_col}_idw")


def interpolate_ok_masked(
    controls: pd.DataFrame,
    grid_ds: xr.Dataset,
    mask: xr.DataArray,
    *,
    value_col: str,
    variogram_model: str = "spherical",
    nlags: int = 6,
    n_closest_points: int | None = None,
    # robustness knobs
    fallback: str = "idw",   # "idw" or "nearest" or "raise"
    fallback_idw_k: int = 8,
    fallback_idw_p: float = 2.0,
    fallback_max_dist_km: float = 500.0,
) -> xr.DataArray:
    """
    Ordinary kriging to a masked grid.

    Robustness:
      - Deduplicate exact lon/lat duplicates by averaging values.
      - Use pseudo-inverse solve (pseudo_inv=True) and exact_values=False to reduce singular failures.
      - If kriging still fails (singular/ill-conditioned), optionally fall back to IDW/NN.

    Notes:
      - Moving window (`n_closest_points`) is much faster but can be numerically fragile if points are clustered.
    """
    if not _HAS_PYKRIGE:
        raise RuntimeError("pykrige not installed; cannot run Ordinary Kriging.")

    df = _coerce_finite_lonlat(controls, name="controls")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col]).reset_index(drop=True)
    if len(df) < 5:
        raise ValueError(f"Not enough points to krige {value_col}: {len(df)} points")

    # Deduplicate exact coordinate repeats (common cause of singular matrices)
    df = (
        df.groupby(["lon", "lat"], as_index=False)[value_col]
        .mean()
        .merge(df.drop(columns=[value_col]).drop_duplicates(subset=["lon", "lat"]), on=["lon", "lat"], how="left")
    )
    # After merge, value_col exists from groupby; ensure clean
    df = df[["lon", "lat", value_col]].copy()

    lon, lat = _grid_lonlat(grid_ds)
    idx, qlon, qlat = _masked_query_points(lon, lat, mask)

    # Build OK model with more stable solve
    ok = OrdinaryKriging(
        df["lon"].to_numpy(float),
        df["lat"].to_numpy(float),
        df[value_col].to_numpy(float),
        variogram_model=str(variogram_model),
        nlags=int(nlags),
        verbose=False,
        enable_plotting=False,
        exact_values=False,
        pseudo_inv=True,
        pseudo_inv_type="pinv",
    )

    qlon_f = qlon.astype(float)
    qlat_f = qlat.astype(float)

    try:
        if n_closest_points is not None:
            kwargs = {"n_closest_points": min(int(n_closest_points), len(df), 50)}
            try:
                z, _ = ok.execute("points", qlon_f, qlat_f, backend="C", **kwargs)
            except Exception:
                z, _ = ok.execute("points", qlon_f, qlat_f, backend="loop", **kwargs)
        else:
            z, _ = ok.execute("points", qlon_f, qlat_f, backend="vectorized")

        grid = _scatter_to_grid(np.asarray(z, dtype=float), idx, (len(lat), len(lon)))
        return xr.DataArray(grid, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name=f"{value_col}_ok")

    except Exception as e:
        msg = f"OK failed for {value_col} ({variogram_model}, nlags={nlags}, n_closest_points={n_closest_points}): {type(e).__name__}: {e}"
        warnings.warn(msg)

        if fallback == "raise":
            raise

        if fallback == "nearest":
            da = interpolate_nn_masked(
                controls=controls,
                grid_ds=grid_ds,
                mask=mask,
                value_col=value_col,
                max_dist_km=float(fallback_max_dist_km),
            )
            da.name = f"{value_col}_ok_fallback_nn"
            return da

        # default: IDW
        da = interpolate_idw_masked(
            controls=controls,
            grid_ds=grid_ds,
            mask=mask,
            value_col=value_col,
            k=int(fallback_idw_k),
            power=float(fallback_idw_p),
            max_dist_km=float(fallback_max_dist_km),
        )
        da.name = f"{value_col}_ok_fallback_idw"
        return da


def interpolate_rbf_masked(
    controls: pd.DataFrame,
    grid_ds: xr.Dataset,
    mask: xr.DataArray,
    *,
    value_col: str,
    neighbors: int = 50,
    kernel: str = "thin_plate_spline",
    epsilon: float | None = None,
) -> xr.DataArray:
    if not _HAS_SCIPY_RBFI:
        raise RuntimeError("SciPy RBFInterpolator not available.")

    df = _coerce_finite_lonlat(controls, name="controls")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[value_col]).reset_index(drop=True)
    if len(df) < 5:
        raise ValueError(f"Not enough points for RBF {value_col}: {len(df)} points")
    # Deduplicate exact coordinate repeats (common cause of singular matrices)
    df = (
        df.groupby(["lon", "lat"], as_index=False)[value_col]
        .mean()
        .merge(df.drop(columns=[value_col]).drop_duplicates(subset=["lon", "lat"]), on=["lon", "lat"], how="left")
    )
    # After merge, value_col exists from groupby; ensure clean
    df = df[["lon", "lat", value_col]].copy()
    
    lon, lat = _grid_lonlat(grid_ds)
    idx, qlon, qlat = _masked_query_points(lon, lat, mask)


    lat0 = float(np.nanmean(df["lat"].to_numpy(dtype=float)))
    c = float(np.cos(np.deg2rad(lat0)))
    X = np.c_[df["lon"].to_numpy(dtype=float) * c, df["lat"].to_numpy(dtype=float)]
    Q = np.c_[qlon.astype(float) * c, qlat.astype(float)]

    nbh = min(int(neighbors), len(df))
    kwargs = {}
    if epsilon is not None:
        kwargs["epsilon"] = float(epsilon)

    rbf = RBFInterpolator(X, df[value_col].to_numpy(dtype=float), neighbors=nbh, kernel=kernel, **kwargs)
    pred = rbf(Q)
    grid = _scatter_to_grid(np.asarray(pred, dtype=float), idx, (len(lat), len(lon)))
    return xr.DataArray(grid, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name=f"{value_col}_rbf")

# =============================================================================
# SPATIAL BLOCKED CV
# =============================================================================

def assign_tiles(df: pd.DataFrame, tile_deg: float) -> pd.DataFrame:
    out = df.copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out = out.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    lat0 = np.floor(out["lat"].min() / tile_deg) * tile_deg
    lon0 = np.floor(out["lon"].min() / tile_deg) * tile_deg

    out["tile_i"] = np.floor((out["lat"] - lat0) / tile_deg).astype(int)
    out["tile_j"] = np.floor((out["lon"] - lon0) / tile_deg).astype(int)
    out["tile"] = out["tile_i"].astype(str) + "_" + out["tile_j"].astype(str)
    return out


def make_folds_from_tiles(df: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    tiles = df["tile"].unique().tolist()
    rng.shuffle(tiles)
    fold_map = {t: i % n_folds for i, t in enumerate(tiles)}
    return df["tile"].map(fold_map).astype(int)


def metrics_scalar_center1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    if m.sum() == 0:
        return {"mae": np.nan, "rmse": np.nan}
    e = np.log(y_pred[m]) - np.log(y_true[m])
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e**2)))}


def metrics_offset_center0(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return {"mae": np.nan, "rmse": np.nan}
    e = y_pred[m] - y_true[m]
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e**2)))}


def predict_at_points(
    method: str, 
    train: pd.DataFrame, 
    test: pd.DataFrame, 
    value_col: str,
    epsilon: float | None = None,
        # NN/IDW knobs
    max_dist_km: float = 300.0,
    idw_k: int = 8,
    idw_p: float = 2.0,
    # OK knobs
    ok_variogram: str = "spherical",
    ok_nlags: int = 6,
    ok_n_closest: int | None = 30,
    # RBF knobs
    rbf_neighbors: int = 50,
    rbf_kernel: str = "thin_plate_spline",
    ) -> np.ndarray:
    
    tr = _coerce_finite_lonlat(train, name="train")
    tr[value_col] = pd.to_numeric(tr[value_col], errors="coerce")
    tr = tr.dropna(subset=[value_col])
    te = _coerce_finite_lonlat(test, name="test")
    te = te.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if tr.empty or te.empty:
        return np.full((len(test),), np.nan, dtype=float)

    Xtr = tr[["lat", "lon"]].to_numpy(float)
    ytr = tr[value_col].to_numpy(float)
    Xte = te[["lat", "lon"]].to_numpy(float)
    
    def _nearest_fallback() -> np.ndarray:
        return _predict_nn(
            train_lon=Xtr[:, 1],
            train_lat=Xtr[:, 0],
            train_val=ytr,
            query_lon=Xte[:, 1],
            query_lat=Xte[:, 0],
            max_dist_km=max_dist_km,
        )
    
    if method == "nearest":
        return _nearest_fallback()

    if method == "idw":
        return _predict_idw(
            train_lon=Xtr[:, 1],
            train_lat=Xtr[:, 0],
            train_val=ytr,
            query_lon=Xte[:, 1],
            query_lat=Xte[:, 0],
            k=idw_k,
            power=idw_p,
            max_dist_km=max_dist_km,
        )

    if method == "rbf":
        tr2 = tr.groupby(["lat", "lon"], as_index=False)[value_col].mean()
        Xtr2 = tr2[["lat", "lon"]].to_numpy(float)
        ytr2 = tr2[value_col].to_numpy(float)
        
        lat0 = float(np.nanmean(tr2["lat"].to_numpy(dtype=float)))
        c = float(np.cos(np.deg2rad(lat0)))
        X = np.c_[Xtr2[:, 1] * c, Xtr2[:, 0]]
        Q = np.c_[Xte[:, 1] * c, Xte[:, 0]]
        try:
            kwargs = {}
            if epsilon is not None:
                kwargs["epsilon"] = float(epsilon)
                kwargs["neighbors"] = min(int(rbf_neighbors), len(tr2))
            return _predict_rbf(
                train = X,
                train_val=ytr2,
                query=Q,
                kernel=rbf_kernel,
                **kwargs
            )
        except np.linalg.LinAlgError:
            return _nearest_fallback()
    
    if method == "ok":
        if not _HAS_PYKRIGE:
            return np.full((len(test),), np.nan, dtype=float)
        tr2 = tr.groupby(["lon", "lat"], as_index=False)[value_col].mean()
        if tr2.empty:
            return np.full((len(test),), np.nan, dtype=float)
        zvals = tr2[value_col].to_numpy(float)
        
        return _predict_ok(
            train_lon=tr2["lon"].to_numpy(float),
            train_lat=tr2["lat"].to_numpy(float),
            train_val=zvals,
            query_lon=te["lon"].to_numpy(float),
            query_lat=te["lat"].to_numpy(float),
            variogram_model=ok_variogram,
            nlags=ok_nlags,
            n_closest_points=ok_n_closest,
        )
        
    raise ValueError(f"Unknown method={method}")

# Spatial blocked-CV tiling
TILE_DEG = 2.0
N_FOLDS = 5
RANDOM_SEED = 42

def run_blocked_cv(controls: pd.DataFrame, mode: str, value_col: str, methods: list[str]) -> pd.DataFrame:
    df = controls.copy()
    df["mode"] = df["mode"].astype(str).str.lower()
    df = df.loc[df["mode"] == mode].copy()
    
    if "mode" not in controls.columns:
        raise ValueError("controls_csv must contain a 'mode' column with 'onshore'/'offshore'.")
    df = _coerce_finite_lonlat(df, name="controls")


    if df.empty or df.shape[0] < 20:
        raise ValueError(f"Not enough control points for CV: mode={mode} value_col={value_col} n={len(df)}")

    df = assign_tiles(df, TILE_DEG)
    df["fold"] = make_folds_from_tiles(df, N_FOLDS, RANDOM_SEED)

    rows = []
    for method in methods:
        for fold in range(N_FOLDS):
            test = df.loc[df["fold"] == fold].copy()
            train = df.loc[df["fold"] != fold].copy()

            pred = predict_at_points(method=method, train=train, test=test, value_col=value_col)
            truth = test[value_col].to_numpy(float)

            met = metrics_scalar_center1(truth, pred) if value_col == "scalar" else metrics_offset_center0(truth, pred)

            rows.append({
                "mode": mode,
                "value": value_col,
                "method": method,
                "fold": fold,
                "n_test": int(np.isfinite(truth).sum()),
                "mae": met["mae"],
                "rmse": met["rmse"],
            })

    return pd.DataFrame(rows)

# ----------------------------
# Merge & plotting (single map)
# ----------------------------

def merge_onshore_offshore(
    *,
    onshore: xr.DataArray,
    offshore: xr.DataArray,
    onshore_mask: xr.DataArray,
    offshore_mask: xr.DataArray,
    name: str,
) -> xr.DataArray:
    merged = xr.full_like(onshore, np.nan, dtype=float)
    merged = merged.where(~onshore_mask, onshore)
    merged = merged.where(~offshore_mask, offshore)
    merged.name = name
    return merged


def plot_merged_map(
    da: xr.DataArray,
    *,
    outpath: Path,
    title: str,
    center: float,
    onshore_geojson: Path,
    offshore_geojson: Path,
    points: pd.DataFrame,
    value_col: str,
    extent: tuple[float, float, float, float] | None = None,
    dpi: int = 250,
) -> None:
    """
    One plot showing BOTH onshore+offshore on the same map:
      - merged surface (pcolormesh)
      - outlines of onshore/offshore polygons
      - control points (onshore circles, offshore triangles)

    Robustness:
      - Handles masked-array lon/lat coords by subsetting to finite coords only.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- sanitize coordinates (matplotlib pcolormesh requires finite X/Y) ---
    lon = da["lon"].values
    lat = da["lat"].values

    # If coords are masked arrays, fill to plain ndarray
    if isinstance(lon, np.ma.MaskedArray):
        lon = lon.filled(np.nan)
    if isinstance(lat, np.ma.MaskedArray):
        lat = lat.filled(np.nan)

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    m_lon = np.isfinite(lon)
    m_lat = np.isfinite(lat)
    if not m_lon.all() or not m_lat.all():
        # subset DataArray to finite coords only
        da = da.isel(lon=np.where(m_lon)[0], lat=np.where(m_lat)[0])
        lon = lon[m_lon]
        lat = lat[m_lat]

    # data array values
    z = da.values
    if isinstance(z, np.ma.MaskedArray):
        z = z.filled(np.nan)
    z = np.asarray(z, dtype=float)

    lon2, lat2 = np.meshgrid(lon, lat)

    vmin, vmax = _sym_vmin_vmax(z, center=center)

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.pcolormesh(
        lon2, lat2, z,
        shading="auto",
        vmin=vmin, vmax=vmax,
        cmap="PRGn",
        rasterized=True,
    )

    # polygon outlines
    g_on = gpd.read_file(onshore_geojson)
    g_off = gpd.read_file(offshore_geojson).clip(gpd.read_file(Path('input/regions/north_sea_shape.geojson')))
    try:
        g_on.boundary.plot(ax=ax, linewidth=0.3, color='black')
        g_off.boundary.plot(ax=ax, linewidth=0.3, color='black')
    except Exception:
        pass

    # points
    p = _coerce_finite_lonlat(points, name="controls")
    p[value_col] = pd.to_numeric(p[value_col], errors="coerce")
    p = p.dropna(subset=[value_col])

    pon = p.loc[p["mode"] == "onshore"]
    poff = p.loc[p["mode"] == "offshore"]

    if len(pon) > 0:
        ax.scatter(
            pon["lon"], pon["lat"],
            s=18, c=pon[value_col],
            vmin=vmin, vmax=vmax, cmap="PRGn",
            edgecolor="k", linewidth=0.1,
            alpha=0.8,
            marker="o", label="Onshore",
        )
    if len(poff) > 0:
        ax.scatter(
            poff["lon"], poff["lat"],
            s=22, c=poff[value_col],
            vmin=vmin, vmax=vmax, cmap="PRGn",
            edgecolor="k", linewidth=0.1,
            alpha=0.8,
            marker="^", label="Offshore",
        )

    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label(value_col.capitalize())

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.legend(loc="lower left", frameon=True,title="Control Points", fontsize=8)

    outpath.parent.mkdir(parents=True, exist_ok=True)

    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(pad=0.2)

    fig.savefig(
        outpath,
        dpi=int(dpi),
        bbox_inches="tight",
        pad_inches=0.02,
        facecolor="white",
    )
    plt.close(fig)


# ----------------------------
# Orchestration
# ----------------------------

def run_bias_to_grid_comparison(
    *,
    controls_csv: Path,
    grid_nc: Path,
    onshore_geojson: Path,
    offshore_geojson: Path,
    out_dir: Path,
    methods: list[str],
    export_nc: bool = True,
    make_plots: bool = True,
    # NN/IDW knobs
    max_dist_km: float = 300.0,
    idw_k: int = 8,
    idw_p: float = 2.0,
    # OK knobs
    ok_variogram: str = "spherical",
    ok_nlags: int = 6,
    ok_n_closest: int | None = 30,
    # RBF knobs
    rbf_neighbors: int = 50,
    rbf_kernel: str = "thin_plate_spline",
    ) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    controls = pd.read_csv(controls_csv)
    if "mode" not in controls.columns:
        raise ValueError("controls_csv must contain a 'mode' column with 'onshore'/'offshore'.")
    controls = _coerce_finite_lonlat(controls, name="controls")

    grid_ds = xr.open_dataset(grid_nc)

    lon, lat = _grid_lonlat(grid_ds)

    # masks
    onshore_mask = mask_from_geojson_fast(geojson=onshore_geojson, lon=lon, lat=lat)
    offshore_mask = mask_from_geojson_fast(geojson=offshore_geojson, lon=lon, lat=lat)

    surfaces_dir = out_dir / "surfaces"
    plots_dir = out_dir / "maps"
    surfaces_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    for value_col, center in [("scalar", 1.0), ("offset", 0.0)]:
        for method in methods:
            # interpolate separately onshore/offshore then merge into ONE map
            ctrl_on = controls.loc[controls["mode"] == "onshore"]
            ctrl_off = controls.loc[controls["mode"] == "offshore"]

            if method == "nearest":
                da_on = interpolate_nn_masked(ctrl_on, grid_ds, onshore_mask, value_col=value_col, max_dist_km=max_dist_km)
                da_off = interpolate_nn_masked(ctrl_off, grid_ds, offshore_mask, value_col=value_col, max_dist_km=max_dist_km)
            elif method == "idw":
                da_on = interpolate_idw_masked(ctrl_on, grid_ds, onshore_mask, value_col=value_col, k=idw_k, power=idw_p, max_dist_km=max_dist_km)
                da_off = interpolate_idw_masked(ctrl_off, grid_ds, offshore_mask, value_col=value_col, k=idw_k, power=idw_p, max_dist_km=max_dist_km)
            elif method == "ok":
                da_on = interpolate_ok_masked(ctrl_on, grid_ds, onshore_mask, value_col=value_col, variogram_model=ok_variogram, nlags=ok_nlags, n_closest_points=ok_n_closest, fallback='idw', fallback_idw_k=idw_k, fallback_idw_p=idw_p, fallback_max_dist_km=max_dist_km)
                da_off = interpolate_ok_masked(ctrl_off, grid_ds, offshore_mask, value_col=value_col, variogram_model=ok_variogram, nlags=ok_nlags, n_closest_points=ok_n_closest, fallback='idw', fallback_idw_k=idw_k, fallback_idw_p=idw_p, fallback_max_dist_km=max_dist_km)
            elif method == "rbf":
                da_on = interpolate_rbf_masked(ctrl_on, grid_ds, onshore_mask, value_col=value_col, neighbors=rbf_neighbors, kernel=rbf_kernel)
                da_off = interpolate_rbf_masked(ctrl_off, grid_ds, offshore_mask, value_col=value_col, neighbors=rbf_neighbors, kernel=rbf_kernel)
            else:
                raise ValueError(f"Unknown method: {method}")

            merged = merge_onshore_offshore(
                onshore=da_on,
                offshore=da_off,
                onshore_mask=onshore_mask,
                offshore_mask=offshore_mask,
                name=f"{value_col}_{method}_merged",
            )
            merged.attrs.update(dict(value_col=value_col, method=method, center=float(center)))

            if export_nc:
                out_nc = surfaces_dir / f"{value_col}_merged_{method}.nc"
                merged.astype("float32").to_netcdf(out_nc)

            if make_plots:
                out_png = plots_dir / f"{value_col}_merged_{method}.png"
                plot_merged_map(
                    merged,
                    outpath=out_png,
                    title=f"{method.upper()} — {value_col.title()} (Merged Onshore+Offshore)",
                    center=center,
                    onshore_geojson=onshore_geojson,
                    offshore_geojson=offshore_geojson,
                    points=controls,
                    value_col=value_col,
                )
    
    return out_dir

def run_cv_comparison(
    *,
    controls_csv: Path,
    out_dir: Path,
    methods: list[str],
    ) -> pd.DataFrame:
        #  load controls 
    controls = pd.read_csv(controls_csv)
    for c in ["mode", "lat", "lon", "scalar", "offset"]:
        if c not in controls.columns:
            raise ValueError(f"{controls_csv} missing required column '{c}'. cols={list(controls.columns)}")
    controls["mode"] = controls["mode"].astype(str).str.lower()

    #  CV compare 
    methods_cv = methods
    if ("ok" in methods_cv) and (not _HAS_PYKRIGE):
        methods_cv = [m for m in methods_cv if m != "ok"]

    all_scores = []
    for mode in ["onshore", "offshore"]:
        for value_col in ["scalar", "offset"]:
            all_scores.append(run_blocked_cv(controls, mode=mode, value_col=value_col, methods=methods_cv))

    scores_df = pd.concat(all_scores, ignore_index=True)
    scores_df.to_csv(out_dir / "interp_cv_scores.csv", index=False)

    summary = (
        scores_df.groupby(["mode", "value", "method"], as_index=False)
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"))
        .sort_values(["mode", "value", "rmse"])
    )
    summary.to_csv(out_dir / "interp_cv_summary.csv", index=False)
    print("\nCV SUMMARY (lower is better):")
    print(summary)
    
    return out_dir

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bias-to-grid comparison (from scratch).")
    p.add_argument("--controls-csv", type=Path, required=True, help="CSV with lon,lat,mode,scalar,offset")
    p.add_argument("--grid-nc", type=Path, required=True, help="NetCDF with 1D coords lon,lat")
    p.add_argument("--onshore-geojson", type=Path, required=True)
    p.add_argument("--offshore-geojson", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--methods", type=str, default="nearest,idw", help="Comma-separated methods: nearest,idw,ok,rbf")

    p.add_argument("--no-plots", action="store_true", help="Do not write PNG maps")
    p.add_argument("--no-nc", action="store_true", help="Do not write NetCDF surfaces")

    p.add_argument("--max-dist-km", type=float, default=300.0)
    p.add_argument("--idw-k", type=int, default=8)
    p.add_argument("--idw-p", type=float, default=2.0)

    p.add_argument("--ok-variogram", type=str, default="spherical")
    p.add_argument("--ok-nlags", type=int, default=6)
    p.add_argument("--ok-n-closest", type=int, default=30)

    p.add_argument("--rbf-neighbors", type=int, default=50)
    p.add_argument("--rbf-kernel", type=str, default="thin_plate_spline")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if "ok" in methods and not _HAS_PYKRIGE:
        raise RuntimeError("Requested method 'ok' but pykrige is not installed.")
    if "rbf" in methods and not _HAS_SCIPY_RBFI:
        raise RuntimeError("Requested method 'rbf' but SciPy RBFInterpolator not available.")

    # out_summary = run_cv_comparison(
    #     controls_csv=args.controls_csv,
    #     out_dir=args.out_dir,
    #     methods=methods,
    # )
    # print(f"CV Comparison COMPLETE. Outputs in: {out_summary}")  
    
    out = run_bias_to_grid_comparison(
        controls_csv=args.controls_csv,
        grid_nc=args.grid_nc,
        onshore_geojson=args.onshore_geojson,
        offshore_geojson=args.offshore_geojson,
        out_dir=args.out_dir,
        methods=methods,
        export_nc=(not args.no_nc),
        make_plots=(not args.no_plots),
        max_dist_km=args.max_dist_km,
        idw_k=args.idw_k,
        idw_p=args.idw_p,
        ok_variogram=args.ok_variogram,
        ok_nlags=args.ok_nlags,
        ok_n_closest=None if args.ok_n_closest <= 0 else int(args.ok_n_closest),
        rbf_neighbors=args.rbf_neighbors,
        rbf_kernel=args.rbf_kernel,
    )
    print(f"Plotting COMPLETE. Outputs in: {out}")


if __name__ == "__main__":
    main()

# python bias_to_grid_comparison.py \
#   --controls-csv out/correction_points.csv \
#   --grid-nc /Users/ellyess/Library/CloudStorage/OneDrive-ImperialCollegeLondon/PhD/pypsa-eur-wind/cutouts/europe-2023-sarah3-era5.nc \
#   --onshore-geojson input/regions/country_shapes.geojson \
#   --offshore-geojson input/regions/offshore_shapes.geojson \
#   --out-dir out/bias_to_grid \
#   --methods nearest,idw,ok,rbf