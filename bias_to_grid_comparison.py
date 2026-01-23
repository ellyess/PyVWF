"""
Compare interpolation methods for PyVWF bias-correction factors (scalar + offset)
and produce:
  1) Spatial blocked-CV scores (MAE/RMSE) per method, per mode (on/off), per variable
  2) Interpolated surfaces on an ERA5/cutout grid (masked to chosen countries)
  3) Plots with diverging colourbars centered at 1 (scalar) or 0 (offset)

Speed-ups in this version
-
- Precompute onshore/offshore masks once.
- Plot on a downsampled grid (PLOT_STRIDE) to reduce points by ~stride^2.
- Interpolate only at masked grid points, then scatter back to full grid (no full-grid interpolation).
- DO NOT produce RBF plots (still allowed in CV if you keep it in CV_METHODS).

Usage:
  python bias_to_grid_comparison.py
"""

from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.neighbors import BallTree

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader

from shapely.geometry import Point
from shapely.ops import unary_union

# Shapely 2.x vectorized contains helper (fast). If not available we fall back.
try:
    from shapely import contains_xy  # shapely>=2
    _HAS_CONTAINS_XY = True
except Exception:
    _HAS_CONTAINS_XY = False

# RBF (used only in CV; no plots)
from scipy.interpolate import RBFInterpolator

# Optional kriging
try:
    from pykrige.ok import OrdinaryKriging
    _HAS_PYKRIGE = True
except Exception:
    _HAS_PYKRIGE = False


# =============================================================================
# CONFIG — EDIT THESE
# =============================================================================

CUTOUT_PATH = Path(
    "/Users/ellyess/Library/CloudStorage/OneDrive-ImperialCollegeLondon/PhD/pypsa-eur-wind/cutouts/europe-2023-sarah3-era5.nc"
)
CONTROLS_CSV = Path("fixed_1_1_controls.csv")

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / "maps").mkdir(exist_ok=True)

COUNTRY_NAMES = [
    "United Kingdom",
    "France",
    "Belgium",
    "Netherlands",
    "Germany",
    "Denmark",
    "Norway",
]

PLOT_EXTENT = (-12, 30, 45, 72)

# Spatial blocked-CV tiling
TILE_DEG = 2.0
N_FOLDS = 5
RANDOM_SEED = 42

# Interpolation hyperparams
IDW_K = 12
IDW_P = 2.0
IDW_MAX_DIST_KM = 300.0

# RBF params (CV only)
RBF_NEIGHBORS = 80
RBF_KERNEL = "thin_plate_spline"
RBF_SMOOTHING = 1e-6

# Kriging settings
KRIG_VARIOMODEL = "spherical"
KRIG_COORDS_TYPE = "geographic"  # lon/lat degrees
KRIG_PSEUDO_INV = True
KRIG_N_CLOSEST = 30              # moving-window local kriging for point prediction
KRIG_BACKEND = "loop"            # avoids backend errors in some pykrige versions

# Mask behaviour
OFFSHORE_BUFFER_DEG = 2.0        # tune 1.5–3.0 to adjust how much ocean is shown

# Speed knobs
PLOT_STRIDE = 2                  # 2 -> ~4x fewer grid points; 3 -> ~9x fewer
PLOT_METHODS = ["nearest", "idw"] + (["ok"] if _HAS_PYKRIGE else [])
CV_METHODS = ["nearest", "idw", "rbf"] + (["ok"] if _HAS_PYKRIGE else [])  # RBF in CV is fine

# Chunk for IDW point prediction (larger = faster but more memory)
IDW_POINT_CHUNK = 200_000


# =============================================================================
# GRID NORMALIZATION (cutout -> lon/lat 1D coords)
# =============================================================================

def normalize_latlon_grid_drop_old(
    ds: xr.Dataset,
    *,
    x_name: str = "x",
    y_name: str = "y",
    lon_name_new: str = "lon",
    lat_name_new: str = "lat",
) -> xr.Dataset:
    """
    Drop any existing lon/lat vars/coords then rename x->lon and y->lat.
    Assumes x/y are already degrees (atlite/ERA5 cutouts typically are).
    """
    out = ds

    drop_names = []
    for nm in ["lon", "lat", lon_name_new, lat_name_new]:
        if nm in out.coords or nm in out.data_vars:
            drop_names.append(nm)
    if drop_names:
        out = out.drop_vars(drop_names, errors="ignore")

    if x_name not in out.dims and x_name not in out.coords:
        raise ValueError(f"Expected '{x_name}' in ds.dims/coords, got dims={list(out.dims)} coords={list(out.coords)}")
    if y_name not in out.dims and y_name not in out.coords:
        raise ValueError(f"Expected '{y_name}' in ds.dims/coords, got dims={list(out.dims)} coords={list(out.coords)}")

    out = out.rename({x_name: lon_name_new, y_name: lat_name_new})

    out[lon_name_new] = xr.DataArray(pd.to_numeric(out[lon_name_new].values, errors="coerce"), dims=(lon_name_new,))
    out[lat_name_new] = xr.DataArray(pd.to_numeric(out[lat_name_new].values, errors="coerce"), dims=(lat_name_new,))

    if out[lon_name_new].isnull().any():
        out = out.dropna(dim=lon_name_new)
    if out[lat_name_new].isnull().any():
        out = out.dropna(dim=lat_name_new)

    return out


# =============================================================================
# MASKS (countries, land, onshore/offshore)
# =============================================================================

def build_countries_union(country_names: list[str]):
    shp = shapereader.natural_earth(
        resolution="110m",
        category="cultural",
        name="admin_0_countries",
    )
    reader = shapereader.Reader(shp)
    geoms = []
    missing = []

    for name in country_names:
        found = False
        for rec in reader.records():
            if rec.attributes.get("NAME_LONG") == name or rec.attributes.get("NAME") == name:
                geoms.append(rec.geometry)
                found = True
        if not found:
            missing.append(name)

    if missing:
        warnings.warn(f"Some country names not found in Natural Earth: {missing}")

    if not geoms:
        raise ValueError("No country geometries found; check COUNTRY_NAMES.")

    return unary_union(geoms)


def build_land_union():
    shp = shapereader.natural_earth(
        resolution="110m",
        category="physical",
        name="land",
    )
    reader = shapereader.Reader(shp)
    return unary_union([rec.geometry for rec in reader.records()])


def mask_grid_to_geom(grid_ds: xr.Dataset, geom, name: str = "mask") -> xr.DataArray:
    lat = grid_ds["lat"].values
    lon = grid_ds["lon"].values
    LON, LAT = np.meshgrid(lon, lat)

    if _HAS_CONTAINS_XY:
        inside = contains_xy(geom, LON, LAT)
    else:
        inside = np.zeros(LON.shape, dtype=bool)
        for i in range(LON.shape[0]):
            for j in range(LON.shape[1]):
                inside[i, j] = geom.contains(Point(float(LON[i, j]), float(LAT[i, j])))

    return xr.DataArray(
        inside,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name=name,
    )


def build_onshore_offshore_masks(
    grid_ds: xr.Dataset,
    country_names: list[str],
    *,
    offshore_buffer_deg: float = 2.0,
    prefer_lsm: bool = True,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Onshore = inside selected countries AND on land.
    Offshore = within buffered selected countries AND NOT on land.
    If grid_ds contains 'lsm' and prefer_lsm=True, use it for land/sea (faster).
    """
    countries_union = build_countries_union(country_names)
    countries_mask = mask_grid_to_geom(grid_ds, countries_union, name="countries_mask")

    if prefer_lsm and ("lsm" in grid_ds.data_vars):
        land_mask = (grid_ds["lsm"] > 0.5).astype(bool)
        land_mask = land_mask.rename("land_mask")
    else:
        land_union = build_land_union()
        land_mask = mask_grid_to_geom(grid_ds, land_union, name="land_mask")

    onshore_mask = (countries_mask & land_mask).rename("onshore_mask")

    offshore_zone = countries_union.buffer(offshore_buffer_deg)
    offshore_zone_mask = mask_grid_to_geom(grid_ds, offshore_zone, name="offshore_zone_mask")
    offshore_mask = (offshore_zone_mask & (~land_mask)).rename("offshore_mask")

    return onshore_mask, offshore_mask


# =============================================================================
# Shared helpers: controls + grid points + masked scatter
# =============================================================================

def _prep_controls(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df.copy()
    out["lat"] = pd.to_numeric(out["lat"], errors="coerce")
    out["lon"] = pd.to_numeric(out["lon"], errors="coerce")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")
    out = out.dropna(subset=["lat", "lon", value_col]).reset_index(drop=True)
    return out


def _grid_pts_and_shape(grid_ds: xr.Dataset):
    lat = grid_ds["lat"].values
    lon = grid_ds["lon"].values
    LON, LAT = np.meshgrid(lon, lat)
    shape = LAT.shape
    pts_latlon = np.column_stack([LAT.ravel(), LON.ravel()])
    return lat, lon, shape, pts_latlon


def _masked_pts(mask: xr.DataArray, pts_latlon: np.ndarray):
    m = mask.values.ravel()
    idx = np.where(m)[0]
    return idx, pts_latlon[idx]


def _scatter_to_grid(pred_1d: np.ndarray, idx: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.full((shape[0] * shape[1],), np.nan, dtype=float)
    out[idx] = pred_1d
    return out.reshape(shape)


# =============================================================================
# INTERPOLATORS (point-wise; used for speed)
# =============================================================================

def _idw_points(
    Xtr_latlon: np.ndarray,
    ytr: np.ndarray,
    Xte_latlon: np.ndarray,
    *,
    k: int,
    power: float,
    max_dist_km: float | None,
    chunk_size: int,
) -> np.ndarray:
    tree = BallTree(np.deg2rad(Xtr_latlon), metric="haversine")
    k_eff = min(int(k), len(Xtr_latlon))
    earth_km = 6371.0
    eps = 1e-12

    out = np.full((Xte_latlon.shape[0],), np.nan, dtype=float)
    Xte_rad = np.deg2rad(Xte_latlon)

    for start in range(0, Xte_rad.shape[0], chunk_size):
        end = min(start + chunk_size, Xte_rad.shape[0])
        dist_rad, idx = tree.query(Xte_rad[start:end], k=k_eff)
        dist_km = dist_rad * earth_km
        neigh = ytr[idx]

        if max_dist_km is not None:
            mask_far = dist_km > float(max_dist_km)
            neigh = np.where(mask_far, np.nan, neigh)
            dist_km = np.where(mask_far, np.nan, dist_km)

        w = 1.0 / np.maximum(dist_km, eps) ** float(power)
        w = np.where(np.isfinite(neigh), w, 0.0)

        wsum = np.sum(w, axis=1)
        num = np.sum(w * np.nan_to_num(neigh, nan=0.0), axis=1)
        out[start:end] = num / np.where(wsum > 0, wsum, np.nan)

    return out


def interpolate_nn_masked(
    controls: pd.DataFrame,
    grid_ds: xr.Dataset,
    mask: xr.DataArray,
    *,
    value_col: str,
    max_dist_km: float | None = None,
    out_name: str = "nn",
) -> xr.DataArray:
    ctrl = _prep_controls(controls, value_col)
    if ctrl.empty:
        raise ValueError("No valid controls for NN.")

    lat, lon, shape, pts = _grid_pts_and_shape(grid_ds)
    idx, pts_m = _masked_pts(mask, pts)

    Xtr = ctrl[["lat", "lon"]].to_numpy(float)
    ytr = ctrl[value_col].to_numpy(float)

    tree = BallTree(np.deg2rad(Xtr), metric="haversine")
    dist_rad, nn_idx = tree.query(np.deg2rad(pts_m), k=1)
    dist_km = dist_rad[:, 0] * 6371.0
    pred = ytr[nn_idx[:, 0]].astype(float)

    if max_dist_km is not None:
        pred = np.where(dist_km <= float(max_dist_km), pred, np.nan)

    grid = _scatter_to_grid(pred, idx, shape)
    return xr.DataArray(grid, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name=out_name)


def interpolate_idw_masked(
    controls: pd.DataFrame,
    grid_ds: xr.Dataset,
    mask: xr.DataArray,
    *,
    value_col: str,
    k: int = IDW_K,
    power: float = IDW_P,
    max_dist_km: float | None = IDW_MAX_DIST_KM,
    out_name: str = "idw",
) -> xr.DataArray:
    ctrl = _prep_controls(controls, value_col)
    if ctrl.empty:
        raise ValueError("No valid controls for IDW.")

    lat, lon, shape, pts = _grid_pts_and_shape(grid_ds)
    idx, pts_m = _masked_pts(mask, pts)

    Xtr = ctrl[["lat", "lon"]].to_numpy(float)
    ytr = ctrl[value_col].to_numpy(float)

    pred = _idw_points(
        Xtr, ytr, pts_m,
        k=k, power=power, max_dist_km=max_dist_km,
        chunk_size=IDW_POINT_CHUNK,
    )

    grid = _scatter_to_grid(pred, idx, shape)
    return xr.DataArray(grid, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name=out_name)


def _ok_kwargs() -> dict:
    # PyKrige version differences exist; these are generally safe.
    return dict(
        variogram_model=KRIG_VARIOMODEL,
        variogram_parameters=None,  # auto-fit
        coordinates_type=KRIG_COORDS_TYPE,
        pseudo_inv=KRIG_PSEUDO_INV,
        pseudo_inv_type="pinv",
        verbose=False,
        enable_plotting=False,
    )


def _dedupe_df_lonlat_mean(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df.groupby(["lon", "lat"], as_index=False)[value_col].mean()


def interpolate_ok_masked(
    controls: pd.DataFrame,
    grid_ds: xr.Dataset,
    mask: xr.DataArray,
    *,
    value_col: str,
    out_name: str = "ok",
) -> xr.DataArray:
    if not _HAS_PYKRIGE:
        raise RuntimeError("pykrige not installed; cannot run Ordinary Kriging.")

    ctrl = _prep_controls(controls, value_col)
    if ctrl.empty:
        raise ValueError("No valid controls for OK.")
    ctrl = _dedupe_df_lonlat_mean(ctrl, value_col)

    lat, lon, shape, pts = _grid_pts_and_shape(grid_ds)
    idx, pts_m = _masked_pts(mask, pts)

    zvals = ctrl[value_col].to_numpy(float)

    ok = OrdinaryKriging(
        ctrl["lon"].to_numpy(float),
        ctrl["lat"].to_numpy(float),
        zvals,
        **_ok_kwargs(),
    )

    z, _ = ok.execute(
        "points",
        pts_m[:, 1].astype(float),  # lon
        pts_m[:, 0].astype(float),  # lat
        n_closest_points=min(int(KRIG_N_CLOSEST), len(ctrl)),
        backend=KRIG_BACKEND,
    )
    pred = np.asarray(z, dtype=float)

    grid = _scatter_to_grid(pred, idx, shape)
    return xr.DataArray(grid, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name=out_name)


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


def predict_at_points(method: str, train: pd.DataFrame, test: pd.DataFrame, value_col: str) -> np.ndarray:
    tr = _prep_controls(train, value_col)
    te = test.copy()
    te["lat"] = pd.to_numeric(te["lat"], errors="coerce")
    te["lon"] = pd.to_numeric(te["lon"], errors="coerce")
    te = te.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if tr.empty or te.empty:
        return np.full((len(test),), np.nan, dtype=float)

    Xtr = tr[["lat", "lon"]].to_numpy(float)
    ytr = tr[value_col].to_numpy(float)
    Xte = te[["lat", "lon"]].to_numpy(float)

    def _nearest_fallback():
        tree = BallTree(np.deg2rad(Xtr), metric="haversine")
        _, idx = tree.query(np.deg2rad(Xte), k=1)
        return ytr[idx[:, 0]].astype(float)

    if method == "nearest":
        return _nearest_fallback()

    if method == "idw":
        return _idw_points(
            Xtr, ytr, Xte,
            k=IDW_K, power=IDW_P, max_dist_km=IDW_MAX_DIST_KM,
            chunk_size=50_000,
        ).astype(float)

    if method == "rbf":
        tr2 = tr.groupby(["lat", "lon"], as_index=False)[value_col].mean()
        if len(tr2) < 4:
            return _nearest_fallback()
        Xtr2 = tr2[["lat", "lon"]].to_numpy(float)
        ytr2 = tr2[value_col].to_numpy(float)
        try:
            rbf = RBFInterpolator(
                Xtr2, ytr2,
                kernel=RBF_KERNEL,
                neighbors=min(int(RBF_NEIGHBORS), len(tr2)),
                smoothing=float(RBF_SMOOTHING),
                degree=(-1 if RBF_KERNEL == "thin_plate_spline" else 0),
            )
            return rbf(Xte).astype(float)
        except np.linalg.LinAlgError:
            return _nearest_fallback()

    if method == "ok":
        if not _HAS_PYKRIGE:
            return np.full((len(test),), np.nan, dtype=float)
        tr2 = _dedupe_df_lonlat_mean(tr, value_col)
        if tr2.empty:
            return np.full((len(test),), np.nan, dtype=float)
        zvals = tr2[value_col].to_numpy(float)
        ok = OrdinaryKriging(
            tr2["lon"].to_numpy(float),
            tr2["lat"].to_numpy(float),
            zvals,
            **_ok_kwargs(),
        )
        z, _ = ok.execute(
            "points",
            te["lon"].to_numpy(float),
            te["lat"].to_numpy(float),
            n_closest_points=min(int(KRIG_N_CLOSEST), len(tr2)),
            backend=KRIG_BACKEND,
        )
        return np.asarray(z, dtype=float)

    raise ValueError(f"Unknown method={method}")


def run_blocked_cv(controls: pd.DataFrame, mode: str, value_col: str, methods: list[str]) -> pd.DataFrame:
    df = controls.copy()
    df["mode"] = df["mode"].astype(str).str.lower()
    df = df.loc[df["mode"] == mode].copy()
    df = _prep_controls(df, value_col)

    if df.empty or df.shape[0] < 20:
        raise ValueError(f"Not enough control points for CV: mode={mode} value_col={value_col} n={len(df)}")

    df = assign_tiles(df, TILE_DEG)
    df["fold"] = make_folds_from_tiles(df, N_FOLDS, RANDOM_SEED)

    rows = []
    for method in methods:
        for fold in range(N_FOLDS):
            test = df.loc[df["fold"] == fold].copy()
            train = df.loc[df["fold"] != fold].copy()

            pred = predict_at_points(method, train, test, value_col=value_col)
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


# =============================================================================
# PLOTTING HELPERS
# =============================================================================

def _sym_vmin_vmax(values: np.ndarray, center: float) -> tuple[float, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return (np.nan, np.nan)
    d = np.nanmax(np.abs(v - center))
    return (center - d, center + d)


def plot_surface(
    da: xr.DataArray,
    *,
    title: str,
    center: float,
    extent=PLOT_EXTENT,
    cmap="RdBu_r",
    overlay_points: pd.DataFrame | None = None,
    point_value_col: str | None = None,
    outpath: Path | None = None,
):
    lon = da["lon"].values
    lat = da["lat"].values
    LON, LAT = np.meshgrid(lon, lat)

    vmin, vmax = _sym_vmin_vmax(da.values, center=center)

    fig = plt.figure(figsize=(11, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    im = ax.pcolormesh(
        LON, LAT, da.values,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.6)
    gl.top_labels = False
    gl.right_labels = False

    cb = plt.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label(str(da.name or "value"))
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax != vmin:
        cb.ax.axhline((center - vmin) / (vmax - vmin), linewidth=1.0)

    if overlay_points is not None and len(overlay_points) > 0:
        p = overlay_points.copy()
        p["lon"] = pd.to_numeric(p["lon"], errors="coerce")
        p["lat"] = pd.to_numeric(p["lat"], errors="coerce")
        p = p.dropna(subset=["lon", "lat"])

        if point_value_col is not None and point_value_col in p.columns:
            p[point_value_col] = pd.to_numeric(p[point_value_col], errors="coerce")
            ax.scatter(
                p["lon"], p["lat"],
                s=10, c=p[point_value_col],
                cmap=cmap, vmin=vmin, vmax=vmax,
                transform=ccrs.PlateCarree(),
                edgecolor="k", linewidth=0.2, zorder=6,
            )
        else:
            ax.scatter(
                p["lon"], p["lat"],
                s=10, transform=ccrs.PlateCarree(),
                edgecolor="k", facecolor="none", linewidth=0.3, zorder=6,
            )

    ax.set_title(title)
    plt.tight_layout()

    if outpath is not None:
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
    else:
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    #  load controls 
    controls = pd.read_csv(CONTROLS_CSV)
    for c in ["mode", "lat", "lon", "scalar", "offset"]:
        if c not in controls.columns:
            raise ValueError(f"{CONTROLS_CSV} missing required column '{c}'. cols={list(controls.columns)}")
    controls["mode"] = controls["mode"].astype(str).str.lower()

    #  load grid 
    ds = xr.open_dataset(CUTOUT_PATH)
    grid_ds_full = normalize_latlon_grid_drop_old(ds, x_name="x", y_name="y")

    #  downsample for plotting 
    if int(PLOT_STRIDE) > 1:
        grid_ds = grid_ds_full.isel(
            lon=slice(None, None, int(PLOT_STRIDE)),
            lat=slice(None, None, int(PLOT_STRIDE)),
        )
    else:
        grid_ds = grid_ds_full

    #  masks (computed on plotting grid) 
    onshore_mask, offshore_mask = build_onshore_offshore_masks(
        grid_ds, COUNTRY_NAMES, offshore_buffer_deg=OFFSHORE_BUFFER_DEG, prefer_lsm=True
    )

    #  CV compare 
    methods_cv = CV_METHODS
    if ("ok" in methods_cv) and (not _HAS_PYKRIGE):
        methods_cv = [m for m in methods_cv if m != "ok"]

    all_scores = []
    for mode in ["onshore", "offshore"]:
        for value_col in ["scalar", "offset"]:
            all_scores.append(run_blocked_cv(controls, mode=mode, value_col=value_col, methods=methods_cv))

    scores_df = pd.concat(all_scores, ignore_index=True)
    scores_df.to_csv(OUT_DIR / "interp_cv_scores.csv", index=False)

    summary = (
        scores_df.groupby(["mode", "value", "method"], as_index=False)
        .agg(mae=("mae", "mean"), rmse=("rmse", "mean"))
        .sort_values(["mode", "value", "rmse"])
    )
    summary.to_csv(OUT_DIR / "interp_cv_summary.csv", index=False)
    print("\nCV SUMMARY (lower is better):")
    print(summary)

    #  full-grid surfaces + plots (NO RBF PLOTS) 
    methods_plot = [m for m in PLOT_METHODS if m != "rbf"]
    if ("ok" in methods_plot) and (not _HAS_PYKRIGE):
        methods_plot = [m for m in methods_plot if m != "ok"]

    for mode in ["onshore", "offshore"]:
        ctrl_m = controls.loc[controls["mode"] == mode].copy()
        mask = onshore_mask if mode == "onshore" else offshore_mask

        for value_col, center in [("scalar", 1.0), ("offset", 0.0)]:
            cmap = "RdBu_r"

            for method in methods_plot:
                if method == "nearest":
                    da = interpolate_nn_masked(
                        ctrl_m, grid_ds, mask,
                        value_col=value_col,
                        max_dist_km=IDW_MAX_DIST_KM,
                        out_name=f"{value_col}_{method}_{mode}",
                    )
                elif method == "idw":
                    da = interpolate_idw_masked(
                        ctrl_m, grid_ds, mask,
                        value_col=value_col,
                        k=IDW_K, power=IDW_P, max_dist_km=IDW_MAX_DIST_KM,
                        out_name=f"{value_col}_{method}_{mode}",
                    )
                elif method == "ok":
                    da = interpolate_ok_masked(
                        ctrl_m, grid_ds, mask,
                        value_col=value_col,
                        out_name=f"{value_col}_{method}_{mode}",
                    )
                else:
                    continue

                out_png = OUT_DIR / "maps" / f"{value_col}_{mode}_{method}.png"
                plot_surface(
                    da,
                    title=f"{method.upper()} {value_col} — {mode} (center={center})",
                    center=center,
                    cmap=cmap,
                    overlay_points=ctrl_m,
                    point_value_col=value_col,
                    outpath=out_png,
                )

    print(f"\nWrote:\n  - {OUT_DIR / 'interp_cv_scores.csv'}\n  - {OUT_DIR / 'interp_cv_summary.csv'}\n  - {OUT_DIR / 'maps'}/*.png")
    if int(PLOT_STRIDE) > 1:
        print(f"NOTE: plotted on downsampled grid with PLOT_STRIDE={PLOT_STRIDE}.")


if __name__ == "__main__":
    main()