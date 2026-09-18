"""Leave-one-region-out transfer of correction factors by a random forest.

The machinery of the transfer re-test (``docs/findings/method-ml-transfer.md``):
per-cluster centroids with their fitted factors, terrain features at those
centroids, and a random forest scored by leave-one-region-out and by random
cross-validation. The run list, the gate and the output belong to the study
driver, ``scripts/analysis/ml_transfer_retest.py``; this module holds what that
driver and the physics-informed drivers share.

Pinned by ``tests/test_pin_ml_transfer.py``. The feature named ``roughness``
here is terrain-elevation roughness, the local standard deviation of
elevation, not the surface roughness length z0 that ``CONTEXT.md`` calls
roughness. The column keeps its name because the recorded feature sets and
outputs use it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

#: Forest settings of the development-branch experiment this re-tests.
RF_KW = dict(n_estimators=100, max_depth=10, min_samples_split=10)
#: The seeds every score is averaged over.
SEEDS = [0, 1, 2, 3, 42]

#: Terrain and position (the primary set).
SET_A = ["elevation", "slope", "aspect", "roughness", "curvature",
         "abs_lat", "lon_norm", "lat_norm"]
#: Terrain only.
SET_B = ["elevation", "slope", "aspect", "roughness", "curvature"]
#: The primary set plus fleet descriptors.
SET_C = SET_A + ["mean_height", "log_capacity", "n_plants"]


def build_centroids(runs: dict, root: Path) -> pd.DataFrame:
    """One row per cluster: centroid, fleet descriptors and fitted factors.

    Args:
        runs: Region code to ``(train directory relative to root, k)``.
        root: The directory the train directories are relative to.

    Returns:
        Frame with the cluster centroid (unweighted), mean hub height, total
        capacity, unit count, the fixed-slice ``scalar`` and ``offset``, the
        region, ``log_capacity`` and ``abs_lat``.
    """
    rows = []
    for region, (rel, k) in runs.items():
        d = root / rel
        turb = pd.read_csv(d / f"train_turb_info_{k}.csv")
        fac = pd.read_csv(d / f"factors_fixed_{k}.csv")
        assert fac["cluster"].is_unique, f"one row per cluster expected: {region}"

        g = turb.groupby("cluster").agg(
            lon=("lon", "mean"), lat=("lat", "mean"),
            mean_height=("height", "mean"),
            capacity=("capacity", "sum"), n_plants=("ID", "count"),
        ).reset_index()
        g = g.merge(fac[["cluster", "scalar", "offset"]], on="cluster")
        g["region"] = region
        rows.append(g)
    df = pd.concat(rows, ignore_index=True)
    df["log_capacity"] = np.log10(df["capacity"])
    df["abs_lat"] = df["lat"].abs()
    return df


def terrain_features(df: pd.DataFrame, etopo_path: Path) -> pd.DataFrame:
    """Add elevation, slope, aspect, roughness and curvature at each centroid.

    The derivation of the development branch's terrain features, applied per
    region to a 30 arc-second ETOPO subset one degree wider than the region's
    centroids, then sampled at the nearest grid point.
    """
    etopo = xr.open_dataset(etopo_path)
    res = 1.0 / 120.0  # 30 arcsec
    m_per_deg = 111132.954
    out = []
    for region, sub in df.groupby("region"):
        lo0, lo1 = sub.lon.min() - 1, sub.lon.max() + 1
        la0, la1 = sub.lat.min() - 1, sub.lat.max() + 1
        z = etopo.z.sel(lon=slice(lo0, lo1), lat=slice(la0, la1)).load()
        elev = z.values.astype("float64")
        lat_avg = float(z.lat.mean())
        dlat_m = res * m_per_deg
        dlon_m = res * m_per_deg * np.cos(np.deg2rad(lat_avg))
        grad_lat = np.gradient(elev, dlat_m, axis=0)
        grad_lon = np.gradient(elev, dlon_m, axis=1)
        slope = np.rad2deg(np.arctan(np.sqrt(grad_lat**2 + grad_lon**2)))
        aspect = (90 - np.rad2deg(np.arctan2(grad_lon, grad_lat))) % 360
        mean_elev = uniform_filter(elev, size=3, mode="nearest")
        roughness = np.sqrt(uniform_filter((elev - mean_elev) ** 2, size=3,
                                           mode="nearest"))
        curvature = (np.gradient(grad_lat, dlat_m, axis=0)
                     + np.gradient(grad_lon, dlon_m, axis=1))
        fields = dict(elevation=elev, slope=slope, aspect=aspect,
                      roughness=roughness, curvature=curvature)
        ii = np.searchsorted(z.lat.values, sub.lat.to_numpy()).clip(0, elev.shape[0] - 1)
        jj = np.searchsorted(z.lon.values, sub.lon.to_numpy()).clip(0, elev.shape[1] - 1)
        feat = {name: arr[ii, jj] for name, arr in fields.items()}
        out.append(pd.DataFrame(feat, index=sub.index))
    return df.join(pd.concat(out))


def rf_eval(train, test, feats, target, seed):
    """Fit a forest on ``train`` and score it on ``test``: ``(R2, MAE)``."""
    m = RandomForestRegressor(random_state=seed, **RF_KW)
    m.fit(train[feats], train[target])
    p = m.predict(test[feats])
    return r2_score(test[target], p), mean_absolute_error(test[target], p)


def loro(df, feats, target, seeds=SEEDS):
    """Leave-one-region-out scores, averaged over seeds, per held-out region."""
    recs = []
    for region in df.region.unique():
        tr, te = df[df.region != region], df[df.region == region]
        for s in seeds:
            r2, mae = rf_eval(tr, te, feats, target, s)
            recs.append(dict(holdout=region, seed=s, r2=r2, mae=mae))
    r = pd.DataFrame(recs).groupby("holdout").agg(
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        mae_mean=("mae", "mean")).reset_index()
    return r


def random_cv(df, feats, target, seeds=SEEDS):
    """Random 5-fold cross-validation over seeds: ``(mean R2, sd R2, mean MAE)``."""
    recs = []
    for s in seeds:
        kf = KFold(n_splits=5, shuffle=True, random_state=s)
        for tr_i, te_i in kf.split(df):
            r2, mae = rf_eval(df.iloc[tr_i], df.iloc[te_i], feats, target, s)
            recs.append((r2, mae))
    a = np.array(recs)
    return a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean()


def variance_decomposition(df, target):
    """The share of the target's variance that lies between regions."""
    grand = df[target].mean()
    between = df.groupby("region")[target].agg(["mean", "count"])
    ss_between = (between["count"] * (between["mean"] - grand) ** 2).sum()
    ss_total = ((df[target] - grand) ** 2).sum()
    return ss_between / ss_total
