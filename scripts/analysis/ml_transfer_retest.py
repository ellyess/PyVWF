#!/usr/bin/env python3
"""ML transfer re-test on five-region, post-fix correction targets.

Re-runs the development-branch ML transfer experiment (leave-one-region-out
prediction of correction factors from terrain + spatial features) against the
post-fix pipeline outputs on this branch. Gates were pre-specified before any
model run; see docs/findings/ml_transfer_retest.md.

Requires the train outputs under output/validation/ (train-sweep2/train-sweep3
runs at commit 8a032d6) and input/reference/terrain/etopo_global.nc.

Run: /opt/anaconda3/bin/python scripts/analysis/ml_transfer_retest.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "ml_retest"

# ---------------------------------------------------------------- dataset ----
RUNS = {  # region -> (train dir, k)  — canonical post-fix runs, commit 8a032d6
    "DK": ("output/validation/DK/train-sweep3", 100),
    "DE": ("output/validation/DE/train-sweep3", 100),
    "UK": ("output/validation/UK/train-sweep3", 100),
    "US": ("output/validation/US/train-sweep2", 100),
    "BR": ("output/validation/BR/train-sweep2", 60),
}
SENSITIVITY_RUNS = {
    "BR120": {"BR": ("output/validation/BR/train-sweep2", 120)},
    "high-k": {
        "DK": ("output/validation/DK/train-sweep3", 500),
        "DE": ("output/validation/DE/train-sweep3", 500),
        "UK": ("output/validation/UK/train-sweep3", 500),
        "US": ("output/validation/US/train-sweep2", 300),
        "BR": ("output/validation/BR/train-sweep2", 120),
    },
}

RF_KW = dict(n_estimators=100, max_depth=10, min_samples_split=10)  # = prior
SEEDS = [0, 1, 2, 3, 42]

SET_A = ["elevation", "slope", "aspect", "roughness", "curvature",
         "abs_lat", "lon_norm", "lat_norm"]
SET_B = ["elevation", "slope", "aspect", "roughness", "curvature"]
SET_C = SET_A + ["mean_height", "log_capacity", "n_plants"]


def build_centroids(runs: dict) -> pd.DataFrame:
    rows = []
    for region, (rel, k) in runs.items():
        d = ROOT / rel
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


# ------------------------------------------------------ terrain features ----
def terrain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same derivation as development:scripts/pyvwf_ml/download_terrain_data.py
    (compute_terrain_derivatives), applied per-region on an ETOPO 2022 30"
    bounding-box subset, then nearest-sampled at centroids."""
    etopo = xr.open_dataset(ROOT / "input/reference/terrain/etopo_global.nc")
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
        ii = np.searchsorted(z.lat.values, sub.lat.values).clip(0, elev.shape[0] - 1)
        jj = np.searchsorted(z.lon.values, sub.lon.values).clip(0, elev.shape[1] - 1)
        feat = {name: arr[ii, jj] for name, arr in fields.items()}
        out.append(pd.DataFrame(feat, index=sub.index))
    return df.join(pd.concat(out))


# ------------------------------------------------------------------ tests ----
def rf_eval(train, test, feats, target, seed):
    m = RandomForestRegressor(random_state=seed, **RF_KW)
    m.fit(train[feats], train[target])
    p = m.predict(test[feats])
    return r2_score(test[target], p), mean_absolute_error(test[target], p)


def loro(df, feats, target):
    recs = []
    for region in df.region.unique():
        tr, te = df[df.region != region], df[df.region == region]
        for s in SEEDS:
            r2, mae = rf_eval(tr, te, feats, target, s)
            recs.append(dict(holdout=region, seed=s, r2=r2, mae=mae))
    r = pd.DataFrame(recs).groupby("holdout").agg(
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        mae_mean=("mae", "mean")).reset_index()
    return r


def random_cv(df, feats, target):
    recs = []
    for s in SEEDS:
        kf = KFold(n_splits=5, shuffle=True, random_state=s)
        for tr_i, te_i in kf.split(df):
            r2, mae = rf_eval(df.iloc[tr_i], df.iloc[te_i], feats, target, s)
            recs.append((r2, mae))
    a = np.array(recs)
    return a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean()


def variance_decomposition(df, target):
    grand = df[target].mean()
    between = df.groupby("region")[target].agg(["mean", "count"])
    ss_between = (between["count"] * (between["mean"] - grand) ** 2).sum()
    ss_total = ((df[target] - grand) ** 2).sum()
    return ss_between / ss_total


def run_suite(df, label):
    print(f"\n{'='*70}\n### {label}  (n={len(df)})")
    # normalise lon/lat over the pooled dataset (cosmetic for trees)
    for c in ("lon", "lat"):
        df[f"{c}_norm"] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
    df = terrain_features(df)
    print("\nPer-region target stats:")
    print(df.groupby("region")[["scalar", "offset"]]
            .agg(["mean", "std", "count"]).round(3).to_string())
    for target in ("scalar", "offset"):
        vb = variance_decomposition(df, target)
        print(f"\nT5 {target}: between-region variance share = {vb:.1%}")
    results = {}
    for name, feats in [("T1 SetA", SET_A), ("T2 SetB", SET_B), ("T3 SetC", SET_C)]:
        for target in ("scalar", "offset"):
            r = loro(df, feats, target)
            results[(name, target)] = r
            print(f"\n{name} LORO [{target}]:")
            print(r.round(3).to_string(index=False))
    for target in ("scalar", "offset"):
        m, sd, mae = random_cv(df, SET_A, target)
        print(f"\nT4 random 5-fold CV SetA [{target}]: "
              f"R2 = {m:.3f} ± {sd:.3f}, MAE = {mae:.3f}")
    # gate check on T1 scalar
    t1 = results[("T1 SetA", "scalar")]
    n_pos = int((t1.r2_mean > 0).sum())
    print(f"\nGATE (T1 scalar, primary set only): {n_pos}/5 regions R2>0 -> "
          f"{'POSITIVE' if n_pos >= 3 else 'NEGATIVE result stands'}")
    return df, results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_centroids(RUNS)
    df, _ = run_suite(df, "PRIMARY (DK/DE/UK/US k=100, BR k=60)")
    df.to_csv(OUT / "ml_retest_centroids_primary.csv", index=False)

    for name, override in SENSITIVITY_RUNS.items():
        runs = {**RUNS, **override}
        d2 = build_centroids(runs)
        run_suite(d2, f"SENSITIVITY {name}")


if __name__ == "__main__":
    main()
