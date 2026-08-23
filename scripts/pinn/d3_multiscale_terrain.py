#!/usr/bin/env python3
"""D3: is the transfer failure a FEATURE-SCALE problem?

The published transfer experiment described terrain with derivatives computed
on a 3-pixel window of a 30 arc-second grid -- roughly 90 metres around the
cluster centroid. The bias being corrected arises at the scale of the ERA5 grid
cell (0.25 deg, ~28 km): that is the terrain the reanalysis smooths away, and
D1 found the ERA5-cell relief correlates with the fitted scalar consistently
and significantly in 4 of 5 regions (BR +0.76, DE +0.70, UK +0.51, US +0.44;
DK null, and DK is flat enough to have almost no relief variance), where the
90-metre features do not.

This tests whether describing terrain at the RIGHT scale, with no physics and
no change of model, moves cross-region transfer. It is the baseline that the
physics-informed model must beat (gate P2 in pinn_prespecification.md), and it
is also gate P3's ablation in embryo: if scale alone closes the gap, the
finding is about feature scale, not physics.

Pre-specified prediction (written before running): the pure-physiography
multi-scale set, which contains NO longitude/latitude and so cannot encode
region identity, beats the published SET_A on wind-space transfer skill in
>= 3 of 5 regions, and specifically improves the two failing regions, US and UK.

Run: PYTHONPATH=src /opt/anaconda3/bin/python scripts/pinn/d3_multiscale_terrain.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.ml_transfer_retest import (  # noqa: E402
    RUNS, SEEDS, SET_A, RF_KW, build_centroids, terrain_features,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pinn" / "d3"
ETOPO = ROOT / "input/reference/terrain/etopo_global.nc"

# Window half-widths in kilometres, and what each is meant to capture:
#   1  km  micro-siting (what the published features saw)
#   5  km  individual hill / ridge
#   28 km  the ERA5 grid cell itself -- the terrain the reanalysis cannot see
#   84 km  the orographic-drag / blocking scale
SCALES_KM = {"1km": 1.0, "5km": 5.0, "28km": 28.0, "84km": 84.0}
WS_GRID = np.arange(4.0, 20.1, 0.5)     # above cut-in: speeds that make power


def multiscale_terrain(df: pd.DataFrame) -> pd.DataFrame:
    """Terrain position, spread and relief at four scales around each centroid."""
    etopo = xr.open_dataset(ETOPO)
    dlat = float(abs(etopo.lat.values[1] - etopo.lat.values[0]))
    km_per_px = dlat * 111.32
    print(f"  ETOPO {dlat*3600:.0f} arcsec  ->  {km_per_px:.2f} km/pixel")

    out = []
    for region, sub in df.groupby("region"):
        pad = 1.0
        z = etopo.z.sel(lon=slice(sub.lon.min() - pad, sub.lon.max() + pad),
                        lat=slice(sub.lat.min() - pad, sub.lat.max() + pad)).load()
        elev = z.values.astype("float32")
        zlat, zlon = z.lat.values, z.lon.values
        ii = np.abs(zlat[None, :] - sub.lat.values[:, None]).argmin(axis=1)
        jj = np.abs(zlon[None, :] - sub.lon.values[:, None]).argmin(axis=1)
        z_site = elev[ii, jj].astype("float64")

        feat = {"z_site": z_site}
        # land fraction at the ERA5 cell scale: offshore vs onshore, continuous
        n28 = max(int(round(28.0 / km_per_px)) | 1, 3)
        feat["land_frac_28km"] = uniform_filter(
            (elev > 0).astype("float32"), size=n28, mode="nearest")[ii, jj]

        for name, km in SCALES_KM.items():
            n = max(int(round(km / km_per_px)) | 1, 3)   # odd window
            mean = uniform_filter(elev, size=n, mode="nearest")
            msq = uniform_filter(elev.astype("float64") ** 2, size=n, mode="nearest")
            std = np.sqrt(np.clip(msq - mean.astype("float64") ** 2, 0, None))
            relief = (maximum_filter(elev, size=n, mode="nearest")
                      - minimum_filter(elev, size=n, mode="nearest"))
            feat[f"tpi_{name}"] = z_site - mean[ii, jj]      # height above surroundings
            feat[f"std_{name}"] = std[ii, jj]                 # roughness of the terrain
            feat[f"relief_{name}"] = relief[ii, jj].astype("float64")
            del mean, msq, std, relief
        del elev
        out.append(pd.DataFrame(feat, index=sub.index))
    return df.join(pd.concat(out))


def windspace_loro(df, feats, label):
    """LORO transfer scored where the correction is applied: wind speed, 4-20 m/s."""
    rows = []
    for region in sorted(df.region.unique()):
        tr, te = df[df.region != region], df[df.region == region]
        w_true = (te["scalar"].to_numpy()[:, None] * WS_GRID[None, :]
                  + te["offset"].to_numpy()[:, None])
        pa = np.mean([RandomForestRegressor(random_state=s, **RF_KW)
                      .fit(tr[feats], tr["scalar"]).predict(te[feats])
                      for s in SEEDS], axis=0)[:, None]
        pb = np.mean([RandomForestRegressor(random_state=s, **RF_KW)
                      .fit(tr[feats], tr["offset"]).predict(te[feats])
                      for s in SEEDS], axis=0)[:, None]
        w_pred = pa * WS_GRID[None, :] + pb
        w_none = np.broadcast_to(WS_GRID[None, :], w_true.shape)
        mse_p = float(np.mean((w_true - w_pred) ** 2))
        mse_n = float(np.mean((w_true - w_none) ** 2))
        rows.append(dict(featureset=label, holdout=region,
                         rmse_uncorr=np.sqrt(mse_n), rmse_pred=np.sqrt(mse_p),
                         skill=1 - mse_p / mse_n))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_centroids(RUNS)
    for c in ("lon", "lat"):
        df[f"{c}_norm"] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
    print("Building published SET_A features...")
    df = terrain_features(df)
    print("Building multi-scale terrain features...")
    df = multiscale_terrain(df)
    df.to_csv(OUT / "d3_features.csv", index=False)

    MULTI = (["land_frac_28km", "abs_lat", "z_site"]
             + [f"{p}_{s}" for s in SCALES_KM for p in ("tpi", "std", "relief")])

    SETS = {
        "SET_A (published)": SET_A,
        "MULTI (physiography only, no lon/lat)": MULTI,
        "SET_A + MULTI": SET_A + MULTI,
        "MULTI + hub height": MULTI + ["mean_height"],
    }

    pd.set_option("display.width", 210)
    print(f"\n{'='*94}\n### Wind-space LORO transfer skill (higher is better; >0 beats no correction)\n")
    allr = []
    for label, feats in SETS.items():
        r = windspace_loro(df, feats, label)
        allr.append(r)
    res = pd.concat(allr, ignore_index=True)
    res.to_csv(OUT / "d3_loro_skill.csv", index=False)

    piv = res.pivot(index="holdout", columns="featureset", values="skill")
    piv = piv[list(SETS.keys())]
    print(piv.round(3).to_string())
    print("\n  regions with skill > 0:")
    for label in SETS:
        print(f"    {label:40s} {int((piv[label]>0).sum())}/5   "
              f"mean skill {piv[label].mean():+.3f}   median {piv[label].median():+.3f}")

    base = piv["SET_A (published)"]
    print(f"\n{'='*94}\n### Pre-specified check: does MULTI beat SET_A?\n")
    for label in list(SETS)[1:]:
        better = piv[label] > base
        print(f"  {label:40s} beats SET_A in {int(better.sum())}/5 "
              f"({', '.join(sorted(piv.index[better]))})")
    m = piv["MULTI (physiography only, no lon/lat)"]
    print(f"\n  US: SET_A {base['US']:+.3f} -> MULTI {m['US']:+.3f}")
    print(f"  UK: SET_A {base['UK']:+.3f} -> MULTI {m['UK']:+.3f}")

    print(f"\n{'='*94}\n### Feature importance, pooled fit on MULTI (which scale matters?)\n")
    rf = RandomForestRegressor(random_state=0, **RF_KW).fit(df[MULTI], df["scalar"])
    imp = pd.Series(rf.feature_importances_, index=MULTI).sort_values(ascending=False)
    print(imp.head(12).round(4).to_string())

    print(f"\nwrote {OUT}/d3_*.csv")


if __name__ == "__main__":
    main()
