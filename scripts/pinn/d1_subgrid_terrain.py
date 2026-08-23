#!/usr/bin/env python3
"""D1: do the correction scalars track SUB-GRID terrain exposure?

Must-distinguish test for the physics-informed route. ERA5 runs on a ~31 km
effective orography, so a turbine on a ridge or in a mountain pass sits in a
grid cell whose mean elevation is far below its own. Linear speed-up theory
(Jackson and Hunt 1975) says the fractional speed-up over a hill scales with
the hill's aspect ratio H/L, so the reanalysis should under-read exactly where
the sub-grid elevation excess is large and positive.

If that is what the extreme US scalars (Tehachapi, Altamont, central
Washington) are made of, a physical speed-up term is the right thing to put in
the model. If the scalars show no relationship to sub-grid exposure, the
terrain-speed-up hypothesis is dead and this route should be abandoned rather
than tuned.

The test is designed so it CAN fail: the prediction is a positive, monotone,
cross-region-consistent relationship between sub-grid elevation excess and the
scalar. A null or region-inconsistent relationship falsifies it.

ERA5's own resolved orography is only on disk for Europe, so the cell-mean
elevation is taken as the ETOPO mean over each 0.25 deg cell -- available
globally and a reasonable stand-in for what the model resolves.

Run: PYTHONPATH=src /opt/anaconda3/bin/python scripts/pinn/d1_subgrid_terrain.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.ml_transfer_retest import RUNS, build_centroids  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pinn" / "d1"
ETOPO = ROOT / "input/reference/terrain/etopo_global.nc"

ERA5_CELL = 0.25          # deg, the reanalysis grid the corrections were fit on
RES = 1.0 / 60.0          # ETOPO 2022 60 arcsec product actually ships 15"; read below


def subgrid_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-centroid sub-grid terrain exposure relative to the ERA5 cell."""
    etopo = xr.open_dataset(ETOPO)
    lat_v, lon_v = etopo.lat.values, etopo.lon.values
    dlat = float(abs(lat_v[1] - lat_v[0]))
    dlon = float(abs(lon_v[1] - lon_v[0]))
    # how many ETOPO pixels span one ERA5 cell
    nlat = max(int(round(ERA5_CELL / dlat)), 1)
    nlon = max(int(round(ERA5_CELL / dlon)), 1)

    recs = []
    for region, sub in df.groupby("region"):
        lo0, lo1 = sub.lon.min() - 0.6, sub.lon.max() + 0.6
        la0, la1 = sub.lat.min() - 0.6, sub.lat.max() + 0.6
        z = etopo.z.sel(lon=slice(lo0, lo1), lat=slice(la0, la1)).load()
        elev = z.values.astype("float64")
        zlat, zlon = z.lat.values, z.lon.values

        for idx, r in sub.iterrows():
            i = int(np.abs(zlat - r.lat).argmin())
            j = int(np.abs(zlon - r.lon).argmin())
            i0, i1 = max(i - nlat // 2, 0), min(i + nlat // 2 + 1, elev.shape[0])
            j0, j1 = max(j - nlon // 2, 0), min(j + nlon // 2 + 1, elev.shape[1])
            box = elev[i0:i1, j0:j1]
            box_land = np.where(box < -50, np.nan, box)   # ignore deep bathymetry
            z_site = float(elev[i, j])
            z_cell = float(np.nanmean(box_land)) if np.isfinite(box_land).any() else np.nan
            z_std = float(np.nanstd(box_land)) if np.isfinite(box_land).any() else np.nan
            z_rng = (float(np.nanmax(box_land) - np.nanmin(box_land))
                     if np.isfinite(box_land).any() else np.nan)
            recs.append(dict(
                index=idx,
                z_site=z_site,
                z_cell_mean=z_cell,
                h_excess=z_site - z_cell,          # metres above the resolved cell
                cell_relief=z_rng,                 # metres, cell max-min
                cell_std=z_std,
                # Jackson-Hunt-flavoured aspect ratio: excess height over the
                # half-width of the ERA5 cell (~12.5 km at the equator).
                aspect_ratio=(z_site - z_cell) / (ERA5_CELL * 111_320.0 / 2.0),
                exposure=((z_site - z_cell) / z_std) if z_std and z_std > 1 else 0.0,
            ))
        etopo_lat_res = dlat
    out = pd.DataFrame(recs).set_index("index")
    print(f"  ETOPO grid: dlat={dlat*3600:.0f}\" dlon={dlon*3600:.0f}\"  "
          f"-> ERA5 cell = {nlat} x {nlon} pixels")
    return df.join(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_centroids(RUNS)
    print("Computing sub-grid terrain exposure...")
    df = subgrid_features(df)
    df.to_csv(OUT / "d1_subgrid.csv", index=False)

    pd.set_option("display.width", 200)
    print(f"\n{'='*88}\n### Sub-grid exposure by region\n")
    print(df.groupby("region")[["h_excess", "cell_relief", "exposure", "scalar"]]
            .agg(["mean", "std", "max"]).round(2).to_string())

    print(f"\n{'='*88}\n### Correlation of scalar with sub-grid terrain, per region\n")
    rows = []
    for region, sub in df.groupby("region"):
        for var in ("h_excess", "cell_relief", "exposure", "aspect_ratio"):
            v, s = sub[var], sub["scalar"]
            ok = v.notna() & s.notna()
            if ok.sum() < 5:
                continue
            pr, pp = stats.pearsonr(v[ok], s[ok])
            sr, sp = stats.spearmanr(v[ok], s[ok])
            rows.append(dict(region=region, var=var, n=int(ok.sum()),
                             pearson_r=pr, pearson_p=pp,
                             spearman_r=sr, spearman_p=sp))
    corr = pd.DataFrame(rows)
    corr.to_csv(OUT / "d1_correlations.csv", index=False)
    for var in ("h_excess", "cell_relief", "exposure", "aspect_ratio"):
        print(f"\n-- {var} --")
        print(corr[corr["var"] == var][
            ["region", "n", "pearson_r", "pearson_p", "spearman_r", "spearman_p"]
        ].round(4).to_string(index=False))

    print(f"\n{'='*88}\n### Pooled (all regions)\n")
    for var in ("h_excess", "cell_relief", "exposure", "aspect_ratio"):
        ok = df[var].notna() & df["scalar"].notna()
        pr, pp = stats.pearsonr(df.loc[ok, var], df.loc[ok, "scalar"])
        sr, sp = stats.spearmanr(df.loc[ok, var], df.loc[ok, "scalar"])
        print(f"  {var:14s} pearson {pr:+.3f} (p={pp:.2e})   "
              f"spearman {sr:+.3f} (p={sp:.2e})")

    print(f"\n{'='*88}\n### The 12 largest scalars: what terrain are they on?\n")
    top = df.nlargest(12, "scalar")[
        ["region", "lon", "lat", "scalar", "z_site", "z_cell_mean",
         "h_excess", "cell_relief", "exposure"]]
    print(top.round(2).to_string(index=False))

    print(f"\n### The 12 smallest scalars\n")
    bot = df.nsmallest(12, "scalar")[
        ["region", "lon", "lat", "scalar", "z_site", "z_cell_mean",
         "h_excess", "cell_relief", "exposure"]]
    print(bot.round(2).to_string(index=False))

    print(f"\nwrote {OUT/'d1_subgrid.csv'}")


if __name__ == "__main__":
    main()
