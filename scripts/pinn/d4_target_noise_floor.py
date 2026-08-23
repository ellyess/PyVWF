#!/usr/bin/env python3
"""D4: how much of the fitted scalar is signal, and how much is noise?

Every transfer experiment so far -- the published one and D3 -- has asked
whether a model can predict the per-cluster scalar in an unseen region. None
has asked how much of that scalar is predictable IN PRINCIPLE. If the fitted
factors carry substantial estimation noise (few turbines per cluster, few
months of observation, curve-matching error), then no model, physics-informed
or otherwise, can explain that part, and the ceiling on any transfer score is
set by the data rather than by the method.

The tool is the empirical variogram. Two clusters whose centroids are close
together sit in the same or adjacent ERA5 cells and experience nearly the same
reanalysis bias, so their TRUE scalars should be nearly equal. The expected
squared difference between two clusters at separation d,

    gamma(d) = 0.5 * E[ (a_i - a_j)^2 ]

therefore tends, as d -> 0, not to zero but to the variance of the estimation
noise: the variogram's NUGGET. The ratio nugget/sill is the fraction of the
scalar's variance that carries no spatial information at all, and one minus it
is an upper bound on the variance any spatial or terrain-based model can
explain.

This is a measurement, not a model: no fitting to the evaluation, no gate.

Run: PYTHONPATH=src /opt/anaconda3/bin/python scripts/pinn/d4_target_noise_floor.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.ml_transfer_retest import RUNS, build_centroids  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pinn" / "d4"

# Bin edges in km. The first bin is deliberately narrower than one ERA5 cell
# (~28 km) so it is dominated by pairs sharing a grid cell.
BINS = np.array([0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600, 900])
EARTH_R = 6371.0


def haversine_matrix(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    dlo = lo[:, None] - lo[None, :]
    dla = la[:, None] - la[None, :]
    a = (np.sin(dla / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2)
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def spherical_model(d, nugget, psill, rng):
    """Classical spherical variogram: nugget + partial sill rising to range."""
    d = np.asarray(d, dtype=float)
    out = np.where(
        d <= rng,
        nugget + psill * (1.5 * d / rng - 0.5 * (d / rng) ** 3),
        nugget + psill,
    )
    return np.where(d <= 0, nugget, out)


def empirical_variogram(lon, lat, val):
    D = haversine_matrix(lon, lat)
    iu = np.triu_indices(len(val), k=1)
    d = D[iu]
    g = 0.5 * (val[iu[0]] - val[iu[1]]) ** 2
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() >= 15:
            rows.append(dict(d_mid=float(d[m].mean()), gamma=float(g[m].mean()),
                             n_pairs=int(m.sum()), lo=lo, hi=hi))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_centroids(RUNS)
    pd.set_option("display.width", 210)

    summ, vgs = [], []
    for region, sub in df.groupby("region"):
        lon = sub.lon.to_numpy(); lat = sub.lat.to_numpy()
        val = sub.scalar.to_numpy()
        vg = empirical_variogram(lon, lat, val)
        vg.insert(0, "region", region)
        vgs.append(vg)

        total_var = float(np.var(val, ddof=1))
        try:
            p0 = [vg.gamma.iloc[0], max(total_var - vg.gamma.iloc[0], 1e-6), 150.0]
            popt, _ = curve_fit(
                spherical_model, vg.d_mid, vg.gamma, p0=p0,
                bounds=([0, 0, 5], [total_var * 2, total_var * 4, 2000]),
                maxfev=20000,
            )
            nugget, psill, rng = popt
        except Exception as e:                      # noqa: BLE001
            print(f"  [{region}] variogram fit failed: {e}")
            nugget, psill, rng = np.nan, np.nan, np.nan

        sill = nugget + psill
        summ.append(dict(
            region=region, n_clusters=len(sub), scalar_var=total_var,
            nugget=nugget, sill=sill, range_km=rng,
            nugget_ratio=nugget / sill if sill else np.nan,
            max_explainable_r2=1 - (nugget / sill) if sill else np.nan,
            gamma_first_bin=float(vg.gamma.iloc[0]),
            first_bin_ratio=float(vg.gamma.iloc[0]) / total_var,
        ))

    s = pd.DataFrame(summ)
    vgall = pd.concat(vgs, ignore_index=True)
    s.to_csv(OUT / "d4_variogram_summary.csv", index=False)
    vgall.to_csv(OUT / "d4_variogram_bins.csv", index=False)

    print(f"{'='*100}\n### Variogram of the fitted scalar, per region\n")
    print(s[["region", "n_clusters", "scalar_var", "nugget", "sill", "range_km",
             "nugget_ratio", "max_explainable_r2"]].round(4).to_string(index=False))
    print("\n  nugget_ratio      = share of scalar variance with NO spatial structure")
    print("  max_explainable_r2 = 1 - nugget_ratio; the ceiling on any spatial/terrain model")

    print(f"\n{'='*100}\n### Empirical variogram bins (gamma should rise from the nugget)\n")
    for region, sub in vgall.groupby("region"):
        tv = float(s.loc[s.region == region, "scalar_var"].iloc[0])
        print(f"\n-- {region}  (total variance {tv:.4f}) --")
        show = sub[["lo", "hi", "d_mid", "n_pairs", "gamma"]].copy()
        show["gamma_over_var"] = show.gamma / tv
        print(show.round(3).to_string(index=False))

    print(f"\n{'='*100}\n### What this means for the published transfer scores\n")
    for _, r in s.iterrows():
        print(f"  {r.region}: published LORO R2 is measured against a target whose "
              f"ceiling is {r.max_explainable_r2:.2f}")
    print(f"\nwrote {OUT}/d4_*.csv")


if __name__ == "__main__":
    main()
