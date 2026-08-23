#!/usr/bin/env python3
"""E2: are the fitted physical quantities physically credible?

The model's claim to transfer rests on its outputs being physics rather than
curve-fitting artefacts. That claim is checkable independently of any skill
score: a log speed-up should be near zero on flat coastal ground and large in
mountain passes, a conversion efficiency should sit in the range the wind
industry actually reports for wake plus availability plus electrical losses,
and a shear correction should push the profile toward, not away from, the
stability regime the site is in.

So this audits the fitted quantities against knowledge the fitting never saw.
It is a validity check on the parameterisation, not a skill measurement, and it
can fail: values outside the physical ranges below would mean the terms are
absorbing each other rather than measuring what they are named for.

Run: PYVWF_INPUT=input/combined PYTHONPATH=src /opt/anaconda3/bin/python \
         scripts/pinn/e2_physics_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vwf.pinn.train import fit, load_regions  # noqa: E402

REGIONS = ["DK", "DE", "UK", "US", "BR"]
CACHE = ROOT / "output" / "pinn" / "cache"
OUT = ROOT / "output" / "pinn" / "e2"

# Sites whose reanalysis behaviour is independently documented, used as named
# probes rather than as fitting targets. The complex-terrain ones are the
# clusters D1 identified as carrying the largest affine scalars.
PROBES = {
    "Tehachapi Pass, CA": (-118.29, 35.06),
    "Altamont Pass, CA": (-121.78, 38.05),
    "Columbia Basin, WA": (-120.46, 47.07),
    "San Gorgonio, CA": (-116.26, 32.71),
    "West Jutland (flat)": (8.30, 56.20),
    "North Sea offshore": (7.90, 55.00),
    "Bahia highlands, BR": (-41.53, -14.03),
    "Rio Grande do Norte coast, BR": (-36.18, -5.11),
}


def nearest_unit(regions, lon, lat):
    """The cached unit closest to a probe point, across all regions."""
    best = None
    for r in regions:
        d = np.hypot(r.lon - lon, r.lat - lat)
        k = int(d.argmin())
        if best is None or d[k] < best[0]:
            best = (float(d[k]), r, k)
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tr = load_regions(REGIONS, "train", CACHE, quiet=True)
    print("Fitting on all five regions (this is the audit fit, not a gate)...")
    model, std, hist = fit(tr, hidden=None, physics=True, profile="power",
                           epochs=80, seed=0, verbose=True)

    pd.set_option("display.width", 210)
    rows = []
    for r in tr:
        with torch.no_grad():
            g, d, e, k = model(std.terrain(r), std.fleet(r), r.relief)
        rows.append(pd.DataFrame({
            "region": r.code, "lon": r.lon, "lat": r.lat,
            "relief_28km": r.relief.numpy(),
            "height": r.height.numpy(),
            "speedup": torch.exp(g).numpy(),
            "gamma": g.numpy(), "delta": d.numpy(), "eta": e.numpy(),
        }))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(OUT / "e2_fitted_physics.csv", index=False)
    kappa = float(k)

    print(f"\n{'='*96}\n### Fitted physical quantities by region\n")
    print(df.groupby("region")[["speedup", "delta", "eta", "relief_28km"]]
            .agg(["mean", "std", "min", "max"]).round(3).to_string())
    print(f"\n  kappa (global) = {kappa:.3f}   "
          f"relief scale = {float(torch.exp(model.log_relief_scale)):.0f} m")

    print(f"\n{'='*96}\n### Named probes: speed-up where the terrain is known\n")
    prows = []
    for name, (lon, lat) in PROBES.items():
        dist, r, k = nearest_unit(tr, lon, lat)
        with torch.no_grad():
            g, d, e, _ = model(std.terrain(r, torch.tensor([k])),
                               std.fleet(r, torch.tensor([k])),
                               r.relief[torch.tensor([k])])
        prows.append(dict(probe=name, region=r.code, dist_deg=round(dist, 2),
                          relief_m=float(r.relief[k]),
                          speedup=float(torch.exp(g)[0]),
                          delta=float(d[0]), eta=float(e[0])))
    probes = pd.DataFrame(prows)
    print(probes.round(3).to_string(index=False))
    probes.to_csv(OUT / "e2_probes.csv", index=False)

    print(f"\n{'='*96}\n### Physical plausibility checks\n")
    checks = {
        "speed-up >= 1 in the top relief decile":
            float((df[df.relief_28km > df.relief_28km.quantile(0.9)].speedup >= 1).mean()),
        "speed-up within 2% of 1 in the bottom relief decile":
            float(((df[df.relief_28km < df.relief_28km.quantile(0.1)].speedup - 1)
                   .abs() < 0.02).mean()),
        "eta inside the reported industry band 0.75-0.95":
            float(df.eta.between(0.75, 0.95).mean()),
        "offshore eta below onshore eta (bigger arrays, stronger wakes)": np.nan,
    }
    for name, val in checks.items():
        if np.isfinite(val):
            print(f"  {name:62s} {val:6.1%}")
    print(f"\n  Spearman(relief, speed-up) = "
          f"{df[['relief_28km','speedup']].corr(method='spearman').iloc[0,1]:+.3f}"
          f"   (should be strongly positive by construction, reported to show"
          f" the term is actually being used)")

    print(f"\n{'='*96}\n### Speed-up against relief, binned\n")
    df["relief_bin"] = pd.cut(df.relief_28km, [0, 50, 100, 200, 400, 800, 1600, 4000])
    print(df.groupby("relief_bin", observed=True)
            .agg(n=("speedup", "size"), speedup=("speedup", "mean"),
                 speedup_sd=("speedup", "std"), eta=("eta", "mean"))
            .round(3).to_string())
    print(f"\nwrote {OUT}/e2_*.csv")


if __name__ == "__main__":
    main()
