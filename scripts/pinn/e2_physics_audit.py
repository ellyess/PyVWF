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

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vwf.pinn.train import fit, load_regions, predict_frame  # noqa: E402

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


def fitted_frame(model, std, regions, density: bool) -> pd.DataFrame:
    """Per-unit fitted physical quantities across every region."""
    rows = []
    for r in regions:
        with torch.no_grad():
            g, d, e, k = model(std.terrain(r), std.fleet(r), r.relief)
        rows.append(pd.DataFrame({
            "region": r.code, "lon": r.lon, "lat": r.lat,
            "relief_28km": r.relief.numpy(), "elevation": r.elevation.numpy(),
            "height": r.height.numpy(),
            "speedup": torch.exp(g).numpy(), "gamma": g.numpy(),
            "delta": d.numpy(), "eta": e.numpy(), "density": density,
        }))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tr = load_regions(args.regions, "train", CACHE, quiet=True)
    pd.set_option("display.width", 210)

    fits = {}
    for density in (False, True):
        label = "with density" if density else "no density"
        print(f"Fitting on all five regions ({label})...")
        model, std, hist = fit(tr, hidden=None, physics=True, profile="power",
                               density=density, epochs=args.epochs, seed=0,
                               verbose=False)
        fits[density] = (model, std, hist)
        print(f"  final training loss {hist[-1]:.6f}  rmse {np.sqrt(hist[-1]):.4f}")

    frames = pd.concat([fitted_frame(m, s, tr, d) for d, (m, s, _) in fits.items()],
                       ignore_index=True)
    frames.to_csv(OUT / "e2_fitted_physics.csv", index=False)
    base = frames[~frames.density]

    print(f"\n{'='*100}\n### 1. Fitted physical quantities by region (no density)\n")
    print(base.groupby("region")[["speedup", "delta", "eta", "relief_28km"]]
              .agg(["mean", "std", "min", "max"]).round(3).to_string())
    for d, (m, _, _) in fits.items():
        print(f"  [{'with' if d else 'no '} density] kappa = "
              f"{float(_scalar(m)):.3f}   relief scale = "
              f"{float(torch.exp(m.log_relief_scale)):.0f} m")

    print(f"\n{'='*100}\n### 2. Addendum 1, prediction 1: does the speed-up at altitude fall?\n")
    hi = frames[frames.elevation > 800.0]
    lo = frames[frames.elevation <= 200.0]
    tab = pd.DataFrame({
        "n": [int((~hi.density).sum()), int((~lo.density).sum())],
        "speedup_no_density": [hi[~hi.density].speedup.mean(),
                               lo[~lo.density].speedup.mean()],
        "speedup_with_density": [hi[hi.density].speedup.mean(),
                                 lo[lo.density].speedup.mean()],
    }, index=["elevation > 800 m", "elevation <= 200 m"])
    tab["change"] = tab.speedup_with_density - tab.speedup_no_density
    print(tab.round(4).to_string())
    fell = tab.loc["elevation > 800 m", "change"] < 0
    print(f"\n  prediction 1 ({'CONFIRMED' if fell else 'FAILED'}): the speed-up above "
          f"800 m {'falls' if fell else 'does not fall'} when density is modelled")

    print(f"\n{'='*100}\n### 3. Named probes: speed-up where the terrain is known\n")
    model, std, _ = fits[False]
    prows = []
    for name, (lon, lat) in PROBES.items():
        dist, r, k = nearest_unit(tr, lon, lat)
        idx = torch.tensor([k])
        with torch.no_grad():
            g, d, e, _ = model(std.terrain(r, idx), std.fleet(r, idx), r.relief[idx])
        prows.append(dict(probe=name, region=r.code, dist_deg=round(dist, 2),
                          elev_m=float(r.elevation[k]), relief_m=float(r.relief[k]),
                          speedup=float(torch.exp(g)[0]), delta=float(d[0]),
                          eta=float(e[0])))
    probes = pd.DataFrame(prows)
    print(probes.round(3).to_string(index=False))
    probes.to_csv(OUT / "e2_probes.csv", index=False)

    print(f"\n{'='*100}\n### 4. Speed-up against relief, binned (is the term being used?)\n")
    b = base.copy()
    b["relief_bin"] = pd.cut(b.relief_28km, [0, 50, 100, 200, 400, 800, 1600, 4000])
    print(b.groupby("relief_bin", observed=True)
           .agg(n=("speedup", "size"), speedup=("speedup", "mean"),
                sd=("speedup", "std"), eta=("eta", "mean")).round(3).to_string())

    print(f"\n{'='*100}\n### 5. Physical plausibility\n")
    top = b[b.relief_28km > b.relief_28km.quantile(0.9)]
    bot = b[b.relief_28km < b.relief_28km.quantile(0.1)]
    print(f"  speed-up >= 1 in the top relief decile        {float((top.speedup>=1).mean()):6.1%}")
    print(f"  speed-up within 2% of 1 in the bottom decile  "
          f"{float(((bot.speedup-1).abs()<0.02).mean()):6.1%}")
    print(f"  eta inside the reported band 0.75-0.95        "
          f"{float(b.eta.between(0.75,0.95).mean()):6.1%}")
    print(f"  spearman(relief, speed-up)                    "
          f"{b[['relief_28km','speedup']].corr(method='spearman').iloc[0,1]:+.3f}")

    print(f"\n{'='*100}\n### 6. What is LEFT: residual structure after correction\n")
    resid = []
    for r in tr:
        f = predict_frame(r, model, std).dropna(subset=["cf_sim"])
        f["resid"] = f.cf_sim - f.cf_obs
        f["region"] = r.code
        resid.append(f)
    res = pd.concat(resid, ignore_index=True)
    res.to_csv(OUT / "e2_residuals.csv", index=False)
    print("  by month (mean residual, simulated minus observed):")
    piv = res.pivot_table(index="region", columns="month", values="resid", aggfunc="mean")
    print(piv.round(3).to_string())
    print("\n  seasonal amplitude of the residual (max minus min monthly mean):")
    print((piv.max(axis=1) - piv.min(axis=1)).round(3).to_string())
    print("\n  by region: mean, sd, and share of variance the month explains")
    for code, sub in res.groupby("region"):
        month_means = sub.groupby("month").resid.transform("mean")
        share = float(month_means.var() / sub.resid.var()) if sub.resid.var() > 0 else np.nan
        print(f"    {code}: mean {sub.resid.mean():+.4f}  sd {sub.resid.std():.4f}  "
              f"month explains {share:6.1%} of residual variance")

    print(f"\nwrote {OUT}/e2_*.csv")


def _scalar(model):
    with torch.no_grad():
        _, _, _, k = model(torch.zeros(1, 14), torch.zeros(1, 4), torch.zeros(1))
    return k


if __name__ == "__main__":
    main()
