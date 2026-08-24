#!/usr/bin/env python3
"""D7: where is the error that remains, and what would removing it require?

E1 established that the correction transfers. This asks the next question: of
the error still left after it, how much sits in structure the model could in
principle represent but does not, and how much is in places nothing in the
current inputs can reach?

The residual is sliced along the axes a physical model could plausibly gain on:

  terrain          is the speed-up term under- or over-reaching?
  elevation        is air density being absorbed rather than modelled?
  hub height       is the profile term extrapolating badly at the extremes?
  capacity density is the efficiency term missing a wake effect?
  offshore         is the marine boundary layer being mismodelled?
  month            is there seasonal physics the static terms cannot carry?
  coastal distance is the land-sea transition unrepresented?

A slice where the mean residual is flat carries no recoverable signal; one with
a strong monotone trend is a term the model is missing. That distinction is the
whole point: it separates "add this physics" from "acquire this data".

Fits once on all regions (in-region, so the residual is the model's floor rather
than a transfer penalty) and reports per-region and pooled.

Run: PYVWF_INPUT=input/combined PYTHONPATH=src /opt/anaconda3/bin/python \
         scripts/pinn/d7_residual_anatomy.py
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

from vwf.pinn.terrain import FEATURES as TERRAIN_FEATURES  # noqa: E402
from vwf.pinn.train import (  # noqa: E402
    FLEET_FEATURES, fit, load_regions, predict_frame,
)

# Looked up by name rather than by position: a hardcoded column index would keep
# running, and quietly slice the residual along the wrong variable, if either
# feature list were ever reordered.
TI = {name: i for i, name in enumerate(TERRAIN_FEATURES)}
FI = {name: i for i, name in enumerate(FLEET_FEATURES)}

CACHE = ROOT / "output" / "pinn" / "cache"
OUT = ROOT / "output" / "pinn" / "d7"
REGIONS = ["DK", "DE", "UK", "US", "BR"]

# Slice, and what a trend along it would mean.
SLICES = {
    "relief_28km": ("ERA5-cell relief (m)", [0, 50, 100, 200, 400, 800, 4000]),
    "z_site": ("elevation (m)", [-100, 0, 100, 300, 600, 1000, 4000]),
    "height": ("hub height (m)", [0, 40, 60, 80, 100, 120, 200]),
    "capdens": ("local capacity density (MW/km2)", None),
    "std_84km": ("terrain spread at 84 km (m)", [0, 25, 50, 100, 200, 400, 2000]),
    "land_frac_28km": ("land fraction in the ERA5 cell", [0, 0.2, 0.5, 0.8, 0.99, 1.01]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    ap.add_argument("--density", action="store_true")
    ap.add_argument("--wake", action="store_true")
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)

    tr = load_regions(args.regions, "train", CACHE, quiet=True)
    print(f"Fitting on {'+'.join(args.regions)} (in-region; this is the model's floor)...")
    model, std, hist = fit(tr, hidden=args.hidden or None, physics=True,
                           profile="power", density=args.density,
                           wake=args.wake, epochs=args.epochs, seed=0,
                           verbose=False)
    print(f"  final training loss {hist[-1]:.6f}  (rmse {np.sqrt(hist[-1]):.4f})")

    frames = []
    for r in tr:
        f = predict_frame(r, model, std, density=args.density).dropna(subset=["cf_sim"])
        f["region"] = r.code
        meta = pd.DataFrame({
            "ID": r.ids.astype(str),
            "relief_28km": r.relief.numpy(),
            "z_site": r.elevation.numpy(),
            "height": r.height.numpy(),
            "capdens": 10 ** r.fleet_raw[:, FI["log_capdens_10km"]].numpy(),
            "std_84km": r.terrain_raw[:, TI["std_84km"]].numpy(),
            "land_frac_28km": r.terrain_raw[:, TI["land_frac_28km"]].numpy(),
            "offshore": r.fleet_raw[:, FI["is_offshore"]].numpy(),
        })
        frames.append(f.merge(meta, on="ID", how="left"))
    res = pd.concat(frames, ignore_index=True)
    res["resid"] = res.cf_sim - res.cf_obs
    res.to_csv(OUT / f"d7_residuals_{args.tag}.csv", index=False)

    print(f"\n{'='*96}\n### Residual by region (simulated minus observed)\n")
    print(res.groupby("region").agg(
        n=("resid", "size"), mean=("resid", "mean"), sd=("resid", "std"),
        rmse=("resid", lambda s: float(np.sqrt((s ** 2).mean()))),
    ).round(4).to_string())

    print(f"\n{'='*96}\n### Residual sliced along axes a physical term could reach\n")
    trends = []
    for col, (label, edges) in SLICES.items():
        if col not in res:
            continue
        v = res[col]
        b = (pd.qcut(v, 6, duplicates="drop") if edges is None
             else pd.cut(v, edges))
        g = res.groupby(b, observed=True).agg(
            n=("resid", "size"), mean_resid=("resid", "mean"),
            rmse=("resid", lambda s: float(np.sqrt((s ** 2).mean()))))
        print(f"\n-- {label} --")
        print(g.round(4).to_string())
        span = float(g.mean_resid.max() - g.mean_resid.min())
        trends.append(dict(slice=col, label=label, mean_resid_span=span,
                           overall_sd=float(res.resid.std())))

    print(f"\n{'='*96}\n### Which slices carry recoverable structure?\n")
    t = pd.DataFrame(trends).sort_values("mean_resid_span", ascending=False)
    t["span_over_sd"] = t.mean_resid_span / t.overall_sd
    print(t[["label", "mean_resid_span", "span_over_sd"]].round(4).to_string(index=False))
    print("\n  span_over_sd is the swing in MEAN residual across the slice, relative")
    print("  to the residual's own spread. Large means a systematic term is missing;")
    print("  near zero means that axis is already handled and nothing is to be won.")
    t.to_csv(OUT / f"d7_slice_trends_{args.tag}.csv", index=False)

    print(f"\n{'='*96}\n### Seasonal structure, per region\n")
    piv = res.pivot_table(index="region", columns="month", values="resid", aggfunc="mean")
    print(piv.round(3).to_string())
    print("\n  amplitude (max minus min monthly mean), and the share of residual")
    print("  variance the month explains:")
    for code, sub in res.groupby("region"):
        amp = sub.groupby("month").resid.mean()
        share = float(sub.groupby("month").resid.transform("mean").var() / sub.resid.var())
        print(f"    {code}: amplitude {float(amp.max()-amp.min()):.4f}   "
              f"month explains {share:6.1%}")

    print(f"\n{'='*96}\n### Offshore against onshore\n")
    print(res.groupby("offshore").agg(
        n=("resid", "size"), mean=("resid", "mean"),
        rmse=("resid", lambda s: float(np.sqrt((s ** 2).mean())))).round(4).to_string())

    print(f"\nwrote {OUT}/d7_*.csv")


if __name__ == "__main__":
    main()
