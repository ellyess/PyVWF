#!/usr/bin/env python3
"""D8: what does Brazil's uniform 100 m hub-height default actually cost?

Every Brazilian complex in `br_md.csv` carries `height = 100.0` with
`height_source = "default-uniform"`, because ONS publishes no hub height and no
turbine model. That is documented and honest, but it has a consequence nobody
had measured: at exactly the reference height of the ERA5 wind field, EVERY
profile form returns a ratio of exactly 1, so Brazil cannot distinguish one
hub-height profile from another. It is invisible to the whole profile question.

Building the missing data means a per-complex turbine-spec table of the kind
Chile and Argentina already have (`configs/curation/*_turbine_specs.csv`,
60 and 65 rows, each with a cited source), for 193 complexes. Before spending
that, this measures what it would buy, in two parts:

  LEVEL        refit at a uniform 80, 90, 100, 110, 120 m. How much does the
               ASSUMED level move the answer?
  HETEROGENEITY refit with heights drawn around 100 m with realistic spread.
               How much does the assumption of UNIFORMITY cost, separately from
               the level? This is the part the uniform default hides.

The heterogeneity arm does not pretend to know Brazil's real heights. It asks
what happens if they are spread as a modern fleet's are, which bounds the error
the uniform default can be making.

Run: PYVWF_INPUT=input/combined PYTHONPATH=src /opt/anaconda3/bin/python \
         scripts/pinn/d8_hub_height_sensitivity.py
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

import vwf.pinn.train as T  # noqa: E402
from vwf.harness.skill import skill_metrics  # noqa: E402

CACHE = ROOT / "output" / "pinn" / "cache"
OUT = ROOT / "output" / "pinn" / "d8"
LEVELS = (80.0, 90.0, 100.0, 110.0, 120.0)
# Spread of a modern onshore fleet: most machines 85-125 m, a tail either way.
HET_MEAN, HET_SD, HET_LO, HET_HI = 100.0, 15.0, 70.0, 140.0


def scored(tr, te, profile, epochs, seed=0):
    m, std, hist = T.fit(tr, hidden=16, physics=True, profile=profile,
                         epochs=epochs, seed=seed, verbose=False)
    f = T.predict_frame(te, m, std, profile=profile).dropna(subset=["cf_sim"])
    return skill_metrics(f), hist[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--region", default="BR")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    tr = T.load_regions([args.region], "train", CACHE, quiet=True)
    te = T.load_regions([args.region], "test", CACHE, quiet=True)[0]
    h0_tr, h0_te = tr[0].height.clone(), te.height.clone()
    print(f"{args.region}: {tr[0].n_units} train / {te.n_units} test units, "
          f"heights {float(h0_tr.min()):.0f}-{float(h0_tr.max()):.0f} m")

    rows = []
    print(f"\n{'='*88}\n### LEVEL: uniform height, swept\n")
    for h in LEVELS:
        tr[0].height = torch.full_like(h0_tr, h)
        te.height = torch.full_like(h0_te, h)
        for profile in ("power", "shear-log"):
            k, loss = scored(tr, te, profile, args.epochs)
            rows.append(dict(arm="level", height=h, spread=0.0, profile=profile,
                             seed=0, rmse=k["rmse"], mbe=k["mbe"], loss=loss))
        r = {x["profile"]: x["rmse"] for x in rows if x["height"] == h and x["arm"] == "level"}
        print(f"  {h:5.0f} m   power {r['power']:.4f}   shear-log {r['shear-log']:.4f}"
              f"   difference {r['shear-log']-r['power']:+.4f}")

    print(f"\n{'='*88}\n### HETEROGENEITY: heights spread around 100 m\n")
    for seed in args.seeds:
        g = torch.Generator().manual_seed(seed)
        draw = lambda n: torch.normal(HET_MEAN, HET_SD, (n,), generator=g).clamp(HET_LO, HET_HI)  # noqa: E731
        tr[0].height = draw(len(h0_tr))
        te.height = draw(len(h0_te))
        sd = float(tr[0].height.std())
        for profile in ("power", "shear-log"):
            k, loss = scored(tr, te, profile, args.epochs, seed=seed)
            rows.append(dict(arm="heterogeneous", height=HET_MEAN, spread=sd,
                             profile=profile, seed=seed, rmse=k["rmse"],
                             mbe=k["mbe"], loss=loss))
        r = {x["profile"]: x["rmse"] for x in rows
             if x["arm"] == "heterogeneous" and x["seed"] == seed}
        print(f"  seed {seed}  sd {sd:4.1f} m   power {r['power']:.4f}   "
              f"shear-log {r['shear-log']:.4f}   difference {r['shear-log']-r['power']:+.4f}")

    tr[0].height, te.height = h0_tr, h0_te
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"d8_{args.region}_height_sensitivity.csv", index=False)

    print(f"\n{'='*88}\n### What the default costs\n")
    lvl = res[res.arm == "level"]
    base = float(lvl[(lvl.height == 100) & (lvl.profile == "power")].rmse.iloc[0])
    print(f"  RMSE at the documented default (100 m, power):        {base:.4f}")
    print(f"  spread across assumed levels 80-120 m:                "
          f"{lvl[lvl.profile=='power'].rmse.max()-lvl[lvl.profile=='power'].rmse.min():.4f}")
    het = res[(res.arm == "heterogeneous") & (res.profile == "power")]
    print(f"  RMSE with realistic height spread (mean over seeds):  {het.rmse.mean():.4f}"
          f"  (sd {het.rmse.std():.4f})")
    print(f"  cost of assuming uniformity:                          "
          f"{het.rmse.mean()-base:+.4f}")
    print()
    print("  Can Brazil tell the two profile forms apart?")
    for arm in ("level", "heterogeneous"):
        sub = res[res.arm == arm]
        d = (sub[sub.profile == "shear-log"].rmse.values
             - sub[sub.profile == "power"].rmse.values)
        print(f"    {arm:14s} mean |power - shear-log| = {np.abs(d).mean():.5f}")
    print(f"\nwrote {OUT}/d8_{args.region}_height_sensitivity.csv")


if __name__ == "__main__":
    main()
