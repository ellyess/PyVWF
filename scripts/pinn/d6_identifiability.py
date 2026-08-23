#!/usr/bin/env python3
"""D6: are the conversion efficiency and the terrain speed-up separable?

Both terms reduce output, and over a narrow wind regime they can trade off
almost freely: a bigger speed-up with a lower efficiency predicts nearly the
same capacity factor as a smaller speed-up with a higher one. If the split is
not pinned down, the fitted numbers cannot be read as physics, and worse, the
terrain term will carry some of the loss into regions whose loss regime differs.

What breaks the degeneracy is FLAT ground. The speed-up is structurally zero
where there is no sub-grid relief, so a flat site forces the efficiency to
account for the whole correction on its own; a fit with flat sites in it has an
anchor that a fit without them does not.

The probe: start the efficiency at four different values, fit everything else
identically, and see where each lands. Convergence to the same place means the
data pins it; divergence at equal loss means it does not.

Run: PYTHONPATH=src /opt/anaconda3/bin/python scripts/pinn/d6_identifiability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import vwf.pinn.train as T  # noqa: E402
from vwf.harness.skill import skill_metrics  # noqa: E402
from vwf.pinn.model import ETA_BOUNDS, PhysicsCorrection, _preactivation_for  # noqa: E402

CACHE = ROOT / "output" / "pinn" / "cache"
OUT = ROOT / "output" / "pinn" / "d6"
STARTS = (0.70, 0.80, 0.90, 0.98)
# BR alone is the complex-terrain case with almost no flat ground; adding
# Denmark, whose median ERA5-cell relief is 78 m against Brazil's 362 m,
# supplies the anchor.
SETUPS = {
    "BR only (little flat ground)": ["BR"],
    "BR + DK (flat ground added)": ["BR", "DK"],
}


def fit_from(tr, start: float, epochs: int = 60, seed: int = 0):
    torch.manual_seed(seed)
    std = T.Standardiser.fit(tr)
    m = PhysicsCorrection(14, 4, hidden=None, physics=True)
    with torch.no_grad():
        m.eta.net.bias.fill_(_preactivation_for(start, *ETA_BOUNDS))
    opt = torch.optim.Adam(m.parameters(), lr=0.05, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    quad = T.gauss_hermite(T.N_QUAD)
    gen = torch.Generator().manual_seed(seed)
    loss = None
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.stack([
            T.region_loss(r, m, std, profile="power", quad=quad, generator=gen)
            for r in tr
        ]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        opt.step()
        sched.step()
    return m, std, float(loss.detach())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    rows = []
    for label, codes in SETUPS.items():
        tr = T.load_regions(codes, "train", CACHE, quiet=True)
        te = T.load_regions(["BR"], "test", CACHE, quiet=True)[0]
        print(f"\n=== {label} ===")
        for start in STARTS:
            m, std, loss = fit_from(tr, start)
            rep = m.report(std.terrain(tr[0]), std.fleet(tr[0]), tr[0].relief)
            k = skill_metrics(T.predict_frame(te, m, std).dropna(subset=["cf_sim"]))
            rows.append(dict(setup=label, eta_start=start, eta=rep["eta_mean"],
                             speedup=rep["speedup_mean"], delta=rep["delta_mean"],
                             train_loss=loss, br_rmse=k["rmse"]))
            print(f"  eta start {start:.2f} -> eta {rep['eta_mean']:.4f}  "
                  f"speedup {rep['speedup_mean']:.4f}  loss {loss:.6f}  "
                  f"BR RMSE {k['rmse']:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "d6_identifiability.csv", index=False)
    print(f"\n{'='*92}\n### Spread across starting points (smaller = better pinned)\n")
    spread = res.groupby("setup").agg(
        eta_range=("eta", lambda s: s.max() - s.min()),
        speedup_range=("speedup", lambda s: s.max() - s.min()),
        loss_range=("train_loss", lambda s: s.max() - s.min()),
        rmse_range=("br_rmse", lambda s: s.max() - s.min()),
    )
    print(spread.round(5).to_string())
    print("\n  A large eta/speedup range at a small loss range is the signature of")
    print("  a flat direction: the prediction is pinned, the decomposition is not.")
    print(f"\nwrote {OUT}/d6_identifiability.csv")


if __name__ == "__main__":
    main()
