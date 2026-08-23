#!/usr/bin/env python3
"""Assemble the E1 comparison table and score the pre-specified gates.

Reads the raw per-seed rows E1 writes and joins them to the INCUMBENT affine
numbers held in the cluster-sweep runs, so every reference point in the table is
read from a stored artefact rather than quoted from prose.

Two incumbent columns, because the choice matters and neither alone is fair:

  affine (best k)  the lowest held-out RMSE over the whole k and time-slice
                   sweep. This is what the validation scorecard reports, and it
                   is optimistic: the configuration is chosen on the test year.
  affine (config)  the configuration the region's TOML actually names as its
                   sweep knee -- what a user of the repository would get.

Run: PYTHONPATH=src /opt/anaconda3/bin/python scripts/pinn/e1_report.py --tag primary
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from vwf.harness.regions import load_region  # noqa: E402

E1 = ROOT / "output" / "pinn" / "e1"
SWEEP = ROOT / "output" / "validation" / "cluster_sweep_2026-07-24"
REGIONS = ["DK", "DE", "UK", "US", "BR"]


def incumbent(code: str) -> dict:
    """Best-of-sweep and configured-k affine RMSE for one region."""
    d = pd.read_csv(SWEEP / f"{code}_metrics.csv")
    if "scope" in d:
        d = d[d.scope == "fleet"]
    aff = d[d.variant == "affine-wind"]
    unc = d[d.variant == "uncorrected"]
    spec = load_region(ROOT / "configs" / "regions" / f"{code.lower()}.toml")
    k = max(spec.cluster_list)
    at_k = aff[aff.num_clu == k]
    best = aff.loc[aff.rmse.idxmin()]
    return {
        "incumbent_uncorrected": float(unc.rmse.iloc[0]) if len(unc) else np.nan,
        "affine_best": float(best.rmse),
        "affine_best_cfg": f"k{int(best.num_clu)} {best.time_res}",
        "affine_config_k": float(at_k.rmse.min()) if len(at_k) else np.nan,
        "config_k": k,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="primary")
    args = ap.parse_args()
    raw = pd.read_csv(E1 / f"e1_{args.tag}_raw.csv")
    pd.set_option("display.width", 230)

    agg = (raw.groupby(["holdout", "arm"])
              .agg(rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
                   mbe=("mbe", "mean"), r=("pearson_r", "mean"),
                   n=("rmse", "size")).reset_index())
    piv = agg.pivot(index="holdout", columns="arm", values="rmse")
    order = [c for c in ("uncorrected", "rf-transfer", "pinn", "pinn-ablation",
                         "pinn-in-region") if c in piv.columns]
    piv = piv[order]

    inc = pd.DataFrame({c: incumbent(c) for c in REGIONS if c in piv.index}).T
    table = piv.join(inc[["affine_best", "affine_config_k", "affine_best_cfg"]])

    print(f"{'='*112}\n### Capacity-factor RMSE on the held-out test year "
          f"(capacity-weighted, harness skill_metrics)\n")
    print(table.round(4).to_string())
    print("\n  pinn / pinn-ablation / pinn-in-region are means over seeds; "
          "spreads below.")
    print("  affine_* are in-region fits, shown as reference points, never as "
          "transfer arms.")

    print(f"\n### Seed spread (sd of RMSE across seeds)\n")
    sd = agg.pivot(index="holdout", columns="arm", values="rmse_sd")
    print(sd[[c for c in order if c in sd.columns]].round(4).to_string())

    print(f"\n### Mean bias error\n")
    mb = agg.pivot(index="holdout", columns="arm", values="mbe")
    print(mb[[c for c in order if c in mb.columns]].round(4).to_string())

    base = piv["uncorrected"]
    skill = 1 - piv.div(base, axis=0) ** 2
    print(f"\n### Skill against uncorrected ERA5  (1 - MSE/MSE_uncorrected)\n")
    print(skill.round(3).to_string())

    print(f"\n{'='*112}\n### Pre-specified gates\n")
    ok1a = (piv["pinn"] < base)
    ok1b = (piv["pinn"] > base * 1.10)
    p1 = bool(ok1a.sum() >= 3 and not ok1b.any())
    print(f"P1  zero-shot beats uncorrected in >=3/5, and degrades none by >10%")
    print(f"      beats uncorrected in {int(ok1a.sum())}/5: "
          f"{', '.join(sorted(piv.index[ok1a])) or 'none'}")
    print(f"      degraded by >10% : "
          f"{', '.join(sorted(piv.index[ok1b])) or 'none'}")
    print(f"      -> P1 {'PASS' if p1 else 'FAIL'}")

    if "rf-transfer" in piv:
        ok2 = piv["pinn"] < piv["rf-transfer"]
        print(f"\nP2  beats the incumbent RF transfer in >=3/5")
        print(f"      beats rf-transfer in {int(ok2.sum())}/5: "
              f"{', '.join(sorted(piv.index[ok2])) or 'none'}")
        print(f"      -> P2 {'PASS' if ok2.sum() >= 3 else 'FAIL'}")

    if "pinn-ablation" in piv:
        d = piv["pinn-ablation"] - piv["pinn"]
        print(f"\nP3  does the physics earn its place? (ablation RMSE minus pinn RMSE;"
              f" positive = constraints help)")
        print(d.round(4).to_string())
        print(f"      physics better in {int((d > 0).sum())}/5 regions")

    table.to_csv(E1 / f"e1_{args.tag}_table.csv")
    print(f"\nwrote {E1}/e1_{args.tag}_table.csv")


if __name__ == "__main__":
    main()
