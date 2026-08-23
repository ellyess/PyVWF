#!/usr/bin/env python3
"""D5: can transfer work at all? How much of each region's physiography the others cover.

`ml_transfer_retest.md` concluded that the binding constraint on transfer is
regime coverage rather than sample count: Brazil transferred once anything like
it existed in training, and the United States failed hardest because nothing in
a European training fleet resembles a mountain pass. That was an inference from
which regions failed. It is directly measurable.

For each region held out, this asks what fraction of its units sit inside the
physiographic envelope the other four span. Two readings, because they answer
different questions:

  per-feature   the share of holdout units whose value falls inside the
                training regions' 1st-99th percentile range, feature by feature.
                Answers "is this variable in range at all".
  joint         the share inside the training envelope on EVERY feature at once,
                and a nearest-neighbour distance in standardised feature space.
                Answers "has the training set seen a site like this one".

A method cannot be blamed for failing where the answer is near zero, and a
method that succeeds there deserves more credit than one that does not have to.

Measurement only: no model, no gate.

Run: PYTHONPATH=src /opt/anaconda3/bin/python scripts/pinn/d5_regime_coverage.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from vwf.pinn.terrain import FEATURES as TERRAIN_FEATURES  # noqa: E402

CACHE = ROOT / "output" / "pinn" / "cache"
OUT = ROOT / "output" / "pinn" / "d5"
REGIONS = ["DK", "DE", "UK", "US", "BR"]
KEY = ["relief_28km", "std_84km", "tpi_28km", "z_site", "land_frac_28km",
       "std_28km", "relief_84km", "relief_5km"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    ap.add_argument("--tag", default="five")
    args = ap.parse_args()
    regions = args.regions
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {c: pd.read_csv(CACHE / f"{c}_train" / "meta.csv", dtype={"ID": str})
            for c in regions}
    test = {c: pd.read_csv(CACHE / f"{c}_test" / "meta.csv", dtype={"ID": str})
            for c in regions}
    pd.set_option("display.width", 220)

    print(f"{'='*104}\n### Physiography by region (training fleets)\n")
    summary = pd.DataFrame({
        c: meta[c][KEY].median() for c in regions
    }).T
    summary.insert(0, "n_units", [len(meta[c]) for c in regions])
    summary.insert(1, "n_unique_loc",
                   [meta[c][["lon", "lat"]].drop_duplicates().shape[0] for c in regions])
    print(summary.round(2).to_string())
    print("\n  n_unique_loc matters: German rows are postcode centroids and British")
    print("  rows are farm generation split across turbines, so the number of")
    print("  INDEPENDENT physiographic samples is far below the row count.")

    rows, joint_rows = [], []
    for holdout in regions:
        train = pd.concat([meta[c] for c in regions if c != holdout])
        te = test[holdout]
        inside_all = np.ones(len(te), dtype=bool)
        for f in TERRAIN_FEATURES:
            lo, hi = np.percentile(train[f], [1, 99])
            inside = te[f].between(lo, hi).to_numpy()
            inside_all &= inside
            rows.append(dict(holdout=holdout, feature=f, covered=float(inside.mean())))

        # Nearest-neighbour distance in standardised feature space, training
        # statistics only -- the same standardisation the model sees.
        mu, sd = train[list(TERRAIN_FEATURES)].mean(), train[list(TERRAIN_FEATURES)].std()
        A = ((train[list(TERRAIN_FEATURES)] - mu) / sd).to_numpy()
        B = ((te[list(TERRAIN_FEATURES)] - mu) / sd).to_numpy()
        # chunked to keep the pairwise distance matrix small
        nn = np.empty(len(B))
        for i in range(0, len(B), 512):
            d = np.linalg.norm(B[i:i + 512, None, :] - A[None, :, :], axis=2)
            nn[i:i + 512] = d.min(axis=1)
        joint_rows.append(dict(holdout=holdout, n_test=len(te),
                               joint_covered=float(inside_all.mean()),
                               nn_median=float(np.median(nn)),
                               nn_p90=float(np.percentile(nn, 90)),
                               nn_max=float(nn.max())))

    cov = pd.DataFrame(rows)
    joint = pd.DataFrame(joint_rows)
    cov.to_csv(OUT / f"d5_per_feature_{args.tag}.csv", index=False)
    joint.to_csv(OUT / f"d5_joint_{args.tag}.csv", index=False)

    print(f"\n{'='*104}\n### Per-feature coverage: share of holdout units inside the "
          f"training 1st-99th percentile range\n")
    piv = cov.pivot(index="feature", columns="holdout", values="covered")
    piv = piv.loc[[f for f in TERRAIN_FEATURES]]
    print(piv.round(3).to_string())
    print("\n  worst-covered feature per holdout:")
    for h in regions:
        f = piv[h].idxmin()
        print(f"    {h}: {f} ({piv.loc[f, h]:.1%} of units in range)")

    print(f"\n{'='*104}\n### Joint coverage and distance to the nearest training site\n")
    print(joint.round(3).to_string(index=False))
    print("\n  joint_covered = share of holdout units inside the training range on")
    print("  EVERY terrain feature at once. nn_* are distances in standardised")
    print("  feature space to the closest training unit (0 = an identical site).")

    print(f"\nwrote {OUT}/d5_*_{args.tag}.csv")


if __name__ == "__main__":
    main()
