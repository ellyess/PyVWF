#!/usr/bin/env python3
"""D0: is leave-one-region-out R2 the decision-relevant transfer metric?

The prior experiment (docs/findings/ml_transfer_retest.md) scored cross-region
transfer as sklearn ``r2_score`` on the per-cluster scalar. That denominator is
the HOLDOUT's own mean -- a quantity nobody has when they arrive in an unseen
region. The decision a practitioner actually faces is "apply a transferred
correction, or apply none", so the baseline that matters is the identity
correction (scalar = 1, offset = 0).

This script re-scores the SAME targets and the SAME model under both
denominators, and adds the trivially transferable predictor the prior
experiment never scored: the pooled training-region mean.

Diagnostic only -- no model is tuned here and no gate is claimed. It measures
existing artefacts to decide which metric later gates should be written in.

Run: PYTHONPATH=src /opt/anaconda3/bin/python scripts/pinn/d0_metric_reframe.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.ml_transfer_retest import (  # noqa: E402
    RUNS, SEEDS, SET_A, build_centroids, terrain_features, RF_KW,
)
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pinn" / "d0"

# The identity correction: what you apply when you decline to correct at all.
IDENTITY = {"scalar": 1.0, "offset": 0.0}


def mse(y, yhat):
    return float(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_centroids(RUNS)
    for c in ("lon", "lat"):
        df[f"{c}_norm"] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
    df = terrain_features(df)

    rows = []
    for target in ("scalar", "offset"):
        ident = IDENTITY[target]
        for region in sorted(df.region.unique()):
            tr = df[df.region != region]
            te = df[df.region == region]
            y = te[target].to_numpy()

            preds = np.column_stack([
                RandomForestRegressor(random_state=s, **RF_KW)
                .fit(tr[SET_A], tr[target]).predict(te[SET_A])
                for s in SEEDS
            ])
            mse_ml = float(np.mean([mse(y, preds[:, i])
                                    for i in range(preds.shape[1])]))

            mse_ident = mse(y, np.full_like(y, ident))
            mse_pooled = mse(y, np.full_like(y, tr[target].mean()))
            mse_oracle = mse(y, np.full_like(y, y.mean()))  # R2 denominator

            rows.append(dict(
                target=target, holdout=region, n=len(te),
                y_mean=y.mean(), y_std=y.std(),
                # published metric: 1 - mse_ml/mse_oracle
                r2_vs_holdout_mean=1 - mse_ml / mse_oracle,
                # decision-relevant: skill against declining to correct
                skill_ml_vs_identity=1 - mse_ml / mse_ident,
                skill_pooledmean_vs_identity=1 - mse_pooled / mse_ident,
                skill_oraclemean_vs_identity=1 - mse_oracle / mse_ident,
                rmse_identity=np.sqrt(mse_ident),
                rmse_pooled=np.sqrt(mse_pooled),
                rmse_ml=np.sqrt(mse_ml),
            ))

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "d0_metric_reframe.csv", index=False)

    pd.set_option("display.width", 200)
    for target in ("scalar", "offset"):
        sub = res[res.target == target]
        print(f"\n{'='*92}\n### target = {target}   (identity = {IDENTITY[target]})\n")
        print(sub[[
            "holdout", "n", "y_mean", "y_std",
            "r2_vs_holdout_mean",
            "skill_ml_vs_identity", "skill_pooledmean_vs_identity",
            "skill_oraclemean_vs_identity",
        ]].round(3).to_string(index=False))
        print(f"\n  RMSE (lower better):")
        print(sub[["holdout", "rmse_identity", "rmse_pooled", "rmse_ml"]]
              .round(3).to_string(index=False))
        n_pos_pub = int((sub.r2_vs_holdout_mean > 0).sum())
        n_pos_ml = int((sub.skill_ml_vs_identity > 0).sum())
        n_pos_pool = int((sub.skill_pooledmean_vs_identity > 0).sum())
        print(f"\n  regions with R2 vs holdout mean   > 0 : {n_pos_pub}/5  (published gate)")
        print(f"  regions with ML skill vs identity > 0 : {n_pos_ml}/5")
        print(f"  regions with pooled-mean skill    > 0 : {n_pos_pool}/5  (zero-ML transfer)")

    print(f"\nwrote {OUT/'d0_metric_reframe.csv'}")


if __name__ == "__main__":
    main()
