#!/usr/bin/env python3
"""D2: the affine correction is rank-1, and that breaks the ML transfer.

D1 showed pearson(scalar, offset) = -0.99 in DE/DK/UK/US and -0.84 in BR. A
correlation that tight means the fitted pair (a, b) in ``w' = a*w + b`` is not
two independent numbers: it lies on a ridge, and only the combination that the
observed capacity factor actually pins down is determined. The prior ML
experiment predicted a and b as SEPARATE targets, so its two prediction errors
add along the ridge instead of cancelling.

This script:
  1. measures the ridge per region, recovering the pivot speed w_p from the
     regression of offset on scalar (slope = -w_p);
  2. reparameterises to (w_p-pivot level ``m = a*w_p + b``, slope ``a``) and
     reports how orthogonal the new pair is;
  3. re-runs the leave-one-region-out transfer test on the reparameterised
     targets, scored BOTH ways (R2 vs holdout mean, and skill against the
     identity correction), so the comparison to the published numbers is
     like-for-like;
  4. checks how much of the correction the level alone carries, by scoring the
     level-only correction (slope pinned to 1) against the full affine pair in
     wind-speed space.

Diagnostic. No gate is claimed here; gates for the physics-informed model are
pre-specified separately in docs/findings/method-physics-informed-prespecification.md.

Run: PYTHONPATH=src /opt/anaconda3/bin/python scripts/pinn/d2_reparameterise.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.ml_transfer_retest import (  # noqa: E402
    RUNS, SEEDS, SET_A, RF_KW, terrain_features,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pinn" / "d2"
D1 = ROOT / "output" / "pinn" / "d1" / "d1_subgrid.csv"


def ridge_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Regress offset on scalar per region; the negated slope is the pivot speed."""
    rows = []
    for region, s in df.groupby("region"):
        res = stats.linregress(s["scalar"], s["offset"])
        rows.append(dict(region=region, n=len(s),
                         pivot_w=-res.slope, intercept=res.intercept,
                         r=res.rvalue, r2=res.rvalue ** 2))
    return pd.DataFrame(rows)


def mse(y, yhat):
    return float(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2))


def loro_both_metrics(df, feats, target, identity):
    """LORO RF, scored as published R2 and as skill vs the identity correction."""
    rows = []
    for region in sorted(df.region.unique()):
        tr, te = df[df.region != region], df[df.region == region]
        y = te[target].to_numpy()
        preds = np.column_stack([
            RandomForestRegressor(random_state=s, **RF_KW)
            .fit(tr[feats], tr[target]).predict(te[feats]) for s in SEEDS
        ])
        m_ml = float(np.mean([mse(y, preds[:, i]) for i in range(preds.shape[1])]))
        m_oracle = mse(y, np.full_like(y, y.mean()))
        m_ident = mse(y, np.full_like(y, identity))
        rows.append(dict(holdout=region, n=len(te),
                         r2_vs_holdout_mean=1 - m_ml / m_oracle,
                         skill_vs_identity=1 - m_ml / m_ident,
                         rmse_ml=np.sqrt(m_ml), rmse_identity=np.sqrt(m_ident)))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(D1)
    for c in ("lon", "lat"):
        df[f"{c}_norm"] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
    df = terrain_features(df)   # the published SET_A features
    pd.set_option("display.width", 210)

    # ---------------------------------------------------------------- 1 ----
    piv = ridge_pivot(df)
    print(f"{'='*94}\n### 1. The (scalar, offset) ridge: offset = c - w_p * scalar\n")
    print(piv.round(3).to_string(index=False))
    w_p = float(piv["pivot_w"].median())
    print(f"\n  median pivot speed across regions: w_p = {w_p:.2f} m/s")
    print("  (a pivot in the 6-10 m/s band is the steep part of a power curve,")
    print("   i.e. the speed at which matching the observed CF is most binding)")

    # ---------------------------------------------------------------- 2 ----
    # Reparameterise onto a SINGLE global pivot so the new target is defined
    # identically in every region (a per-region pivot would leak region identity).
    df["level"] = df["scalar"] * w_p + df["offset"]      # corrected speed at w_p
    df["gain"] = df["scalar"]                             # slope, unchanged
    print(f"\n{'='*94}\n### 2. Orthogonality after reparameterising at w_p = {w_p:.2f} m/s\n")
    rows = []
    for region, s in df.groupby("region"):
        rows.append(dict(
            region=region,
            r_scalar_offset=stats.pearsonr(s.scalar, s.offset)[0],
            r_gain_level=stats.pearsonr(s.gain, s.level)[0],
            level_mean=s.level.mean(), level_std=s.level.std(),
            gain_mean=s.gain.mean(), gain_std=s.gain.std(),
        ))
    orth = pd.DataFrame(rows)
    print(orth.round(3).to_string(index=False))
    print(f"\n  pooled r(scalar,offset) = {stats.pearsonr(df.scalar, df.offset)[0]:+.3f}"
          f"   pooled r(gain,level) = {stats.pearsonr(df.gain, df.level)[0]:+.3f}")

    # ---------------------------------------------------------------- 3 ----
    print(f"\n{'='*94}\n### 3. LORO transfer on old vs new targets (same RF, same seeds)\n")
    # identity correction: scalar 1, offset 0  ->  level = w_p, gain = 1
    specs = [("scalar", 1.0), ("offset", 0.0), ("level", w_p), ("gain", 1.0)]
    all_res = []
    for target, ident in specs:
        r = loro_both_metrics(df, SET_A, target, ident)
        r.insert(0, "target", target)
        all_res.append(r)
        print(f"\n-- target = {target}  (identity = {ident:.3f}) --")
        print(r.round(3).to_string(index=False))
        print(f"   R2>0 in {int((r.r2_vs_holdout_mean>0).sum())}/5   |   "
              f"skill-vs-identity>0 in {int((r.skill_vs_identity>0).sum())}/5")
    res = pd.concat(all_res, ignore_index=True)
    res.to_csv(OUT / "d2_loro_reparameterised.csv", index=False)

    # ---------------------------------------------------------------- 4 ----
    # The published experiment predicted scalar and offset as two free targets.
    # Because they sit on a ridge, the constrained alternative is to predict the
    # scalar only and READ the offset off the ridge: b = c - w_p * a. Score both
    # in wind-speed space, which is where the correction is actually applied.
    print(f"\n{'='*94}\n### 4. Free vs ridge-constrained offset, scored in wind-speed space\n")
    ws = np.arange(4.0, 20.1, 0.5)   # above cut-in: the speeds that make power
    rows = []
    for region in sorted(df.region.unique()):
        tr, te = df[df.region != region], df[df.region == region]
        # ridge constants are fitted on the TRAINING regions only (no leakage)
        rr = stats.linregress(tr["scalar"], tr["offset"])
        a_true = te["scalar"].to_numpy()[:, None]
        b_true = te["offset"].to_numpy()[:, None]
        w_true = a_true * ws[None, :] + b_true

        pa = np.mean([RandomForestRegressor(random_state=s_, **RF_KW)
                      .fit(tr[SET_A], tr["scalar"]).predict(te[SET_A])
                      for s_ in SEEDS], axis=0)[:, None]
        pb = np.mean([RandomForestRegressor(random_state=s_, **RF_KW)
                      .fit(tr[SET_A], tr["offset"]).predict(te[SET_A])
                      for s_ in SEEDS], axis=0)[:, None]

        w_free = pa * ws[None, :] + pb                      # as published
        w_ridge = pa * ws[None, :] + (rr.intercept + rr.slope * pa)
        w_none = np.broadcast_to(ws[None, :], w_true.shape)  # no correction

        def r(x):
            return float(np.sqrt(np.mean((w_true - x) ** 2)))
        rows.append(dict(
            holdout=region,
            rmse_uncorrected=r(w_none),
            rmse_pred_free=r(w_free),
            rmse_pred_ridge=r(w_ridge),
            skill_free=1 - r(w_free) ** 2 / r(w_none) ** 2,
            skill_ridge=1 - r(w_ridge) ** 2 / r(w_none) ** 2,
        ))
    wsp = pd.DataFrame(rows)
    print(wsp.round(3).to_string(index=False))
    print(f"\n  skill = 1 - MSE/MSE_uncorrected in m/s over 4-20 m/s;")
    print(f"  positive means the transferred correction beats leaving ERA5 alone.")
    print(f"  free  > 0 in {int((wsp.skill_free>0).sum())}/5 regions")
    print(f"  ridge > 0 in {int((wsp.skill_ridge>0).sum())}/5 regions")
    wsp.to_csv(OUT / "d2_windspace.csv", index=False)

    df.to_csv(OUT / "d2_targets.csv", index=False)
    print(f"\nwrote {OUT}/d2_*.csv")


if __name__ == "__main__":
    main()
