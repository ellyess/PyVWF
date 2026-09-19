#!/usr/bin/env python3
"""ML transfer re-test on five-region, post-fix correction targets.

Re-runs the development-branch ML transfer experiment (leave-one-region-out
prediction of correction factors from terrain + spatial features) against the
post-fix pipeline outputs on this branch. Gates were pre-specified before any
model run; see docs/findings/method-ml-transfer.md.

Requires the train outputs under output/validation/ (train-sweep2/train-sweep3
runs at commit 8a032d6) and input/reference/terrain/etopo_global.nc.

Run: /opt/anaconda3/bin/python scripts/analysis/ml_transfer_retest.py
"""

from pathlib import Path

import pandas as pd

from vwf.extensions.ml import transfer
from vwf.extensions.ml.transfer import (  # noqa: F401  (the drivers in scripts/pinn/ import these from here)
    RF_KW,
    SEEDS,
    SET_A,
    SET_B,
    SET_C,
    loro,
    random_cv,
    rf_eval,
    variance_decomposition,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "ml_retest"

# ---------------------------------------------------------------- dataset ----
RUNS = {  # region -> (train dir, k); canonical post-fix runs, commit 8a032d6
    "DK": ("output/validation/DK/train-sweep3", 100),
    "DE": ("output/validation/DE/train-sweep3", 100),
    "UK": ("output/validation/UK/train-sweep3", 100),
    "US": ("output/validation/US/train-sweep2", 100),
    "BR": ("output/validation/BR/train-sweep2", 60),
}
SENSITIVITY_RUNS = {
    "BR120": {"BR": ("output/validation/BR/train-sweep2", 120)},
    "high-k": {
        "DK": ("output/validation/DK/train-sweep3", 500),
        "DE": ("output/validation/DE/train-sweep3", 500),
        "UK": ("output/validation/UK/train-sweep3", 500),
        "US": ("output/validation/US/train-sweep2", 300),
        "BR": ("output/validation/BR/train-sweep2", 120),
    },
}


def build_centroids(runs: dict) -> pd.DataFrame:
    """:func:`vwf.extensions.ml.transfer.build_centroids`, rooted at ``ROOT``.

    ``ROOT`` is read when called, so a caller that repoints it (as
    ml_transfer_expanded.py does) is honoured.
    """
    return transfer.build_centroids(runs, ROOT)


def terrain_features(df: pd.DataFrame) -> pd.DataFrame:
    """:func:`vwf.extensions.ml.transfer.terrain_features` on the ETOPO grid
    under ``ROOT``."""
    return transfer.terrain_features(df, ROOT / "input/reference/terrain/etopo_global.nc")


def run_suite(df, label):
    print(f"\n{'=' * 70}\n### {label}  (n={len(df)})")
    # normalise lon/lat over the pooled dataset (cosmetic for trees)
    for c in ("lon", "lat"):
        df[f"{c}_norm"] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
    df = terrain_features(df)
    print("\nPer-region target stats:")
    print(
        df.groupby("region")[["scalar", "offset"]]
        .agg(["mean", "std", "count"])
        .round(3)
        .to_string()
    )
    for target in ("scalar", "offset"):
        vb = variance_decomposition(df, target)
        print(f"\nT5 {target}: between-region variance share = {vb:.1%}")
    results = {}
    for name, feats in [("T1 SetA", SET_A), ("T2 SetB", SET_B), ("T3 SetC", SET_C)]:
        for target in ("scalar", "offset"):
            r = loro(df, feats, target)
            results[(name, target)] = r
            print(f"\n{name} LORO [{target}]:")
            print(r.round(3).to_string(index=False))
    for target in ("scalar", "offset"):
        m, sd, mae = random_cv(df, SET_A, target)
        print(f"\nT4 random 5-fold CV SetA [{target}]: R2 = {m:.3f} ± {sd:.3f}, MAE = {mae:.3f}")
    # gate check on T1 scalar
    t1 = results[("T1 SetA", "scalar")]
    n_pos = int((t1.r2_mean > 0).sum())
    print(
        f"\nGATE (T1 scalar, primary set only): {n_pos}/5 regions R2>0 -> "
        f"{'POSITIVE' if n_pos >= 3 else 'NEGATIVE result stands'}"
    )
    return df, results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_centroids(RUNS)
    df, _ = run_suite(df, "PRIMARY (DK/DE/UK/US k=100, BR k=60)")
    df.to_csv(OUT / "ml_retest_centroids_primary.csv", index=False)

    for name, override in SENSITIVITY_RUNS.items():
        runs = {**RUNS, **override}
        d2 = build_centroids(runs)
        run_suite(d2, f"SENSITIVITY {name}")


if __name__ == "__main__":
    main()
