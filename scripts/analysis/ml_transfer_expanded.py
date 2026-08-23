"""ML transfer re-test with expanded regime coverage (NZ/CL/AR added).

The 2026 re-test (`ml_transfer_retest.py`) found leave-one-region-out (LORO)
transfer negative (1/5), and traced the one catastrophic case, US (R2 -1.05),
to the mountain-pass regime (Tehachapi, Altamont, central Washington) that the
Europe-only training pool contained no analogue of: "transfer fails hardest
exactly where the training set has no coverage of the physical regime."

This session added New Zealand (Cook Strait / Manawatu funnelling), Chile
(Atacama coast, Andean foothills) and Argentina (Cuyo / La Rioja) -- the same
class of complex-terrain, ERA5-under-resolves-the-ridge-wind regime. This
re-runs LORO on the original 5 regions and on the expanded 8, so the effect of
adding those analogues is read directly.

PRE-SPECIFIED GATES (written before any model run):

  G1 (primary headline). US-holdout scalar R2 (Set A, LORO). Baseline is the
     5-region value. "Materially rescued" if the 8-region value rises above
     -0.30; "transfers" if it rises above 0.

  G2 (majority). Scalar R2 (Set A) > 0 in >= 5 of the 8 regions.

  G3 (new regimes transfer). Each of NZ / CL / AR, as a holdout in the
     8-region run, has scalar R2 > 0.

A NEGATIVE overall result is the null and stands unless a gate is met. Reports
per-region R2 for both datasets so the delta is explicit.

Reuses the feature/model machinery of ml_transfer_retest.py unchanged
(Set A/B/C, RandomForest, ETOPO terrain features, 5 seeds).

Run after the cluster sweep has produced the train dirs, with
PYVWF_INPUT unset (this reads factors, no simulation).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ml_transfer_retest as _mtr
from ml_transfer_retest import (  # reuse machinery unchanged
    SET_A,
    build_centroids,
    loro,
    terrain_features,
)

# Pin the imported module's ROOT to this file's repo root, so build_centroids
# and terrain_features resolve the same way however either script is invoked.
_mtr.ROOT = Path(__file__).resolve().parents[2]

# Absolute so it wins over the imported build_centroids' own ROOT anchor.
_REPO = Path(__file__).resolve().parents[2]
SWEEP = str(_REPO / "output/validation/cluster_sweep_2026-07-24")

# region -> (train dir, k). k is chosen from the sweep grid: enough centroids
# for training rows, below the fake-plateau ceiling. The five originals keep the
# prior k for comparability; the three new regions take the top of their grid.
RUNS_5 = {
    "DK": (f"{SWEEP}/DK/train-dk", 100),
    "DE": (f"{SWEEP}/DE/train-de", 100),
    "UK": (f"{SWEEP}/UK/train-uk", 100),
    "US": (f"{SWEEP}/US/train-us", 100),
    "BR": (f"{SWEEP}/BR/train-br", 60),
}
RUNS_8 = {
    **RUNS_5,
    # New regions at their sweep-knee k (stable factors, not the edge-best):
    # NZ 5 (avoid the 7-farm ceiling), CL 8 (the k=3,5 blow-ups are the northern
    # extreme-scalar trap; k=8 is the stable optimum), AR 12 (k=35 edge-overfits).
    "NZ": (f"{SWEEP}/NZ/train-nz", 5),
    "CL": (f"{SWEEP}/CL/train-cl", 8),
    "AR": (f"{SWEEP}/AR/train-ar", 12),
}


def loro_setA(runs: dict, target: str = "scalar", cap: float | None = None) -> pd.DataFrame:
    """Build centroids, add terrain features, run leave-one-region-out (Set A).

    ``cap`` clips the scalar target to ``[0.3, cap]`` before training. The new
    complex-terrain regions carry extreme-scalar artifacts (CL 39, AR 15) from
    ERA5's desert/mountain wind under-resolution; without a cap they poison the
    pool numerically, so the capped run is the honest comparison.
    """
    df = build_centroids(runs)
    for c in ("lon", "lat"):
        df[f"{c}_norm"] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
    df = terrain_features(df)
    if cap is not None and target == "scalar":
        df["scalar"] = df["scalar"].clip(0.3, cap)
    return loro(df, SET_A, target)


def main() -> int:
    r5 = loro_setA(RUNS_5).set_index("holdout")
    r8 = loro_setA(RUNS_8).set_index("holdout")

    print("=" * 66)
    print("Scalar R2 (Set A, leave-one-region-out)")
    print("=" * 66)
    print(f"{'holdout':8} {'5-region':>10} {'8-region':>10} {'delta':>8}")
    for region in ["DK", "DE", "UK", "US", "BR", "NZ", "CL", "AR"]:
        a = r5.loc[region, "r2_mean"] if region in r5.index else float("nan")
        b = r8.loc[region, "r2_mean"] if region in r8.index else float("nan")
        delta = b - a if a == a else float("nan")
        d = f"{delta:+.3f}" if delta == delta else "   new"
        a_s = f"{a:.3f}" if a == a else "   -"
        print(f"{region:8} {a_s:>10} {b:>10.3f} {d:>8}")

    us5 = r5.loc["US", "r2_mean"]
    us8 = r8.loc["US", "r2_mean"]
    n_pos = int((r8["r2_mean"] > 0).sum())
    new_ok = all(r8.loc[reg, "r2_mean"] > 0 for reg in ("NZ", "CL", "AR"))

    print("\nPRE-SPECIFIED GATES")
    print(f"  G1 US-holdout: {us5:.3f} (5-region) -> {us8:.3f} (8-region)  "
          f"[{'TRANSFERS' if us8 > 0 else 'RESCUED' if us8 > -0.30 else 'still fails'}]")
    print(f"  G2 majority:   {n_pos}/8 regions R2>0  "
          f"[{'MET' if n_pos >= 5 else 'not met'}]")
    print(f"  G3 new regimes NZ/CL/AR all R2>0:  "
          f"[{'MET' if new_ok else 'not met'}]")

    # Capped variant: removes the numerical poisoning from the extreme-scalar
    # artifacts, so the transfer question is asked on tamed targets.
    c5 = loro_setA(RUNS_5, cap=3.0).set_index("holdout")
    c8 = loro_setA(RUNS_8, cap=3.0).set_index("holdout")
    print("\n" + "=" * 66)
    print("Scalar R2 with scalar capped at 3.0 (artifacts tamed)")
    print("=" * 66)
    print(f"{'holdout':8} {'5-region':>10} {'8-region':>10}")
    for region in ["DK", "DE", "UK", "US", "BR", "NZ", "CL", "AR"]:
        a = c5.loc[region, "r2_mean"] if region in c5.index else float("nan")
        b = c8.loc[region, "r2_mean"]
        a_s = f"{a:.3f}" if a == a else "   -"
        print(f"{region:8} {a_s:>10} {b:>10.3f}")

    out = Path("output/ml_retest")
    out.mkdir(parents=True, exist_ok=True)
    r8.reset_index().to_csv(out / "expanded_loro_scalar.csv", index=False)
    c8.reset_index().to_csv(out / "expanded_loro_scalar_capped.csv", index=False)
    print(f"\nwrote {out/'expanded_loro_scalar.csv'} (+ capped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
