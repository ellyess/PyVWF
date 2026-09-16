"""Is the control-point pool a good training set for the transfer problem?

The manuscript asks whether a correction can be predicted for a country with no
control points, from terrain and other features. That makes the pool a
**supervised training set**, not an interpolation input, and changes what makes
it good: feature coverage and label quality rather than density.

Three read-only measurements, before any pool is designed:

1. **Feature-space redundancy.** Denmark onshore contributes 884 control points
   on near-continuous low terrain. If those are near-identical feature vectors
   they are rows without information. Measured by comparing the same fleet
   clustered at 884 and at 200: how much terrain-feature coverage is lost.
2. **Label quality beyond the visible failures.** Seven of 1,729 points are
   implausible by scalar bounds or zero crossing. A label can be noisy without
   leaving bounds: fitted on few units, or with a scalar and offset that trade
   off against each other so the pair is under-determined.
3. The leave-one-country-out premise is not measured here; it is already
   answered in ``method-ml-transfer.md`` and is cited rather than re-run.

Read-only. Writes its tables under ``<out_dir>``.

Usage, from the repository root:

    PYVWF_INPUT=input/combined PYTHONPATH=src python \\
        scripts/analysis/pool_as_training_set.py <out_dir>
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "ml_transfer_retest", REPO / "scripts" / "analysis" / "ml_transfer_retest.py")
_ml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ml)

POOL = REPO / "output/pyvwf_to_grid/all_corrections_centroids.csv"
RUNS = REPO / "output/runs/turbine_grid"
DK_FINAL = REPO / "output/cluster_selection_2026-09-15/DK/train-onshore-final"

#: The terrain features the ML transfer work used, so coverage is measured in
#: the space the model actually sees.
FEATURES = ("elevation", "slope", "roughness", "curvature")

#: Bins per feature when counting occupied cells in standardised space. Coarse
#: on purpose: the question is whether a regime is represented at all, not how
#: finely.
BINS = 8


def centroids(turb_info: pd.DataFrame) -> pd.DataFrame:
    """Cluster centroids, capacity-weighted, as the pool's own points are."""
    g = turb_info.assign(w=turb_info["capacity"].astype(float))
    out = g.groupby("cluster").apply(
        lambda d: pd.Series({"lon": np.average(d["lon"], weights=d["w"]),
                             "lat": np.average(d["lat"], weights=d["w"]),
                             "units": len(d), "capacity": d["w"].sum()}),
        include_groups=False)
    return out.reset_index()


def coverage(frame: pd.DataFrame, reference: pd.DataFrame) -> dict:
    """Occupied cells and redundancy, in the reference's standardised space.

    Standardising on one reference keeps two clusterings on one scale, so the
    counts are comparable rather than each normalised to itself.
    """
    mu = reference[list(FEATURES)].mean()
    sd = reference[list(FEATURES)].std().replace(0, 1.0)
    z = ((frame[list(FEATURES)] - mu) / sd).to_numpy()
    edges = [np.linspace(-3, 3, BINS + 1) for _ in FEATURES]
    idx = np.stack([np.digitize(z[:, i], edges[i]) for i in range(len(FEATURES))], 1)
    occupied = len({tuple(r) for r in idx})
    # Redundancy: how close each point is to its nearest neighbour in feature
    # space. A dense pool of near-identical rows has a small median.
    d = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return {"points": len(frame), "occupied_cells": occupied,
            "points_per_cell": round(len(frame) / occupied, 2),
            "median_nn_distance": round(float(np.median(d.min(1))), 4),
            "5pct_nn_distance": round(float(np.percentile(d.min(1), 5)), 4)}


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== 1. Feature-space coverage: Denmark onshore at 884 against 200")
    frames = {}
    for k in (884, 200, 100):
        info = pd.read_csv(DK_FINAL / f"train_turb_info_{k}.csv")
        c = centroids(info).assign(region="DK")
        frames[k] = _ml.terrain_features(c)
    rows = [{"clustering": f"k={k}", **coverage(f, frames[884])}
            for k, f in frames.items()]
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(out / "dk_onshore_coverage.csv", index=False)

    print("\n=== 2. Label quality across the pool")
    pool = pd.read_csv(POOL)
    units = []
    for code, mode, level, year in (("BE","all","country",2023),("DE","onshore","turbine",2019),
                                    ("DK","offshore","turbine",2020),("DK","onshore","turbine",2020),
                                    ("ES","all","country",2023),("FR","all","country",2023),
                                    ("IE","all","country",2023),("IT","all","country",2023),
                                    ("NL","all","country",2023),("NO","all","country",2023),
                                    ("PT","all","country",2023),("SE","all","country",2023),
                                    ("UK","offshore","turbine",2019),("UK","onshore","turbine",2019)):
        f = RUNS / f"{code}-{mode}-obs_{level}-corrected-calc_z0" / "training" / \
            "simulated-turbines" / f"{code}_{year}_turb_info.csv"
        info = pd.read_csv(f)
        pool_code = {"DE": "DE-onshore", "DK": f"DK-{mode}", "UK": f"UK-{mode}"}.get(code, code)
        if code in ("DK", "UK"):
            pool_code = f"{code}-{mode}"
        n = info.groupby("cluster").size().rename("units").reset_index()
        n["country_code"] = pool_code
        units.append(n)
    counts = pd.concat(units, ignore_index=True)
    merged = pool.merge(counts, on=["country_code", "cluster"], how="left")

    low, high = 0.2, 3.0
    merged["crossing"] = np.where((merged.offset < 0) & (merged.scalar > 0),
                                  -merged.offset / merged.scalar, np.nan)
    merged["out_of_bounds"] = (merged.scalar < low) | (merged.scalar > high) | \
                              (merged.crossing > 4.0)
    merged["few_units"] = merged["units"].fillna(0) < 3
    merged["single_unit"] = merged["units"].fillna(0) <= 1
    merged.to_csv(out / "pool_label_quality.csv", index=False)

    print(f"  points: {len(merged)}, with a unit count: {int(merged.units.notna().sum())}")
    print(f"  out of bounds (the visible seven): {int(merged.out_of_bounds.sum())}")
    print(f"  fitted on a single unit: {int(merged.single_unit.sum())}")
    print(f"  fitted on fewer than three units: {int(merged.few_units.sum())}")
    print("\n  units per control point, by row:")
    by = merged.groupby("country_code")["units"].describe()[["count","min","25%","50%","max"]]
    print(by.round(1).to_string())

    print("\n  the scalar and offset trade off against each other, by row")
    print("  (a strong negative correlation means the pair is under-determined:")
    print("   a larger scalar with a more negative offset fits the same mean)")
    corr = merged.groupby("country_code").apply(
        lambda d: pd.Series({"n": len(d),
                             "corr_scalar_offset": round(float(d.scalar.corr(d.offset)), 3)
                             if len(d) > 2 else np.nan}), include_groups=False)
    print(corr.to_string())
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
