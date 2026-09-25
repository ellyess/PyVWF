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
        scripts/studies/method-why-corrections-do-not-transfer/pool_as_training_set.py <out_dir>
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from pyvwf.cli.common import make_parser  # noqa: E402
from pyvwf.extensions.grid.surface import flag_implausible, zero_crossing_speed  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "ml_transfer_retest", REPO / "scripts" / "analysis" / "ml_transfer_retest.py"
)
_ml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ml)

POOL = REPO / "output/pyvwf_to_grid/all_corrections_centroids.csv"
RUNS = REPO / "output/runs/turbine_grid"
SEL = REPO / "output/cluster_selection_2026-09-15"

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
        lambda d: pd.Series(
            {
                "lon": np.average(d["lon"], weights=d["w"]),
                "lat": np.average(d["lat"], weights=d["w"]),
                "units": len(d),
                "capacity": d["w"].sum(),
            }
        ),
        include_groups=False,
    )
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
    return {
        "points": len(frame),
        "occupied_cells": occupied,
        "points_per_cell": round(len(frame) / occupied, 2),
        "median_nn_distance": round(float(np.median(d.min(1))), 4),
        "5pct_nn_distance": round(float(np.percentile(d.min(1), 5)), 4),
    }


def main(out_dir: str, pool_path: Path = POOL, runs: Path = RUNS, selection: Path = SEL) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== 1. Feature-space coverage: Denmark onshore at 884 against 200")
    frames = {}
    for k in (884, 200, 100):
        info = pd.read_csv(
            Path(selection) / "DK" / "train-onshore-final" / f"train_turb_info_{k}.csv"
        )
        c = centroids(info).assign(region="DK")
        frames[k] = _ml.terrain_features(c)
    rows = [{"clustering": f"k={k}", **coverage(f, frames[884])} for k, f in frames.items()]
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(out / "dk_onshore_coverage.csv", index=False)

    print("\n=== 2. Label quality across the pool")
    pool = pd.read_csv(pool_path)
    # Country rows carry cluster membership in the fleet file, so their unit
    # counts join exactly. Turbine rows do not: the chapter-era runs stored no
    # per-cluster membership, so their cluster sizes are taken from the
    # selection study's run at the SAME cluster count, which is the chapter's
    # count in every case because it was evaluated as baseline B1. That is a
    # current training fleet rather than the chapter-era one, so the
    # distribution is indicative and the join is by size, not by identity.
    units = []
    for code, year in (
        ("BE", 2023),
        ("ES", 2023),
        ("FR", 2023),
        ("IE", 2023),
        ("IT", 2023),
        ("NL", 2023),
        ("NO", 2023),
        ("PT", 2023),
        ("SE", 2023),
    ):
        info = pd.read_csv(
            Path(runs)
            / f"{code}-all-obs_country-corrected-calc_z0"
            / "training"
            / "simulated-turbines"
            / f"{code}_{year}_turb_info.csv"
        )
        n = info.groupby("cluster").size().rename("units").reset_index()
        n["country_code"] = code
        units.append(n)
    counts = pd.concat(units, ignore_index=True)
    merged = pool.merge(counts, on=["country_code", "cluster"], how="left")

    sel = Path(selection)
    turbine_sizes = {}
    for pool_code, region, mode, k in (
        ("DE-onshore", "DE", "onshore", 500),
        ("DK-offshore", "DK", "offshore", 2),
        ("DK-onshore", "DK", "onshore", 884),
        ("UK-offshore", "UK", "offshore", 10),
        ("UK-onshore", "UK", "onshore", 300),
    ):
        # Two spellings: rows re-run after the run-path fix carry the fleet
        # mode, the three clean rows predate it. Missing is reported, not
        # skipped: a row that quietly vanishes from a table is the same defect
        # as a detector that never fires.
        candidates = [
            sel / region / f"train-{mode}-final" / f"train_turb_info_{k}.csv",
            sel / region / "train-final" / f"train_turb_info_{k}.csv",
        ]
        found = next((c for c in candidates if c.is_file()), None)
        if found is None:
            raise FileNotFoundError(
                f"no clustering at k={k} for {pool_code}; tried "
                + ", ".join(str(c) for c in candidates)
            )
        turbine_sizes[pool_code] = pd.read_csv(found).groupby("cluster").size()

    merged["crossing"] = zero_crossing_speed(merged["scalar"], merged["offset"])
    merged["out_of_bounds"] = flag_implausible(merged["scalar"], merged["offset"])
    merged["few_units"] = merged["units"].fillna(0) < 3
    merged["single_unit"] = merged["units"].fillna(0) <= 1
    merged.to_csv(out / "pool_label_quality.csv", index=False)

    print(f"  points: {len(merged)}")
    print(f"  out of bounds (the visible seven): {int(merged.out_of_bounds.sum())}")
    print("\n  country rows, exact units per control point:")
    by = (
        merged.dropna(subset=["units"])
        .groupby("country_code")["units"]
        .describe()[["count", "min", "25%", "50%", "max"]]
    )
    print(by.round(1).to_string())
    c = merged.dropna(subset=["units"])
    print(
        f"    of {len(c)} country points: {int((c.units <= 1).sum())} on one grid point, "
        f"{int((c.units < 3).sum())} on fewer than three"
    )

    print("\n  turbine rows, cluster sizes at the pool's own count, from the")
    print("  selection study's run at the same count (indicative, see the code):")
    rows = []
    for name, sizes in turbine_sizes.items():
        rows.append(
            {
                "row": name,
                "clusters": len(sizes),
                "min": int(sizes.min()),
                "median": float(sizes.median()),
                "max": int(sizes.max()),
                "singletons": int((sizes <= 1).sum()),
                "under_three": int((sizes < 3).sum()),
                "share_under_three": round(float((sizes < 3).mean()), 3),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n  the scalar and offset trade off against each other, by row")
    print("  (a strong negative correlation means the pair is under-determined:")
    print("   a larger scalar with a more negative offset fits the same mean)")
    corr = merged.groupby("country_code").apply(
        lambda d: pd.Series(
            {
                "n": len(d),
                "corr_scalar_offset": round(float(d.scalar.corr(d.offset)), 3)
                if len(d) > 2
                else np.nan,
            }
        ),
        include_groups=False,
    )
    print(corr.to_string())
    print(f"\nwritten: {out}")


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<out_dir>``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("out_dir", help="Directory for the outputs, under output/")
    parser.add_argument(
        "--pool", type=Path, default=POOL, help=f"The control-point pool (default: {POOL})"
    )
    parser.add_argument(
        "--runs", type=Path, default=RUNS, help=f"The chapter's thesis-era runs (default: {RUNS})"
    )
    parser.add_argument(
        "--selection", type=Path, default=SEL, help=f"The cluster selection runs (default: {SEL})"
    )
    args = parser.parse_args(argv)
    main(args.out_dir, pool_path=args.pool, runs=args.runs, selection=args.selection)


if __name__ == "__main__":
    cli()
