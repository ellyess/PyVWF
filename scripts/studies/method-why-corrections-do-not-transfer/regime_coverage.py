"""What regimes the pool covers, and whether coverage explains the holdouts.

`method-ml-transfer.md` named regime coverage, not sample count, as the
binding constraint on transfer, and the alternatives have since been tested
and eliminated: the pool's density is redundant, the target's conditioning
changes no geometry-weighted prediction, and the identifiability defect
explains neither collapse. Regime coverage is now the standing hypothesis, so
this measures it.

Three parts, the third being the test:

1. **What the 1,729 control points cover**, in the terrain-feature space the
   machine-learning work used, and where the gaps are.
2. **Which other regions would extend that coverage rather than add density**,
   measured as feature cells they occupy that Europe does not.
3. **Whether the folds that do least badly are the ones whose regime is
   represented elsewhere in the pool.** For each country holdout, how far its
   points sit from the nearest training point in feature space, set against
   the leave-one-country-out error already measured. **If nothing separates
   them, that is reported**: it would say the binding constraint is not
   coverage either.

Read-only. Writes under ``<out_dir>``.

Usage, from the repository root:

    PYVWF_INPUT=input/combined PYTHONPATH=src python \\
        scripts/studies/method-why-corrections-do-not-transfer/regime_coverage.py <out_dir>
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from vwf.cli.common import make_parser

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "ml_transfer_retest", REPO / "scripts" / "analysis" / "ml_transfer_retest.py"
)
_ml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ml)

POOL = REPO / "output/pyvwf_to_grid/all_corrections_centroids.csv"
LOCO = REPO / "output/loco_reference_2026-09-16/loco_reference_wind.csv"
REFRESH = REPO / "output/validation/refresh_2026-08-24"

FEATURES = ("elevation", "slope", "roughness", "curvature")
BINS = 8

#: Regions outside the European pool, with the cluster count their scorecard
#: run used. Candidates for extending coverage rather than adding density.
CANDIDATES = {"US": 250, "BR": 60, "CL": 10, "AR": 10, "NZ": 7, "AU-NEM": 45}

#: The pool's own country codes carry the mode; the LOCO folds do not.
FOLD_OF = {
    "DE-onshore": "DE",
    "DK-onshore": "DK",
    "DK-offshore": "DK",
    "UK-onshore": "UK",
    "UK-offshore": "UK",
}


def featurise(frame: pd.DataFrame, region: str) -> pd.DataFrame:
    return _ml.terrain_features(frame.assign(region=region))


def cells(z: np.ndarray) -> set:
    edges = np.linspace(-3, 3, BINS + 1)
    return {tuple(r) for r in np.stack([np.digitize(z[:, i], edges) for i in range(z.shape[1])], 1)}


def main(
    out_dir: str, pool_path: Path = POOL, loco_path: Path = LOCO, refresh: Path = REFRESH
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(pool_path)
    pool = pd.concat([featurise(g, code) for code, g in pool.groupby("country_code")])
    pool["fold"] = pool["country_code"].map(FOLD_OF).fillna(pool["country_code"])

    mu = pool[list(FEATURES)].mean()
    sd = pool[list(FEATURES)].std().replace(0, 1.0)
    z_pool = ((pool[list(FEATURES)] - mu) / sd).to_numpy()
    print(
        f"=== the pool: {len(pool)} points, {len(cells(z_pool))} occupied cells "
        f"of {BINS ** len(FEATURES)}"
    )
    print("\n  feature ranges, raw units")
    print(
        pool[list(FEATURES)]
        .describe()
        .loc[["min", "25%", "50%", "75%", "max"]]
        .round(2)
        .to_string()
    )

    print("\n=== 2. what other regions would add")
    rows = []
    base = cells(z_pool)
    for region, k in CANDIDATES.items():
        f = Path(refresh) / region / "train-refresh" / f"train_turb_info_{k}.csv"
        if not f.is_file():
            raise FileNotFoundError(f"no fleet for {region} at {f}")
        other = featurise(pd.read_csv(f), region)
        z = ((other[list(FEATURES)] - mu) / sd).to_numpy()
        new = cells(z) - base
        rows.append(
            {
                "region": region,
                "points": len(other),
                "cells": len(cells(z)),
                "cells_new_to_europe": len(new),
                "share_of_its_cells_new": round(len(new) / max(len(cells(z)), 1), 3),
                "max_elevation": round(float(other["elevation"].max()), 0),
                "max_roughness": round(float(other["roughness"].max()), 1),
            }
        )
    added = pd.DataFrame(rows).sort_values("cells_new_to_europe", ascending=False)
    print(added.to_string(index=False))
    added.to_csv(out / "candidate_regions.csv", index=False)

    print("\n=== 3. does coverage explain the holdouts?")
    loco = pd.read_csv(loco_path)
    loco = loco[(loco.metric == "great_circle") & (loco.method == "idw")]
    rows = []
    for fold, held in pool.groupby("fold"):
        train = pool[pool["fold"] != fold]
        zt = ((train[list(FEATURES)] - mu) / sd).to_numpy()
        zh = ((held[list(FEATURES)] - mu) / sd).to_numpy()
        d = np.sqrt(((zh[:, None, :] - zt[None, :, :]) ** 2).sum(-1)).min(1)
        skill = loco[loco.fold == fold]
        rows.append(
            {
                "fold": fold,
                "points": len(held),
                "feature_distance_median": round(float(np.median(d)), 3),
                "feature_distance_90pct": round(float(np.percentile(d, 90)), 3),
                "share_outside_training_cells": round(
                    float(
                        np.mean(
                            [
                                tuple(r) not in cells(zt)
                                for r in np.stack(
                                    [
                                        np.digitize(zh[:, i], np.linspace(-3, 3, BINS + 1))
                                        for i in range(zh.shape[1])
                                    ],
                                    1,
                                )
                            ]
                        )
                    ),
                    3,
                ),
                "loco_mae": round(float(skill["from_reference_mae"].iloc[0]), 3)
                if len(skill)
                else np.nan,
                "loco_r2": round(float(skill["from_reference_r2"].iloc[0]), 3)
                if len(skill)
                else np.nan,
            }
        )
    table = pd.DataFrame(rows).sort_values("loco_mae")
    print(table.to_string(index=False))
    table.to_csv(out / "fold_coverage.csv", index=False)

    ok = table.dropna(subset=["loco_mae"])
    for col in (
        "feature_distance_median",
        "feature_distance_90pct",
        "share_outside_training_cells",
    ):
        r = float(ok[col].corr(ok["loco_mae"]))
        rho = float(ok[col].corr(ok["loco_mae"], method="spearman"))
        print(f"  {col} against LOCO MAE: pearson {r:+.3f}, spearman {rho:+.3f}")
    print(
        "\nIf these are near zero, coverage does not explain the holdouts "
        "either, and that is the result."
    )


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<out_dir>``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("out_dir", help="Directory for the outputs, under output/")
    parser.add_argument(
        "--pool", type=Path, default=POOL, help=f"The control-point pool (default: {POOL})"
    )
    parser.add_argument(
        "--loco", type=Path, default=LOCO, help=f"The LOCO reference-wind scores (default: {LOCO})"
    )
    parser.add_argument(
        "--refresh",
        type=Path,
        default=REFRESH,
        help=f"The refresh runs holding each region's fleet (default: {REFRESH})",
    )
    args = parser.parse_args(argv)
    main(args.out_dir, pool_path=args.pool, loco_path=args.loco, refresh=args.refresh)


if __name__ == "__main__":
    cli()
