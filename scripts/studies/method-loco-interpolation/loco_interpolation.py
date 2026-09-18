"""Leave-one-country-out for the interpolators. Read-only; committed before it runs.

Registered in ``docs/findings/method-loco-interpolation-prereg.md``. Chapter 5
tested its machine learning with country holdouts and found collapse; chapter 4
never ran one, so the two methods were judged at different rigour. This scores
the interpolators on the same holdouts, over the same 1,729 centroid-level
control points.

Everything the pre-registration fixes is fixed here and not on the command
line: twelve folds by country with onshore and offshore combined, every fold
scored including the hard ones, the Netherlands fold scored and primary, mean
absolute error primary with R-squared beside it, scalar error in both linear
and log space, and distance great-circle with Euclidean degrees beside it.

The interpolators are ``vwf.extensions.grid.interpolation``, ported from the
chapter's own script, so this study and the offshore pool study share one
definition of every number.

**RBF has no metric option.** ``scipy``'s interpolator fits on the coordinates
it is given, so it runs in degrees under both distance columns, and its rows
say so.

Usage, from the repository root:

    PYTHONPATH=src python scripts/studies/method-loco-interpolation/loco_interpolation.py <out_dir>
"""
from pathlib import Path

import numpy as np
import pandas as pd

from vwf.cli.common import make_parser
from vwf.extensions.grid import interpolation as interp

POOL = Path("output/pyvwf_to_grid/all_corrections_centroids.csv")

#: The IDW product masks a cell beyond this many degrees from any control
#: point. Reported per fold so a hard fold is visible as hard.
MASK_DEG = interp.MAX_DISTANCE_DEG

#: Great circle is this study's metric; degrees is chapter 4's and is reported
#: beside it. Kriging takes the equivalent through pykrige's coordinate type.
METRICS = {"great_circle": "geographic", "degrees": "euclidean"}


def folds(pool: pd.DataFrame) -> dict[str, np.ndarray]:
    """One fold per country, onshore and offshore combined.

    The pool's ``country_code`` carries the mode for the turbine-level rows,
    as ``DK-onshore`` and ``DK-offshore``. Chapter 5's reported holdout sizes,
    500 for Germany and 303 for the United Kingdom, are the combined counts, so
    combining follows it rather than inventing a partition.
    """
    country = pool["country_code"].str.split("-").str[0]
    return {name: np.flatnonzero((country == name).to_numpy())
            for name in sorted(country.unique())}


def skill(predicted: np.ndarray, actual: np.ndarray) -> dict:
    """MAE and R-squared, with R-squared reported however unstable it is.

    A negative R-squared is a result: it says the fold would have been better
    served by its own mean than by the prediction. Chapter 5's own country
    holdouts are negative, so hiding it here would break the comparison this
    study exists to make.
    """
    error = predicted - actual
    ss_res = float((error ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    return {"mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt((error ** 2).mean())),
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")}


def predict(method: str, train: pd.DataFrame, test: pd.DataFrame, metric: str):
    """One method's scalar and offset predictions for one held-out fold."""
    lon, lat = test["lon"].to_numpy(), test["lat"].to_numpy()
    if method == "idw":
        return interp.idw_at(train, lon, lat, metric=metric)
    if method == "nearest":
        return interp.nearest_at(train, lon, lat, metric=metric)
    if method == "rbf":
        return interp.rbf_at(train, lon, lat)
    if method == "kriging":
        return interp.kriging_at(train, lon, lat, coordinates_type=METRICS[metric])
    raise ValueError(f"unknown method {method!r}")


def main(out_dir: str, pool_path: Path = POOL) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(pool_path)
    rows, geometry = [], []

    for name, index in folds(pool).items():
        test = pool.iloc[index]
        train = pool.drop(index=pool.index[index])
        km = interp.distance_to_nearest(train, test["lon"], test["lat"],
                                        metric="great_circle")
        deg = interp.distance_to_nearest(train, test["lon"], test["lat"],
                                         metric="degrees")
        geometry.append({
            "fold": name, "n": len(test),
            "km_to_nearest_min": float(km.min()), "km_to_nearest_median": float(np.median(km)),
            "km_to_nearest_max": float(km.max()),
            "deg_to_nearest_median": float(np.median(deg)),
            "share_beyond_mask": float((deg > MASK_DEG).mean()),
        })
        for metric in METRICS:
            for method in ("idw", "kriging", "nearest", "rbf"):
                try:
                    scalar, offset = predict(method, train, test, metric)
                except Exception as error:  # recorded, never silently skipped
                    rows.append({"fold": name, "n": len(test), "metric": metric,
                                 "method": method, "failed": f"{type(error).__name__}: {error}"})
                    continue
                row = {"fold": name, "n": len(test), "metric": metric, "method": method,
                       "metric_applies": method != "rbf"}
                for key, value in skill(np.asarray(scalar),
                                        test["scalar"].to_numpy()).items():
                    row[f"scalar_{key}"] = value
                positive = (np.asarray(scalar) > 0) & (test["scalar"].to_numpy() > 0)
                row["scalar_log_mae"] = (
                    float(np.abs(np.log(np.asarray(scalar)[positive])
                                 - np.log(test["scalar"].to_numpy()[positive])).mean())
                    if positive.any() else float("nan"))
                row["scalar_log_n"] = int(positive.sum())
                for key, value in skill(np.asarray(offset),
                                        test["offset"].to_numpy()).items():
                    row[f"offset_{key}"] = value
                rows.append(row)

    frame = pd.DataFrame(rows)
    geo = pd.DataFrame(geometry)
    frame.to_csv(out / "loco_scores.csv", index=False)
    geo.to_csv(out / "loco_fold_geometry.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("=== fold geometry")
        print(geo.round(3).to_string(index=False))
        for metric in METRICS:
            print(f"\n=== {metric}: scalar MAE by fold and method")
            part = frame[frame["metric"] == metric]
            print(part.pivot(index="fold", columns="method", values="scalar_mae")
                  .round(4).to_string())
            print(f"--- {metric}: offset MAE")
            print(part.pivot(index="fold", columns="method", values="offset_mae")
                  .round(4).to_string())
            print(f"--- {metric}: scalar R2")
            print(part.pivot(index="fold", columns="method", values="scalar_r2")
                  .round(3).to_string())
    print(f"\nwritten: {out / 'loco_scores.csv'}, {out / 'loco_fold_geometry.csv'}")


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<out_dir>``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("out_dir", help="Directory for the outputs, under output/")
    parser.add_argument("--pool", type=Path, default=POOL,
                        help=f"The control-point pool (default: {POOL})")
    args = parser.parse_args(argv)
    main(args.out_dir, pool_path=args.pool)


if __name__ == "__main__":
    cli()
