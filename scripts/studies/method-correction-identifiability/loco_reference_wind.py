"""Leave-one-country-out on an identified target, not on two coefficients.

`method-correction-identifiability.md` shows the affine fit solves one equation
in two unknowns, so the split between scalar and offset is set by a tie-break
and chapter 4 interpolates two coordinates of it independently. This scores the
same holdouts on the corrected speed at a reference wind, which is identified.

**The reference winds are fixed here before the run: 8 and 12 m/s.** They are
inside the range the objective can see: the pivot near 4 m/s, where the
corrections coincide, is **rejected as a reference wind** because a pencil of
lines has least spread at its crossing point by construction, so a target there
is well conditioned only because it is unconstrained, and under 2.5% of the
capacity-factor mass the fit matches lies below it. At 8 to 12 m/s the
within-row spread is 0.08 to 0.21 rather than 0.02, which is the evidence the
target carries something.

**This is not framed as a rescue.** Interpolating two unidentified coefficients
independently is wrong whether or not a better target scores better, and the
question here is only whether the target's conditioning changes the result.

**An improvement could be the target varying less rather than transfer
working**, so the target's own within-row and between-row variance is reported
beside every score and no improvement is read without it.

Read-only. Writes under ``<out_dir>``.

Usage, from the repository root:

    PYTHONPATH=src python scripts/studies/method-correction-identifiability/loco_reference_wind.py <out_dir>
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from pyvwf.cli.common import make_parser


REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "loco_interpolation",
    REPO / "scripts" / "studies" / "method-loco-interpolation" / "loco_interpolation.py",
)
_loco = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loco)

POOL = REPO / "output/pyvwf_to_grid/all_corrections_centroids.csv"

#: Fixed before the run. See the module docstring for why the pivot is not one.
REFERENCE_WINDS = (8.0, 12.0)

METHODS = ("idw", "kriging", "nearest", "rbf")


def as_reference(frame: pd.DataFrame) -> pd.DataFrame:
    """The same corrections in the reference-wind basis.

    ``(u1, u2) = (a v1 + b, a v2 + b)`` is an invertible linear map of
    ``(a, b)``, so nothing is lost and only the conditioning changes.
    """
    v1, v2 = REFERENCE_WINDS
    return frame.assign(
        scalar=frame["scalar"] * v1 + frame["offset"], offset=frame["scalar"] * v2 + frame["offset"]
    )


def main(out_dir: str, pool_path: Path = POOL) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(pool_path)
    v1, v2 = REFERENCE_WINDS

    print(f"=== the target, at {v1:g} and {v2:g} m/s")
    ref = as_reference(pool)
    for name, col in (("u8", "scalar"), ("u12", "offset")):
        grand = float(ref[col].mean())
        within = float(
            ref.groupby("country_code")[col].apply(lambda s: ((s - s.mean()) ** 2).sum()).sum()
        )
        between = float(
            ref.groupby("country_code")[col].apply(lambda s: len(s) * (s.mean() - grand) ** 2).sum()
        )
        print(
            f"  {name}: mean {grand:.3f}, within-row share of variance "
            f"{within / (within + between):.3f}, between-row "
            f"{between / (within + between):.3f}"
        )
    for col in ("scalar", "offset"):
        grand = float(pool[col].mean())
        within = float(
            pool.groupby("country_code")[col].apply(lambda s: ((s - s.mean()) ** 2).sum()).sum()
        )
        between = float(
            pool.groupby("country_code")[col]
            .apply(lambda s: len(s) * (s.mean() - grand) ** 2)
            .sum()
        )
        print(
            f"  {col}: mean {grand:.3f}, within-row share {within / (within + between):.3f}, "
            f"between-row {between / (within + between):.3f}"
        )

    rows = []
    for name, index in _loco.folds(pool).items():
        test_c = pool.iloc[index]
        train_c = pool.drop(index=pool.index[index])
        test_r, train_r = as_reference(test_c), as_reference(train_c)
        for metric in _loco.METRICS:
            for method in METHODS:
                try:
                    a_hat, b_hat = _loco.predict(method, train_c, test_c, metric)
                    u1_hat, u2_hat = _loco.predict(method, train_r, test_r, metric)
                except Exception as error:  # recorded, never silently skipped
                    rows.append(
                        {
                            "fold": name,
                            "metric": metric,
                            "method": method,
                            "failed": f"{type(error).__name__}: {error}",
                        }
                    )
                    continue
                # The coefficient prediction, converted into the same units, so
                # the two are compared on one scale.
                converted = np.asarray(a_hat) * v1 + np.asarray(b_hat)
                actual = test_r["scalar"].to_numpy()
                row = {"fold": name, "n": len(test_c), "metric": metric, "method": method}
                for prefix, pred in (
                    ("from_coefficients", converted),
                    ("from_reference", np.asarray(u1_hat)),
                ):
                    for key, value in _loco.skill(pred, actual).items():
                        row[f"{prefix}_{key}"] = value
                row["prediction_difference"] = float(np.abs(converted - np.asarray(u1_hat)).max())
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "loco_reference_wind.csv", index=False)
    done = frame[frame.get("failed").isna()] if "failed" in frame else frame
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(
            f"\n=== do the two bases predict the same thing? "
            f"(largest disagreement at {v1:g} m/s, per method)"
        )
        print(
            done.groupby("method")["prediction_difference"].describe()[["max"]].round(9).to_string()
        )
        print("\n=== error at the reference wind, by method, averaged over folds")
        cols = [
            "from_coefficients_mae",
            "from_reference_mae",
            "from_coefficients_r2",
            "from_reference_r2",
        ]
        print(done.groupby(["metric", "method"])[cols].mean().round(4).to_string())
        print("\n=== every fold, great circle")
        gc = done[done.metric == "great_circle"]
        print(
            gc[
                [
                    "fold",
                    "n",
                    "method",
                    "from_coefficients_mae",
                    "from_reference_mae",
                    "from_coefficients_r2",
                    "from_reference_r2",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )
    print(f"\nwritten: {out / 'loco_reference_wind.csv'}")


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<out_dir>``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("out_dir", help="Directory for the outputs, under output/")
    parser.add_argument(
        "--pool", type=Path, default=POOL, help=f"The control-point pool (default: {POOL})"
    )
    args = parser.parse_args(argv)
    main(args.out_dir, pool_path=args.pool)


if __name__ == "__main__":
    cli()
