"""Are the scalar and offset separately identified, and what is, if they are not?

Thesis chapter 4 reports a strong negative relationship between the scalar and
the offset across clusters, r = -0.867 pooled, and reads it as "compensatory
behaviour between multiplicative and additive adjustments". That reading treats
it as a property of the correction. This asks whether it is instead a statement
that the pair is **not identified**: if a larger scalar with a more negative
offset fits the same observations, then where a fit lands along that line is
arbitrary, and both chapters then operate on an arbitrary coordinate. Chapter 4
interpolates the two coefficients independently across space; chapter 5 trains
a model to predict one of them.

Three measurements, all from the existing pool and none needing a re-fit:

1. **How collinear, pooled and per row.** Pooling across rows mixes regimes and
   can only understate it.
2. **Where the pencil of lines pivots.** Corrected speed is
   ``v_corrected = scalar * v + offset``, so a row's corrections are a pencil
   of straight lines. Its variance across a row is
   ``v^2 Var(a) + 2v Cov(a, b) + Var(b)``, minimised at
   ``v* = -Cov(a, b) / Var(a)``. If that pivot sits at a physically ordinary
   wind speed, the row's corrections nearly agree there and disagree about the
   slope, which is a different and more useful description than two
   coefficients.
3. **Whether the corrected speed at the pivot is better conditioned** than
   either coefficient, measured as relative spread.

``(v_corrected(v1), v_corrected(v2))`` is a linear bijection of
``(scalar, offset)``, so a reparameterisation loses nothing and only changes
which coordinate is well determined.

Read-only. Writes its tables under ``<out_dir>``.

Usage, from the repository root:

    PYVWF_INPUT=input/combined PYTHONPATH=src python \\
        scripts/studies/method-correction-identifiability/correction_identifiability.py <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

POOL = Path("output/pyvwf_to_grid/all_corrections_centroids.csv")

#: Reference winds a correction is read at. 8 m/s is near these fleets' daily
#: mean; 5 and 11 bracket the steep part of a power curve.
REFERENCE_WINDS = (5.0, 8.0, 11.0)


def pivot(a: np.ndarray, b: np.ndarray) -> float:
    """The wind speed at which a row's corrected speeds agree most closely."""
    va = float(np.var(a, ddof=1))
    if va <= 0:
        return float("nan")
    return float(-np.cov(a, b, ddof=1)[0, 1] / va)


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(POOL)

    pooled_r = float(pool["scalar"].corr(pool["offset"]))
    print(f"=== pooled across all {len(pool)} control points: r = {pooled_r:.3f}")
    print("    the chapter reports -0.867 for the same quantity")

    rows = []
    for code, g in pool.groupby("country_code"):
        if len(g) < 3:
            rows.append({"row": code, "n": len(g)})
            continue
        a = g["scalar"].to_numpy(float)
        b = g["offset"].to_numpy(float)
        r = float(np.corrcoef(a, b)[0, 1])
        # Share of the joint variation lying along one line, after putting the
        # two coefficients on a common scale.
        z = np.column_stack([(a - a.mean()) / a.std(ddof=1),
                             (b - b.mean()) / b.std(ddof=1)])
        eig = np.linalg.eigvalsh(np.cov(z.T, ddof=1))
        along = float(eig.max() / eig.sum())
        p = pivot(a, b)
        record = {"row": code, "n": len(g), "r": round(r, 3),
                  "variance_along_the_line": round(along, 3),
                  "pivot_speed": round(p, 2),
                  "cv_scalar": round(float(a.std(ddof=1) / abs(a.mean())), 3),
                  "cv_offset": round(float(b.std(ddof=1) / abs(b.mean())), 3)}
        for v in REFERENCE_WINDS:
            corrected = a * v + b
            record[f"cv_v{v:g}"] = round(float(corrected.std(ddof=1)
                                               / abs(corrected.mean())), 3)
        if np.isfinite(p):
            at_pivot = a * p + b
            record["cv_at_pivot"] = round(float(at_pivot.std(ddof=1)
                                                / abs(at_pivot.mean())), 3)
            record["mean_at_pivot"] = round(float(at_pivot.mean()), 3)
        rows.append(record)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "identifiability_by_row.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n=== per row")
        print(frame.to_string(index=False))

        dense = frame[frame["n"] >= 10].dropna(subset=["r"])
        print("\n=== the rows with enough points to say anything")
        print(dense[["row", "n", "r", "variance_along_the_line", "pivot_speed",
                     "cv_scalar", "cv_offset", "cv_at_pivot"]].to_string(index=False))

        print("\n=== relative spread of the corrected speed, by reference wind")
        cols = ["row", "n"] + [f"cv_v{v:g}" for v in REFERENCE_WINDS] + ["cv_at_pivot"]
        print(frame.dropna(subset=["r"])[cols].to_string(index=False))
    print(f"\nwritten: {out / 'identifiability_by_row.csv'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
