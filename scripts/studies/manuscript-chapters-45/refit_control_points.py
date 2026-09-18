"""Do Italy's and Portugal's control points come back inside the scalar bounds?

Seven of the pool's 1,729 control points are implausible by this project's own
definitions, and they concentrate: all three of Italy's, two of Portugal's, one
of France's, one of Germany's. Italy and Portugal are also the two rows with
the largest chapter-era extrapolated capacity shares, 89.35% and 84.92%, so
their simulated output was built from winds that did not exist and the fitted
scalar is observed over simulated. **That makes extrapolation a candidate cause
and not a demonstrated one**, which is what this checks.

It refits each named configuration on ``era5/EU_2026-09``, which covers every
unit with no extrapolation, at the cluster count the grid points carry, and
prints the fitted scalars and offsets against the pool's current values with
the plausibility screen applied to both.

Three things it cannot separate, stated rather than buried:

- **The archive and the roughness treatment move together.** The newer files
  carry no stored roughness, so the run derives it per timestep. Both changes
  are in every number here.
- **The comparison is cross-pipeline.** The pool was built by the legacy path
  under ``output/runs/turbine_grid``; this refits through the harness.
  **Belgium is therefore run as a control**: its control points are plausible
  and its fleet needs no extrapolation under either archive, so if Belgium does
  not reproduce, the Italian and Portuguese comparisons cannot be read either.
- **Portugal's observed series is still defective.** Its capacity register is
  flat for five years and is not repaired here, so a Portuguese scalar that
  stays implausible has two candidate causes and this run rules out neither.

Read-only apart from the run directories it writes under ``<out_dir>``.

Usage, from the repository root:

    PYVWF_INPUT=input/combined PYTHONPATH=src python \\
        scripts/studies/manuscript-chapters-45/refit_control_points.py <out_dir> [stem ...]
"""
import dataclasses
import sys
from pathlib import Path

import pandas as pd

from vwf.extensions.grid.surface import (
    MAX_ZERO_CROSSING_SPEED,
    PLAUSIBLE_SCALAR,
    flag_implausible,
    zero_crossing_speed,
)
from vwf.harness import driver, regions

POOL = Path("output/pyvwf_to_grid/all_corrections_centroids.csv")
ERA5_PATH, ROUGHNESS, TIME_SLICE = "era5/EU_2026-09", "derived", "fixed"

#: Belgium first, as the control. See the module docstring.
DEFAULT_STEMS = ("be", "it", "pt")


def screen(frame: pd.DataFrame) -> pd.DataFrame:
    """The plausibility screen correction_surface applies, on fitted pairs."""
    low, high = PLAUSIBLE_SCALAR
    out = frame.copy()
    out["crossing"] = zero_crossing_speed(out["scalar"], out["offset"])
    out["scalar_bad"] = (out["scalar"] < low) | (out["scalar"] > high)
    out["crossing_bad"] = out["crossing"] > MAX_ZERO_CROSSING_SPEED
    out["implausible"] = flag_implausible(out["scalar"], out["offset"])
    return out


def main(out_dir: str, *stems: str) -> None:
    out = Path(out_dir)
    pool = pd.read_csv(POOL)
    results = []

    for stem in (stems or DEFAULT_STEMS):
        spec = regions.load_region(Path("configs/regions") / f"{stem}.toml")
        current = pool[pool["country_code"] == spec.code]
        clusters = len(current)
        print(f"\n=== {spec.code}: refitting {clusters} clusters on {ERA5_PATH}",
              flush=True)

        refit = dataclasses.replace(
            spec, cluster_list=(clusters,), time_slices=(TIME_SLICE,),
            era5_path=ERA5_PATH, roughness=ROUGHNESS)
        run_dir = driver.run_train(refit, out, run_name="refit")
        factors = pd.read_csv(run_dir / f"factors_{TIME_SLICE}_{clusters}.csv")

        before = screen(current[["cluster", "scalar", "offset"]]).assign(
            code=spec.code, condition="pool, chapter era")
        after = screen(factors[["cluster", "scalar", "offset"]]).assign(
            code=spec.code, condition=f"refit, {ERA5_PATH}")
        results.append(pd.concat([before, after], ignore_index=True))

    frame = pd.concat(results, ignore_index=True)
    frame.to_csv(out / "refit_control_points.csv", index=False)
    low, high = PLAUSIBLE_SCALAR
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(f"\n=== every fitted pair, screened at scalar [{low}, {high}] and "
              f"crossing {MAX_ZERO_CROSSING_SPEED} m/s")
        print(frame[["code", "condition", "cluster", "scalar", "offset",
                     "crossing", "implausible"]].round(3).to_string(index=False))
        print("\n=== implausible pairs, before and after")
        print(frame.groupby(["code", "condition"], sort=False).agg(
            clusters=("scalar", "size"), implausible=("implausible", "sum"),
            min_scalar=("scalar", "min"), max_scalar=("scalar", "max"),
            ).round(3).to_string())
    print(f"\nwritten: {out / 'refit_control_points.csv'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1], *sys.argv[2:])
