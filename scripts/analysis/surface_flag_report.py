"""What share of the product's cells does the plausibility flag reject, by band?

`docs/findings/method-distance-mask.md` establishes that geometry does not
select the cells holding unusable corrections: they sit near the control
points, not far from them. `correction_surface` therefore guards on the
correction's own behaviour instead, and this reports what that guard actually
marks, on the real pool, through the route the product uses.

**This is the product's surface, not the chapter's.** It is split by declared
``cluster_mode`` into an onshore and an offshore pool, each kriged separately,
and it corrects every cell rather than neutralising any. The finding's tables
are from one undivided pool with no mask at all. The two are different
surfaces and the flag's share is a property of this one.

Read-only with respect to the tree. Writes its results under ``<out_dir>``.

Usage, from the repository root:

    PYVWF_INPUT=input/combined PYTHONPATH=src python \\
        scripts/analysis/surface_flag_report.py <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vwf.extensions.grid import surface

POOL = Path("output/pyvwf_to_grid/all_corrections_centroids.csv")
SHAPES = Path("input/reference/shapes")
ONSHORE, OFFSHORE = SHAPES / "country_shapes.geojson", SHAPES / "offshore_shapes.geojson"

GRID_LON = np.arange(-10.0, 30.01, 0.25)
GRID_LAT = np.arange(35.0, 72.01, 0.25)

EDGES = (0.0, 1.0, 2.0, 5.0, np.inf)
LABELS = ["0 to 1", "1 to 2", "2 to 5", "beyond 5"]


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(POOL)

    print(f"building the product surface from {len(pool)} control points ...", flush=True)
    field = surface.correction_surface(
        pool, GRID_LON, GRID_LAT, onshore_geojson=ONSHORE, offshore_geojson=OFFSHORE,
        method="kriging", n_closest_onshore=None, n_closest_offshore=None)
    print(f"  onshore {field.attrs['n_control_points_onshore']}, "
          f"offshore {field.attrs['n_control_points_offshore']}, "
          f"outside areas: {field.attrs['outside_areas']}", flush=True)

    cells = pd.DataFrame({
        "scalar": field["scalar"].values.ravel(),
        "offset": field["offset"].values.ravel(),
        "zero_crossing_speed": field["zero_crossing_speed"].values.ravel(),
        "plausible": field["plausible"].values.ravel(),
        "distance_degrees": field["distance_to_control_deg"].values.ravel(),
        "n_control_within_horizon": field["n_control_within_horizon"].values.ravel(),
        "in_region": (field["is_onshore_area"] | field["is_offshore_area"]).values.ravel(),
    })
    cells["band"] = pd.cut(cells["distance_degrees"], list(EDGES), labels=LABELS,
                           right=False, include_lowest=True)
    cells.to_csv(out / "surface_flag_cells.csv", index=False)

    low, high = surface.PLAUSIBLE_SCALAR
    cells["scalar_out_of_bounds"] = (cells["scalar"] < low) | (cells["scalar"] > high)
    cells["crossing_too_high"] = (
        cells["zero_crossing_speed"] > surface.MAX_ZERO_CROSSING_SPEED)

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(f"\n=== the flag, by band. {len(cells)} cells, "
              f"{int((~cells['plausible']).sum())} marked implausible")
        table = cells.groupby("band", observed=False).agg(
            cells=("plausible", "size"),
            implausible=("plausible", lambda s: int((~s).sum())),
            scalar_out_of_bounds=("scalar_out_of_bounds", "sum"),
            crossing_too_high=("crossing_too_high", "sum"))
        table["implausible share"] = table["implausible"] / table["cells"]
        print(table.round(4).to_string())

        print("\n=== the same, inside a region of interest only")
        inside = cells[cells["in_region"]]
        table = inside.groupby("band", observed=False).agg(
            cells=("plausible", "size"),
            implausible=("plausible", lambda s: int((~s).sum())))
        table["implausible share"] = table["implausible"] / table["cells"]
        print(table.round(4).to_string())

        print("\n=== what the flag rejects, against what it keeps")
        print(cells.groupby("plausible", observed=False).agg(
            cells=("scalar", "size"),
            median_distance_degrees=("distance_degrees", "median"),
            median_support=("n_control_within_horizon", "median"),
            median_scalar=("scalar", "median"),
            median_offset=("offset", "median")).round(3).to_string())
    print(f"\nwritten: {out / 'surface_flag_cells.csv'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
