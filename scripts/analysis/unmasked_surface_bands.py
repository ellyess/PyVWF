"""What does the correction surface hold where the chapter refused to answer?

Thesis chapter 4 neutralises every grid cell more than 5 degrees from any
control point, which is 34.5% of the 23,989 cells of the shipped grid. Removing
that mask means those cells receive a real correction instead of unity, and the
LOCO result says a prediction that far from any control point carries no
within-country information. So the values have to be read before they ship.

This measures, for the undivided pool of 1,729 control points kriged over the
shipped grid with **no mask at all**:

- how the cells divide into distance bands, on both metrics, since the
  chapter's 5-degree threshold is stated in Euclidean degrees and new work uses
  great-circle distance;
- what scalar and offset each band holds;
- what those turn into at a reference wind speed, which is the form a user
  meets them in: a scalar and an offset are hard to read, and 8 m/s becoming
  2 m/s is not;
- how many cells of each band lie inside any region of interest at all. A cell
  over open ocean or the Sahara receives a correction that nothing will ever
  read, and separating those from the cells a user could plausibly sample is
  the difference between a large number and a consequential one.

Read-only with respect to the tree. Writes its results under ``<out_dir>``.

Usage, from the repository root:

    PYVWF_INPUT=input/combined PYTHONPATH=src python \\
        scripts/analysis/unmasked_surface_bands.py <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vwf.extensions.grid import interpolation as interp, surface

POOL = Path("output/pyvwf_to_grid/all_corrections_centroids.csv")
SHAPES = Path("input/reference/shapes")
ONSHORE, OFFSHORE = SHAPES / "country_shapes.geojson", SHAPES / "offshore_shapes.geojson"

GRID_LON = np.arange(-10.0, 30.01, 0.25)
GRID_LAT = np.arange(35.0, 72.01, 0.25)

#: Band edges in Euclidean degrees, the units the chapter's threshold is in.
#: 5.0 is the chapter's mask; 2.0 is the lower edge of the band the deliverable
#: treats as its standard.
EDGES = (0.0, 1.0, 2.0, 5.0, np.inf)

#: The speed a correction is read at, for the physical column. Close to the
#: daily-mean speeds these fleets actually see.
REFERENCE_SPEED = 8.0

#: The ends of the curve table, which is what a corrected speed has to land
#: inside to produce a capacity factor at all.
CURVE_MIN, CURVE_MAX = 0.0, 40.0


def bands(distance: np.ndarray) -> pd.Categorical:
    labels = ["0 to 1", "1 to 2", "2 to 5", "beyond 5"]
    return pd.cut(distance, list(EDGES), labels=labels, right=False,
                  include_lowest=True)


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(POOL)

    print(f"kriging {len(pool)} control points over "
          f"{len(GRID_LON)} by {len(GRID_LAT)} cells, unmasked ...", flush=True)
    scalar, offset = interp.to_grid(
        interp.kriging_at, pool, GRID_LON, GRID_LAT,
        variogram_model=interp.KRIGING_VARIOGRAM,
        coordinates_type=interp.KRIGING_COORDINATES)

    lon_grid, lat_grid = np.meshgrid(GRID_LON, GRID_LAT)
    flat_lon, flat_lat = lon_grid.ravel(), lat_grid.ravel()
    degrees = interp.distance_to_nearest(pool, flat_lon, flat_lat, metric="degrees")
    great_circle = interp.distance_to_nearest(pool, flat_lon, flat_lat,
                                              metric="great_circle")

    on_area = surface.area_mask(GRID_LON, GRID_LAT, ONSHORE, name="on").values.ravel()
    off_area = surface.area_mask(GRID_LON, GRID_LAT, OFFSHORE, name="off").values.ravel()

    corrected = REFERENCE_SPEED * scalar.ravel() + offset.ravel()
    cells = pd.DataFrame({
        "lon": flat_lon, "lat": flat_lat,
        "scalar": scalar.ravel(), "offset": offset.ravel(),
        "distance_degrees": degrees, "distance_great_circle_degrees": great_circle,
        "in_region": on_area | off_area,
        "corrected_at_reference": corrected,
        "off_curve_at_reference": (corrected < CURVE_MIN) | (corrected > CURVE_MAX),
        "band": bands(degrees)})
    cells.to_csv(out / "unmasked_surface_cells.csv", index=False)

    total = len(cells)
    print(f"\n=== the grid: {total} cells, {len(GRID_LON)} by {len(GRID_LAT)}")
    print(f"    reference speed {REFERENCE_SPEED} m/s, curve table "
          f"{CURVE_MIN} to {CURVE_MAX} m/s")

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n=== how the cells divide, and how much of each band anyone can reach")
        split = cells.groupby("band", observed=False).agg(
            cells=("scalar", "size"), in_region=("in_region", "sum"))
        split["share of grid"] = split["cells"] / total
        split["share in a region"] = split["in_region"] / split["cells"]
        print(split.round(4).to_string())

        print("\n=== what each band holds")
        for column in ("scalar", "offset", "corrected_at_reference"):
            stat = cells.groupby("band", observed=False)[column].describe(
                percentiles=[0.05, 0.5, 0.95])
            print(f"\n{column}:")
            print(stat[["min", "5%", "50%", "95%", "max"]].round(3).to_string())

        print("\n=== the same, for cells inside a region of interest only")
        inside = cells[cells["in_region"]]
        for column in ("scalar", "corrected_at_reference"):
            stat = inside.groupby("band", observed=False)[column].describe(
                percentiles=[0.05, 0.5, 0.95])
            print(f"\n{column}, in region:")
            print(stat[["min", "5%", "50%", "95%", "max"]].round(3).to_string())

        print("\n=== corrections that cannot produce a capacity factor at the "
              "reference speed")
        bad = cells.groupby("band", observed=False).agg(
            off_curve=("off_curve_at_reference", "sum"),
            off_curve_in_region=("off_curve_at_reference",
                                 lambda s: int((s & cells.loc[s.index, "in_region"]).sum())))
        bad["share of band"] = bad["off_curve"] / split["cells"]
        print(bad.round(4).to_string())

        print("\n=== the two metrics disagree on which cells are beyond 5")
        both = pd.crosstab(bands(degrees), bands(great_circle))
        print(both.to_string())

    print(f"\nwritten: {out / 'unmasked_surface_cells.csv'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
