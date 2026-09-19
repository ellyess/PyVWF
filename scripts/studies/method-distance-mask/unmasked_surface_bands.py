"""What does the correction surface hold where the chapter refused to answer?

Thesis chapter 4 neutralises every grid cell more than 5 degrees from any
control point, which is 34.5% of the 23,989 cells of the shipped grid. Removing
that mask means those cells receive a real correction instead of unity, and the
LOCO result says a prediction that far from any control point carries no
within-country information. So the values have to be read before they ship.

This measures, for the undivided pool of 1,729 control points kriged over the
shipped grid with **no mask at all**:

- how the cells divide into distance bands. The chapter's 5-degree threshold
  is Euclidean in degrees, and the great-circle metric this project adopted for
  new work returns kilometres, so the two cannot share a threshold. Both are
  carried per cell and the bands are cut on the chapter's;
- what scalar and offset each band holds, and **the kriging variance**, which
  is what the chapter's distance threshold is a proxy for and which the
  estimator already computes;
- what those turn into at a reference wind speed, which is the form a user
  meets them in: a scalar and an offset are hard to read, and 8 m/s becoming
  2 m/s is not;
- where the cells holding unusable corrections sit under **two** screens, which
  disagree: whether 8 m/s corrects off the curve table, and whether the
  correction's zero crossing removes ordinary low winds;
- where the cells holding unusable corrections actually sit, and whether the
  interpolation overshoots its own control points or faithfully reports a pool
  that disagrees with itself;
- how many cells of each band lie inside any region of interest at all. A cell
  over open ocean or the Sahara receives a correction that nothing will ever
  read, and separating those from the cells a user could plausibly sample is
  the difference between a large number and a consequential one.

Read-only with respect to the tree. Writes its results under ``<out_dir>``.

Usage, from the repository root:

    PYVWF_INPUT=input/combined PYTHONPATH=src python \\
        scripts/studies/method-distance-mask/unmasked_surface_bands.py <out_dir>
"""

from pathlib import Path

import numpy as np
import pandas as pd

from vwf.cli.common import make_parser
from vwf.extensions.grid import interpolation as interp, surface
from vwf.extensions.grid.surface import PLAUSIBLE_SCALAR, flag_implausible, zero_crossing_speed

POOL = Path("output/pyvwf_to_grid/all_corrections_centroids.csv")
SHAPES = Path("input/reference/shapes")

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

#: A corrected speed at the reference outside these is on the curve but is not
#: a credible value. Read as a screen, not as a threshold anything acts on.
EXTREME_LOW, EXTREME_HIGH = 1.0, 20.0

#: How many of a cell's nearest control points are read when asking whether the
#: interpolation overshot them or reported them faithfully.
NEIGHBOURS = 5


def bands(distance: np.ndarray) -> pd.Categorical:
    labels = ["0 to 1", "1 to 2", "2 to 5", "beyond 5"]
    return pd.cut(distance, list(EDGES), labels=labels, right=False, include_lowest=True)


def main(out_dir: str, pool_path: Path = POOL, shapes: Path = SHAPES) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(pool_path)
    onshore = Path(shapes) / "country_shapes.geojson"
    offshore = Path(shapes) / "offshore_shapes.geojson"

    lon_grid, lat_grid = np.meshgrid(GRID_LON, GRID_LAT)
    flat_lon, flat_lat = lon_grid.ravel(), lat_grid.ravel()

    print(
        f"kriging {len(pool)} control points over "
        f"{len(GRID_LON)} by {len(GRID_LAT)} cells, unmasked ...",
        flush=True,
    )
    # Called directly rather than through to_grid, because the variance is the
    # point: the estimator computes it either way and the chapter threw it away.
    (scalar, offset), (scalar_var, offset_var) = interp.kriging_at(
        pool,
        flat_lon,
        flat_lat,
        variogram_model=interp.KRIGING_VARIOGRAM,
        coordinates_type=interp.KRIGING_COORDINATES,
        with_variance=True,
    )

    degrees = interp.distance_to_nearest(pool, flat_lon, flat_lat, metric="degrees")
    great_circle_km = interp.distance_to_nearest(pool, flat_lon, flat_lat, metric="great_circle")

    on_area = surface.area_mask(GRID_LON, GRID_LAT, onshore, name="on").values.ravel()
    off_area = surface.area_mask(GRID_LON, GRID_LAT, offshore, name="off").values.ravel()

    corrected = REFERENCE_SPEED * scalar + offset
    cells = pd.DataFrame(
        {
            "lon": flat_lon,
            "lat": flat_lat,
            "scalar": scalar,
            "offset": offset,
            "scalar_variance": scalar_var,
            "offset_variance": offset_var,
            "distance_degrees": degrees,
            "distance_great_circle_km": great_circle_km,
            "in_region": on_area | off_area,
            "corrected_at_reference": corrected,
            "off_curve_at_reference": (corrected < CURVE_MIN) | (corrected > CURVE_MAX),
            "band": bands(degrees),
        }
    )
    cells.to_csv(out / "unmasked_surface_cells.csv", index=False)

    total = len(cells)
    print(f"\n=== the grid: {total} cells, {len(GRID_LON)} by {len(GRID_LAT)}")
    print(f"    reference speed {REFERENCE_SPEED} m/s, curve table {CURVE_MIN} to {CURVE_MAX} m/s")

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\n=== how the cells divide, and how much of each band anyone can reach")
        split = cells.groupby("band", observed=False).agg(
            cells=("scalar", "size"), in_region=("in_region", "sum")
        )
        split["share of grid"] = split["cells"] / total
        split["share in a region"] = split["in_region"] / split["cells"]
        print(split.round(4).to_string())

        print("\n=== what each band holds")
        for column in ("scalar", "offset", "corrected_at_reference", "scalar_variance"):
            stat = cells.groupby("band", observed=False)[column].describe(
                percentiles=[0.05, 0.5, 0.95]
            )
            print(f"\n{column}:")
            print(stat[["min", "5%", "50%", "95%", "max"]].round(3).to_string())

        print("\n=== the same, for cells inside a region of interest only")
        inside = cells[cells["in_region"]]
        for column in ("scalar", "corrected_at_reference"):
            stat = inside.groupby("band", observed=False)[column].describe(
                percentiles=[0.05, 0.5, 0.95]
            )
            print(f"\n{column}, in region:")
            print(stat[["min", "5%", "50%", "95%", "max"]].round(3).to_string())

        print("\n=== corrections that cannot produce a capacity factor at the reference speed")
        bad = cells.groupby("band", observed=False).agg(
            off_curve=("off_curve_at_reference", "sum"),
            off_curve_in_region=(
                "off_curve_at_reference",
                lambda s: int((s & cells.loc[s.index, "in_region"]).sum()),
            ),
        )
        bad["share of band"] = bad["off_curve"] / split["cells"]
        print(bad.round(4).to_string())

        print("\n=== does the kriging variance separate what distance does not?")
        extreme = (cells["corrected_at_reference"] < EXTREME_LOW) | (
            cells["corrected_at_reference"] > EXTREME_HIGH
        )
        print(
            f"    cells whose corrected speed at {REFERENCE_SPEED} m/s is "
            f"below {EXTREME_LOW:g} or above {EXTREME_HIGH:g} m/s: "
            f"{int(extreme.sum())}"
        )
        print(
            cells.groupby("band", observed=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "cells": len(g),
                        "extreme": int(
                            (
                                (g["corrected_at_reference"] < EXTREME_LOW)
                                | (g["corrected_at_reference"] > EXTREME_HIGH)
                            ).sum()
                        ),
                        "median scalar variance": g["scalar_variance"].median(),
                    }
                ),
                include_groups=False,
            )
            .round(4)
            .to_string()
        )
        print("\n    the same cells, by scalar-variance quintile:")
        quintile = pd.qcut(
            cells["scalar_variance"], 5, labels=["lowest", "2nd", "3rd", "4th", "highest"]
        )
        print(
            cells.assign(q=quintile)
            .groupby("q", observed=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "cells": len(g),
                        "extreme": int(
                            (
                                (g["corrected_at_reference"] < EXTREME_LOW)
                                | (g["corrected_at_reference"] > EXTREME_HIGH)
                            ).sum()
                        ),
                        "median distance degrees": g["distance_degrees"].median(),
                    }
                ),
                include_groups=False,
            )
            .round(3)
            .to_string()
        )

        print(f"\n=== do the {NEIGHBOURS} nearest control points explain the extreme cells?")
        near = cells[cells["distance_degrees"] < interp.MAX_DISTANCE_DEG].copy()
        distances = interp.degree_distances(
            near[["lon", "lat"]].to_numpy(float), pool[["lon", "lat"]].to_numpy(float), "degrees"
        )
        nearest = np.argpartition(distances, NEIGHBOURS, axis=1)[:, :NEIGHBOURS]
        neighbour_scalars = pool["scalar"].to_numpy(float)[nearest]
        low, high = neighbour_scalars.min(axis=1), neighbour_scalars.max(axis=1)
        near["neighbour_spread"] = high - low
        near["outside_neighbours"] = (near["scalar"] > high) | (near["scalar"] < low)
        near["extreme"] = extreme[near.index]
        print(
            near.groupby("extreme")
            .agg(
                cells=("scalar", "size"),
                median_neighbour_spread=("neighbour_spread", "median"),
                share_outside_neighbour_range=("outside_neighbours", "mean"),
                median_distance=("distance_degrees", "median"),
            )
            .round(3)
            .to_string()
        )

        print("\n=== the second screen: does the correction refuse ordinary low winds?")
        cells["zero_crossing_speed"] = zero_crossing_speed(cells["scalar"], cells["offset"])
        cells["implausible"] = flag_implausible(cells["scalar"], cells["offset"])
        low_bound, high_bound = PLAUSIBLE_SCALAR
        screen = cells.groupby("band", observed=False).agg(
            cells=("scalar", "size"),
            implausible=("implausible", "sum"),
            scalar_out_of_bounds=(
                "scalar",
                lambda s: int(((s < low_bound) | (s > high_bound)).sum()),
            ),
        )
        screen["share"] = screen["implausible"] / screen["cells"]
        print(screen.round(4).to_string())
        caught = int(cells.loc[cells["band"] == "beyond 5", "implausible"].sum())
        total_bad = int(cells["implausible"].sum())
        beyond = int((cells["band"] == "beyond 5").sum())
        print(
            f"    a 5-degree cut deletes {beyond} cells to remove {caught} of "
            f"{total_bad} implausible ones: {caught / total_bad:.1%} of the defect, "
            f"and {beyond - caught} good cells discarded"
        )

        print("\n=== the two metrics are not in the same units")
        print(
            cells.groupby("band", observed=False)["distance_great_circle_km"]
            .describe(percentiles=[0.5])[["min", "50%", "max"]]
            .round(1)
            .to_string()
        )

    print(f"\nwritten: {out / 'unmasked_surface_cells.csv'}")


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<out_dir>``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("out_dir", help="Directory for the outputs, under output/")
    parser.add_argument(
        "--pool", type=Path, default=POOL, help=f"The control-point pool (default: {POOL})"
    )
    parser.add_argument(
        "--shapes",
        type=Path,
        default=SHAPES,
        help=f"The onshore and offshore GeoJSON directory (default: {SHAPES})",
    )
    args = parser.parse_args(argv)
    main(args.out_dir, pool_path=args.pool, shapes=args.shapes)


if __name__ == "__main__":
    cli()
