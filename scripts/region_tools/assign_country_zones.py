"""Assign country grid points to bidding zones using real zone polygons.

The zonal path pairs each grid-point cluster with the bidding zone whose
observations carry the same index (`SE_1` to cluster 0, and so on). That only
works if the clusters are the zones.

Two things had to be fixed for that to hold. The shipped grids' cluster labels
were not the zones at all: measured against the bounding boxes
`generate_country_level_training_data` declares, 0% of the points in three of
the four Swedish clusters fell inside the zone they were labelled with. And
those bounding boxes are themselves approximations of the market boundaries,
which for Norway is not a detail: the NO3/NO5 boundary runs south of Sunnfjord
rather than along the Vestland county line, so a box drawn on the county puts
Guleslettene, Lutelandet, Hennoy and Mehuken in the wrong zone.

This uses real polygons instead (see `configs/curation/zones/README.md` for
sources and licences). Points outside every polygon, which happens offshore
because the Swedish and Danish polygons are land-only, fall back to the nearest
polygon by distance and are reported.

Usage:
    PYTHONPATH=src python scripts/region_tools/assign_country_zones.py SE --dry-run
    PYTHONPATH=src python scripts/region_tools/assign_country_zones.py SE NO
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vwf.config import PyVWFPaths  # noqa: E402

ZONE_DIR = REPO_ROOT / "configs" / "curation" / "zones"

BACKUP_SUFFIX = ".boxzones.bak.csv"


def load_zones(country: str) -> dict[int, "gpd.GeoSeries"]:
    """Cluster index to zone polygon, for one country.

    Zone files are named ``<CC>_<n>.geojson``. Clusters are numbered ``n - 1``,
    matching how ``generate_country_level_training_data`` stamps them onto the
    grid and how ``EntsoeZonalFileSource`` tags the observations.
    """
    paths = sorted(ZONE_DIR.glob(f"{country.upper()}_*.geojson"))
    if not paths:
        raise FileNotFoundError(
            f"No zone polygons for {country} in {ZONE_DIR}. "
            "See that directory's README for how to fetch them."
        )
    zones = {}
    for path in paths:
        suffix = path.stem.split("_", 1)[1]
        if not suffix.isdigit():
            continue  # named zones (Italy's IT_NORD etc.) are not numbered
        zones[int(suffix) - 1] = gpd.read_file(path).geometry.union_all()
    if not zones:
        raise ValueError(f"{country}: zone files exist but none are numbered")
    return zones


def assign(points: pd.DataFrame, zones) -> tuple[pd.Series, int]:
    """Cluster per point: the containing polygon, else the nearest one.

    Returns the assignment and how many points needed the nearest-polygon
    fallback. Sweden's and Denmark's polygons are land-only, so any offshore
    point falls back; Norway's extend offshore and generally do not.
    """
    ordered = sorted(zones)
    geoms = [zones[c] for c in ordered]
    out, fallbacks = [], 0

    for row in points.itertuples():
        point = Point(row.lon, row.lat)
        hit = next((c for c, g in zip(ordered, geoms) if g.contains(point)), None)
        if hit is None:
            fallbacks += 1
            hit = min(ordered, key=lambda c: zones[c].distance(point))
        out.append(hit)

    return pd.Series(out, index=points.index), fallbacks


def process(country: str, *, dry_run: bool) -> None:
    code = country.upper()
    zones = load_zones(code)
    grid_dir = PyVWFPaths.COUNTRY_LEVEL_DATA / "grid_points" / code.lower()
    paths = sorted(
        p for p in grid_dir.glob(f"{code.lower()}_grid_points*.csv")
        if ".bak." not in p.name
    )
    if not paths:
        print(f"{code}: no grid points under {grid_dir}")
        return

    print(f"\n{code}: {len(zones)} zone polygons, {len(paths)} grid file(s)")
    for path in paths:
        grid = pd.read_csv(path)
        new, fallbacks = assign(grid, zones)
        moved = int((grid["cluster"] != new).sum())
        after = grid.assign(cluster=new).groupby("cluster")["capacity"].sum().round(0)

        print(f"  {path.name}: {moved}/{len(grid)} moved, {fallbacks} nearest-polygon "
              f"fallback(s); MW by zone {after.to_dict()}")

        if dry_run:
            continue
        backup = path.with_suffix(BACKUP_SUFFIX)
        if not backup.exists():
            grid.to_csv(backup, index=False)
        grid.assign(cluster=new).to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("countries", nargs="+", help="zonal region codes, e.g. SE NO")
    parser.add_argument("--dry-run", action="store_true", help="report only")
    args = parser.parse_args()
    for country in args.countries:
        process(country, dry_run=args.dry_run)
    if not args.dry_run:
        print("\nRe-run weight_country_grid_points.py --zone-aware so fleet capacity "
              "is not summed across a zone boundary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
