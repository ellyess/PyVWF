"""Give country-level grid points real capacity weights from the GWPT.

The country-level workflow simulates a uniform grid where every point carries
the same synthetic capacity, then compares the aggregate against national
generation over national installed capacity. Those are two different spatial
averages: the model's is weighted by land area, the observation's by where the
fleet actually is. Real fleets are concentrated, so the gap between them is
absorbed into the correction factors as if it were reanalysis bias.

This script closes the gap. Every operating wind project in the Global Wind
Power Tracker is assigned to its nearest grid point, and each point's capacity
becomes the sum of what it carries. Points with no fleet are dropped, so the
simulated country stops averaging over empty countryside.

GWPT records a start year and a retirement year, so ``--year`` builds the fleet
as it stood at a point in time. That matters here: NL's country-level run
failed because the fleet doubled between training and test, and uniform weights
cannot represent that at all.

The GWPT is CC-BY-4.0. See docs/guides/data-sources.md.

Usage:
    PYTHONPATH=src python scripts/region_tools/weight_country_grid_points.py NL --dry-run
    PYTHONPATH=src python scripts/region_tools/weight_country_grid_points.py NL FR ES
    PYTHONPATH=src python scripts/region_tools/weight_country_grid_points.py --all --year 2021
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vwf.config import PyVWFPaths  # noqa: E402

GWPT_PATH = PyVWFPaths.INPUT_ROOT / "reference" / "gwpt" / (
    "Global-Wind-Power-Tracker-February-2026.xlsx"
)

#: Region code to the GWPT's ``Country/Area`` spelling.
GWPT_COUNTRY = {
    "BE": "Belgium",
    "ES": "Spain",
    "FR": "France",
    "IE": "Ireland",
    "IT": "Italy",
    "NL": "Netherlands",
    "NO": "Norway",
    "PT": "Portugal",
    "SE": "Sweden",
}

BACKUP_SUFFIX = ".uniform.bak.csv"

#: Records the tracker marks operating that independent registers contradict.
#: Each row states its evidence; see the file.
EXCLUSIONS_PATH = REPO_ROOT / "configs" / "curation" / "gwpt_exclusions.csv"


def load_exclusions(path: Path = EXCLUSIONS_PATH) -> set[str]:
    """GEM phase IDs to drop, keyed so a renamed project stays excluded."""
    if not path.is_file():
        return set()
    return set(pd.read_csv(path)["gem_phase_id"].astype(str))


def tag_zones(frame: pd.DataFrame, country: str) -> pd.DataFrame:
    """Add a ``zone`` column from the real bidding-zone polygons.

    Points outside every polygon take the nearest one, which matters offshore:
    the Swedish and Danish polygons are land-only.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    zone_dir = REPO_ROOT / "configs" / "curation" / "zones"
    paths = sorted(zone_dir.glob(f"{country.upper()}_*.geojson"))
    zones = {
        p.stem: gpd.read_file(p).geometry.union_all()
        for p in paths
        if p.stem.split("_", 1)[1].isdigit()
    }
    if not zones:
        raise FileNotFoundError(f"No numbered zone polygons for {country} in {zone_dir}")

    def zone_of(lon, lat):
        point = Point(lon, lat)
        hit = next((n for n, g in zones.items() if g.contains(point)), None)
        return hit if hit else min(zones, key=lambda n: zones[n].distance(point))

    out = frame.copy()
    out["zone"] = [zone_of(r.lon, r.lat) for r in out.itertuples()]
    return out


def load_gwpt(path: Path = GWPT_PATH) -> pd.DataFrame:
    """Read the tracker's Data sheet."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Global Wind Power Tracker not found at {path}. "
            "See docs/guides/data-sources.md for where to download it."
        )
    return pd.read_excel(path, sheet_name="Data")


def fleet_for(gwpt: pd.DataFrame, country: str, year: int | None) -> pd.DataFrame:
    """Operating, geolocated projects for one country, optionally as of a year."""
    name = GWPT_COUNTRY.get(country.upper())
    if name is None:
        raise KeyError(f"No GWPT country name mapped for {country!r}")

    fleet = gwpt[gwpt["Country/Area"] == name].copy()
    fleet = fleet[fleet["Status"].astype(str).str.lower() == "operating"]
    fleet = fleet.dropna(subset=["Latitude", "Longitude", "Capacity (MW)"])

    excluded = load_exclusions()
    if excluded and "GEM phase ID" in fleet.columns:
        drop = fleet["GEM phase ID"].astype(str).isin(excluded)
        if drop.any():
            names = ", ".join(fleet.loc[drop, "Project Name"].astype(str))
            print(f"  excluding {int(drop.sum())} curated GWPT record(s): {names}")
            fleet = fleet[~drop]

    if year is not None:
        start = pd.to_numeric(fleet["Start year"], errors="coerce")
        retired = pd.to_numeric(fleet["Retired year"], errors="coerce")
        # Projects with no start year are kept: the tracker often omits it for
        # older sites, and dropping them would understate the historic fleet.
        fleet = fleet[(start.isna() | (start <= year)) & (retired.isna() | (retired > year))]

    return fleet[["Latitude", "Longitude", "Capacity (MW)"]].rename(
        columns={"Latitude": "lat", "Longitude": "lon", "Capacity (MW)": "mw"}
    )


def assign_to_grid(
    grid: pd.DataFrame, fleet: pd.DataFrame, *, zone_aware: bool = False
) -> pd.Series:
    """Sum fleet capacity onto the nearest grid point, indexed like ``grid``.

    Longitude is scaled by cos(latitude) so "nearest" is nearest on the ground
    rather than nearest in degrees, which at Nordic latitudes differ by roughly
    a factor of two.

    With ``zone_aware``, a project can only be assigned to a grid point in its
    own bidding zone. Without it, "nearest" happily crosses a zone boundary, and
    the zonal path then compares a cluster's simulated output against an
    observation for a zone whose turbines are partly counted somewhere else.
    Both frames need a ``zone`` column for this; projects whose zone has no grid
    point fall back to the unrestricted nearest point and are counted in the
    return value's ``attrs``.
    """
    if fleet.empty:
        return pd.Series(0.0, index=grid.index)

    scale = float(np.cos(np.deg2rad(grid["lat"].mean())))
    grid_xy = np.column_stack([grid["lon"].to_numpy() * scale, grid["lat"].to_numpy()])
    fleet_xy = np.column_stack([fleet["lon"].to_numpy() * scale, fleet["lat"].to_numpy()])

    # One grid has a few hundred points and one fleet a few thousand projects,
    # so the dense distance matrix is small and avoids a scipy dependency here.
    d2 = ((fleet_xy[:, None, :] - grid_xy[None, :, :]) ** 2).sum(axis=2)

    stranded = 0
    if zone_aware:
        if "zone" not in fleet.columns or "zone" not in grid.columns:
            raise ValueError("zone_aware assignment needs a 'zone' column on both frames")
        grid_zone = grid["zone"].to_numpy()
        masked = d2.copy()
        for i, zone in enumerate(fleet["zone"].to_numpy()):
            allowed = grid_zone == zone
            if not allowed.any():
                stranded += 1
                continue
            masked[i, ~allowed] = np.inf
        d2 = masked

    nearest = d2.argmin(axis=1)
    weights = np.zeros(len(grid))
    np.add.at(weights, nearest, fleet["mw"].to_numpy())
    out = pd.Series(weights, index=grid.index)
    out.attrs["stranded"] = stranded
    return out


def report(country: str, grid: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Per-cluster comparison of area weighting against fleet weighting."""
    frame = grid[["cluster"]].copy()
    frame["mw"] = weights.to_numpy()
    frame["n"] = 1

    by_cluster = frame.groupby("cluster", as_index=False)[["n", "mw"]].sum()
    by_cluster["area_share"] = by_cluster["n"] / by_cluster["n"].sum()
    total_mw = by_cluster["mw"].sum()
    by_cluster["fleet_share"] = (
        by_cluster["mw"] / total_mw if total_mw > 0 else np.nan
    )
    by_cluster["shift_pp"] = 100 * (by_cluster["fleet_share"] - by_cluster["area_share"])
    return by_cluster


def _apply(grid: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    out = grid.copy()
    out["capacity"] = weights.to_numpy()
    if "weight" in out.columns:
        # create_sampling_points writes weight=1.0 for grid methods. Nothing in
        # the correction path reads it, but leaving it at 1.0 next to a real
        # capacity makes the file look uniformly weighted when it is not.
        out["weight"] = out["capacity"]
    return out


def process_per_year(
    country: str,
    gwpt: pd.DataFrame,
    years: range,
    *,
    dry_run: bool,
    zone_aware: bool = False,
) -> None:
    """Write one grid per year, on a point set fixed across the whole range.

    The point set is the union over the range, so every year's file carries the
    same points and the same clusters and only the weights move. That keeps the
    cluster count stable between the train and test splits, which the run's
    ``cluster_list`` has to match, while still letting the fleet grow: NL more
    than doubled between its training window and its test year, and a single
    snapshot cannot represent that at all.
    """
    code = country.upper()
    grid_dir = PyVWFPaths.COUNTRY_LEVEL_DATA / "grid_points" / code.lower()
    base_path = grid_dir / f"{code.lower()}_grid_points.csv"
    if not base_path.is_file():
        print(f"{code}: no grid points at {base_path}, skipped")
        return

    base = pd.read_csv(base_path)
    # Rebuild from the uniform grid when one exists, so the union is taken over
    # every candidate point rather than only those the 2021 snapshot kept.
    uniform_path = base_path.with_suffix(BACKUP_SUFFIX)
    grid = pd.read_csv(uniform_path) if uniform_path.is_file() else base

    if zone_aware:
        # Zones are re-derived here rather than read from the grid's cluster
        # column, because the point set above comes from the uniform backup,
        # which predates any zone assignment. Deriving them removes the ordering
        # trap: this step is correct whether or not assign_country_zones.py has
        # run, and it writes the cluster it used.
        grid = tag_zones(grid, code)
        grid["cluster"] = grid["zone"].str.rsplit("_", n=1).str[-1].astype(int) - 1

    def weights_for(year: int) -> pd.Series:
        fleet = fleet_for(gwpt, code, year)
        if zone_aware and not fleet.empty:
            fleet = tag_zones(fleet, code)
        return assign_to_grid(grid, fleet, zone_aware=zone_aware)

    per_year = {year: weights_for(year) for year in years}
    stranded = sum(w.attrs.get("stranded", 0) for w in per_year.values())
    if stranded:
        print(f"  {stranded} project-years had no grid point in their own zone")
    ever = pd.concat(per_year.values(), axis=1).max(axis=1)
    kept = grid[ever > 0].reset_index(drop=True)
    keep_mask = (ever > 0).to_numpy()

    print(
        f"\n{code} {years.start}-{years.stop - 1}: {len(kept)} of {len(grid)} points "
        f"ever carry fleet, {kept['cluster'].nunique()} clusters"
    )
    totals = {y: float(w[keep_mask].sum()) for y, w in per_year.items()}
    span = f"{totals[years.start]:,.0f} MW -> {totals[years.stop - 1]:,.0f} MW"
    print(f"  fleet {span}")

    if dry_run:
        print("  dry run, nothing written")
        return

    for year, weights in per_year.items():
        out = _apply(kept, weights[keep_mask].reset_index(drop=True))
        out.to_csv(grid_dir / f"{code.lower()}_grid_points_{year}.csv", index=False)
    print(f"  wrote {len(per_year)} year-specific grids to {grid_dir.name}/")


def process(
    country: str,
    gwpt: pd.DataFrame,
    *,
    year: int | None,
    dry_run: bool,
    keep_empty: bool,
) -> None:
    code = country.upper()
    grid_path = (
        PyVWFPaths.COUNTRY_LEVEL_DATA / "grid_points" / code.lower()
        / f"{code.lower()}_grid_points.csv"
    )
    if not grid_path.is_file():
        print(f"{code}: no grid points at {grid_path}, skipped")
        return

    grid = pd.read_csv(grid_path)
    fleet = fleet_for(gwpt, code, year)
    weights = assign_to_grid(grid, fleet)

    table = report(code, grid, weights)
    as_of = f" as of {year}" if year is not None else ""
    print(f"\n{code}{as_of}: {len(fleet)} projects, {fleet['mw'].sum():,.0f} MW "
          f"over {len(grid)} grid points")
    print(table.round(3).to_string(index=False))
    empty = int((weights == 0).sum())
    print(f"  grid points with no fleet: {empty} of {len(grid)}")
    worst = table["shift_pp"].abs().max()
    print(f"  largest cluster reweighting: {worst:.1f} percentage points")

    out = _apply(grid, weights)
    if not keep_empty:
        out = out[out["capacity"] > 0].reset_index(drop=True)

    lost = set(grid["cluster"].unique()) - set(out["cluster"].unique())
    if lost:
        print(f"  WARNING: clusters {sorted(lost)} have no fleet and were emptied")

    if dry_run:
        print("  dry run, nothing written")
        return

    backup = grid_path.with_suffix(BACKUP_SUFFIX)
    if not backup.exists():
        grid.to_csv(backup, index=False)
        print(f"  backed up the uniform grid to {backup.name}")
    out.to_csv(grid_path, index=False)
    print(f"  wrote {len(out)} capacity-weighted points to {grid_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("countries", nargs="*", help="region codes, e.g. NL FR ES")
    parser.add_argument("--all", action="store_true", help="every mapped country")
    parser.add_argument(
        "--year",
        type=int,
        help="build the fleet as it stood in this year (uses GWPT start/retired years)",
    )
    parser.add_argument(
        "--per-year",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="also write one grid per year over this inclusive range, on a "
        "point set fixed across it",
    )
    parser.add_argument(
        "--zone-aware",
        action="store_true",
        help="with --per-year: never assign a project to a grid point outside "
        "its own bidding zone (requires configs/curation/zones polygons)",
    )
    parser.add_argument("--dry-run", action="store_true", help="report only")
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep grid points with no fleet, at zero capacity",
    )
    parser.add_argument("--gwpt", type=Path, default=GWPT_PATH)
    args = parser.parse_args()

    countries = sorted(GWPT_COUNTRY) if args.all else [c.upper() for c in args.countries]
    if not countries:
        parser.error("give at least one country code, or --all")

    gwpt = load_gwpt(args.gwpt)
    for country in countries:
        if args.per_year:
            start, end = args.per_year
            process_per_year(
                country, gwpt, range(start, end + 1),
                dry_run=args.dry_run, zone_aware=args.zone_aware,
            )
        else:
            process(
                country,
                gwpt,
                year=args.year,
                dry_run=args.dry_run,
                keep_empty=args.keep_empty,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
