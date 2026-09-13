"""Join correction factors to their cluster geometries, for maps and spatial work.

Ported on 2026-09-13 from the `development` branch. **This is off the critical
path**: neither registered study needs it, and nothing else in
``vwf.extensions.grid`` imports it. It is here because the chapter's figures
come through it and the manuscript will want them.

The original merged geometries to factors with a left join and printed the
result. Two silent failures came out of that and are refused here instead: a
cluster present in the geometries and absent from the factors produced a row of
missing values, and a cluster present in the factors and absent from the
geometries vanished. Either means the two files describe different fits, which
is worth stopping for rather than mapping.

The second function in the original hardcoded a chapter-era directory layout,
``input/country_level_data`` and ``output/runs/turbine_grid``, neither of which
is where this repository keeps anything now. Those are parameters here, with no
defaults, so a caller states where its files are rather than discovering that a
default points nowhere.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

#: The time-slice columns a factors table can carry, in the order they are
#: looked for. One of these names the slice; the rest are absent.
TIME_COLUMNS = ("fixed", "season", "bimonth", "month")

#: What a ``fixed`` factors table calls its single slice.
FIXED_SLICE = "1/1"


def _time_column(factors: pd.DataFrame) -> str | None:
    for name in TIME_COLUMNS:
        if name in factors.columns:
            return name
    return None


def correction_geodataframe(
    factors_csv: str | Path,
    cluster_geoms_geojson: str | Path,
    *,
    time_slice: str | None = None,
    output_path: str | Path | None = None,
) -> gpd.GeoDataFrame:
    """Factors joined to the geometry of the cluster each was fitted for.

    Args:
        factors_csv: a run's ``factors_<slice>_<n>.csv``, or the chapter-era
            ``<CODE>_factors_<slice>_<n>.csv``.
        cluster_geoms_geojson: polygons carrying a ``cluster`` column.
        time_slice: keep only this slice, for example ``winter`` or ``1/1``.
            None keeps every slice, which gives one row per cluster and slice.
        output_path: write the result here. The driver is taken from the
            suffix: ``.geojson``, ``.gpkg`` or ``.shp``, and anything else is
            refused rather than silently written as GeoJSON.

    Returns:
        The joined GeoDataFrame.

    Raises:
        ValueError: if the two files disagree about which clusters exist, if
            the requested slice is not in the table, or if the output suffix is
            not one of the three supported.
    """
    factors = pd.read_csv(factors_csv)
    geoms = gpd.read_file(cluster_geoms_geojson)
    for frame, name in ((factors, "factors"), (geoms, "geometries")):
        if "cluster" not in frame.columns:
            raise ValueError(f"{name} has no 'cluster' column")

    if time_slice is not None:
        column = _time_column(factors)
        if column is None:
            raise ValueError(
                f"a time slice {time_slice!r} was asked for and the factors table has "
                f"none of {list(TIME_COLUMNS)}")
        available = sorted(factors[column].astype(str).unique())
        if time_slice not in available:
            raise ValueError(
                f"time slice {time_slice!r} is not in the {column!r} column; "
                f"it holds {available}")
        factors = factors[factors[column].astype(str) == time_slice]

    factors = factors.assign(cluster=factors["cluster"].astype(int))
    geoms = geoms.assign(cluster=geoms["cluster"].astype(int))
    only_geoms = sorted(set(geoms["cluster"]) - set(factors["cluster"]))
    only_factors = sorted(set(factors["cluster"]) - set(geoms["cluster"]))
    if only_geoms or only_factors:
        raise ValueError(
            f"the two files describe different fits: {len(only_geoms)} clusters have a "
            f"geometry and no factors {only_geoms[:5]}, {len(only_factors)} have factors "
            f"and no geometry {only_factors[:5]}. A left join would have mapped the "
            "first as missing values and dropped the second.")

    joined = geoms.merge(factors, on="cluster", how="left")
    if output_path is not None:
        out = Path(output_path)
        drivers = {".geojson": "GeoJSON", ".gpkg": "GPKG", ".shp": "ESRI Shapefile"}
        if out.suffix not in drivers:
            raise ValueError(
                f"unsupported output suffix {out.suffix!r}; use one of {list(drivers)}")
        out.parent.mkdir(parents=True, exist_ok=True)
        joined.to_file(out, driver=drivers[out.suffix])
    return joined


def country_correction_geodataframes(
    country: str,
    *,
    factors_dir: str | Path,
    geometry_file: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, gpd.GeoDataFrame]:
    """Every factors table of one row, joined to that row's cluster geometry.

    Args:
        country: the region code, used to find ``<CODE>_factors_*.csv`` and to
            name the outputs.
        factors_dir: where that row's factors tables are. **No default**: the
            original pointed at a chapter-era layout that no longer exists, and
            a default that points nowhere fails later and less clearly than an
            argument that is required.
        geometry_file: the cluster polygons for that row.
        output_dir: write one file per factors table here, or None to return
            them without writing.

    Returns:
        One GeoDataFrame per factors table, keyed by the file's stem.

    Raises:
        ValueError: if no factors table is found, which otherwise returns an
            empty result that reads as a row with no corrections.
    """
    factors_dir, geometry = Path(factors_dir), Path(geometry_file)
    tables = sorted(factors_dir.glob(f"{country}_factors_*.csv")) or \
        sorted(factors_dir.glob("factors_*.csv"))
    if not tables:
        raise ValueError(
            f"no factors table for {country} in {factors_dir}; looked for "
            f"'{country}_factors_*.csv' and 'factors_*.csv'")

    out: dict[str, gpd.GeoDataFrame] = {}
    for table in tables:
        # The slice is read from the table's own column rather than parsed out
        # of the file name, which the original did positionally and which
        # breaks on any code or slice containing an underscore.
        columns = pd.read_csv(table, nrows=0).columns
        column = _time_column(pd.DataFrame(columns=columns))
        destination = None
        if output_dir is not None:
            destination = Path(output_dir) / country.lower() / f"{table.stem}.geojson"
        out[table.stem] = correction_geodataframe(
            table, geometry,
            time_slice=FIXED_SLICE if column == "fixed" else None,
            output_path=destination)
    return out
