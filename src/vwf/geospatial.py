"""Label points as onshore or offshore against region geometries.

Ported from the `development` branch on 2026-09-13, where it supported the
gridded-correction work of thesis chapter 4 (`vwf.extensions.grid` calls
:func:`categorize_points_spatial_join` to split control points by domain before
interpolating them). It is kept separate from :mod:`vwf.clustering`, which owns
the region shapes for fitting, because this answers a different question: not
which cluster a unit belongs to, but which side of the coastline it is on.

Two implementations of the same classification are offered, and they are
expected to agree. The spatial join is the one to use; the point-in-polygon
loop is the fallback when no spatial index is available, and it doubles as the
independent route for checking the join.

**A point is onshore or offshore only if it lies strictly inside a geometry.**
A point on a boundary, and a point in neither set of shapes, is ``unknown``,
which is a third answer and not a synonym for offshore. Callers that need a
binary split have to say what they do with ``unknown`` rather than inheriting a
default from here.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.prepared import prep

#: The three values the classification can take. ``unknown`` is a result.
DOMAINS = ("onshore", "offshore", "unknown")


def load_region_geometry(geojson_path: Path | str) -> gpd.GeoDataFrame:
    """Load a GeoJSON file and return its geometries in EPSG:4326.

    Args:
        geojson_path: Path to the GeoJSON file holding region geometries.

    Returns:
        GeoDataFrame in EPSG:4326.

    Raises:
        ValueError: If the file holds no geometries. An empty shape file
            classifies every point as ``unknown``, silently, so it is refused
            here rather than discovered in a result.
    """
    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        raise ValueError(f"No geometries found in {geojson_path}")
    return gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")


def union_geometries(geojson_path: Path | str):
    """The union of every geometry in a GeoJSON file."""
    return load_region_geometry(geojson_path).geometry.union_all()


def _resolve(
    is_onshore: np.ndarray, is_offshore: np.ndarray, index: pd.Index, prefer_onshore: bool
) -> pd.Series:
    """Turn two membership masks into one label per point.

    A point inside both sets of shapes takes ``prefer_onshore``'s answer. The
    two shape files overlap wherever a coastline is drawn differently in each,
    which is often, so this case is ordinary rather than pathological.
    """
    out = pd.Series("unknown", index=index, name="domain", dtype=object)
    first, second = ("offshore", "onshore") if prefer_onshore else ("onshore", "offshore")
    out.iloc[np.flatnonzero(is_offshore if first == "offshore" else is_onshore)] = first
    out.iloc[np.flatnonzero(is_onshore if second == "onshore" else is_offshore)] = second
    return out


def categorize_points_by_region(
    df: pd.DataFrame,
    *,
    onshore_geojson: Path | str,
    offshore_geojson: Path | str,
    lon_col: str = "lon",
    lat_col: str = "lat",
    prefer_onshore: bool = True,
) -> pd.Series:
    """Classify points by testing each one against the prepared geometries.

    The slow route, and the independent one: it shares no code with the spatial
    join beyond loading the shapes, so the two agreeing is evidence about the
    join rather than about a helper they both call.

    Args:
        df: Points, with longitude and latitude columns.
        onshore_geojson: Shapes defining the onshore domain.
        offshore_geojson: Shapes defining the offshore domain.
        lon_col: Longitude column name.
        lat_col: Latitude column name.
        prefer_onshore: Which label a point inside both domains takes.

    Returns:
        A ``domain`` series aligned to ``df``, valued in :data:`DOMAINS`.
    """
    onshore = prep(union_geometries(onshore_geojson))
    offshore = prep(union_geometries(offshore_geojson))
    points = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
    return _resolve(
        np.array([onshore.contains(p) for p in points], dtype=bool),
        np.array([offshore.contains(p) for p in points], dtype=bool),
        df.index,
        prefer_onshore,
    )


def _joined_mask(points: gpd.GeoDataFrame, regions: gpd.GeoDataFrame, n: int) -> np.ndarray:
    """Which points fall strictly inside ``regions``, by spatial join.

    The join is keyed on a position column rather than on the frame's own
    index. A point inside two overlapping polygons comes back as two rows, so
    reading the result through the index would give a mask longer than the
    frame, and a frame whose index has repeated labels would misalign on top of
    that. Positions are unique by construction.
    """
    hit = np.zeros(n, dtype=bool)
    joined = gpd.sjoin(points, regions, predicate="within", how="inner")
    if len(joined):
        hit[joined["_position"].to_numpy(dtype=int)] = True
    return hit


def categorize_points_spatial_join(
    df: pd.DataFrame,
    *,
    onshore_geojson: Path | str,
    offshore_geojson: Path | str,
    lon_col: str = "lon",
    lat_col: str = "lat",
    prefer_onshore: bool = True,
) -> pd.Series:
    """Classify points by spatial join, falling back to the point-in-polygon route.

    Args:
        df: Points, with longitude and latitude columns.
        onshore_geojson: Shapes defining the onshore domain.
        offshore_geojson: Shapes defining the offshore domain.
        lon_col: Longitude column name.
        lat_col: Latitude column name.
        prefer_onshore: Which label a point inside both domains takes.

    Returns:
        A ``domain`` series aligned to ``df``, valued in :data:`DOMAINS`.
    """
    onshore_gdf = load_region_geometry(onshore_geojson)
    offshore_gdf = load_region_geometry(offshore_geojson)
    points = gpd.GeoDataFrame(
        df.assign(_position=np.arange(len(df))),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )
    try:
        is_onshore = _joined_mask(points, onshore_gdf, len(df))
        is_offshore = _joined_mask(points, offshore_gdf, len(df))
    except Exception as error:  # pragma: no cover - needs a broken spatial index
        warnings.warn(
            f"Spatial join failed ({type(error).__name__}: {error}), falling back to "
            "point in polygon. Install rtree for the indexed path.",
            stacklevel=2,
        )
        return categorize_points_by_region(
            df,
            onshore_geojson=onshore_geojson,
            offshore_geojson=offshore_geojson,
            lon_col=lon_col,
            lat_col=lat_col,
            prefer_onshore=prefer_onshore,
        )
    return _resolve(is_onshore, is_offshore, df.index, prefer_onshore)


def add_domain_column(
    df: pd.DataFrame,
    *,
    onshore_geojson: Path | str,
    offshore_geojson: Path | str,
    lon_col: str = "lon",
    lat_col: str = "lat",
    prefer_onshore: bool = True,
    method: Literal["spatial_join", "point_in_polygon"] = "spatial_join",
    overwrite: bool = False,
) -> pd.DataFrame:
    """Add a ``domain`` column labelling each point onshore, offshore or unknown.

    Args:
        df: Points, modified in place and returned.
        onshore_geojson: Shapes defining the onshore domain.
        offshore_geojson: Shapes defining the offshore domain.
        lon_col: Longitude column name.
        lat_col: Latitude column name.
        prefer_onshore: Which label a point inside both domains takes.
        method: ``spatial_join`` or ``point_in_polygon``.
        overwrite: Whether an existing ``domain`` column may be replaced.

    Returns:
        ``df``, with a ``domain`` column.

    Raises:
        ValueError: If ``domain`` exists and ``overwrite`` is false, or if
            ``method`` is not one of the two supported values.
    """
    if "domain" in df.columns and not overwrite:
        raise ValueError("Column 'domain' already exists. Pass overwrite=True to replace it.")
    if method not in ("spatial_join", "point_in_polygon"):
        raise ValueError(f"Unknown method: {method!r}. Use 'spatial_join' or 'point_in_polygon'.")
    classify = (
        categorize_points_spatial_join if method == "spatial_join" else categorize_points_by_region
    )
    df["domain"] = classify(
        df,
        onshore_geojson=onshore_geojson,
        offshore_geojson=offshore_geojson,
        lon_col=lon_col,
        lat_col=lat_col,
        prefer_onshore=prefer_onshore,
    )
    return df


def filter_by_domain(
    df: pd.DataFrame,
    domain: Literal["onshore", "offshore", "known"],
    *,
    domain_col: str = "domain",
) -> pd.DataFrame:
    """Select the points of one domain, or of both known domains.

    Args:
        df: Points carrying a domain column.
        domain: ``onshore``, ``offshore``, or ``known`` for the two together.
        domain_col: The domain column's name.

    Returns:
        A copy holding the selected points.

    Raises:
        ValueError: If the domain column is absent, or ``domain`` is not one of
            the three accepted values.
    """
    if domain_col not in df.columns:
        raise ValueError(f"Column '{domain_col}' not found in the frame.")
    if domain not in ("onshore", "offshore", "known"):
        raise ValueError(f"Unknown domain: {domain!r}. Use 'onshore', 'offshore' or 'known'.")
    if domain == "known":
        return df[df[domain_col].isin(["onshore", "offshore"])].copy()
    return df[df[domain_col] == domain].copy()
