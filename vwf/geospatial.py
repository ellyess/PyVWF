"""
Geospatial utilities for PyVWF.

Functions for categorizing turbines and points by their location
relative to onshore and offshore regions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep


def load_region_geometry(geojson_path: Path | str) -> gpd.GeoDataFrame:
    """Load a GeoJSON file and return a GeoDataFrame in EPSG:4326.

    Args:
        geojson_path: Path to the GeoJSON file containing region geometries.

    Returns:
        GeoDataFrame with geometries in EPSG:4326.

    Raises:
        ValueError: If no geometries are found in the file.
    """
    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        raise ValueError(f"No geometries found in {geojson_path}")
    
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    
    return gdf


def union_geometries(geojson_path: Path | str):
    """Load geometries from GeoJSON and return their union.

    Args:
        geojson_path: Path to the GeoJSON file.

    Returns:
        Union of all geometries in the file.
    """
    gdf = load_region_geometry(geojson_path)
    
    # GeoPandas >= 0.14: unary_union deprecated -> use union_all()
    return gdf.geometry.union_all() if hasattr(gdf.geometry, "union_all") else gdf.geometry.unary_union


def categorize_points_by_region(
    df: pd.DataFrame,
    *,
    onshore_geojson: Path | str,
    offshore_geojson: Path | str,
    lon_col: str = "lon",
    lat_col: str = "lat",
    prefer_onshore: bool = True,
) -> pd.Series:
    """Categorize points as onshore, offshore, or unknown based on GeoJSON regions.

    Args:
        df: DataFrame containing point coordinates.
        onshore_geojson: Path to GeoJSON file defining onshore regions.
        offshore_geojson: Path to GeoJSON file defining offshore regions.
        lon_col: Name of longitude column.
        lat_col: Name of latitude column.
        prefer_onshore: If True and a point is in both regions, classify as onshore.

    Returns:
        Series with values ``"onshore"``, ``"offshore"``, or ``"unknown"``.

    Examples:
        >>> df = pd.DataFrame({'lon': [0.1, 2.5], 'lat': [51.5, 52.0]})
        >>> categories = categorize_points_by_region(
        ...     df,
        ...     onshore_geojson='regions/onshore.geojson',
        ...     offshore_geojson='regions/offshore.geojson'
        ... )
    """
    # Load and prepare geometries
    onshore_geom = union_geometries(onshore_geojson)
    offshore_geom = union_geometries(offshore_geojson)
    
    onshore_prep = prep(onshore_geom)
    offshore_prep = prep(offshore_geom)
    
    # Create point geometries
    points = [Point(lon, lat) for lon, lat in zip(df[lon_col], df[lat_col])]
    
    # Classify each point
    categories = []
    for point in points:
        in_onshore = onshore_prep.contains(point)
        in_offshore = offshore_prep.contains(point)
        
        if in_onshore and in_offshore:
            # Both regions overlap at this point
            cat = "onshore" if prefer_onshore else "offshore"
        elif in_onshore:
            cat = "onshore"
        elif in_offshore:
            cat = "offshore"
        else:
            cat = "unknown"
        
        categories.append(cat)
    
    return pd.Series(categories, index=df.index, name="domain")


def categorize_points_spatial_join(
    df: pd.DataFrame,
    *,
    onshore_geojson: Path | str,
    offshore_geojson: Path | str,
    lon_col: str = "lon",
    lat_col: str = "lat",
    prefer_onshore: bool = True,
) -> pd.Series:
    """Categorize points using spatial join (faster for large datasets).

    Requires rtree or pygeos for optimal performance.

    Args:
        df: DataFrame containing point coordinates.
        onshore_geojson: Path to GeoJSON file defining onshore regions.
        offshore_geojson: Path to GeoJSON file defining offshore regions.
        lon_col: Name of longitude column.
        lat_col: Name of latitude column.
        prefer_onshore: If True and a point is in both regions, classify as onshore.

    Returns:
        Series with values ``"onshore"``, ``"offshore"``, or ``"unknown"``.
    """
    # Load regions
    onshore_gdf = load_region_geometry(onshore_geojson)
    offshore_gdf = load_region_geometry(offshore_geojson)
    
    # Create points GeoDataFrame
    points_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326"
    )
    
    # Spatial joins
    try:
        in_onshore = gpd.sjoin(points_gdf, onshore_gdf, predicate="within", how="left")
        in_offshore = gpd.sjoin(points_gdf, offshore_gdf, predicate="within", how="left")
    except Exception as e:
        # Fall back to slower method if spatial index fails
        import warnings
        warnings.warn(
            f"Spatial join failed ({type(e).__name__}), falling back to point-in-polygon. "
            "Install rtree or pygeos for better performance."
        )
        return categorize_points_by_region(
            df, 
            onshore_geojson=onshore_geojson,
            offshore_geojson=offshore_geojson,
            lon_col=lon_col,
            lat_col=lat_col,
            prefer_onshore=prefer_onshore
        )
    
    # Determine categories
    is_onshore = in_onshore["index_right"].notna()
    is_offshore = in_offshore["index_right"].notna()
    
    categories = pd.Series("unknown", index=df.index, name="domain")
    
    if prefer_onshore:
        categories.loc[is_offshore] = "offshore"
        categories.loc[is_onshore] = "onshore"  # overwrites if both
    else:
        categories.loc[is_onshore] = "onshore"
        categories.loc[is_offshore] = "offshore"  # overwrites if both
    
    return categories


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
    """Add a ``"domain"`` column that labels points as onshore or offshore.

    Args:
        df: DataFrame to modify (modified in place and returned).
        onshore_geojson: Path to GeoJSON file defining onshore regions.
        offshore_geojson: Path to GeoJSON file defining offshore regions.
        lon_col: Name of longitude column.
        lat_col: Name of latitude column.
        prefer_onshore: If True and a point is in both regions, classify as onshore.
        method: Classification method. ``"spatial_join"`` is faster for large datasets.
        overwrite: If True, overwrite existing ``"domain"`` column.

    Returns:
        DataFrame with the ``"domain"`` column added.

    Raises:
        ValueError: If ``"domain"`` exists and ``overwrite`` is False.
        ValueError: If ``method`` is not a supported option.

    Examples:
        >>> df = pd.DataFrame({
        ...     'turbine_id': [1, 2, 3],
        ...     'lon': [0.1, 2.5, -1.0],
        ...     'lat': [51.5, 52.0, 53.0]
        ... })
        >>> df = add_domain_column(
        ...     df,
        ...     onshore_geojson='input/regions/country_shapes.geojson',
        ...     offshore_geojson='input/regions/north_sea_shape.geojson'
        ... )
        >>> print(df['domain'])
        0    onshore
        1    onshore
        2    offshore
        Name: domain, dtype: object
    """
    if "domain" in df.columns and not overwrite:
        raise ValueError("Column 'domain' already exists. Set overwrite=True to replace it.")
    
    if method == "spatial_join":
        categories = categorize_points_spatial_join(
            df,
            onshore_geojson=onshore_geojson,
            offshore_geojson=offshore_geojson,
            lon_col=lon_col,
            lat_col=lat_col,
            prefer_onshore=prefer_onshore,
        )
    elif method == "point_in_polygon":
        categories = categorize_points_by_region(
            df,
            onshore_geojson=onshore_geojson,
            offshore_geojson=offshore_geojson,
            lon_col=lon_col,
            lat_col=lat_col,
            prefer_onshore=prefer_onshore,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spatial_join' or 'point_in_polygon'.")
    
    df["domain"] = categories
    return df


def filter_by_domain(
    df: pd.DataFrame,
    domain: Literal["onshore", "offshore", "known"],
    *,
    domain_col: str = "domain",
) -> pd.DataFrame:
    """Filter a DataFrame by domain type.

    Args:
        df: DataFrame with a domain column.
        domain: Domain to filter for. ``"known"`` returns onshore and offshore only.
        domain_col: Name of the domain column.

    Returns:
        Filtered DataFrame.

    Raises:
        ValueError: If ``domain_col`` is not present in the DataFrame.

    Examples:
        >>> df_onshore = filter_by_domain(df, "onshore")
        >>> df_offshore = filter_by_domain(df, "offshore")
        >>> df_all_known = filter_by_domain(df, "known")
    """
    if domain_col not in df.columns:
        raise ValueError(f"Column '{domain_col}' not found in DataFrame.")
    
    if domain == "known":
        return df[df[domain_col].isin(["onshore", "offshore"])].copy()
    
    return df[df[domain_col] == domain].copy()
