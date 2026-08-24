"""Clustering utilities for turbine metadata.

This module provides spatial clustering of turbine coordinates for training and
evaluation workflows.
"""
# Lazy (string) annotations so optional geometry types (e.g. Polygon) in
# function signatures don't require shapely to be importable.
from __future__ import annotations

from pathlib import Path

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import warnings

from vwf.config import PyVWFPaths

# shapely powers the optional Voronoi/geometry helpers below; the core
# KMeans-based cluster_turbines() does not need it, so guard the import to keep
# the package importable in minimal environments.
try:
    from shapely.geometry import Point, Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    Point = Polygon = None
    warnings.warn("shapely not installed. Geometry-based features will be limited.")

try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    warnings.warn("geopandas not installed. Geometry-based features will be limited.")


# Cached region shapes
_COUNTRY_SHAPES = None
_OFFSHORE_SHAPES = None


def load_region_shapes():
    """Load country and offshore region shapes from GeoJSON files.

    Returns:
        Tuple of (country_gdf, offshore_gdf) GeoDataFrames.
    """
    global _COUNTRY_SHAPES, _OFFSHORE_SHAPES

    # Resolved through PyVWFPaths rather than a literal relative path, so an
    # installed PyVWF finds them via PYVWF_INPUT instead of only when the
    # working directory happens to be a repository checkout. The shapes are too
    # large to bundle, so a missing file stays a warning, not an error.
    if _COUNTRY_SHAPES is None:
        country_path = PyVWFPaths.COUNTRY_SHAPES
        if country_path.exists():
            _COUNTRY_SHAPES = gpd.read_file(country_path)
        else:
            warnings.warn(f"Country shapes not found at {country_path}")
            _COUNTRY_SHAPES = gpd.GeoDataFrame()

    if _OFFSHORE_SHAPES is None:
        offshore_path = PyVWFPaths.OFFSHORE_SHAPES
        if offshore_path.exists():
            _OFFSHORE_SHAPES = gpd.read_file(offshore_path)
        else:
            warnings.warn(f"Offshore shapes not found at {offshore_path}")
            _OFFSHORE_SHAPES = gpd.GeoDataFrame()

    return _COUNTRY_SHAPES, _OFFSHORE_SHAPES


# Projected CRS for the area/distance comparisons in repair_region_shape.
# Equal-area over Europe; never used for returned geometry, which stays EPSG:4326.
_METRIC_CRS = "EPSG:3035"

# Repaired shapes are cached per (code, mode): polygonising the coastline is far
# too slow to repeat on every call.
_REPAIRED_SHAPES: dict = {}


def repair_region_shape(geom, region_code: str, fleet_xy=None, coastline_path=None):
    """Add landmasses that ``country_shapes.geojson`` omits for a region.

    The bundled country shapes drop smaller islands. Denmark is the worst case
    in this dataset: its polygon holds only Jutland, Zealand and Funen
    (39,208 km² against a published land area of 42,943), so Lolland, Falster,
    Bornholm, Als, Langeland, Møn, Mors, Samsø, Læsø and Anholt are absent and
    12.9% of the DK onshore fleet falls outside its own country polygon, a
    median 15 km from the nearest included land. Greece (-10.1%) and Croatia
    (-9.0%) are similarly truncated.

    Closed rings in the coastline linework are polygonised and each resulting
    landmass is claimed for a region when either:

    1. it carries a point of ``fleet_xy``, when one is supplied, or
    2. the nearest *offshore* (EEZ) zone belongs to this region.

    Territorial waters are used rather than nearest land because land distance
    gets sovereignty wrong: Bornholm is 34 km from Sweden and 143 km from the
    Danish mainland, so nearest-land hands it to Sweden. It sits inside Denmark's
    EEZ (distance 0.0 km, against 14.7 km to Sweden's), so the EEZ test is
    correct without needing a fleet to disambiguate.

    Args:
        geom: The region's shapely geometry, as returned by
            :func:`get_country_shape`.
        region_code: Two-letter code used to match the offshore zones
            (``'GB'``/``'UK'`` are treated as the same region).
        fleet_xy: Optional (n, 2) array of ``(lon, lat)`` positions. Any
            landmass carrying one is claimed regardless of the EEZ test, which
            keeps a fleet's own sites from being dropped.
        coastline_path: Coastline linework to polygonise. Defaults to
            ``PyVWFPaths.COASTLINES``.

    Returns:
        The merged shapely geometry, or ``geom`` unchanged when the coastline
        file is missing or nothing new is claimed.
    """
    if not HAS_GEOPANDAS or geom is None or geom.is_empty:
        return geom

    coast_path = Path(coastline_path) if coastline_path else PyVWFPaths.COASTLINES
    if not coast_path.exists():
        warnings.warn(
            f"Coastline file not found at {coast_path}; returning the "
            f"unrepaired {region_code} shape, which may omit islands"
        )
        return geom

    from shapely.geometry import MultiPoint, box as shapely_box
    from shapely.ops import polygonize, unary_union

    _, offshore_shapes = load_region_shapes()
    code_map = {'UK': 'GB', 'GB': 'UK'}
    codes = {region_code, code_map.get(region_code, region_code)}

    minx, miny, maxx, maxy = geom.bounds
    window = shapely_box(minx - 3.0, miny - 2.0, maxx + 3.0, maxy + 2.0)
    coast = gpd.read_file(coast_path)
    near = coast[coast.intersects(window)]
    if near.empty:
        return geom
    lines = unary_union([g.intersection(window) for g in near.geometry])

    offshore_metric = (
        offshore_shapes.to_crs(_METRIC_CRS)
        if offshore_shapes is not None and not offshore_shapes.empty
        else None
    )
    if offshore_metric is None:
        warnings.warn(
            "Offshore shapes unavailable; island attribution falls back to "
            "nearest land, which misassigns outlying islands such as Bornholm"
        )
        country_shapes, _ = load_region_shapes()
        if country_shapes is None or country_shapes.empty:
            # Nothing to attribute against; claiming every island would be worse
            # than leaving the shape as it came.
            return geom
        offshore_metric = country_shapes.to_crs(_METRIC_CRS)

    fleet = None
    if fleet_xy is not None and len(fleet_xy):
        fleet = MultiPoint(np.asarray(fleet_xy, dtype=float))

    extra = []
    for piece in polygonize(lines):
        if piece.is_empty or piece.within(geom.buffer(1e-9)):
            continue
        if fleet is not None and piece.intersects(fleet):
            extra.append(piece)
            continue
        pm = gpd.GeoSeries([piece], crs="EPSG:4326").to_crs(_METRIC_CRS).iloc[0]
        nearest = offshore_metric.iloc[offshore_metric.geometry.distance(pm).idxmin()]
        if str(nearest["name"]) in codes:
            extra.append(piece)

    if not extra:
        return geom
    return unary_union([geom] + extra)


def get_country_shape(country_code: str, cluster_mode: str = "onshore",
                      repair: bool = False):
    """Get country or offshore shape by country code.

    Args:
        country_code: Two-letter country code (e.g., 'NL', 'UK', 'DE').
        cluster_mode: Either 'onshore' or 'offshore'.
        repair: If True, merge in islands the bundled country shapes omit
            (see :func:`repair_region_shape`). Off by default so existing
            callers keep the exact geometry they have always received; turn it
            on wherever a shape is used to clip or draw a whole region.
            Ignored for ``cluster_mode='offshore'``, whose zones are not
            island-truncated.

    Returns:
        Shapely geometry for the country/offshore region, or None if not found.
    """
    if not HAS_GEOPANDAS:
        return None

    country_shapes, offshore_shapes = load_region_shapes()

    # Map UK <-> GB
    code_map = {'UK': 'GB', 'GB': 'UK'}
    codes_to_try = [country_code, code_map.get(country_code, country_code)]

    if cluster_mode == "offshore":
        shapes_df = offshore_shapes
    else:
        shapes_df = country_shapes

    if shapes_df.empty:
        return None

    # Try to find the country
    for code in codes_to_try:
        matches = shapes_df[shapes_df['name'] == code]
        if not matches.empty:
            # Return the union of all geometries for this country (in case of multi-part)
            geom = matches.geometry.unary_union
            if repair and cluster_mode != "offshore":
                key = (code, cluster_mode)
                if key not in _REPAIRED_SHAPES:
                    _REPAIRED_SHAPES[key] = repair_region_shape(geom, code)
                return _REPAIRED_SHAPES[key]
            return geom

    warnings.warn(f"Country/offshore shape not found for {country_code} (mode={cluster_mode})")
    return None

def _cluster_coords(frame, geographic: bool):
    """Coordinates to cluster on: raw degrees, or unit-sphere Cartesian.

    Clustering raw ``(lat, lon)`` degrees treats one degree of longitude as one
    degree of latitude, which they are not: a degree of longitude is 111 km at
    the equator but 71 km at 50N, so clusters come out stretched east-west by
    a factor that varies across the domain (CONUS spans cos(lat) 0.91 to 0.64;
    Brazil crosses the equator).

    Projecting onto the unit sphere makes Euclidean distance the CHORD length,
    which is a monotone function of great-circle distance, so nearest-centroid
    assignment becomes geographically correct. It needs no reference latitude,
    so it stays valid over wide domains where a single cos(lat0) scaling would
    not.
    """
    lat = np.radians(pd.to_numeric(frame["lat"], errors="coerce").to_numpy(float))
    lon = np.radians(pd.to_numeric(frame["lon"], errors="coerce").to_numpy(float))
    if not geographic:
        return frame[["lat", "lon"]]
    return np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def _merge_undersized(labels, centres, min_size):
    """Map undersized cluster labels onto their nearest surviving centroid.

    A cluster with one or two members is fitted to those members' idiosyncrasy,
    which is how a correction ends up with a wind scalar of 80 (see
    docs/findings/method-hourly-resolution.md: every degenerate Chilean cluster was
    both tiny and in the Atacama, where ERA5 badly under-resolves the wind).
    Merging the smallest cluster into its nearest neighbour and repeating is a
    cheap structural guard against fitting one farm.

    Args:
        labels: Cluster label per training row.
        centres: KMeans cluster centres, indexed by label.
        min_size: Minimum members a surviving cluster must have.

    Returns:
        dict mapping every original label to its surviving label.
    """
    labels = np.asarray(labels)
    remap = {int(c): int(c) for c in range(len(centres))}
    alive = {int(c) for c in np.unique(labels)}

    while len(alive) > 1:
        sizes = {c: int((labels == c).sum()) for c in alive}
        small = [c for c, n in sizes.items() if n < min_size]
        if not small:
            break
        victim = min(small, key=lambda c: (sizes[c], c))
        others = [c for c in alive if c != victim]
        # Nearest surviving centroid in the same space the fit used.
        d = np.linalg.norm(centres[others] - centres[victim], axis=1)
        target = int(others[int(np.argmin(d))])
        labels = np.where(labels == victim, target, labels)
        alive.discard(victim)
        for src, dst in list(remap.items()):
            if dst == victim:
                remap[src] = target
    return remap


def cluster_turbines(
    num_clu,
    turb_info_train,
    train=False,
    *args,
    random_state=42,
    weight_col=None,
    geographic=False,
    min_cluster_size=1,
):
    """Cluster turbines by spatial coordinates.

    Args:
        num_clu: Number of clusters.
        turb_info_train: Training turbine metadata with ``lat`` and ``lon``.
        train: If True, assign clusters to ``turb_info_train`` and return it.
        *args: Optional turbine metadata to cluster using the fitted model.
        random_state: KMeans seed. Exposed so the partition's seed-stability is
            testable; the result must not depend on it (see
            ``tests/test_clustering.py``).
        weight_col: Column to weight the fit by, typically ``"capacity"``, or
            None for an unweighted fit. Unweighted, a 5 MW site pulls a centroid
            as hard as a 500 MW one even though skill is scored capacity-
            weighted. Weighting moves cluster boundaries toward where the
            capacity is. OFF by default: it is a modelling choice about whether
            clusters represent METEOROLOGY (unweighted: bias is a property of
            location) or where the ENERGY is, and the two are not the same.
            Falls back to unweighted if the column is missing or carries no
            usable positive weights.
        geographic: If True, cluster on unit-sphere Cartesian coordinates so
            distance is geographic rather than degree-space (see
            :func:`_cluster_coords`). Default False preserves the legacy
            degree-space behaviour.
        min_cluster_size: Minimum TRAINING sites a cluster must retain. Clusters
            below this are merged into their nearest surviving centroid before
            any correction is fitted, because a one-site cluster fits that
            site's idiosyncrasy rather than a regional bias. Default 1 disables
            merging and reproduces the legacy partition exactly.

    Returns:
        Turbine metadata with an added ``cluster`` column.
    """
    # Fit clusters to the training data.
    #
    # init="k-means++", not "random". Random initialisation settles into a
    # different local optimum per seed, and n_init=10 picks among those by
    # INERTIA, which does not track held-out skill. On the real DK fleet at a
    # fixed k=500, varying only the seed moved held-out MAE across
    # 0.0514-0.0842, a wider range than the whole k=10..1000 curve, so the
    # k-sweep was measuring initialisation luck. k-means++ collapses that
    # spread to 0.0507-0.0513 and beats the luckiest random draw.
    kmeans = KMeans(
            init="k-means++",
            n_clusters = num_clu,
            n_init = 10,
            max_iter = 300,
            random_state = random_state
        )
    sample_weight = None
    if weight_col is not None and weight_col in turb_info_train.columns:
        w = pd.to_numeric(turb_info_train[weight_col], errors="coerce").to_numpy(float)
        # A partial or non-positive weight vector would silently distort the
        # partition, so only use weights when every row has a usable one.
        if np.all(np.isfinite(w)) and np.all(w > 0):
            sample_weight = w
        else:
            warnings.warn(
                f"cluster_turbines: {weight_col!r} has missing or non-positive "
                "values; falling back to an unweighted fit"
            )

    train_coords = _cluster_coords(turb_info_train, geographic)
    kmeans.fit(train_coords, sample_weight=sample_weight)

    # The merge is derived from the TRAINING partition, so train and apply
    # resolve identical labels given the same training fleet and seed.
    remap = None
    if min_cluster_size and min_cluster_size > 1:
        remap = _merge_undersized(
            kmeans.predict(train_coords), kmeans.cluster_centers_, int(min_cluster_size)
        )
        merged = sum(1 for src, dst in remap.items() if src != dst)
        if merged:
            warnings.warn(
                f"cluster_turbines: merged {merged} cluster(s) with fewer than "
                f"{int(min_cluster_size)} training sites into their nearest "
                f"neighbour; {len(set(remap.values()))} of {num_clu} clusters remain"
            )

    def _labels(frame):
        out = kmeans.predict(_cluster_coords(frame, geographic))
        return np.array([remap[int(c)] for c in out]) if remap else out

    if train:
        turb_info_train['cluster'] = _labels(turb_info_train)
        return turb_info_train
    else:
        turb_info = args[0]
        turb_info['cluster'] = _labels(turb_info)
        return turb_info


def add_turbine_metadata(
    sampling_points: pd.DataFrame,
    default_height: float = 100.0,
    default_model: str = "2019COE_Market_Average_2.6MW_121",
    default_capacity: float = 3.0,
    default_type: str = "onshore",
) -> pd.DataFrame:
    """Add turbine metadata to sampling points for simulation.

    For grid-based or random sampling points that don't have turbine metadata,
    this function adds representative values needed by interpolate_wind() and
    simulate_wind() functions.

    Args:
        sampling_points: DataFrame with lat, lon columns.
        default_height: Hub height in meters (default: 100m for modern turbines).
        default_model: Turbine model name for power curve selection.
            The default is the bundled open library's market-average onshore
            composite. Must match a column in your power curve file.
        default_capacity: Capacity in MW (default: 3.0 MW).
        default_type: Turbine type, either "onshore" or "offshore" (default: "onshore").

    Returns:
        DataFrame with added columns: ID, height, model, capacity, type.
        Preserves any existing columns (lat, lon, weight, cluster, etc.).

    Example:
        >>> grid_points = create_sampling_points(nl_bounds, method="grid")
        >>> grid_points = add_turbine_metadata(
        ...     grid_points,
        ...     default_height=100.0,
        ...     default_model="2019COE_Market_Average_2.6MW_121",
        ...     default_capacity=3.0
        ... )
        >>> # Now ready for simulate_wind()
    """
    points = sampling_points.copy()

    # Add ID if not present
    if 'ID' not in points.columns:
        points['ID'] = [f'grid_{i:04d}' for i in range(len(points))]

    # Add turbine metadata
    if 'height' not in points.columns:
        points['height'] = default_height

    if 'model' not in points.columns:
        points['model'] = default_model

    if 'capacity' not in points.columns:
        points['capacity'] = default_capacity

    if 'type' not in points.columns:
        points['type'] = default_type

    return points


def create_sampling_points(
    country_bounds: Polygon | None = None,
    method: str = "grid",
    resolution: float = 0.25,
    turbine_locs: pd.DataFrame | None = None,
    n_random: int | None = None,
    add_metadata: bool = False,
    default_height: float = 100.0,
    default_model: str = "2019COE_Market_Average_2.6MW_121",
    default_capacity: float = 3.0,
) -> pd.DataFrame:
    """Create spatial sampling points for simulation.

    This function creates sampling points for simulating wind capacity factors.
    It supports three methods:
    - "grid": Regular grid within country boundaries (for country-level data)
    - "turbines": Use actual turbine locations (for turbine-level data)
    - "random": Random points within boundaries (for sensitivity analysis)

    Args:
        country_bounds: Polygon defining country/zone boundaries.
            Required for "grid" and "random" methods.
        method: Sampling method - "grid", "turbines", or "random".
        resolution: Grid resolution in degrees (for method="grid").
            Default 0.25° ≈ 25km.
        turbine_locs: Turbine locations DataFrame with lat, lon, capacity
            (required for method="turbines").
        n_random: Number of random points (for method="random").
            If None, calculated from resolution.
        add_metadata: If True, add turbine metadata (ID, height, model, capacity)
            needed for simulation. Useful for grid/random methods.
        default_height: Hub height in meters if add_metadata=True (default: 100m).
        default_model: Turbine model name if add_metadata=True (default: the
            bundled open library's market-average onshore composite).
        default_capacity: Capacity in MW if add_metadata=True (default: 3.0).

    Returns:
        DataFrame with columns:

        - lat: Latitude
        - lon: Longitude
        - weight: Capacity for turbines, 1.0 for grid/random points

        If add_metadata=True, also includes:

        - ID: Point identifier
        - height: Hub height (meters)
        - model: Turbine model name
        - capacity: Capacity (MW)

    Raises:
        ValueError: If required parameters are missing.

    Example:
        >>> # Grid-based for country-level data WITHOUT metadata
        >>> from shapely.geometry import box
        >>> nl_bounds = box(3.3, 50.7, 7.2, 53.6)
        >>> grid_points = create_sampling_points(
        ...     country_bounds=nl_bounds,
        ...     method="grid",
        ...     resolution=0.25
        ... )
        >>> print(len(grid_points))  # Has: lat, lon, weight
        150

        >>> # Grid-based WITH turbine metadata for simulation
        >>> grid_points = create_sampling_points(
        ...     country_bounds=nl_bounds,
        ...     method="grid",
        ...     resolution=0.25,
        ...     add_metadata=True,
        ...     default_height=100.0,
        ...     default_model="2019COE_Market_Average_2.6MW_121",
        ...     default_capacity=3.0
        ... )
        >>> print(grid_points.columns)
        >>> # Has: lat, lon, weight, ID, height, model, capacity

        >>> # Turbine-based for turbine-level data (UK, DE)
        >>> turbines = pd.DataFrame({
        ...     'lat': [51.5, 52.1, 51.8],
        ...     'lon': [1.2, 1.5, 1.3],
        ...     'capacity': [3.0, 2.5, 3.5]
        ... })
        >>> turb_points = create_sampling_points(
        ...     method="turbines",
        ...     turbine_locs=turbines
        ... )
    """
    if method == "turbines":
        if turbine_locs is None:
            raise ValueError("turbine_locs required for method='turbines'")

        if not all(col in turbine_locs.columns for col in ['lat', 'lon']):
            raise ValueError("turbine_locs must have 'lat' and 'lon' columns")

        # Use actual turbine locations
        points = turbine_locs[['lat', 'lon']].copy()

        # Add weight (capacity if available, otherwise 1.0)
        if 'capacity' in turbine_locs.columns:
            points['weight'] = turbine_locs['capacity']
        else:
            points['weight'] = 1.0

        # Optionally add metadata
        if add_metadata:
            points = add_turbine_metadata(
                points,
                default_height=default_height,
                default_model=default_model,
                default_capacity=default_capacity
            )

        return points.reset_index(drop=True)

    elif method == "grid":
        if country_bounds is None:
            raise ValueError("country_bounds required for method='grid'")

        # Create regular grid within bounds
        minx, miny, maxx, maxy = country_bounds.bounds

        lats = np.arange(miny, maxy, resolution)
        lons = np.arange(minx, maxx, resolution)

        grid_points = []
        for lat in lats:
            for lon in lons:
                point = Point(lon, lat)
                if country_bounds.contains(point):
                    grid_points.append({
                        'lat': lat,
                        'lon': lon,
                        'weight': 1.0  # Equal weighting for grid points
                    })

        if len(grid_points) == 0:
            raise ValueError("No grid points within country_bounds. Check bounds and resolution.")

        points = pd.DataFrame(grid_points)

        # Optionally add metadata for simulation
        if add_metadata:
            points = add_turbine_metadata(
                points,
                default_height=default_height,
                default_model=default_model,
                default_capacity=default_capacity
            )

        return points

    elif method == "random":
        if country_bounds is None:
            raise ValueError("country_bounds required for method='random'")

        # Random sampling within bounds
        minx, miny, maxx, maxy = country_bounds.bounds

        if n_random is None:
            # Calculate from resolution
            n_random = int((maxx - minx) * (maxy - miny) / (resolution ** 2))

        # Distinct name from the DataFrame `points` bound in the other branches:
        # this one accumulates dicts before being framed up below.
        sampled: list[dict[str, float]] = []
        attempts = 0
        max_attempts = n_random * 100  # Prevent infinite loop

        while len(sampled) < n_random and attempts < max_attempts:
            lon = np.random.uniform(minx, maxx)
            lat = np.random.uniform(miny, maxy)
            point = Point(lon, lat)

            if country_bounds.contains(point):
                sampled.append({
                    'lat': lat,
                    'lon': lon,
                    'weight': 1.0
                })
            attempts += 1

        if len(sampled) < n_random:
            warnings.warn(
                f"Only found {len(sampled)} points after {max_attempts} attempts. "
                f"Requested {n_random}. Try increasing resolution or checking bounds."
            )

        points_df = pd.DataFrame(sampled)

        # Optionally add metadata for simulation
        if add_metadata:
            points_df = add_turbine_metadata(
                points_df,
                default_height=default_height,
                default_model=default_model,
                default_capacity=default_capacity
            )

        return points_df

    else:
        raise ValueError(f"Unknown method: {method}. Use 'grid', 'turbines', or 'random'.")


def cluster_with_geometries(
    sampling_points: pd.DataFrame,
    num_clusters: int,
    method: str = "kmeans",
    country_bounds: Polygon | None = None,
    country_code: str | None = None,
    cluster_mode: str = "onshore",
    geometry_type: str = "convex_hull",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Cluster sampling points and return cluster geometries.

    This extends the standard clustering to return spatial geometries (polygons)
    for each cluster. This enables:
    - Applying corrections based on lat/lon rather than turbine ID
    - Visualizing correction factors as maps
    - Supporting both turbine-level and grid-level data

    Args:
        sampling_points: DataFrame with lat, lon, weight columns.
        num_clusters: Number of clusters to create.
        method: Clustering method - "kmeans" or "predefined".
        country_bounds: Country boundary for clipping (polygon or box).
                       If None and country_code is provided, will load from GeoJSON files.
        country_code: Two-letter country code (e.g., 'NL', 'UK', 'DE').
                     If provided, will load actual country shapes for clipping.
        cluster_mode: Either 'onshore' or 'offshore' (default: 'onshore').
                     Used to select appropriate region shape when country_code is provided.
        geometry_type: Type of geometry to create - "convex_hull" or "voronoi".
                      - convex_hull: Simple convex hull of cluster points (may have gaps)
                      - voronoi: Voronoi tessellation for complete spatial coverage (no gaps)

    Returns:
        Tuple of:
        - sampling_points with added 'cluster' column
        - GeoDataFrame with cluster geometries (or None if geopandas not available)

    Example:
        >>> # Use Voronoi tessellation for complete coverage
        >>> points_clustered, cluster_geoms = cluster_with_geometries(
        ...     sampling_points=points,
        ...     num_clusters=5,
        ...     method="kmeans",
        ...     country_code="NL",
        ...     cluster_mode="onshore",
        ...     geometry_type="voronoi"
        ... )
        >>> # Save cluster geometries
        >>> cluster_geoms.to_file("output/nl_correction_regions.geojson")
    """
    # Get clipping boundary
    clip_boundary = country_bounds
    if country_code is not None and clip_boundary is None:
        # Try to load actual country shape
        clip_boundary = get_country_shape(country_code, cluster_mode)
        if clip_boundary is not None:
            print(f"    Using actual {cluster_mode} shape for {country_code}")

    if clip_boundary is None and country_bounds is None:
        print("    Warning: No clipping boundary provided - using raw convex hulls")
    if method == "kmeans":
        # Standard KMeans clustering
        kmeans = KMeans(
            n_clusters=num_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42
        )

        sampling_points['cluster'] = kmeans.fit_predict(
            sampling_points[['lat', 'lon']]
        )

        # Create Voronoi diagram from cluster centers for geometries
        if not HAS_GEOPANDAS:
            warnings.warn(
                "geopandas not installed. Returning None for cluster geometries. "
                "Install with: pip install geopandas"
            )
            return sampling_points, None

        centers = kmeans.cluster_centers_  # [lat, lon]

        # Import geometry types needed for both Voronoi and convex hull
        from shapely.geometry import MultiPoint

        # Try Voronoi tessellation if requested and feasible
        use_voronoi = geometry_type == "voronoi" and num_clusters >= 3
        voronoi_success = False

        if use_voronoi:
            # Create Voronoi tessellation for complete spatial coverage (no gaps)
            # Requires at least 3 points for Voronoi diagram
            from scipy.spatial import qhull

            try:
                # Convert centers to (lon, lat) for Voronoi
                centers_lonlat = centers[:, [1, 0]]  # [lon, lat]

                # Use shapely's voronoi_diagram for bounded Voronoi
                # This automatically handles infinite regions properly
                from shapely.ops import voronoi_diagram as shapely_voronoi

                # Create MultiPoint from cluster centers
                points = MultiPoint([Point(lon, lat) for lon, lat in centers_lonlat])

                # Create bounding envelope - extend beyond points for proper edge cells
                min_lon, min_lat = centers_lonlat.min(axis=0)
                max_lon, max_lat = centers_lonlat.max(axis=0)
                lon_range = max(max_lon - min_lon, 5.0)
                lat_range = max(max_lat - min_lat, 5.0)
                padding = max(lon_range, lat_range) * 0.3  # 30% padding

                from shapely.geometry import box as create_box
                envelope = create_box(
                    min_lon - padding, min_lat - padding,
                    max_lon + padding, max_lat + padding
                )

                # Generate bounded Voronoi diagram
                voronoi_polys = shapely_voronoi(points, envelope=envelope)

                # voronoi_polys is a GeometryCollection - extract individual polygons
                # and assign them to the nearest cluster center
                cluster_geoms = []

                for cluster_id in range(num_clusters):
                    center_lon, center_lat = centers_lonlat[cluster_id]
                    center_point = Point(center_lon, center_lat)

                    # Find the Voronoi polygon that contains this center point
                    polygon = None
                    for geom in voronoi_polys.geoms:
                        if geom.contains(center_point) or geom.touches(center_point):
                            polygon = geom
                            break

                    if polygon is None:
                        # Fallback: find closest polygon
                        min_dist = float('inf')
                        for geom in voronoi_polys.geoms:
                            dist = center_point.distance(geom)
                            if dist < min_dist:
                                min_dist = dist
                                polygon = geom

                    # Clip to country boundary if provided
                    if clip_boundary is not None and polygon is not None:
                        polygon = polygon.intersection(clip_boundary)

                    # Count points in this cluster
                    n_points = (sampling_points['cluster'] == cluster_id).sum()

                    cluster_geoms.append({
                        'cluster': cluster_id,
                        'geometry': polygon,
                        'n_points': n_points
                    })

                cluster_gdf = gpd.GeoDataFrame(cluster_geoms, crs="EPSG:4326")
                voronoi_success = True

            except (qhull.QhullError, Exception) as e:
                # Voronoi failed due to degenerate geometry (e.g., collinear points)
                # Fall back to convex hull
                warnings.warn(
                    f"Voronoi tessellation failed (degenerate geometry): {str(e)[:100]}. "
                    f"Falling back to convex hull."
                )
                voronoi_success = False

        # Use convex hull if Voronoi wasn't requested, failed, or too few clusters
        if not voronoi_success:
            if geometry_type == "voronoi" and num_clusters < 3:
                warnings.warn(
                    f"Voronoi tessellation requires at least 3 clusters, "
                    f"but only {num_clusters} requested. Using convex hull instead."
                )

            cluster_geoms = []
            for cluster_id in range(num_clusters):
                cluster_points = sampling_points[
                    sampling_points['cluster'] == cluster_id
                ][['lon', 'lat']].values

                if len(cluster_points) >= 3:
                    # Create convex hull
                    multipoint = MultiPoint([(lon, lat) for lon, lat in cluster_points])
                    polygon = multipoint.convex_hull

                    # Clip to country bounds if provided
                    if clip_boundary is not None:
                        polygon = polygon.intersection(clip_boundary)

                    cluster_geoms.append({
                        'cluster': cluster_id,
                        'geometry': polygon,
                        'n_points': len(cluster_points)
                    })
                elif len(cluster_points) > 0:
                    # Single or two points - create small buffer
                    center_lon, center_lat = cluster_points.mean(axis=0)
                    point = Point(center_lon, center_lat)
                    polygon = point.buffer(0.1)  # 0.1 degree buffer (~10km)

                    if clip_boundary is not None:
                        polygon = polygon.intersection(clip_boundary)

                    cluster_geoms.append({
                        'cluster': cluster_id,
                        'geometry': polygon,
                        'n_points': len(cluster_points)
                    })

            cluster_gdf = gpd.GeoDataFrame(cluster_geoms, crs="EPSG:4326")

        return sampling_points, cluster_gdf

    elif method == "predefined":
        # Use predefined cluster assignments (e.g., for bidding zones)
        # Assumes sampling_points already has 'cluster' column

        if 'cluster' not in sampling_points.columns:
            raise ValueError("For method='predefined', sampling_points must have 'cluster' column")

        if not HAS_GEOPANDAS:
            return sampling_points, None

        # Create geometries from point clusters
        cluster_geoms = []
        for cluster_id in sampling_points['cluster'].unique():
            cluster_points = sampling_points[
                sampling_points['cluster'] == cluster_id
            ][['lon', 'lat']].values

            if len(cluster_points) >= 3:
                multipoint = MultiPoint([(lon, lat) for lon, lat in cluster_points])
                polygon = multipoint.convex_hull

                cluster_geoms.append({
                    'cluster': cluster_id,
                    'geometry': polygon,
                    'n_points': len(cluster_points)
                })

        cluster_gdf = gpd.GeoDataFrame(cluster_geoms, crs="EPSG:4326")

        return sampling_points, cluster_gdf

    else:
        raise ValueError(f"Unknown method: {method}. Use 'kmeans' or 'predefined'.")
