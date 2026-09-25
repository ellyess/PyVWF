"""Clustering utilities for turbine metadata.

This module provides spatial clustering of turbine coordinates for training and
evaluation workflows, and the country shapes both it and ``pyvwf.sampling`` use.
The country-level sampling grids and their Voronoi geometries moved to
``pyvwf.sampling`` on 2026-09-25.
"""

from __future__ import annotations

from pathlib import Path

from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import geopandas as gpd
import numpy as np
import pandas as pd
import warnings

from pyvwf.config import PyVWFPaths

#: Every KMeans call runs on one thread. Its partition depends on the thread
#: count: on the synthetic NL grid, 1 and 8 threads give one labelling and 2
#: and 4 give two others, because the threaded reductions sum in a different
#: order and a near-tie falls the other way. Until 2026-09-24 importing pyvwf set
#: OMP/BLAS threads to 1 for the whole process as a side effect of the removed
#: legacy module, which is what every published partition was made under, so
#: one thread reproduces them and is the same on any machine.
_ONE_THREAD = {"limits": 1}


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
    if geom is None or geom.is_empty:
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
    code_map = {"UK": "GB", "GB": "UK"}
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


def get_country_shape(country_code: str, cluster_mode: str = "onshore", repair: bool = False):
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
    country_shapes, offshore_shapes = load_region_shapes()

    # Map UK <-> GB
    code_map = {"UK": "GB", "GB": "UK"}
    codes_to_try = [country_code, code_map.get(country_code, country_code)]

    if cluster_mode == "offshore":
        shapes_df = offshore_shapes
    else:
        shapes_df = country_shapes

    if shapes_df.empty:
        return None

    # Try to find the country
    for code in codes_to_try:
        matches = shapes_df[shapes_df["name"] == code]
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
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


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
        init="k-means++", n_clusters=num_clu, n_init=10, max_iter=300, random_state=random_state
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
    with threadpool_limits(**_ONE_THREAD):
        kmeans.fit(train_coords, sample_weight=sample_weight)

    # The merge is derived from the TRAINING partition, so train and apply
    # resolve identical labels given the same training fleet and seed.
    remap = None
    if min_cluster_size and min_cluster_size > 1:
        with threadpool_limits(**_ONE_THREAD):
            train_labels = kmeans.predict(train_coords)
        remap = _merge_undersized(train_labels, kmeans.cluster_centers_, int(min_cluster_size))
        merged = sum(1 for src, dst in remap.items() if src != dst)
        if merged:
            warnings.warn(
                f"cluster_turbines: merged {merged} cluster(s) with fewer than "
                f"{int(min_cluster_size)} training sites into their nearest "
                f"neighbour; {len(set(remap.values()))} of {num_clu} clusters remain"
            )

    def _labels(frame):
        with threadpool_limits(**_ONE_THREAD):
            out = kmeans.predict(_cluster_coords(frame, geographic))
        return np.array([remap[int(c)] for c in out]) if remap else out

    if train:
        turb_info_train["cluster"] = _labels(turb_info_train)
        return turb_info_train
    else:
        turb_info = args[0]
        turb_info["cluster"] = _labels(turb_info)
        return turb_info
