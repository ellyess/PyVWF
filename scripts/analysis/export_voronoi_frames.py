"""Export PyVWF cluster solutions as Voronoi cell geometry for TouchDesigner.

Reads a cluster sweep directory (``train-k*/`` subdirectories produced by
training) and writes a single ``.npz`` holding, for every ``k``, a triangulated
Voronoi tessellation of the cluster centroids clipped to the region boundary,
with per-cell colours from the learned ``scalar`` (alpha) correction.

Why Voronoi over centroids reproduces the clusters exactly: training partitions
turbines with KMeans in raw ``(lat, lon)`` degree space
(:func:`vwf.clustering.cluster_turbines`, ``geographic=False``). At convergence
each centroid is the mean of its assigned points, so nearest-centroid
assignment is the partition, and the Voronoi diagram of the centroids is its
exact geometric dual. The tessellation is therefore built in degree space and
only projected afterwards for display.

TouchDesigner ships numpy but not scipy/shapely, so all geometry work happens
here and TD only loads flat float32 arrays.

Usage::

    PYTHONPATH=src python scripts/analysis/export_voronoi_frames.py \
        --sweep output/validation/dk_onshore_sweep_2026-07-24 \
        --region DK --out output/viz/dk_voronoi_frames.npz
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import geopandas as gpd
import mapbox_earcut
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial import Voronoi
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union


# The correction applies ``cor_ws = ws * scalar + offset`` (vwf.wind), so it is
# a no-op at scalar=1 and offset=0. Those anchor the two diverging scales.
NEUTRAL = 1.0
OFF_NEUTRAL = 0.0

# Fixed colour limits, shared by every frame so cell colours stay comparable
# across the animation. Set from the pooled scalar distribution over the whole
# sweep: p2 on the low side (the tail below this is a handful of tiny
# high-k clusters) and the true max on the high side, which is already close
# to p100. Values outside are clamped, not rescaled.
VMIN, VMAX = 0.42, 1.33

# Offset (m/s) spans the full pooled range; unlike scalar its tails are not
# dominated by a few tiny clusters, so nothing is clipped.
OFF_VMIN, OFF_VMAX = -1.40, 2.78

# Equal-area projection, used only to report boundary areas. Matches the CRS
# vwf.clustering uses for the same comparisons; never used for the exported
# display coordinates.
METRIC_CRS = "EPSG:3035"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", required=True, type=Path,
                   help="Sweep directory containing <region>/train-k*/")
    p.add_argument("--region", default="DK", help="Region code (default: DK)")
    p.add_argument("--out", required=True, type=Path, help="Output .npz path")
    p.add_argument("--shapes", type=Path,
                   default=Path("input/reference/shapes/country_shapes.geojson"),
                   help="GeoJSON of country polygons with a 'name' column")
    p.add_argument("--max-k", type=int, default=1000,
                   help="Highest cluster count to include (default: 1000)")
    p.add_argument("--time-res", default="fixed",
                   help="Factors temporal resolution to read (default: fixed)")
    p.add_argument("--cmap", default="RdBu_r", help="Diverging colormap name")
    p.add_argument("--coastline", type=Path,
                   default=Path("input/reference/terrain/coastlines.geojson"),
                   help="Coastline GeoJSON used to recover islands the country "
                        "shape omits; pass a missing path to disable")
    return p.parse_args()


def discover_ks(region_dir: Path, max_k: int) -> list[int]:
    """Cluster counts with a train directory, ascending, up to ``max_k``."""
    ks = []
    for d in region_dir.glob("train-k*"):
        m = re.fullmatch(r"train-k(\d+)", d.name)
        if m and int(m.group(1)) <= max_k:
            ks.append(int(m.group(1)))
    if not ks:
        raise SystemExit(f"no train-k* directories under {region_dir}")
    return sorted(ks)


def load_boundary(shapes_path: Path, region: str, coastline_path: Path | None,
                  fleet_xy: np.ndarray | None):
    """Land polygon for ``region``, dissolved to a single geometry.

    The bundled country shapes are coarse: the DK polygon holds only Jutland,
    Zealand and Funen (39,208 km² against Denmark's published 42,943), so
    Lolland, Falster, Bornholm, Als, Langeland, Mors and the rest are missing
    and any turbine on them ends up with no land drawn beneath it. The repair
    lives in :func:`vwf.clustering.repair_region_shape`; the fleet is passed in
    so a landmass carrying turbines is kept whatever the EEZ test says.
    """
    from vwf.clustering import repair_region_shape

    gdf = gpd.read_file(shapes_path)
    sel = gdf[gdf["name"].astype(str).str.upper() == region.upper()]
    if sel.empty:
        raise SystemExit(
            f"region {region!r} not in {shapes_path} "
            f"(have: {sorted(gdf['name'].astype(str).unique())})"
        )
    base_geom = unary_union(sel.geometry)
    if coastline_path is not None and not coastline_path.exists():
        return base_geom

    merged = repair_region_shape(base_geom, region.upper(), fleet_xy=fleet_xy,
                                 coastline_path=coastline_path)
    if merged is base_geom or merged.equals(base_geom):
        return base_geom

    def _km2(g):
        return gpd.GeoSeries([g], crs=gdf.crs).to_crs(METRIC_CRS).area.iloc[0] / 1e6

    print(f"boundary: merged coastline landmasses, "
          f"{_km2(base_geom):,.0f} -> {_km2(merged):,.0f} km2")
    return merged


def load_skill(sweep: Path, ks: list[int], time_res: str
               ) -> tuple[np.ndarray, float] | tuple[None, None]:
    """Fleet RMSE per solution, plus the uncorrected baseline.

    Read from the sweep's ``combined_metrics.csv`` so the number on screen is
    the same one the validation tables report: variant ``affine-wind``, scope
    ``fleet``, matching ``time_res``. The baseline is the ``uncorrected``
    variant, constant across k.

    Returns ``(None, None)`` when the metrics file is absent, so the export
    still succeeds without a skill readout.
    """
    path = sweep / "combined_metrics.csv"
    if not path.exists():
        print(f"note: {path.name} not found; exporting without RMSE")
        return None, None

    m = pd.read_csv(path)
    base = m[m["variant"] == "uncorrected"]["rmse"]
    corrected = m[(m["variant"] == "affine-wind")
                  & (m["time_res"] == time_res)
                  & (m["scope"] == "fleet")].set_index("num_clu")["rmse"]

    missing = [k for k in ks if k not in corrected.index]
    if base.empty or missing:
        print(f"note: metrics incomplete (missing k={missing}); exporting without RMSE")
        return None, None

    return corrected.reindex(ks).to_numpy(dtype=float), float(base.iloc[0])


def voronoi_cells(centroids: np.ndarray, boundary) -> list:
    """Voronoi cell per centroid, clipped to ``boundary``.

    ``centroids`` is (n, 2) as (lon, lat). Eight points on a far circle bound
    every real cell, so no region runs to infinity and the clip is exact.
    Returns a list of shapely geometries, one per input row and index-aligned
    with it (possibly empty where a cell misses the boundary entirely).
    """
    if len(centroids) == 1:
        return [boundary]

    centre = centroids.mean(axis=0)
    radius = float(np.abs(centroids - centre).max()) * 100.0 + 10.0
    angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    far = centre + radius * np.column_stack([np.cos(angles), np.sin(angles)])

    vor = Voronoi(np.vstack([centroids, far]))

    cells = []
    for i in range(len(centroids)):
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region:
            # Should not happen given the bounding ring, but never emit a
            # half-open cell silently.
            cells.append(Polygon())
            continue
        poly = Polygon(vor.vertices[region])
        if not poly.is_valid:
            poly = poly.buffer(0)
        cells.append(poly.intersection(boundary))
    return cells


def triangulate(geom) -> np.ndarray:
    """Triangulate a (Multi)Polygon into an (n_tri*3, 2) vertex array.

    Uses earcut per polygon part, which handles concave rings and holes, so
    cells clipped against a real coastline come out correct.
    """
    if geom.is_empty:
        return np.zeros((0, 2), dtype=np.float64)

    parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    out = []
    for part in parts:
        if not isinstance(part, Polygon) or part.is_empty:
            continue
        rings = [np.asarray(part.exterior.coords[:-1], dtype=np.float64)]
        rings += [np.asarray(r.coords[:-1], dtype=np.float64)
                  for r in part.interiors]
        rings = [r for r in rings if len(r) >= 3]
        if not rings:
            continue
        verts = np.vstack(rings)
        ring_ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
        idx = mapbox_earcut.triangulate_float64(verts, ring_ends)
        if len(idx):
            out.append(verts[idx])
    if not out:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack(out)


def outline_segments(geom) -> np.ndarray:
    """Boundary of a (Multi)Polygon as an (n_seg*2, 2) line-segment array."""
    if geom.is_empty:
        return np.zeros((0, 2), dtype=np.float64)
    parts = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    out = []
    for part in parts:
        if not isinstance(part, Polygon) or part.is_empty:
            continue
        for ring in [part.exterior, *part.interiors]:
            c = np.asarray(ring.coords, dtype=np.float64)
            if len(c) < 2:
                continue
            seg = np.empty((2 * (len(c) - 1), 2), dtype=np.float64)
            seg[0::2] = c[:-1]
            seg[1::2] = c[1:]
            out.append(seg)
    if not out:
        return np.zeros((0, 2), dtype=np.float64)
    return np.vstack(out)


def main() -> None:
    args = parse_args()
    region_dir = args.sweep / args.region
    if not region_dir.is_dir():
        raise SystemExit(f"not a directory: {region_dir}")

    ks = discover_ks(region_dir, args.max_k)
    # The fleet is identical at every k, so any solution supplies the points
    # used to claim coastline landmasses for this region.
    seed_turb = pd.read_csv(region_dir / f"train-k{ks[0]}" / f"train_turb_info_{ks[0]}.csv")
    boundary = load_boundary(args.shapes, args.region, args.coastline,
                             seed_turb[["lon", "lat"]].to_numpy(dtype=float))
    print(f"region {args.region}: {len(ks)} solutions {ks[0]}..{ks[-1]}")

    # Equal-area-ish display transform: degrees are the clustering space, but
    # 1 deg lon is only cos(lat) as wide as 1 deg lat, so plotting raw lon/lat
    # would stretch the map east-west by ~1.8x at Denmark's latitude.
    minx, miny, maxx, maxy = boundary.bounds
    lon0, lat0 = 0.5 * (minx + maxx), 0.5 * (miny + maxy)
    kx = float(np.cos(np.radians(lat0)))

    def to_display(xy: np.ndarray) -> np.ndarray:
        if len(xy) == 0:
            return xy.astype(np.float32)
        out = np.empty_like(xy)
        out[:, 0] = (xy[:, 0] - lon0) * kx
        out[:, 1] = xy[:, 1] - lat0
        return out

    # Uniform scale from the boundary extent so the map fills roughly [-1, 1]
    # with aspect preserved.
    half = max((maxx - minx) * kx, maxy - miny) / 2.0
    scale = 1.0 / half

    norm = TwoSlopeNorm(vcenter=NEUTRAL, vmin=VMIN, vmax=VMAX)
    off_norm = TwoSlopeNorm(vcenter=OFF_NEUTRAL, vmin=OFF_VMIN, vmax=OFF_VMAX)
    cmap = colormaps[args.cmap]

    payload: dict[str, np.ndarray] = {}
    summary = []

    # Per-turbine tracks, for the particle layer. The turbine set is identical
    # and identically ordered at every k (verified: same IDs, same positions),
    # so alpha per turbine is a dense (n_k, n_turb) array that interpolates
    # continuously between solutions even though the cell topology does not.
    turb_xy: np.ndarray | None = None
    turb_alpha_rows: list[np.ndarray] = []
    turb_offset_rows: list[np.ndarray] = []
    turb_centroid_rows: list[np.ndarray] = []

    for k in ks:
        tdir = region_dir / f"train-k{k}"
        turb = pd.read_csv(tdir / f"train_turb_info_{k}.csv")
        fac = pd.read_csv(tdir / f"factors_{args.time_res}_{k}.csv")

        # Centroid = mean of assigned turbines, in the KMeans (lat, lon) space.
        cen = turb.groupby("cluster")[["lon", "lat"]].mean().sort_index()
        cluster_ids = cen.index.to_numpy()
        centroids = cen.to_numpy(dtype=float)

        alpha = (fac.groupby("cluster")["scalar"].mean()
                 .reindex(cluster_ids).to_numpy(dtype=float))
        offset = (fac.groupby("cluster")["offset"].mean()
                  .reindex(cluster_ids).to_numpy(dtype=float))
        if np.isnan(alpha).any() or np.isnan(offset).any():
            raise SystemExit(f"k={k}: factors table missing clusters present in turb info")

        cells = voronoi_cells(centroids, boundary)
        rgba = cmap(norm(np.clip(alpha, VMIN, VMAX)))
        rgba_off = cmap(off_norm(np.clip(offset, OFF_VMIN, OFF_VMAX)))

        tri_xy, tri_rgb, tri_rgb_off, edge_xy = [], [], [], []
        for cell, colour, colour_off in zip(cells, rgba, rgba_off):
            tris = triangulate(cell)
            if len(tris):
                tri_xy.append(tris)
                tri_rgb.append(np.repeat(colour[None, :3], len(tris), axis=0))
                tri_rgb_off.append(np.repeat(colour_off[None, :3], len(tris), axis=0))
            segs = outline_segments(cell)
            if len(segs):
                edge_xy.append(segs)

        tri_xy = np.vstack(tri_xy) if tri_xy else np.zeros((0, 2))
        tri_rgb = np.vstack(tri_rgb) if tri_rgb else np.zeros((0, 3))
        tri_rgb_off = np.vstack(tri_rgb_off) if tri_rgb_off else np.zeros((0, 3))
        edge_xy = np.vstack(edge_xy) if edge_xy else np.zeros((0, 2))

        payload[f"tri_xy_{k}"] = (to_display(tri_xy) * scale).astype(np.float32)
        payload[f"tri_rgb_{k}"] = tri_rgb.astype(np.float32)
        payload[f"tri_rgb_off_{k}"] = tri_rgb_off.astype(np.float32)
        payload[f"edge_xy_{k}"] = (to_display(edge_xy) * scale).astype(np.float32)
        payload[f"alpha_{k}"] = alpha.astype(np.float32)
        payload[f"offset_{k}"] = offset.astype(np.float32)

        # Particle tracks: each turbine's own alpha and its cluster's centroid.
        pos = turb[["lon", "lat"]].to_numpy(dtype=float)
        if turb_xy is None:
            turb_xy = pos
        elif not np.allclose(pos, turb_xy):
            raise SystemExit(
                f"k={k}: turbine positions differ from the first solution; the "
                "particle layer assumes one fixed, identically ordered fleet"
            )
        lookup = {int(c): j for j, c in enumerate(cluster_ids)}
        rows = turb["cluster"].map(lookup).to_numpy()
        turb_alpha_rows.append(alpha[rows])
        turb_offset_rows.append(offset[rows])
        turb_centroid_rows.append(centroids[rows])

        n_tri = len(tri_xy) // 3
        summary.append((k, len(cells), n_tri))
        print(f"  k={k:5d}  cells={len(cells):5d}  triangles={n_tri:6d}  "
              f"alpha[{alpha.min():.3f}, {alpha.max():.3f}]")

    coast = outline_segments(boundary)
    payload["coast_xy"] = (to_display(coast) * scale).astype(np.float32)
    payload["ks"] = np.asarray(ks, dtype=np.int32)
    payload["colour_limits"] = np.asarray([VMIN, NEUTRAL, VMAX], dtype=np.float32)

    payload["turb_xy"] = (to_display(turb_xy) * scale).astype(np.float32)
    payload["turb_alpha"] = np.vstack(turb_alpha_rows).astype(np.float32)
    payload["turb_offset"] = np.vstack(turb_offset_rows).astype(np.float32)
    payload["turb_centroid_xy"] = np.stack(
        [to_display(c) * scale for c in turb_centroid_rows]).astype(np.float32)

    # Colour lookup table so TouchDesigner can shade a continuously
    # interpolated alpha without reimplementing TwoSlopeNorm: the norm is baked
    # in by sampling alpha uniformly across [VMIN, VMAX], so TD only needs a
    # linear index. Keep in step with the cell colours, which use the same norm.
    lut_alpha = np.linspace(VMIN, VMAX, 256)
    payload["alpha_lut"] = cmap(norm(lut_alpha))[:, :3].astype(np.float32)
    lut_off = np.linspace(OFF_VMIN, OFF_VMAX, 256)
    payload["offset_lut"] = cmap(off_norm(lut_off))[:, :3].astype(np.float32)
    payload["offset_limits"] = np.asarray(
        [OFF_VMIN, OFF_NEUTRAL, OFF_VMAX], dtype=np.float32)

    rmse, rmse_unc = load_skill(args.sweep, ks, args.time_res)
    if rmse is not None:
        payload["rmse"] = rmse.astype(np.float32)
        payload["rmse_uncorrected"] = np.float32(rmse_unc)
        print(f"\nfleet RMSE ({args.time_res}): uncorrected {rmse_unc:.5f}, "
              f"corrected {rmse.min():.5f}..{rmse.max():.5f} "
              f"({100*(rmse.min()-rmse_unc)/rmse_unc:+.1f}%.."
              f"{100*(rmse.max()-rmse_unc)/rmse_unc:+.1f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)

    meta = {
        "region": args.region,
        "sweep": str(args.sweep),
        "time_res": args.time_res,
        "ks": ks,
        "cmap": args.cmap,
        "colour_limits": {"vmin": VMIN, "vcenter": NEUTRAL, "vmax": VMAX},
        "display_transform": {"lon0": lon0, "lat0": lat0, "kx": kx, "scale": scale},
        "n_turbines": int(len(turb_xy)),
        "frames": [{"k": k, "cells": c, "triangles": t} for k, c, t in summary],
    }
    args.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))

    size_mb = args.out.stat().st_size / 1e6
    print(f"\nwrote {args.out} ({size_mb:.2f} MB) and {args.out.with_suffix('.json').name}")


if __name__ == "__main__":
    main()
