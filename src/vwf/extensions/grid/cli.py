"""``pyvwf-grid``: kriging-based export of correction fields to an atlite cutout.

Thin argparse wrapper around :func:`vwf.extensions.grid.export_pyvwf_grid`.
For the full multi-method interpolation comparison and CV, use the research
script ``scripts/pyvwf_to_grid/compare_unified_corrections_to_grid.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vwf.extensions.grid import export_pyvwf_grid


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pyvwf-grid",
        description="Interpolate PyVWF correction points onto an atlite cutout grid.",
    )
    p.add_argument("--cutout-nc", type=Path, required=True, help="Atlite cutout NetCDF.")
    p.add_argument(
        "--points-csv",
        type=Path,
        required=True,
        help="CSV of correction control points (typically all_corrections_centroids.csv).",
    )
    p.add_argument("--out-nc", type=Path, required=True, help="Output NetCDF path.")
    p.add_argument(
        "--onshore-geojson",
        type=Path,
        required=True,
        help="GeoJSON of onshore polygons.",
    )
    p.add_argument(
        "--offshore-geojson",
        type=Path,
        required=True,
        help="GeoJSON of offshore polygons.",
    )
    p.add_argument(
        "--variogram-model",
        type=str,
        default="spherical",
        choices=["spherical", "exponential", "gaussian", "linear"],
        help="Variogram model for kriging.",
    )
    p.add_argument(
        "--n-closest-onshore",
        type=int,
        default=50,
        help="Nearest neighbours for local kriging onshore.",
    )
    p.add_argument(
        "--n-closest-offshore",
        type=int,
        default=80,
        help="Nearest neighbours for local kriging offshore.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for kriging.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    out = export_pyvwf_grid(
        cutout_nc=args.cutout_nc,
        points_csv=args.points_csv,
        out_nc=args.out_nc,
        onshore_geojson=args.onshore_geojson,
        offshore_geojson=args.offshore_geojson,
        variogram_model=args.variogram_model,
        n_closest_onshore=args.n_closest_onshore,
        n_closest_offshore=args.n_closest_offshore,
        workers=args.workers,
    )
    print(f"[pyvwf-grid] wrote {out}")


if __name__ == "__main__":
    main()
