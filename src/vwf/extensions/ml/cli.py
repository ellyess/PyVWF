"""``pyvwf-ml``: train ML correction models and export gridded predictions.

Thin argparse wrapper around :func:`vwf.extensions.ml.export_ml_correction_grid`.
For model-comparison experiments and feature ablations use the research scripts
under ``scripts/pyvwf_ml/``.

Boosted-tree backends require the ``[ml]`` extra::

    pip install pyvwf[ml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vwf.extensions.ml import export_ml_correction_grid


MODEL_CHOICES = [
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "lightgbm",
    "ridge",
    "lasso",
    "elastic_net",
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pyvwf-ml",
        description="Train a terrain-aware ML correction model and export a grid.",
    )
    p.add_argument(
        "--corrections-csv",
        type=Path,
        required=True,
        help="CSV of correction points (typically all_corrections_centroids.csv).",
    )
    p.add_argument(
        "--grid-nc",
        type=Path,
        required=True,
        help="NetCDF defining the output grid (an atlite cutout works).",
    )
    p.add_argument("--out-nc", type=Path, required=True, help="Output NetCDF path.")
    p.add_argument(
        "--terrain-nc",
        type=Path,
        default=None,
        help="Terrain feature NetCDF (elevation, slope, roughness, ...).",
    )
    p.add_argument(
        "--coastline-geojson",
        type=Path,
        default=None,
        help="Coastline GeoJSON for distance-to-coast feature.",
    )
    p.add_argument(
        "--onshore-mask-geojson",
        type=Path,
        default=None,
        help="GeoJSON mask applied to onshore predictions.",
    )
    p.add_argument(
        "--offshore-mask-geojson",
        type=Path,
        default=None,
        help="GeoJSON mask applied to offshore predictions.",
    )
    p.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=MODEL_CHOICES,
        help="ML model backend.",
    )
    p.add_argument("--n-estimators", type=int, default=200, help="Trees for tree-based models.")
    p.add_argument("--max-depth", type=int, default=15, help="Max depth for tree-based models.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    model_kwargs = {"n_estimators": args.n_estimators, "max_depth": args.max_depth}
    out = export_ml_correction_grid(
        corrections_csv=args.corrections_csv,
        grid_nc=args.grid_nc,
        out_nc=args.out_nc,
        terrain_nc=args.terrain_nc,
        coastline_geojson=args.coastline_geojson,
        onshore_mask_geojson=args.onshore_mask_geojson,
        offshore_mask_geojson=args.offshore_mask_geojson,
        model_type=args.model_type,
        **model_kwargs,
    )
    print(f"[pyvwf-ml] wrote {out}")


if __name__ == "__main__":
    main()
