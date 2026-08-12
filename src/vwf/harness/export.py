"""Export a validated region's correction as an atlite-ready gridded field.

This turns a trained harness run into a self-describing NetCDF holding, for every
ERA5 grid cell in the region and every time slice, the affine wind-speed
correction (``scalar``, ``offset``) that cell would receive, plus which trained
cluster it came from and how far that cell is from the cluster's centroid. The
result drops into an existing atlite or PyPSA-Eur wind pipeline at wind-speed
level, without that pipeline needing PyVWF itself.

Spatial extension is nearest-centroid (Voronoi) from the trained clusters: the
corrections were fitted AT FARM LOCATIONS, so cells far from any training farm
carry extrapolated corrections. That caveat is written into the file's global
attributes, and every cell ships with ``km_to_centroid`` so the user can judge
the extrapolation themselves. Nearest-centroid is deliberately conservative;
a smoother interpolation (IDW from control points, thesis Chapter 4) is a future
refinement, not a claim this file makes.

Only fitted parameters enter the file (scalars and offsets are results); no
licensed curve content is included, so the artifact is redistributable subject
only to the ERA5/Copernicus and observation-source attributions carried in the
attributes.

Generalises the one-off ``scripts/region_tools/export_au_grid_netcdf.py`` to any
region config and any of the four time slices, and (unlike that demo, which
emitted a full applied time series) exports the compact static factor field that
is the actual drop-in product.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import vwf
from vwf.config import PyVWFPaths
from vwf.datasets.era5 import prep_era5
from vwf.harness.regions import RegionSpec


def _order_slices(labels: list, time_res: str, spec: RegionSpec) -> list:
    """Put slice labels in natural reading order for the ``slice`` coordinate."""
    labels = list(labels)
    if time_res == "month":
        return sorted(labels, key=int)
    if time_res == "bimonth":
        return sorted(labels, key=lambda s: int(str(s).split("/")[0]))
    if time_res == "season":
        first_month = {name: min(months) for name, months in spec.seasons.items()}
        return sorted(labels, key=lambda s: first_month.get(s, 99))
    return labels  # fixed: single "1/1"


def _nearest_centroid(lon: np.ndarray, lat: np.ndarray, centroids: pd.DataFrame):
    """Cluster id and distance-to-centroid (km) for every cell of a lon/lat grid."""
    lon2d, lat2d = np.meshgrid(lon, lat)
    coslat = np.cos(np.radians(lat2d))
    dists = np.stack([
        np.hypot((lon2d - c.lon) * 111.0 * coslat, (lat2d - c.lat) * 111.0)
        for c in centroids.itertuples()
    ])
    cluster_2d = centroids.index.values[np.argmin(dists, axis=0)]
    km_2d = dists.min(axis=0).astype("float32")
    return cluster_2d, km_2d


def _validation_note(
    metrics_csv: Path | None, num_clu: int, time_res: str, model: str
) -> str:
    """One-line uncorrected-vs-corrected summary for this (k, slice), from metrics.csv."""
    if metrics_csv is None or not Path(metrics_csv).exists():
        return "not attached (pass --metrics to embed the held-out validation result)"
    m = pd.read_csv(metrics_csv)
    # Prefer the fleet/national scope; fall back to whatever is present.
    scope = "fleet" if "fleet" in m.get("scope", pd.Series()).values else (
        "national" if "national" in m.get("scope", pd.Series()).values else None
    )
    if scope is not None:
        m = m[m["scope"] == scope]
    unc = m[m["variant"] == "uncorrected"]
    cor = m[(m["variant"] == model) & (m["num_clu"] == num_clu)
            & (m["time_res"] == time_res)]
    if unc.empty or cor.empty:
        return "metrics.csv present but no matching rows for this (k, slice)"
    u, c = unc.iloc[0], cor.iloc[0]
    return (
        f"held-out RMSE {u['rmse']:.4f} -> {c['rmse']:.4f}, "
        f"MBE {u['mbe']:+.4f} -> {c['mbe']:+.4f}, "
        f"r {u['pearson_r']:.3f} -> {c['pearson_r']:.3f} "
        f"(uncorrected -> corrected; {model}, k={num_clu}, {time_res})"
    )


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def export_correction_field(
    spec: RegionSpec,
    train_run_dir: str | Path,
    *,
    num_clu: int,
    time_res: str,
    out_path: str | Path,
    metrics_csv: str | Path | None = None,
    era5_dir: str | Path | None = None,
) -> Path:
    """Write the gridded correction-factor field for one trained (k, slice).

    Args:
        spec: The region config (bbox, seasons, correction model, name).
        train_run_dir: A harness train run holding ``factors_<slice>_<k>.csv``
            and ``train_turb_info_<k>.csv``.
        num_clu: Cluster count to export (must match a trained factors file).
        time_res: Time slice to export (``fixed``/``season``/``bimonth``/``month``).
        out_path: Destination ``.nc`` path.
        metrics_csv: Optional evaluate-run ``metrics.csv`` whose held-out result
            for this ``(k, slice)`` is embedded as the validation note.
        era5_dir: Optional ERA5 directory; defaults to ``<INPUT>/<spec.era5_path>``.

    Returns:
        The written NetCDF path.
    """
    train_run_dir = Path(train_run_dir)
    factors = pd.read_csv(train_run_dir / f"factors_{time_res}_{num_clu}.csv")
    fleet = pd.read_csv(train_run_dir / f"train_turb_info_{num_clu}.csv")
    centroids = fleet.groupby("cluster")[["lon", "lat"]].mean()
    # Training support behind each cluster: a cell whose correction comes from a
    # one-farm cluster is far less trustworthy than one backed by many farms.
    n_train = fleet.groupby("cluster").size()
    cap_train = fleet.groupby("cluster")["capacity"].sum() / 1000.0  # kW -> MW

    era5_dir = Path(era5_dir) if era5_dir else PyVWFPaths.INPUT_ROOT / spec.era5_path
    grid = prep_era5(spec.code, calc_z0=False, bbox=spec.bbox, era5_dir=era5_dir)
    grid = grid.isel(time=0)
    lon, lat = grid.lon.values, grid.lat.values

    cluster_2d, km_2d = _nearest_centroid(lon, lat, centroids)
    n_train_2d = np.vectorize(lambda c: int(n_train.get(c, 0)))(cluster_2d).astype("int16")
    cap_train_2d = np.vectorize(lambda c: float(cap_train.get(c, 0.0)))(cluster_2d).astype("float32")

    labels = _order_slices(list(factors[time_res].drop_duplicates()), time_res, spec)
    scal = np.empty((len(labels), *cluster_2d.shape), dtype="float32")
    off = np.empty_like(scal)
    for i, lab in enumerate(labels):
        sub = factors[factors[time_res] == lab].set_index("cluster")
        s_map = sub["scalar"].to_dict()
        o_map = sub["offset"].to_dict()

        def _s(c, s_map=s_map):
            v = s_map.get(c, np.nan)
            return 1.0 if pd.isna(v) else float(v)  # missing/NaN scalar -> identity

        def _o(c, o_map=o_map):
            v = o_map.get(c, np.nan)
            return 0.0 if pd.isna(v) else float(v)  # missing/NaN offset -> no shift

        scal[i] = np.vectorize(_s)(cluster_2d).astype("float32")
        off[i] = np.vectorize(_o)(cluster_2d).astype("float32")

    slice_coord = [str(s) for s in labels]
    dims2d = ("lat", "lon")
    coords = {"slice": slice_coord, "lat": lat, "lon": lon}
    note = _validation_note(
        Path(metrics_csv) if metrics_csv else None, num_clu, time_res,
        spec.correction_model,
    )

    out = xr.Dataset(
        {
            "scalar": xr.DataArray(scal, coords=coords, dims=("slice", *dims2d)).assign_attrs(
                units="1",
                long_name="affine wind-speed correction multiplier (cor_ws = "
                          "scalar * era5_ws + offset), nearest trained cluster",
            ),
            "offset": xr.DataArray(off, coords=coords, dims=("slice", *dims2d)).assign_attrs(
                units="m s-1",
                long_name="affine wind-speed correction offset, nearest trained cluster",
            ),
            "cluster": xr.DataArray(
                cluster_2d.astype("int16"), coords={"lat": lat, "lon": lon}, dims=dims2d
            ).assign_attrs(long_name="trained cluster applied to this cell (nearest centroid)"),
            "km_to_centroid": xr.DataArray(
                km_2d, coords={"lat": lat, "lon": lon}, dims=dims2d
            ).assign_attrs(
                units="km",
                long_name="distance from this cell to its applied cluster's centroid; "
                          "large values flag extrapolation beyond the training fleet",
            ),
            "cluster_n_train": xr.DataArray(
                n_train_2d, coords={"lat": lat, "lon": lon}, dims=dims2d
            ).assign_attrs(
                units="1",
                long_name="number of training farms behind the applied cluster; low "
                          "counts (especially 1) mean the correction here rests on thin "
                          "support and should be treated as indicative",
            ),
            "cluster_capacity_mw": xr.DataArray(
                cap_train_2d, coords={"lat": lat, "lon": lon}, dims=dims2d
            ).assign_attrs(
                units="MW",
                long_name="installed capacity behind the applied cluster in the training "
                          "fleet (a second measure of how well-supported the correction is)",
            ),
        },
        attrs={
            "title": f"PyVWF gridded wind-speed correction field, {spec.name}",
            "summary": f"Per-cluster affine wind-speed corrections ({spec.correction_model}, "
                       f"{num_clu} clusters x {time_res} slices) trained on observed wind-farm "
                       f"capacity factors and mapped to the ERA5 0.25deg grid by nearest cluster "
                       f"centroid. Apply as cor_ws = scalar * era5_ws + offset before a power "
                       f"curve; drops into an atlite wind pipeline at wind-speed level.",
            "region_code": spec.code,
            "region_name": spec.name,
            "correction_model": spec.correction_model,
            "num_clusters": int(num_clu),
            "time_resolution": time_res,
            "seasons_definition": "; ".join(
                f"{name}={sorted(months)}" for name, months in spec.seasons.items()
            ),
            "validation_note": note,
            "confidence_layers": "Per-cell trust is carried by three fields: km_to_centroid "
                                 "(distance from the training fleet), cluster_n_train and "
                                 "cluster_capacity_mw (how much observed data trained the "
                                 "applied cluster). A cell that is far from the fleet or "
                                 "drawn from a one-farm cluster is indicative only. A "
                                 "per-cluster held-out error layer is a planned addition.",
            "EXTRAPOLATION_CAVEAT": "Corrections were trained at wind-farm locations and "
                                    "extended to the grid by nearest cluster centroid. Cells "
                                    "far from any training farm (see km_to_centroid) carry "
                                    "extrapolated corrections of undemonstrated validity; "
                                    "consult km_to_centroid before using a cell, and treat "
                                    "cells beyond the fleet's spatial footprint as indicative "
                                    "only.",
            "usage": "cor_ws = scalar * era5_wnd100m + offset, then evaluate a power curve at "
                     "hub height. scalar/offset vary by the 'slice' coordinate (this file's "
                     f"time resolution is '{time_res}').",
            "content_note": "Only fitted parameters (scalars, offsets) are included; no "
                            "power-curve content and no observation records. Redistributable "
                            "under the attributions below.",
            "pyvwf_version": vwf.__version__,
            "git_commit": _git_commit(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "attribution_era5": "Grid derived from ERA5. Contains modified Copernicus Climate "
                                "Change Service information. Neither the European Commission nor "
                                "ECMWF is responsible for any use of this data.",
            "attribution_observations": f"Correction factors trained on {spec.code} wind-farm "
                                        f"capacity-factor observations via the '{spec.source}' "
                                        f"source adapter; see the region's runbook for the "
                                        f"upstream provider and its terms.",
            "history": "vwf.harness.export.export_correction_field",
        },
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(out_path, encoding=enc)
    return out_path
