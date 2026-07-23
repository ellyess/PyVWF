#!/usr/bin/env python
"""D4: export the gridded corrected-wind/CF NetCDF for the AU-NEM demo.

Produces a self-describing artifact over the NEM box for the held-out year:

- ``wnd100m_uncorrected``  daily ERA5 100 m wind speed (m/s)
- ``wnd100m_corrected``    the above with PyVWF's per-cluster SEASONAL
                           affine correction applied (m/s)
- ``cf_iec2_corrected``    illustrative capacity factor from the corrected
                           wind via the OPEN IEC Class 2 industry composite
                           (dimensionless; generic machine, not a site fleet)
- ``cluster``              which trained cluster's factors each cell uses
- ``km_to_centroid``       distance to that cluster's centroid (judge the
                           extrapolation yourself)

Spatial extension is nearest-centroid (Voronoi) from the 5 trained clusters:
the corrections were TRAINED AT FARM LOCATIONS; cells far from any training
farm carry extrapolated corrections. That caveat is written into the file's
attributes, deliberately loudly.

The factors are fitted parameters (results); no licensed curve content
enters the file. Attribution attributes cover ERA5/Copernicus, AEMO, and
the Global Wind Power Tracker (CC BY 4.0).

    PYVWF_INPUT=<stage> python scripts/region_tools/export_au_grid_netcdf.py \\
        --train-run <dir with factors_season_5.csv + train_turb_info_5.csv> \\
        --open-curves <power_curves_open_smoothed_cf.csv> --year 2023
"""
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import vwf
from vwf.datasets.era5 import prep_era5
from vwf.time_utils import add_time_resolution_columns

SH_SEASONS = {"summer": [12, 1, 2], "autumn": [3, 4, 5],
              "winter": [6, 7, 8], "spring": [9, 10, 11]}
BBOX = (129.0, 154.0, -44.0, -10.0)
IEC2 = "IEC_Class2_Normalized_Industry_Composite"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-run", required=True)
    ap.add_argument("--open-curves", required=True)
    ap.add_argument("--era5-dir", default=None, help="default: <PYVWF_INPUT>/era5/AU_daily")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--out", default="output/validation/AU-NEM/au_nem_grid.nc")
    args = ap.parse_args()

    train_run = Path(args.train_run)
    factors = pd.read_csv(train_run / "factors_season_5.csv")
    fleet = pd.read_csv(train_run / "train_turb_info_5.csv")
    centroids = fleet.groupby("cluster")[["lon", "lat"]].mean()

    from vwf.config import PyVWFPaths
    era5_dir = Path(args.era5_dir) if args.era5_dir else PyVWFPaths.INPUT_ROOT / "era5" / "AU_daily"
    ds = prep_era5("AU-NEM", calc_z0=True, bbox=BBOX, era5_dir=era5_dir)
    ds = ds.sel(time=str(args.year))

    # --- nearest-centroid cluster per cell (approx km via lat-scaled degrees) ---
    lon2d, lat2d = np.meshgrid(ds.lon.values, ds.lat.values)
    coslat = np.cos(np.radians(lat2d))
    dists = np.stack([
        np.hypot((lon2d - c.lon) * 111.0 * coslat, (lat2d - c.lat) * 111.0)
        for c in centroids.itertuples()
    ])
    cluster_2d = centroids.index.values[np.argmin(dists, axis=0)]
    km_2d = dists.min(axis=0).astype("float32")

    # --- seasonal factors per (time, cell) -----------------------------------
    months = pd.DataFrame({"month": ds.time.dt.month.values})
    months = add_time_resolution_columns(months, SH_SEASONS)
    fac = factors.set_index(["cluster", "season"])
    scal_t = np.empty((ds.sizes["time"], *cluster_2d.shape), dtype="float32")
    off_t = np.empty_like(scal_t)
    for season in SH_SEASONS:
        t_idx = np.where(months["season"].values == season)[0]
        scal_2d = np.vectorize(lambda c: fac.loc[(c, season), "scalar"])(cluster_2d)
        off_2d = np.vectorize(lambda c: fac.loc[(c, season), "offset"])(cluster_2d)
        scal_t[t_idx] = scal_2d.astype("float32")
        off_t[t_idx] = off_2d.astype("float32")

    unc = ds["wnd100m"].astype("float32")
    cor = (unc * scal_t + off_t).clip(min=0.0)

    curves = pd.read_csv(args.open_curves)
    speeds = curves[curves.columns[0]].to_numpy(float)
    cf_curve = curves[IEC2].to_numpy(float)
    cf = xr.DataArray(
        np.interp(cor.values, speeds, cf_curve).astype("float32"),
        coords=cor.coords, dims=cor.dims,
    )

    def _git(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=10, check=True).stdout.strip()
        except Exception:
            return "unknown"

    out = xr.Dataset(
        {
            "wnd100m_uncorrected": unc.assign_attrs(
                units="m s-1", long_name="ERA5 100 m wind speed, daily mean"),
            "wnd100m_corrected": cor.assign_attrs(
                units="m s-1",
                long_name="PyVWF seasonally corrected 100 m wind speed, daily mean"),
            "cf_iec2_corrected": cf.assign_attrs(
                units="1",
                long_name="Illustrative capacity factor (corrected wind through the "
                          "open IEC Class 2 industry composite; generic machine, "
                          "not a site fleet)"),
            "cluster": xr.DataArray(cluster_2d.astype("int8"),
                                    coords={"lat": ds.lat, "lon": ds.lon},
                                    dims=("lat", "lon")).assign_attrs(
                long_name="trained correction cluster applied (nearest centroid)"),
            "km_to_centroid": xr.DataArray(km_2d,
                                           coords={"lat": ds.lat, "lon": ds.lon},
                                           dims=("lat", "lon")).assign_attrs(
                units="km", long_name="distance to the applied cluster's centroid"),
        },
        attrs={
            "title": f"PyVWF corrected wind and illustrative CF, Australian NEM, {args.year}",
            "summary": "Per-cluster seasonal affine wind-speed corrections trained on "
                       "AEMO 2020-2022 farm observations, applied to the ERA5 grid. "
                       "Headline finding: ERA5 over-amplifies South Australia's "
                       "seasonal cycle; the correction compresses it toward "
                       "observation (docs/findings/pillar_a_au.md).",
            "EXTRAPOLATION_CAVEAT": "Corrections were trained at wind-farm locations "
                                    "and extended to the grid by nearest cluster "
                                    "centroid. Cells far from training farms (see "
                                    "km_to_centroid) carry extrapolated corrections "
                                    "of undemonstrated validity. In particular, "
                                    "far-north cells (lat > -20) lie far from any "
                                    "training cluster after the far-north exclusion "
                                    "(see FACTORS_PROVENANCE): corrections there are "
                                    "DISTANT extrapolations - consult km_to_centroid "
                                    "before using them.",
            "FACTORS_PROVENANCE": "Factors come from the far-north-exclusion training "
                                  "run (sites at lat > -20 excluded) per the "
                                  "robustness analysis in docs/findings/pillar_a_au.md, "
                                  "which removes a known n=1 overfit cluster whose "
                                  "extreme factors (scalars up to 2.4) would otherwise "
                                  "be applied across the tropical north. The findings "
                                  "doc's HEADLINE gate uses the all-farms run; this "
                                  "artifact deliberately ships only corrections the "
                                  "analysis stands behind, so its factors differ from "
                                  "the headline tables.",
            "correction_training": "PyVWF affine-in-wind, 5 clusters x 4 SH seasons, "
                                   "trained 2020-2021-2022, REAL curve-library run, "
                                   "far-north sites excluded "
                                   "(fitted parameters only; no curve content included)",
            "cf_layer_curve": f"{IEC2} (open library; BSD-3-derived, "
                              "NatLabRockies/turbine-models)",
            "seasons_definition": "SH explicit months: winter=JJA, summer=DJF",
            "pyvwf_version": vwf.__version__,
            "git_commit": _git(["git", "rev-parse", "HEAD"]),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "attribution_era5": "Contains modified Copernicus Climate Change Service "
                                "information (ERA5). Neither the European Commission "
                                "nor ECMWF is responsible for any use of this data.",
            "attribution_aemo": "Training observations derived from AEMO data "
                                "(Source: AEMO). AEMO makes no representation as to "
                                "accuracy or completeness.",
            "attribution_gwpt": "Farm coordinates from the Global Wind Power Tracker, "
                                "Global Energy Monitor (CC BY 4.0).",
            "history": "scripts/region_tools/export_au_grid_netcdf.py",
        },
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in out.data_vars}
    out.to_netcdf(out_path, encoding=enc)
    print(f"written: {out_path} ({out_path.stat().st_size/1e6:.0f} MB), "
          f"{out.sizes['time']} days x {out.sizes['lat']}x{out.sizes['lon']} cells")


if __name__ == "__main__":
    main()
