#!/usr/bin/env python3
"""Compute the Global Wind Atlas speed-up ratio for every unit of every cache.

For the `gwa-ratio` arm of docs/findings/method-physics-informed-turbine-prereg.md.
Each unit gets ``R = GWA / E``: the atlas's mean 100 m speed within 2.5 km,
over the unit's mean daily ERA5 100 m speed in its own split, clipped to the
speed-up's bounds, and 1 where the atlas has no value (``vwf.pinn.gwa``).

Reads the caches and the atlas files, and writes ``<CODE>_<split>.csv`` per
region and split into ``--out``. It also writes ``gwa_record.json``, with the
clipped and neutral shares of capacity, and a ``run_manifest.json``. It
refuses a dirty tree unless ``--allow-dirty`` is given.

Run: PYTHONPATH=src /opt/anaconda3/bin/python scripts/pinn/gwa_ratio.py \
         --cache output/pinn_loco_2026-09-16/cache --gwa input/raw/gwa4 \
         --out output/pinn_turbine_2026-09-18/gwa
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vwf.harness.provenance import build_manifest, write_manifest  # noqa: E402
from vwf.pinn.cache import load_cache  # noqa: E402
from vwf.pinn.gwa import RATIO_BOUNDS, gwa_ratio  # noqa: E402

ISO3 = {"DK": "DNK", "DE": "DEU", "UK": "GBR", "US": "USA", "BR": "BRA",
        "AR": "ARG", "AU-NEM": "AUS", "CL": "CHL", "NZ": "NZL"}
RADIUS_KM = 2.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--gwa", required=True, help="directory of <ISO3>_wind-speed_100m.tif")
    ap.add_argument("--out", required=True)
    ap.add_argument("--regions", nargs="+", default=list(ISO3))
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    launch = build_manifest()
    if launch["git_dirty"] and not args.allow_dirty:
        raise SystemExit("refusing to run on a dirty tree. Commit first, or pass --allow-dirty.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    record, rasters = {}, {}
    for code in args.regions:
        raster = Path(args.gwa) / f"{ISO3[code]}_wind-speed_100m.tif"
        rasters[code] = {"path": str(raster),
                         "sha256": hashlib.sha256(raster.read_bytes()).hexdigest()}
        for split in args.splits:
            cache = load_cache(code, split, args.cache)
            meta = cache.meta
            with np.errstate(invalid="ignore"):
                era5_mean = np.nanmean(np.asarray(cache.w_mean, dtype=float), axis=0)
            table = gwa_ratio(meta["ID"], meta["lon"], meta["lat"], era5_mean, raster,
                              radius_km=RADIUS_KM)
            table["capacity"] = meta["capacity"].to_numpy(dtype=float)
            table.to_csv(out / f"{code}_{split}.csv", index=False)
            cap = table["capacity"].sum()
            record[f"{code}/{split}"] = {
                "units": int(len(table)),
                "capacity_share_clipped": float(table.loc[table.clipped, "capacity"].sum() / cap),
                "capacity_share_neutral": float(table.loc[table.neutral, "capacity"].sum() / cap),
                "units_clipped": int(table.clipped.sum()),
                "units_neutral": int(table.neutral.sum()),
                "ratio_median": float(table.loc[~table.neutral, "ratio"].median()),
            }
            print(f"[{code}/{split}] {record[f'{code}/{split}']}", flush=True)

    with open(out / "gwa_record.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    write_manifest(out, build_manifest(extra={
        "run_mode": "pinn-gwa-ratio", "argv": sys.argv[1:], "cache": args.cache,
        "radius_km": RADIUS_KM, "ratio_bounds": list(RATIO_BOUNDS), "rasters": rasters,
        "record": record,
    }))


if __name__ == "__main__":
    main()
