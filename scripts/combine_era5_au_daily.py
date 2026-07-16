#!/usr/bin/env python
"""Pre-combine the monthly ERA5-AU files to yearly DAILY files.

A 4-year hourly load of the full NEM box is ~8-16 GB in memory; the pipeline
only consumes daily-mean wind speeds, so this reduces each year to ~29 MB.
Order matters and is preserved from the legacy pipeline: wind SPEED is
computed from HOURLY components first and only then daily-averaged
(mean-of-speed, not speed-of-mean-components), and roughness is derived from
the hourly 10 m/100 m shear exactly as vwf.datasets.era5.prep_era5 does.
prep_era5 detects the precomputed wnd100m/roughness and skips recomputation.

    python scripts/combine_era5_au_daily.py            # 2020-2023
    python scripts/combine_era5_au_daily.py --years 2023
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr

Z0_CLIP = (1e-6, 2.0)


def combine_year(in_dir: Path, out_dir: Path, year: int) -> Path:
    target = out_dir / f"era5_au_daily_{year}.nc"
    if target.is_file():
        print(f"{year}: exists, skipping")
        return target
    files = sorted(in_dir.glob(f"era5_au_{year}_*.nc"))
    if len(files) != 12:
        raise FileNotFoundError(f"{year}: expected 12 monthly files, found {len(files)}")
    days = []
    for f in files:
        ds = xr.open_dataset(f)
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        if tname != "time":
            ds = ds.rename({tname: "time"})
        for old, new in [("longitude", "lon"), ("latitude", "lat")]:
            if old in ds.coords:
                ds = ds.rename({old: new})
        w100 = np.hypot(ds["u100"], ds["v100"])
        w10 = np.hypot(ds["u10"], ds["v10"]).clip(min=1e-4)
        w100c = w100.clip(min=1e-4)
        num = w100c * np.log(10) - w10 * np.log(100)
        den = (w100c - w10).where(lambda x: np.abs(x) > 1e-4)
        z0log = (num / den).where(lambda x: x < 0)
        z0log = z0log.bfill("time").clip(min=np.log(Z0_CLIP[0]), max=np.log(Z0_CLIP[1]))
        rough = np.exp(z0log).clip(min=Z0_CLIP[0])
        daily = xr.Dataset({"wnd100m": w100, "roughness": rough}).resample(time="1D").mean()
        days.append(daily.astype("float32"))
        ds.close()
    combined = xr.concat(days, dim="time").sortby("time")
    enc = {v: {"zlib": True, "complevel": 4} for v in combined.data_vars}
    combined.to_netcdf(target, encoding=enc)
    print(f"{year}: {combined.sizes['time']} days -> {target.name} "
          f"({target.stat().st_size / 1e6:.0f} MB)")
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", default="input/era5/AU")
    ap.add_argument("--out-dir", default="input/era5/AU_daily")
    ap.add_argument("--years", type=int, nargs="+", default=[2020, 2021, 2022, 2023])
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in args.years:
        combine_year(Path(args.in_dir), out_dir, year)


if __name__ == "__main__":
    main()
