#!/usr/bin/env python
"""Pre-combine a region's monthly ERA5 files to yearly DAILY files.

Only the large boxes need this: a multi-year hourly load of a continent-sized
box (US, BR, and the NEM) is tens of GB in memory, but the pipeline only
consumes daily-mean wind speeds, so this reduces each year to a small daily
file. Small boxes (NZ, CL, AR) skip it — their config ``[era5] path`` points
straight at the raw ``era5/<tag>`` dir.

One script for every region: the input tag and the year span come from the
region TOML (this replaced three near-identical ``combine_era5_<code>_daily``
scripts). It reads ``era5/<file_tag>/era5_<code>_<YYYY>_<MM>.nc`` and writes
``era5/<file_tag>_daily/era5_<code>_daily_<YYYY>.nc`` — the ``_daily`` dir the
US/BR configs point ``[era5] path`` at.

    python scripts/era5/combine.py --region br            # all years, from config
    python scripts/era5/combine.py --region us --years 2022

Order matters and is preserved from the legacy pipeline: wind SPEED is computed
from HOURLY components first and only then daily-averaged (mean-of-speed, not
speed-of-mean-components), and roughness is derived from the hourly 10 m/100 m
shear exactly as ``vwf.datasets.era5.prep_era5`` does; prep_era5 detects the
precomputed ``wnd100m``/``roughness`` and skips recomputation.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from vwf.harness.regions import load_region  # noqa: E402

Z0_CLIP = (1e-6, 2.0)
CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "regions"


def region_spec(code: str):
    path = CONFIG_DIR / f"{code.lower()}.toml"
    if not path.is_file():
        sys.exit(f"no region config at {path} — is {code!r} a shipped region?")
    return load_region(path)


def combine_year(in_dir: Path, out_dir: Path, code: str, year: int) -> Path:
    target = out_dir / f"era5_{code}_daily_{year}.nc"
    if target.is_file():
        print(f"{year}: exists, skipping")
        return target
    files = sorted(in_dir.glob(f"era5_{code}_{year}_*.nc"))
    if len(files) != 12:
        raise FileNotFoundError(f"{year}: expected 12 monthly files, found {len(files)}")
    days = []
    for f in files:
        # Recent CDS ERA5 files carry an object-dtype ``expver`` variable
        # (ERA5/ERA5T mixing for near-real-time months); decoding it trips
        # xarray in some environments. It is unused here, so drop at open.
        ds = xr.open_dataset(f, drop_variables=["expver"])
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
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True, help="Region code (e.g. br, us, au)")
    ap.add_argument("--years", type=int, nargs="+", default=None,
                    help="Override the year span (default: train[0]..test[-1])")
    args = ap.parse_args()

    spec = region_spec(args.region)
    # Filenames key on file_tag, not the region code: AU-NEM writes era5_au_*.
    tag = spec.file_tag.lower()
    years = args.years or list(range(spec.train_years[0], spec.test_years[-1] + 1))
    root = Path(os.environ.get("PYVWF_INPUT", "input"))
    in_dir = root / "era5" / spec.file_tag
    out_dir = root / "era5" / f"{spec.file_tag}_daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        combine_year(in_dir, out_dir, tag, year)


if __name__ == "__main__":
    main()
