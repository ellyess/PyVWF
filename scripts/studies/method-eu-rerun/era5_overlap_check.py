"""G0 of the European re-run: do the old and new ERA5 files agree? Read-only.

The eleven European scorecard rows move from ``era5/EU`` to
``era5/EU_2026-09``, which changes two things at once: the roughness treatment,
because the new files carry no roughness field, and the loaded extent, because
the new box reaches further south and east. The re-run plan
(``docs/findings/method-eu-rerun-prereg.md``) separates those two. Neither
separation survives if the wind components themselves changed between the two
downloads, because then every comparison measures a data change as well.

So this is a gate, not a diagnostic. Its three outcomes are registered in that
plan and are not chosen once the numbers are visible: the values agree and the
comparisons measure the treatment; they differ with a bounded cause and every
figure travels with that bound; they differ with no cause established and the
treatment claim is withdrawn for every row.

**Fixed before running, so the sample cannot be chosen to suit the answer:**

- three years, the first, middle and last of the window: 2015, 2019, 2023;
- four months of each, one per season: January, April, July, October;
- every hour of those months, and every eighth grid cell in each direction,
  which is one cell each two degrees across the overlap;
- the overlap is the old files' own extent, since the new box contains it;
- all four wind components, ``u10``, ``v10``, ``u100`` and ``v100``. The
  roughness is not compared: the new files carry none, which is the point.

Equality is reported three ways, because they mean different things. Bit
identity says the archive did not change. Equality within float32 round-trip
says any difference is storage, not data. Anything larger is a data change and
needs a cause before the re-run proceeds.

Usage, from the repository root:

    PYTHONPATH=src python scripts/studies/method-eu-rerun/era5_overlap_check.py <out_dir>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from vwf.cli.common import make_parser
import xarray as xr

OLD_DIR = Path("input/era5/EU")
NEW_DIR = Path("input/era5/EU_2026-09")
YEARS = (2015, 2019, 2023)
MONTHS = (1, 4, 7, 10)
STRIDE = 8
VARIABLES = ("u10", "v10", "u100", "v100")
# float32 carries about seven significant decimal digits, so a value of order
# 10 m/s round-trips to within about 1e-5. A difference under this is storage;
# anything above it is the archive.
FLOAT32_TOLERANCE = 1e-5


def _standardise(ds: xr.Dataset) -> xr.Dataset:
    renames = {"longitude": "lon", "latitude": "lat", "valid_time": "time"}
    return ds.rename({k: v for k, v in renames.items() if k in ds.coords or k in ds.dims})


def _old_year(year: int, old_dir: Path = OLD_DIR) -> xr.Dataset:
    return _standardise(xr.open_dataset(Path(old_dir) / f"era5_combined_{year}_EU.nc"))


def _new_month(year: int, month: int, new_dir: Path = NEW_DIR) -> xr.Dataset:
    return _standardise(xr.open_dataset(Path(new_dir) / f"era5_eu_2026-09_{year}_{month:02d}.nc"))


def compare_month(year: int, month: int, old_dir: Path = OLD_DIR, new_dir: Path = NEW_DIR) -> dict:
    """One month of one year, on the sampled cells of the overlap."""
    old = _old_year(year, old_dir)
    new = _new_month(year, month, new_dir)
    # The overlap is the old extent; sample it the same way on both sides by
    # selecting the old coordinates after the stride, so a grid registration
    # difference shows up as a missing label rather than as a silent shift.
    lats = old["lat"].values[::STRIDE]
    lons = old["lon"].values[::STRIDE]
    times = old["time"].values[old["time"].dt.month.values == month]
    o = old.sel(lat=lats, lon=lons).sel(time=times)
    n = new.sel(lat=lats, lon=lons).sel(time=times)

    row = {
        "year": year,
        "month": month,
        "cells": len(lats) * len(lons),
        "hours": len(times),
        "coords_match": bool(
            np.array_equal(o["lat"].values, n["lat"].values)
            and np.array_equal(o["lon"].values, n["lon"].values)
            and np.array_equal(o["time"].values, n["time"].values)
        ),
    }
    for name in VARIABLES:
        a = np.asarray(o[name].values, dtype="float64")
        b = np.asarray(n[name].values, dtype="float64")
        diff = np.abs(a - b)
        row[f"{name}_max_abs_diff"] = float(diff.max())
        row[f"{name}_identical"] = bool((diff == 0).all())
    old.close()
    new.close()
    return row


def main(out_dir: str, old_dir: Path = OLD_DIR, new_dir: Path = NEW_DIR) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in YEARS:
        for month in MONTHS:
            rows.append(compare_month(year, month, old_dir, new_dir))
            r = rows[-1]
            worst = max(r[f"{v}_max_abs_diff"] for v in VARIABLES)
            print(
                f"{year}-{month:02d}: {r['cells']} cells x {r['hours']} hours, "
                f"coords match {r['coords_match']}, worst |diff| {worst:.3e}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "era5_overlap_check.csv", index=False)

    identical = bool(frame[[f"{v}_identical" for v in VARIABLES]].all().all())
    worst = float(frame[[f"{v}_max_abs_diff" for v in VARIABLES]].max().max())
    coords = bool(frame["coords_match"].all())
    print(f"\ncoordinates match everywhere: {coords}")
    print(f"worst absolute difference across all samples: {worst:.3e}")
    if identical and coords:
        print("G0: the two downloads are BIT-IDENTICAL on the sampled overlap.")
    elif coords and worst <= FLOAT32_TOLERANCE:
        print(
            f"G0: equal within float32 round-trip ({FLOAT32_TOLERANCE:.0e}); "
            "the difference is storage, not data."
        )
    else:
        print(
            "G0: the downloads DIFFER. The plan's second or third branch applies: "
            "bound the cause, or withdraw the treatment claim for every row."
        )
        sys.exit(1)


def cli(argv: list[str] | None = None) -> None:
    """Parse the recorded command line, ``<out_dir>``, and run :func:`main`."""
    parser = make_parser(__doc__)
    parser.add_argument("out_dir", help="Directory for the outputs, under output/")
    parser.add_argument(
        "--old-dir",
        type=Path,
        default=OLD_DIR,
        help=f"The annual combined ERA5 files (default: {OLD_DIR})",
    )
    parser.add_argument(
        "--new-dir",
        type=Path,
        default=NEW_DIR,
        help=f"The 2026-09 monthly ERA5 files (default: {NEW_DIR})",
    )
    args = parser.parse_args(argv)
    main(args.out_dir, old_dir=args.old_dir, new_dir=args.new_dir)


if __name__ == "__main__":
    cli()
