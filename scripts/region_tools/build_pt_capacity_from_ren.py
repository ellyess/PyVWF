"""Give Portugal's country-level observations the wind fleet REN counts.

Portugal's generation from ENTSO-E is REN's: the hourly mean of REN's
15-minute wind series equals the stored value on sample hours in 2015 and
2023, onshore plus offshore. Its capacity is not. The training files carry
4,486 MW in every year from 2015 to 2019, a register ENTSO-E stopped updating,
while REN's fleet grew from 4,538 to 5,208 MW; and from 2020 they carry
ENTSO-E's onshore register while the generation includes WindFloat Atlantic's
offshore output.

REN's Data Hub gives the installed capacity by technology at each month's end
(``ElectricityInstalledPowerMonthly``, type ``WIND``), mainland Portugal, the
scope of the generation; it steps by 8 to 9 MW in December 2019, April 2020
and August 2020, as WindFloat Atlantic's three turbines were connected, so it
counts offshore too. This keeps each file's generation and divides it by that
fleet interpolated linearly through each month, from one month's end to the
next. REN publishes no reuse licence for the Data Hub ("All rights
reserved"), so the table is kept local, as ENTSO-E's data is.

Every national Portuguese file is rewritten. Nothing is overwritten unless
``--replace`` is given, and then the old file is kept beside it with the
suffix ``.entsoe-capacity.bak.csv``.

Usage:
    PYTHONPATH=src python scripts/region_tools/build_pt_capacity_from_ren.py --out <dir>
    PYTHONPATH=src python scripts/region_tools/build_pt_capacity_from_ren.py --replace
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pandas as pd


from pyvwf.config import PyVWFPaths
from pyvwf.datasets.country_obs_rebuild import recapacitate, write_keeping_backup
from pyvwf.loaders.country_obs_checks import check_country_cf

REN_MONTHLY = (
    "https://servicebus.ren.pt/datahubapi/electricity/ElectricityInstalledPowerMonthly"
    "?culture=en-US&year={year}&month={month}"
)
BACKUP_SUFFIX = ".entsoe-capacity.bak.csv"


def fetch_fleet(cache: Path, first: int, last: int) -> pd.Series:
    """Installed wind capacity at each month's end in MW, indexed by month.

    Read for December of ``first - 1`` to December of ``last``. The months are
    downloaded into ``cache`` if it is not there: one call of about 520 bytes
    each.
    """
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for month in pd.period_range(f"{first - 1}-12", f"{last}-12", freq="M"):
            url = REN_MONTHLY.format(year=month.year, month=month.month)
            with urllib.request.urlopen(url, timeout=600) as response:
                types = json.load(response)
            wind = next(t["monthly_Accumulation"] for t in types if t["type"] == "WIND")
            rows.append({"month": str(month), "wind_mw": wind})
        pd.DataFrame(rows).to_csv(cache, index=False)
        print(f"fetched {cache}")
    table = pd.read_csv(cache)
    return pd.Series(table["wind_mw"].to_numpy(), index=pd.PeriodIndex(table["month"], freq="M"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--obs",
        type=Path,
        default=PyVWFPaths.COUNTRY_LEVEL_DATA / "observations" / "pt",
        help="directory holding the pt_<split>_<years>.csv files to read",
    )
    parser.add_argument("--out", type=Path, help="where to write (default: --obs)")
    parser.add_argument(
        "--cache",
        type=Path,
        default=PyVWFPaths.INPUT_ROOT
        / "raw"
        / "ren"
        / "installed_power_monthly_wind_2014_2023.csv",
        help="where the REN month-end table is kept",
    )
    parser.add_argument("--replace", action="store_true", help="replace existing files")
    args = parser.parse_args()

    fleet = fetch_fleet(args.cache, 2015, 2023)
    out_dir = args.out or args.obs
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(p for p in args.obs.glob("pt_*.csv") if ".bak." not in p.name):
        # A backup holds the generation as fetched; read it once it exists, so a
        # second run reproduces the first instead of reading its own output.
        source = path.with_suffix(BACKUP_SUFFIX)
        frame = pd.read_csv(source if source.exists() else path, index_col=0)
        new = recapacitate(frame, fleet)
        new.index.name = frame.index.name
        report = check_country_cf(new, path.name, warn=False)
        print(f"{path.name}: {len(new)} hours, mean CF {report.mean_cf:.3f}")
        print(f"  gates: {'pass' if report.ok else '; '.join(report.issues)}")
        target = out_dir / path.name
        if target == path:
            write_keeping_backup(new, target, replace=args.replace, backup_suffix=BACKUP_SUFFIX)
        else:
            new.to_csv(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
