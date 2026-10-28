"""Rebuild Norway's country-level observations from NVE's metered farm output and register.

The ENTSO-E files on disk do not measure one fleet. Their generation sums the
bidding zones ENTSO-E reports, which in 2015-21 come to 0.87 to 0.99 of the
official wind production (Statistics Norway, table 12824) each year, and in
October 2015-18 they double the clock-change hour. Their capacity is ENTSO-E's
register, 26% below NVE's fleet in 2015 and 32% above it in 2020. In 2023 the
generation matches Statistics Norway month by month (0.974 to 1.006).

NVE publishes each farm's hourly output, metered by Statnett at the grid
connection (``vindprod2002-2021_kraftverk_utcplus1.xlsx``, in MWh, stamped
UTC+1 at the start of the hour), and its turbine register gives every turbine
group's size and its commissioning and decommissioning dates. The farms' sum
equals Statistics Norway's monthly wind production in every month of 2015-21.
NVE's ``timeseries_utc_kwh_2022_all.zip`` is not used: its output is modelled
back to 2003 for every farm, including farms built in 2021.

For 2015-21, each farm counts in a calendar month, both its output and its
capacity, only when it has a metered value in every hour, its registered
capacity is above zero and the same all month, and its output never exceeds
1.05 times that capacity. The register dates turbines when they are formally
commissioned, often months after they start producing, so without the last two
conditions a farm being built shows output against capacity it has not been
given. The rule keeps 91 to 99.8% of each year's metered energy.

The 2023 test year keeps ENTSO-E's generation and divides it by the register's
whole fleet: NVE's metered file ends in 2021. Statistics Norway's monthly
totals, which NVE's farms sum to in 2015-21 and ENTSO-E's generation matches
in 2023, tie the two sources to one measure.

The files keep their names and the ENTSO-E layout: one row per hour, a UTC
calendar year per year, stamped in Oslo time. Nothing is overwritten unless
``--replace`` is given, and then the old file is kept beside it with the
suffix ``.entsoe.bak.csv``.

Usage:
    PYTHONPATH=src python scripts/region_tools/build_no_obs_from_nve.py --out <dir>
    PYTHONPATH=src python scripts/region_tools/build_no_obs_from_nve.py --replace
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


from pyvwf.config import PyVWFPaths
from pyvwf.loaders.country_obs_checks import check_country_cf

TZ = "Europe/Oslo"
BACKUP_SUFFIX = ".entsoe.bak.csv"
#: Output above this multiple of a farm's registered capacity means turbines
#: the register has not dated yet.
OVERSHOOT = 1.05

#: The splits the scorecard and study configs read, and where each comes from.
METERED_SPANS = [
    ("train", 2015, 2018),
    ("train", 2015, 2019),
    ("train", 2015, 2021),
    ("test", 2019, 2019),
]
ENTSOE_TEST_YEAR = 2023


def read_metered(path: Path) -> pd.DataFrame:
    """Hourly output per farm in MW, indexed in UTC, one column per register ID.

    Row 1 of the sheet names each farm, row 2 gives its ``VindkraftAnleggId``
    and row 3 is a header; each value is the MWh of the hour starting at its
    UTC+1 stamp. A farm's column is empty before it first produced.
    """
    raw = pd.read_excel(path, header=None)
    out = raw.iloc[3:, 1:].astype(float)
    out.index = pd.DatetimeIndex(raw.iloc[3:, 0]).tz_localize("Etc/GMT-1").tz_convert("UTC")
    out.columns = raw.iloc[1, 1:].astype(int).to_numpy()
    return out


def register_fleet(path: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    """MW registered per farm at each hour, from the turbine groups' dates.

    A group counts from its commissioning day (``DatoIdriftsatt``) until its
    decommissioning day (``DatoUtavdrift``), each read as the UTC day.
    """
    plants = json.loads(path.read_text(encoding="utf-8"))
    day = index.normalize()
    fleet = pd.DataFrame(0.0, index=index, columns=[p["VindkraftAnleggId"] for p in plants])
    for plant in plants:
        for group in plant["Turbiner"]:
            on = pd.Timestamp(group["DatoIdriftsatt"], tz="UTC")
            live = day >= on
            if group["DatoUtavdrift"]:
                live &= day < pd.Timestamp(group["DatoUtavdrift"], tz="UTC")
            mw = group["AntallTurbiner"] * group["TurbinStorrelse_kW"] / 1e3
            fleet.loc[live, plant["VindkraftAnleggId"]] += mw
    return fleet


def national(metered: pd.DataFrame, fleet: pd.DataFrame) -> pd.DataFrame:
    """National generation over the capacity of the farms that count each month."""
    fleet = fleet[metered.columns]
    month = metered.index.tz_convert(None).to_period("M")
    counts = (
        metered.notna().groupby(month).all()
        & (fleet > 0).groupby(month).all()
        & (fleet.groupby(month).max() == fleet.groupby(month).min())
        & ~(metered > OVERSHOOT * fleet).groupby(month).any()
    )
    mask = counts.reindex(month).set_axis(metered.index)
    out = pd.DataFrame(
        {
            "generation_mw": metered.where(mask).sum(axis=1),
            "capacity_mw": fleet.where(mask).sum(axis=1),
        }
    )
    out["capacity_factor"] = out["generation_mw"] / out["capacity_mw"]
    return out


def split(series: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    """The UTC calendar years first to last, stamped in Oslo time."""
    years = series.index.year
    out = series[(years >= first) & (years <= last)].copy()
    out.index = out.index.tz_convert(TZ)
    out.index.name = None
    return out


def write(frame: pd.DataFrame, path: Path, *, replace: bool) -> None:
    """Write one split, keeping any file it replaces as a backup."""
    if path.exists():
        if not replace:
            raise FileExistsError(f"{path} exists; pass --replace to keep it as a backup")
        backup = path.with_suffix(BACKUP_SUFFIX)
        if not backup.exists():
            path.rename(backup)
            print(f"  kept the old file as {backup.name}")
    frame.to_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    obs_dir = PyVWFPaths.COUNTRY_LEVEL_DATA / "observations" / "no"
    raw_dir = PyVWFPaths.INPUT_ROOT / "raw" / "nve"
    parser.add_argument("--obs", type=Path, default=obs_dir, help="directory of the no_*.csv files")
    parser.add_argument("--out", type=Path, help="where to write (default: --obs)")
    parser.add_argument(
        "--metered",
        type=Path,
        default=raw_dir / "vindprod2002-2021_kraftverk_utcplus1.xlsx",
        help="NVE's metered output per farm "
        "(https://www.nve.no/media/13399/vindprod2002-2021_kraftverk_utcplus1.xlsx)",
    )
    parser.add_argument(
        "--register",
        type=Path,
        default=raw_dir / "api_GetWindPowerPlants_2026-09-18.json",
        help="NVE's turbine register (https://api.nve.no/web/WindPowerplant/GetWindPowerPlants)",
    )
    parser.add_argument("--replace", action="store_true", help="replace existing files")
    args = parser.parse_args()
    out_dir = args.out or args.obs
    out_dir.mkdir(parents=True, exist_ok=True)

    def target(stem: str) -> Path:
        # Norway's files carry the generator's ``_aggregated`` suffix; keep
        # whichever name the observation source would read.
        plain = args.obs / f"{stem}.csv"
        return plain if plain.exists() else args.obs / f"{stem}_aggregated.csv"

    def save(frame: pd.DataFrame, source: Path) -> None:
        result = check_country_cf(frame, source.name, warn=False)
        energy = frame["generation_mw"].groupby(frame.index.tz_convert("UTC").year).sum() / 1e3
        print(
            f"{source.name}: {len(frame)} hours, mean CF {result.mean_cf:.3f}, GWh {energy.round(0).to_dict()}"
        )
        print(f"  gates: {'pass' if result.ok else '; '.join(result.issues)}")
        if out_dir == args.obs:
            write(frame, source, replace=args.replace)
        else:
            frame.to_csv(out_dir / source.name)

    metered = read_metered(args.metered)
    series = national(metered, register_fleet(args.register, metered.index))
    for kind, first, last in METERED_SPANS:
        stem = f"no_test_{first}" if kind == "test" else f"no_train_{first}_{last}"
        save(split(series, first, last), target(stem))

    # The 2023 generation is ENTSO-E's; read it from the backup once one
    # exists, so a second run reproduces the first.
    test = target(f"no_test_{ENTSOE_TEST_YEAR}")
    backup = test.with_suffix(BACKUP_SUFFIX)
    frame = pd.read_csv(backup if backup.exists() else test, index_col=0)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True)).tz_convert(TZ)
    frame["capacity_mw"] = (
        register_fleet(args.register, frame.index.tz_convert("UTC")).sum(axis=1).to_numpy()
    )
    frame["capacity_factor"] = frame["generation_mw"] / frame["capacity_mw"]
    save(frame, test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
