"""Rebuild France's country-level observations from RTE, every wind farm in every year.

The ENTSO-E files on disk do not measure one fleet. Matched hour by hour
against ENTSO-E's per-type series (audited 2026-10-23), the 2023 test file is
onshore generation alone, while the 2023 grid simulates Saint-Nazaire; and
ENTSO-E holds no French offshore before June 2023 at all. Their capacity is
ENTSO-E's installed-capacity register, which follows the fleet to 2017 and
then falls 1.0 to 2.9 GW behind it in 2018 and 2019, so those years' mean
capacity factors are overstated by about 8% and 12%.

RTE is the French TSO and the source ENTSO-E republishes for France. It
publishes its definitive eCO2mix national series on ODRE (Open Data Réseaux
Énergies, dataset ``eco2mix-national-cons-def``) under Licence Ouverte v2.0,
with national wind output (``eolien``, onshore and offshore) every half hour
from the metering of the transmission and distribution networks; its
``parc-prod-par-filiere`` dataset, under the same licence, gives the wind fleet
connected on 31 December of each year. Both cover the same fleet. This takes
the half-hourly output, averages each hour, and divides by the capacity
interpolated linearly through the year from one 31 December to the next:
France's fleet grew 6 to 15% a year, so a capacity held at the start of the
year would overstate each year's capacity factor by about half its growth.
The files keep the ENTSO-E layout: one row per hour, a UTC calendar year per
year, stamped in Paris time. RTE's own annual eCO2mix files are not read:
their terms forbid redistributing them.

RTE records each day in Paris wall-clock time with 48 half-hours, so the clock
changes cannot be placed, and ODRE's UTC stamps inherit that: in March it gives
01:00 and 01:30 UTC twice, and in October it puts the once-written 02:00 and
02:30 at 01:00 and 01:30 UTC and leaves 00:00 and 00:30 empty, an assignment
the source cannot support. So each stamp is taken back to Paris time, a stamp
given twice is dropped, and so is one that occurs twice in Paris time. Three
UTC hours a year go empty: 01:00 in March, and 00:00 and 01:00 in October. An
hour counts only when both its half-hours are present.

Nothing is overwritten unless ``--replace`` is given, and then the old file is
kept beside it with the suffix ``.entsoe.bak.csv``.

Usage:
    PYTHONPATH=src python scripts/region_tools/build_fr_obs_from_rte.py --out <dir>
    PYTHONPATH=src python scripts/region_tools/build_fr_obs_from_rte.py --replace
"""

from __future__ import annotations

import argparse
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


from pyvwf.config import PyVWFPaths
from pyvwf.datasets.country_obs_rebuild import (
    interpolated_capacity,
    split_utc_years,
    write_keeping_backup,
)
from pyvwf.loaders.country_obs_checks import check_country_cf

ODRE = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/{dataset}/exports/csv"
TZ = "Europe/Paris"
BACKUP_SUFFIX = ".entsoe.bak.csv"

#: The splits the scorecard and study configs read.
SPANS = [
    ("train", 2015, 2018),
    ("train", 2015, 2019),
    ("train", 2015, 2021),
    ("test", 2019, 2019),
    ("test", 2023, 2023),
]


def download(dataset: str, query: dict[str, str], path: Path) -> None:
    """Export one ODRE dataset into ``path`` as semicolon-separated CSV, unless it is there."""
    if path.exists():
        return
    url = f"{ODRE.format(dataset=dataset)}?{urllib.parse.urlencode({**query, 'delimiter': ';'})}"
    with urllib.request.urlopen(url, timeout=600) as response:
        path.write_bytes(response.read())
    print(f"fetched {path}")


def fetch(cache: Path, first: int, last: int) -> tuple[pd.DataFrame, pd.Series]:
    """RTE's half-hourly national wind output and its 31 December wind fleet.

    Output is read for the UTC calendar years ``first`` to ``last``. Missing
    exports are downloaded into ``cache``.

    Returns:
        The output as ``date_heure`` (UTC) and ``eolien`` rows, and the fleet
        in MW indexed by year.
    """
    cache.mkdir(parents=True, exist_ok=True)
    output_path = cache / f"odre_eco2mix_national_wind_{first}_{last}.csv"
    where = (
        f'date_heure >= "{first}-01-01T00:00:00Z" and date_heure < "{last + 1}-01-01T00:00:00Z"'
        " and eolien is not null"
    )
    query = {"select": "date_heure,eolien", "where": where, "order_by": "date_heure"}
    download("eco2mix-national-cons-def", {**query, "timezone": "UTC"}, output_path)
    fleet_path = cache / "parc_prod_par_filiere.csv"
    download(
        "parc-prod-par-filiere", {"select": "annee,parc_eolien", "order_by": "annee"}, fleet_path
    )
    output = pd.read_csv(output_path, sep=";", encoding="utf-8-sig")
    fleet = pd.read_csv(fleet_path, sep=";", encoding="utf-8-sig")
    return output, fleet.set_index("annee")["parc_eolien"]


def hourly_output(raw: pd.DataFrame) -> pd.Series:
    """Hourly national wind output in MW, indexed in UTC.

    An hour counts only when both its half-hours are present, so a missing
    half-hour shows as a missing hour rather than as output that halved.

    Args:
        raw: ``date_heure`` and ``eolien`` as ODRE exports them, in UTC.
    """
    utc = pd.DatetimeIndex(pd.to_datetime(raw["date_heure"], utc=True))
    wall = pd.Series(
        raw["eolien"].to_numpy(dtype=float), index=utc.tz_convert(TZ).tz_localize(None)
    )
    wall = wall[~wall.index.duplicated(keep=False)]
    local = pd.DatetimeIndex(wall.index).tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    series = pd.Series(wall.to_numpy(), index=local.tz_convert("UTC"))
    hourly = series[series.index.notna()].resample("1h")
    return hourly.mean()[hourly.count() == 2]


def national(raw: pd.DataFrame, fleet: pd.Series) -> pd.DataFrame:
    """One hourly national series: generation, capacity, and their ratio."""
    generation = hourly_output(raw)
    out = pd.DataFrame(
        {"generation_mw": generation, "capacity_mw": interpolated_capacity(generation.index, fleet)}
    )
    out["capacity_factor"] = out["generation_mw"] / out["capacity_mw"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PyVWFPaths.COUNTRY_LEVEL_DATA / "observations" / "fr",
        help="directory for the fr_<split>_<years>.csv files",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=PyVWFPaths.INPUT_ROOT / "raw" / "rte",
        help="where the ODRE exports are kept",
    )
    parser.add_argument("--replace", action="store_true", help="replace existing files")
    args = parser.parse_args()

    first = min(s[1] for s in SPANS)
    last = max(s[2] for s in SPANS)
    series = national(*fetch(args.cache, first, last))
    args.out.mkdir(parents=True, exist_ok=True)
    for kind, start, end in SPANS:
        name = f"fr_test_{start}.csv" if kind == "test" else f"fr_train_{start}_{end}.csv"
        frame = split_utc_years(series, start, end, TZ)
        report = check_country_cf(frame, name, warn=False)
        utc_year = frame.index.tz_convert("UTC").year
        energy = (frame["generation_mw"].groupby(utc_year).sum() / 1e3).round(0)
        print(f"{name}: {len(frame)} hours, mean CF {report.mean_cf:.3f}, GWh {energy.to_dict()}")
        print(f"  gates: {'pass' if report.ok else '; '.join(report.issues)}")
        write_keeping_backup(
            frame, args.out / name, replace=args.replace, backup_suffix=BACKUP_SUFFIX
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
