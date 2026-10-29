"""Rebuild Belgium's country-level observations from Elia, offshore and onshore together.

The ENTSO-E files on disk do not measure one thing. Matched hour by hour against
Elia's per-type series (Open Data dataset ``ods031``, audited 2026-10-22), the
2015-19 and 2023 rows are onshore generation only (r 0.998 to 1.000, ratio of
means 1.00), while 2020-21 are onshore plus offshore. The model simulates the
whole fleet, offshore points included since ``fc8c7f5``, so the correction was
fitted to turn whole-fleet output into onshore output: the reported row's
offshore and coastal cluster carries wind scalars of 0.48 to 0.52.

Elia is the Belgian TSO and the source ENTSO-E republishes for Belgium.
``ods031`` gives measured and upscaled output and the monitored capacity behind
it, every 15 minutes, for each segment. This sums the segments, averages each
hour and divides by the summed monitored capacity, so numerator and denominator
cover the same fleet in every hour. The files keep the ENTSO-E layout: one row
per hour, a UTC calendar year per year, stamped in Brussels time.

Nothing is overwritten unless ``--replace`` is given, and then the old file is
kept beside it with the suffix ``.entsoe-mixed.bak.csv``.

Usage:
    python scripts/region_tools/build_be_obs_from_elia.py --out <dir>
    python scripts/region_tools/build_be_obs_from_elia.py --replace
"""

from __future__ import annotations

import argparse
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


from pyvwf.config import PyVWFPaths
from pyvwf.datasets.country_obs_rebuild import split_utc_years, write_keeping_backup
from pyvwf.loaders.country_obs_checks import check_country_cf

ELIA_EXPORT = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods031/exports/csv"
TZ = "Europe/Brussels"
BACKUP_SUFFIX = ".entsoe-mixed.bak.csv"

#: The splits the scorecard and study configs read, plus 2024 as a test year
#: that no country row has used.
SPANS = [
    ("train", 2015, 2018),
    ("train", 2015, 2019),
    ("train", 2015, 2021),
    ("test", 2019, 2019),
    ("test", 2023, 2023),
    ("test", 2024, 2024),
]


def fetch(cache: Path, first: int, last: int) -> pd.DataFrame:
    """Elia's 15-minute series summed by segment type, cached as CSV."""
    if not cache.exists():
        query = {
            "select": "datetime,offshoreonshore,sum(measured) as measured,"
            "sum(monitoredcapacity) as capacity",
            "group_by": "datetime,offshoreonshore",
            "where": f"datetime >= '{first - 1}-12-31T23:00:00Z' "
            f"and datetime < '{last + 1}-01-01T00:00:00Z'",
            "timezone": "UTC",
        }
        cache.parent.mkdir(parents=True, exist_ok=True)
        url = f"{ELIA_EXPORT}?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(url, timeout=600) as response:
            cache.write_bytes(response.read())
        print(f"fetched {cache}")
    return pd.read_csv(cache, sep=";", encoding="utf-8-sig", parse_dates=["datetime"])


def hourly_national(raw: pd.DataFrame) -> pd.DataFrame:
    """One hourly national series: generation, monitored capacity, and their ratio.

    An hour counts only when both segments reported in it, so a missing segment
    shows as a missing hour rather than as a fleet that shrank. A 15-minute step
    with no monitored capacity is a gap, not a fleet of zero: Elia reports 108
    such steps in 2015-24, all with generation above zero, which would otherwise
    become infinite capacity factors.

    Args:
        raw: rows of ``datetime`` (UTC), ``offshoreonshore``, ``measured`` and
            ``capacity`` in MW, one per segment type and 15-minute step.

    Returns:
        Hourly frame indexed in UTC with ``generation_mw``, ``capacity_mw`` and
        ``capacity_factor``.
    """
    raw = raw[raw["capacity"] > 0]
    wide = raw.pivot_table(
        index="datetime", columns="offshoreonshore", values=["measured", "capacity"]
    )
    hourly = wide.resample("1h").mean()
    complete = hourly.notna().all(axis=1)
    out = pd.DataFrame(
        {
            "generation_mw": hourly["measured"].sum(axis=1),
            "capacity_mw": hourly["capacity"].sum(axis=1),
        }
    )[complete]
    out["capacity_factor"] = out["generation_mw"] / out["capacity_mw"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=PyVWFPaths.COUNTRY_LEVEL_DATA / "observations" / "be",
        help="directory for the be_<split>_<years>.csv files",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=PyVWFPaths.INPUT_ROOT / "raw" / "elia" / "ods031_by_type_2015_2024.csv",
        help="where the raw Elia export is kept",
    )
    parser.add_argument("--replace", action="store_true", help="replace existing files")
    args = parser.parse_args()

    first = min(s[1] for s in SPANS)
    last = max(s[2] for s in SPANS)
    series = hourly_national(fetch(args.cache, first, last))
    args.out.mkdir(parents=True, exist_ok=True)
    for kind, start, end in SPANS:
        name = f"be_test_{start}.csv" if kind == "test" else f"be_train_{start}_{end}.csv"
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
