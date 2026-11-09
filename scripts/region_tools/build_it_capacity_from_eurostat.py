"""Give Italy's country-level observations the wind fleet Eurostat counts.

Italy's generation from ENTSO-E is sound: each year's energy from 2015 to 2023
is 0.97 to 0.99 of Eurostat's gross wind production (checked 2026-11-09). Its
capacity is not. ENTSO-E's register for Italy runs about two years behind the
fleet: the value it gives for a year is within 0.1 to 3.5% of the fleet
Eurostat counts at the end of the year two before (11,234 MW for 2023 against
11,254 MW at the end of 2021). Against the fleet installed during the year, it
is 5 to 8% low in every year, so each capacity factor was 3 to 7% too high.

Eurostat's capacity table (``nrg_inf_epcrw``, wind ``RA300``, net maximum
capacity ``CAP_NET_ELC``) gives the fleet installed on 31 December of each
year, onshore and offshore. Its metadata page no longer resolves, so the date
was checked against two fleets counted on 31 December: Eurostat gives
Energimyndigheten's Swedish fleet to the megawatt in nine of the ten years
2014-23, and RTE's French fleet within 2%. This keeps each file's generation
and divides it by that fleet interpolated linearly through the year, as
Sweden's and France's builders do.

Eurostat revises its tables, so the response is kept in ``--cache`` and read
from there on every later run. Eurostat allows reuse with the source
acknowledged.

Every national Italian file is rewritten. Nothing is overwritten unless
``--replace`` is given, and then the old file is kept beside it with the
suffix ``.entsoe-capacity.bak.csv``.

Usage:
    python scripts/region_tools/build_it_capacity_from_eurostat.py --out <dir>
    python scripts/region_tools/build_it_capacity_from_eurostat.py --replace
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

NRG_INF_EPCRW = (
    "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nrg_inf_epcrw"
    "?geo=IT&siec=RA300&unit=MW&plant_tec=CAP_NET_ELC&format=JSON&lang=EN"
)
BACKUP_SUFFIX = ".entsoe-capacity.bak.csv"


def fetch_fleet(cache: Path) -> pd.Series:
    """Installed wind capacity on 31 December in MW, indexed by year.

    The table is downloaded into ``cache`` if it is not there: one JSON-stat
    response of about 2 kB. The query fixes every dimension but time, so each
    value's position is its year's position in the time index.
    """
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(NRG_INF_EPCRW, timeout=600) as response:
            cache.write_bytes(response.read())
        print(f"fetched {cache}")
    table = json.loads(cache.read_text())
    years = {
        pos: int(year) for year, pos in table["dimension"]["time"]["category"]["index"].items()
    }
    return pd.Series(
        {years[int(pos)]: float(mw) for pos, mw in table["value"].items()}
    ).sort_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--obs",
        type=Path,
        default=PyVWFPaths.COUNTRY_LEVEL_DATA / "observations" / "it",
        help="directory holding the it_<split>_<years>.csv files to read",
    )
    parser.add_argument("--out", type=Path, help="where to write (default: --obs)")
    parser.add_argument(
        "--cache",
        type=Path,
        default=PyVWFPaths.INPUT_ROOT / "raw" / "eurostat" / "nrg_inf_epcrw_IT.json",
        help="where the Eurostat response is kept",
    )
    parser.add_argument("--replace", action="store_true", help="replace existing files")
    args = parser.parse_args()

    fleet = fetch_fleet(args.cache)
    out_dir = args.out or args.obs
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(p for p in args.obs.glob("it_*.csv") if ".bak." not in p.name):
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
