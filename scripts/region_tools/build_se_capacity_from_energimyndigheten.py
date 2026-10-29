"""Give Sweden's country-level observations the fleet Energimyndigheten counts.

Sweden's generation from ENTSO-E is sound: each year's energy is 0.976 to 1.007
of Energimyndigheten's official wind production (audited 2026-10-25). Its
capacity is not. The files built from the four bidding zones carry 8,354 MW in
every year from 2015 to 2019, the sum of four zonal capacities an old fetcher
derived as peak generation over 0.9 when ENTSO-E returned none
(``docs/results.md#the-country-level-estimator``). The files built from the national
series carry ENTSO-E's register, which counts onshore wind at the start of each
year: it leaves out the year's growth and about 200 MW offshore that the
generation includes, and its 2015 value, 3,500 MW, puts that year's capacity
factor at 0.54.

Energimyndigheten's official wind statistics (``Vindkraftsstatistik``, table
EN0105_1) give the fleet installed on 31 December of each year, onshore and
offshore, from the electricity-certificate register. This keeps each file's
generation and divides it by that fleet interpolated linearly through the year,
as France's builder does with RTE's: Sweden's fleet grew 3 to 20% a year, so a
capacity held at the start of the year overstates each year's capacity factor
by about half its growth. The register keeps a decommissioned turbine for up to
three years, so the fleet is slightly overstated.

Every national Swedish file is rewritten, both the zone sums (``_aggregated``)
and the national ones, since the observation source resolves either depending
on the split. Nothing is overwritten unless ``--replace`` is given, and then the
old file is kept beside it with the suffix ``.entsoe-capacity.bak.csv``.

Usage:
    python scripts/region_tools/build_se_capacity_from_energimyndigheten.py --out <dir>
    python scripts/region_tools/build_se_capacity_from_energimyndigheten.py --replace
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

EN0105_1 = (
    "https://pxexternal.energimyndigheten.se/api/v1/sv/Energimyndighetens_statistikdatabas/"
    "Officiell_energistatistik/Vindkraftsstatistik/EN0105_1.px"
)
BACKUP_SUFFIX = ".entsoe-capacity.bak.csv"


def fetch_fleet(cache: Path) -> pd.Series:
    """Installed wind capacity on 31 December in MW, indexed by year.

    The table is downloaded into ``cache`` if it is not there: one CSV of about
    1.5 kB, in Latin-1, with ``.`` for a missing value.
    """
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"query": [], "response": {"format": "csv"}}).encode()
        request = urllib.request.Request(
            EN0105_1, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            cache.write_bytes(response.read())
        print(f"fetched {cache}")
    table = pd.read_csv(cache, encoding="latin-1", na_values=".")
    capacity = next(c for c in table.columns if "effekt" in c)
    return table.set_index(table.columns[0])[capacity]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--obs",
        type=Path,
        default=PyVWFPaths.COUNTRY_LEVEL_DATA / "observations" / "se",
        help="directory holding the se_<split>_<years>[_aggregated].csv files to read",
    )
    parser.add_argument("--out", type=Path, help="where to write (default: --obs)")
    parser.add_argument(
        "--cache",
        type=Path,
        default=PyVWFPaths.INPUT_ROOT / "raw" / "energimyndigheten" / "EN0105_1.csv",
        help="where the Energimyndigheten table is kept",
    )
    parser.add_argument("--replace", action="store_true", help="replace existing files")
    args = parser.parse_args()

    fleet = fetch_fleet(args.cache)
    out_dir = args.out or args.obs
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(p for p in args.obs.glob("se_*.csv") if ".bak." not in p.name):
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
