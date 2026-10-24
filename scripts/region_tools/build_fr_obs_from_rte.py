"""Rebuild France's country-level observations from RTE, every wind farm in every year.

The ENTSO-E files on disk do not measure one fleet. Matched hour by hour
against ENTSO-E's per-type series (audited 2026-10-23), the 2023 test file is
onshore generation alone, while the 2023 grid simulates Saint-Nazaire; and
ENTSO-E holds no French offshore before June 2023 at all. Their capacity is
ENTSO-E's installed-capacity register, which follows the fleet to 2017 and
then falls 1.0 to 2.9 GW behind it in 2018 and 2019, so those years' mean
capacity factors are overstated by about 8% and 12%.

RTE is the French TSO and the source ENTSO-E republishes for France. Its annual
definitive eCO2mix files give national wind output (``Eolien``, onshore and
offshore) every half hour, from the metering of the transmission and
distribution networks; its ``parc-prod-par-filiere`` dataset gives the wind
fleet connected on 31 December of each year. Both cover the same fleet. This
takes the half-hourly output, averages each hour, and divides by the capacity
interpolated linearly through the year from one 31 December to the next:
France's fleet grew 6 to 15% a year, so a capacity held at the start of the
year would overstate each year's capacity factor by about half its growth.
The files keep the ENTSO-E layout: one row per hour, a UTC calendar year per
year, stamped in Paris time.

RTE stamps its rows in Paris wall-clock time but writes each day with 48
half-hours, so the clock changes cannot be placed. The 02:00 and 02:30 rows of
those two days are dropped: in March they do not exist, and in October they
occur twice but are written once, which leaves two UTC hours empty. An hour
counts only when both its half-hours are present.

Nothing is overwritten unless ``--replace`` is given, and then the old file is
kept beside it with the suffix ``.entsoe.bak.csv``.

Usage:
    PYTHONPATH=src python scripts/region_tools/build_fr_obs_from_rte.py --out <dir>
    PYTHONPATH=src python scripts/region_tools/build_fr_obs_from_rte.py --replace
"""

from __future__ import annotations

import argparse
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyvwf.config import PyVWFPaths  # noqa: E402
from pyvwf.loaders.country_obs_checks import check_country_cf  # noqa: E402

RTE_ANNUAL = (
    "https://eco2mix.rte-france.com/download/eco2mix/eCO2mix_RTE_Annuel-Definitif_{year}.zip"
)
RTE_FLEET = (
    "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/parc-prod-par-filiere/"
    "exports/csv"
)
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


def fetch(cache: Path, first: int, last: int) -> tuple[pd.DataFrame, pd.Series]:
    """RTE's half-hourly national wind output and its 31 December wind fleet.

    The annual files for ``first`` to ``last + 1`` are read, the last for the
    final UTC hour of ``last``, which falls on 1 January in Paris time.
    Missing files are downloaded into ``cache``.

    Returns:
        The output as ``Date``, ``Heures`` and ``Eolien`` rows, and the fleet
        in MW indexed by year.
    """
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(first, last + 2):
        path = cache / f"eCO2mix_RTE_Annuel-Definitif_{year}.zip"
        if not path.exists():
            with urllib.request.urlopen(RTE_ANNUAL.format(year=year), timeout=600) as response:
                path.write_bytes(response.read())
            print(f"fetched {path}")
        with zipfile.ZipFile(path) as archive, archive.open(archive.namelist()[0]) as handle:
            frames.append(
                pd.read_csv(handle, sep="\t", encoding="latin-1", dtype=str, index_col=False)
            )
    fleet_path = cache / "parc_prod_par_filiere.csv"
    if not fleet_path.exists():
        query = {"select": "annee,parc_eolien", "order_by": "annee", "delimiter": ";"}
        url = f"{RTE_FLEET}?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(url, timeout=600) as response:
            fleet_path.write_bytes(response.read())
        print(f"fetched {fleet_path}")
    fleet = pd.read_csv(fleet_path, sep=";", encoding="utf-8-sig")
    return pd.concat(frames)[["Date", "Heures", "Eolien"]], fleet.set_index("annee")["parc_eolien"]


def hourly_output(raw: pd.DataFrame) -> pd.Series:
    """Hourly national wind output in MW, indexed in UTC.

    Rows without a value (RTE's quarter-hour forecast rows, ``ND`` and the
    closing notice) are dropped, and an hour counts only when both its
    half-hours are present, so a missing half-hour shows as a missing hour
    rather than as output that halved.

    Args:
        raw: ``Date``, ``Heures`` and ``Eolien`` as RTE writes them, in Paris
            wall-clock time.
    """
    output = pd.to_numeric(raw["Eolien"], errors="coerce")
    stamps = pd.to_datetime(
        raw["Date"] + " " + raw["Heures"], format="%Y-%m-%d %H:%M", errors="coerce"
    )
    keep = output.notna() & stamps.notna()
    local = pd.DatetimeIndex(stamps[keep]).tz_localize(TZ, ambiguous="NaT", nonexistent="NaT")
    series = pd.Series(output[keep].to_numpy(), index=local.tz_convert("UTC"))
    series = series[series.index.notna()]
    series = series[~series.index.duplicated()]
    hourly = series.resample("1h")
    return hourly.mean()[hourly.count() == 2]


def interpolated_capacity(index: pd.DatetimeIndex, fleet: pd.Series) -> pd.Series:
    """The wind fleet at each hour, linear from one 31 December to the next.

    Args:
        index: UTC hours.
        fleet: MW connected on 31 December, indexed by year.
    """
    year = index.year
    start = pd.to_datetime(year.astype(str) + "-01-01").tz_localize("UTC")
    end = pd.to_datetime((year + 1).astype(str) + "-01-01").tz_localize("UTC")
    fraction = (index - start) / (end - start)
    before = fleet.reindex(year - 1).to_numpy()
    after = fleet.reindex(year).to_numpy()
    return pd.Series(before + (after - before) * fraction, index=index)


def national(raw: pd.DataFrame, fleet: pd.Series) -> pd.DataFrame:
    """One hourly national series: generation, capacity, and their ratio."""
    generation = hourly_output(raw)
    out = pd.DataFrame(
        {"generation_mw": generation, "capacity_mw": interpolated_capacity(generation.index, fleet)}
    )
    out["capacity_factor"] = out["generation_mw"] / out["capacity_mw"]
    return out


def split(series: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    """The UTC calendar years first to last, stamped in Paris time."""
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
        help="where the RTE annual files and fleet table are kept",
    )
    parser.add_argument("--replace", action="store_true", help="replace existing files")
    args = parser.parse_args()

    first = min(s[1] for s in SPANS)
    last = max(s[2] for s in SPANS)
    series = national(*fetch(args.cache, first, last))
    args.out.mkdir(parents=True, exist_ok=True)
    for kind, start, end in SPANS:
        name = f"fr_test_{start}.csv" if kind == "test" else f"fr_train_{start}_{end}.csv"
        frame = split(series, start, end)
        report = check_country_cf(frame, name, warn=False)
        utc_year = frame.index.tz_convert("UTC").year
        energy = (frame["generation_mw"].groupby(utc_year).sum() / 1e3).round(0)
        print(f"{name}: {len(frame)} hours, mean CF {report.mean_cf:.3f}, GWh {energy.to_dict()}")
        print(f"  gates: {'pass' if report.ok else '; '.join(report.issues)}")
        write(frame, args.out / name, replace=args.replace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
