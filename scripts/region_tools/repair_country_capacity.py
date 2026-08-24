"""Rebuild a country CF series against a real installed-capacity register.

ENTSO-E's installed-capacity endpoint does not always track the fleet. Ireland's
returns 1907.13 MW for every year from 2015 to 2021 while the real fleet nearly
doubled, so every derived capacity factor is inflated: the series peaks at 1.87
before the fetcher's clip, and 5.7% of the 2023 rows sit on the ceiling with
their true value discarded. Generation is fine; only the denominator is wrong.

This rewrites ``capacity_mw`` from the Global Wind Power Tracker, which carries
a start year and a retirement year per project and so gives an annual register,
then recomputes ``capacity_factor``. On Ireland it moves the mean CF from 0.480
to 0.332 and the annual peak CF to 0.93 to 1.00 for 2017 onward, which is what a
national fleet looks like. GWPT undercounts Ireland before 2017 (peak CF still
1.05 in 2015), so ``--from-year`` also writes a trimmed training file over the
window where the register is trustworthy.

Check the result with scripts/analysis/audit_country_observations.py.

Usage:
    PYTHONPATH=src python scripts/region_tools/repair_country_capacity.py IE --dry-run
    PYTHONPATH=src python scripts/region_tools/repair_country_capacity.py IE --from-year 2017
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vwf.config import PyVWFPaths  # noqa: E402
from vwf.loaders.country_obs_checks import check_country_cf  # noqa: E402
from weight_country_grid_points import GWPT_PATH, fleet_for, load_gwpt  # noqa: E402

#: Matches the ceiling applied by the ENTSO-E fetcher.
CLIP = 1.5

BACKUP_SUFFIX = ".entsoe-capacity.bak.csv"


def annual_capacity(gwpt: pd.DataFrame, country: str, years) -> dict[int, float]:
    """Installed capacity in MW as of each year, from the tracker."""
    return {int(y): float(fleet_for(gwpt, country, int(y))["mw"].sum()) for y in years}


def repair(path: Path, gwpt: pd.DataFrame, country: str, *, dry_run: bool) -> pd.DataFrame:
    """Rewrite one observation file's capacity and capacity factor."""
    obs = pd.read_csv(path, index_col=0)
    obs.index = pd.to_datetime(obs.index, utc=True, format="mixed")
    obs = obs.sort_index()

    if "generation_mw" not in obs.columns:
        raise ValueError(f"{path} has no 'generation_mw' column to rebuild the ratio from")

    capacity = annual_capacity(gwpt, country, sorted(obs.index.year.unique()))
    if any(mw <= 0 for mw in capacity.values()):
        raise ValueError(f"{country}: the tracker reports no capacity for some year: {capacity}")

    before = check_country_cf(obs, f"{path.name} before", warn=False)

    out = obs.copy()
    out["capacity_mw"] = out.index.year.map(capacity)
    out["capacity_factor"] = (out["generation_mw"] / out["capacity_mw"]).clip(0, CLIP)

    after = check_country_cf(out, f"{path.name} after", warn=False)
    print(f"\n{path.name}")
    print(f"  capacity  {obs['capacity_mw'].min():.0f}-{obs['capacity_mw'].max():.0f} MW "
          f"-> {out['capacity_mw'].min():.0f}-{out['capacity_mw'].max():.0f} MW")
    print(f"  mean CF   {before.mean_cf:.3f} -> {after.mean_cf:.3f}")
    print(f"  peak CF   {before.peak_cf:.3f} -> {after.peak_cf:.3f}")
    peaks = (out.groupby(out.index.year)["capacity_factor"].max()).round(2)
    print(f"  peak CF by year: {peaks.to_dict()}")
    print(f"  gates: {'pass' if after.ok else '; '.join(after.issues)}")

    if not dry_run:
        backup = path.with_suffix(BACKUP_SUFFIX)
        if not backup.exists():
            obs.to_csv(backup)
            print(f"  kept the ENTSO-E-capacity version as {backup.name}")
        out.to_csv(path)
        print(f"  rewrote {path.name}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("country", help="region code, e.g. IE")
    parser.add_argument(
        "--from-year",
        type=int,
        help="also write a training file trimmed to start at this year, for the "
        "window where the tracker's register is trustworthy",
    )
    parser.add_argument("--dry-run", action="store_true", help="report only")
    parser.add_argument("--gwpt", type=Path, default=GWPT_PATH)
    args = parser.parse_args()

    code = args.country.upper()
    obs_dir = PyVWFPaths.COUNTRY_LEVEL_DATA / "observations" / code.lower()
    if not obs_dir.is_dir():
        print(f"{code}: no observations at {obs_dir}")
        return 1

    gwpt = load_gwpt(args.gwpt)
    paths = sorted(p for p in obs_dir.glob(f"{code.lower()}_*.csv") if ".bak." not in p.name)
    repaired = {p: repair(p, gwpt, code, dry_run=args.dry_run) for p in paths}

    if args.from_year:
        for path, frame in repaired.items():
            if "_train_" not in path.name:
                continue
            trimmed = frame[frame.index.year >= args.from_year]
            if trimmed.empty:
                continue
            end = int(trimmed.index.year.max())
            out = path.parent / f"{code.lower()}_train_{args.from_year}_{end}.csv"
            print(f"\n{out.name}: {len(trimmed)} rows, {args.from_year}-{end}")
            if not args.dry_run:
                trimmed.to_csv(out)
                print(f"  wrote {out.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
