"""Give a region's plants real power curves and hub heights from a spec table.

The newer turbine-level regions (CL, AR, and until now AU-NEM) shipped a
metadata file with one uniform power curve and one uniform hub height for every
plant, because their generation source carries no turbine detail. That is
indefensible for a paid simulation: a 1.5 MW / 77 m machine and a 6 MW / 163 m
machine reach rated power at completely different wind speeds, and the uniform
curve mis-states where most of the fleet sits. It also lets the bias correction
launder a curve error into what looks like a reanalysis-bias correction.

This joins a per-farm turbine-spec table (manufacturer, model, turbine count,
rotor diameter, hub height) onto the region's metadata and:

  - assigns each plant a real open-library curve by scale and specific power,
    reusing the exact matcher the US and NZ regions use
    (``vwf.datasets.eia_us.assign_curves_from_library``), and
  - sets each plant's hub height from the spec table.

Plants the spec table cannot cover keep the uniform default, recorded in
``model_source``/``height_source`` so a run manifest stays honest about which
plants are real and which are proxies. The uniform metadata is backed up.

The spec table's schema (one row per plant, joined on ``ID``):

    ID, turbine_count, rotor_diameter_m, hub_height_m [, unit_mw, ...]

Extra columns are ignored. Rows with a blank turbine_count or rotor_diameter_m
fall back to the uniform curve.

Usage:
    PYTHONPATH=src python scripts/region_tools/apply_turbine_specs.py CL \\
        --specs configs/curation/cl_turbine_specs.csv --dry-run
    PYTHONPATH=src python scripts/region_tools/apply_turbine_specs.py CL \\
        --specs configs/curation/cl_turbine_specs.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vwf.config import PyVWFPaths  # noqa: E402
from vwf.datasets.eia_us import assign_curves_from_library  # noqa: E402

#: Region code to the metadata directory and file stem.
REGION_DIR = {"CL": "CL", "AR": "AR", "AU-NEM": "AU_NEM", "BR": "BR"}

DEFAULT_MODEL = "2019COE_Market_Average_2.6MW_121"
DEFAULT_HEIGHT = 100.0
BACKUP_SUFFIX = ".uniform.bak.csv"


def md_path(code: str) -> Path:
    folder = REGION_DIR.get(code.upper(), code.upper())
    stem = folder.lower()
    return PyVWFPaths.TURBINE_DATA / folder / f"{stem}_md.csv"


def apply_specs(md: pd.DataFrame, specs: pd.DataFrame) -> pd.DataFrame:
    """Match curves and set heights on a copy of ``md`` from ``specs``."""
    md = md.copy()
    md["ID"] = md["ID"].astype(str)
    specs = specs.copy()
    specs["ID"] = specs["ID"].astype(str)

    keep = ["ID", "turbine_count", "rotor_diameter_m"]
    for optional in ("hub_height_m", "unit_mw"):
        if optional in specs.columns:
            keep.append(optional)
    joined = md.merge(specs[keep], on="ID", how="left")

    # The US matcher matches on per-turbine rating and specific power. It derives
    # the per-turbine rating as capacity / n_turbines, which is only right when
    # the metadata capacity is the plant's real nameplate. For AR it is not: the
    # generation source carries no capacity, so it was joined from GWPT, where
    # phases sharing a site carry a duplicated site-total. Dividing that by a
    # per-phase turbine count inflates the rating and picks the wrong curve.
    #
    # The spec table's own unit_mw is the reliable per-turbine rating, so it is
    # used directly where present. The matcher only reads `capacity` and
    # `n_turbines` to form that ratio, so feeding it unit_mw with n_turbines = 1
    # gives the correct match without disturbing the real capacity, which is
    # restored afterwards for the simulation and weighting.
    diameter = pd.to_numeric(joined["rotor_diameter_m"], errors="coerce")
    count = pd.to_numeric(joined["turbine_count"], errors="coerce")
    unit_kw = pd.to_numeric(joined.get("unit_mw"), errors="coerce") * 1000.0
    per_turbine_kw = unit_kw.where(
        unit_kw.notna(), pd.to_numeric(joined["capacity"], errors="coerce") / count
    )

    match_input = joined.copy()
    match_input["diameter"] = diameter
    match_input["n_turbines"] = 1.0
    match_input["capacity"] = per_turbine_kw  # per-turbine kW, for matching only
    picked = assign_curves_from_library(match_input, fallback_model=DEFAULT_MODEL)

    matched = joined.copy()  # real capacity intact
    matched["model"] = picked["model"].to_numpy()
    matched["model_source"] = picked["model_source"].to_numpy()

    # Hub height from the spec table where present, uniform default otherwise.
    if "hub_height_m" in joined.columns:
        h = pd.to_numeric(joined["hub_height_m"], errors="coerce")
        matched["height"] = h.where(h.notna(), DEFAULT_HEIGHT)
        matched["height_source"] = np.where(
            h.notna(), "spec-table", "default-uniform"
        )
    else:
        matched["height"] = DEFAULT_HEIGHT
        matched["height_source"] = "default-uniform"

    return matched.drop(columns=["turbine_count", "rotor_diameter_m",
                                 "hub_height_m", "unit_mw"],
                        errors="ignore")


def summarise(code: str, before: pd.DataFrame, after: pd.DataFrame) -> None:
    cap = after["capacity"]
    matched = after["model_source"].eq("matched-scale-and-specific-power")
    real_h = after["height_source"].eq("spec-table")
    print(f"\n{code}: {len(after)} plants")
    print(f"  curves: {after['model'].nunique()} distinct (was "
          f"{before['model'].nunique()})")
    print(f"  matched to a real curve: {matched.sum()}/{len(after)} plants, "
          f"{100 * cap[matched].sum() / cap.sum():.0f}% of capacity")
    print(f"  real hub height: {real_h.sum()}/{len(after)} plants")
    src = after["model_source"].value_counts().to_dict()
    print(f"  model_source: {src}")
    if matched.any():
        h = after.loc[real_h, "height"]
        if len(h):
            print(f"  hub heights: {h.min():.0f}-{h.max():.0f} m")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region", help="region code, e.g. CL AR AU-NEM")
    parser.add_argument("--specs", type=Path, required=True,
                        help="per-farm turbine spec CSV")
    parser.add_argument("--dry-run", action="store_true", help="report only")
    args = parser.parse_args()

    code = args.region.upper()
    path = md_path(code)
    if not path.is_file():
        print(f"{code}: metadata not found at {path}")
        return 1
    if not args.specs.is_file():
        print(f"spec table not found at {args.specs}")
        return 1

    before = pd.read_csv(path)
    specs = pd.read_csv(args.specs)
    after = apply_specs(before, specs)
    summarise(code, before, after)

    if args.dry_run:
        print("\ndry run, nothing written")
        return 0

    backup = path.with_suffix(BACKUP_SUFFIX)
    if not backup.exists():
        before.to_csv(backup, index=False)
        print(f"\nbacked up uniform metadata to {backup.name}")
    after.to_csv(path, index=False)
    print(f"wrote matched metadata to {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
