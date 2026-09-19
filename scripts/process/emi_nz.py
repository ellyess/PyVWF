#!/usr/bin/env python
"""Build the EMINewZealandSource inputs from the EMI downloads.

Reads (all local, downloaded by scripts/fetch/emi_nz.py, user-executed):
    input/raw/emi/<YYYYMM>_Generation_MD.csv   half-hourly kWh per plant
    input/raw/emi/DispatchedGenerationPlant.csv  (register; report-only here)
and the curated farm tables committed in configs/:
    configs/curation/nz_wind_farms.csv       per-farm metadata with provenance
    configs/curation/nz_capacity_stages.csv  stable capacity plateaus for staged builds
    configs/curation/nz_mask_windows.csv     commissioning-ramp months to mask

Writes (under <out>, default input/observations/turbine/NZ/):
    nz_md.csv          EMINewZealandSource metadata contract
    nz_obs.csv         monthly CF wide frame (UTC bins, coverage-screened)
    nz_build_mask.csv  (ID, year, month) commissioning months to NaN
    join_report.md     coverage/matching report

Wind rows are selected by Fuel_Code in {"Wind", "WIN"} (the coding style
changed with Kaiwera Downs 2) and keyed on Gen_Code, case-normalised;
Site_Code is NOT stable across years (Te Apiti was TAP, later WDV) and
capitalisation varies (Harapaki, KaiweraDowns). An unmapped wind Gen_Code is
a hard error: it means a new farm entered the fleet and the curated table
needs a row (or an explicit exclusion below).

Power-curve keys are assigned per farm by scale-then-specific-power matching
against the active curve library (the same guarded matcher the US fleet
uses), with the true manufacturer/model string carried in ``turbine_model``
for provenance, never as the curve key.

    python scripts/process/emi_nz.py
    python scripts/process/emi_nz.py --years 2019 2024   # inclusive window
"""

import argparse
import glob
import sys
from pathlib import Path

from vwf.cli.common import add_input_path
from vwf.datasets.emi_nz import (
    capacity_history_from_curation,
    gen_code_map,
    load_curated_tables,
    mask_from_windows,
    metadata_contract,
    monthly_cf,
    wind_half_hourly,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_input_path(ap, "--raw", "raw", "emi", help="Directory of <YYYYMM>_Generation_MD.csv files")
    ap.add_argument(
        "--configs",
        default="configs/curation",
        help="Directory holding the curated nz_*.csv tables",
    )
    ap.add_argument(
        "--years",
        type=int,
        nargs=2,
        default=[2019, 2024],
        metavar=("START", "END"),
        help="Inclusive UTC year window",
    )
    add_input_path(ap, "--out", "observations", "turbine", "NZ")
    ap.add_argument(
        "--fallback-model",
        default="2019COE_Market_Average_2.6MW_121",
        help="Uniform curve key for farms the matcher cannot place "
        "(must be a column of power_curves.csv)",
    )
    args = ap.parse_args()

    raw_paths = sorted(glob.glob(str(Path(args.raw) / "*_Generation_MD.csv")))
    if not raw_paths:
        sys.exit(
            f"no *_Generation_MD.csv under {args.raw}; run "
            "scripts/fetch/emi_nz.py first (user-executed)."
        )

    farms, stages, windows = load_curated_tables(Path(args.configs))
    half_hourly, unmapped = wind_half_hourly(raw_paths, gen_code_map(farms))
    if unmapped:
        sys.exit(
            "wind Gen_Codes with no row in configs/curation/nz_wind_farms.csv (new "
            f"farm(s)? add rows or an explicit exclusion): {sorted(unmapped)}"
        )
    if half_hourly is None:
        sys.exit("no wind rows found in any Generation_MD file.")

    # --- monthly CF + build mask -------------------------------------------
    history = capacity_history_from_curation(farms, stages)
    y0, y1 = args.years
    wide = monthly_cf(half_hourly, history, y0, y1)
    mask = mask_from_windows(windows)

    # --- metadata contract --------------------------------------------------
    out_md = metadata_contract(farms, args.fallback_model)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_md.to_csv(out / "nz_md.csv", index=False)
    wide.to_csv(out / "nz_obs.csv", index=False)
    mask.to_csv(out / "nz_build_mask.csv", index=False)

    # --- report -------------------------------------------------------------
    obs_cols = [f"obs_{m}" for m in range(1, 13)]
    n_months = int(wide[obs_cols].notna().sum().sum())
    lines = [
        "# New Zealand (EMI) farm report",
        "",
        f"- farms in curated table: {len(farms)}; farm-years in window "
        f"{y0}-{y1}: {len(wide)}; non-NaN farm-months: {n_months}",
        f"- total final-build capacity: {farms['capacity'].sum() / 1e6:.2f} GW",
        f"- build-mask months: {len(mask)} "
        "(commissioning ramps; curated windows in configs/curation/nz_mask_windows.csv)",
        f"- power curves: {out_md['model'].nunique()} distinct "
        f"({int(out_md['model_source'].str.startswith('matched').sum())} matched "
        "on scale then specific power, "
        f"{int(out_md['model_source'].str.startswith('default-uniform').sum())} "
        "on the uniform fallback)",
        "",
        "Hub heights are per-farm from the curated table; height_source marks "
        "the unverified ones (tararua_3, mill_creek, kaiwera_downs_2).",
        "Te Rere Hau is degraded late-window (turbines stopped/derated): its "
        "observed CF understates the resource (standing caveat).",
        "Mahinerangi is excluded: metered inside the Waipori hydro scheme, "
        "never appears as wind in Generation_MD.",
        "",
    ]
    (out / "join_report.md").write_text("\n".join(lines))
    print(f"metadata: {len(out_md)} farms -> {out / 'nz_md.csv'}")
    print(f"observations: {len(wide)} farm-years -> {out / 'nz_obs.csv'}")
    print(f"build mask: {len(mask)} farm-months -> {out / 'nz_build_mask.csv'}")
    print(f"join report -> {out / 'join_report.md'}")


if __name__ == "__main__":
    main()
