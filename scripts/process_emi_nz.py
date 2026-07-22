#!/usr/bin/env python
"""Build the EMINewZealandSource inputs from the EMI downloads.

Reads (all local, downloaded by scripts/fetch_emi_nz.py — user-executed):
    input/emi_raw/<YYYYMM>_Generation_MD.csv   half-hourly kWh per plant
    input/emi_raw/DispatchedGenerationPlant.csv  (register; report-only here)
and the curated farm tables committed in configs/:
    configs/nz_wind_farms.csv       per-farm metadata with provenance
    configs/nz_capacity_stages.csv  stable capacity plateaus for staged builds
    configs/nz_mask_windows.csv     commissioning-ramp months to mask

Writes (under <out>, default input/turbine_level_data/NZ/):
    nz_md.csv          EMINewZealandSource metadata contract
    nz_obs.csv         monthly CF wide frame (UTC bins, coverage-screened)
    nz_build_mask.csv  (ID, year, month) commissioning months to NaN
    join_report.md     coverage/matching report

Wind rows are selected by Fuel_Code in {"Wind", "WIN"} (the coding style
changed with Kaiwera Downs 2) and keyed on Gen_Code, case-normalised —
Site_Code is NOT stable across years (Te Apiti was TAP, later WDV) and
capitalisation varies (Harapaki, KaiweraDowns). An unmapped wind Gen_Code is
a hard error: it means a new farm entered the fleet and the curated table
needs a row (or an explicit exclusion below).

Power-curve keys are assigned per farm by scale-then-specific-power matching
against the active curve library (the same guarded matcher the US fleet
uses), with the true manufacturer/model string carried in ``turbine_model``
for provenance — never as the curve key.

    python scripts/process_emi_nz.py
    python scripts/process_emi_nz.py --years 2019 2024   # inclusive window
"""
import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

from vwf.datasets.eia_us import assign_curves_from_library
from vwf.datasets.emi_nz import half_hourly_from_generation_md, monthly_cf

#: Register-listed wind Gen_Codes that never carry wind rows in Generation_MD,
#: or embedded farms outside the dispatched dataset. Documented exclusions —
#: an unmapped Gen_Code NOT in this set is an error.
KNOWN_ABSENT = {
    # Mahinerangi is metered inside the Waipori hydro scheme; its output never
    # appears as wind rows in Generation_MD (verified 2013/2025/2026 files).
    "mahinerangi",
}

WIND_FUELS = {"wind", "win"}


def load_curated(configs: Path):
    farms = pd.read_csv(configs / "nz_wind_farms.csv")
    farms["ID"] = farms["ID"].astype(str)
    stages = pd.read_csv(configs / "nz_capacity_stages.csv")
    windows = pd.read_csv(configs / "nz_mask_windows.csv")
    return farms, stages, windows


def gen_code_map(farms: pd.DataFrame) -> dict[str, str]:
    """Lower-cased Gen_Code -> farm ID."""
    mapping: dict[str, str] = {}
    for _, row in farms.iterrows():
        for code in str(row["gen_codes"]).split(";"):
            mapping[code.strip().lower()] = row["ID"]
    return mapping


def capacity_history(farms: pd.DataFrame, stages: pd.DataFrame) -> pd.DataFrame:
    """Stable-plateau capacity history: curated stages where given, else one
    row per farm from first generation at final capacity."""
    rows = [
        pd.DataFrame({
            "ID": stages["ID"].astype(str),
            "effective_from": pd.to_datetime(stages["effective_from"]),
            "capacity": pd.to_numeric(stages["capacity"]),
        })
    ]
    staged = set(stages["ID"].astype(str))
    plain = farms[~farms["ID"].isin(staged)]
    rows.append(pd.DataFrame({
        "ID": plain["ID"],
        "effective_from": pd.to_datetime(plain["first_generation"]),
        "capacity": pd.to_numeric(plain["capacity"]),
    }))
    return pd.concat(rows, ignore_index=True).sort_values(
        ["ID", "effective_from"]
    ).reset_index(drop=True)


def mask_from_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """Expand curated (from_month, to_month) windows to (ID, year, month)."""
    rows = []
    for _, w in windows.iterrows():
        months = pd.period_range(w["from_month"], w["to_month"], freq="M")
        for p in months:
            rows.append({"ID": str(w["ID"]), "year": p.year, "month": p.month})
    return pd.DataFrame(rows, columns=["ID", "year", "month"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="input/emi_raw",
                    help="Directory of <YYYYMM>_Generation_MD.csv files")
    ap.add_argument("--configs", default="configs",
                    help="Directory holding the curated nz_*.csv tables")
    ap.add_argument("--years", type=int, nargs=2, default=[2019, 2024],
                    metavar=("START", "END"), help="Inclusive UTC year window")
    ap.add_argument("--out", default="input/turbine_level_data/NZ")
    ap.add_argument("--fallback-model", default="2019COE_Market_Average_2.6MW_121",
                    help="Uniform curve key for farms the matcher cannot place "
                    "(must be a column of power_curves.csv)")
    args = ap.parse_args()

    raw_paths = sorted(glob.glob(str(Path(args.raw) / "*_Generation_MD.csv")))
    if not raw_paths:
        sys.exit(f"no *_Generation_MD.csv under {args.raw} — run "
                 "scripts/fetch_emi_nz.py first (user-executed).")

    farms, stages, windows = load_curated(Path(args.configs))
    mapping = gen_code_map(farms)

    # --- melt every monthly file to half-hourly UTC farm rows ---------------
    pieces = []
    unmapped: set[str] = set()
    for path in raw_paths:
        gen = pd.read_csv(path, low_memory=False)
        fuel = gen["Fuel_Code"].astype(str).str.strip().str.lower()
        wind = gen[fuel.isin(WIND_FUELS)].copy()
        if wind.empty:
            continue
        codes = wind["Gen_Code"].astype(str).str.strip().str.lower()
        unmapped |= set(codes[~codes.isin(mapping)]) - KNOWN_ABSENT
        wind["farm"] = codes.map(mapping)
        wind = wind[wind["farm"].notna()]
        if wind.empty:
            continue
        hh = half_hourly_from_generation_md(wind, id_col="farm")
        pieces.append(hh)

    if unmapped:
        sys.exit(
            "wind Gen_Codes with no row in configs/nz_wind_farms.csv (new "
            f"farm(s)? add rows or an explicit exclusion): {sorted(unmapped)}"
        )
    if not pieces:
        sys.exit("no wind rows found in any Generation_MD file.")

    half_hourly = (
        pd.concat(pieces, ignore_index=True)
        .groupby(["ID", "timestamp"], as_index=False)["kwh"].sum()
    )

    # --- monthly CF + build mask -------------------------------------------
    history = capacity_history(farms, stages)
    y0, y1 = args.years
    wide = monthly_cf(half_hourly, history, y0, y1)
    mask = mask_from_windows(windows)

    # --- metadata contract --------------------------------------------------
    md = farms.rename(columns={"turbine_model": "true_model"}).copy()
    md["model"] = args.fallback_model
    md["model_source"] = "default-uniform"
    md = assign_curves_from_library(md, fallback_model=args.fallback_model)
    md["commissioning_date"] = pd.to_datetime(md["first_generation"])
    out_md = md[
        ["ID", "site_name", "lon", "lat", "height", "capacity", "model",
         "type", "commissioning_date", "height_source", "model_source",
         "true_model", "n_turbines", "diameter", "operator"]
    ]

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
        "(commissioning ramps; curated windows in configs/nz_mask_windows.csv)",
        f"- power curves: {out_md['model'].nunique()} distinct "
        f"({int(out_md['model_source'].str.startswith('matched').sum())} matched "
        "on scale then specific power, "
        f"{int(out_md['model_source'].str.startswith('default-uniform').sum())} "
        "on the uniform fallback)",
        "",
        "Hub heights are per-farm from the curated table; height_source marks "
        "the unverified ones (tararua_3, mill_creek, kaiwera_downs_2).",
        "Te Rere Hau is degraded late-window (turbines stopped/derated): its "
        "observed CF understates the resource — standing caveat.",
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
