"""Which capacity weights thesis chapter 4's country-level runs carried.

The design record and the country-level review describe the chapter's country
grids as uniform, every point holding the same synthetic capacity. The
chapter's own run directories, `output/runs/turbine_grid/*-obs_country-*`,
hold capacities that vary by point. This reconstructs them from the Global
Wind Power Tracker with the rule in `development:scripts/
regenerate_grid_points_with_gwpt.py`, and reports what that rule does.

The rule, as written there: a project counts in year Y if its start year is Y
or earlier or blank, and its retirement year is after Y or blank, with no
filter on status; each grid point's capacity is the sum over every counted
project within a haversine radius, floored at 3 MW.

**A reproduction is only evidence if a wrong parameter fails it**, so every
country is also checked at a neighbouring year and radius, which must not
reproduce.

Read-only. Writes under ``<out_dir>``.

Usage, from the repository root:

    PYTHONPATH=src python scripts/analysis/chapter_capacity_weights.py <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "output/runs/turbine_grid"
GWPT = REPO / "input/reference/gwpt/Global-Wind-Power-Tracker-February-2026.xlsx"

COUNTRY = {"BE": "Belgium", "ES": "Spain", "FR": "France", "IE": "Ireland",
           "IT": "Italy", "NL": "Netherlands", "NO": "Norway", "PT": "Portugal",
           "SE": "Sweden"}

YEAR, RADIUS_KM, FLOOR_MW = 2015, 50.0, 3.0
#: Wrong parameters, each of which must fail to reproduce.
CONTROLS = ((2016, 50.0), (2015, 25.0))


def haversine_km(lat, lon, lats, lons):
    p1, p2 = np.radians(lat), np.radians(lats)
    a = (np.sin(np.radians(lats - lat) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lons - lon) / 2) ** 2)
    return 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def counted(projects: pd.DataFrame, year: int) -> pd.DataFrame:
    start, retired = projects["Start year"], projects["Retired year"]
    keep = (start.isna() | (start <= year)) & (retired.isna() | (retired > year))
    return projects[keep]


def weights(points: pd.DataFrame, projects: pd.DataFrame, radius: float) -> np.ndarray:
    lats, lons = projects["Latitude"].to_numpy(), projects["Longitude"].to_numpy()
    cap = projects["Capacity (MW)"].to_numpy()
    summed = [cap[haversine_km(a, b, lats, lons) <= radius].sum()
              for a, b in zip(points["lat"], points["lon"])]
    return np.maximum(summed, FLOOR_MW)


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tracker = pd.read_excel(GWPT, sheet_name="Data")
    for col in ("Start year", "Retired year"):
        tracker[col] = pd.to_numeric(tracker[col], errors="coerce")
    tracker = tracker.dropna(subset=["Latitude", "Longitude", "Capacity (MW)"])

    rows = []
    for code, name in COUNTRY.items():
        info = RUNS / f"{code}-all-obs_country-corrected-calc_z0/training/simulated-turbines"
        train = pd.read_csv(info / f"{code}_train_turb_info.csv")
        test = pd.read_csv(info / f"{code}_2023_turb_info.csv")
        projects = tracker[tracker["Country/Area"] == name]
        fleet = counted(projects, YEAR)
        rebuilt = weights(train, fleet, RADIUS_KM)
        run_cap = train["capacity"].to_numpy()

        # Each project's share of the summed weight is its capacity times the
        # number of points within the radius, which is how a status or a blank
        # start year reaches the weights rather than only the project list.
        lats, lons = fleet["Latitude"].to_numpy(), fleet["Longitude"].to_numpy()
        hits = np.zeros(len(fleet))
        for a, b in zip(train["lat"], train["lon"]):
            hits += haversine_km(a, b, lats, lons) <= RADIUS_KM
        contribution = fleet["Capacity (MW)"].to_numpy() * hits
        non_operating = (fleet["Status"] != "operating").to_numpy()
        blank_start = fleet["Start year"].isna().to_numpy()

        row = {
            "country": code,
            "points": len(train),
            "max_abs_diff_mw": float(np.abs(rebuilt - run_cap).max()),
            "test_equals_train": bool(np.array_equal(test["capacity"], train["capacity"])),
            "counted_fleet_mw": round(float(fleet["Capacity (MW)"].sum())),
            "operating_fleet_mw": round(float(fleet.loc[~non_operating, "Capacity (MW)"].sum())),
            "summed_weight_mw": round(float(run_cap.sum())),
            "weight_over_counted_fleet": round(float(run_cap.sum() / fleet["Capacity (MW)"].sum()), 2),
            "non_operating_share_of_counted_mw": round(float(
                fleet.loc[non_operating, "Capacity (MW)"].sum() / fleet["Capacity (MW)"].sum()), 3),
            "blank_start_share_of_counted_mw": round(float(
                fleet.loc[blank_start, "Capacity (MW)"].sum() / fleet["Capacity (MW)"].sum()), 3),
            "non_operating_share_of_weight": round(float(
                contribution[non_operating].sum() / max(contribution.sum(), 1e-9)), 3),
            "points_at_floor": int((run_cap == FLOOR_MW).sum()),
            "floor_share_of_weight": round(float(
                (run_cap == FLOOR_MW).sum() * FLOOR_MW / run_cap.sum()), 4),
        }
        for year, radius in CONTROLS:
            wrong = weights(train, counted(projects, year), radius)
            row[f"control_{year}_{int(radius)}km_max_abs_diff_mw"] = float(
                np.abs(wrong - run_cap).max())
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(out / "chapter_capacity_weights.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.to_string(index=False))

    reproduced = (table["max_abs_diff_mw"] < 1e-6).all()
    controls_fail = all((table[c] > 1.0).all() for c in table.columns if c.startswith("control_"))
    print(f"\nreproduced in all nine: {reproduced}; every wrong parameter fails: {controls_fail}")


if __name__ == "__main__":
    main(sys.argv[1])
