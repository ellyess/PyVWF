"""The country weighting study's denominator table and row rule (#67).

For each row and compared training year Y, the mid-year grid total against the
observed capacity-factor denominator, the mean of the 1 January figures for Y
and Y+1. A row is excluded if the two disagree by more than 10% in any compared
year, or if its 1 January figures are flat (within 1 MW) over three or more
consecutive training years. Both rules, and the reading of "1 January figure",
are fixed in ``docs/findings/method-country-weighting-prereg.md``.

A year's 1 January figure is ``capacity_mw`` at the first timestamp of that
year's January rows in the training observation file. Where the January rows
carry more than one value, the spread is recorded beside it, so a reading that
depends on the choice is visible.

Usage, from the repository root:

    PYVWF_INPUT=input/combined python \\
        scripts/studies/method-country-weighting/denominators.py \\
        output/country_weighting_2026-10-16/denominators
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pyvwf.cli.common import make_parser
from pyvwf.config import PyVWFPaths
from pyvwf.harness.regions import load_region

ROWS = ["FR", "BE", "IE", "SE", "NO", "ES"]
#: Registered thresholds.
MAX_DISAGREEMENT, FLAT_MW, FLAT_YEARS = 0.10, 1.0, 3


def observation_file(code: str, first: int, last: int) -> Path:
    base = PyVWFPaths.COUNTRY_LEVEL_DATA / "observations" / code.lower()
    stem = f"{code.lower()}_train_{first}_{last}"
    for path in (base / f"{stem}.csv", base / f"{stem}_aggregated.csv"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"{code}: no training observation file for {stem}")


def january_figures(obs: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """The 1 January capacity of each year, with the spread over its January rows."""
    rows = []
    for year in years:
        jan = obs[(obs.index.year == year) & (obs.index.month == 1)]["capacity_mw"].dropna()
        rows.append(
            {
                "year": year,
                "jan1_mw": float(jan.iloc[0]) if len(jan) else float("nan"),
                "january_spread_mw": float(jan.max() - jan.min()) if len(jan) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def is_flat(jan1: pd.Series) -> bool:
    """Equal to within ``FLAT_MW`` over ``FLAT_YEARS`` or more consecutive years."""
    values = jan1.to_numpy()
    for i in range(len(values) - FLAT_YEARS + 1):
        window = values[i : i + FLAT_YEARS]
        if pd.notna(window).all() and window.max() - window.min() <= FLAT_MW:
            return True
    return False


def row_table(code: str) -> tuple[pd.DataFrame, dict]:
    spec = load_region(f"configs/regions/scorecard/{code.lower()}_country.toml")
    first, last = spec.train_years
    obs_path = observation_file(code, first, last)
    obs = pd.read_csv(obs_path, index_col=0)
    obs.index = pd.to_datetime(obs.index, utc=True)
    years = list(range(first, last + 1))
    jan = january_figures(obs, years).set_index("year")

    grid_dir = PyVWFPaths.COUNTRY_LEVEL_DATA / "grid_points" / code.lower()
    rows = []
    # No file carries a figure for the year after the last, so it is not compared.
    for year in years[:-1]:
        grid_total = float(
            pd.read_csv(grid_dir / f"{code.lower()}_grid_points_{year}_midyear.csv")[
                "capacity"
            ].sum()
        )
        observed = (jan.loc[year, "jan1_mw"] + jan.loc[year + 1, "jan1_mw"]) / 2.0
        rows.append(
            {
                "region": code,
                "year": year,
                "grid_total_midyear_mw": grid_total,
                "jan1_mw": jan.loc[year, "jan1_mw"],
                "jan1_next_mw": jan.loc[year + 1, "jan1_mw"],
                "observed_denominator_mw": observed,
                "relative_disagreement": abs(grid_total - observed) / observed,
                "january_spread_mw": jan.loc[year, "january_spread_mw"],
            }
        )
    table = pd.DataFrame(rows)
    flat = is_flat(jan["jan1_mw"])
    worst = float(table["relative_disagreement"].max())
    reasons = []
    if worst > MAX_DISAGREEMENT:
        reasons.append(f"disagreement {worst:.1%} > {MAX_DISAGREEMENT:.0%}")
    if flat:
        reasons.append(f"1 January figures flat over {FLAT_YEARS}+ consecutive years")
    verdict = {
        "region": code,
        "train_years": [first, last],
        "observation_file": str(obs_path),
        "max_relative_disagreement": worst,
        "flat": flat,
        "jan1_mw": {str(y): float(v) for y, v in jan["jan1_mw"].items()},
        "excluded": bool(reasons),
        "reasons": reasons,
        "note": (
            "IE's denominator is built from the same register as its grid, so its "
            "agreement is not independent evidence."
            if code == "IE"
            else ""
        ),
    }
    return table, verdict


def main(out_dir: str) -> None:
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} already holds a denominator table")
    out.mkdir(parents=True, exist_ok=True)
    tables, verdicts = [], []
    for code in ROWS:
        table, verdict = row_table(code)
        tables.append(table)
        verdicts.append(verdict)
    pd.concat(tables).to_csv(out / "denominators.csv", index=False)
    scoreable = [v["region"] for v in verdicts if not v["excluded"]]
    summary = {"rows": verdicts, "scoreable": scoreable, "n_scoreable": len(scoreable)}
    (out / "row_rule.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.set_option("display.width", 200)
    print(pd.concat(tables).round(4).to_string(index=False))
    for v in verdicts:
        print(v["region"], "EXCLUDED" if v["excluded"] else "scoreable", "; ".join(v["reasons"]))
    print(f"scoreable: {scoreable} ({len(scoreable)})")


def cli(argv: list[str] | None = None) -> None:
    parser = make_parser("The country weighting study's denominator table and row rule.")
    parser.add_argument("out_dir", help="directory for denominators.csv and row_rule.json")
    args = parser.parse_args(argv)
    main(args.out_dir)


if __name__ == "__main__":
    cli()
