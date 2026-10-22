"""Audit every country-level observed CF series against physical bounds.

Runs the gates in :mod:`pyvwf.loaders.country_obs_checks` over each region config
with ``obs_level = "country"``, for both the train and test splits, and prints
one row per split. Use it before trusting any country-level result: a series
that fails here produces plausible-looking correction factors and a silently
wrong model, because the affine correction absorbs a constant error into the
scalar.

It also compares each year's observed energy with Eurostat's national wind
production (``configs/curation/eurostat_wind_gwh.csv``). The physical gates
cannot see a series that measures only part of the fleet: Belgium's onshore-only
years had plausible capacity factors because their capacity was onshore too.

Usage:
    PYTHONPATH=src python scripts/analysis/audit_country_observations.py
    PYTHONPATH=src python scripts/analysis/audit_country_observations.py --csv out.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyvwf.harness.driver import resolve_source  # noqa: E402
from pyvwf.harness.regions import load_region  # noqa: E402
from pyvwf.loaders.country_obs_checks import check_country_cf  # noqa: E402

CONFIG_DIR = REPO_ROOT / "configs" / "regions"
REFERENCE = REPO_ROOT / "configs" / "curation" / "eurostat_wind_gwh.csv"

#: Observed annual energy over the national statistic. A ratio outside FAIL
#: fails the series; outside NOTE it is noted. The bands were set on
#: 2026-10-22 AFTER seeing that Elia's complete Belgian series (offshore plus
#: onshore) sits at 0.82 to 0.93 of Eurostat's gross production, so they are
#: wide enough to pass a complete series and narrow enough to fail one that
#: drops a whole segment (Belgium's onshore-only years sit at 0.34 to 0.44).
FAIL_RATIO = (0.70, 1.30)
NOTE_RATIO = (0.85, 1.15)

#: A year is compared only when the series covers this share of its hours,
#: so a test file's partial first or last day does not read as a shortfall.
MIN_YEAR_COVERAGE = 0.9


def annual_energy_ratios(obs: pd.DataFrame, reference: pd.DataFrame, country: str) -> pd.Series:
    """Observed annual wind energy over the national statistic, per UTC year.

    Args:
        obs: observed series with ``generation_mw`` in MW on a regular step.
        reference: ``country``, ``year`` and ``gwh`` rows (Eurostat gross
            production).
        country: the region code to look up in ``reference``.

    Returns:
        Ratio per year, for the years the series covers and the reference holds.
        Empty when the series has no ``generation_mw`` column.
    """
    if "generation_mw" not in obs.columns:
        return pd.Series(dtype=float)
    index = pd.to_datetime(obs.index, utc=True, format="mixed")
    generation = pd.Series(obs["generation_mw"].to_numpy(), index=index).sort_index()
    step_hours = generation.index.to_series().diff().median() / pd.Timedelta(hours=1)
    years = generation.index.year
    energy_gwh = generation.groupby(years).sum() * step_hours / 1e3
    steps = generation.groupby(years).size()
    hours_in_year = pd.Series(
        [
            (pd.Timestamp(f"{y + 1}-01-01") - pd.Timestamp(f"{y}-01-01")).days * 24
            for y in steps.index
        ],
        index=steps.index,
    )
    covered = steps * step_hours / hours_in_year >= MIN_YEAR_COVERAGE
    national = reference[reference["country"] == country].set_index("year")["gwh"]
    ratio = (energy_gwh[covered] / national).dropna()
    ratio.index.name = "year"
    return ratio


def energy_findings(ratios: pd.Series) -> tuple[list[str], list[str]]:
    """Failures and notes for the years outside the bands."""
    failures, notes = [], []
    for year, ratio in ratios.items():
        message = f"{year} energy is {ratio:.2f} of the national statistic"
        if not FAIL_RATIO[0] <= ratio <= FAIL_RATIO[1]:
            failures.append(message)
        elif not NOTE_RATIO[0] <= ratio <= NOTE_RATIO[1]:
            notes.append(message)
    return failures, notes


def audit(config_dir: Path) -> pd.DataFrame:
    """Check every country-level region's train and test series."""
    reference = pd.read_csv(REFERENCE)
    rows = []
    for config in sorted(config_dir.glob("*.toml")):
        spec = load_region(config)
        if spec.obs_level != "country":
            continue
        for split in ("train", "test"):
            label = f"{spec.code} {split}"
            try:
                source = resolve_source(spec, split)
                with warnings.catch_warnings():
                    # The gate is re-run below to capture the report itself.
                    warnings.simplefilter("ignore", UserWarning)
                    obs = source.load_observations()
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                rows.append(
                    {
                        "label": label,
                        "ok": False,
                        "issues": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            report = check_country_cf(obs, label, warn=False)
            ratios = annual_energy_ratios(obs, reference, spec.code)
            failures, notes = energy_findings(ratios)
            report.failures.extend(failures)
            report.notes.extend(notes)
            row = report.as_row()
            row["energy_ratio_min"] = ratios.min() if len(ratios) else None
            row["energy_ratio_max"] = ratios.max() if len(ratios) else None
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir", type=Path, default=CONFIG_DIR, help="region config directory"
    )
    parser.add_argument("--csv", type=Path, help="also write the table to this path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any series fails a gate",
    )
    args = parser.parse_args()

    table = audit(args.config_dir)
    if table.empty:
        print("No country-level regions found.")
        return 0

    columns = [
        c
        for c in (
            "label",
            "step_hours",
            "mean_cf",
            "peak_cf",
            "n_clipped",
            "frac_clipped",
            "longest_unchanged_years",
            "unchanged_span",
            "energy_ratio_min",
            "energy_ratio_max",
            "ok",
        )
        if c in table.columns
    ]
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(table[columns].to_string(index=False))

    failures = table[~table["ok"].astype(bool)]
    # Notes and warnings do not clear ``ok``, so they have to be counted here
    # or the tiering hides them: a series that used to fail loudly for one bad
    # hour now passes, and passing silently is not the intent.
    noted = int(table.get("n_notes", pd.Series(dtype=int)).fillna(0).sum())
    warned = int(table.get("n_warnings", pd.Series(dtype=int)).fillna(0).sum())
    print(
        f"\n{len(failures)} of {len(table)} series failed a gate; "
        f"{warned} warning{'' if warned == 1 else 's'} and "
        f"{noted} note{'' if noted == 1 else 's'} on series that passed."
    )
    if len(failures):
        print()
        for _, row in failures.iterrows():
            print(f"  FAIL {row['label']}: {row['issues']}")
    for _, row in table[table["ok"].astype(bool)].iterrows():
        if row.get("issues"):
            print(f"  note {row['label']}: {row['issues']}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")

    return 1 if (args.strict and len(failures)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
