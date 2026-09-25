"""Audit every country-level observed CF series against physical bounds.

Runs the gates in :mod:`pyvwf.loaders.country_obs_checks` over each region config
with ``obs_level = "country"``, for both the train and test splits, and prints
one row per split. Use it before trusting any country-level result: a series
that fails here produces plausible-looking correction factors and a silently
wrong model, because the affine correction absorbs a constant error into the
scalar.

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


def audit(config_dir: Path) -> pd.DataFrame:
    """Check every country-level region's train and test series."""
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
            rows.append(report.as_row())
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
