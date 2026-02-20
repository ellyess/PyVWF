"""Evaluate all PyVWF correction runs and output error metrics to CSV.

This script searches for all PyVWF runs (country-level and turbine-level),
calculates evaluation metrics using vwf.metrics.overall_error for consistency,
and outputs a comprehensive CSV file.

Usage:
    python scripts/evaluate_all_pyvwf_runs.py --prefix turbine_grid
    python scripts/evaluate_all_pyvwf_runs.py --prefix turbine_grid --output output/runs/turbine_grid/pyvwf_evaluation_metrics.csv
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from vwf.metrics import overall_error


def parse_run_directory(run_dir: Path) -> dict:
    """Parse PyVWF run directory name to extract metadata.

    Expected format: {COUNTRY}-{MODE}-obs_{OBS_LEVEL}-corrected-calc_z0
    Examples:
        - NL-all-obs_country-corrected-calc_z0
        - DE-onshore-obs_turbine-corrected-calc_z0
        - DK-offshore-obs_turbine-corrected-calc_z0

    Args:
        run_dir: Path to run directory.

    Returns:
        Dictionary with country, mode, obs_level, and full run_name.
    """
    dir_name = run_dir.name

    # Parse directory name
    match = re.match(r"([A-Z]{2})-([a-z]+)-obs_([a-z]+)-", dir_name)

    if not match:
        return {
            "country": "UNKNOWN",
            "mode": "UNKNOWN",
            "obs_level": "UNKNOWN",
            "run_name": dir_name,
        }

    country, mode, obs_level = match.groups()

    return {
        "country": country,
        "mode": mode,  # all, onshore, offshore
        "obs_level": obs_level,  # country, turbine
        "run_name": dir_name,
    }


def _aggregate_sim_to_country(df_sim: pd.DataFrame, turb_info: pd.DataFrame) -> pd.Series:
    """Aggregate grid-level simulation to capacity-weighted country average.

    Args:
        df_sim: Simulation DataFrame with 'time' column and per-grid columns.
        turb_info: Turbine info with 'ID' and 'capacity' columns.

    Returns:
        Series of capacity-weighted average CF values aligned with df_sim index.
    """
    grid_cols = [c for c in df_sim.columns if c != 'time']
    cap_map = turb_info.set_index('ID')['capacity']
    # Only use grid columns that have capacity info
    valid_cols = [c for c in grid_cols if c in cap_map.index]
    caps = cap_map[valid_cols].values.astype(float)
    total_cap = caps.sum()
    if total_cap == 0:
        return pd.Series(np.nan, index=df_sim.index)
    # Use nansum to skip NaN grid cells, adjusting weights accordingly
    vals = df_sim[valid_cols].values
    valid_mask = ~np.isnan(vals)
    weighted = np.where(valid_mask, vals * caps, 0.0)
    weight_sums = np.where(valid_mask, caps, 0.0).sum(axis=1)
    result = np.where(weight_sums > 0, weighted.sum(axis=1) / weight_sums, np.nan)
    return pd.Series(result, index=df_sim.index)


def _country_level_metrics(df_sim: pd.DataFrame, df_obs: pd.DataFrame,
                           turb_info: pd.DataFrame) -> tuple[float, float, float]:
    """Compute MAE, RMSE, MBE for country-level obs vs aggregated sim.

    Args:
        df_sim: Grid-level simulation with 'time' and per-grid columns.
        df_obs: Country-level observation with 'time' and 'obs' columns.
        turb_info: Turbine info for capacity weighting.

    Returns:
        Tuple of (rmse, mae, mbe).
    """
    sim = df_sim.copy()
    obs = df_obs.copy()
    sim['time'] = pd.to_datetime(sim['time'])
    obs['time'] = pd.to_datetime(obs['time'])

    # Aggregate sim to country level
    sim['cf_sim'] = _aggregate_sim_to_country(sim, turb_info)

    # Aggregate both to monthly
    sim['month'] = sim['time'].dt.month
    sim['year'] = sim['time'].dt.year
    sim_monthly = sim.groupby(['year', 'month'])['cf_sim'].mean().reset_index()

    obs['month'] = obs['time'].dt.month
    obs['year'] = obs['time'].dt.year
    obs_monthly = obs.groupby(['year', 'month'])['obs'].mean().reset_index()
    obs_monthly = obs_monthly.rename(columns={'obs': 'cf_obs'})

    merged = pd.merge(sim_monthly, obs_monthly, on=['year', 'month'])
    merged = merged.dropna(subset=['cf_sim', 'cf_obs'])

    if len(merged) == 0:
        return np.nan, np.nan, np.nan

    diff = merged['cf_sim'] - merged['cf_obs']
    rmse = np.sqrt((diff ** 2).mean())
    mae = np.abs(diff).mean()
    mbe = diff.mean()
    return rmse, mae, mbe


def evaluate_run(run_dir: Path) -> list[dict]:
    """Evaluate a single PyVWF run directory.

    For turbine-level obs, uses vwf.metrics.overall_error.
    For country-level obs, aggregates grid simulations to country level first.

    Args:
        run_dir: Path to PyVWF run directory.

    Returns:
        List of dictionaries with evaluation metrics for each correction variant.
    """
    metadata = parse_run_directory(run_dir)
    results_dir = run_dir / "results" / "capacity-factor"

    if not results_dir.exists():
        print(f"  ⚠ No results directory found: {results_dir}")
        return []

    # Find observation file to extract year
    obs_files = list(results_dir.glob("*_obs_cf.csv"))
    if not obs_files:
        print(f"  ⚠ No observation file found in {results_dir}")
        return []

    obs_file = obs_files[0]
    print(f"  Loading observations: {obs_file.name}")

    # Extract year from filename
    year_match = re.search(r"_(\d{4})_", obs_file.name)
    year = int(year_match.group(1)) if year_match else None

    # Load turbine info for capacity weighting
    info_dir = run_dir / "training" / "simulated-turbines"
    candidates = []
    if year is not None:
        candidates.append(info_dir / f"{metadata['country']}_{year}_turb_info.csv")
    candidates.append(info_dir / f"{metadata['country']}_train_turb_info.csv")

    turb_info_path = next((path for path in candidates if path.exists()), None)
    if turb_info_path is None:
        print(f"  ⚠ No turbine info file found for capacity weighting")
        return []

    turb_info = pd.read_csv(turb_info_path)
    print(f"  Using turbine info: {turb_info_path.name}")

    # Determine cluster counts and time resolutions from available files
    cor_files = list(results_dir.glob("*_cor_cf.csv"))
    cluster_set = set()
    time_res_set = set()

    for cor_file in cor_files:
        match = re.search(r"_(\w+)_(\d+)_cor_cf\.csv", cor_file.name)
        if match:
            time_res, n_clusters = match.groups()
            # Strip year prefix from time_res if present (e.g., "2020_fixed" -> "fixed")
            time_res_clean = re.sub(r'^\d{4}_', '', time_res)
            cluster_set.add(int(n_clusters))
            time_res_set.add(time_res_clean)

    if not cluster_set or not time_res_set:
        print(f"  ⚠ Could not parse cluster counts or time resolutions from filenames")
        return []

    cluster_list = sorted(list(cluster_set))
    time_res_list = sorted(list(time_res_set))

    print(f"  Found clusters: {cluster_list}")
    print(f"  Found time resolutions: {time_res_list}")

    # Country-level obs: aggregate sim to country level and compute metrics directly
    if metadata['obs_level'] == 'country':
        obs_df = pd.read_csv(obs_file)
        results = []

        # Uncorrected
        unc_file = results_dir / f"{metadata['country']}_{year}_unc_cf.csv"
        if unc_file.exists():
            unc_df = pd.read_csv(unc_file)
            rmse, mae, mbe = _country_level_metrics(unc_df, obs_df, turb_info)
            results.append({
                **metadata, "year": year,
                "correction_type": "uncorrected", "time_res": None,
                "n_clusters": None, "mae": mae, "rmse": rmse,
                "r2": np.nan, "bias": mbe, "rel_bias": np.nan,
                "mean_obs": np.nan, "mean_sim": np.nan, "n_points": np.nan,
            })

        # Corrected variants
        for num_clu in cluster_list:
            for time_res in time_res_list:
                cor_file = results_dir / f"{metadata['country']}_{year}_{time_res}_{num_clu}_cor_cf.csv"
                if cor_file.exists():
                    cor_df = pd.read_csv(cor_file)
                    rmse, mae, mbe = _country_level_metrics(cor_df, obs_df, turb_info)
                    results.append({
                        **metadata, "year": year,
                        "correction_type": "corrected", "time_res": time_res,
                        "n_clusters": num_clu, "mae": mae, "rmse": rmse,
                        "r2": np.nan, "bias": mbe, "rel_bias": np.nan,
                        "mean_obs": np.nan, "mean_sim": np.nan, "n_points": np.nan,
                    })

        return results

    # Turbine-level obs: use overall_error as before
    try:
        metrics_df = overall_error(
            'total',
            str(run_dir.resolve()),
            metadata['country'],
            turb_info,
            cluster_list,
            time_res_list,
            False,
            year
        )
    except Exception as e:
        print(f"  ⚠ Error calculating metrics: {e}")
        return []

    # Convert overall_error output to expected format
    results = []
    for _, row in metrics_df.iterrows():
        result = {
            **metadata,
            "year": year,
            "correction_type": "uncorrected" if pd.isna(row['time_res']) else "corrected",
            "time_res": row['time_res'] if not pd.isna(row['time_res']) else None,
            "n_clusters": row['num_clu'] if row['num_clu'] != 1 or not pd.isna(row['time_res']) else None,
            "mae": row['mae'],
            "rmse": row['rmse'],
            "r2": np.nan,  # overall_error doesn't return R²
            "bias": row['mbe'],
            "rel_bias": np.nan,  # overall_error doesn't return rel_bias
            "mean_obs": np.nan,  # overall_error doesn't return these
            "mean_sim": np.nan,
            "n_points": np.nan,
        }
        results.append(result)

    return results



def find_all_runs(base_dir: Path) -> list[Path]:
    """Find all PyVWF run directories.

    Args:
        base_dir: Base output directory (output/run).

    Returns:
        List of paths to run directories.
    """
    run_dirs = []

    if not base_dir.exists():
        print(f"Base directory not found: {base_dir}")
        return run_dirs

    for item in base_dir.iterdir():
        if item.is_dir() and "obs_" in item.name:
            run_dirs.append(item)

    return sorted(run_dirs)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Evaluate all PyVWF runs and output metrics to CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV file path (defaults to <base-dir>/pyvwf_evaluation_metrics.csv)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("output/runs"),
        help="Base directory containing PyVWF runs",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="Prefix folder under base-dir (e.g. output/runs/<prefix>)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("PyVWF Evaluation Metrics")
    print("=" * 80)

    # Find all runs
    runs_dir = args.base_dir / args.prefix

    if args.output is None:
        args.output = runs_dir / "pyvwf_evaluation_metrics.csv"

    print(f"\nSearching for runs in: {runs_dir}")
    run_dirs = find_all_runs(runs_dir)

    if not run_dirs:
        print("\n✗ No PyVWF runs found!")
        return 1

    print(f"\n✓ Found {len(run_dirs)} runs:")
    for run_dir in run_dirs:
        print(f"  - {run_dir.name}")

    # Evaluate all runs
    print("\n" + "-" * 80)
    print("Evaluating runs...")
    print("-" * 80)

    all_results = []

    for run_dir in run_dirs:
        print(f"\n{run_dir.name}:")
        results = evaluate_run(run_dir)
        all_results.extend(results)

    if not all_results:
        print("\n✗ No results collected!")
        return 1

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Reorder columns for clarity
    column_order = [
        "country",
        "mode",
        "obs_level",
        "year",
        "correction_type",
        "time_res",
        "n_clusters",
        "mae",
        "rmse",
        "r2",
        "bias",
        "rel_bias",
        "mean_obs",
        "mean_sim",
        "n_points",
        "run_name",
    ]

    df = df[column_order]

    # Sort by country, obs_level, correction_type
    df = df.sort_values(["country", "obs_level", "correction_type", "time_res"])

    # Save to CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print(f"\nNote: Metrics calculated using vwf.metrics.overall_error(type='total')")
    print(f"Total evaluations: {len(df)}")
    print(f"\nCountries: {sorted(df['country'].unique())}")
    print(f"Observation levels: {sorted(df['obs_level'].unique())}")
    print(f"Correction types: {sorted(df['correction_type'].unique())}")

    # Summary statistics
    print("\n" + "-" * 80)
    print("Mean Metrics by Correction Type:")
    print("-" * 80)

    summary = df.groupby(["correction_type", "obs_level"]).agg({
        "mae": "mean",
        "rmse": "mean",
        "r2": "mean",
        "bias": "mean",
        "rel_bias": "mean",
    }).round(4)

    print(summary)

    print(f"\n✓ Results saved to: {args.output}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
