#!/usr/bin/env python3
"""Evaluate grid-interpolated bias corrections at observation locations.

Validates gridded bias corrections by simulating capacity factors at
observation locations and comparing against observed CFs, uncorrected
CFs, and cluster-based CFs.

For turbine-level countries (DK, DE, UK): computes per-turbine CFs
and evaluates using capacity-weighted metrics matching the existing
PyVWF evaluation pipeline.

For country-level countries (NL, FR, BE, etc.): computes per-grid-point
CFs, aggregates to country level, and compares to country-level obs.

Usage:
    # Quick test with one country and one method
    python scripts/pyvwf_to_grid/evaluate_grid_corrections.py \\
        --countries DK --methods idw

    # Full evaluation (all countries, IDW + Kriging)
    python scripts/pyvwf_to_grid/evaluate_grid_corrections.py

    # Custom paths
    python scripts/pyvwf_to_grid/evaluate_grid_corrections.py \\
        --runs-dir output/runs/turbine_grid \\
        --grid-dir output/pyvwf_to_grid/grid_comparison \\
        --methods idw kriging \\
        --output-dir output/pyvwf_to_grid/grid_evaluation
"""

import argparse
import re
import sys
import time as time_module
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from vwf.datasets.era5 import prep_era5
from vwf.wind import interpolate_wind, _get_power_curve_cache
from vwf.data import load_power_curves
from vwf.metrics import calculate_error

# Import thesis plotting style
from vwf.viz.style import thesis_plot_style

# Style setup
STYLE = thesis_plot_style()
cm = STYLE['cm']


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_run_directory(run_dir):
    """Parse run directory name -> dict with country, mode, obs_level."""
    match = re.match(r"([A-Z]{2})-([a-z]+)-obs_([a-z]+)-", run_dir.name)
    if not match:
        return None
    c, m, o = match.groups()
    return {'country': c, 'mode': m, 'obs_level': o}


def detect_test_year(run_dir, country):
    """Get test year from observation file name."""
    results_dir = run_dir / "results" / "capacity-factor"
    for f in results_dir.glob(f"{country}_*_obs_cf.csv"):
        m = re.search(r"_(\d{4})_", f.name)
        if m:
            return int(m.group(1))
    return None


def load_turb_info(run_dir, country, year_test):
    """Load turbine info from saved training files."""
    info_dir = run_dir / "training" / "simulated-turbines"
    candidates = [
        info_dir / f"{country}_{year_test}_turb_info.csv",
        info_dir / f"{country}_train_turb_info.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df['ID'] = df['ID'].astype(str)
            for col in ['lat', 'lon', 'height', 'capacity']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['lat', 'lon', 'height', 'capacity']).reset_index(drop=True)
            return df
    return None


def load_existing_cluster_metrics(runs_dir):
    """Load existing pyvwf_evaluation_metrics.csv for comparison."""
    path = runs_dir / "pyvwf_evaluation_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def extract_grid_corrections(grid_ds, turb_info):
    """Extract scalar/offset from gridded corrections at each location.

    Uses bilinear interpolation from the 0.25deg grid to each
    turbine/grid-point lat/lon. Points outside the grid or in
    distance-masked regions get neutral values (scalar=1, offset=0).
    """
    lats = xr.DataArray(turb_info['lat'].values, dims='points')
    lons = xr.DataArray(turb_info['lon'].values, dims='points')

    scalar = grid_ds['scalar'].interp(y=lats, x=lons, method='linear').values
    offset = grid_ds['offset'].interp(y=lats, x=lons, method='linear').values

    df = pd.DataFrame({
        'ID': turb_info['ID'].values,
        'scalar': np.where(np.isnan(scalar), 1.0, scalar),
        'offset': np.where(np.isnan(offset), 0.0, offset),
    })
    return df


def apply_corrections_and_cf(unc_ws, corrections, power_curves):
    """Apply scalar/offset corrections to wind speeds and convert to CFs.

    Args:
        unc_ws: DataArray (time, turbine) with model and capacity coords
                from interpolate_wind().
        corrections: DataFrame with 'ID', 'scalar', 'offset'.
        power_curves: Power curve lookup table.

    Returns:
        DataFrame in wide format (time, turbine_id_1, turbine_id_2, ...).
    """
    turb_ids = unc_ws.turbine.values
    corr_map = corrections.set_index('ID')

    # Build aligned scalar/offset arrays
    s = np.array([corr_map.at[str(tid), 'scalar'] for tid in turb_ids])
    o = np.array([corr_map.at[str(tid), 'offset'] for tid in turb_ids])

    scalars = xr.DataArray(s, dims='turbine', coords={'turbine': turb_ids})
    offsets = xr.DataArray(o, dims='turbine', coords={'turbine': turb_ids})

    # Apply correction at wind speed level
    cor_ws = unc_ws * scalars + offsets

    # Power curve conversion
    _, curve_by_model = _get_power_curve_cache(power_curves)

    def speed_to_cf(da):
        model = da.model[0].item()
        akima = curve_by_model[model]
        vals = np.clip(akima(da.data), 0.0, 1.0)
        return xr.DataArray(vals, coords=da.coords, dims=da.dims)

    cor_cf = cor_ws.groupby("model").map(speed_to_cf)

    # Convert to wide DataFrame matching existing CF CSV format
    df = cor_cf.to_pandas().reset_index()
    df.columns = [str(c) for c in df.columns]
    return df


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(df_sim, df_obs, turb_info, obs_level):
    """Compute MAE, RMSE, bias appropriate to the observation level."""
    if obs_level == 'turbine':
        return _turbine_metrics(df_sim, df_obs, turb_info)
    else:
        return _country_metrics(df_sim, df_obs, turb_info)


def _turbine_metrics(df_sim, df_obs, turb_info):
    """Turbine-level metrics using calculate_error('total').

    Matches the existing PyVWF evaluation methodology:
    capacity-weighted per-turbine MAE, RMSE, bias.
    """
    try:
        rmse, mae, mbe = calculate_error(
            'total', df_sim.copy(), df_obs.copy(), turb_info.copy()
        )
        return {'mae': mae, 'rmse': rmse, 'bias': mbe}
    except Exception as e:
        print(f"      Error computing turbine metrics: {e}")
        return {'mae': np.nan, 'rmse': np.nan, 'bias': np.nan}


def _country_metrics(df_sim, df_obs, turb_info):
    """Country-level: aggregate per-grid-point sim to country, compare to obs.

    Matches the methodology from evaluate_all_pyvwf_runs._country_level_metrics().
    """
    sim = df_sim.copy()
    obs = df_obs.copy()
    sim['time'] = pd.to_datetime(sim['time'])
    obs['time'] = pd.to_datetime(obs['time'])

    # Aggregate sim to country level (capacity-weighted)
    grid_cols = [c for c in sim.columns if c != 'time']
    cap_map = turb_info.set_index('ID')['capacity']
    valid_cols = [c for c in grid_cols if c in cap_map.index]

    if not valid_cols:
        return {'mae': np.nan, 'rmse': np.nan, 'bias': np.nan}

    caps = cap_map[valid_cols].values.astype(float)
    vals = sim[valid_cols].values
    valid_mask = ~np.isnan(vals)
    weighted = np.where(valid_mask, vals * caps, 0.0)
    weight_sums = np.where(valid_mask, caps, 0.0).sum(axis=1)
    sim['cf_sim'] = np.where(weight_sums > 0,
                              weighted.sum(axis=1) / weight_sums, np.nan)

    # Monthly aggregation
    sim['year'] = sim['time'].dt.year
    sim['month'] = sim['time'].dt.month
    sim_m = sim.groupby(['year', 'month'])['cf_sim'].mean().reset_index()

    obs['year'] = obs['time'].dt.year
    obs['month'] = obs['time'].dt.month
    obs_m = obs.groupby(['year', 'month'])['obs'].mean().reset_index()
    obs_m = obs_m.rename(columns={'obs': 'cf_obs'})

    merged = pd.merge(sim_m, obs_m, on=['year', 'month']).dropna()
    if len(merged) == 0:
        return {'mae': np.nan, 'rmse': np.nan, 'bias': np.nan}

    diff = merged['cf_sim'] - merged['cf_obs']
    return {
        'mae': np.abs(diff).mean(),
        'rmse': np.sqrt((diff ** 2).mean()),
        'bias': diff.mean(),
    }


# =============================================================================
# SINGLE RUN EVALUATION
# =============================================================================

def evaluate_single_run(run_dir, methods, grid_dir, existing_metrics=None):
    """Evaluate grid corrections for a single PyVWF run directory.

    Steps:
    1. Load turb_info, observations, existing uncorrected CFs
    2. Compute uncorrected metrics (from saved CFs)
    3. Extract cluster-best metrics from existing evaluation
    4. Load ERA5, compute wind speeds, apply grid corrections, simulate CFs
    5. Compute grid-corrected metrics

    Returns list of result dicts.
    """
    metadata = parse_run_directory(run_dir)
    if metadata is None:
        return []

    country = metadata['country']
    mode = metadata['mode']
    obs_level = metadata['obs_level']

    print(f"\n{'='*60}")
    print(f"  {country}-{mode} ({obs_level}-level)")
    print(f"{'='*60}")

    # Detect test year
    year_test = detect_test_year(run_dir, country)
    if year_test is None:
        print("  ! Cannot detect test year, skipping")
        return []
    print(f"  Test year: {year_test}")

    # Load turb_info
    turb_info = load_turb_info(run_dir, country, year_test)
    if turb_info is None:
        print("  ! Cannot load turbine info, skipping")
        return []
    print(f"  Locations: {len(turb_info)} "
          f"({'turbines' if obs_level == 'turbine' else 'grid points'})")

    # Load observation and uncorrected CF files
    results_dir = run_dir / "results" / "capacity-factor"
    obs_path = results_dir / f"{country}_{year_test}_obs_cf.csv"
    unc_path = results_dir / f"{country}_{year_test}_unc_cf.csv"

    if not obs_path.exists() or not unc_path.exists():
        print(f"  ! Missing obs or unc CF files, skipping")
        return []

    obs_cf = pd.read_csv(obs_path)
    unc_cf = pd.read_csv(unc_path)

    results = []

    # --- 1. Uncorrected metrics (from saved CFs) ---
    print("  [1] Uncorrected metrics...")
    unc_metrics = compute_metrics(unc_cf, obs_cf, turb_info, obs_level)
    results.append({
        **metadata, 'year': year_test,
        'correction_source': 'uncorrected', 'method': '-',
        **unc_metrics,
    })
    print(f"      MAE={unc_metrics['mae']:.4f}  RMSE={unc_metrics['rmse']:.4f}  "
          f"Bias={unc_metrics['bias']:+.4f}")

    # --- 2. Cluster-best metrics (from existing evaluation) ---
    if existing_metrics is not None:
        country_fixed = existing_metrics[
            (existing_metrics['country'] == country) &
            (existing_metrics['mode'] == mode) &
            (existing_metrics['obs_level'] == obs_level) &
            (existing_metrics['correction_type'] == 'corrected') &
            (existing_metrics['time_res'] == 'fixed')
        ]
        if len(country_fixed) > 0:
            best = country_fixed.loc[country_fixed['mae'].idxmin()]
            n_clu = int(best['n_clusters'])
            results.append({
                **metadata, 'year': year_test,
                'correction_source': 'cluster_fixed',
                'method': f'n={n_clu}',
                'mae': best['mae'], 'rmse': best['rmse'], 'bias': best['bias'],
            })
            print(f"  [2] Cluster best (n={n_clu}, fixed): "
                  f"MAE={best['mae']:.4f}  RMSE={best['rmse']:.4f}  "
                  f"Bias={best['bias']:+.4f}")
        else:
            print("  [2] No fixed cluster metrics found for comparison")
    else:
        print("  [2] No existing metrics file for comparison")

    # --- 3. Grid-corrected metrics ---
    # Load ERA5 reanalysis
    print("  [3] Loading ERA5 reanalysis...")
    t0 = time_module.time()
    try:
        reanalysis = prep_era5(country, False, True)  # train=False, calc_z0=True
        reanalysis = reanalysis.sel(time=str(year_test))
    except Exception as e:
        print(f"      ! Failed to load ERA5: {e}")
        return results
    print(f"      ERA5 loaded ({time_module.time() - t0:.0f}s)")

    # Load power curves
    power_curves = load_power_curves()

    # Compute uncorrected wind speeds (shared across grid methods)
    print("  [4] Computing wind speeds...")
    t0 = time_module.time()
    try:
        unc_ws = interpolate_wind(reanalysis, turb_info)
    except Exception as e:
        print(f"      ! Wind interpolation failed: {e}")
        return results
    elapsed = time_module.time() - t0
    print(f"      Wind interpolation done ({elapsed:.0f}s, "
          f"shape={unc_ws.shape})")

    # Evaluate each grid method
    for method in methods:
        print(f"  [5] Grid method: {method.upper()}")
        nc_path = grid_dir / f'europe_corrections_{method}.nc'
        if not nc_path.exists():
            print(f"      ! Grid file not found: {nc_path}")
            continue

        grid_ds = xr.open_dataset(nc_path)

        # Extract corrections at turbine/grid-point locations
        corrections = extract_grid_corrections(grid_ds, turb_info)
        n_non_neutral = ((corrections['scalar'] != 1.0) |
                         (corrections['offset'] != 0.0)).sum()
        print(f"      Corrections: {n_non_neutral}/{len(corrections)} "
              f"non-neutral values")

        # Summary of correction magnitudes
        print(f"      Scalar: mean={corrections['scalar'].mean():.3f}, "
              f"std={corrections['scalar'].std():.3f}, "
              f"range=[{corrections['scalar'].min():.3f}, "
              f"{corrections['scalar'].max():.3f}]")
        print(f"      Offset: mean={corrections['offset'].mean():.3f}, "
              f"std={corrections['offset'].std():.3f}, "
              f"range=[{corrections['offset'].min():.3f}, "
              f"{corrections['offset'].max():.3f}]")

        # Apply corrections and simulate CFs
        t0 = time_module.time()
        try:
            cor_cf_df = apply_corrections_and_cf(unc_ws, corrections,
                                                 power_curves)
        except Exception as e:
            print(f"      ! CF simulation failed: {e}")
            grid_ds.close()
            continue
        print(f"      CF simulation ({time_module.time() - t0:.0f}s)")

        # Compute metrics
        grid_metrics = compute_metrics(cor_cf_df, obs_cf, turb_info,
                                       obs_level)
        results.append({
            **metadata, 'year': year_test,
            'correction_source': 'grid', 'method': method,
            **grid_metrics,
        })
        print(f"      MAE={grid_metrics['mae']:.4f}  "
              f"RMSE={grid_metrics['rmse']:.4f}  "
              f"Bias={grid_metrics['bias']:+.4f}")

        grid_ds.close()

    return results


# =============================================================================
# PLOTTING
# =============================================================================

def generate_comparison_plots(results_df, output_dir):
    """Generate bar chart comparing MAE across correction approaches."""
    print("\nGenerating comparison plots...")

    # Create combined label for country+mode
    results_df = results_df.copy()
    results_df['label'] = results_df.apply(
        lambda r: f"{r['country']}-{r['mode']}"
                  + (f" ({r['obs_level']})" if r['obs_level'] == 'country'
                     else ''),
        axis=1
    )
    labels = results_df['label'].unique()

    # Unique correction source + method pairs (in order)
    pairs = results_df.drop_duplicates(
        subset=['correction_source', 'method']
    )[['correction_source', 'method']].values.tolist()

    # Human-readable names and colors
    pair_names = []
    for src, meth in pairs:
        if src == 'uncorrected':
            pair_names.append('Uncorrected')
        elif src == 'cluster_fixed':
            pair_names.append(f'Cluster ({meth})')
        else:
            pair_names.append(f'Grid {meth.upper()}')

    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12']

    # Bar chart
    n_groups = len(pairs)
    bar_width = 0.8 / max(n_groups, 1)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(16, len(labels) * 3) * cm, 10 * cm))

    for i, ((src, meth), name) in enumerate(zip(pairs, pair_names)):
        subset = results_df[
            (results_df['correction_source'] == src) &
            (results_df['method'] == meth)
        ]
        vals = []
        for lbl in labels:
            row = subset[subset['label'] == lbl]
            vals.append(row['mae'].values[0] if len(row) > 0 else 0)

        offset = i * bar_width - (n_groups - 1) * bar_width / 2
        bars = ax.bar(x + offset, vals, bar_width,
                      label=name, color=colors[i % len(colors)],
                      edgecolor='black', linewidth=0.4, alpha=0.85)

        # Add value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=4,
                        rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('MAE (capacity factor)')
    ax.set_title('Grid vs Cluster Corrections: MAE Comparison',
                 fontweight='bold')
    ax.legend(fontsize=6, frameon=True, framealpha=0.9, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    path = output_dir / 'grid_vs_cluster_comparison.pdf'
    plt.savefig(path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()

    # --- Improvement summary bar chart ---
    # Show % improvement over uncorrected for each country
    unc_rows = results_df[results_df['correction_source'] == 'uncorrected']
    if len(unc_rows) == 0:
        return

    fig, ax = plt.subplots(figsize=(max(16, len(labels) * 3) * cm, 10 * cm))

    corrected_pairs = [(s, m) for s, m in pairs if s != 'uncorrected']
    n_groups = len(corrected_pairs)
    bar_width = 0.8 / max(n_groups, 1)

    for i, (src, meth) in enumerate(corrected_pairs):
        subset = results_df[
            (results_df['correction_source'] == src) &
            (results_df['method'] == meth)
        ]
        improvements = []
        for lbl in labels:
            unc = unc_rows[unc_rows['label'] == lbl]
            cor = subset[subset['label'] == lbl]
            if len(unc) > 0 and len(cor) > 0:
                unc_mae = unc['mae'].values[0]
                cor_mae = cor['mae'].values[0]
                if unc_mae > 0:
                    improvements.append((unc_mae - cor_mae) / unc_mae * 100)
                else:
                    improvements.append(0)
            else:
                improvements.append(0)

        if src == 'cluster_fixed':
            name = f'Cluster ({meth})'
        else:
            name = f'Grid {meth.upper()}'

        offset = i * bar_width - (n_groups - 1) * bar_width / 2
        bars = ax.bar(x + offset, improvements, bar_width,
                      label=name, color=colors[(i + 1) % len(colors)],
                      edgecolor='black', linewidth=0.4, alpha=0.85)

        for bar, val in zip(bars, improvements):
            if val != 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('MAE Improvement over Uncorrected (%)')
    ax.set_title('MAE Improvement: Grid vs Cluster Corrections',
                 fontweight='bold')
    ax.legend(fontsize=6, frameon=True, framealpha=0.9)
    ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    path = output_dir / 'grid_improvement_comparison.pdf'
    plt.savefig(path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Evaluate grid-interpolated corrections at observation "
                    "locations"
    )
    parser.add_argument(
        '--runs-dir', type=str,
        default='output/runs/turbine_grid',
        help='PyVWF runs directory (default: output/runs/turbine_grid)'
    )
    parser.add_argument(
        '--grid-dir', type=str,
        default='output/pyvwf_to_grid/grid_comparison',
        help='Directory with gridded correction NetCDFs'
    )
    parser.add_argument(
        '--methods', nargs='+', default=['idw', 'kriging'],
        help='Grid interpolation methods to evaluate (default: idw kriging)'
    )
    parser.add_argument(
        '--countries', nargs='+', default=None,
        help='Countries to evaluate (default: all available)'
    )
    parser.add_argument(
        '--output-dir', type=str,
        default='output/pyvwf_to_grid/grid_evaluation',
        help='Output directory for results'
    )

    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    grid_dir = Path(args.grid_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GRID CORRECTION EVALUATION")
    print("=" * 70)
    print(f"  Runs directory:  {runs_dir}")
    print(f"  Grid directory:  {grid_dir}")
    print(f"  Methods:         {args.methods}")
    print(f"  Output:          {output_dir}")

    # Find run directories
    run_dirs = sorted([
        d for d in runs_dir.iterdir()
        if d.is_dir() and 'obs_' in d.name
    ])

    if not run_dirs:
        print("\n! No run directories found")
        return 1

    # Filter by country if specified
    if args.countries:
        countries_upper = [c.upper() for c in args.countries]
        run_dirs = [d for d in run_dirs
                    if d.name[:2] in countries_upper]

    print(f"\n  Runs to evaluate ({len(run_dirs)}):")
    for d in run_dirs:
        print(f"    - {d.name}")

    # Load existing cluster-based metrics for comparison
    existing_metrics = load_existing_cluster_metrics(runs_dir)
    if existing_metrics is not None:
        print(f"\n  Loaded {len(existing_metrics)} existing evaluation rows")

    # Evaluate each run
    all_results = []
    t_total = time_module.time()

    for run_dir in run_dirs:
        results = evaluate_single_run(
            run_dir, args.methods, grid_dir, existing_metrics
        )
        all_results.extend(results)

    if not all_results:
        print("\n! No results collected")
        return 1

    # Save results
    results_df = pd.DataFrame(all_results)
    csv_path = output_dir / 'grid_evaluation_metrics.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    print(f"\n{'Country':>8} {'Mode':>10} {'Level':>8} "
          f"{'Source':>15} {'Method':>10} "
          f"{'MAE':>8} {'RMSE':>8} {'Bias':>9}")
    print("-" * 90)
    for _, r in results_df.iterrows():
        print(f"{r['country']:>8} {r['mode']:>10} {r['obs_level']:>8} "
              f"{r['correction_source']:>15} {r['method']:>10} "
              f"{r['mae']:>8.4f} {r['rmse']:>8.4f} {r['bias']:>+9.4f}")

    # Generate plots
    generate_comparison_plots(results_df, output_dir)

    elapsed = time_module.time() - t_total
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  Results CSV: {csv_path}")
    print(f"  Plots:       {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
