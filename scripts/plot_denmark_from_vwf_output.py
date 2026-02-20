"""Generate all Denmark thesis figures from PyVWF output directory.

This script loads data from a PyVWF run output directory and generates
all publication-ready figures for the Denmark wind power paper.

Usage:
    python scripts/plot_denmark_from_vwf_output.py \
        --data_dir output/runs/turbine_dk_research/DK-onshore-obs_turbine-corrected-calc_z0 \
        --output_dir output/runs/turbine_dk_research/plots
"""

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add project root to path for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import thesis plotting style
from plotting_style import thesis_plot_style


# =============================================================================
# GLOBAL STYLE SETUP
# =============================================================================
STYLE = thesis_plot_style()
cm = STYLE['cm']

# Temporal resolution display order and colours
TIME_RES_ORDER = {'fixed': 0, 'season': 1, 'bimonth': 2, 'month': 3}
TIME_RES_LABELS = {'fixed': 'Fixed', 'season': 'Seasonal', 'bimonth': 'Bimonthly', 'month': 'Monthly'}
TIME_RES_COLOURS = {'fixed': '#2E86AB', 'season': '#A23B72', 'bimonth': '#06A77D', 'month': '#D84A05'}


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_vwf_data(data_dir):
    """Load all data from PyVWF output directory.

    Returns:
        dict with keys:
            - train_turbines: DataFrame with training turbine info
            - val_turbines: DataFrame with validation turbine info
            - correction_factors: dict of DataFrames by (n_clu, time_res)
            - capacity_factors: dict of file paths by (n_clu, time_res)
            - results_summary: DataFrame with error metrics
            - mode: 'onshore' or 'offshore' (parsed from dir name)
    """
    data_dir = Path(data_dir)
    print(f"\nLoading data from: {data_dir}")

    data = {}

    # Parse mode from directory name (e.g., DK-onshore-obs_turbine-corrected-calc_z0)
    dir_name = data_dir.name
    if 'offshore' in dir_name:
        data['mode'] = 'offshore'
    else:
        data['mode'] = 'onshore'
    print(f"  Mode: {data['mode']}")

    # Load turbine info files
    train_path = data_dir / "training/simulated-turbines/DK_train_turb_info.csv"
    val_path = data_dir / "training/simulated-turbines/DK_2020_turb_info.csv"

    print("  Loading turbine metadata...")
    if train_path.exists():
        data['train_turbines'] = pd.read_csv(train_path)
        data['train_turbines']['offshore'] = data['train_turbines']['type'] == 'offshore'
        print(f"    Training turbines: {len(data['train_turbines'])}")
    else:
        data['train_turbines'] = None
        print("    ! Training turbines not found")

    if val_path.exists():
        data['val_turbines'] = pd.read_csv(val_path)
        data['val_turbines']['offshore'] = data['val_turbines']['type'] == 'offshore'
        # Mark new turbines
        if data['train_turbines'] is not None:
            train_ids = set(data['train_turbines']['ID'])
            data['val_turbines']['is_new'] = ~data['val_turbines']['ID'].isin(train_ids)
        print(f"    Validation turbines: {len(data['val_turbines'])}")
    else:
        data['val_turbines'] = None
        print("    ! Validation turbines not found")

    # Load correction factors
    print("  Loading correction factors...")
    corrections_dir = data_dir / "training/correction-factors"
    correction_factors = {}

    if corrections_dir.exists():
        for cf_file in corrections_dir.glob("DK_factors_*.csv"):
            # Parse filename: DK_factors_{time_res}_{n_clu}.csv
            parts = cf_file.stem.split('_')
            if len(parts) >= 4:
                time_res = parts[2]
                n_clu = parts[3]

                try:
                    n_clu_int = int(n_clu)
                    df = pd.read_csv(cf_file)
                    correction_factors[(n_clu_int, time_res)] = df
                except ValueError:
                    pass

        print(f"    Loaded {len(correction_factors)} correction factor files")

    data['correction_factors'] = correction_factors

    # Load capacity factors (store paths, not data - files are large)
    print("  Loading capacity factor paths...")
    cf_dir = data_dir / "results/capacity-factor"
    capacity_factors = {}

    if cf_dir.exists():
        for cf_file in cf_dir.glob("DK_2020_*_cor_cf.csv"):
            # Parse filename: DK_2020_{time_res}_{n_clu}_cor_cf.csv
            parts = cf_file.stem.split('_')
            if len(parts) >= 5:
                time_res = parts[2]
                n_clu = parts[3]

                try:
                    n_clu_int = int(n_clu)
                    capacity_factors[(n_clu_int, time_res)] = cf_file
                except ValueError:
                    pass

        print(f"    Found {len(capacity_factors)} capacity factor files")

    data['capacity_factors'] = capacity_factors

    # Load evaluation metrics from parent run directory
    print("  Loading evaluation metrics...")
    data['results_summary'] = load_evaluation_metrics(data_dir, data['mode'])

    return data


def load_evaluation_metrics(data_dir, mode):
    """Load real evaluation metrics from the run's CSV file.

    Looks for pyvwf_evaluation_metrics.csv in the parent run directory.
    """
    data_dir = Path(data_dir)

    # The metrics CSV is at the run prefix level (parent of DK-onshore-...)
    parent_dir = data_dir.parent
    metrics_path = parent_dir / "pyvwf_evaluation_metrics.csv"

    if not metrics_path.exists():
        print(f"    ! Metrics file not found: {metrics_path}")
        return None

    df = pd.read_csv(metrics_path)
    print(f"    Loaded {len(df)} metric rows from {metrics_path.name}")

    # Filter to current mode
    df_mode = df[df['mode'] == mode].copy()
    print(f"    Filtered to {mode}: {len(df_mode)} rows")

    # Separate corrected and uncorrected
    corrected = df_mode[df_mode['time_res'] != 'uncorrected'].copy()
    uncorrected = df_mode[df_mode['time_res'] == 'uncorrected'].copy()

    if len(uncorrected) > 0:
        unc_mae = uncorrected['mae'].values[0]
        unc_rmse = uncorrected['rmse'].values[0]
        unc_bias = uncorrected['bias'].values[0]
        print(f"    Uncorrected baseline: MAE={unc_mae:.4f}, RMSE={unc_rmse:.4f}, Bias={unc_bias:.4f}")

    # Build summary with n_clu as int
    corrected['n_clu'] = corrected['n_clusters'].astype(int)
    corrected = corrected.rename(columns={'time_res': 'time_res', 'mae': 'mae', 'rmse': 'rmse', 'bias': 'bias'})

    # Store uncorrected baseline in the dataframe attrs
    corrected.attrs['uncorrected_mae'] = unc_mae if len(uncorrected) > 0 else np.nan
    corrected.attrs['uncorrected_rmse'] = unc_rmse if len(uncorrected) > 0 else np.nan
    corrected.attrs['uncorrected_bias'] = unc_bias if len(uncorrected) > 0 else np.nan

    return corrected


def load_sensitivity_metrics(runs_dir):
    """Load evaluation metrics from sensitivity run directories and the standard model.

    Looks for pyvwf_evaluation_metrics.csv in each run directory under runs_dir.

    Args:
        runs_dir: Path to the runs directory (e.g., output/runs)

    Returns:
        dict mapping scenario key to {'data': DataFrame, 'label': str}
    """
    runs_dir = Path(runs_dir)

    # Scenario definitions: directory name -> display label
    scenarios = {
        'standard': {
            'dir': 'turbine_dk_research',
            'label': 'Standard model run.',
        },
        'missing_30pct': {
            'dir': 'sensitivity_missing_30pct',
            'label': '(a) 30% of observations removed.',
        },
        'missing_50pct': {
            'dir': 'sensitivity_missing_50pct',
            'label': '(b) 50% of observations removed.',
        },
        'fix_train_ge15se': {
            'dir': 'sensitivity_fix_train_ge15se',
            'label': "(c) Training power curve set to 'GE.1.5se'.",
        },
        'fix_train_vestas': {
            'dir': 'sensitivity_fix_train_vestas_v66',
            'label': "(d) Training power curve set to 'Vestas.V66.2000'.",
        },
        'fix_test_ge15se': {
            'dir': 'sensitivity_fix_test_ge15se',
            'label': "(e) Validation powercurve set to 'GE.1.5se'.",
        },
        'fix_test_vestas': {
            'dir': 'sensitivity_fix_test_vestas_v66',
            'label': "(f) Validation power curve set to 'Vestas.V66.2000'.",
        },
    }

    results = {}
    for key, info in scenarios.items():
        metrics_path = runs_dir / info['dir'] / 'pyvwf_evaluation_metrics.csv'
        if not metrics_path.exists():
            print(f"    ! Metrics not found for {key}: {metrics_path}")
            continue

        df = pd.read_csv(metrics_path)
        # Filter to onshore, corrected only
        df_onshore = df[(df['mode'] == 'onshore') & (df['time_res'] != 'uncorrected')].copy()
        df_onshore['n_clu'] = df_onshore['n_clusters'].astype(int)

        results[key] = {
            'data': df_onshore,
            'label': info['label'],
        }
        print(f"    Loaded {key}: {len(df_onshore)} rows")

    return results


def load_country_shapefile():
    """Load Denmark country boundaries if available."""
    shape_path = "input/regions/country_shapes.geojson"
    if Path(shape_path).exists():
        gdf = gpd.read_file(shape_path)
        # Try 'name' or 'country' column
        country_col = 'name' if 'name' in gdf.columns else 'country'
        dk = gdf[gdf[country_col] == 'DK']
        if len(dk) > 0:
            return dk
    return None


def setup_map_axes(ax, denmark_extent=True, show_labels=False):
    """Setup axes for Denmark maps."""
    if denmark_extent:
        ax.set_xlim(7, 16)
        ax.set_ylim(54, 58)

    if show_labels:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linewidth=0.5)


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_turbine_locations(data, output_dir):
    """Figure 1: Turbine locations (training and validation)."""
    print("\n[ 1/11] Plotting turbine locations...")

    train_df = data['train_turbines']
    val_df = data['val_turbines']

    if train_df is None and val_df is None:
        print("  ! Skipping - no turbine data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18*cm, 12*cm))
    dk_shape = load_country_shapefile()

    # LEFT: Training turbines
    if train_df is not None:
        onshore = train_df[~train_df['offshore']]
        offshore = train_df[train_df['offshore']]

        if dk_shape is not None:
            dk_shape.plot(ax=ax1, facecolor='none', edgecolor='gray', linewidth=0.8)

        ax1.scatter(onshore['lon'], onshore['lat'],
                   s=1, c='#2E86AB', alpha=0.6, label='Onshore')
        if len(offshore) > 0:
            ax1.scatter(offshore['lon'], offshore['lat'],
                       s=3, c='#A23B72', alpha=0.6, label='Offshore', marker='^')

        ax1.set_title(f'Training Turbines\n(n={len(train_df)})')
        ax1.legend(loc='upper left', markerscale=3, framealpha=0.9)
        setup_map_axes(ax1)

    # RIGHT: Validation turbines
    if val_df is not None:
        if dk_shape is not None:
            dk_shape.plot(ax=ax2, facecolor='none', edgecolor='gray', linewidth=0.8)

        # Separate by type and new/existing
        existing_onshore = val_df[(~val_df.get('is_new', True)) & (~val_df['offshore'])]
        new_onshore = val_df[val_df.get('is_new', False) & (~val_df['offshore'])]
        existing_offshore = val_df[(~val_df.get('is_new', True)) & (val_df['offshore'])]
        new_offshore = val_df[val_df.get('is_new', False) & (val_df['offshore'])]

        if len(existing_onshore) > 0:
            ax2.scatter(existing_onshore['lon'], existing_onshore['lat'],
                       s=1, c='#2E86AB', alpha=0.4, label='Existing onshore')
        if len(new_onshore) > 0:
            ax2.scatter(new_onshore['lon'], new_onshore['lat'],
                       s=2, c='#06A77D', alpha=0.8, label='New onshore', marker='s')
        if len(existing_offshore) > 0:
            ax2.scatter(existing_offshore['lon'], existing_offshore['lat'],
                       s=3, c='#A23B72', alpha=0.4, label='Existing offshore', marker='^')
        if len(new_offshore) > 0:
            ax2.scatter(new_offshore['lon'], new_offshore['lat'],
                       s=4, c='#D84A05', alpha=0.8, label='New offshore', marker='D')

        n_new = val_df.get('is_new', pd.Series([False])).sum()
        ax2.set_title(f'Validation Turbines (2020)\n(n={len(val_df)}, new={n_new})')
        ax2.legend(loc='upper left', markerscale=3, framealpha=0.9, fontsize=5)
        setup_map_axes(ax2)

    plt.tight_layout()
    output_path = output_dir / 'fig1_turbine_locations.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_clustering_overview(data, output_dir):
    """Figure 2: Clustering visualization at different resolutions."""
    print("\n[ 2/11] Plotting clustering overview...")

    train_df = data['train_turbines']
    if train_df is None:
        print("  ! Skipping - no training data")
        return

    onshore = train_df[~train_df['offshore']]
    coords = onshore[['lon', 'lat']].values

    n_clusters_list = [2, 10, 100]
    fig, axes = plt.subplots(1, 3, figsize=(20*cm, 10*cm))
    dk_shape = load_country_shapefile()

    for idx, n_clu in enumerate(n_clusters_list):
        ax = axes[idx]

        kmeans = KMeans(n_clusters=n_clu, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)

        if dk_shape is not None:
            dk_shape.plot(ax=ax, facecolor='none', edgecolor='gray', linewidth=0.8)

        ax.scatter(coords[:, 0], coords[:, 1],
                           c=labels, s=2, cmap='tab20', alpha=0.7)

        centroids = kmeans.cluster_centers_
        ax.scatter(centroids[:, 0], centroids[:, 1],
                  c='red', s=20, marker='x', linewidths=2, alpha=0.8)

        ax.set_title(f'n_clusters = {n_clu}')
        setup_map_axes(ax)

    plt.tight_layout()
    output_path = output_dir / 'fig2_clustering_overview.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_clustering_heuristics(data, output_dir):
    """Figure 3: Elbow method and silhouette score."""
    print("\n[ 3/11] Plotting clustering heuristics...")

    train_df = data['train_turbines']
    if train_df is None:
        print("  ! Skipping - no training data")
        return

    onshore = train_df[~train_df['offshore']]
    coords = onshore[['lon', 'lat']].values

    # Range of clusters from available correction factors
    available_n_clu = sorted(set([k[0] for k in data['correction_factors'].keys()]))
    n_clusters_range = [n for n in available_n_clu if n >= 2 and n <= len(coords)]

    if len(n_clusters_range) == 0:
        print("  ! No valid cluster ranges found")
        return

    sse_values = []
    silhouette_values = []

    print(f"  Computing metrics for {len(n_clusters_range)} cluster sizes...")
    for n_clu in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clu, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords)

        sse = kmeans.inertia_
        sse_values.append(sse)

        # Silhouette score (sample if too large)
        if len(coords) > 5000:
            sample_idx = np.random.choice(len(coords), 5000, replace=False)
            sil = silhouette_score(coords[sample_idx], labels[sample_idx])
        else:
            sil = silhouette_score(coords, labels)
        silhouette_values.append(sil)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18*cm, 9*cm))

    # Elbow plot
    ax1.plot(n_clusters_range, sse_values, 'o-', linewidth=STYLE['lw'],
            markersize=STYLE['ms'], color='#2E86AB')
    ax1.set_xlabel('Number of Clusters (n_clu)')
    ax1.set_ylabel('Sum of Squared Distance (SSE)')
    ax1.set_title('Elbow Method')

    # Mark elbow region
    if 500 in n_clusters_range and 700 in n_clusters_range:
        ax1.axvspan(500, 700, color='red', alpha=0.1)
        ax1.text(600, max(sse_values)*0.9, 'Elbow\nRegion', ha='center', color='red')

    # Silhouette plot
    ax2.plot(n_clusters_range, silhouette_values, 'o-', linewidth=STYLE['lw'],
            markersize=STYLE['ms'], color='#A23B72')
    ax2.set_xlabel('Number of Clusters (n_clu)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'fig3_clustering_heuristics.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_correction_factors_spatial(data, output_dir):
    """Figure 4: Spatial distribution of correction factors."""
    print("\n[ 4/11] Plotting correction factors...")

    # Use fixed_700 as example (good balance of spatial resolution)
    key = (700, 'fixed')
    if key not in data['correction_factors']:
        # Try to find closest match
        available = [(k, abs(k[0]-700)) for k in data['correction_factors'].keys() if k[1] == 'fixed']
        if available:
            key = min(available, key=lambda x: x[1])[0]
        else:
            print("  ! No suitable correction factors found")
            return

    cf_df = data['correction_factors'][key]
    train_df = data['train_turbines']

    if train_df is None:
        print("  ! No training data for cluster locations")
        return

    # Get cluster locations by computing kmeans on training data
    onshore = train_df[~train_df['offshore']]
    coords = onshore[['lon', 'lat']].values

    n_clu = key[0]
    kmeans = KMeans(n_clusters=n_clu, random_state=42, n_init=10)
    kmeans.fit(coords)

    # Create DataFrame with cluster centroids and correction factors
    centroids = kmeans.cluster_centers_
    cf_with_coords = cf_df.copy()
    cf_with_coords['lon'] = centroids[:, 0]
    cf_with_coords['lat'] = centroids[:, 1]

    fig = plt.figure(figsize=(20*cm, 9*cm))
    dk_shape = load_country_shapefile()

    # (a) Scalar vs offset
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(cf_with_coords['scalar'], cf_with_coords['offset'],
               s=10, alpha=0.5, c='#2E86AB')
    ax1.set_xlabel('Scalar')
    ax1.set_ylabel('Offset (m/s)')
    ax1.set_title('(a) Scalar vs Offset')
    ax1.axvline(1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # (b) Spatial map of scalars
    ax2 = fig.add_subplot(1, 3, 2)
    if dk_shape is not None:
        dk_shape.plot(ax=ax2, facecolor='none', edgecolor='gray', linewidth=0.8)

    norm = TwoSlopeNorm(vmin=cf_with_coords['scalar'].min(),
                       vcenter=1.0,
                       vmax=cf_with_coords['scalar'].max())
    scatter = ax2.scatter(cf_with_coords['lon'], cf_with_coords['lat'],
                        c=cf_with_coords['scalar'], s=10,
                        cmap='RdBu_r', norm=norm, alpha=0.7, edgecolors='black', linewidths=0.3)
    cbar = plt.colorbar(scatter, ax=ax2, pad=0.02, aspect=20)
    cbar.set_label('Scalar')
    cbar.ax.axhline(1.0, color='black', linewidth=1.5, linestyle='--')

    ax2.set_title(f'(b) Scalar (n={n_clu})')
    setup_map_axes(ax2)

    # (c) Spatial map of offsets
    ax3 = fig.add_subplot(1, 3, 3)
    if dk_shape is not None:
        dk_shape.plot(ax=ax3, facecolor='none', edgecolor='gray', linewidth=0.8)

    norm = TwoSlopeNorm(vmin=cf_with_coords['offset'].min(),
                       vcenter=0.0,
                       vmax=cf_with_coords['offset'].max())
    scatter = ax3.scatter(cf_with_coords['lon'], cf_with_coords['lat'],
                        c=cf_with_coords['offset'], s=10,
                        cmap='RdBu_r', norm=norm, alpha=0.7, edgecolors='black', linewidths=0.3)
    cbar = plt.colorbar(scatter, ax=ax3, pad=0.02, aspect=20)
    cbar.set_label('Offset (m/s)')
    cbar.ax.axhline(0.0, color='black', linewidth=1.5, linestyle='--')

    ax3.set_title(f'(c) Offset (n={n_clu})')
    setup_map_axes(ax3)

    plt.tight_layout()
    output_path = output_dir / 'fig4_correction_factors.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_correction_factors_distribution(data, output_dir):
    """Figure 5: Distribution of correction factors across cluster counts."""
    print("\n[ 5/11] Plotting correction factor distributions...")

    time_res = 'fixed'

    # Pick 4 representative n_clu values from available data
    available_fixed = sorted([k[0] for k in data['correction_factors'].keys()
                              if k[1] == time_res and k[0] >= 2])
    if len(available_fixed) == 0:
        print("  ! No fixed correction factors found")
        return

    # Select ~evenly spaced subset of 4
    if len(available_fixed) <= 4:
        n_clu_list = available_fixed
    else:
        indices = np.linspace(0, len(available_fixed) - 1, 4, dtype=int)
        n_clu_list = [available_fixed[i] for i in indices]

    nplots = len(n_clu_list)
    ncols = min(nplots, 2)
    nrows = (nplots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16*cm, 7*nrows*cm))
    if nplots == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for idx, n_clu in enumerate(n_clu_list):
        key = (n_clu, time_res)
        if key not in data['correction_factors']:
            continue

        cf_df = data['correction_factors'][key]
        ax = axes[idx]

        # Histogram of scalars and offsets
        ax2 = ax.twinx()

        ax.hist(cf_df['scalar'], bins=20, alpha=0.6, color='#2E86AB', label='Scalar')
        ax2.hist(cf_df['offset'], bins=20, alpha=0.6, color='#A23B72', label='Offset')

        ax.axvline(1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        ax.set_xlabel('Value')
        ax.set_ylabel('Count (Scalar)', color='#2E86AB')
        ax2.set_ylabel('Count (Offset)', color='#A23B72')
        ax.set_title(f'n_clusters = {n_clu}')

        # Statistics
        s_std = cf_df['scalar'].std()
        o_std = cf_df['offset'].std()
        stats_text = (f'Scalar: \u03bc={cf_df["scalar"].mean():.2f}, \u03c3={s_std:.2f}\n'
                     f'Offset: \u03bc={cf_df["offset"].mean():.2f}, \u03c3={o_std:.2f}')
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=5)

    # Hide unused axes
    for idx in range(len(n_clu_list), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    output_path = output_dir / 'fig5_correction_factor_distributions.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_temporal_slicing_overview(data, output_dir):
    """Figure 6: Temporal slicing scheme."""
    print("\n[ 6/11] Plotting temporal slicing overview...")

    fig, axes = plt.subplots(4, 1, figsize=(16*cm, 12*cm))

    time_res_schemes = {
        'fixed': [list(range(1, 13))],
        'season': [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        'bimonth': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
        'month': [[i] for i in range(1, 13)]
    }

    colors_scheme = ['#2E86AB', '#A23B72', '#06A77D', '#D84A05', '#F18F01', '#C73E1D']

    for idx, (tres, scheme) in enumerate(time_res_schemes.items()):
        ax = axes[idx]

        for group_idx, months in enumerate(scheme):
            color = colors_scheme[group_idx % len(colors_scheme)]
            for month in months:
                ax.barh(0, 1, left=month-1, height=0.5, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)

        ax.set_xlim(0, 12)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks(np.arange(0.5, 12, 1))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticks([])
        ax.set_ylabel(TIME_RES_LABELS.get(tres, tres), rotation=0, ha='right', va='center')

        if idx == 3:
            ax.set_xlabel('Month')
        else:
            ax.set_xticklabels([])

    plt.suptitle('Temporal Slicing Schemes', fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / 'fig6_temporal_slicing.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_error_vs_clusters(data, output_dir):
    """Figure 7: MAE and RMSE vs number of clusters for all temporal resolutions."""
    print("\n[ 7/11] Plotting error vs clusters...")

    summary = data['results_summary']
    if summary is None or len(summary) == 0:
        print("  ! No evaluation metrics, skipping")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18*cm, 9*cm))

    # Get available time resolutions (sorted)
    time_res_vals = sorted(summary['time_res'].unique(),
                           key=lambda x: TIME_RES_ORDER.get(x, 99))

    # Uncorrected baselines
    unc_mae = summary.attrs.get('uncorrected_mae', np.nan)
    unc_rmse = summary.attrs.get('uncorrected_rmse', np.nan)

    for tres in time_res_vals:
        subset = summary[summary['time_res'] == tres].sort_values('n_clu')
        colour = TIME_RES_COLOURS.get(tres, '#999999')
        label = TIME_RES_LABELS.get(tres, tres)

        ax1.plot(subset['n_clu'], subset['mae'], 'o-',
                color=colour, linewidth=STYLE['lw'], markersize=3,
                label=label, alpha=0.85)
        ax2.plot(subset['n_clu'], subset['rmse'], 'o-',
                color=colour, linewidth=STYLE['lw'], markersize=3,
                label=label, alpha=0.85)

    # Uncorrected baselines
    if not np.isnan(unc_mae):
        ax1.axhline(unc_mae, color='red', linestyle='--', linewidth=1.2, alpha=0.7,
                    label=f'Uncorrected ({unc_mae:.4f})')
    if not np.isnan(unc_rmse):
        ax2.axhline(unc_rmse, color='red', linestyle='--', linewidth=1.2, alpha=0.7,
                    label=f'Uncorrected ({unc_rmse:.4f})')

    for ax, metric in [(ax1, 'MAE'), (ax2, 'RMSE')]:
        ax.set_xscale('log')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel(f'{metric} (capacity factor)')
        ax.set_title(f'{metric} vs Spatial Resolution')
        ax.legend(fontsize=6, frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    # Find and annotate best configuration
    best_idx = summary['mae'].idxmin()
    best = summary.loc[best_idx]
    fig.suptitle(
        f'Denmark {data["mode"].capitalize()} Error Analysis\n'
        f'Best: n={int(best["n_clu"])}, {best["time_res"]} '
        f'(MAE={best["mae"]:.4f})',
        fontsize=8, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'fig7_error_vs_clusters.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_temporal_comparison(data, output_dir):
    """Figure 8: Heatmap of MAE across n_clusters and temporal resolutions."""
    print("\n[ 8/11] Plotting temporal comparison heatmap...")

    summary = data['results_summary']
    if summary is None or len(summary) == 0:
        print("  ! No evaluation metrics, skipping")
        return

    # Pivot to create heatmap matrix
    time_res_vals = sorted(summary['time_res'].unique(),
                           key=lambda x: TIME_RES_ORDER.get(x, 99))
    n_clu_vals = sorted(summary['n_clu'].unique())

    # Create pivot table
    pivot = summary.pivot_table(values='mae', index='n_clu', columns='time_res')
    # Reorder columns
    pivot = pivot[[t for t in time_res_vals if t in pivot.columns]]

    fig, ax = plt.subplots(figsize=(12*cm, max(10, len(n_clu_vals) * 0.5)*cm))

    # Plot heatmap
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')

    # Annotate cells
    for i in range(len(n_clu_vals)):
        for j in range(len(time_res_vals)):
            if j < pivot.shape[1]:
                val = pivot.values[i, j]
                if not np.isnan(val):
                    # Highlight best per row
                    row_min = np.nanmin(pivot.values[i, :])
                    fontweight = 'bold' if val == row_min else 'normal'
                    ax.text(j, i, f'{val:.4f}', ha='center', va='center',
                           fontsize=5, fontweight=fontweight,
                           color='white' if val > np.nanmedian(pivot.values) else 'black')

    ax.set_xticks(range(len(time_res_vals)))
    ax.set_xticklabels([TIME_RES_LABELS.get(t, t) for t in time_res_vals], fontsize=7)
    ax.set_yticks(range(len(n_clu_vals)))
    ax.set_yticklabels(n_clu_vals, fontsize=6)

    ax.set_xlabel('Temporal Resolution')
    ax.set_ylabel('Number of Clusters')
    ax.set_title(f'MAE Heatmap ({data["mode"].capitalize()})', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label('MAE', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Add uncorrected reference
    unc_mae = summary.attrs.get('uncorrected_mae', np.nan)
    if not np.isnan(unc_mae):
        ax.text(1.0, -0.08, f'Uncorrected MAE: {unc_mae:.4f}',
               transform=ax.transAxes, ha='right', fontsize=6,
               color='red', fontstyle='italic')

    plt.tight_layout()
    output_path = output_dir / 'fig8_temporal_comparison.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_summary_figure(data, output_dir):
    """Figure 9: Overview summary figure."""
    print("\n[ 9/11] Creating summary figure...")

    train_df = data['train_turbines']
    val_df = data['val_turbines']

    if train_df is None:
        print("  ! Skipping - no training data")
        return

    fig = plt.figure(figsize=(20*cm, 16*cm))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    dk_shape = load_country_shapefile()

    # (a) Training turbines
    ax1 = fig.add_subplot(gs[0, 0])
    if dk_shape is not None:
        dk_shape.plot(ax=ax1, facecolor='none', edgecolor='gray', linewidth=0.8)
    onshore = train_df[~train_df['offshore']]
    ax1.scatter(onshore['lon'], onshore['lat'], s=1, c='#2E86AB', alpha=0.6)
    ax1.set_title(f'(a) Training\nn={len(train_df)}')
    setup_map_axes(ax1)

    # (b) Validation turbines
    ax2 = fig.add_subplot(gs[0, 1])
    if dk_shape is not None:
        dk_shape.plot(ax=ax2, facecolor='none', edgecolor='gray', linewidth=0.8)
    if val_df is not None:
        onshore_val = val_df[~val_df['offshore']]
        ax2.scatter(onshore_val['lon'], onshore_val['lat'], s=1, c='#06A77D', alpha=0.6)
        ax2.set_title(f'(b) Validation\nn={len(val_df)}')
    setup_map_axes(ax2)

    # (c) Example clustering
    ax3 = fig.add_subplot(gs[0, 2])
    coords = onshore[['lon', 'lat']].values
    kmeans = KMeans(n_clusters=100, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    if dk_shape is not None:
        dk_shape.plot(ax=ax3, facecolor='none', edgecolor='gray', linewidth=0.8)
    ax3.scatter(coords[:, 0], coords[:, 1], c=labels, s=1, cmap='tab20', alpha=0.7)
    ax3.set_title('(c) Spatial Clusters\nn=100')
    setup_map_axes(ax3)

    # (d) Correction factors scalar
    ax4 = fig.add_subplot(gs[1, 0])
    key = (700, 'fixed') if (700, 'fixed') in data['correction_factors'] else list(data['correction_factors'].keys())[0]
    cf_df = data['correction_factors'][key]
    n_clu = key[0]

    kmeans_cf = KMeans(n_clusters=n_clu, random_state=42, n_init=10)
    kmeans_cf.fit(coords)
    centroids = kmeans_cf.cluster_centers_

    if dk_shape is not None:
        dk_shape.plot(ax=ax4, facecolor='none', edgecolor='gray', linewidth=0.8)

    norm = TwoSlopeNorm(vmin=cf_df['scalar'].min(), vcenter=1.0, vmax=cf_df['scalar'].max())
    scatter = ax4.scatter(centroids[:, 0], centroids[:, 1],
                         c=cf_df['scalar'], s=10, cmap='RdBu_r', norm=norm,
                         alpha=0.7, edgecolors='black', linewidths=0.3)
    plt.colorbar(scatter, ax=ax4, label='Scalar', pad=0.02, aspect=15)
    ax4.set_title(f'(d) Correction Scalar\nn={n_clu}')
    setup_map_axes(ax4)

    # (e) Correction factors offset
    ax5 = fig.add_subplot(gs[1, 1])
    if dk_shape is not None:
        dk_shape.plot(ax=ax5, facecolor='none', edgecolor='gray', linewidth=0.8)

    norm = TwoSlopeNorm(vmin=cf_df['offset'].min(), vcenter=0.0, vmax=cf_df['offset'].max())
    scatter = ax5.scatter(centroids[:, 0], centroids[:, 1],
                         c=cf_df['offset'], s=10, cmap='RdBu_r', norm=norm,
                         alpha=0.7, edgecolors='black', linewidths=0.3)
    plt.colorbar(scatter, ax=ax5, label='Offset (m/s)', pad=0.02, aspect=15)
    ax5.set_title(f'(e) Correction Offset\nn={n_clu}')
    setup_map_axes(ax5)

    # (f) Scalar vs offset
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(cf_df['scalar'], cf_df['offset'], s=5, alpha=0.5, c='#2E86AB')
    ax6.axvline(1, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax6.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax6.set_xlabel('Scalar')
    ax6.set_ylabel('Offset (m/s)')
    ax6.set_title('(f) Factor Correlation')

    plt.suptitle('Denmark Wind Power Modelling: Overview', fontweight='bold', y=0.98)

    output_path = output_dir / 'fig9_summary_overview.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_sensitivity_analysis(data, output_dir, runs_dir):
    """Figure 10: Sensitivity analysis of data quality on model performance.

    Reproduces Paper Fig 10: 3x3 grid with 7 subplots showing RMSE vs n_clusters
    for the standard model and 6 data quality scenarios.

    Layout:
        Row 0: [legend],            (a) 30% missing,       (b) 50% missing
        Row 1: Standard model,      (c) Train GE.1.5se,    (d) Train Vestas.V66.2000
        Row 2: [empty],             (e) Val GE.1.5se,      (f) Val Vestas.V66.2000
    """
    print("\n[10/11] Plotting sensitivity analysis...")

    # Load sensitivity metrics from run directories
    print("  Loading sensitivity metrics...")
    sensitivity_data = load_sensitivity_metrics(runs_dir)

    if len(sensitivity_data) == 0:
        print("  ! No sensitivity data found, skipping")
        return

    # Subplot positions for each scenario
    subplot_map = {
        'standard':           (1, 0),
        'missing_30pct':      (0, 1),
        'missing_50pct':      (0, 2),
        'fix_train_ge15se':   (1, 1),
        'fix_train_vestas':   (1, 2),
        'fix_test_ge15se':    (2, 1),
        'fix_test_vestas':    (2, 2),
    }

    # Line styles matching the paper (colour + linestyle for each t_freq)
    time_res_styles = {
        'month':   {'color': '#2E86AB', 'linestyle': '-',  'label': 'Monthly'},
        'bimonth': {'color': '#A23B72', 'linestyle': '--', 'label': 'Bimonthly'},
        'season':  {'color': '#06A77D', 'linestyle': ':',  'label': 'Seasonal'},
        'fixed':   {'color': '#D84A05', 'linestyle': '-.', 'label': 'Fixed'},
    }

    # Determine consistent y-axis limits across all scenarios
    all_rmse = []
    for key in subplot_map:
        if key in sensitivity_data:
            all_rmse.extend(sensitivity_data[key]['data']['rmse'].dropna().tolist())
    if all_rmse:
        y_min = max(0, min(all_rmse) - 0.005)
        y_max = max(all_rmse) + 0.005
    else:
        y_min, y_max = 0.08, 0.17

    # Filter standard model to sensitivity cluster range for comparison consistency
    sensitivity_clusters = [1, 10, 100, 500, 1000]
    if 'standard' in sensitivity_data:
        std_df = sensitivity_data['standard']['data']
        sensitivity_data['standard']['data'] = std_df[std_df['n_clu'].isin(sensitivity_clusters)]

    fig, axes = plt.subplots(3, 3, figsize=(20*cm, 18*cm))

    for key, (row, col) in subplot_map.items():
        ax = axes[row, col]

        if key not in sensitivity_data:
            ax.text(0.5, 0.5, 'Data not available',
                    ha='center', va='center', transform=ax.transAxes, fontsize=7)
            ax.set_xscale('log')
            ax.set_xlim(0.8, 1200)
            ax.set_ylim(y_min, y_max)
            continue

        df = sensitivity_data[key]['data']
        label = sensitivity_data[key]['label']

        for tres, style in time_res_styles.items():
            subset = df[df['time_res'] == tres].sort_values('n_clu')
            if len(subset) > 0:
                ax.plot(subset['n_clu'], subset['rmse'],
                        color=style['color'], linestyle=style['linestyle'],
                        linewidth=STYLE['lw'], markersize=2, marker='o',
                        label=style['label'])

        ax.set_xscale('log')
        ax.set_xlabel(r'Number of Clusters ($n_{\mathrm{clu}}$)', fontsize=6)
        ax.set_ylabel('RMSE', fontsize=6)
        ax.set_title(label, fontsize=6)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(labelsize=5)
        ax.grid(True, alpha=0.3, linewidth=0.5)

    # Hide unused cells
    axes[0, 0].set_visible(False)  # Top-left: will use for legend
    axes[2, 0].set_visible(False)  # Bottom-left: empty

    # Create legend in the top-left cell area
    legend_ax = fig.add_axes(axes[0, 0].get_position(fig))
    legend_ax.set_axis_off()
    lines = []
    labels = []
    for tres, style in time_res_styles.items():
        line, = legend_ax.plot([], [], color=style['color'],
                               linestyle=style['linestyle'],
                               linewidth=STYLE['lw'] * 1.5)
        lines.append(line)
        labels.append(style['label'])
    legend_ax.legend(lines, labels, loc='center', frameon=True,
                     title=r'Temporal Frequency ($t_{\mathrm{freq}}$)',
                     fontsize=7, title_fontsize=7)

    # Add horizontal separator lines between experiment groups
    for y_frac in [0.355, 0.66]:
        fig.add_artist(plt.Line2D([0.02, 0.98], [y_frac, y_frac],
                                  transform=fig.transFigure,
                                  color='black', linewidth=0.5, alpha=0.3))

    plt.tight_layout()
    output_path = output_dir / 'fig10_sensitivity_analysis.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def create_results_summary_figure(data, output_dir):
    """Figure 11: Results summary with error analysis and improvement."""
    print("\n[11/11] Creating results summary figure...")

    summary = data['results_summary']
    if summary is None or len(summary) == 0:
        print("  ! No evaluation metrics, skipping")
        return

    fig = plt.figure(figsize=(20*cm, 16*cm))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    unc_mae = summary.attrs.get('uncorrected_mae', np.nan)
    unc_rmse = summary.attrs.get('uncorrected_rmse', np.nan)

    # (a) MAE vs clusters (best time_res highlighted)
    ax1 = fig.add_subplot(gs[0, 0])
    time_res_vals = sorted(summary['time_res'].unique(),
                           key=lambda x: TIME_RES_ORDER.get(x, 99))
    for tres in time_res_vals:
        subset = summary[summary['time_res'] == tres].sort_values('n_clu')
        colour = TIME_RES_COLOURS.get(tres, '#999')
        ax1.plot(subset['n_clu'], subset['mae'], 'o-',
                color=colour, linewidth=STYLE['lw'], markersize=2,
                label=TIME_RES_LABELS.get(tres, tres), alpha=0.85)

    if not np.isnan(unc_mae):
        ax1.axhline(unc_mae, color='red', linestyle='--', linewidth=1, alpha=0.7)

    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('MAE')
    ax1.set_title('(a) MAE vs Clusters')
    ax1.legend(fontsize=5, frameon=True, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linewidth=0.5)

    # (b) Improvement over uncorrected
    ax2 = fig.add_subplot(gs[0, 1])
    if not np.isnan(unc_mae):
        # Show best MAE per time_res
        best_per_tres = summary.groupby('time_res')['mae'].min()
        best_per_tres = best_per_tres.reindex(time_res_vals)
        improvement = ((unc_mae - best_per_tres) / unc_mae * 100)

        bars = ax2.bar(range(len(time_res_vals)),
                      improvement.values,
                      color=[TIME_RES_COLOURS.get(t, '#999') for t in time_res_vals],
                      edgecolor='black', linewidth=0.4, alpha=0.85)

        for bar, val, mae_val in zip(bars, improvement.values, best_per_tres.values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}%\n({mae_val:.4f})',
                    ha='center', va='bottom', fontsize=5)

        ax2.set_xticks(range(len(time_res_vals)))
        ax2.set_xticklabels([TIME_RES_LABELS.get(t, t) for t in time_res_vals], fontsize=7)
        ax2.set_ylabel('MAE Improvement (%)')
        ax2.set_title(f'(b) Improvement over Uncorrected\n(baseline MAE={unc_mae:.4f})')

    # (c) Best n_clu per time_res
    ax3 = fig.add_subplot(gs[1, 0])
    best_rows = []
    for tres in time_res_vals:
        subset = summary[summary['time_res'] == tres]
        if len(subset) > 0:
            best = subset.loc[subset['mae'].idxmin()]
            best_rows.append({
                'time_res': tres,
                'n_clu': int(best['n_clu']),
                'mae': best['mae'],
                'rmse': best['rmse'],
                'bias': best['bias'],
            })
    best_df = pd.DataFrame(best_rows)

    x = np.arange(len(best_df))
    width = 0.35
    ax3.bar(x - width/2, best_df['mae'], width,
            color=[TIME_RES_COLOURS.get(t, '#999') for t in best_df['time_res']],
            edgecolor='black', linewidth=0.4, label='MAE', alpha=0.85)
    ax3.bar(x + width/2, best_df['rmse'], width,
            color=[TIME_RES_COLOURS.get(t, '#999') for t in best_df['time_res']],
            edgecolor='black', linewidth=0.4, label='RMSE', alpha=0.5)

    ax3.set_xticks(x)
    labels = [f'{TIME_RES_LABELS.get(r["time_res"], r["time_res"])}\nn={r["n_clu"]}'
              for _, r in best_df.iterrows()]
    ax3.set_xticklabels(labels, fontsize=6)
    ax3.set_ylabel('Error')
    ax3.set_title('(c) Best Configuration per Time Res')
    ax3.legend(fontsize=6, frameon=True)

    # (d) Bias analysis
    ax4 = fig.add_subplot(gs[1, 1])
    for tres in time_res_vals:
        subset = summary[summary['time_res'] == tres].sort_values('n_clu')
        colour = TIME_RES_COLOURS.get(tres, '#999')
        ax4.plot(subset['n_clu'], subset['bias'], 'o-',
                color=colour, linewidth=STYLE['lw'], markersize=2,
                label=TIME_RES_LABELS.get(tres, tres), alpha=0.85)

    unc_bias = summary.attrs.get('uncorrected_bias', np.nan)
    if not np.isnan(unc_bias):
        ax4.axhline(unc_bias, color='red', linestyle='--', linewidth=1, alpha=0.7,
                    label=f'Uncorrected ({unc_bias:.4f})')
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    ax4.set_xscale('log')
    ax4.set_xlabel('Number of Clusters')
    ax4.set_ylabel('Mean Bias')
    ax4.set_title('(d) Bias vs Clusters')
    ax4.legend(fontsize=5, frameon=True, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linewidth=0.5)

    # Overall best
    best_idx = summary['mae'].idxmin()
    best = summary.loc[best_idx]
    fig.suptitle(
        f'Denmark {data["mode"].capitalize()}: Results Summary\n'
        f'Best: n={int(best["n_clu"])}, {TIME_RES_LABELS.get(best["time_res"], best["time_res"])} '
        f'(MAE={best["mae"]:.4f}, RMSE={best["rmse"]:.4f})',
        fontsize=8, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    output_path = output_dir / 'fig11_results_summary.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Generate Denmark thesis figures from PyVWF output"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="PyVWF output directory (e.g., output/runs/turbine_dk_research/DK-onshore-...)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for figures (default: <run_prefix>/plots)"
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="output/runs",
        help="Base directory containing all run outputs, for sensitivity analysis (default: output/runs)"
    )

    args = parser.parse_args()

    # Default output dir: parent of data_dir / plots
    if args.output_dir is None:
        args.output_dir = str(Path(args.data_dir).parent / 'plots')

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("DENMARK THESIS FIGURES - From PyVWF Output")
    print("="*80)
    print(f"\nData directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nUsing thesis plotting style:")
    print(f"  - DPI: {STYLE['dpi']}")
    print(f"  - Line width: {STYLE['lw']}")
    print(f"  - Marker size: {STYLE['ms']}")

    # Load all data
    data = load_vwf_data(args.data_dir)

    # Generate all figures
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)

    plot_turbine_locations(data, output_dir)          # Fig 1
    plot_clustering_overview(data, output_dir)         # Fig 2
    plot_clustering_heuristics(data, output_dir)       # Fig 3
    plot_correction_factors_spatial(data, output_dir)   # Fig 4
    plot_correction_factors_distribution(data, output_dir)  # Fig 5
    plot_temporal_slicing_overview(data, output_dir)    # Fig 6
    plot_error_vs_clusters(data, output_dir)            # Fig 7 (NEW)
    plot_temporal_comparison(data, output_dir)           # Fig 8 (NEW)
    create_summary_figure(data, output_dir)             # Fig 9
    plot_sensitivity_analysis(data, output_dir, args.runs_dir)  # Fig 10
    create_results_summary_figure(data, output_dir)     # Fig 11 (NEW)

    print("\n" + "="*80)
    print("ALL FIGURES COMPLETE")
    print("="*80)
    print(f"\nFigures saved to: {output_dir}")
    print("\nGenerated files:")
    for pdf_file in sorted(output_dir.glob("*.pdf")):
        print(f"  - {pdf_file.name}")

    # Print summary statistics
    summary = data['results_summary']
    if summary is not None and len(summary) > 0:
        print("\n" + "="*80)
        print("ERROR SUMMARY")
        print("="*80)
        unc_mae = summary.attrs.get('uncorrected_mae', np.nan)
        if not np.isnan(unc_mae):
            print(f"\n  Uncorrected MAE: {unc_mae:.4f}")

        best_idx = summary['mae'].idxmin()
        best = summary.loc[best_idx]
        print(f"  Best corrected:  MAE={best['mae']:.4f} "
              f"(n={int(best['n_clu'])}, {best['time_res']})")
        if not np.isnan(unc_mae):
            improvement = (unc_mae - best['mae']) / unc_mae * 100
            print(f"  Improvement:     {improvement:.1f}%")


if __name__ == "__main__":
    main()
