"""Generate all Denmark thesis figures from PyVWF output directory.

This script loads data from a PyVWF run output directory and generates
all publication-ready figures for the Denmark wind power paper.

Usage:
    python plot_denmark_from_vwf_output.py \\
        --data_dir output/run-working/DK-onshore-obs_turbine-corrected-calc_z0 \\
        --output_dir output/denmark_figures
"""

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Import thesis plotting style
from plotting_style import thesis_plot_style

# Import individual plotting functions from the main script
import sys
sys.path.insert(0, str(Path(__file__).parent))


# =============================================================================
# GLOBAL STYLE SETUP
# =============================================================================
STYLE = thesis_plot_style()
cm = STYLE['cm']


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
            - capacity_factors: dict of DataFrames by (n_clu, time_res)
            - results_summary: DataFrame with error metrics
    """
    data_dir = Path(data_dir)
    print(f"\nLoading data from: {data_dir}")

    data = {}

    # Load turbine info files
    train_path = data_dir / "training/simulated-turbines/DK_train_turb_info.csv"
    val_path = data_dir / "training/simulated-turbines/DK_2020_turb_info.csv"

    print("  Loading turbine metadata...")
    if train_path.exists():
        data['train_turbines'] = pd.read_csv(train_path)
        data['train_turbines']['offshore'] = data['train_turbines']['type'] == 'offshore'
        print(f"    ✓ Training turbines: {len(data['train_turbines'])}")
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
        print(f"    ✓ Validation turbines: {len(data['val_turbines'])}")
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

        print(f"    ✓ Loaded {len(correction_factors)} correction factor files")

    data['correction_factors'] = correction_factors

    # Load capacity factors and compute errors
    print("  Loading capacity factors and computing errors...")
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
                    # Only load small sample for speed (or aggregate first)
                    # Full file is too large, we'll compute summary statistics
                    capacity_factors[(n_clu_int, time_res)] = cf_file
                except ValueError:
                    pass

        print(f"    ✓ Found {len(capacity_factors)} capacity factor files")

    data['capacity_factors'] = capacity_factors

    # Build results summary from correction factors metadata
    print("  Building results summary...")
    data['results_summary'] = build_results_summary(data)

    return data


def build_results_summary(data):
    """Build summary DataFrame of results from correction factors.

    This creates a placeholder results summary. In practice, you would
    compute actual RMSE/MAE values from the capacity factor files.
    """
    rows = []

    for (n_clu, time_res), cf_df in data['correction_factors'].items():
        # Placeholder values - replace with actual computation
        row = {
            'n_clu': n_clu,
            'time_res': time_res,
            'rmse_train': np.nan,  # Compute from training data
            'rmse_full': np.nan,   # Compute from validation data
            'mae_full': np.nan,
            'rmse_temporal': np.nan,
            'rmse_spatial': np.nan,
            'mbe': np.nan,
        }
        rows.append(row)

    if rows:
        return pd.DataFrame(rows).sort_values(['time_res', 'n_clu'])
    return None


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
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
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
    print("\n[1/9] Plotting turbine locations...")

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
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_clustering_overview(data, output_dir):
    """Figure 4: Clustering visualization at different resolutions."""
    print("\n[2/9] Plotting clustering overview...")

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

        scatter = ax.scatter(coords[:, 0], coords[:, 1],
                           c=labels, s=2, cmap='tab20', alpha=0.7)

        centroids = kmeans.cluster_centers_
        ax.scatter(centroids[:, 0], centroids[:, 1],
                  c='red', s=20, marker='x', linewidths=2, alpha=0.8)

        ax.set_title(f'n_clusters = {n_clu}')
        setup_map_axes(ax)

    plt.tight_layout()
    output_path = output_dir / 'fig4_clustering_overview.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_clustering_heuristics(data, output_dir):
    """Figure 5: Elbow method and silhouette score."""
    print("\n[3/9] Plotting clustering heuristics...")

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
    output_path = output_dir / 'fig5_clustering_heuristics.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_correction_factors_spatial(data, output_dir):
    """Figure 9: Spatial distribution of correction factors."""
    print("\n[4/9] Plotting correction factors...")

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
    output_path = output_dir / 'fig9_correction_factors.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_correction_factors_distribution(data, output_dir):
    """Additional figure: Distribution of correction factors across all clusters."""
    print("\n[5/9] Plotting correction factor distributions...")

    # Plot distributions for different spatial resolutions
    n_clu_list = [1, 10, 100, 700]
    time_res = 'fixed'

    fig, axes = plt.subplots(2, 2, figsize=(16*cm, 14*cm))
    axes = axes.flatten()

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
        stats_text = (f'Scalar: μ={cf_df["scalar"].mean():.2f}, σ={cf_df["scalar"].std():.2f}\\n'
                     f'Offset: μ={cf_df["offset"].mean():.2f}, σ={cf_df["offset"].std():.2f}')
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=5)

    plt.tight_layout()
    output_path = output_dir / 'fig_correction_factor_distributions.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_temporal_slicing_overview(data, output_dir):
    """Figure showing temporal slicing scheme."""
    print("\n[6/9] Plotting temporal slicing overview...")

    fig, axes = plt.subplots(4, 1, figsize=(16*cm, 12*cm))

    time_res_schemes = {
        'fixed': [list(range(1, 13))],
        'seasonal': [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        'bimonthly': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
        'monthly': [[i] for i in range(1, 13)]
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
        ax.set_ylabel(tres, rotation=0, ha='right', va='center')

        if idx == 3:
            ax.set_xlabel('Month')
        else:
            ax.set_xticklabels([])

    plt.suptitle('Temporal Slicing Schemes', fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / 'fig_temporal_slicing.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_summary_figure(data, output_dir):
    """Create an overview summary figure."""
    print("\n[7/9] Creating summary figure...")

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
    ax1.set_title(f'(a) Training\\nn={len(train_df)}')
    setup_map_axes(ax1)

    # (b) Validation turbines
    ax2 = fig.add_subplot(gs[0, 1])
    if dk_shape is not None:
        dk_shape.plot(ax=ax2, facecolor='none', edgecolor='gray', linewidth=0.8)
    if val_df is not None:
        onshore_val = val_df[~val_df['offshore']]
        ax2.scatter(onshore_val['lon'], onshore_val['lat'], s=1, c='#06A77D', alpha=0.6)
        ax2.set_title(f'(b) Validation\\nn={len(val_df)}')
    setup_map_axes(ax2)

    # (c) Example clustering
    ax3 = fig.add_subplot(gs[0, 2])
    coords = onshore[['lon', 'lat']].values
    kmeans = KMeans(n_clusters=100, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    if dk_shape is not None:
        dk_shape.plot(ax=ax3, facecolor='none', edgecolor='gray', linewidth=0.8)
    ax3.scatter(coords[:, 0], coords[:, 1], c=labels, s=1, cmap='tab20', alpha=0.7)
    ax3.set_title('(c) Spatial Clusters\\nn=100')
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
    ax4.set_title(f'(d) Correction Scalar\\nn={n_clu}')
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
    ax5.set_title(f'(e) Correction Offset\\nn={n_clu}')
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

    output_path = output_dir / 'fig_summary_overview.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def plot_available_configurations(data, output_dir):
    """Plot matrix showing available model configurations."""
    print("\n[8/9] Plotting available configurations...")

    # Extract all combinations
    n_clu_vals = sorted(set([k[0] for k in data['correction_factors'].keys()]))
    time_res_set = set([k[1] for k in data['correction_factors'].keys()])

    # Sort time_res in logical order
    order = {'fixed': 0, 'season': 1, 'seasonal': 1, 'bimonth': 2, 'bimonthly': 2, 'month': 3, 'monthly': 3}
    time_res_vals = sorted(time_res_set, key=lambda x: order.get(x, 99))

    # Create matrix
    matrix = np.zeros((len(time_res_vals), len(n_clu_vals)))
    for i, tres in enumerate(time_res_vals):
        for j, nclu in enumerate(n_clu_vals):
            if (nclu, tres) in data['correction_factors']:
                matrix[i, j] = 1

    fig, ax = plt.subplots(figsize=(16*cm, 10*cm))

    im = ax.imshow(matrix, cmap='Blues', aspect='auto', interpolation='nearest')

    ax.set_xticks(range(len(n_clu_vals)))
    ax.set_xticklabels(n_clu_vals, rotation=45, ha='right')
    ax.set_yticks(range(len(time_res_vals)))
    ax.set_yticklabels(time_res_vals)

    ax.set_xlabel('Number of Clusters (n_clu)')
    ax.set_ylabel('Temporal Resolution')
    ax.set_title(f'Available Model Configurations (Total: {len(data["correction_factors"])})')

    # Add text annotations
    for i in range(len(time_res_vals)):
        for j in range(len(n_clu_vals)):
            if matrix[i, j] == 1:
                ax.text(j, i, '✓', ha='center', va='center', color='white', fontsize=8)

    plt.tight_layout()
    output_path = output_dir / 'fig_available_configurations.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def create_paper_figures_summary(data, output_dir):
    """Create a single page summary matching paper figures."""
    print("\n[9/9] Creating paper figures summary...")

    # This would include the key results figures from Figure 6, 7, 8
    # For now, create a placeholder

    fig = plt.figure(figsize=(20*cm, 24*cm))

    fig.text(0.5, 0.5,
            'PAPER FIGURES SUMMARY\\n\\n'
            'To generate Figures 6, 7, 8:\\n'
            '- Compute RMSE/MAE from capacity factor files\\n'
            '- Compare observed vs simulated CF\\n'
            '- Analyze temporal and spatial errors\\n\\n'
            'Data files are ready in:\\n'
            'results/capacity-factor/',
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.axis('off')
    output_path = output_dir / 'fig_paper_summary_placeholder.pdf'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
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
        help="PyVWF output directory (e.g., output/run-working/DK-onshore-...)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/plots/denmark_figures",
        help="Output directory for figures"
    )

    args = parser.parse_args()

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

    plot_turbine_locations(data, output_dir)
    plot_clustering_overview(data, output_dir)
    plot_clustering_heuristics(data, output_dir)
    plot_correction_factors_spatial(data, output_dir)
    plot_correction_factors_distribution(data, output_dir)
    plot_temporal_slicing_overview(data, output_dir)
    create_summary_figure(data, output_dir)
    plot_available_configurations(data, output_dir)
    create_paper_figures_summary(data, output_dir)

    print("\n" + "="*80)
    print("✓ ALL FIGURES COMPLETE")
    print("="*80)
    print(f"\nFigures saved to: {output_dir}")
    print("\nGenerated files:")
    for pdf_file in sorted(output_dir.glob("*.pdf")):
        print(f"  - {pdf_file.name}")


if __name__ == "__main__":
    main()
