"""Visualize correction factors before and after interpolation.

This script creates comprehensive maps showing:
1. Raw correction clusters (pre-interpolation)
2. Interpolated correction surfaces (all methods)
3. Comparison between methods
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import thesis plotting style
from plotting_style import thesis_plot_style

# Apply thesis style globally for this script
STYLE = thesis_plot_style()
cm = STYLE['cm']


# Paths
CORRECTIONS_POLYGONS = "output/pyvwf_to_grid/all_corrections_polygons.geojson"
CORRECTIONS_CENTROIDS = "output/pyvwf_to_grid/all_corrections_centroids.geojson"
GRID_DIR = Path("output/pyvwf_to_grid/grid_comparison")
OUTPUT_DIR = Path("output/pyvwf_to_grid/grid_comparison/maps")

# Interpolation methods
METHODS = ['nearest', 'idw', 'kriging', 'rbf']

# Map extent (Europe)
EXTENT = [-10, 30, 35, 72]  # lon_min, lon_max, lat_min, lat_max


def setup_axes(ax):
    """Setup axes for geographic plotting."""
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_aspect('equal')


def plot_raw_corrections_polygons():
    """Plot raw correction clusters as polygons."""
    print("\nPlotting raw correction clusters (polygons)...")

    # Load data
    gdf = gpd.read_file(CORRECTIONS_POLYGONS)

    # Create figures for scalar and offset
    for var, title, vcenter in [
        ('scalar', 'Correction Scalar (Raw Clusters)', 1.0),
        ('offset', 'Correction Offset (Raw Clusters)', 0.0)
    ]:
        fig, ax = plt.subplots(figsize=(14*cm, 10*cm))
        setup_axes(ax)

        # Plot polygons with diverging colormap centered at identity
        vmin = gdf[var].min()
        vmax = gdf[var].max()
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        gdf.plot(column=var, ax=ax, cmap='RdBu_r', norm=norm,
                 edgecolor='black', linewidth=0.2, alpha=0.7,
                 legend=True, legend_kwds={'label': var.capitalize()})

        ax.set_title(title, fontweight='bold')

        output_path = OUTPUT_DIR / f'raw_corrections_{var}_polygons.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()


def plot_raw_corrections_centroids():
    """Plot raw correction clusters as centroids (points)."""
    print("\nPlotting raw correction centroids (points)...")

    # Load data
    gdf = gpd.read_file(CORRECTIONS_CENTROIDS)

    # Create figures for scalar and offset
    for var, title, vcenter in [
        ('scalar', 'Correction Scalar (Centroids)', 1.0),
        ('offset', 'Correction Offset (Centroids)', 0.0)
    ]:
        fig, ax = plt.subplots(figsize=(14*cm, 10*cm))
        setup_axes(ax)

        # Plot points with diverging colormap centered at identity
        vmin = gdf[var].min()
        vmax = gdf[var].max()
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        scatter = ax.scatter(gdf.geometry.x, gdf.geometry.y,
                            c=gdf[var], cmap='RdBu_r', norm=norm,
                            s=20, alpha=0.8, edgecolors='black', linewidth=0.3)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
        cbar.set_label(var.capitalize())
        # Mark identity value
        cbar.ax.axhline(vcenter, color='black', linewidth=1.5, linestyle='--', alpha=0.7)

        ax.set_title(title, fontweight='bold')

        # Add statistics
        stats_text = f'n = {len(gdf)}\nmean = {gdf[var].mean():.2f}\nstd = {gdf[var].std():.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        output_path = OUTPUT_DIR / f'raw_corrections_{var}_centroids.png'
        plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()


def plot_interpolated_surfaces():
    """Plot interpolated correction surfaces for all methods."""
    print("\nPlotting interpolated correction surfaces...")

    # Load raw centroids for overlay
    centroids = gpd.read_file(CORRECTIONS_CENTROIDS)

    for var, title_prefix, vcenter in [
        ('scalar', 'Correction Scalar', 1.0),
        ('offset', 'Correction Offset', 0.0)
    ]:
        for method in METHODS:
            # Load interpolated grid
            nc_path = GRID_DIR / f'europe_corrections_{method}.nc'
            if not nc_path.exists():
                print(f"  ! Skipping {method} - file not found")
                continue

            ds = xr.open_dataset(nc_path)

            # Create figure
            fig, ax = plt.subplots(figsize=(16*cm, 10*cm))
            setup_axes(ax)

            # Plot interpolated surface with diverging colormap
            vmin = ds[var].min().values
            vmax = ds[var].max().values
            norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

            im = ax.pcolormesh(ds['x'], ds['y'], ds[var],
                              cmap='RdBu_r', norm=norm, shading='auto', alpha=0.7)

            # Overlay raw control points
            ax.scatter(centroids.geometry.x, centroids.geometry.y,
                      c='black', s=3, alpha=0.4, marker='x', label='Control points')

            # Colorbar
            cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.7)
            cbar.set_label(var.capitalize())
            # Mark identity value
            cbar.ax.axhline(vcenter, color='black', linewidth=1.5, linestyle='--', alpha=0.7)

            # Title
            title = f'{title_prefix} - {method.upper()} Interpolation'
            ax.set_title(title, fontweight='bold')

            # Stats (moved to upper right to avoid legend overlap)
            data_mean = ds[var].mean().values
            data_std = ds[var].std().values
            stats_text = f'Method: {method.upper()}\nmean = {data_mean:.2f}\nstd = {data_std:.2f}\nn = {len(centroids)} points'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            output_path = OUTPUT_DIR / f'interpolated_{var}_{method}.png'
            plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
            print(f"  ✓ Saved: {output_path}")
            plt.close()


def plot_method_comparison():
    """Create side-by-side comparison of all interpolation methods."""
    print("\nPlotting method comparison...")

    # Load raw centroids
    centroids = gpd.read_file(CORRECTIONS_CENTROIDS)

    for var, title_prefix, vcenter in [
        ('scalar', 'Correction Scalar', 1.0),
        ('offset', 'Correction Offset', 0.0)
    ]:
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(20*cm, 16*cm))
        axes = axes.flatten()

        # Global range for consistent colormap
        all_data = []
        for method in METHODS:
            nc_path = GRID_DIR / f'europe_corrections_{method}.nc'
            if nc_path.exists():
                ds = xr.open_dataset(nc_path)
                all_data.append(ds[var].values)

        vmin = np.min([d.min() for d in all_data])
        vmax = np.max([d.max() for d in all_data])
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        for idx, method in enumerate(METHODS):
            # Load interpolated grid
            nc_path = GRID_DIR / f'europe_corrections_{method}.nc'
            if not nc_path.exists():
                print(f"  ! Skipping {method} - file not found")
                continue

            ds = xr.open_dataset(nc_path)
            ax = axes[idx]
            setup_axes(ax)

            # Plot interpolated surface
            im = ax.pcolormesh(ds['x'], ds['y'], ds[var],
                              cmap='RdBu_r', norm=norm, shading='auto', alpha=0.7)

            # Overlay control points
            ax.scatter(centroids.geometry.x, centroids.geometry.y,
                      c='black', s=2, alpha=0.3, marker='x')

            # Title
            ax.set_title(f'{method.upper()}', fontweight='bold')

            # Stats
            data_mean = ds[var].mean().values
            data_std = ds[var].std().values
            stats_text = f'μ={data_mean:.2f}\nσ={data_std:.2f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Overall title
        fig.suptitle(f'{title_prefix} - Method Comparison', fontweight='bold', y=0.98)

        # Shared colorbar
        cbar = fig.colorbar(im, ax=axes, orientation='vertical',
                           pad=0.02, shrink=0.7, aspect=30)
        cbar.set_label(var.capitalize())
        # Mark identity value
        cbar.ax.axhline(vcenter, color='black', linewidth=1.5, linestyle='--', alpha=0.7)

        output_path = OUTPUT_DIR / f'comparison_{var}_all_methods.png'
        plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()


def plot_difference_maps():
    """Plot difference maps between methods."""
    print("\nPlotting difference maps (IDW vs other methods)...")

    # Load IDW as reference
    ds_idw = xr.open_dataset(GRID_DIR / 'europe_corrections_idw.nc')

    for var, title_prefix in [
        ('scalar', 'Scalar Difference'),
        ('offset', 'Offset Difference')
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(20*cm, 6*cm))
        methods_to_compare = ['nearest', 'kriging', 'rbf']

        for idx, method in enumerate(methods_to_compare):
            nc_path = GRID_DIR / f'europe_corrections_{method}.nc'
            if not nc_path.exists():
                continue

            ds = xr.open_dataset(nc_path)
            ax = axes[idx]
            setup_axes(ax)

            # Calculate difference
            diff = ds[var] - ds_idw[var]

            # Symmetric color scale around zero
            vmax = max(abs(diff.min().values), abs(diff.max().values))
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

            im = ax.pcolormesh(ds['x'], ds['y'], diff,
                              cmap='RdBu_r', norm=norm, shading='auto', alpha=0.7)

            # Title
            ax.set_title(f'{method.upper()} - IDW', fontweight='bold')

            # Stats
            diff_mean = diff.mean().values
            diff_std = diff.std().values
            diff_max = abs(diff).max().values
            stats_text = f'μ={diff_mean:.3f}\nσ={diff_std:.3f}\n|max|={diff_max:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # Overall title
        fig.suptitle(f'{title_prefix} from IDW (Reference)', fontweight='bold', y=0.98)

        # Shared colorbar
        cbar = fig.colorbar(im, ax=axes, orientation='vertical',
                           pad=0.02, shrink=0.7, aspect=30)
        cbar.set_label(f'Δ {var.capitalize()}')

        output_path = OUTPUT_DIR / f'difference_{var}_from_idw.png'
        plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()


def plot_summary_figure():
    """Create comprehensive summary figure."""
    print("\nCreating summary figure...")

    # Load data
    centroids = gpd.read_file(CORRECTIONS_CENTROIDS)
    polygons = gpd.read_file(CORRECTIONS_POLYGONS)
    ds_idw = xr.open_dataset(GRID_DIR / 'europe_corrections_idw.nc')

    # Create figure with 6 subplots (2 rows × 3 cols)
    fig = plt.figure(figsize=(24*cm, 14*cm))

    for row, (var, title_suffix, vcenter) in enumerate([
        ('scalar', 'Scalar', 1.0),
        ('offset', 'Offset (m/s)', 0.0)
    ]):
        # Get data range
        vmin = min(polygons[var].min(), ds_idw[var].min().values)
        vmax = max(polygons[var].max(), ds_idw[var].max().values)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

        # Column 1: Raw polygons
        ax1 = fig.add_subplot(2, 3, row*3 + 1)
        setup_axes(ax1)

        polygons.plot(column=var, ax=ax1, cmap='RdBu_r', norm=norm,
                     edgecolor='black', linewidth=0.1, alpha=0.7)
        ax1.set_title(f'Raw Clusters\n{title_suffix}', fontweight='bold')

        # Column 2: Raw centroids
        ax2 = fig.add_subplot(2, 3, row*3 + 2)
        setup_axes(ax2)

        scatter = ax2.scatter(centroids.geometry.x, centroids.geometry.y,
                            c=centroids[var], cmap='RdBu_r', norm=norm,
                            s=15, alpha=0.8, edgecolors='black', linewidth=0.2)
        ax2.set_title(f'Control Points (n={len(centroids)})\n{title_suffix}', fontweight='bold')

        # Column 3: IDW interpolated
        ax3 = fig.add_subplot(2, 3, row*3 + 3)
        setup_axes(ax3)

        im = ax3.pcolormesh(ds_idw['x'], ds_idw['y'], ds_idw[var],
                           cmap='RdBu_r', norm=norm, shading='auto', alpha=0.7)
        ax3.scatter(centroids.geometry.x, centroids.geometry.y,
                   c='black', s=2, alpha=0.3, marker='x')
        ax3.set_title(f'IDW Interpolated\n{title_suffix}', fontweight='bold')

        # Add colorbar for this row
        cbar = fig.colorbar(im, ax=[ax1, ax2, ax3],
                           orientation='vertical', pad=0.02, aspect=20)
        cbar.set_label(title_suffix)
        # Mark identity value
        cbar.ax.axhline(vcenter, color='black', linewidth=1.5, linestyle='--', alpha=0.7)

    plt.suptitle('Bias Correction Factors: Raw Clusters → Interpolated Surface',
                fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = OUTPUT_DIR / 'summary_corrections_workflow.png'
    plt.savefig(output_path, dpi=STYLE['dpi'], bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def main():
    """Main execution."""
    print("="*80)
    print("Visualizing Correction Factors and Interpolations")
    print("="*80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Raw correction clusters (polygons)
    plot_raw_corrections_polygons()

    # 2. Raw correction centroids (points)
    plot_raw_corrections_centroids()

    # 3. Interpolated surfaces (all methods)
    plot_interpolated_surfaces()

    # 4. Method comparison (side-by-side)
    plot_method_comparison()

    # 5. Difference maps (vs IDW)
    plot_difference_maps()

    # 6. Summary figure
    plot_summary_figure()

    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)
    print(f"\nAll maps saved to: {OUTPUT_DIR}")
    print("\nMaps created:")
    print("  - Raw correction clusters (polygons): 2 maps")
    print("  - Raw correction centroids (points): 2 maps")
    print("  - Interpolated surfaces: 8 maps (4 methods × 2 variables)")
    print("  - Method comparison: 2 maps (side-by-side)")
    print("  - Difference maps: 2 maps (vs IDW reference)")
    print("  - Summary workflow: 1 map")
    print(f"\nTotal: 17 maps generated")


if __name__ == "__main__":
    main()
