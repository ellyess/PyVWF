#!/usr/bin/env python3
"""Prepare turbine fleet features for ML correction model.

Loads turbine metadata from country-specific CSVs, standardizes
column names, and links turbines to correction centroids.

Usage:
    python prepare_turbine_fleet_features.py \
        --corrections output/pyvwf_to_grid/all_corrections_centroids.csv \
        --turbine-dir input/turbine_level_data \
        --output output/pyvwf_to_grid/fleet_features.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def load_dk_metadata(turbine_dir: Path) -> pd.DataFrame:
    """Load and standardize Danish turbine metadata."""
    path = turbine_dir / 'DK' / 'dk_md.csv'
    if not path.exists():
        print(f"  Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path)
    # Columns: ID, connection_date, capacity, diameter, height,
    #          manufacturer, model, x_utm32, y_utm32, location_type,
    #          municipality, lon, lat
    out = pd.DataFrame({
        'turbine_id': df['ID'].astype(str),
        'country': 'DK',
        'capacity': pd.to_numeric(df['capacity'], errors='coerce'),
        'diameter': pd.to_numeric(df['diameter'], errors='coerce'),
        'height': pd.to_numeric(df['height'], errors='coerce'),
        'lon': pd.to_numeric(df['lon'], errors='coerce'),
        'lat': pd.to_numeric(df['lat'], errors='coerce'),
    })
    print(f"  DK: {len(out)} turbines loaded")
    return out


def load_uk_metadata(turbine_dir: Path) -> pd.DataFrame:
    """Load and standardize UK turbine metadata."""
    path = turbine_dir / 'UK' / 'uk_md.csv'
    if not path.exists():
        print(f"  Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path)
    # Columns: ID, manufacturer, capacity, diameter, height, lon, lat, type
    out = pd.DataFrame({
        'turbine_id': df['ID'].astype(str),
        'country': 'UK',
        'capacity': pd.to_numeric(df['capacity'], errors='coerce'),
        'diameter': pd.to_numeric(df['diameter'], errors='coerce'),
        'height': pd.to_numeric(df['height'], errors='coerce'),
        'lon': pd.to_numeric(df['lon'], errors='coerce'),
        'lat': pd.to_numeric(df['lat'], errors='coerce'),
    })
    print(f"  UK: {len(out)} turbines loaded")
    return out


def load_de_metadata(turbine_dir: Path) -> pd.DataFrame:
    """Load and standardize German turbine metadata (no coordinates)."""
    path = turbine_dir / 'DE' / 'DE_md.csv'
    if not path.exists():
        print(f"  Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path)
    # Columns: V1, Manufacturer, kW, Rotor..m., Tower..m., StartDate, ...
    out = pd.DataFrame({
        'turbine_id': df['V1'].astype(str),
        'country': 'DE',
        'capacity': pd.to_numeric(df['kW'], errors='coerce'),
        'diameter': pd.to_numeric(df['Rotor..m.'], errors='coerce'),
        'height': pd.to_numeric(df['Tower..m.'], errors='coerce'),
    })
    # No coordinates available for DE
    # Filter obvious outliers
    out = out[
        (out['capacity'] > 0) & (out['capacity'] < 20000)
        & (out['diameter'] > 5) & (out['diameter'] < 300)
        & (out['height'] > 10) & (out['height'] < 300)
    ]
    print(f"  DE: {len(out)} turbines loaded (no coordinates)")
    return out


def assign_turbines_to_centroids(
    turbines: pd.DataFrame,
    centroids: pd.DataFrame,
) -> pd.DataFrame:
    """Assign turbines to nearest correction centroid using cKDTree.

    Returns per-centroid aggregates.
    """
    turb_coords = turbines[['lon', 'lat']].dropna()
    if turb_coords.empty:
        return pd.DataFrame()

    turb = turbines.loc[turb_coords.index].copy()
    tree = cKDTree(centroids[['lon', 'lat']].values)
    _, indices = tree.query(turb[['lon', 'lat']].values)

    turb['centroid_idx'] = centroids.index[indices]

    agg = turb.groupby('centroid_idx').agg(
        mean_hub_height=('height', 'mean'),
        mean_rotor_diameter=('diameter', 'mean'),
        mean_capacity=('capacity', 'mean'),
        n_turbines=('turbine_id', 'count'),
    )
    return agg


def main():
    parser = argparse.ArgumentParser(
        description='Prepare turbine fleet features for ML',
    )
    parser.add_argument(
        '--corrections', type=str,
        default='output/pyvwf_to_grid/all_corrections_centroids.csv',
        help='Corrections centroids CSV',
    )
    parser.add_argument(
        '--turbine-dir', type=str,
        default='input/turbine_level_data',
        help='Directory with country-level turbine metadata',
    )
    parser.add_argument(
        '--output', type=str,
        default='output/pyvwf_to_grid/fleet_features.csv',
        help='Output CSV with per-centroid fleet features',
    )
    args = parser.parse_args()

    corrections = pd.read_csv(args.corrections)
    turbine_dir = Path(args.turbine_dir)
    output_path = Path(args.output)

    print("=" * 70)
    print("Prepare Turbine Fleet Features")
    print("=" * 70)
    print(f"  Corrections: {len(corrections)} centroids")
    print(f"  Turbine data: {turbine_dir}")

    # Load all metadata
    print("\nLoading turbine metadata...")
    dk_turb = load_dk_metadata(turbine_dir)
    uk_turb = load_uk_metadata(turbine_dir)
    de_turb = load_de_metadata(turbine_dir)

    # Initialize feature columns
    corrections['mean_hub_height'] = np.nan
    corrections['mean_rotor_diameter'] = np.nan
    corrections['mean_capacity'] = np.nan
    corrections['n_turbines'] = np.nan

    # DK - spatial matching
    if not dk_turb.empty:
        print("\nAssigning DK turbines to centroids...")
        dk_mask = corrections['country_code'].str.startswith('DK')
        dk_centroids = corrections.loc[dk_mask]
        if not dk_centroids.empty:
            agg = assign_turbines_to_centroids(dk_turb, dk_centroids)
            for col in agg.columns:
                corrections.loc[agg.index, col] = agg[col]
            print(f"  Matched {len(agg)} centroids")

    # UK - spatial matching
    if not uk_turb.empty:
        print("\nAssigning UK turbines to centroids...")
        uk_mask = corrections['country_code'].str.startswith('UK')
        uk_centroids = corrections.loc[uk_mask]
        if not uk_centroids.empty:
            agg = assign_turbines_to_centroids(uk_turb, uk_centroids)
            for col in agg.columns:
                corrections.loc[agg.index, col] = agg[col]
            print(f"  Matched {len(agg)} centroids")

    # DE - country-wide medians (no coordinates)
    if not de_turb.empty:
        print("\nApplying DE country-wide medians...")
        de_mask = corrections['country_code'].str.startswith('DE')
        corrections.loc[de_mask, 'mean_hub_height'] = de_turb['height'].median()
        corrections.loc[de_mask, 'mean_rotor_diameter'] = de_turb['diameter'].median()
        corrections.loc[de_mask, 'mean_capacity'] = de_turb['capacity'].median()
        print(f"  DE medians: height={de_turb['height'].median():.0f}m, "
              f"diameter={de_turb['diameter'].median():.0f}m, "
              f"capacity={de_turb['capacity'].median():.0f}kW")

    # Fill remaining with global median
    for col in ['mean_hub_height', 'mean_rotor_diameter', 'mean_capacity']:
        n_missing = corrections[col].isna().sum()
        if n_missing > 0:
            corrections[col] = corrections[col].fillna(corrections[col].median())
            print(f"  Filled {n_missing} missing {col} with global median")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fleet_cols = [
        'country_code', 'cluster', 'lon', 'lat',
        'mean_hub_height', 'mean_rotor_diameter', 'mean_capacity', 'n_turbines',
    ]
    corrections[fleet_cols].to_csv(output_path, index=False)

    print(f"\nSaved fleet features: {output_path}")
    print(f"\nSummary:")
    print(corrections[['mean_hub_height', 'mean_rotor_diameter', 'mean_capacity']].describe())

    print("\n" + "=" * 70)
    print("FLEET FEATURES COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
