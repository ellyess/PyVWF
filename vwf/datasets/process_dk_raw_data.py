"""
Process raw Denmark wind turbine data from Danish Energy Agency.

This script processes:
1. anlaeg.xlsx -> dk_md.csv (turbine metadata)
2. maanedsdata_2002_2020.xlsx -> dk_obs_2002_2020.csv (monthly observations)

Output files are compatible with PyVWF format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, date
import warnings


def process_dk_metadata(
    input_file: Path,
    output_file: Path,
    verbose: bool = True
) -> pd.DataFrame:
    """Process Denmark turbine metadata file.

    Args:
        input_file: Path to anlaeg.xlsx
        output_file: Path to save dk_md.csv
        verbose: Print progress information

    Returns:
        Processed metadata DataFrame
    """
    if verbose:
        print("="*80)
        print("PROCESSING DENMARK METADATA")
        print("="*80)
        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")

    # Read metadata file (header is at row 9)
    df_raw = pd.read_excel(input_file, header=9)

    if verbose:
        print(f"Initial shape: {df_raw.shape}")

    # Remove rows with missing GSRN (turbine ID)
    df = df_raw.dropna(subset=['Turbine identifier (GSRN)']).copy()

    # Keep only rows where GSRN is numeric (valid turbine IDs)
    df['GSRN_str'] = df['Turbine identifier (GSRN)'].astype(str).str.strip()
    df = df[df['GSRN_str'].str.isdigit()].copy()

    if verbose:
        print(f"After filtering: {len(df)} turbines")

    # Extract relevant columns
    columns_map = {
        'Turbine identifier (GSRN)': 'ID',
        'Date of original connection to grid': 'connection_date',
        'Capacity (kW)': 'capacity',
        'Rotor diameter (m)': 'diameter',
        'Hub height (m)': 'height',
        'Manufacture': 'manufacturer',
        'Model': 'model',
        'X (east) coordinate\nUTM 32 Euref89': 'x_utm32',
        'Y (north) coordinate\nUTM 32 Euref89': 'y_utm32',
        'Type of location': 'location_type',
        'Local authority\nname': 'municipality'
    }

    # Select and rename columns
    available_cols = [col for col in columns_map.keys() if col in df.columns]
    df_clean = df[available_cols].copy()
    df_clean = df_clean.rename(columns=columns_map)

    # Convert data types
    df_clean['ID'] = df_clean['ID'].astype(str).str.strip()

    # Convert connection date
    df_clean['connection_date'] = pd.to_datetime(
        df_clean['connection_date'],
        errors='coerce'
    )

    # Convert numeric columns
    numeric_cols = ['capacity', 'diameter', 'height', 'x_utm32', 'y_utm32']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Convert UTM coordinates to lat/lon
    try:
        from pyproj import Transformer

        # Create transformer from UTM32 EUREF89 to WGS84
        transformer = Transformer.from_crs(
            "EPSG:25832",  # UTM 32N EUREF89
            "EPSG:4326",   # WGS84 (lat/lon)
            always_xy=True
        )

        # Only convert valid coordinates
        valid_coords = df_clean['x_utm32'].notna() & df_clean['y_utm32'].notna()

        if valid_coords.any():
            lons, lats = transformer.transform(
                df_clean.loc[valid_coords, 'x_utm32'].values,
                df_clean.loc[valid_coords, 'y_utm32'].values
            )
            df_clean.loc[valid_coords, 'lon'] = lons
            df_clean.loc[valid_coords, 'lat'] = lats

            if verbose:
                print(f"Converted {valid_coords.sum()} coordinates to lat/lon")

    except ImportError:
        warnings.warn(
            "pyproj not installed. Cannot convert UTM to lat/lon. "
            "Install with: pip install pyproj"
        )
    except Exception as e:
        warnings.warn(f"Error converting coordinates: {e}")

    # Clean manufacturer and model names
    if 'manufacturer' in df_clean.columns:
        df_clean['manufacturer'] = df_clean['manufacturer'].fillna('Unknown')
        df_clean['manufacturer'] = df_clean['manufacturer'].str.strip()
        # Replace 'Ukendt' (Danish for 'Unknown')
        df_clean['manufacturer'] = df_clean['manufacturer'].replace('Ukendt', 'Unknown')

    if 'model' in df_clean.columns:
        df_clean['model'] = df_clean['model'].fillna('Unknown')
        df_clean['model'] = df_clean['model'].str.strip()
        df_clean['model'] = df_clean['model'].replace('Ukendt', 'Unknown')

    # Remove duplicates
    df_clean = df_clean.drop_duplicates(subset=['ID'])

    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_file, index=False)

    if verbose:
        print(f"\n✓ Saved metadata: {len(df_clean)} turbines")
        print(f"  Columns: {', '.join(df_clean.columns)}")
        print(f"  Date range: {df_clean['connection_date'].min()} to {df_clean['connection_date'].max()}")
        if 'lat' in df_clean.columns:
            valid_coords = df_clean['lat'].notna().sum()
            print(f"  Valid coordinates: {valid_coords} ({valid_coords/len(df_clean)*100:.1f}%)")

    return df_clean


def process_dk_monthly_observations(
    input_file: Path,
    output_file: Path,
    year_start: int = 2002,
    year_end: int = 2020,
    verbose: bool = True
) -> pd.DataFrame:
    """Process Denmark monthly production data.

    Args:
        input_file: Path to maanedsdata_2002_2020.xlsx
        output_file: Path to save dk_obs_2002_2020.csv
        year_start: First year in data file
        year_end: Last year in data file
        verbose: Print progress information

    Returns:
        Processed observations DataFrame
    """
    if verbose:
        print("\n" + "="*80)
        print("PROCESSING DENMARK MONTHLY OBSERVATIONS")
        print("="*80)
        print(f"Input:  {input_file}")
        print(f"Output: {output_file}")

    all_records = []

    # Process each year sheet
    for year in range(year_start, year_end + 1):
        sheet_name = f"Månedsprod_{year}"

        if verbose:
            print(f"\nProcessing {year}...", end=" ")

        try:
            # Read sheet (header is at row 7)
            df_raw = pd.read_excel(input_file, sheet_name=sheet_name, header=7)

            # Rename columns
            # First column is turbine ID (Møllenummer)
            col_rename = {}
            first_cols = df_raw.columns[:3]
            col_rename[first_cols[0]] = 'ID'
            col_rename[first_cols[1]] = 'connection_date'
            col_rename[first_cols[2]] = 'decommission_date'

            df = df_raw.rename(columns=col_rename)

            # Remove header rows (row with "Møllenummer (identikationsnummer)")
            df = df[df['ID'].astype(str).str.strip() != 'Møllenummer (identikationsnummer)'].copy()

            # Keep only valid turbine IDs (numeric)
            df['ID'] = df['ID'].astype(str).str.strip()
            df = df[df['ID'].str.isdigit()].copy()

            # Identify month columns (they should be datetime columns)
            month_cols = []
            for col in df.columns:
                if col not in ['ID', 'connection_date', 'decommission_date']:
                    # Check if column name is a date (pd.Timestamp or datetime)
                    if isinstance(col, (pd.Timestamp, datetime, date)):
                        month_cols.append(col)
                    elif hasattr(col, 'year') and hasattr(col, 'month'):
                        month_cols.append(col)

            # Convert month columns to numeric
            for col in month_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Fill NaN values with 0 (turbine offline)
            df[month_cols] = df[month_cols].fillna(0)

            # Convert dates
            df['connection_date'] = pd.to_datetime(df['connection_date'], errors='coerce')
            df['decommission_date'] = pd.to_datetime(df['decommission_date'], errors='coerce')

            # Extract year and month from column timestamps
            # Reshape to long format: ID, year, month, generation
            for _, row in df.iterrows():
                turbine_id = row['ID']
                connection_date = row['connection_date']
                decommission_date = row['decommission_date']

                for month_col in month_cols:
                    month = month_col.month
                    generation = row[month_col]

                    all_records.append({
                        'ID': turbine_id,
                        'year': year,
                        'month': month,
                        'generation_kwh': generation,
                        'connection_date': connection_date,
                        'decommission_date': decommission_date
                    })

            if verbose:
                print(f"✓ {len(df)} turbines, {len(month_cols)} months")

        except Exception as e:
            if verbose:
                print(f"✗ Error: {e}")
            continue

    df_long = pd.DataFrame(all_records)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Combined data:")
        print(f"  Total turbine-months: {len(df_long):,}")
        print(f"  Unique turbines: {df_long['ID'].nunique():,}")
        print(f"  Year range: {df_long['year'].min()}-{df_long['year'].max()}")

    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_csv(output_file, index=False)

    if verbose:
        print(f"\n✓ Saved observations to: {output_file}")

        # Summary by year
        year_summary = df_long.groupby('year').agg({
            'ID': 'nunique',
            'generation_kwh': lambda x: x.sum() / 1e6  # Convert to GWh
        })
        year_summary.columns = ['Turbines', 'Generation (GWh)']
        print("\nGeneration by year:")
        print(year_summary.to_string())

    return df_long


def main():
    """Main processing function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Process raw Denmark wind turbine data"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input/turbine_level_data/DK"),
        help="Input directory containing raw data files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input/turbine_level_data/DK"),
        help="Output directory for processed CSV files"
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only process metadata file"
    )
    parser.add_argument(
        "--observations-only",
        action="store_true",
        help="Only process observations file"
    )

    args = parser.parse_args()

    print("="*80)
    print("DENMARK WIND TURBINE DATA PROCESSOR")
    print("="*80)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Define file paths
    metadata_input = args.input_dir / "anlaeg.xlsx"
    observations_input = args.input_dir / "maanedsdata_2002_2020.xlsx"
    metadata_output = args.output_dir / "dk_md.csv"
    observations_output = args.output_dir / "dk_obs_2002_2020.csv"

    # Process metadata
    if not args.observations_only:
        if metadata_input.exists():
            try:
                df_md = process_dk_metadata(
                    metadata_input,
                    metadata_output,
                    verbose=True
                )
            except Exception as e:
                print(f"\n✗ Error processing metadata: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"✗ Metadata file not found: {metadata_input}")

    # Process observations
    if not args.metadata_only:
        if observations_input.exists():
            try:
                df_obs = process_dk_monthly_observations(
                    observations_input,
                    observations_output,
                    verbose=True
                )
            except Exception as e:
                print(f"\n✗ Error processing observations: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"✗ Observations file not found: {observations_input}")

    print("\n" + "="*80)
    print("✓ PROCESSING COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    if metadata_output.exists():
        print(f"  ✓ {metadata_output}")
    if observations_output.exists():
        print(f"  ✓ {observations_output}")


if __name__ == "__main__":
    main()
