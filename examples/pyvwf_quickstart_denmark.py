#!/usr/bin/env python3
"""
PyVWF Training Quickstart - Denmark Example

This script demonstrates the complete PyVWF workflow:
1. Load turbine metadata and observations
2. Prepare ERA5 reanalysis data
3. Train bias correction factors
4. Simulate wind power with corrections
5. Validate against observations
6. Export results and plots

Example usage:
    python examples/pyvwf_quickstart_denmark.py
    python examples/pyvwf_quickstart_denmark.py --year-test 2020
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def check_data_availability(country: str = "DK", year_test: int = 2020):
    """Check if required input data exists."""
    print_section("Checking Data Availability")
    
    country_dir = Path(f"input/country-data/{country}")
    era5_train = Path(f"input/era5/{country}/train")
    era5_test = Path(f"input/era5/{country}/test")
    
    checks = {
        "Turbine metadata": country_dir / f"{country.lower()}_md.csv",
        "Observations (train)": country_dir / "observations",
        "ERA5 data (train)": era5_train,
        "ERA5 data (test)": era5_test,
        "Power curves": Path("input/power_curves.csv"),
    }
    
    all_ok = True
    for name, path in checks.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_ok = False
    
    if not all_ok:
        print("\n⚠ Warning: Some required files are missing.")
        print("  See README.md for data preparation instructions.")
        return False
    
    print("\n✓ All required data files found!")
    return True


def run_pyvwf_training(
    country: str = "DK",
    year_test: int = 2020,
    calc_z0: bool = True,
    cluster_mode: str = "onshore",
    n_clusters: int = 5,
    time_res: str = "month",
    output_dir: Path = None,
):
    """
    Run the complete PyVWF training and simulation workflow.
    
    Parameters
    ----------
    country : str
        Country code (e.g., 'DK' for Denmark)
    year_test : int
        Test year for simulation
    calc_z0 : bool
        Calculate surface roughness from wind profiles
    cluster_mode : str
        Clustering mode: 'all', 'onshore', 'offshore'
    n_clusters : int
        Number of clusters for spatial aggregation
    time_res : str
        Time resolution for corrections: 'fixed', 'season', 'bimonth', 'month'
    output_dir : Path
        Output directory for results
    """
    
    if output_dir is None:
        z0_str = "calc_z0" if calc_z0 else "era5_z0"
        output_dir = Path(f"out/quickstart_{country}_{year_test}_{z0_str}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_section(f"PyVWF Training - {country} (Test Year: {year_test})")
    print(f"Configuration:")
    print(f"  Country: {country}")
    print(f"  Test year: {year_test}")
    print(f"  Surface roughness: {'Calculated' if calc_z0 else 'ERA5 forecast'}")
    print(f"  Cluster mode: {cluster_mode}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Time resolution: {time_res}")
    print(f"  Output directory: {output_dir}")
    
    # Import PyVWF modules
    print("\nImporting PyVWF modules...")
    from vwf.data import prep_country
    from vwf.datasets.era5 import prep_era5
    from vwf.clustering import cluster_turbines
    from vwf.correction import calculate_scalar, calculate_offset
    from vwf.wind import train_simulate_wind, simulate_country_cf
    from vwf.metrics import calculate_metrics
    
    # =================================================================
    # STEP 1: Load and prepare data
    # =================================================================
    print_section("STEP 1: Loading Turbine Metadata and Observations")
    
    turb_info, obs_data = prep_country(
        country=country,
        year_test=year_test,
        obs_level="turbine",  # Use turbine-level observations
    )
    
    print(f"\nTurbine metadata loaded:")
    print(f"  Total turbines: {len(turb_info)}")
    print(f"  Onshore: {(turb_info['type'] == 'onshore').sum()}")
    print(f"  Offshore: {(turb_info['type'] == 'offshore').sum()}")
    print(f"  Avg capacity: {turb_info['capacity'].mean():.2f} MW")
    print(f"  Avg hub height: {turb_info['height'].mean():.1f} m")
    
    print(f"\nObservation data loaded:")
    print(f"  Date range: {obs_data.index.min()} to {obs_data.index.max()}")
    print(f"  Number of days: {len(obs_data)}")
    print(f"  Turbines with data: {len(obs_data.columns)}")
    
    # Save turbine info
    turb_info_path = output_dir / "turbine_metadata.csv"
    turb_info.to_csv(turb_info_path, index=False)
    print(f"\n✓ Saved turbine metadata: {turb_info_path}")
    
    # =================================================================
    # STEP 2: Load ERA5 reanalysis data
    # =================================================================
    print_section("STEP 2: Loading ERA5 Reanalysis Data")
    
    # Training data (years before test year)
    print("\nLoading ERA5 training data...")
    era5_train = prep_era5(
        country=country,
        train=True,
        calc_z0=calc_z0,
    )
    print(f"  Training period: {era5_train.time.min().values} to {era5_train.time.max().values}")
    print(f"  Grid size: {len(era5_train.lat)} × {len(era5_train.lon)}")
    print(f"  Variables: {list(era5_train.data_vars)}")
    
    # Test data
    print("\nLoading ERA5 test data...")
    era5_test = prep_era5(
        country=country,
        train=False,
        calc_z0=calc_z0,
    )
    print(f"  Test period: {era5_test.time.min().values} to {era5_test.time.max().values}")
    
    # =================================================================
    # STEP 3: Spatial clustering
    # =================================================================
    print_section("STEP 3: Spatial Clustering of Turbines")
    
    print(f"\nClustering turbines (mode: {cluster_mode}, n_clusters: {n_clusters})...")
    turb_info_clustered = cluster_turbines(
        turb_info=turb_info,
        obs_data=obs_data,
        mode=cluster_mode,
        n_clusters=n_clusters,
    )
    
    cluster_counts = turb_info_clustered['cluster'].value_counts().sort_index()
    print(f"\nCluster distribution:")
    for cluster_id, count in cluster_counts.items():
        print(f"  Cluster {cluster_id}: {count} turbines")
    
    # Save clustered turbine info
    cluster_path = output_dir / "turbines_clustered.csv"
    turb_info_clustered.to_csv(cluster_path, index=False)
    print(f"\n✓ Saved clustered turbines: {cluster_path}")
    
    # =================================================================
    # STEP 4: Train bias correction factors
    # =================================================================
    print_section("STEP 4: Training Bias Correction Factors")
    
    print(f"\nSimulating training period with ERA5 data...")
    train_results = train_simulate_wind(
        turb_info=turb_info_clustered,
        obs_data=obs_data,
        reanalysis=era5_train,
    )
    
    print(f"  Generated {len(train_results)} simulated capacity factor time series")
    
    # Calculate scalar correction
    print(f"\nCalculating scalar correction (time_res: {time_res})...")
    scalar_corrections = calculate_scalar(
        gen_cf=train_results,
        time_res=time_res,
    )
    
    print(f"  Scalar corrections by cluster:")
    for cluster_id in sorted(scalar_corrections['cluster'].unique()):
        cluster_data = scalar_corrections[scalar_corrections['cluster'] == cluster_id]
        print(f"    Cluster {cluster_id}: mean={cluster_data['scalar'].mean():.3f}, "
              f"std={cluster_data['scalar'].std():.3f}")
    
    # Calculate offset correction
    print(f"\nCalculating offset correction (time_res: {time_res})...")
    offset_corrections = calculate_offset(
        gen_cf=train_results,
        time_res=time_res,
    )
    
    print(f"  Offset corrections by cluster:")
    for cluster_id in sorted(offset_corrections['cluster'].unique()):
        cluster_data = offset_corrections[cluster_id == offset_corrections['cluster']]
        print(f"    Cluster {cluster_id}: mean={cluster_data['offset'].mean():.3f}, "
              f"std={cluster_data['offset'].std():.3f}")
    
    # Save correction factors
    scalar_path = output_dir / "scalar_corrections.csv"
    offset_path = output_dir / "offset_corrections.csv"
    scalar_corrections.to_csv(scalar_path, index=False)
    offset_corrections.to_csv(offset_path, index=False)
    print(f"\n✓ Saved scalar corrections: {scalar_path}")
    print(f"✓ Saved offset corrections: {offset_path}")
    
    # =================================================================
    # STEP 5: Simulate test year with corrections
    # =================================================================
    print_section(f"STEP 5: Simulating Test Year ({year_test})")
    
    print("\nSimulating with bias corrections applied...")
    cf_corrected = simulate_country_cf(
        turb_info=turb_info_clustered,
        reanalysis=era5_test,
        scalar_factor=scalar_corrections,
        offset_factor=offset_corrections,
        time_res=time_res,
    )
    
    print(f"  Generated corrected CF time series: {cf_corrected.shape}")
    print(f"  Date range: {cf_corrected.index.min()} to {cf_corrected.index.max()}")
    
    # Also simulate without corrections for comparison
    print("\nSimulating without corrections (baseline)...")
    cf_uncorrected = simulate_country_cf(
        turb_info=turb_info_clustered,
        reanalysis=era5_test,
        scalar_factor=None,  # No corrections
        offset_factor=None,
        time_res=time_res,
    )
    
    # Save simulated capacity factors
    cf_corrected_path = output_dir / "cf_corrected.csv"
    cf_uncorrected_path = output_dir / "cf_uncorrected.csv"
    cf_corrected.to_csv(cf_corrected_path)
    cf_uncorrected.to_csv(cf_uncorrected_path)
    print(f"\n✓ Saved corrected CF: {cf_corrected_path}")
    print(f"✓ Saved uncorrected CF: {cf_uncorrected_path}")
    
    # =================================================================
    # STEP 6: Validation against observations
    # =================================================================
    print_section("STEP 6: Validation Against Observations")
    
    # Get test year observations
    test_start = pd.Timestamp(f"{year_test}-01-01")
    test_end = pd.Timestamp(f"{year_test}-12-31")
    obs_test = obs_data.loc[test_start:test_end]
    
    if len(obs_test) == 0:
        print(f"\n⚠ Warning: No observations available for {year_test}")
        print("  Skipping validation step.")
    else:
        print(f"\nObservations available: {len(obs_test)} days")
        
        # Aggregate to country level for comparison
        cf_corrected_country = cf_corrected.mean(axis=1)
        cf_uncorrected_country = cf_uncorrected.mean(axis=1)
        obs_country = obs_test.mean(axis=1)
        
        # Align dates
        common_dates = cf_corrected_country.index.intersection(obs_country.index)
        print(f"  Common dates for validation: {len(common_dates)}")
        
        if len(common_dates) > 0:
            # Calculate metrics
            print("\nCalculating validation metrics...")
            
            metrics_corrected = calculate_metrics(
                obs_country.loc[common_dates],
                cf_corrected_country.loc[common_dates],
            )
            
            metrics_uncorrected = calculate_metrics(
                obs_country.loc[common_dates],
                cf_uncorrected_country.loc[common_dates],
            )
            
            # Display results
            print("\n" + "-"*70)
            print("Validation Results - Country-Level Aggregation")
            print("-"*70)
            print(f"{'Metric':<20} {'Uncorrected':>15} {'Corrected':>15} {'Improvement':>15}")
            print("-"*70)
            
            metrics_to_show = ['r2', 'rmse', 'mae', 'bias', 'correlation']
            for metric in metrics_to_show:
                if metric in metrics_uncorrected and metric in metrics_corrected:
                    uncorr = metrics_uncorrected[metric]
                    corr = metrics_corrected[metric]
                    
                    # Calculate improvement
                    if metric in ['rmse', 'mae', 'bias']:
                        improvement = ((uncorr - corr) / abs(uncorr)) * 100
                        imp_str = f"{improvement:+.1f}%"
                    else:  # r2, correlation (higher is better)
                        improvement = ((corr - uncorr) / abs(uncorr)) * 100
                        imp_str = f"{improvement:+.1f}%"
                    
                    print(f"{metric.upper():<20} {uncorr:>15.4f} {corr:>15.4f} {imp_str:>15}")
            
            print("-"*70)
            
            # Save metrics
            metrics_df = pd.DataFrame({
                'metric': list(metrics_corrected.keys()),
                'uncorrected': list(metrics_uncorrected.values()),
                'corrected': list(metrics_corrected.values()),
            })
            metrics_path = output_dir / "validation_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            print(f"\n✓ Saved validation metrics: {metrics_path}")
            
            # Save comparison time series
            comparison_df = pd.DataFrame({
                'date': common_dates,
                'observed': obs_country.loc[common_dates].values,
                'simulated_uncorrected': cf_uncorrected_country.loc[common_dates].values,
                'simulated_corrected': cf_corrected_country.loc[common_dates].values,
            })
            comparison_path = output_dir / "timeseries_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            print(f"✓ Saved time series comparison: {comparison_path}")
    
    # =================================================================
    # STEP 7: Generate summary plots (optional)
    # =================================================================
    print_section("STEP 7: Summary")
    
    print(f"\n✓ PyVWF training complete!")
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  • turbine_metadata.csv - Turbine information")
    print(f"  • turbines_clustered.csv - Turbines with cluster assignments")
    print(f"  • scalar_corrections.csv - Multiplicative correction factors")
    print(f"  • offset_corrections.csv - Additive correction factors")
    print(f"  • cf_corrected.csv - Simulated CF with corrections")
    print(f"  • cf_uncorrected.csv - Simulated CF without corrections")
    if len(obs_test) > 0 and len(common_dates) > 0:
        print(f"  • validation_metrics.csv - Performance metrics")
        print(f"  • timeseries_comparison.csv - Observed vs simulated time series")
    
    print(f"\nNext steps:")
    print(f"  1. Visualize results with your plotting tools")
    print(f"  2. Export corrections to grid: vwf.export_pyvwf_grid()")
    print(f"  3. Try different cluster configurations or time resolutions")
    print(f"  4. Compare with other countries")
    
    return output_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PyVWF Training Quickstart - Denmark Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Denmark, 2020, calculated roughness
  python examples/pyvwf_quickstart_denmark.py
  
  # Different test year
  python examples/pyvwf_quickstart_denmark.py --year-test 2019
  
  # Use ERA5 forecast roughness instead of calculated
  python examples/pyvwf_quickstart_denmark.py --no-calc-z0
  
  # More clusters and finer time resolution
  python examples/pyvwf_quickstart_denmark.py --clusters 10 --time-res bimonth
  
  # Offshore only
  python examples/pyvwf_quickstart_denmark.py --cluster-mode offshore
        """
    )
    
    parser.add_argument(
        "--country",
        type=str,
        default="DK",
        help="Country code (default: DK for Denmark)",
    )
    parser.add_argument(
        "--year-test",
        type=int,
        default=2020,
        help="Test year for simulation (default: 2020)",
    )
    parser.add_argument(
        "--calc-z0",
        action="store_true",
        default=True,
        help="Calculate surface roughness from wind profiles (default: True)",
    )
    parser.add_argument(
        "--no-calc-z0",
        action="store_false",
        dest="calc_z0",
        help="Use ERA5 forecast surface roughness instead",
    )
    parser.add_argument(
        "--cluster-mode",
        type=str,
        choices=["all", "onshore", "offshore"],
        default="onshore",
        help="Clustering mode (default: onshore)",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
        help="Number of clusters (default: 5)",
    )
    parser.add_argument(
        "--time-res",
        type=str,
        choices=["fixed", "season", "bimonth", "month"],
        default="month",
        help="Time resolution for corrections (default: month)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: auto-generated)",
    )
    
    args = parser.parse_args()
    
    # Print banner
    print("\n" + "="*70)
    print("  PyVWF - Python Virtual Wind Farm Model")
    print("  Training Quickstart - Denmark Example")
    print("="*70)
    
    # Check data availability
    if not check_data_availability(args.country, args.year_test):
        print("\n⚠ Warning: Some data files are missing.")
        print("  The script will continue but may fail if required files are not found.")
        response = input("\nContinue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Run training
    try:
        output_dir = run_pyvwf_training(
            country=args.country,
            year_test=args.year_test,
            calc_z0=args.calc_z0,
            cluster_mode=args.cluster_mode,
            n_clusters=args.clusters,
            time_res=args.time_res,
            output_dir=args.output_dir,
        )
        
        print("\n" + "="*70)
        print("  ✓ SUCCESS - PyVWF training completed!")
        print("="*70)
        print(f"\nResults saved to: {output_dir}")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: Required file not found - {e}")
        print("\nPlease ensure all required data files are available.")
        print("See README.md for data preparation instructions.")
        return 1
        
    except Exception as e:
        print(f"\n✗ Error during PyVWF training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
