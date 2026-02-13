#!/usr/bin/env python3
"""Train all bias correction configurations for PyVWF.

This script trains bias correction factors for:
- Turbine-level workflows (DK, DE, UK)
- Country-level workflows (NL, FR, BE, NO, SE, ES, IT, PT, IE)
- Multiple temporal resolutions (fixed, season, bimonth)
- Multiple spatial clusters

Output structure: output/runs/<prefix>/<country-run>/
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import traceback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "input" / "country_level_data"))

from vwf.vwf import PyVWF
try:
    from pyvwf_config import get_all_configs
    HAS_COUNTRY_CONFIG = True
except ImportError:
    HAS_COUNTRY_CONFIG = False
    print("Warning: Country-level config not found. Only turbine-level will be available.")


# ============================================================================
# TRAINING CONFIGURATIONS
# ============================================================================

def get_training_sets():
    """Define all training set configurations.

    Returns:
        Dict mapping prefix to configuration dict.
    """
    configs = {}

    # -------------------------------------------------------------------------
    # TURBINE-LEVEL: Simple baseline (fixed temporal, various clusters)
    # -------------------------------------------------------------------------
    configs['turbine_fixed_2015_2019'] = {
        'name': 'Turbine-Level Simple (2015-2019 → 2020)',
        'obs_level': 'turbine',
        'train_years': (2015, 2019),
        'test_year': 2020,
        'countries': ['DK', 'DE', 'UK'],
        'cluster_mode': 'all',
        'cluster_list': [5, 10, 15],
        'time_res_list': ['fixed'],
        'calc_z0': True,
    }

    # -------------------------------------------------------------------------
    # TURBINE-LEVEL: Denmark Research Run (2015-2019 → 2020)
    # -------------------------------------------------------------------------
    configs['turbine_dk_research'] = {
        'name': 'Denmark Turbine-Level Research Run (2015-2019 → 2020)',
        'obs_level': 'turbine',
        'train_years': (2015, 2019),
        'test_year': 2020,
        'countries': ['DK_onshore','DK_offshore'],  # Only DK has data up to 2023
        'cluster_mode': 'all',
        'cluster_list': [10],
        'time_res_list': ['fixed', 'season', 'bimonth', 'month'],
        'calc_z0': True,
        'per_country_config': {
            'DK_onshore': { 'cluster_list': [1,2,3,5,7,10,20,50,70,100,200,500,700,800,900,1000,2000,3000,3300]}, 
            'DK_offshore': { 'cluster_list': [1,2,3,5]},
        }
    }
    
    configs['turbine_dk_research_fixed'] = {
        'name': 'Denmark Turbine-Level Research Run (2015-2019 → 2020)',
        'obs_level': 'turbine',
        'train_years': (2015, 2019),
        'test_year': 2020,
        'countries': ['DK'],  # Only DK has data up to 2023
        'cluster_mode': 'onshore',
        'cluster_list': [1,10,100,500,700,1000],
        'time_res_list': ['fixed'],
        'calc_z0': True,
        }
    # -------------------------------------------------------------------------
    # GRID RESEARCH
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # TURBINE-LEVEL
    # -------------------------------------------------------------------------
    # This example shows how to train the same country for onshore and offshore separately
    configs['turbine_grid'] = {
        'name': 'Turbine-Level Grid Research',
        'obs_level': 'turbine',
        # Base/default configuration (used if not overridden)
        'train_years': (2015, 2019),
        'test_year': 2020,
        'countries': ['DK_onshore', 'DK_offshore', 'UK_onshore', 'UK_offshore','DE_onshore'],
        'cluster_mode': 'all',  # Overridden by _onshore/_offshore suffix
        'cluster_list': [10],
        'time_res_list': ['fixed'],
        'calc_z0': True,
        # Per-variant overrides (optional)
        'per_country_config': {
            'DK_onshore': {
                # DK onshore: Test more clusters
                'cluster_list': [700],
                'train_years': (2015, 2019),
                'test_year': 2020,
            },
            'DK_offshore': {
                # DK offshore: Fewer clusters (less complex)
                'cluster_list': [2],
                'train_years': (2015, 2019),
                'test_year': 2020,
            },
            'UK_onshore': {
                # UK onshore: Default
                'cluster_list': [300],
                'train_years': (2015, 2018),
                'test_year': 2019,
            },
            'UK_offshore': {
                # UK offshore: More clusters (large offshore presence)
                'cluster_list': [10],
                'train_years': (2015, 2018),
                'test_year': 2019,
            },
            'DE_onshore': {
                # DE onshore: Default
                'cluster_list':[500],
                'train_years': (2015, 2018),
                'test_year': 2019,
            },
        }
    }
    # -------------------------------------------------------------------------
    # COUNTRY-LEVEL
    # -------------------------------------------------------------------------
    if HAS_COUNTRY_CONFIG:
        configs['country_grid_2015_2021_2023'] = {
            'name': 'Country-Level Grid Research (2015-2021 → 2023)',
            'obs_level': 'country',
            'train_years': (2015, 2021),
            'test_year': 2023,
            'countries': ['NL', 'FR', 'BE', 'NO', 'SE', 'ES', 'IT', 'PT', 'IE'],
            'use_config': True,
            'use_year_specific_weighting': True,  # Toggle year-specific capacity weighting
        }
        
        configs['country_grid_2015_2021_2023_no_weight'] = {
            'name': 'Country-Level Grid Research (2015-2021 → 2023) - No Weighting',
            'obs_level': 'country',
            'train_years': (2015, 2021),
            'test_year': 2023,
            'countries': ['NL', 'FR', 'BE', 'NO', 'SE', 'ES', 'IT', 'PT', 'IE'],
            'use_config': True,
            'use_year_specific_weighting': False,  # Static grid points (no year-specific)
        }

    return configs


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def parse_country_variant(country_key: str) -> tuple[str, str | None]:
    """Parse country key that may include a variant suffix.

    Supports formats:
    - 'DK' -> ('DK', None)
    - 'DK_onshore' -> ('DK', 'onshore')
    - 'DK_offshore' -> ('DK', 'offshore')

    Args:
        country_key: Country code, optionally with _onshore or _offshore suffix

    Returns:
        Tuple of (base_country_code, variant_cluster_mode or None)
    """
    if '_onshore' in country_key:
        base_country = country_key.replace('_onshore', '')
        return base_country, 'onshore'
    elif '_offshore' in country_key:
        base_country = country_key.replace('_offshore', '')
        return base_country, 'offshore'
    else:
        return country_key, None


def get_country_config(country_key: str, base_config: dict) -> dict:
    """Get configuration for a specific country with per-country overrides.

    Supports country variants like 'DK_onshore' or 'DK_offshore' to train
    the same country with different cluster modes.

    Args:
        country_key: Country code, optionally with _onshore/_offshore suffix
        base_config: Base configuration with defaults

    Returns:
        Configuration dict with country-specific overrides applied
    """
    # Start with base config
    country_cfg = base_config.copy()

    # Parse country variant (e.g., 'DK_onshore' -> 'DK', 'onshore')
    base_country, variant_mode = parse_country_variant(country_key)

    # If variant has a cluster mode specified, override the base
    if variant_mode is not None:
        country_cfg['cluster_mode'] = variant_mode

    # Apply per-country overrides if defined
    if 'per_country_config' in base_config and country_key in base_config['per_country_config']:
        overrides = base_config['per_country_config'][country_key]
        country_cfg.update(overrides)

    # Store the base country code for data loading
    country_cfg['_base_country'] = base_country
    country_cfg['_country_key'] = country_key

    return country_cfg


def train_turbine_level(country_key: str, config: dict, output_dir: Path, args) -> dict:
    """Train turbine-level bias corrections.

    Supports country variants like 'DK_onshore' or 'DK_offshore' to train
    the same country with different cluster modes in separate runs.

    Args:
        country_key: Country code, optionally with _onshore/_offshore suffix
        config: Training configuration (may include per_country_config)
        output_dir: Output directory for this run

    Returns:
        Result dictionary with status
    """
    # Get country-specific configuration (handles variants)
    country_cfg = get_country_config(country_key, config)
    base_country = country_cfg['_base_country']  # e.g., 'DK' from 'DK_onshore'

    print(f"\n{'='*80}")
    print(f"TRAINING TURBINE-LEVEL: {country_key}")
    print(f"{'='*80}")
    print(f"Base country: {base_country}")
    print(f"Training years: {country_cfg['train_years'][0]}-{country_cfg['train_years'][1]}")
    print(f"Test year: {country_cfg['test_year']}")
    print(f"Clusters: {country_cfg['cluster_list']}")
    print(f"Time resolutions: {country_cfg['time_res_list']}")
    print(f"Cluster mode: {country_cfg['cluster_mode']}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")

    try:
        # Initialize PyVWF with base country code for data loading
        model = PyVWF(
            path=str(output_dir),
            country=base_country,  # Use base country for data loading
            correct=True,
            calc_z0=country_cfg['calc_z0'],
            cluster_mode=country_cfg['cluster_mode'],
            cluster_list=country_cfg['cluster_list'],
            time_res_list=country_cfg['time_res_list'],
            obs_level='turbine',
        )

        # Train
        print(f"\nTraining {country_key}...")
        model.train(
            check=False,
            dask_n_workers=args.dask_n_workers,
            dask_threads_per_worker=args.dask_threads_per_worker,
            dask_use_processes=not args.dask_no_processes,
            dask_use_distributed=not args.dask_no_distributed,
            dask_npartitions=args.dask_npartitions,
        )
        print(f"✓ Training completed for {country_key}")

        # Simulate
        print(f"\nSimulating {country_key} for year {country_cfg['test_year']}...")
        model.simulate_cf(country_cfg['test_year'])
        print(f"✓ Simulation completed for {country_key}")

        return {
            'country': country_key,
            'base_country': base_country,
            'status': 'success',
            'config': {
                'train_years': country_cfg['train_years'],
                'test_year': country_cfg['test_year'],
                'clusters': country_cfg['cluster_list'],
                'time_res': country_cfg['time_res_list'],
                'cluster_mode': country_cfg['cluster_mode'],
            }
        }

    except Exception as e:
        print(f"\n✗ Failed for {country_key}: {e}")
        traceback.print_exc()
        return {'country': country_key, 'status': 'failed', 'error': str(e)}


def train_country_level(country: str, config: dict, output_dir: Path,
                        country_configs: dict, args) -> dict:
    """Train country-level bias corrections with optional year-specific grid points.

    Can use either:
    - Year-specific grid points (use_year_specific_weighting=True): Loads capacity
      data for each year, accounting for the actual turbine fleet at the time.
    - Static grid points (use_year_specific_weighting=False): Uses single grid point
      file representing the fleet at one point in time.

    Args:
        country: Country code
        config: Training configuration (must include 'use_year_specific_weighting')
        output_dir: Output directory for this run
        country_configs: Country-specific configurations from pyvwf_config

    Returns:
        Result dictionary with status
    """
    print(f"\n{'='*80}")
    print(f"TRAINING COUNTRY-LEVEL: {country}")
    print(f"{'='*80}")

    country_config = country_configs[country]
    use_year_specific = config.get('use_year_specific_weighting', False)
    
    print(f"Country: {country_config['name']}")
    print(f"Training years: {config['train_years'][0]}-{config['train_years'][1]}")
    print(f"Test year: {config['test_year']}")
    print(f"Clusters: {country_config['cluster_list']}")
    print(f"Time resolutions: {country_config['time_res_list']}")
    print(f"Year-specific weighting: {use_year_specific}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}")

    try:
        # Adjust paths based on training years
        train_start = config['train_years'][0]
        train_end = config['train_years'][1]
        train_path = str(country_config['train_obs_path']).replace(
            '2015_2019', f'{train_start}_{train_end}'
        )

        if not Path(train_path).exists():
            # Fallback to original path
            train_path = country_config['train_obs_path']

        print(f"\nLoading observations for {country}...")
        obs_train = pd.read_csv(train_path, index_col=0, parse_dates=True)
        obs_test = pd.read_csv(country_config['test_obs_path'], index_col=0, parse_dates=True)

        # Ensure indices are DatetimeIndex and convert to UTC
        if not isinstance(obs_train.index, pd.DatetimeIndex):
            obs_train.index = pd.to_datetime(obs_train.index, utc=True)
        else:
            obs_train.index = obs_train.index.tz_convert('UTC') if obs_train.index.tz is not None else obs_train.index
            
        if not isinstance(obs_test.index, pd.DatetimeIndex):
            obs_test.index = pd.to_datetime(obs_test.index, utc=True)
        else:
            obs_test.index = obs_test.index.tz_convert('UTC') if obs_test.index.tz is not None else obs_test.index

        print(f"  Training obs: {len(obs_train)} timesteps ({obs_train.index.year.min()}-{obs_train.index.year.max()})")
        print(f"  Test obs: {len(obs_test)} timesteps")

        # Initialize PyVWF
        model = PyVWF(
            path=str(output_dir),
            country=country_config['country'],
            correct=True,
            calc_z0=country_config['calc_z0'],
            cluster_mode=country_config['cluster_mode'],
            cluster_list=country_config['cluster_list'],
            time_res_list=country_config['time_res_list'],
            obs_level='country',
        )

        # Load country data (with or without year-specific weighting)
        if use_year_specific:
            print(f"\nLoading year-specific grid points for {country}...")
            grid_points_dir = Path(country_config['grid_points_path']).parent
            model.load_country_data_with_year_specific(
                obs_train, 
                obs_test,
                grid_points_dir=grid_points_dir
            )
        else:
            print(f"\nLoading static grid points for {country}...")
            grid_points = pd.read_csv(country_config['grid_points_path'])
            print(f"  Grid points: {len(grid_points)} points")
            model.load_country_data(grid_points, obs_train, obs_test)

        # Train
        print(f"\nTraining {country}...")
        model.train(
            check=False,
            dask_n_workers=args.dask_n_workers,
            dask_threads_per_worker=args.dask_threads_per_worker,
            dask_use_processes=not args.dask_no_processes,
            dask_use_distributed=not args.dask_no_distributed,
            dask_npartitions=args.dask_npartitions,
        )
        print(f"✓ Training completed for {country}")

        # Simulate
        print(f"\nSimulating {country} for year {config['test_year']}...")
        model.simulate_cf(config['test_year'])
        print(f"✓ Simulation completed for {country}")

        return {'country': country, 'status': 'success'}

    except Exception as e:
        print(f"\n✗ Failed for {country}: {e}")
        traceback.print_exc()
        return {'country': country, 'status': 'failed', 'error': str(e)}


def train_configuration_set(prefix: str, config: dict, base_dir: Path,
                            country_configs: dict = None, args=None) -> list:
    """Train all countries in a configuration set.

    Args:
        prefix: Prefix for this training set
        config: Training configuration
        base_dir: Base output directory (out/runs)
        country_configs: Country-level configurations (if applicable)

    Returns:
        List of result dictionaries
    """
    print(f"\n{'#'*80}")
    print(f"TRAINING SET: {config['name']}")
    print(f"Prefix: {prefix}")
    print(f"{'#'*80}")

    # Create output directory for this set
    set_dir = base_dir / prefix
    set_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for country in config['countries']:
        if config['obs_level'] == 'turbine':
            result = train_turbine_level(country, config, set_dir, args)
        else:
            if country_configs is None:
                print(f"✗ Skipping {country}: No country configs available")
                results.append({'country': country, 'status': 'skipped',
                              'error': 'No country configs'})
                continue
            result = train_country_level(country, config, set_dir, country_configs, args)

        results.append(result)

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training orchestrator."""
    parser = argparse.ArgumentParser(
        description='Train all bias correction configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--sets',
        nargs='+',
        help='Training set prefixes to run (default: all)',
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default='output/runs',
        help='Base output directory (default: out/runs)',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available training sets and exit',
    )
    parser.add_argument(
        '--dask-n-workers',
        type=int,
        default=3,
        help='Number of Dask workers for offset optimization (default: 3)',
    )
    parser.add_argument(
        '--dask-threads-per-worker',
        type=int,
        default=1,
        help='Threads per Dask worker (default: 1)',
    )
    parser.add_argument(
        '--dask-no-processes',
        action='store_true',
        help='Use threads instead of processes for Dask workers',
    )
    parser.add_argument(
        '--dask-no-distributed',
        action='store_true',
        help='Disable Dask distributed LocalCluster and use scheduler="processes"',
    )
    parser.add_argument(
        '--dask-npartitions',
        type=int,
        default=0,
        help='Override Dask partition count (default: 4 * workers; 0 to auto)',
    )

    args = parser.parse_args()

    # Get all training sets
    training_sets = get_training_sets()

    # List available sets
    if args.list:
        print("Available training sets:\n")
        for prefix, config in training_sets.items():
            print(f"  {prefix}")
            print(f"    Name: {config['name']}")
            print(f"    Level: {config['obs_level']}")
            print(f"    Countries: {', '.join(config['countries'])}")
            print()
        return 0

    # Determine which sets to run
    if args.sets:
        sets_to_run = {k: v for k, v in training_sets.items() if k in args.sets}
        if not sets_to_run:
            print(f"Error: No matching training sets found for: {args.sets}")
            print(f"Available: {list(training_sets.keys())}")
            return 1
    else:
        sets_to_run = training_sets

    # Load country configs if available
    country_configs = None
    if HAS_COUNTRY_CONFIG:
        country_configs = get_all_configs()

    # Setup output directory
    base_dir = Path(args.outdir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    # Start timestamp
    start_time = datetime.now()

    print(f"\n{'#'*80}")
    print(f"BIAS CORRECTION TRAINING - FULL RUN")
    print(f"{'#'*80}")
    print(f"Output directory: {base_dir}")
    print(f"Training sets: {len(sets_to_run)}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}\n")

    # Train all sets
    all_results = {}
    for prefix, config in sets_to_run.items():
        try:
            results = train_configuration_set(prefix, config, base_dir, country_configs, args)
            all_results[prefix] = results
        except Exception as e:
            print(f"\n✗ Training set {prefix} failed: {e}")
            traceback.print_exc()
            all_results[prefix] = [{'country': 'ALL', 'status': 'failed', 'error': str(e)}]

    # End timestamp
    end_time = datetime.now()
    duration = end_time - start_time

    # Summary
    print(f"\n{'#'*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'#'*80}")
    print(f"Duration: {duration}")
    print(f"Output: {base_dir}")
    print(f"{'#'*80}\n")

    for prefix, results in all_results.items():
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] == 'failed']

        print(f"\n{prefix}:")
        print(f"  ✓ Successful: {len(successful)}/{len(results)}")
        for r in successful:
            print(f"      - {r['country']}")

        if failed:
            print(f"  ✗ Failed: {len(failed)}/{len(results)}")
            for r in failed:
                print(f"      - {r['country']}: {r.get('error', 'Unknown error')}")

    print(f"\n{'#'*80}\n")

    # Return exit code
    total_failed = sum(len([r for r in results if r['status'] == 'failed'])
                      for results in all_results.values())
    return 0 if total_failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
