"""Run Country-Level PyVWF Workflow for All Countries.

This script processes NL, FR, BE, NO, and Phase 1 countries (ES, SE, IT, PT, IE) using the integrated country-level workflow.

Prerequisites:
    1. Run vwf/datasets/generate_country_level_training_data.py first
    2. Ensure ERA5 reanalysis data is available for all countries

Usage:
    # Run all countries with default settings
    python scripts/run_all_countries.py

    # Specify time resolutions
    python scripts/run_all_countries.py --time-res fixed season

    # Run only specific countries
    python scripts/run_all_countries.py --countries NL FR ES SE

    # Dry run (check config without training)
    python scripts/run_all_countries.py --dry-run
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vwf.vwf import PyVWF


def run_country(
    country: str,
    time_res_list: list[str],
    config_dir: str = "input/country_level_data",
    dry_run: bool = False,
) -> dict:
    """Run complete workflow for a single country.

    Args:
        country: Country code (NL, FR, BE, NO).
        time_res_list: List of temporal resolutions.
        config_dir: Configuration directory.
        dry_run: If True, only load config without training.

    Returns:
        Dictionary with results and timing info.
    """
    print("\n" + "=" * 80)
    print(f"Processing {country}")
    print("=" * 80)

    start_time = time.time()

    try:
        # Load model from config
        print(f"\n[{country}] Loading configuration...")
        model = PyVWF.from_config(
            country_code=country,
            path="",
            config_dir=config_dir,
            time_res_list=time_res_list,
        )

        if dry_run:
            print(f"\n[{country}] ✓ Dry run - configuration loaded successfully")
            return {
                "country": country,
                "status": "dry_run_success",
                "time": time.time() - start_time,
            }

        # Train model
        print(f"\n[{country}] Training bias corrections...")
        train_start = time.time()
        model.train(check=False)
        train_time = time.time() - train_start

        # Get test year from config
        sys.path.insert(0, config_dir)
        from pyvwf_config import get_config
        config = get_config(country)
        test_year = config["test_year"]

        # Simulate test year
        print(f"\n[{country}] Simulating test year {test_year}...")
        sim_start = time.time()
        model.simulate_cf(test_year)
        sim_time = time.time() - sim_start

        total_time = time.time() - start_time

        print(f"\n[{country}] ✓ Completed successfully!")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Simulation time: {sim_time:.2f}s")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Results saved in: {model.directory_path}")

        return {
            "country": country,
            "status": "success",
            "test_year": test_year,
            "train_time": train_time,
            "sim_time": sim_time,
            "total_time": total_time,
            "output_dir": model.directory_path,
        }

    except FileNotFoundError as e:
        print(f"\n[{country}] ✗ Error: Configuration not found")
        print(f"  {e}")
        print("  Run: python vwf/datasets/generate_country_level_training_data.py")
        return {
            "country": country,
            "status": "config_not_found",
            "error": str(e),
            "time": time.time() - start_time,
        }

    except Exception as e:
        print(f"\n[{country}] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "country": country,
            "status": "error",
            "error": str(e),
            "time": time.time() - start_time,
        }


def print_summary(results: list[dict]):
    """Print summary of all results.

    Args:
        results: List of result dictionaries from run_country().
    """
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successes = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]
    config_errors = [r for r in results if r["status"] == "config_not_found"]
    dry_runs = [r for r in results if r["status"] == "dry_run_success"]

    print(f"\nTotal countries processed: {len(results)}")
    print(f"  ✓ Successful: {len(successes)}")
    print(f"  ✗ Errors: {len(errors)}")
    print(f"  ⚠ Config not found: {len(config_errors)}")
    if dry_runs:
        print(f"  ℹ Dry runs: {len(dry_runs)}")

    if successes:
        print("\n" + "-" * 80)
        print("Successful Countries:")
        print("-" * 80)
        print(f"{'Country':<10} {'Test Year':<12} {'Train Time':<12} {'Sim Time':<12} {'Total Time':<12}")
        print("-" * 80)
        for r in successes:
            print(
                f"{r['country']:<10} "
                f"{r['test_year']:<12} "
                f"{r['train_time']:>10.2f}s  "
                f"{r['sim_time']:>10.2f}s  "
                f"{r['total_time']:>10.2f}s"
            )

        total_train_time = sum(r["train_time"] for r in successes)
        total_sim_time = sum(r["sim_time"] for r in successes)
        total_time = sum(r["total_time"] for r in successes)

        print("-" * 80)
        print(
            f"{'TOTAL':<10} {'':12} "
            f"{total_train_time:>10.2f}s  "
            f"{total_sim_time:>10.2f}s  "
            f"{total_time:>10.2f}s"
        )

    if errors:
        print("\n" + "-" * 80)
        print("Errors:")
        print("-" * 80)
        for r in errors:
            print(f"  {r['country']}: {r['error']}")

    if config_errors:
        print("\n" + "-" * 80)
        print("Configuration Not Found:")
        print("-" * 80)
        for r in config_errors:
            print(f"  {r['country']}: Run data generation script first")

    # Output directories
    if successes:
        print("\n" + "-" * 80)
        print("Output Directories:")
        print("-" * 80)
        for r in successes:
            print(f"  {r['country']}: {r['output_dir']}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Run country-level PyVWF workflow for all countries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=["NL", "FR", "BE", "NO", "ES", "SE", "IT", "PT", "IE"],
        choices=["NL", "FR", "BE", "NO", "ES", "SE", "IT", "PT", "IE"],
        help="Countries to process (default: NL FR BE NO ES SE IT PT IE)",
    )
    parser.add_argument(
        "--time-res",
        nargs="+",
        default=["fixed"],
        choices=["fixed", "season", "bimonth", "month"],
        help="Temporal resolutions (default: fixed season)",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="input/country_level_data",
        help="Configuration directory (default: input/country_level_data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load configurations without training",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=True,
        help="Continue processing other countries if one fails (default: True)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Country-Level PyVWF Workflow - Batch Processing")
    print("=" * 80)
    print(f"\nCountries: {', '.join(args.countries)}")
    print(f"Time resolutions: {', '.join(args.time_res)}")
    print(f"Configuration directory: {args.config_dir}")
    if args.dry_run:
        print("\nℹ DRY RUN MODE - No training will be performed")

    results = []
    total_start = time.time()

    for country in args.countries:
        result = run_country(
            country=country,
            time_res_list=args.time_res,
            config_dir=args.config_dir,
            dry_run=args.dry_run,
        )
        results.append(result)

        # Stop if error and continue_on_error is False
        if result["status"] == "error" and not args.continue_on_error:
            print(f"\n✗ Stopping due to error in {country}")
            break

    total_time = time.time() - total_start

    # Print summary
    print_summary(results)

    print("\n" + "=" * 80)
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print("=" * 80)

    # Exit code based on results
    if any(r["status"] == "error" for r in results):
        return 1
    if any(r["status"] == "config_not_found" for r in results):
        print("\n⚠ Some configurations were not found.")
        print("Run: python vwf/datasets/generate_country_level_training_data.py")
        return 2

    print("\n✓ All countries processed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
