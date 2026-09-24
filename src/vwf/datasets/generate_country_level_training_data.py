"""Generate the inputs of the country-level regions.

For NL, FR, BE, NO, ES, SE, IT, PT and IE this builds two things, each in its
own module:

1. grid points with a representative turbine and their clusters
   (:mod:`vwf.datasets.country_grid`);
2. ENTSO-E observations for the training years and the test year
   (:mod:`vwf.datasets.entsoe_country_obs`).

The harness reads both through each region's config in ``configs/regions/``.
Until 2026-09-24 a third step wrote a config module for the legacy batch path,
which was removed.

The observation fetch needs the ``data`` extra and an ENTSO-E API key, passed
in the environment for the one command.

Usage, from the repository root:

    ENTSOE_API_KEY=<key> PYTHONPATH=src python -m vwf.datasets.generate_country_level_training_data
    PYTHONPATH=src python -m vwf.datasets.generate_country_level_training_data \
        --countries NL FR --train-years 2018 2019 --test-year 2020
    PYTHONPATH=src python -m vwf.datasets.generate_country_level_training_data \
        --skip-observations
"""

import argparse
import os
import sys
from pathlib import Path

from vwf.datasets.country_grid import COUNTRY_CONFIGS, generate_grid_points
from vwf.datasets.entsoe_country_obs import fetch_observations
from vwf.datasets.fetch_entsoe_capacity_factors import ENTSOEWindDataFetcher


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Generate training and test data for country-level PyVWF workflows",
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
        "--train-years",
        nargs="+",
        type=int,
        default=[2015, 2016, 2017, 2018],
        help="Training years (default: 2015 2016 2017 2018)",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=2019,
        help="Test year (default: 2019)",
    )
    parser.add_argument(
        "--psr-type",
        choices=["onshore", "offshore", "all"],
        default="onshore",
        help="Wind type (default: onshore)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input/observations/country"),
        help="Output directory (default: input/observations/country)",
    )
    parser.add_argument(
        "--skip-observations",
        action="store_true",
        help="Skip fetching ENTSO-E observations (only generate grids)",
    )
    parser.add_argument(
        "--skip-grids",
        action="store_true",
        help="Skip generating grid points (only fetch observations)",
    )

    args = parser.parse_args()

    # Check API key
    if not args.skip_observations and not os.getenv("ENTSOE_API_KEY"):
        print("=" * 70)
        print("ERROR: ENTSOE_API_KEY environment variable not set!")
        print("=" * 70)
        print("\nYou need an ENTSO-E API key to fetch observations.")
        print("\nSteps:")
        print("1. Register at: https://transparency.entsoe.eu/")
        print("2. Generate API key in your account")
        print("3. Set environment variable:")
        print("   export ENTSOE_API_KEY='your-key-here'")
        print("\nAlternatively, use --skip-observations to only generate grid points.")
        return 1

    # Validate years
    if args.test_year in args.train_years:
        print("ERROR: Test year cannot be in training years!")
        return 1

    if max(args.train_years) >= args.test_year:
        print("Warning: Training years should be before test year!")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Country-Level Training Data Generation for PyVWF")
    print("=" * 70)
    print(f"\nCountries: {', '.join(args.countries)}")
    print(f"Training years: {min(args.train_years)}-{max(args.train_years)}")
    print(f"Test year: {args.test_year}")
    print(f"Output directory: {args.output_dir}")
    print(f"Wind type: {args.psr_type}")

    # Generate grid points
    if not args.skip_grids:
        print("\n" + "=" * 70)
        print("STEP 1: Generate Grid Points with Turbine Metadata")
        print("=" * 70)

        for country in args.countries:
            config = COUNTRY_CONFIGS.get(country.upper())
            if config is None:
                print(f"\n✗ No configuration for {country}")
                continue

            try:
                grid_points, cluster_geoms = generate_grid_points(
                    country=country.upper(),
                    config=config,
                    output_dir=args.output_dir,
                    save_geojson=True,
                )
            except Exception as e:
                print(f"\n✗ Error generating grid for {country}: {e}")
                import traceback

                traceback.print_exc()

    # Fetch observations
    if not args.skip_observations:
        print("\n" + "=" * 70)
        print("STEP 2: Fetch Country-Level Observations from ENTSO-E")
        print("=" * 70)

        try:
            fetcher = ENTSOEWindDataFetcher()
            fetch_observations(
                fetcher=fetcher,
                countries=[c.upper() for c in args.countries],
                train_years=args.train_years,
                test_year=args.test_year,
                output_dir=args.output_dir,
                psr_type=args.psr_type,
            )
        except Exception as e:
            print(f"\n✗ Error fetching observations: {e}")
            import traceback

            traceback.print_exc()
            return 1

    # Print summary
    print("\n" + "=" * 70)
    print("✓ DATA GENERATION COMPLETE")
    print("=" * 70)

    print(f"\nGenerated files in: {args.output_dir}")
    print("\nDirectory structure:")
    print(f"  {args.output_dir}/")
    print("    ├── grid_points/")
    for country in args.countries:
        print(f"    │   ├── {country.lower()}/")
        print(f"    │   │   ├── {country.lower()}_grid_points.csv")
        print(f"    │   │   └── {country.lower()}_correction_regions.geojson")

    if not args.skip_observations:
        print("    ├── observations/")
        for country in args.countries:
            print(f"    │   ├── {country.lower()}/")
            print(
                f"    │   │   ├── {country.lower()}_train_{min(args.train_years)}_{max(args.train_years)}.csv"
            )
            print(f"    │   │   └── {country.lower()}_test_{args.test_year}.csv")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)

    print("""
1. Train and evaluate a country through the harness, from its region config:

   pyvwf-validate train --region configs/regions/<code>.toml
   pyvwf-validate evaluate --region configs/regions/<code>.toml \\
       --train-run output/validation/<CODE>/train-<run>

2. Visualise the correction regions:

   import geopandas as gpd
   gpd.read_file("<output_dir>/grid_points/<code>/<code>_correction_regions.geojson").plot(
       column="cluster", cmap="Set3", edgecolor="black"
   )

3. For Sweden, the bidding-zone region (configs/regions/se_bz.toml) fits
   per-zone observations instead of KMeans clusters.
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
