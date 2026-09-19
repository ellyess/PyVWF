"""ENTSO-E observations for the country-level regions.

Fetches national, or per-zone, wind capacity factors for the training years
and the test year, and writes them under ``<output>/observations/<country>/``.
Norway and Sweden are fetched zone by zone and also summed to a national
series. Needs the ``data`` extra and an ENTSO-E API key in the environment.

Split from ``generate_country_level_training_data.py``, whose ``main`` runs it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from vwf.datasets.country_grid import NORWAY_ZONES, SWEDEN_ZONES
from vwf.datasets.fetch_entsoe_capacity_factors import ENTSOEWindDataFetcher


def fetch_observations(
    fetcher: ENTSOEWindDataFetcher,
    countries: list[str],
    train_years: list[int],
    test_year: int,
    output_dir: Path,
    psr_type: Literal["onshore", "offshore", "all"] = "all",
) -> dict:
    """Fetch ENTSO-E observations for training and test periods.

    For Norway and Sweden, automatically fetches each bidding zone separately.

    Args:
        fetcher: ENTSO-E API client.
        countries: List of country codes.
        train_years: List of training years.
        test_year: Test year.
        output_dir: Output directory.
        psr_type: Wind type ('onshore', 'offshore', 'all').

    Returns:
        Dictionary with training and test data for each country.
    """
    results = {}

    for country in countries:
        print(f"\n{'=' * 70}")
        print(f"Fetching Observations for {country}")
        print(f"{'=' * 70}")

        # Special handling for Norway - fetch zones separately
        if country.upper() == "NO":
            zone_results = fetch_norway_zone_observations(
                fetcher, train_years, test_year, output_dir, psr_type
            )
            results["NO"] = zone_results
            continue

        # Special handling for Sweden - fetch zones separately
        if country.upper() == "SE":
            zone_results = fetch_sweden_zone_observations(
                fetcher, train_years, test_year, output_dir, psr_type
            )
            results["SE"] = zone_results
            continue

        # Standard country-level fetch
        results[country] = fetch_country_observations(
            fetcher, country, train_years, test_year, output_dir, psr_type
        )

    return results


def fetch_norway_zone_observations(
    fetcher: ENTSOEWindDataFetcher,
    train_years: list[int],
    test_year: int,
    output_dir: Path,
    psr_type: Literal["onshore", "offshore", "all"] = "all",
) -> dict:
    """Fetch observations for Norwegian bidding zones.

    Args:
        fetcher: ENTSO-E API client.
        train_years: List of training years.
        test_year: Test year.
        output_dir: Output directory.
        psr_type: Wind type.

    Returns:
        Dictionary with zone data and aggregated country data.
    """
    print("\n✓ Norway detected - fetching bidding zones separately")

    zone_train_data = {}
    zone_test_data = {}

    # Fetch each zone
    for zone_id in NORWAY_ZONES.keys():
        print(f"\n{'─' * 70}")
        print(f"Fetching {zone_id}")
        print(f"{'─' * 70}")

        zone_results = fetch_country_observations(
            fetcher, zone_id, train_years, test_year, output_dir, psr_type
        )

        if zone_results:
            zone_train_data[zone_id] = zone_results["train"]
            zone_test_data[zone_id] = zone_results["test"]

    # Aggregate zones for country-level
    print(f"\n{'─' * 70}")
    print("Aggregating zones to country-level")
    print(f"{'─' * 70}")

    if zone_train_data:
        # Combine all zones with sum
        train_combined = pd.concat(
            [df[["generation_mw", "capacity_mw"]] for df in zone_train_data.values()], axis=1
        )

        train_agg = pd.DataFrame(
            {
                "generation_mw": train_combined.filter(like="generation").sum(axis=1),
                "capacity_mw": train_combined.filter(like="capacity").sum(axis=1),
            }
        )
        train_agg["capacity_factor"] = train_agg["generation_mw"] / train_agg["capacity_mw"]
        train_agg["capacity_factor"] = train_agg["capacity_factor"].clip(0, 1.5)

        print(f"  ✓ Aggregated training: {len(train_agg)} data points")
        print(f"    Mean CF: {train_agg['capacity_factor'].mean():.2%}")

    else:
        train_agg = pd.DataFrame()

    if zone_test_data:
        test_combined = pd.concat(
            [df[["generation_mw", "capacity_mw"]] for df in zone_test_data.values()], axis=1
        )

        test_agg = pd.DataFrame(
            {
                "generation_mw": test_combined.filter(like="generation").sum(axis=1),
                "capacity_mw": test_combined.filter(like="capacity").sum(axis=1),
            }
        )
        test_agg["capacity_factor"] = test_agg["generation_mw"] / test_agg["capacity_mw"]
        test_agg["capacity_factor"] = test_agg["capacity_factor"].clip(0, 1.5)

        print(f"  ✓ Aggregated test: {len(test_agg)} data points")
        print(f"    Mean CF: {test_agg['capacity_factor'].mean():.2%}")

    else:
        test_agg = pd.DataFrame()

    # Save aggregated country-level data
    obs_dir = output_dir / "observations" / "no"
    obs_dir.mkdir(parents=True, exist_ok=True)

    if not train_agg.empty:
        train_path = obs_dir / f"no_train_{min(train_years)}_{max(train_years)}_aggregated.csv"
        train_agg.to_csv(train_path)
        print(f"\n✓ Saved aggregated training: {train_path}")

    if not test_agg.empty:
        test_path = obs_dir / f"no_test_{test_year}_aggregated.csv"
        test_agg.to_csv(test_path)
        print(f"✓ Saved aggregated test: {test_path}")

    return {
        "zones": {
            "train": zone_train_data,
            "test": zone_test_data,
        },
        "aggregated": {
            "train": train_agg,
            "test": test_agg,
        },
    }


def fetch_sweden_zone_observations(
    fetcher: ENTSOEWindDataFetcher,
    train_years: list[int],
    test_year: int,
    output_dir: Path,
    psr_type: Literal["onshore", "offshore", "all"] = "all",
) -> dict:
    """Fetch observations for Swedish bidding zones.

    Args:
        fetcher: ENTSO-E API client.
        train_years: List of training years.
        test_year: Test year.
        output_dir: Output directory.
        psr_type: Wind type.

    Returns:
        Dictionary with zone data and aggregated country data.
    """
    print("\n✓ Sweden detected - fetching bidding zones separately")

    zone_train_data = {}
    zone_test_data = {}

    # Fetch each zone
    for zone_id in SWEDEN_ZONES.keys():
        print(f"\n{'─' * 70}")
        print(f"Fetching {zone_id}")
        print(f"{'─' * 70}")

        zone_results = fetch_country_observations(
            fetcher, zone_id, train_years, test_year, output_dir, psr_type
        )

        if zone_results:
            zone_train_data[zone_id] = zone_results["train"]
            zone_test_data[zone_id] = zone_results["test"]

    # Aggregate zones for country-level
    print(f"\n{'─' * 70}")
    print("Aggregating zones to country-level")
    print(f"{'─' * 70}")

    if zone_train_data:
        # Combine all zones with sum
        train_combined = pd.concat(
            [df[["generation_mw", "capacity_mw"]] for df in zone_train_data.values()], axis=1
        )

        train_agg = pd.DataFrame(
            {
                "generation_mw": train_combined.filter(like="generation").sum(axis=1),
                "capacity_mw": train_combined.filter(like="capacity").sum(axis=1),
            }
        )
        train_agg["capacity_factor"] = train_agg["generation_mw"] / train_agg["capacity_mw"]
        train_agg["capacity_factor"] = train_agg["capacity_factor"].clip(0, 1.5)

        print(f"  ✓ Aggregated training: {len(train_agg)} data points")
        print(f"    Mean CF: {train_agg['capacity_factor'].mean():.2%}")

    else:
        train_agg = pd.DataFrame()

    if zone_test_data:
        test_combined = pd.concat(
            [df[["generation_mw", "capacity_mw"]] for df in zone_test_data.values()], axis=1
        )

        test_agg = pd.DataFrame(
            {
                "generation_mw": test_combined.filter(like="generation").sum(axis=1),
                "capacity_mw": test_combined.filter(like="capacity").sum(axis=1),
            }
        )
        test_agg["capacity_factor"] = test_agg["generation_mw"] / test_agg["capacity_mw"]
        test_agg["capacity_factor"] = test_agg["capacity_factor"].clip(0, 1.5)

        print(f"  ✓ Aggregated test: {len(test_agg)} data points")
        print(f"    Mean CF: {test_agg['capacity_factor'].mean():.2%}")

    else:
        test_agg = pd.DataFrame()

    # Save aggregated country-level data
    obs_dir = output_dir / "observations" / "se"
    obs_dir.mkdir(parents=True, exist_ok=True)

    if not train_agg.empty:
        train_path = obs_dir / f"se_train_{min(train_years)}_{max(train_years)}_aggregated.csv"
        train_agg.to_csv(train_path)
        print(f"\n✓ Saved aggregated training: {train_path}")

    if not test_agg.empty:
        test_path = obs_dir / f"se_test_{test_year}_aggregated.csv"
        test_agg.to_csv(test_path)
        print(f"✓ Saved aggregated test: {test_path}")

    return {
        "zones": {
            "train": zone_train_data,
            "test": zone_test_data,
        },
        "aggregated": {
            "train": train_agg,
            "test": test_agg,
        },
    }


def fetch_country_observations(
    fetcher: ENTSOEWindDataFetcher,
    country: str,
    train_years: list[int],
    test_year: int,
    output_dir: Path,
    psr_type: Literal["onshore", "offshore", "all"] = "all",
) -> dict:
    """Fetch observations for a single country or zone.

    Args:
        fetcher: ENTSO-E API client.
        country: Country or zone code.
        train_years: List of training years.
        test_year: Test year.
        output_dir: Output directory.
        psr_type: Wind type.

    Returns:
        Dictionary with train and test data, or None if no data.
    """
    # Training data
    print(f"\nTraining period: {min(train_years)}-{max(train_years)}")
    train_start = pd.Timestamp(f"{min(train_years)}-01-01", tz="UTC")
    train_end = pd.Timestamp(f"{max(train_years)}-12-31 23:59", tz="UTC")

    train_data = fetcher.calculate_capacity_factor(
        country=country,
        start=train_start,
        end=train_end,
        psr_type=psr_type,
    )

    if train_data.empty:
        print(f"  ✗ No training data for {country}")
        return None

    print(f"  ✓ Training: {len(train_data)} data points")
    print(f"    Mean CF: {train_data['capacity_factor'].mean():.2%}")
    print(f"    Capacity: {train_data['capacity_mw'].mean():.0f} MW")

    # Test data
    print(f"\nTest period: {test_year}")
    test_start = pd.Timestamp(f"{test_year}-01-01", tz="UTC")
    test_end = pd.Timestamp(f"{test_year}-12-31 23:59", tz="UTC")

    test_data = fetcher.calculate_capacity_factor(
        country=country,
        start=test_start,
        end=test_end,
        psr_type=psr_type,
    )

    if test_data.empty:
        print(f"  ✗ No test data for {country}")
        return None

    print(f"  ✓ Test: {len(test_data)} data points")
    print(f"    Mean CF: {test_data['capacity_factor'].mean():.2%}")
    print(f"    Capacity: {test_data['capacity_mw'].mean():.0f} MW")

    # Save training and test data separately
    obs_dir = output_dir / "observations" / country.lower()
    obs_dir.mkdir(parents=True, exist_ok=True)

    train_path = obs_dir / f"{country.lower()}_train_{min(train_years)}_{max(train_years)}.csv"
    train_data.to_csv(train_path)
    print(f"\n✓ Saved training data: {train_path}")

    test_path = obs_dir / f"{country.lower()}_test_{test_year}.csv"
    test_data.to_csv(test_path)
    print(f"✓ Saved test data: {test_path}")

    return {
        "train": train_data,
        "test": test_data,
    }
