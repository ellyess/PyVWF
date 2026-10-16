"""ENTSO-E observations for the country-level regions.

Fetches national, or per-zone, wind capacity factors for the training years
and the test year, and writes them under ``<output>/observations/<country>/``.
Norway and Sweden are fetched zone by zone and also summed to a national
series. Needs the ``data`` extra and an ENTSO-E API key in the environment.

Split from ``generate_country_level_training_data.py``, whose ``main`` runs it.
"""

from __future__ import annotations

import logging

from pathlib import Path
from typing import Any, Literal

import pandas as pd

from pyvwf.config import AGGREGATED_SUFFIX, country_obs_stem
from pyvwf.datasets.country_grid import NORWAY_ZONES, SWEDEN_ZONES
from pyvwf.datasets.fetch_entsoe_capacity_factors import (
    CAPACITY_FACTOR_CEILING,
    ENTSOEWindDataFetcher,
)

logger = logging.getLogger(__name__)


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
    results: dict[str, Any] = {}

    for country in countries:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Fetching Observations for {country}")
        logger.info(f"{'=' * 70}")

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
    """Norway's bidding zones, and their sum as the national series."""
    return fetch_zone_observations(
        "NO", NORWAY_ZONES, fetcher, train_years, test_year, output_dir, psr_type
    )


def fetch_sweden_zone_observations(
    fetcher: ENTSOEWindDataFetcher,
    train_years: list[int],
    test_year: int,
    output_dir: Path,
    psr_type: Literal["onshore", "offshore", "all"] = "all",
) -> dict:
    """Sweden's bidding zones, and their sum as the national series."""
    return fetch_zone_observations(
        "SE", SWEDEN_ZONES, fetcher, train_years, test_year, output_dir, psr_type
    )


def _sum_zones(zone_data: dict) -> pd.DataFrame:
    """Generation and capacity summed over the zones that returned data."""
    if not zone_data:
        return pd.DataFrame()
    combined = pd.concat(
        [df[["generation_mw", "capacity_mw"]] for df in zone_data.values()], axis=1
    )
    summed = pd.DataFrame(
        {
            "generation_mw": combined.filter(like="generation").sum(axis=1),
            "capacity_mw": combined.filter(like="capacity").sum(axis=1),
        }
    )
    summed["capacity_factor"] = summed["generation_mw"] / summed["capacity_mw"]
    summed["capacity_factor"] = summed["capacity_factor"].clip(0, CAPACITY_FACTOR_CEILING)
    return summed


def fetch_zone_observations(
    code: str,
    zones: dict,
    fetcher: ENTSOEWindDataFetcher,
    train_years: list[int],
    test_year: int,
    output_dir: Path,
    psr_type: Literal["onshore", "offshore", "all"] = "all",
) -> dict:
    """Fetch each bidding zone of a country, and write their sum as its national series.

    Norway and Sweden had one copy of this each, differing only in names (#84).
    A zone with no data is left out of both sums.

    Args:
        code: The country code (``"NO"``, ``"SE"``).
        zones: Its zone table, from :mod:`pyvwf.datasets.country_grid`.
        fetcher: ENTSO-E API client.
        train_years: Training years.
        test_year: Test year.
        output_dir: Output directory.
        psr_type: Wind type.

    Returns:
        Dictionary with zone data and aggregated country data.
    """
    logger.info(f"\n✓ {code} detected - fetching bidding zones separately")

    zone_train_data = {}
    zone_test_data = {}
    for zone_id in zones:
        logger.info(f"\n{'─' * 70}")
        logger.info(f"Fetching {zone_id}")
        logger.info(f"{'─' * 70}")
        zone_results = fetch_country_observations(
            fetcher, zone_id, train_years, test_year, output_dir, psr_type
        )
        if zone_results:
            zone_train_data[zone_id] = zone_results["train"]
            zone_test_data[zone_id] = zone_results["test"]

    logger.info(f"\n{'─' * 70}")
    logger.info("Aggregating zones to country-level")
    logger.info(f"{'─' * 70}")
    train_agg = _sum_zones(zone_train_data)
    test_agg = _sum_zones(zone_test_data)
    for label, agg in (("training", train_agg), ("test", test_agg)):
        if not agg.empty:
            logger.info(f"  ✓ Aggregated {label}: {len(agg)} data points")
            logger.info(f"    Mean CF: {agg['capacity_factor'].mean():.2%}")

    obs_dir = output_dir / "observations" / code.lower()
    obs_dir.mkdir(parents=True, exist_ok=True)
    years = (min(train_years), max(train_years))
    if not train_agg.empty:
        train_path = obs_dir / f"{country_obs_stem(code, 'train', years)}{AGGREGATED_SUFFIX}.csv"
        train_agg.to_csv(train_path)
        logger.info(f"\n✓ Saved aggregated training: {train_path}")
    if not test_agg.empty:
        test_path = obs_dir / f"{country_obs_stem(code, 'test', test_year)}{AGGREGATED_SUFFIX}.csv"
        test_agg.to_csv(test_path)
        logger.info(f"✓ Saved aggregated test: {test_path}")

    return {
        "zones": {"train": zone_train_data, "test": zone_test_data},
        "aggregated": {"train": train_agg, "test": test_agg},
    }


def fetch_country_observations(
    fetcher: ENTSOEWindDataFetcher,
    country: str,
    train_years: list[int],
    test_year: int,
    output_dir: Path,
    psr_type: Literal["onshore", "offshore", "all"] = "all",
) -> dict | None:
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
    logger.info(f"\nTraining period: {min(train_years)}-{max(train_years)}")
    train_start = pd.Timestamp(f"{min(train_years)}-01-01", tz="UTC")
    train_end = pd.Timestamp(f"{max(train_years)}-12-31 23:59", tz="UTC")

    train_data = fetcher.calculate_capacity_factor(
        country=country,
        start=train_start,
        end=train_end,
        psr_type=psr_type,
    )

    if train_data.empty:
        logger.info(f"  ✗ No training data for {country}")
        return None

    logger.info(f"  ✓ Training: {len(train_data)} data points")
    logger.info(f"    Mean CF: {train_data['capacity_factor'].mean():.2%}")
    logger.info(f"    Capacity: {train_data['capacity_mw'].mean():.0f} MW")

    # Test data
    logger.info(f"\nTest period: {test_year}")
    test_start = pd.Timestamp(f"{test_year}-01-01", tz="UTC")
    test_end = pd.Timestamp(f"{test_year}-12-31 23:59", tz="UTC")

    test_data = fetcher.calculate_capacity_factor(
        country=country,
        start=test_start,
        end=test_end,
        psr_type=psr_type,
    )

    if test_data.empty:
        logger.info(f"  ✗ No test data for {country}")
        return None

    logger.info(f"  ✓ Test: {len(test_data)} data points")
    logger.info(f"    Mean CF: {test_data['capacity_factor'].mean():.2%}")
    logger.info(f"    Capacity: {test_data['capacity_mw'].mean():.0f} MW")

    # Save training and test data separately
    obs_dir = output_dir / "observations" / country.lower()
    obs_dir.mkdir(parents=True, exist_ok=True)

    train_years_span = (min(train_years), max(train_years))
    train_path = obs_dir / f"{country_obs_stem(country, 'train', train_years_span)}.csv"
    train_data.to_csv(train_path)
    logger.info(f"\n✓ Saved training data: {train_path}")

    test_path = obs_dir / f"{country_obs_stem(country, 'test', test_year)}.csv"
    test_data.to_csv(test_path)
    logger.info(f"✓ Saved test data: {test_path}")

    return {
        "train": train_data,
        "test": test_data,
    }
