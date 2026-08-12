"""Fetch wind generation and capacity data from ENTSO-E API.

This script pulls data from the ENTSO-E Transparency Platform to calculate
capacity factor time series for NL, FR, BE, and NO.

Requirements:
    pip install entsoe-py pandas

Setup:
    1. Register at https://transparency.entsoe.eu/
    2. Create API key: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
    3. Set environment variable: export ENTSOE_API_KEY='your-key-here'
       Or pass directly to the script

Usage:
    # Fetch all four countries
    python vwf/datasets/fetch_entsoe_capacity_factors.py --countries NL FR BE NO --year-start 2015 --year-end 2020

    # Fetch specific Norwegian zones
    python vwf/datasets/fetch_entsoe_capacity_factors.py --countries NO_1 NO_3 --year-start 2020 --year-end 2021
"""

import argparse
import os
from pathlib import Path
from typing import Literal

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

# ENTSO-E country and zone codes
COUNTRY_CODES = {
    # Currently implemented (country-level)
    "NL": "NL",  # Netherlands
    "FR": "FR",  # France
    "BE": "BE",  # Belgium
    "NO": "NO",  # Norway (aggregate across zones)

    # Norwegian bidding zones
    "NO_1": "NO_1",
    "NO_2": "NO_2",
    "NO_3": "NO_3",
    "NO_4": "NO_4",
    "NO_5": "NO_5",

    # Phase 1: High priority (large wind capacity >5 GW)
    "ES": "ES",  # Spain (~30 GW)
    "IT": "IT",  # Italy (~12 GW)
    "PT": "PT",  # Portugal (~6 GW)
    "PL": "PL",  # Poland (~8 GW)
    "SE": "SE",  # Sweden (~12 GW) - aggregate
    "FI": "FI",  # Finland (~5 GW)
    "IE": "IE",  # Ireland (~5 GW)
    "IE_SEM": "IE_SEM",  # Ireland Single Electricity Market

    # Swedish bidding zones
    "SE_1": "SE_1",  # Luleå / Northern Sweden
    "SE_2": "SE_2",  # Sundsvall / North-Central Sweden
    "SE_3": "SE_3",  # Stockholm / Central Sweden
    "SE_4": "SE_4",  # Malmö / Southern Sweden

    # Phase 2: Medium priority (1-5 GW)
    "AT": "AT",  # Austria (~3 GW)
    "GR": "GR",  # Greece (~5 GW)
    "RO": "RO",  # Romania (~3 GW)

    # Phase 3: Lower priority (<1 GW)
    "CZ": "CZ",  # Czech Republic (~0.3 GW)
    "HU": "HU",  # Hungary (~0.3 GW)
    "HR": "HR",  # Croatia (~0.8 GW)
    "BG": "BG",  # Bulgaria (~0.7 GW)
    "LT": "LT",  # Lithuania (~0.6 GW)
    "LV": "LV",  # Latvia (~0.07 GW)
    "EE": "EE",  # Estonia (~0.3 GW)
    "SK": "SK",  # Slovakia (~0.003 GW)
    "SI": "SI",  # Slovenia (~0.003 GW)
}

# Norwegian bidding zones (detailed info)
NORWAY_ZONES = {
    "NO_1": "Oslo / Eastern Norway",
    "NO_2": "Kristiansand / Southern Norway",
    "NO_3": "Trondheim / Mid-Norway",
    "NO_4": "Tromsø / Northern Norway",
    "NO_5": "Bergen / Western Norway",
}

# Swedish bidding zones (detailed info)
SWEDEN_ZONES = {
    "SE_1": "Luleå / Northern Sweden",
    "SE_2": "Sundsvall / North-Central Sweden",
    "SE_3": "Stockholm / Central Sweden",
    "SE_4": "Malmö / Southern Sweden",
}

# Production types for wind
PSR_TYPES = {
    "onshore": "B19",  # Wind Onshore
    "offshore": "B18",  # Wind Offshore
}


def _wind_kinds(columns) -> set[str]:
    """Which wind kinds a set of entsoe-py columns covers, e.g. {"onshore"}.

    Columns arrive as plain strings or as ``(type, aggregation)`` tuples
    depending on the query, so the kind is recovered from the text rather than
    from the column structure.
    """
    kinds = set()
    for column in columns:
        text = " ".join(str(part) for part in column) if isinstance(column, tuple) else str(column)
        lowered = text.lower()
        if "offshore" in lowered:
            kinds.add("offshore")
        elif "onshore" in lowered:
            kinds.add("onshore")
        else:
            kinds.add("wind")
    return kinds


class ENTSOEWindDataFetcher:
    """Fetch wind generation and capacity data from ENTSO-E API."""

    def __init__(self, api_key: str | None = None):
        """Initialize the ENTSO-E client.

        Args:
            api_key: ENTSO-E API key. If None, reads from ENTSOE_API_KEY env var.
        """
        if api_key is None:
            api_key = os.getenv("ENTSOE_API_KEY")
            if api_key is None:
                raise ValueError(
                    "ENTSO-E API key not provided. Either:\n"
                    "  1. Pass api_key parameter, or\n"
                    "  2. Set ENTSOE_API_KEY environment variable\n"
                    "Get your key at: https://transparency.entsoe.eu/"
                )

        self.client = EntsoePandasClient(api_key=api_key)

    def fetch_generation(
        self,
        country: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Literal["onshore", "offshore", "all"] = "all",
    ) -> pd.DataFrame:
        """Fetch actual wind generation data.

        Args:
            country: Country code (NL, FR, BE, NO).
            start: Start timestamp (timezone-aware).
            end: End timestamp (timezone-aware).
            psr_type: Production type - 'onshore', 'offshore', or 'all'.

        Returns:
            DataFrame with DatetimeIndex and generation in MW.
        """
        country_code = COUNTRY_CODES.get(country.upper())
        if country_code is None:
            raise ValueError(f"Country {country} not supported")

        # Special handling for Norway aggregate (multiple bidding zones)
        if country.upper() == "NO":
            return self._fetch_norway_generation(start, end, psr_type)

        print(f"Fetching {psr_type} wind generation for {country} ({start.date()} to {end.date()})...")

        try:
            if psr_type == "all":
                # Fetch both onshore and offshore
                gen_data = self.client.query_generation(
                    country_code=country_code,
                    start=start,
                    end=end,
                    psr_type=None,  # All types
                )

                # Filter wind columns
                wind_cols = [col for col in gen_data.columns if "Wind" in str(col)]
                if not wind_cols:
                    raise NoMatchingDataError(f"No wind generation data found for {country}")

                # Sum onshore + offshore
                gen = gen_data[wind_cols].sum(axis=1)
                if not isinstance(gen, pd.DataFrame):
                    gen = gen.to_frame(name="generation_mw")
                gen.attrs["wind_columns"] = _wind_kinds(wind_cols)

            else:
                # Fetch specific type
                psr_code = PSR_TYPES[psr_type]
                gen = self.client.query_generation(
                    country_code=country_code,
                    start=start,
                    end=end,
                    psr_type=psr_code,
                )
                # Handle case where result is already a DataFrame
                if isinstance(gen, pd.DataFrame):
                    # If multiple columns, sum them
                    if len(gen.columns) > 1:
                        gen = gen.sum(axis=1).to_frame(name="generation_mw")
                    else:
                        gen.columns = ["generation_mw"]
                else:
                    gen = gen.to_frame(name="generation_mw")

            print(f"  ✓ Fetched {len(gen)} data points")
            return gen

        except NoMatchingDataError:
            print(f"  ✗ No data available for {country} (period {start.date()} to {end.date()})")
            return pd.DataFrame()

        except Exception as e:
            print(f"  ✗ Error fetching generation for {country}: {e}")
            return pd.DataFrame()

    def _fetch_norway_generation(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Literal["onshore", "offshore", "all"] = "all",
    ) -> pd.DataFrame:
        """Fetch generation data for all Norwegian bidding zones and aggregate.

        Args:
            start: Start timestamp (timezone-aware).
            end: End timestamp (timezone-aware).
            psr_type: Production type - 'onshore', 'offshore', or 'all'.

        Returns:
            DataFrame with aggregated Norwegian generation in MW.
        """
        print(f"Fetching {psr_type} wind generation for Norway (all zones)...")

        all_zones_data = []

        for zone_code, zone_name in NORWAY_ZONES.items():
            print(f"  Fetching {zone_code} ({zone_name})...")

            try:
                if psr_type == "all":
                    gen_data = self.client.query_generation(
                        country_code=zone_code,
                        start=start,
                        end=end,
                        psr_type=None,
                    )
                    wind_cols = [col for col in gen_data.columns if "Wind" in str(col)]
                    if wind_cols:
                        gen = gen_data[wind_cols].sum(axis=1)
                        all_zones_data.append(gen)
                        print(f"    ✓ {zone_code}: {len(gen)} data points")
                    else:
                        print(f"    ✗ {zone_code}: No wind data")
                else:
                    psr_code = PSR_TYPES[psr_type]
                    gen = self.client.query_generation(
                        country_code=zone_code,
                        start=start,
                        end=end,
                        psr_type=psr_code,
                    )
                    if isinstance(gen, pd.DataFrame):
                        gen = gen.sum(axis=1)
                    all_zones_data.append(gen)
                    print(f"    ✓ {zone_code}: {len(gen)} data points")

            except NoMatchingDataError:
                print(f"    ✗ {zone_code}: No data available")
            except Exception as e:
                print(f"    ✗ {zone_code}: Error - {e}")

        if not all_zones_data:
            print("  ✗ No data available from any Norwegian zone")
            return pd.DataFrame()

        # Aggregate all zones
        aggregated = pd.concat(all_zones_data, axis=1).sum(axis=1)
        aggregated_df = aggregated.to_frame(name="generation_mw")

        print(f"  ✓ Aggregated {len(all_zones_data)} zones: {len(aggregated_df)} total data points")
        return aggregated_df

    def fetch_installed_capacity(
        self,
        country: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Literal["onshore", "offshore", "all"] = "all",
    ) -> pd.DataFrame:
        """Fetch installed wind capacity data.

        Args:
            country: Country code (NL, FR, BE, NO).
            start: Start timestamp (timezone-aware).
            end: End timestamp (timezone-aware).
            psr_type: Production type - 'onshore', 'offshore', or 'all'.

        Returns:
            DataFrame with DatetimeIndex and capacity in MW.
        """
        country_code = COUNTRY_CODES.get(country.upper())
        if country_code is None:
            raise ValueError(f"Country {country} not supported")

        # Special handling for Norway aggregate (multiple bidding zones)
        if country.upper() == "NO":
            return self._fetch_norway_capacity(start, end, psr_type)

        print(f"Fetching {psr_type} wind capacity for {country}...")

        try:
            if psr_type == "all":
                # Fetch both types
                cap_data = self.client.query_installed_generation_capacity(
                    country_code=country_code,
                    start=start,
                    end=end,
                    psr_type=None,
                )

                # Filter wind columns
                wind_cols = [col for col in cap_data.columns if "Wind" in str(col)]
                if not wind_cols:
                    raise NoMatchingDataError(f"No wind capacity data found for {country}")

                # Sum onshore + offshore
                cap = cap_data[wind_cols].sum(axis=1)
                if not isinstance(cap, pd.DataFrame):
                    cap = cap.to_frame(name="capacity_mw")
                cap.attrs["wind_columns"] = _wind_kinds(wind_cols)

            else:
                psr_code = PSR_TYPES[psr_type]
                cap = self.client.query_installed_generation_capacity(
                    country_code=country_code,
                    start=start,
                    end=end,
                    psr_type=psr_code,
                )
                # Handle case where result is already a DataFrame
                if isinstance(cap, pd.DataFrame):
                    if len(cap.columns) > 1:
                        cap = cap.sum(axis=1).to_frame(name="capacity_mw")
                    else:
                        cap.columns = ["capacity_mw"]
                else:
                    cap = cap.to_frame(name="capacity_mw")

            print(f"  ✓ Fetched capacity: {cap['capacity_mw'].iloc[-1]:.0f} MW")
            return cap

        except NoMatchingDataError:
            print(f"  ✗ No capacity data available for {country}")
            return pd.DataFrame()

        except Exception as e:
            print(f"  ✗ Error fetching capacity for {country}: {e}")
            return pd.DataFrame()

    def _fetch_norway_capacity(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Literal["onshore", "offshore", "all"] = "all",
    ) -> pd.DataFrame:
        """Fetch capacity data for all Norwegian bidding zones and aggregate.

        Args:
            start: Start timestamp (timezone-aware).
            end: End timestamp (timezone-aware).
            psr_type: Production type - 'onshore', 'offshore', or 'all'.

        Returns:
            DataFrame with aggregated Norwegian capacity in MW.
        """
        print(f"Fetching {psr_type} wind capacity for Norway (all zones)...")

        all_zones_data = []

        for zone_code, zone_name in NORWAY_ZONES.items():
            print(f"  Fetching capacity for {zone_code}...")

            try:
                if psr_type == "all":
                    cap_data = self.client.query_installed_generation_capacity(
                        country_code=zone_code,
                        start=start,
                        end=end,
                        psr_type=None,
                    )
                    wind_cols = [col for col in cap_data.columns if "Wind" in str(col)]
                    if wind_cols:
                        cap = cap_data[wind_cols].sum(axis=1)
                        all_zones_data.append(cap)
                        print(f"    ✓ {zone_code}: {cap.iloc[-1]:.0f} MW")
                    else:
                        print(f"    ✗ {zone_code}: No wind capacity data")
                else:
                    psr_code = PSR_TYPES[psr_type]
                    cap = self.client.query_installed_generation_capacity(
                        country_code=zone_code,
                        start=start,
                        end=end,
                        psr_type=psr_code,
                    )
                    if isinstance(cap, pd.DataFrame):
                        cap = cap.sum(axis=1)
                    all_zones_data.append(cap)
                    print(f"    ✓ {zone_code}: {cap.iloc[-1]:.0f} MW")

            except NoMatchingDataError:
                print(f"    ✗ {zone_code}: No capacity data available")
            except Exception as e:
                print(f"    ✗ {zone_code}: Error - {e}")

        if not all_zones_data:
            print("  ✗ No capacity data available from any Norwegian zone")
            return pd.DataFrame()

        # Aggregate all zones
        aggregated = pd.concat(all_zones_data, axis=1).sum(axis=1)
        aggregated_df = aggregated.to_frame(name="capacity_mw")

        print(f"  ✓ Aggregated capacity: {aggregated_df['capacity_mw'].iloc[-1]:.0f} MW")
        return aggregated_df

    def calculate_capacity_factor(
        self,
        country: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Literal["onshore", "offshore", "all"] = "all",
    ) -> pd.DataFrame:
        """Calculate capacity factor time series.

        Args:
            country: Country code (NL, FR, BE, NO).
            start: Start timestamp (timezone-aware).
            end: End timestamp (timezone-aware).
            psr_type: Production type - 'onshore', 'offshore', or 'all'.

        Returns:
            DataFrame with columns: generation_mw, capacity_mw, capacity_factor.
        """
        # Fetch generation
        gen = self.fetch_generation(country, start, end, psr_type)
        if gen.empty:
            return pd.DataFrame()

        # Fetch capacity
        cap = self.fetch_installed_capacity(country, start, end, psr_type)
        if cap.empty:
            print("  ⚠ No capacity data - using mean generation as proxy")
            # Fallback: estimate capacity as max generation / 0.9 (assuming 90% availability)
            estimated_cap = gen["generation_mw"].max() / 0.9
            cap = pd.DataFrame(
                {"capacity_mw": estimated_cap},
                index=gen.index,
            )

        # The numerator and denominator have to cover the same fleet. NL's
        # series does not: its capacity factor climbs 4.3x from 2015 to 2021
        # while installed capacity climbs 2.9x, which is what a generation
        # series covering a growing share of the counted fleet looks like. No
        # constant rescaling repairs that, and nothing downstream can see it,
        # so the mismatch is refused here.
        gen_kinds = gen.attrs.get("wind_columns")
        cap_kinds = cap.attrs.get("wind_columns")
        if gen_kinds and cap_kinds and gen_kinds != cap_kinds:
            raise ValueError(
                f"{country}: generation covers {sorted(gen_kinds)} but installed "
                f"capacity covers {sorted(cap_kinds)}. A capacity factor built "
                "from these is wrong by the share of the fleet the two disagree "
                "about. Fetch a psr_type both sides report, or fetch each kind "
                "separately and add the energies."
            )

        # Align timeseries (capacity is usually less frequent than generation)
        # Use both forward-fill and backward-fill to handle capacity timestamps
        # that may come before or after generation timestamps
        df = gen.join(cap, how="left")
        df["capacity_mw"] = df["capacity_mw"].ffill().bfill()

        # Calculate capacity factor
        df["capacity_factor"] = df["generation_mw"] / df["capacity_mw"]

        # Clip to [0, 1] range (sometimes exceeds due to short-term overgeneration)
        # NOTE: saturation here destroys the true value, so
        # vwf.loaders.country_obs_checks flags any series that reaches it.
        df["capacity_factor"] = df["capacity_factor"].clip(0, 1.5)

        return df

    def fetch_multiple_countries(
        self,
        countries: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Literal["onshore", "offshore", "all"] = "all",
        resample: str | None = "h",
    ) -> dict[str, pd.DataFrame]:
        """Fetch capacity factors for multiple countries.

        Args:
            countries: List of country codes.
            start: Start timestamp (timezone-aware).
            end: End timestamp (timezone-aware).
            psr_type: Production type - 'onshore', 'offshore', or 'all'.
            resample: Resampling frequency (h=hourly, D=daily, M=monthly). None = no resampling.

        Returns:
            Dictionary mapping country code to DataFrame.
        """
        results = {}

        for country in countries:
            print(f"\n{'='*70}")
            print(f"Processing {country.upper()}")
            print(f"{'='*70}")

            df = self.calculate_capacity_factor(country, start, end, psr_type)

            if not df.empty:
                # Resample if requested
                if resample is not None:
                    print(f"  Resampling to {resample}...")
                    df = df.resample(resample).agg({
                        "generation_mw": "mean",
                        "capacity_mw": "mean",
                        "capacity_factor": "mean",
                    })

                results[country.upper()] = df
                print(f"  ✓ {country.upper()} complete: {len(df)} data points")
                print(f"    Mean CF: {df['capacity_factor'].mean():.2%}")
                print(f"    Capacity: {df['capacity_mw'].mean():.0f} MW")

        return results


def save_capacity_factors(
    data: dict[str, pd.DataFrame],
    output_dir: Path,
    format: Literal["csv", "parquet", "both"] = "both",
):
    """Save capacity factor data to files.

    Args:
        data: Dictionary mapping country code to DataFrame.
        output_dir: Output directory.
        format: Output format - 'csv', 'parquet', or 'both'.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for country, df in data.items():
        # Add country column
        df["country"] = country

        if format in ["csv", "both"]:
            csv_path = output_dir / f"{country.lower()}_capacity_factors.csv"
            df.to_csv(csv_path)
            print(f"  ✓ Saved: {csv_path}")

        if format in ["parquet", "both"]:
            parquet_path = output_dir / f"{country.lower()}_capacity_factors.parquet"
            df.to_parquet(parquet_path)
            print(f"  ✓ Saved: {parquet_path}")

    # Save combined file
    if len(data) > 1:
        combined = pd.concat(data.values(), keys=data.keys())
        combined.index.names = ["country", "timestamp"]

        if format in ["csv", "both"]:
            combined_csv = output_dir / "combined_capacity_factors.csv"
            combined.to_csv(combined_csv)
            print(f"  ✓ Saved combined: {combined_csv}")

        if format in ["parquet", "both"]:
            combined_parquet = output_dir / "combined_capacity_factors.parquet"
            combined.to_parquet(combined_parquet)
            print(f"  ✓ Saved combined: {combined_parquet}")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Fetch wind capacity factors from ENTSO-E API"
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=["NL", "FR", "BE"],
        help="Country codes (default: NL FR BE). Norway zones: NO_1..NO_5; use NO to aggregate.",
    )
    parser.add_argument(
        "--year-start",
        type=int,
        required=True,
        help="Start year (inclusive)",
    )
    parser.add_argument(
        "--year-end",
        type=int,
        required=True,
        help="End year (inclusive)",
    )
    parser.add_argument(
        "--psr-type",
        choices=["onshore", "offshore", "all"],
        default="all",
        help="Wind type to fetch (default: all)",
    )
    parser.add_argument(
        "--resample",
        default="h",
        help="Resampling frequency: h=hourly, D=daily, M=monthly (default: h)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/entsoe_capacity_factors"),
        help="Output directory (default: output/entsoe_capacity_factors)",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="ENTSO-E API key (or set ENTSOE_API_KEY env var)",
    )

    args = parser.parse_args()

    # Check for Norway
    if "NO" in [c.upper() for c in args.countries]:
        print("\nℹ️  Norway has 5 bidding zones in ENTSO-E (NO_1 to NO_5).")
        print("   Data will be aggregated from all zones automatically.")
        print("   Zones: Oslo, Kristiansand, Trondheim, Tromsø, Bergen\n")

    # Initialize fetcher
    print("Initializing ENTSO-E API client...")
    fetcher = ENTSOEWindDataFetcher(api_key=args.api_key)
    print("  ✓ Connected to ENTSO-E API\n")

    # Create timestamps (use UTC or local timezone)
    start = pd.Timestamp(f"{args.year_start}-01-01", tz="UTC")
    end = pd.Timestamp(f"{args.year_end}-12-31 23:59:59", tz="UTC")

    print(f"Fetching data for: {', '.join(args.countries)}")
    print(f"Period: {start.date()} to {end.date()}")
    print(f"Type: {args.psr_type}")
    print(f"Resampling: {args.resample}\n")

    # Fetch data
    data = fetcher.fetch_multiple_countries(
        countries=args.countries,
        start=start,
        end=end,
        psr_type=args.psr_type,
        resample=args.resample,
    )

    # Save results
    if data:
        print(f"\n{'='*70}")
        print("Saving results...")
        print(f"{'='*70}")
        save_capacity_factors(data, args.output_dir, format=args.format)

        print(f"\n✓ Complete! Data saved to: {args.output_dir}")
    else:
        print("\n✗ No data fetched. Check country codes and date ranges.")


if __name__ == "__main__":
    main()
