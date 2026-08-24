"""Turbine-level data loaders for PyVWF.

This module provides functions to load turbine metadata and observations
from supported countries (DK, DE, UK).
"""
import pandas as pd

from vwf.config import PyVWFPaths
from vwf.utils import ensure_numeric


def _standardise_turb_info_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize turbine metadata for interpolation.

    Args:
        df: Turbine metadata with at least ``ID``, ``capacity``, ``height``, ``lon``, and ``lat``.

    Returns:
        Cleaned DataFrame with required columns and types.

    Raises:
        ValueError: If required columns are missing or no valid turbines remain.
    """
    df = df.copy()

    if "ID" not in df.columns:
        raise ValueError("turbine metadata must contain 'ID'")
    df["ID"] = df["ID"].astype(str)

    if "type" not in df.columns:
        df["type"] = "onshore"

    df = ensure_numeric(df, ["capacity", "diameter", "height", "lon", "lat"])

    # enforce physical hub heights
    df = df[df["height"].notna()]
    df = df[df["height"] > 1.0]   # or >0, but >1 m avoids pathological cases

    if df.empty:
        raise ValueError("No turbines with valid hub height (>1 m) after standardisation")

    return df.reset_index(drop=True)


def load_turbine_metadata(country: str) -> pd.DataFrame:
    """Load raw turbine metadata for a country.

    Supported countries: DK (Denmark), DE (Germany), UK (United Kingdom)

    Args:
        country: Country code (case-insensitive, e.g., 'DK', 'DE', 'UK').

    Returns:
        DataFrame with columns: ``ID``, ``capacity``, ``diameter``, ``height``,
        ``manufacturer``, ``lon``, ``lat``, and ``type`` (onshore/offshore).

    Raises:
        ValueError: If country is not supported.

    Examples:
        >>> dk_turbines = load_turbine_metadata('DK')
        >>> print(dk_turbines.columns)
        Index(['ID', 'manufacturer', 'capacity', 'diameter', 'height', 'lon', 'lat', 'type'], dtype='object')
    """
    country = country.upper()

    if country == "DK":
        # Load processed metadata from observations/turbine
        dk_md = pd.read_csv(PyVWFPaths.TURBINE_DATA / "DK/dk_md.csv")

        # Select and rename columns
        columns_map = {
            "ID": "ID",
            "manufacturer": "manufacturer",
            "capacity": "capacity",
            "diameter": "diameter",
            "height": "height",
            "lon": "lon",
            "lat": "lat",
            "location_type": "type"
        }

        # Keep only available columns
        available_cols = [col for col in columns_map.keys() if col in dk_md.columns]
        dk_md = dk_md[available_cols].copy()
        dk_md = dk_md.rename(columns=columns_map)

        # Standardize location type (Land -> onshore, Hav -> offshore)
        if "type" in dk_md.columns:
            dk_md["type"] = dk_md["type"].str.lower().replace({"land": "onshore", "hav": "offshore"})

        # Ensure numeric columns
        dk_md = ensure_numeric(dk_md, ["capacity", "diameter", "height", "lon", "lat"])

        # Drop rows with missing essential data
        dk_md = dk_md.dropna(subset=["capacity", "diameter", "lon", "lat"]).reset_index(drop=True)

        # Clean manufacturer names
        dk_md["manufacturer"] = dk_md["manufacturer"].astype(str).str.split(" ").str[0]

        return _standardise_turb_info_minimal(dk_md)

    if country == "DE":
        # Load Germany data from observations/turbine
        de_geo = pd.read_csv(PyVWFPaths.TURBINE_DATA / "DE/geolocate.germany.csv")
        de_md = pd.read_csv(PyVWFPaths.TURBINE_DATA / "DE/DE_md.csv")

        de_md = de_md[["V1", "Manufacturer", "kW", "Rotor..m.", "Tower..m."]]
        de_md.columns = ["ID", "manufacturer", "capacity", "diameter", "height"]
        de_md["postcode"] = de_md["ID"].astype(str).str[:5].astype(int)

        de_md = pd.merge(de_md, de_geo[["postcode", "lon", "lat"]], on="postcode", how="left").drop(columns=["postcode"])
        de_md = de_md.dropna(subset=["capacity", "diameter", "lon", "lat"]).reset_index(drop=True)
        de_md["type"] = "onshore"
        return _standardise_turb_info_minimal(de_md)

    if country == "UK":
        # Load UK data from observations/turbine
        uk_md = pd.read_csv(PyVWFPaths.TURBINE_DATA / "UK/uk_md.csv")
        return _standardise_turb_info_minimal(uk_md)

    raise ValueError(f"Unsupported country={country}. Supported: DK, DE, UK")


def load_turbine_observations(country: str, year_start: int, year_end: int) -> pd.DataFrame:
    """Load turbine-level monthly generation in wide format.

    Supported countries: DK (Denmark), DE (Germany), UK (United Kingdom)

    Args:
        country: Country code (case-insensitive).
        year_start: First year to include (inclusive).
        year_end: Last year to include (inclusive).

    Returns:
        DataFrame with columns: ``ID``, ``year``, and month columns (1-12)
        containing generation values.

    Raises:
        ValueError: If country is not supported or no observations loader is available.

    Examples:
        >>> dk_obs = load_turbine_observations('DK', 2015, 2019)
        >>> print(dk_obs.columns[:5])
        Index(['ID', 'year', '1', '2', '3'], dtype='object')
    """
    country = country.upper()

    if country == "DK":
        # Load processed observations from observations/turbine (long format)
        dk_data = pd.read_csv(PyVWFPaths.TURBINE_DATA / "DK/dk_obs_2002_2020.csv")

        # Filter by year range
        dk_data = dk_data.loc[(dk_data["year"] >= year_start) & (dk_data["year"] <= year_end)].copy()

        # Ensure proper data types
        dk_data["ID"] = dk_data["ID"].astype(str)
        dk_data = dk_data.dropna(subset=["ID", "year", "month"])

        # Pivot from long to wide format: (ID, year) x month -> generation_kwh
        obs = (
            dk_data.pivot(index=["ID", "year"], columns="month", values="generation_kwh")
            .reset_index()
            .fillna(0)
            .infer_objects()
        )

        # Rename columns to match expected format (1, 2, 3, ..., 12)
        month_cols = {i: str(i) for i in range(1, 13)}
        obs = obs.rename(columns=month_cols)

        return obs

    if country == "DE":
        # Load Germany observations from observations/turbine
        de_data = pd.read_csv(PyVWFPaths.TURBINE_DATA / "DE/DE_data.csv")
        de_data = (
            de_data.loc[(de_data["Year"] >= year_start) & (de_data["Year"] <= year_end)]
            .drop(columns=["Downtime"])
            .reset_index(drop=True)
        )
        de_data.columns = ["ID", "year", "month", "output"]
        de_data = de_data.dropna(subset=["ID", "year", "month"])
        obs = (
            de_data.pivot(index=["ID", "year"], columns="month", values="output")
            .reset_index()
            .fillna(0)
            .infer_objects()
        )
        return obs

    if country == "UK":
        # Load UK observations from observations/turbine
        obs = pd.read_csv(PyVWFPaths.TURBINE_DATA / "UK/ukobs.csv")
        # Filter by year (UK data may not need year filtering, but include for consistency)
        if "year" in obs.columns:
            obs = obs.loc[(obs["year"] >= year_start) & (obs["year"] <= year_end)].copy()
        return obs

    raise ValueError(f"No turbine-level observation loader implemented for {country}.")
