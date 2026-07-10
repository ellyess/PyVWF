"""Turbine-level observation source for Denmark, Germany, and the UK.

This adapter wraps the existing CSV loaders in :mod:`vwf.loaders.turbine_loaders`
and reproduces the metadata and capacity-factor preparation that
``vwf.data.prep_country`` performed inline before the source refactor.
"""
from __future__ import annotations

from calendar import monthrange
from typing import ClassVar

import pandas as pd

from vwf.loaders.turbine_loaders import load_turbine_metadata, load_turbine_observations
from vwf.sources.base import ObservationSource
from vwf.sources.registry import register

#: Training window used when no explicit year range is requested.
DEFAULT_TRAIN_YEARS: dict[str, tuple[int, int]] = {
    "DK": (2015, 2019),
    "DE": (2015, 2018),
    "UK": (2015, 2018),
}

_FALLBACK_TRAIN_YEARS: tuple[int, int] = (2015, 2018)


@register
class EuropeanTurbineSource(ObservationSource):
    """Per-turbine metadata and monthly generation for DK, DE, and UK."""

    name: ClassVar[str] = "european-turbine"
    obs_level: ClassVar[str] = "turbine"
    countries: ClassVar[tuple[str, ...]] = ("DK", "DE", "UK")

    def __init__(self, country: str) -> None:
        country = country.upper()
        if country not in self.countries:
            raise ValueError(
                f"{type(self).__name__} supports {self.countries}, got {country!r}"
            )
        self.country: str = country
        self._metadata: pd.DataFrame | None = None

    @property
    def default_train_years(self) -> tuple[int, int]:
        """Inclusive training window for this country."""
        return DEFAULT_TRAIN_YEARS.get(self.country, _FALLBACK_TRAIN_YEARS)

    def load_metadata(self) -> pd.DataFrame:
        """Load turbine metadata and assign a power-curve model to each turbine."""
        if self._metadata is None:
            # Imported lazily: vwf.data imports this package at module scope, so a
            # top-level import here would close the cycle.
            from vwf.data import add_models

            self._metadata = add_models(load_turbine_metadata(self.country))
        return self._metadata.copy()

    def load_observations(
        self,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> pd.DataFrame:
        """Load monthly generation and convert it to capacity factor.

        Generation arrives in kWh per turbine-month. Dividing by the hours in the
        month and the turbine's rated capacity in kW yields a capacity factor.

        Returns:
            Wide DataFrame with ``ID``, ``year``, and ``obs_1`` to ``obs_12``.
        """
        if year_start is None or year_end is None:
            year_start, year_end = self.default_train_years

        turb_info = self.load_metadata()
        obs_gen = load_turbine_observations(self.country, int(year_start), int(year_end)).copy()

        if not {"ID", "year"}.issubset(obs_gen.columns):
            raise ValueError("Turbine observations must contain columns ['ID','year', ...months...]")

        # Standardise month columns to obs_1..obs_12 (if not already)
        month_cols = [c for c in obs_gen.columns if c not in ["ID", "year"]]
        if not any(str(c).startswith("obs_") for c in month_cols):
            obs_gen.columns = [f"obs_{c}" if c not in ["ID", "year"] else c for c in obs_gen.columns]

        obs_gen["ID"] = obs_gen["ID"].astype(str)
        obs_gen["year"] = pd.to_numeric(obs_gen["year"], errors="coerce").astype("Int64")

        # Merge capacity for CF conversion
        obs_gen = obs_gen.merge(turb_info[["ID", "capacity"]], how="left", on="ID")
        obs_gen = obs_gen.dropna(subset=["capacity", "year"]).reset_index(drop=True)

        for m in range(1, 13):
            col = f"obs_{m}"
            if col not in obs_gen.columns:
                continue
            days = (
                obs_gen["year"]
                .astype(int)
                .map(lambda y, month=m: monthrange(int(y), int(month))[1])
                .astype(float)
            )
            obs_gen[col] = pd.to_numeric(obs_gen[col], errors="coerce") / (
                days * 24.0 * obs_gen["capacity"].astype(float)
            )

        return obs_gen.drop(columns=["capacity"])
