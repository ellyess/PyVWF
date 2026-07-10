"""Country-level observation source backed by caller-supplied DataFrames.

This is the adapter behind :meth:`vwf.vwf.PyVWF.load_country_data`. Country-level
observations (ENTSO-E derived, for example) are fetched and cached outside the
library, then handed to PyVWF as a pair of DataFrames. Wrapping them in a source
lets the country-level path travel the same seam as the turbine-level path.
"""
from __future__ import annotations

from typing import ClassVar

import pandas as pd

from vwf.sources.base import ObservationSource
from vwf.sources.registry import register


@register
class InMemoryCountrySource(ObservationSource):
    """Grid points and an observed capacity-factor series held in memory.

    ``countries`` is deliberately empty: this source is never resolved
    automatically from a country code, it is always constructed by the caller
    with the data it should serve.
    """

    name: ClassVar[str] = "in-memory-country"
    obs_level: ClassVar[str] = "country"
    countries: ClassVar[tuple[str, ...]] = ()

    def __init__(self, grid_points: pd.DataFrame, observations: pd.DataFrame) -> None:
        """Store defensive copies of the supplied frames.

        Args:
            grid_points: Sites to simulate. Must carry ``cluster`` alongside the
                usual metadata columns, since no clustering step runs on the
                country-level path.
            observations: DatetimeIndexed frame with a ``capacity_factor`` column.
        """
        self._grid_points: pd.DataFrame = grid_points.copy()
        self._observations: pd.DataFrame = observations.copy()

    def load_metadata(self) -> pd.DataFrame:
        """Return the grid points."""
        return self._grid_points.copy()

    def load_observations(
        self,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> pd.DataFrame:
        """Return the observed series.

        The year arguments are ignored. The caller already scoped this frame when
        it split observations into training and test periods.
        """
        return self._observations.copy()
