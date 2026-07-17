"""The ``ObservationSource`` interface.

An observation source supplies the two things the PyVWF pipeline needs before it
can derive bias corrections for a region:

1. Site metadata: the locations to simulate, with the physical properties needed
   to interpolate wind and apply a power curve.
2. Observed generation: the measured capacity factors those sites achieved.

Everything downstream (interpolation, power curves, clustering, the correction
maths) is source agnostic. Adding a new country therefore means writing one
adapter, registering it, and nothing else.

See ``docs/ADDING_AN_OBSERVATION_SOURCE.md`` for a worked example.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import pandas as pd

#: The two granularities at which observed generation is available.
ObsLevel = Literal["turbine", "country"]


class ObservationSource(ABC):
    """Base class for every observed-generation adapter.

    Subclasses declare three class attributes:

    ``name``
        Unique registry key, for example ``"european-turbine"``.
    ``obs_level``
        Either ``"turbine"`` or ``"country"``. This selects which branch of the
        pipeline consumes the source, and therefore which observation shape the
        source must return.
    ``countries``
        Country codes this source can be resolved for automatically. Leave empty
        for sources that must be constructed explicitly by the caller, such as
        :class:`~vwf.sources.in_memory.InMemoryCountrySource`.

    Auto-resolvable sources (those with a non-empty ``countries``) must accept a
    single country code as their only constructor argument, because that is how
    :func:`vwf.sources.registry.resolve` instantiates them.
    """

    name: ClassVar[str]
    obs_level: ClassVar[ObsLevel]
    countries: ClassVar[tuple[str, ...]] = ()

    def __init__(self, country: str) -> None:
        """Construct an auto-resolvable source for one country.

        This is the constructor contract :func:`vwf.sources.registry.resolve`
        relies on, declared here so it is type-checked rather than assumed.
        Sources that must be built explicitly by the caller (those with an
        empty ``countries``, such as
        :class:`~vwf.sources.in_memory.InMemoryCountrySource`) override it with
        whatever signature they need.
        """
        self.country = country.upper()

    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        """Return the sites to simulate.

        Required columns, whatever the ``obs_level``:

        ``ID``
            Site identifier, string. Joins metadata to observations and to
            simulated output.
        ``lon``, ``lat``
            Location in degrees.
        ``height``
            Hub height in metres, strictly greater than 1.
        ``capacity``
            Rated capacity in kW.
        ``model``
            Power curve key, matching a column of ``input/power_curves.csv``.
        ``type``
            ``"onshore"`` or ``"offshore"``. Used when ``cluster_mode`` filters
            the fleet.

        Country-level sources must additionally supply ``cluster``, the cluster
        assignment for each grid point, because country-level corrections are
        derived per cluster and no clustering step runs on that path.
        """

    @abstractmethod
    def load_observations(
        self,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> pd.DataFrame:
        """Return observed generation, already converted to capacity factor.

        ``year_start`` and ``year_end`` are inclusive. When both are ``None`` the
        source returns its own default training window. Sources whose data is
        already scoped by the caller may ignore both arguments.

        The returned shape depends on ``obs_level``:

        ``obs_level == "turbine"``
            Wide, one row per ``(ID, year)``, with columns ``ID``, ``year`` and
            ``obs_1`` through ``obs_12`` holding the monthly mean capacity
            factor. Missing months are ``NaN``.

        ``obs_level == "country"``
            A ``DatetimeIndex``ed frame with a ``capacity_factor`` column, at
            whatever native resolution the source provides. The pipeline
            resamples to monthly means for training and keeps the native
            resolution for validation.

        Capacity factors are dimensionless and normally lie in ``[0, 1]``.
        """
