"""Per-bidding-zone country-level observations.

This is RQ4 option 1. The national path fits N cluster offsets against a single
national observation per period, which is under-determined by N-1: the offsets
are wherever the optimiser stopped, not estimates. When each cluster carries its
own observation the problem becomes N observations against N offsets, exactly
determined, and the country path collapses onto the turbine path's estimator.

The zonal regions already have the data. ``generate_country_level_training_data``
writes one observation file per bidding zone alongside the national aggregate,
and numbers the grid points' clusters from the zone id (``SE_1`` to cluster 0,
``SE_2`` to cluster 1, and so on), so zone and cluster are already the same
partition. Nothing pointed the loader at those files.

Layout read here::

    observations/country/
      grid_points/<c>/<c>_grid_points[_<year>].csv
      observations/<c>_1/<c>_1_train_<start>_<end>.csv
      observations/<c>_2/<c>_2_train_<start>_<end>.csv
      ...

Every zone found is loaded; a zone whose file is missing for this split is
reported rather than silently dropped, because a missing zone shifts the
national aggregate the metrics are scored against.
"""
from __future__ import annotations

from typing import ClassVar

import pandas as pd

from vwf.loaders.country_obs_checks import check_country_cf
from vwf.sources.base import ObsLevel
from vwf.sources.entsoe_files import EntsoeFileSource
from vwf.sources.registry import register

#: Highest zone number to look for. NO and SE have five and four; no ENTSO-E
#: bidding-zone country has more than this.
MAX_ZONES = 8


def _base_country(code: str) -> str:
    """Country from a region code, e.g. ``'SE-BZ'`` to ``'SE'``.

    The zonal run is a different region from the national one so their codes
    stay distinct (the same convention ``ES-WS`` uses), but both read the same
    country's files.
    """
    return code.split("-", 1)[0].upper()


@register
class EntsoeZonalFileSource(EntsoeFileSource):
    """Country-level observations resolved per bidding zone rather than nationally.

    Returns a long frame with ``cluster`` alongside ``capacity_factor``, which
    is what tells the training path each cluster has its own constraint. Grid
    point loading is inherited unchanged.
    """

    name: ClassVar[str] = "entsoe-zonal"
    obs_level: ClassVar[ObsLevel] = "country"
    countries: ClassVar[tuple[str, ...]] = ()  # caller-constructed per split

    def __init__(self, country: str, *args, **kwargs) -> None:
        super().__init__(_base_country(country), *args, **kwargs)

    def _zone_dir(self, zone: int):
        return self._base / "observations" / f"{self._c}_{zone}"

    def _zone_candidates(self, zone: int) -> list:
        stem = self._obs_stem().replace(self._c, f"{self._c}_{zone}", 1)
        directory = self._zone_dir(zone)
        return [directory / f"{stem}.csv", directory / f"{stem}_aggregated.csv"]

    def available_zones(self) -> list[int]:
        """Zone numbers with a directory present, whether or not this split has a file."""
        return [z for z in range(1, MAX_ZONES + 1) if self._zone_dir(z).is_dir()]

    def load_observations(
        self,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> pd.DataFrame:
        """Return every zone's CF series, tagged with its cluster.

        Returns:
            DataFrame indexed by time with ``capacity_factor`` and ``cluster``,
            where ``cluster == zone - 1`` to match how the grid points were
            numbered.

        Raises:
            FileNotFoundError: If no zone has a file for this split.
        """
        zones = self.available_zones()
        if not zones:
            raise FileNotFoundError(
                f"no per-zone observation directories under "
                f"{self._base / 'observations'} for {self.country}; expected "
                f"{self._c}_1, {self._c}_2, ..."
            )

        frames: list[pd.DataFrame] = []
        missing: list[int] = []
        for zone in zones:
            path = next((p for p in self._zone_candidates(zone) if p.is_file()), None)
            if path is None:
                missing.append(zone)
                continue
            obs = pd.read_csv(path, index_col=0, parse_dates=True)
            if "capacity_factor" not in obs.columns:
                raise ValueError(f"{path} has no 'capacity_factor' column")
            check_country_cf(obs, f"{self.country}_{zone} {self.split} ({path.name})")
            obs = obs.copy()
            # generate_country_level_training_data numbers clusters from the
            # zone id as zone - 1; the two partitions have to line up or the
            # merge in train_set drops every row.
            obs["cluster"] = zone - 1
            frames.append(obs)

        if not frames:
            raise FileNotFoundError(
                f"no per-zone observations for {self.country} split={self.split!r}; "
                f"looked for zones {zones} under {self._base / 'observations'}"
            )

        if missing:
            raise FileNotFoundError(
                f"{self.country} split={self.split!r} is missing zones {missing} "
                f"(found {[z for z in zones if z not in missing]}). A partial zone "
                "set shifts the national aggregate the metrics are scored "
                "against, so it is refused rather than silently used."
            )

        return pd.concat(frames).sort_index()
