"""File-backed country-level observation source over the observations/country layout.

This is the "entsoe-country" adapter the validation harness resolves for the
nine ENTSO-E regions. It reads the on-disk convention produced by
``vwf.datasets.generate_country_level_training_data``:

    observations/country/
      grid_points/<c>/<c>_grid_points.csv        # static sites, with cluster
      observations/<c>/<c>_train_<start>_<end>.csv
      observations/<c>/<c>_test_<year>.csv

Like :class:`~vwf.sources.in_memory.InMemoryCountrySource` it is
caller-constructed (``countries`` is empty, so it is never auto-resolved): the
driver builds one per split, because train and test read different files.
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd

from vwf.config import PyVWFPaths
from vwf.sources.base import ObservationSource, ObsLevel
from vwf.sources.registry import register

Split = Literal["train", "test"]


@register
class EntsoeFileSource(ObservationSource):
    """Country-level grid points and observed CF series, read from files."""

    name: ClassVar[str] = "entsoe-country"
    obs_level: ClassVar[ObsLevel] = "country"
    countries: ClassVar[tuple[str, ...]] = ()  # caller-constructed per split

    def __init__(
        self,
        country: str,
        split: Split,
        train_years: tuple[int, int],
        test_year: int,
        cl_data_dir: str | Path | None = None,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.country = country.upper()
        self._c = country.lower()
        self.split: Split = split
        self.train_years = (int(train_years[0]), int(train_years[1]))
        self.test_year = int(test_year)
        self._base = Path(cl_data_dir) if cl_data_dir else PyVWFPaths.COUNTRY_LEVEL_DATA

    def _grid_points_path(self) -> Path:
        return self._base / "grid_points" / self._c / f"{self._c}_grid_points.csv"

    def _obs_path(self) -> Path:
        obs_dir = self._base / "observations" / self._c
        if self.split == "train":
            return obs_dir / f"{self._c}_train_{self.train_years[0]}_{self.train_years[1]}.csv"
        return obs_dir / f"{self._c}_test_{self.test_year}.csv"

    def load_metadata(self) -> pd.DataFrame:
        path = self._grid_points_path()
        if not path.is_file():
            raise FileNotFoundError(f"country grid points not found at {path}")
        gp = pd.read_csv(path)
        if "cluster" not in gp.columns:
            raise ValueError(
                f"{path} has no 'cluster' column; country-level corrections are "
                "derived per cluster and no clustering step runs on that path"
            )
        gp["ID"] = gp["ID"].astype(str)
        return gp

    def load_observations(
        self,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> pd.DataFrame:
        """Return the split's observed capacity-factor series (DatetimeIndexed).

        The year arguments are ignored: the split already scopes the file.
        """
        path = self._obs_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"country observations not found at {path} (split={self.split!r})"
            )
        obs = pd.read_csv(path, index_col=0, parse_dates=True)
        if "capacity_factor" not in obs.columns:
            raise ValueError(f"{path} has no 'capacity_factor' column")
        return obs
