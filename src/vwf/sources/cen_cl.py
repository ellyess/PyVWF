"""Plant-level observation source for Chile (Coordinador / CEN SIP data).

The unit of observation is the **plant** (``obs_unit = "plant"``): CEN's SIP
service reports generation per *central* (``id_central``), and the processing
step reduces the hourly ``generacion-real`` series to monthly capacity factors:
the same design as the AU (farm), US (plant), BR (complex), and NZ (farm)
adapters, where the mechanical ``obs_level`` stays "turbine".

Decisions baked into the finalisation, documented in the region config:

- **UTC bins from fixed Chilean standard time.** CEN publishes hourly data in
  a fixed UTC-4 offset with NO daylight saving (verified on the 2024
  spring-forward day); the processing step shifts to UTC before binning,
  matching the ERA5/simulation convention: ``time_convention =
  "utc-monthly-bins"``.
- **Wind filtered on the accented label.** ``tipo_tecnologia == "Eólica"``;
  the unaccented spelling silently returns nothing.
- **Metadata from the generation stream, coordinates from GWPT.** The CEN rows
  carry plant name, owner, and ``potencia_maxima``; coordinates are absent
  from the entire CEN API and are joined from the Global Wind Power Tracker in
  the processing step. Any plant reaching training without a coordinate is a
  hard error, not a silent drop.
"""
from __future__ import annotations

from typing import ClassVar

import pandas as pd

from vwf.config import PyVWFPaths
from vwf.sources.base import ObservationSource, ObsLevel
from vwf.sources.registry import register

#: Training window used when no explicit year range is requested. Matches the
#: ERA5 fetch (2021-2024) with 2024 held out for evaluation.
DEFAULT_TRAIN_YEARS: tuple[int, int] = (2021, 2023)


@register
class CENChileSource(ObservationSource):
    """Per-plant monthly capacity factors for Chile.

    Reads pre-processed files from ``input/observations/turbine/CL/`` (built by
    ``scripts/process/cen_cl.py`` from the CEN downloads + the GWPT join):

    ``cl_md.csv``
        Plant metadata: ``ID`` (``id_central``), ``lon``, ``lat``, ``height``
        (m), ``capacity`` (kW), ``model``, ``type``, plus provenance/context
        (``height_source``, ``model_source``, ``site_name``, ``propietario``).
    ``cl_obs.csv``
        Monthly observations, wide: ``ID``, ``year``, ``obs_1``..``obs_12``.
        Monthly mean CF in UTC bins against ``potencia_maxima``,
        coverage-screened, as written by the processing step.
    """

    name: ClassVar[str] = "cen-cl"
    obs_level: ClassVar[ObsLevel] = "turbine"  # mechanical contract; obs_unit is "plant"
    countries: ClassVar[tuple[str, ...]] = ("CL", "CHL")

    def __init__(self, country: str = "CL") -> None:
        country = country.upper()
        if country not in self.countries:
            raise ValueError(
                f"{type(self).__name__} supports {self.countries}, got {country!r}"
            )
        self.country: str = country

    @property
    def default_train_years(self) -> tuple[int, int]:
        """Inclusive training window."""
        return DEFAULT_TRAIN_YEARS

    def _data_dir(self):
        return PyVWFPaths.TURBINE_DATA / "CL"

    def load_metadata(self) -> pd.DataFrame:
        path = self._data_dir() / "cl_md.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"CEN-CL plant metadata not found at {path}. This adapter reads "
                "pre-processed files; see the class docstring for the schema "
                "(data acquisition is a user-executed step: docs/runbooks/CL.md)."
            )
        meta = pd.read_csv(path)
        required = {"ID", "lon", "lat", "height", "capacity", "model"}
        missing = required - set(meta.columns)
        if missing:
            raise ValueError(f"{path} is missing required columns {sorted(missing)}")
        meta["ID"] = meta["ID"].astype(str)
        if "type" not in meta.columns:
            meta["type"] = "onshore"
        return meta

    def load_observations(
        self,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> pd.DataFrame:
        """Monthly capacity factors from the pre-processed wide frame."""
        if year_start is None or year_end is None:
            year_start, year_end = self.default_train_years

        path = self._data_dir() / "cl_obs.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"CEN-CL observations not found at {path}. This adapter reads "
                "pre-processed files; see the class docstring for the schema "
                "(data acquisition is a user-executed step: docs/runbooks/CL.md)."
            )
        wide = pd.read_csv(path)
        wide["ID"] = wide["ID"].astype(str)
        return wide[
            (wide["year"] >= int(year_start)) & (wide["year"] <= int(year_end))
        ].reset_index(drop=True)


__all__ = ["CENChileSource"]
