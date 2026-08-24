"""Plant-level observation source for Argentina (CAMMESA renewables base).

The unit of observation is the **plant** (``obs_unit = "plant"``): CAMMESA
reports generated energy per *central*, one row per plant per month, already
at the monthly resolution PyVWF trains on; the processing step only converts
energy (GWh) to capacity factor against a GWPT-joined capacity. There is no
sub-hourly reshaping, timezone, or DST handling, so this is the simplest of
the adapters; the complexity all lives in the coordinate/capacity join.

Decisions baked into the processing (documented in the region config):

- **Monthly native.** CAMMESA's "Tabla Resumen x Central" is monthly GWh per
  central from 2011; no time conversion is needed or done.
- **Capacity and coordinates from GWPT.** CAMMESA carries neither, so both are
  joined from the Global Wind Power Tracker. A plant lacking either is a hard
  error in ``build_ar_metadata``, and a plant whose implied CF is implausibly
  high (bad capacity) is flagged for re-curation.
- **Commissioning prefix stripped.** Pre-operational leading months (CF~0
  against the static nameplate) are removed.
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
class CAMMESAArgentinaSource(ObservationSource):
    """Per-plant monthly capacity factors for Argentina.

    Reads pre-processed files from ``input/observations/turbine/AR/`` (built by
    ``scripts/process/cammesa_ar.py`` from the CAMMESA download + the GWPT
    join):

    ``ar_md.csv``
        Plant metadata: ``ID`` (central name), ``lon``, ``lat``, ``height``
        (m), ``capacity`` (kW), ``model``, ``type``, plus provenance/context
        (``height_source``, ``model_source``, ``site_name``, ``region``,
        ``provincia``).
    ``ar_obs.csv``
        Monthly observations, wide: ``ID``, ``year``, ``obs_1``..``obs_12``.
        Monthly CF (GWh against GWPT capacity), commissioning prefix stripped.
    """

    name: ClassVar[str] = "cammesa-ar"
    obs_level: ClassVar[ObsLevel] = "turbine"  # mechanical contract; obs_unit is "plant"
    countries: ClassVar[tuple[str, ...]] = ("AR", "ARG")

    def __init__(self, country: str = "AR") -> None:
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
        return PyVWFPaths.TURBINE_DATA / "AR"

    def load_metadata(self) -> pd.DataFrame:
        path = self._data_dir() / "ar_md.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"CAMMESA-AR plant metadata not found at {path}. This adapter "
                "reads pre-processed files; see the class docstring for the "
                "schema (data acquisition is user-executed: docs/runbooks/ar.md)."
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

        path = self._data_dir() / "ar_obs.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"CAMMESA-AR observations not found at {path}. This adapter "
                "reads pre-processed files; see the class docstring for the "
                "schema (data acquisition is user-executed: docs/runbooks/ar.md)."
            )
        wide = pd.read_csv(path)
        wide["ID"] = wide["ID"].astype(str)
        return wide[
            (wide["year"] >= int(year_start)) & (wide["year"] <= int(year_end))
        ].reset_index(drop=True)


__all__ = ["CAMMESAArgentinaSource"]
