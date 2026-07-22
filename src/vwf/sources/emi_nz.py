"""Farm-level observation source for New Zealand (EA EMI open data).

The unit of observation is the **farm** (``obs_unit = "farm"``): EMI's
``Generation_MD`` dataset reports metered half-hourly grid injection per
generating plant, and the processing step aggregates a farm's units onto one
series — the same design as the AU (farm/DUID), US (plant), and BR (complex)
adapters, where the mechanical ``obs_level`` stays "turbine".

Decisions baked into the finalisation, documented in the region config:

- **UTC bins from NZ trading periods.** EMI trading periods are New Zealand
  civil time with DST (46/48/50 periods per day); the processing step maps
  them to UTC through ``Pacific/Auckland`` before anything is binned, matching
  the ERA5/simulation convention — ``time_convention = "utc-monthly-bins"``.
- **Then-current capacity denominator.** Monthly CF is computed against the
  farm's effective-dated registered capacity from the EMI plant register (the
  ONS-style denominator), so staged builds (Turitea, Harapaki) are not biased
  low. A below-final-build mask (the AU registered-capacity pattern) is
  applied on top when present, removing the erratic partially-erected months.
- **Registry-grade metadata is hand-compiled.** EMI carries no coordinates or
  hub heights; the curated farm table (``configs/nz_wind_farms.csv``, with
  per-farm provenance) supplies coordinates, capacity, turbine model, and hub
  height. The fleet is ~20 farms, which is what makes hand-compilation viable
  — and makes NZ one of the few regions outside Europe with per-farm hub
  heights.
"""
from __future__ import annotations

from typing import ClassVar

import pandas as pd

from vwf.config import PyVWFPaths
from vwf.sources.base import ObservationSource, ObsLevel
from vwf.sources.registry import register

#: Training window used when no explicit year range is requested. Matches the
#: ERA5 fetch (2019-2024) with 2024 held out for evaluation.
DEFAULT_TRAIN_YEARS: tuple[int, int] = (2019, 2023)


def apply_month_mask(wide: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """NaN the ``(ID, year, month)`` cells listed in ``mask``.

    The same finalisation shape as the BR curtailment mask: an explicit list
    of months whose observed CF is unrepresentative (here: below-final-build
    commissioning months), applied at the source layer so the pure transforms
    stay policy free.
    """
    if mask is None or not len(mask):
        return wide
    out = wide.copy()
    mask_keys = set(
        zip(
            mask["ID"].astype(str),
            mask["year"].astype(int),
            mask["month"].astype(int),
        )
    )
    for m in range(1, 13):
        col = f"obs_{m}"
        hit = [
            (str(i), int(y), m) in mask_keys
            for i, y in zip(out["ID"], out["year"])
        ]
        out.loc[hit, col] = float("nan")
    return out


@register
class EMINewZealandSource(ObservationSource):
    """Per-farm monthly capacity factors for New Zealand.

    Reads pre-processed files from ``input/turbine_level_data/NZ/`` (built by
    ``scripts/process_emi_nz.py`` from the EMI downloads):

    ``nz_md.csv``
        Farm metadata: ``ID`` (farm key), ``lon``, ``lat``, ``height`` (m),
        ``capacity`` (kW, final build), ``model``, ``type``, optional
        ``commissioning_date``, plus provenance/context columns
        (``height_source``, ``model_source``, ``site_name``, ``island``).
        Joined from the curated farm table (``configs/nz_wind_farms.csv``)
        and the EMI plant register.
    ``nz_obs.csv``
        Monthly observations, wide: ``ID``, ``year``, ``obs_1``..``obs_12``
        — monthly mean CF in UTC bins against then-current registered
        capacity, coverage-screened, as written by the processing step from
        the half-hourly ``Generation_MD`` melt.
    ``nz_build_mask.csv`` (optional)
        ``ID``, ``year``, ``month`` rows to NaN because the farm was below
        its final build (commissioning months), as written by the processing
        step from the register capacity history.
    """

    name: ClassVar[str] = "emi-nz"
    obs_level: ClassVar[ObsLevel] = "turbine"  # mechanical contract; obs_unit is "farm"
    countries: ClassVar[tuple[str, ...]] = ("NZ", "NZL")

    def __init__(self, country: str = "NZ") -> None:
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
        return PyVWFPaths.TURBINE_DATA / "NZ"

    def load_metadata(self) -> pd.DataFrame:
        path = self._data_dir() / "nz_md.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"EMI-NZ farm metadata not found at {path}. This adapter reads "
                "pre-processed files; see the class docstring for the schema "
                "(data acquisition is a user-executed step — docs/RUNBOOK_NZ.md)."
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
        """Monthly capacity factors, build-mask applied when present."""
        if year_start is None or year_end is None:
            year_start, year_end = self.default_train_years

        path = self._data_dir() / "nz_obs.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"EMI-NZ observations not found at {path}. This adapter reads "
                "pre-processed files; see the class docstring for the schema "
                "(data acquisition is a user-executed step — docs/RUNBOOK_NZ.md)."
            )
        wide = pd.read_csv(path)
        wide["ID"] = wide["ID"].astype(str)
        wide = wide[
            (wide["year"] >= int(year_start)) & (wide["year"] <= int(year_end))
        ].reset_index(drop=True)

        mask_path = self._data_dir() / "nz_build_mask.csv"
        mask = pd.read_csv(mask_path) if mask_path.is_file() else None
        return apply_month_mask(wide, mask)
