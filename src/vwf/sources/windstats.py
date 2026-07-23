"""Per-turbine observation source for the CONFIDENTIAL WindStats regions.

⚠ CONFIDENTIAL / COMMERCIAL (WindStats) generation, with open (GWPT)
coordinates for Spain — a MIXED-LICENCE region. Nothing derived may be
committed or redistributed; the adapter reads pre-processed files under
``input/turbine_level_data/<CC>/`` that the user builds locally with
``scripts/process/windstats.py`` from data they hold.

Currently serves **Spain (ES)**. Sweden (SE) and Finland (FI) share the
WindStats format but need a thewindpower.net coordinate table (GWPT
under-covers their fleets); they register here once that is supplied — add the
code to ``countries`` and ship their ``<cc>_md.csv``/``<cc>_obs.csv``.

Data-window caveat: the WindStats extracts are historical — ES generation is
1998-2000, SE 1998-2013, FI 2005-2012 — so these are old-fleet regions, not
contemporaries of the 2015-2019 reference set. Training them needs ERA5 for the
matching years.
"""
from __future__ import annotations

from typing import ClassVar

import pandas as pd

from vwf.config import PyVWFPaths
from vwf.sources.base import ObservationSource, ObsLevel
from vwf.sources.registry import register

#: Default training window per WindStats country (inclusive), from each
#: extract's actual coverage; the last covered year is held out for test.
DEFAULT_TRAIN_YEARS: dict[str, tuple[int, int]] = {
    "ES": (1998, 1999),
}


def _base_country(code: str) -> str:
    """WindStats country from a region code, e.g. 'ES-WS' -> 'ES'.

    The region code carries a '-WS' suffix so it does not collide with the
    country-level ENTSO-E region of the same country (e.g. the ENTSO-E 'ES').
    """
    return code.upper().split("-")[0]


@register
class WindStatsSource(ObservationSource):
    """Per-turbine monthly capacity factors for the WindStats regions.

    Reads pre-processed files from ``input/turbine_level_data/<CC>/`` (built by
    ``scripts/process/windstats.py`` from the confidential WindStats extract +
    the coordinate join):

    ``<cc>_md.csv``
        Turbine metadata: ``ID``, ``lon``, ``lat``, ``height`` (m),
        ``capacity`` (kW), ``diameter`` (m), ``model``, ``type``, plus
        provenance (``height_source``, ``model_source``, ``coord_source``).
    ``<cc>_obs.csv``
        Monthly CF, wide: ``ID``, ``year``, ``obs_1``..``obs_12``.
    """

    name: ClassVar[str] = "windstats"
    obs_level: ClassVar[ObsLevel] = "turbine"
    #: Region codes carry a '-WS' suffix to avoid colliding with country-level
    #: ENTSO-E regions. SE-WS/FI-WS join once their coordinates are supplied.
    countries: ClassVar[tuple[str, ...]] = ("ES-WS",)

    def __init__(self, country: str) -> None:
        self.country: str = country.upper()
        if _base_country(self.country) not in {"ES", "SE", "FI"}:
            raise ValueError(
                f"{type(self).__name__} supports ES-WS/SE-WS/FI-WS, got {country!r}"
            )

    @property
    def default_train_years(self) -> tuple[int, int]:
        return DEFAULT_TRAIN_YEARS.get(_base_country(self.country), (1998, 1999))

    def _data_dir(self):
        return PyVWFPaths.TURBINE_DATA / _base_country(self.country)

    def _prefix(self) -> str:
        return _base_country(self.country).lower()

    def load_metadata(self) -> pd.DataFrame:
        path = self._data_dir() / f"{self._prefix()}_md.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"WindStats metadata not found at {path}. Built from CONFIDENTIAL "
                "WindStats data by scripts/process/windstats.py — see "
                "docs/RUNBOOK_ES.md. (Data is git-ignored and commercial.)"
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
        if year_start is None or year_end is None:
            year_start, year_end = self.default_train_years
        path = self._data_dir() / f"{self._prefix()}_obs.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"WindStats observations not found at {path}. Built from "
                "CONFIDENTIAL WindStats data by scripts/process/windstats.py — "
                "see docs/RUNBOOK_ES.md."
            )
        wide = pd.read_csv(path)
        wide["ID"] = wide["ID"].astype(str)
        return wide[
            (wide["year"] >= int(year_start)) & (wide["year"] <= int(year_end))
        ].reset_index(drop=True)


__all__ = ["WindStatsSource"]
