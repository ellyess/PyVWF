"""Pluggable observation sources for PyVWF.

Importing this package registers the built-in adapters. To add a new region,
subclass :class:`ObservationSource`, decorate it with :func:`register`, and
import the module so the decorator runs. See
``docs/guides/ADDING_AN_OBSERVATION_SOURCE.md``.
"""
from vwf.sources.base import ObservationSource, ObsLevel
from vwf.sources.registry import (
    available_sources,
    get_source,
    register,
    resolve,
)

# Imported for their registration side effect.
from vwf.sources.aemo import AEMONemSource
from vwf.sources.cammesa_ar import CAMMESAArgentinaSource
from vwf.sources.cen_cl import CENChileSource
from vwf.sources.client_csv import ClientCsvTurbineSource
from vwf.sources.eia_us import EIAUSSource
from vwf.sources.emi_nz import EMINewZealandSource
from vwf.sources.entsoe_files import EntsoeFileSource
from vwf.sources.entsoe_zonal import EntsoeZonalFileSource
from vwf.sources.european import EuropeanTurbineSource
from vwf.sources.in_memory import InMemoryCountrySource
from vwf.sources.ons_br import ONSBrazilSource
from vwf.sources.windstats import WindStatsSource

__all__ = [
    "ObservationSource",
    "ObsLevel",
    "available_sources",
    "get_source",
    "register",
    "resolve",
    "AEMONemSource",
    "CAMMESAArgentinaSource",
    "CENChileSource",
    "ClientCsvTurbineSource",
    "EIAUSSource",
    "EMINewZealandSource",
    "EntsoeFileSource",
    "EntsoeZonalFileSource",
    "EuropeanTurbineSource",
    "InMemoryCountrySource",
    "ONSBrazilSource",
    "WindStatsSource",
]
