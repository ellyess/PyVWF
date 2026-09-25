"""Pluggable observation sources for PyVWF.

Importing this package registers the built-in adapters. To add a new region,
subclass :class:`ObservationSource`, decorate it with :func:`register`, and
import the module so the decorator runs. See
``docs/guides/adding-an-adapter.md``.
"""

from pyvwf.sources.base import ObservationSource, ObsLevel
from pyvwf.sources.registry import (
    available_sources,
    get_source,
    register,
    resolve,
)

# Imported for their registration side effect.
from pyvwf.sources.aemo import AEMONemSource
from pyvwf.sources.cammesa_ar import CAMMESAArgentinaSource
from pyvwf.sources.cen_cl import CENChileSource
from pyvwf.sources.client_csv import ClientCsvTurbineSource
from pyvwf.sources.eia_us import EIAUSSource
from pyvwf.sources.emi_nz import EMINewZealandSource
from pyvwf.sources.entsoe_files import EntsoeFileSource
from pyvwf.sources.entsoe_zonal import EntsoeZonalFileSource
from pyvwf.sources.european import EuropeanTurbineSource
from pyvwf.sources.in_memory import InMemoryCountrySource
from pyvwf.sources.ons_br import ONSBrazilSource
from pyvwf.sources.windstats import WindStatsSource

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
