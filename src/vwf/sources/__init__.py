"""Pluggable observation sources for PyVWF.

Importing this package registers the built-in adapters. To add a new region,
subclass :class:`ObservationSource`, decorate it with :func:`register`, and
import the module so the decorator runs. See
``docs/ADDING_AN_OBSERVATION_SOURCE.md``.
"""
from vwf.sources.base import ObservationSource, ObsLevel
from vwf.sources.registry import (
    available_sources,
    get_source,
    register,
    resolve,
)

# Imported for their registration side effect.
from vwf.sources.european import EuropeanTurbineSource
from vwf.sources.in_memory import InMemoryCountrySource

__all__ = [
    "ObservationSource",
    "ObsLevel",
    "available_sources",
    "get_source",
    "register",
    "resolve",
    "EuropeanTurbineSource",
    "InMemoryCountrySource",
]
