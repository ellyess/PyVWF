"""
PyVWF package.

Virtual Wind Farm (VWF) framework for bias-corrected wind resource modelling
based on ERA5 reanalysis and observational data.

Main public API
---------------
- PyVWF : core model class
- timeutils : shared time helper utilities
"""

# Version (optional but recommended)
__version__ = "0.1.0"

# Public API
from vwf.vwf import PyVWF
from vwf import timeutils

__all__ = [
    "PyVWF",
    "timeutils",
]
