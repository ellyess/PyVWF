"""The package version, in the one place it is written.

A leaf module, so that code below the root package (the harness provenance and
export) can record the version without importing ``vwf`` itself, which pulls
in the legacy class and matplotlib. ``vwf.__version__`` re-exports it, and
``pyproject.toml`` reads it from here.
"""

__version__ = "0.5.1"
