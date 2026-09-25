"""The package version, in the one place it is written.

A leaf module, so that code below the root package (the harness provenance and
export) can record the version without importing ``pyvwf`` itself, which pulls
in the harness and matplotlib. ``pyvwf.__version__`` re-exports it, and
``pyproject.toml`` reads it from here.
"""

__version__ = "0.6.0"
