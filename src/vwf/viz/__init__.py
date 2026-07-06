"""Plotting and diagnostic visualisation for PyVWF.

Submodules:
- ``vwf.viz.distribution``: distributional diagnostics (CF histograms, ECDFs, QQ)
  plus the ``Results`` loader used by the JOSS paper figures.
- ``vwf.viz.style``: shared matplotlib style for publication-quality figures.
- ``vwf.viz.palettes``: Okabe-Ito categorical colour palettes.

The most commonly used objects are re-exported at the package level so existing
``from vwf.viz import ...`` imports keep working.
"""

from vwf.viz.distribution import (
    Results,
    load_results,
    plot_cf_distribution,
    plot_qq,
)

__all__ = [
    "Results",
    "load_results",
    "plot_cf_distribution",
    "plot_qq",
]
