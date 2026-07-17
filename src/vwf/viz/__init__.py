"""Plotting and diagnostic visualisation for PyVWF.

Submodules:
- ``vwf.viz.distribution``: distributional diagnostics (CF histograms, ECDFs, QQ)
  plus the ``Results`` loader used by the JOSS paper figures.
- ``vwf.viz.factors``: what the correction learned: factor maps per cluster
  and the scalar/offset joint distribution.
- ``vwf.viz.evaluation``: error vs cluster count / temporal resolution, and
  the per-turbine sim-vs-obs bias scatter.
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
from vwf.viz.evaluation import plot_error_vs_clusters, plot_sim_vs_obs
from vwf.viz.factors import plot_correction_factor_map, plot_factor_joint

__all__ = [
    "Results",
    "load_results",
    "plot_cf_distribution",
    "plot_correction_factor_map",
    "plot_error_vs_clusters",
    "plot_factor_joint",
    "plot_qq",
    "plot_sim_vs_obs",
]
