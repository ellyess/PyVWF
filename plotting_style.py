
"""
Shared plotting style for thesis figures.

Conventions:
- Serif fonts, small sizes (print-ready)
- Log spatial resolution axes are ALWAYS coarse -> fine (10^5 -> 10^3)
- Clean axes (no top/right spines)
- PDF-safe fonts
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


def thesis_plot_style(
    *,
    font_family: str = "serif",
    base_fontsize: float = 7.0,
    tick_labelsize: float = 6.0,
    line_width: float = 1.2,
    marker_size: float = 3.5,
    dpi: int = 600,
) -> dict:
    """
    Apply thesis-wide matplotlib rcParams.

    Returns:
        dict with convenience constants:
        - cm: cm to inch conversion
        - lw: default line width
        - ms: default marker size
        - dpi: default dpi
    """
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.size": base_fontsize,
            "axes.titlesize": base_fontsize,
            "axes.labelsize": base_fontsize,
            "xtick.labelsize": tick_labelsize,
            "ytick.labelsize": tick_labelsize,
            "legend.fontsize": tick_labelsize,
            "mathtext.fontset": "dejavuserif",
            "mathtext.default": "it",
            "lines.linewidth": line_width,
            "lines.markersize": marker_size,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "xtick.bottom": True,
            "ytick.left": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "axes.grid": False,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    cm = 1.0 / 2.54
    return {"cm": cm, "lw": line_width, "ms": marker_size, "dpi": dpi}

