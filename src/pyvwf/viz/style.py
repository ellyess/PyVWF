"""
Shared plotting style for PyVWF figures.

Conventions:
- Serif fonts, small sizes (print-ready)
- Clean axes (no top/right spines)
- PDF-safe fonts
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_style(
    *,
    font_family: str = "serif",
    base_fontsize: float = 7.0,
    tick_labelsize: float = 6.0,
    line_width: float = 1.2,
    marker_size: float = 3.5,
    dpi: int = 600,
) -> dict:
    """
    Apply PyVWF-wide matplotlib rcParams.

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
            # Prevent scientific notation on tick labels
            "axes.formatter.useoffset": False,
            "axes.formatter.limits": [-99, 99],
        }
    )

    cm = 1.0 / 2.54

    # Page dimensions (A4, KOMA-Script, BCOR=25mm, DIV=12)
    # textwidth = 418.26 pt = 14.7 cm = 5.787 in
    FULL_WIDTH = 14.7 * cm  # \textwidth  – use for single-column figures
    HALF_WIDTH = 7.2 * cm  # ≈ 0.49\textwidth – use for side-by-side subfigures
    THIRD_WIDTH = 4.6 * cm  # ≈ 0.31\textwidth – use for 3-across subfigure grids
    MAP_WIDTH = 12.5 * cm  # ≈ 0.85\textwidth – use for maps with margins

    return {
        "cm": cm,
        "lw": line_width,
        "ms": marker_size,
        "dpi": dpi,
        "FULL_WIDTH": FULL_WIDTH,
        "HALF_WIDTH": HALF_WIDTH,
        "THIRD_WIDTH": THIRD_WIDTH,
        "MAP_WIDTH": MAP_WIDTH,
    }


def savefig(fig, path, *, close: bool = True, pdf_size_limit_mb: float = 5.0, **kwargs):
    """Save a figure in PDF format with publication defaults, then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str or Path
        Output path.  Extension is forced to '.pdf' unless the figure
        contains rasterised artists, in which case '.png' is kept.
    close : bool
        Whether to close the figure after saving (default True).
    pdf_size_limit_mb : float
        If the saved PDF exceeds this size (in MB), re-save as 600 dpi PNG
        and delete the oversized PDF.  Set to 0 to disable.
    **kwargs
        Forwarded to ``fig.savefig``.
    """
    import os
    from pathlib import Path as _P

    path = _P(path)
    # Detect rasterised content (e.g. scatter with rasterized=True)
    has_raster = any(
        getattr(artist, "get_rasterized", lambda: False)()
        for ax in fig.get_axes()
        for artist in ax.get_children()
    )
    if not has_raster:
        path = path.with_suffix(".pdf")
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("pad_inches", 0.02)
    fig.savefig(path, **kwargs)
    # PDF size guard: fall back to high-res PNG if PDF is too large
    if (
        pdf_size_limit_mb > 0
        and path.suffix == ".pdf"
        and path.exists()
        and os.path.getsize(path) > pdf_size_limit_mb * 1_000_000
    ):
        size_mb = os.path.getsize(path) / 1_000_000
        png_path = path.with_suffix(".png")
        fig.savefig(png_path, dpi=600, **kwargs)
        path.unlink()
        print(f"  Warning: {path.name} was {size_mb:.1f} MB → saved as {png_path.name}")
    if close:
        plt.close(fig)
