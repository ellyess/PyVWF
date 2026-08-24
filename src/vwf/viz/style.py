
"""
Shared plotting style for PyVWF figures.

Conventions:
- Serif fonts, small sizes (print-ready)
- Log spatial resolution axes are ALWAYS coarse -> fine (10^5 -> 10^3)
- Clean axes (no top/right spines)
- PDF-safe fonts
- Standard form with comma separators; scientific notation for >= 1,000,000
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


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
    FULL_WIDTH = 14.7 * cm   # \textwidth  – use for single-column figures
    HALF_WIDTH = 7.2 * cm    # ≈ 0.49\textwidth – use for side-by-side subfigures
    THIRD_WIDTH = 4.6 * cm   # ≈ 0.31\textwidth – use for 3-across subfigure grids
    MAP_WIDTH = 12.5 * cm    # ≈ 0.85\textwidth – use for maps with margins

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


def savefig(
    fig, path, *, close: bool = True, pdf_size_limit_mb: float = 5.0, **kwargs
):
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


def apply_spatial_resolution_axis(
    ax: plt.Axes,
    *,
    xlabel: str = r"Spatial Resolution ($A_{region}^{max}$) [km$^{2}$]",
    annotate: bool = True,
) -> None:
    """
    Apply standard spatial-resolution axis formatting.

    Enforces:
    - log scale
    - inverted axis (COARSE -> FINE, 10^5 -> 10^3)
    - standard label
    - optional 'Coarse' / 'Fine' annotations
    """
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(xlabel)

    if annotate:
        fs = mpl.rcParams.get("xtick.labelsize", 6)
        ax.annotate("Coarse", xy=(0.02, -0.28), xycoords="axes fraction",
                    fontsize=fs, ha="left", va="top")
        ax.annotate("Fine", xy=(0.98, -0.28), xycoords="axes fraction",
                    fontsize=fs, ha="right", va="top")


def add_resolution_markers(ax: plt.Axes, xvals) -> None:
    """Add dashed vertical lines at the coarsest/finest resolution with labels.

    Note: the x-axis is inverted (fine = smallest value on the right).
    """
    xvals = np.asarray(xvals, dtype=float)
    x_min = float(np.nanmin(xvals))
    x_max = float(np.nanmax(xvals))

    ax.axvline(x_min, linestyle="--", linewidth=0.8, color="grey", alpha=0.7)
    ax.axvline(x_max, linestyle="--", linewidth=0.8, color="grey", alpha=0.7)

    y0, y1 = ax.get_ylim()
    y = y1 - 0.02 * (y1 - y0)

    ax.text(x_min, y, "Fine", ha="right", va="top",
            fontsize=mpl.rcParams.get("xtick.labelsize", 6))
    ax.text(x_max, y, "Coarse", ha="left", va="top",
            fontsize=mpl.rcParams.get("xtick.labelsize", 6))


# ---------------------------------------------------------------------------
# Standard-form tick formatting (commas, no scientific notation)
# ---------------------------------------------------------------------------

def _comma_formatter(x, pos):
    """Format tick values with commas (for axes where values < 1,000,000)."""
    if x != x:  # NaN
        return ""
    if x == 0:
        return "0"
    abs_x = abs(x)
    # Integer-valued: use commas
    if abs_x >= 1 and float(x) == int(x):
        return f"{int(x):,}"
    # Large non-integer: commas with minimal decimals
    if abs_x >= 1000:
        return f"{x:,.1f}".rstrip("0").rstrip(".")
    # Moderate numbers
    if abs_x >= 1:
        return f"{x:.2f}".rstrip("0").rstrip(".")
    # Small numbers
    if abs_x >= 0.01:
        return f"{x:.3f}".rstrip("0").rstrip(".")
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _axis_has_large_values(axis) -> bool:
    """Check whether an axis has tick values >= 1,000,000."""
    try:
        ticks = axis.get_ticklocs()
        if len(ticks) == 0:
            return False
        return float(np.max(np.abs(ticks))) >= 1_000_000
    except Exception:
        return False


class _CleanSciFormatter(mticker.ScalarFormatter):
    """ScalarFormatter that shows '10^n' instead of '×10^n' in the offset."""

    def get_offset(self):
        text = super().get_offset()
        return text.replace(r"\times", "")


def format_axes_standard(fig):
    """Apply appropriate tick formatting to all axes in a figure.

    - Axes with values < 1,000,000: comma-separated (e.g. 100,000)
    - Axes with values >= 1,000,000: matplotlib offset notation (e.g. x10^6
      label on the axis, with simple tick numbers like 1, 2, 3)
    - Log axes: comma-separated (e.g. 100,000 instead of 10^5)
    - Custom formatters (FuncFormatter, etc.) are left untouched.
    """
    _default_types = (mticker.ScalarFormatter, mticker.LogFormatterSciNotation)
    for ax in fig.get_axes():
        for axis in [ax.xaxis, ax.yaxis]:
            formatter = axis.get_major_formatter()
            if not isinstance(formatter, _default_types):
                continue
            # Log axes always get comma formatter
            if isinstance(formatter, mticker.LogFormatterSciNotation):
                axis.set_major_formatter(mticker.FuncFormatter(_comma_formatter))
                continue
            # Linear axes: use offset notation for large values
            if _axis_has_large_values(axis):
                sf = _CleanSciFormatter(useMathText=True)
                sf.set_powerlimits((-3, 5))  # use offset for >= 10^6
                axis.set_major_formatter(sf)
            else:
                axis.set_major_formatter(mticker.FuncFormatter(_comma_formatter))
