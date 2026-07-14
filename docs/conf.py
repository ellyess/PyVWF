"""Sphinx configuration for the PyVWF documentation.

The API reference is generated with autodoc from the Google-style docstrings in
``src/vwf``; the narrative guides in this folder are plain Markdown, rendered by
MyST. Build with::

    pip install -e ".[docs]"
    sphinx-build -b html docs docs/_build/html -W
"""
from __future__ import annotations

import vwf

project = "PyVWF"
author = "Ellyess F. Benmoufok"
copyright = "2026, Ellyess F. Benmoufok"
release = vwf.__version__
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # Google-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",              # the narrative guides are Markdown
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # This folder's own index; superseded by index.md on the built site.
    "README.md",
    # Working notes from specific research runs rather than maintained
    # documentation. They stay in the repository but are not published, so the
    # site does not present stale run-specific numbers as guidance.
    "TRAINING_2015_2021_SUMMARY.md",
    "TURBINE_GRID_EVALUATION_ANALYSIS.md",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
# Render a class's "Attributes:" section as :ivar: fields rather than as separate
# object descriptions, which would collide with the same attributes picked up by
# autodoc's :members: (dataclasses like Results document both).
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

myst_enable_extensions = ["colon_fence", "deflist"]
# The guides link to each other as .md; let MyST resolve those to pages.
myst_heading_anchors = 3
suppress_warnings = ["myst.xref_missing"]

html_theme = "sphinx_rtd_theme"
html_title = f"PyVWF {release}"
