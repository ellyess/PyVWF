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
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    # MathJax is Sphinx's default HTML math renderer; listed explicitly so the
    # dollar-math in index.md keeps rendering if the defaults ever change.
    "sphinx.ext.mathjax",
    "myst_parser",  # the narrative guides are Markdown
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # This folder's own index; superseded by index.md on the built site.
    "README.md",
    # Working notes from specific research runs rather than maintained
    # documentation. They stay in the repository, and are readable on GitHub,
    # but are not published, so the site does not present stale run-specific
    # numbers as guidance. The whole findings/ tree is excluded on that same
    # reasoning: each document answers one research question against a single
    # test year, and several record negative results whose value is the
    # reasoning rather than the number. The maintained documentation is
    # guides/, runbooks/ and design/, all of which are in the toctree.
    "findings/**",
    # A working plan kept in design/ for its history, but not explanation for
    # a reader new to the project: the open decision register for the thesis
    # chapters 4 and 5 manuscript.
    "design/manuscript-chapters-45.md",
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

# dollarmath: index.md writes math as $...$ / $$...$$, which GitHub also
# renders natively, so the same source works on both surfaces.
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
# The guides link to each other as .md; let MyST resolve those to pages.
myst_heading_anchors = 3
suppress_warnings = ["myst.xref_missing"]

# Furo: a light and dark theme with a two-column layout, so the page's own
# headings stay visible beside the site navigation. It needs no custom CSS
# here: the guides carry no diagrams of their own, and the one Mermaid
# flowchart in the project README is rendered by GitHub rather than by Sphinx,
# because README.md is excluded from this build.
html_theme = "furo"
html_title = f"PyVWF {release}"
html_theme_options = {
    # Puts a "view this page's source" link on every page, pointing at the
    # Markdown in the repository rather than at the generated reStructuredText.
    "source_repository": "https://github.com/ellyess/PyVWF/",
    "source_branch": "main",
    "source_directory": "docs/",
}
