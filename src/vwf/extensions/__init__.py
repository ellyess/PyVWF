"""Experimental, research-oriented extensions to PyVWF.

These submodules build on the core PyVWF model and are intended for research
use, not production. Each extension declares its own optional dependency group
in ``pyproject.toml``:

- ``vwf.extensions.grid``  - spatial interpolation of correction fields onto an
  ERA5/atlite grid. Requires ``pykrige`` (already a core dependency today).
- ``vwf.extensions.ml``    - machine-learning correction models. Requires
  ``scikit-learn`` (core) and optionally ``xgboost``/``lightgbm``
  (``pip install pyvwf[ml]``).

Drivers for these extensions live in ``scripts/pyvwf_to_grid/`` and
``scripts/pyvwf_ml/``; see ``PIPELINE.md`` for the full workflow.
"""
