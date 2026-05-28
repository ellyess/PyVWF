"""PyVWF ML extension: learn correction factors from terrain/environment features.

Public API (mirrors what the JOSS-paper companion scripts use):
- ``train_correction_model``
- ``predict_correction_grid``
- ``create_feature_matrix``
- ``export_ml_correction_grid``
- ``compare_interpolation_methods``

Optional model backends (``xgboost``, ``lightgbm``) are installed via the
``[ml]`` extra: ``pip install pyvwf[ml]``.

Drivers in ``scripts/pyvwf_ml/`` orchestrate terrain feature engineering,
turbine-level training, and figure generation.
"""

from vwf.extensions.ml.correction import (
    compare_interpolation_methods,
    create_feature_matrix,
    export_ml_correction_grid,
    predict_correction_grid,
    train_correction_model,
)

__all__ = [
    "compare_interpolation_methods",
    "create_feature_matrix",
    "export_ml_correction_grid",
    "predict_correction_grid",
    "train_correction_model",
]
