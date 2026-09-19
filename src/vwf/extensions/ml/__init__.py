"""Machine-learning transfer of correction factors across regions.

Holds the shared machinery of the leave-one-region-out transfer studies
(``docs/findings/method-ml-transfer.md``). It needs only the core
dependencies: scikit-learn is one of them.
"""

from vwf.extensions.ml.transfer import (
    RF_KW,
    SEEDS,
    SET_A,
    SET_B,
    SET_C,
    build_centroids,
    loro,
    random_cv,
    rf_eval,
    terrain_features,
    variance_decomposition,
)

__all__ = [
    "RF_KW",
    "SEEDS",
    "SET_A",
    "SET_B",
    "SET_C",
    "build_centroids",
    "loro",
    "random_cv",
    "rf_eval",
    "terrain_features",
    "variance_decomposition",
]
