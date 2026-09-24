"""The power-curve interpolator cache cannot serve another table's curves.

The cache is keyed by ``id()`` of the curve table, because the lookup runs on
every objective evaluation. ``id()`` is only unique while an object is alive,
so an entry used to outlive its table: a later table given the same id, with
the same columns, got the dead table's interpolators, and entries accumulated
for the life of the process. Entries are now tied to their table's lifetime.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pandas as pd

from vwf import wind


def curves(level):
    speed = np.arange(0.0, 30.5, 0.5)
    return pd.DataFrame({"data$speed": speed, "M": np.clip(speed / 15.0, 0, 1) * level})


def test_a_live_table_is_served_from_the_cache():
    table = curves(1.0)
    first = wind._get_power_curve_cache(table)[1]
    assert wind._get_power_curve_cache(table)[1] is first


def test_the_entry_goes_when_the_table_is_collected():
    table = curves(1.0)
    wind._get_power_curve_cache(table)
    key = id(table)
    assert key in wind._power_curve_cache
    del table
    gc.collect()
    assert key not in wind._power_curve_cache


def test_a_stale_entry_under_a_reused_id_is_not_served():
    class Dead:
        pass

    table = curves(0.5)
    stale = object()
    # What a dead table's entry looked like once its id was reused: same
    # columns, interpolators built from other values.
    wind._power_curve_cache[id(table)] = {
        "table": weakref.ref(Dead()),
        "columns": tuple(table.columns),
        "x": table["data$speed"].to_numpy(),
        "curve_by_model": stale,
    }
    _, curve_by_model = wind._get_power_curve_cache(table)
    assert curve_by_model is not stale
    assert curve_by_model["M"](15.0) == 0.5
