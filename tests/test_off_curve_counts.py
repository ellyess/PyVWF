"""Simulated values the power curves cannot convert are counted, not hidden.

A corrected speed below 0 m/s or above the curve table's last speed has no
value on the curve, and the interpolator returns NaN rather than zero output.
A monthly mean then skips it, so a unit-month can be scored on only the steps
the simulation could handle. In the Spanish country row, fitted offsets pushed
more than half of two clusters' days below zero, and the dropped days made the
corrected capacity factor read high. These tests pin the count and its record.
"""

import json

import numpy as np
import pandas as pd
import pytest

from test_harness_driver import make_spec
from vwf.harness import driver
from vwf.harness.driver import run_evaluate, run_train
from vwf.wind import off_curve_record

CURVES = pd.DataFrame({"data$speed": np.linspace(0.0, 40.0, 401), "m": 0.5})


def _frames():
    time = pd.date_range("2016-01-01", "2016-02-29", freq="D")  # 31 + 29 days
    ws = pd.DataFrame({"time": time, "a": 8.0, "b": 8.0})
    ws.loc[:4, "a"] = -1.0  # 5 January days below the curve
    ws.loc[31:, "b"] = 45.0  # every February day above it
    ws.loc[10, "a"] = np.nan  # one day with no speed
    cf = ws.copy()
    for col in ("a", "b"):
        off = (ws[col] < 0) | (ws[col] > 40) | ws[col].isna()
        cf[col] = np.where(off, np.nan, 0.3)
    return ws, cf


def test_the_record_separates_the_three_routes():
    ws, cf = _frames()
    capacity = pd.Series({"a": 1.0, "b": 3.0})
    r = off_curve_record(ws, cf, capacity, CURVES)
    steps = 60 * 4.0  # 60 days, total weight 4
    assert r["off_curve_below_share"] == pytest.approx(5 * 1.0 / steps)
    assert r["off_curve_above_share"] == pytest.approx(29 * 3.0 / steps)
    assert r["no_speed_share"] == pytest.approx(1 * 1.0 / steps)
    # a/Jan loses 6 days (partly); b/Feb loses all of February (wholly).
    assert r["unit_months_wholly_missing"] == 1
    assert r["unit_months_partly_missing"] == 1


def test_a_clean_frame_records_zeros():
    time = pd.date_range("2016-01-01", periods=31, freq="D")
    ws = pd.DataFrame({"time": time, "a": 8.0})
    cf = ws.assign(a=0.3)
    r = off_curve_record(ws, cf, pd.Series({"a": 1.0}), CURVES)
    assert r == {
        "off_curve_below_share": 0.0,
        "off_curve_above_share": 0.0,
        "no_speed_share": 0.0,
        "unit_months_wholly_missing": 0,
        "unit_months_partly_missing": 0,
    }


def test_evaluate_records_off_curve_values_per_variant(synthetic_dk, monkeypatch):
    """End to end: one unit's first five corrected days are pushed below the
    curve, as a large negative offset would. The corrected row records them,
    the uncorrected row does not, and the month they fall in counts as partly
    missing."""
    spec = make_spec()
    out = synthetic_dk["root"] / "validation"
    train_dir = run_train(spec, out, mode="onshore", run_name="t1")

    clean_dir = run_evaluate(spec, train_dir, out, mode="onshore", run_name="clean")
    clean = pd.read_csv(clean_dir / "metrics.csv")
    for col in ("off_curve_below_share", "off_curve_above_share", "no_speed_share"):
        assert (clean[col] == 0).all()

    real_get = driver.get_correction

    class PushBelowCurve:
        def __init__(self, model):
            self._model = model

        def apply(self, *args, **kwargs):
            ws, cf = self._model.apply(*args, **kwargs)
            unit = [c for c in cf.columns if c != "time"][0]
            ws, cf = ws.copy(), cf.copy()
            ws.loc[:4, unit] = -1.0
            cf.loc[:4, unit] = np.nan
            return ws, cf

    monkeypatch.setattr(driver, "get_correction", lambda name: PushBelowCurve(real_get(name)))
    with pytest.warns(UserWarning, match="could not convert"):
        eval_dir = run_evaluate(spec, train_dir, out, mode="onshore", run_name="pushed")

    metrics = pd.read_csv(eval_dir / "metrics.csv")
    unc = metrics[metrics["variant"] == "uncorrected"].iloc[0]
    cor = metrics[metrics["variant"] == "affine-wind"].iloc[0]
    assert unc["off_curve_below_share"] == 0
    assert cor["off_curve_below_share"] > 0
    assert cor["unit_months_partly_missing"] == 1
    assert cor["unit_months_wholly_missing"] == 0
    # The fit-quality columns keep their place after the metrics.
    cols = list(metrics.columns)
    assert cols.index("n_clusters") < cols.index("off_curve_below_share")

    record = json.loads((eval_dir / "run_manifest.json").read_text())["off_curve"]
    assert set(record) == {"uncorrected", "fixed_2"}
    assert record["fixed_2"]["off_curve_below_share"] == pytest.approx(cor["off_curve_below_share"])
