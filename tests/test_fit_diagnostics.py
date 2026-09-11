"""What a fitted pair does to the speeds it was fitted on.

``fit_quality`` bounds the scalar and checks each offset converged, but an
affine pair with a negative offset sends every speed below ``-offset/scalar``
to a negative corrected speed, which drops out of the fit's own objective. The
Spanish country row's clusters 0 and 3 did that to more than half of their
training days while passing both checks. These tests pin the diagnostics that
record it, and that ``fit_quality`` reports them without setting the dagger.
"""
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from test_harness_driver import make_spec, synthetic_dk  # noqa: F401  (fixture)
from vwf.harness.corrections import fit_quality
from vwf.harness.driver import run_evaluate, run_train
from vwf.wind import fit_diagnostics

CURVES = pd.DataFrame({"data$speed": np.linspace(0.0, 40.0, 401), "m": 0.5})


def _reanalysis():
    times = pd.date_range("2015-01-01", "2016-12-31", freq="D")
    shape = (len(times), 3, 3)
    return xr.Dataset(
        {"wnd100m": (("time", "lat", "lon"), np.full(shape, 8.0)),
         "roughness": (("time", "lat", "lon"), np.full(shape, 0.05))},
        coords={"time": times, "lat": [55.0, 55.5, 56.0], "lon": [8.0, 8.75, 9.5]},
    )


def _fleet():
    return pd.DataFrame({
        "ID": ["a", "b", "c"], "lon": [8.2, 8.6, 9.2], "lat": [55.2, 55.6, 55.8],
        "height": 100.0, "capacity": [1.0, 3.0, 2.0], "model": "m", "cluster": [0, 1, 2],
    })


def _factors():
    # Speeds are 8 m/s everywhere (hub at 100 m). Cluster 0 crosses zero at
    # 9 m/s, so every day goes negative; cluster 2's scalar of 6 sends every day
    # above 40 m/s; cluster 1 is clean.
    return pd.DataFrame({"cluster": [0, 1, 2], "fixed": "1/1",
                         "scalar": [1.0, 1.0, 6.0], "offset": [-9.0, 0.5, 0.0]})


def test_diagnostics_record_where_the_pair_sends_the_training_speeds():
    d = fit_diagnostics(_reanalysis(), _fleet(), _factors(), "fixed", CURVES, years=(2015, 2016))
    assert set(d["year"]) == {2015, 2016}
    by = d.groupby("cluster")[["weight_steps", "weight_below_zero", "weight_above_curve"]].sum()
    assert by.loc[0, "weight_below_zero"] == pytest.approx(by.loc[0, "weight_steps"])
    assert by.loc[2, "weight_above_curve"] == pytest.approx(by.loc[2, "weight_steps"])
    assert by.loc[1, ["weight_below_zero", "weight_above_curve"]].sum() == 0
    z = d.groupby("cluster")["zero_crossing_speed"].first()
    assert z[0] == pytest.approx(9.0)
    assert np.isnan(z[1]) and np.isnan(z[2])


def test_the_training_years_filter_the_steps():
    d = fit_diagnostics(_reanalysis(), _fleet(), _factors(), "fixed", CURVES, years=(2016, 2016))
    assert set(d["year"]) == {2016}
    assert d.loc[d["cluster"] == 0, "unit_steps"].sum() == 366


def test_fit_quality_reports_the_shares_beside_the_dagger():
    d = fit_diagnostics(_reanalysis(), _fleet(), _factors(), "fixed", CURVES)
    q = fit_quality(_factors(), diagnostics=d)
    assert q["max_below_zero_share"] == pytest.approx(1.0)
    assert q["max_above_curve_share"] == pytest.approx(1.0)
    # Per period, cluster 0 (weight 1) and cluster 2 (weight 2) of 6 are dropped.
    assert q["max_period_dropped_share"] == pytest.approx(3 / 6)
    # Recorded, not flagged: the dagger comes from the scalar bound only here.
    assert q["degenerate_clusters"] == "2"
    assert q["n_implausible_scalar"] == 1


def test_fit_quality_without_diagnostics_reports_nan():
    q = fit_quality(_factors())
    assert all(np.isnan(q[k]) for k in
               ("max_below_zero_share", "max_above_curve_share", "max_period_dropped_share"))


def test_train_writes_diagnostics_and_evaluate_reports_them(synthetic_dk):  # noqa: F811
    spec = make_spec()
    out = synthetic_dk["root"] / "validation"
    train_dir = run_train(spec, out, mode="onshore", run_name="t")
    d = pd.read_csv(train_dir / "fit_diagnostics_fixed_2.csv")
    assert {"cluster", "fixed", "year", "zero_crossing_speed", "weight_steps",
            "weight_below_zero", "weight_above_curve"} <= set(d.columns)
    assert set(d["year"]) == {2015}
    record = json.loads((train_dir / "run_manifest.json").read_text())["fit_diagnostics"]
    assert set(record["fixed_2"]) == {"max_below_zero_share", "max_above_curve_share",
                                      "max_period_dropped_share"}

    eval_dir = run_evaluate(spec, train_dir, out, mode="onshore", run_name="e")
    metrics = pd.read_csv(eval_dir / "metrics.csv")
    cor = metrics[metrics["variant"] == "affine-wind"].iloc[0]
    assert np.isfinite(cor["max_below_zero_share"])
    assert cor["max_below_zero_share"] == pytest.approx(record["fixed_2"]["max_below_zero_share"])
