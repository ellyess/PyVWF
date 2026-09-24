"""The country-level joint offset fit refuses a failed fit instead of returning a value.

Before these checks, `find_offsets_country_level` returned whatever L-BFGS-B
stopped on: offsets from a fit that had not converged, offsets pinned to the
+/-10 m/s bound, and all-zero offsets when the optimiser raised. Each of those
was then counted as an accepted year. A refused period is NaN for every
cluster, which `format_bc_factors` leaves out of the accepted years.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

from vwf import correction


@pytest.fixture
def fleet():
    """Two clusters of two turbines inside the conftest grid."""
    return pd.DataFrame(
        {
            "ID": ["a", "b", "c", "d"],
            "lat": [55.2, 55.4, 55.6, 55.8],
            "lon": [8.2, 8.4, 8.6, 8.8],
            "height": [100.0] * 4,
            "capacity": [1000.0, 3000.0, 2000.0, 2000.0],
            "model": ["GE.1.5sle"] * 4,
            "cluster": [0, 0, 1, 1],
        }
    )


def _fit(obs, fleet, reanalysis, power_curve):
    return correction.find_offsets_country_level(
        year=2020,
        time_slice="1/1",
        obs_country_cf=obs,
        scalars_by_cluster={0: 1.0, 1: 1.0},
        turb_info=fleet,
        reanalysis=reanalysis,
        powerCurveFile=power_curve,
    )


def test_a_solvable_observation_gives_finite_offsets(fleet, reanalysis, power_curve):
    offsets = _fit(0.3, fleet, reanalysis, power_curve)
    assert set(offsets) == {0, 1}
    assert all(np.isfinite(v) and -10 < v < 10 for v in offsets.values())


def test_an_offset_on_a_bound_refuses_the_period(fleet, reanalysis, power_curve, monkeypatch):
    # The optimiser reports success with one offset pinned to +10 m/s; the old
    # code accepted it. The fit is joint, so the other offset is refused too.
    def pinned(fun, x0, *args, **kwargs):
        return OptimizeResult(x=np.array([10.0, 0.5]), success=True, status=0, message="ok")

    monkeypatch.setattr(correction, "minimize", pinned)
    with pytest.warns(UserWarning, match="on a bound"):
        offsets = _fit(0.3, fleet, reanalysis, power_curve)
    assert all(np.isnan(v) for v in offsets.values())


def test_an_optimiser_error_is_refused_not_zeroed(fleet, reanalysis, power_curve, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(correction, "minimize", boom)
    with pytest.warns(UserWarning, match="raised"):
        offsets = _fit(0.3, fleet, reanalysis, power_curve)
    assert all(np.isnan(v) for v in offsets.values())


def test_a_fit_that_did_not_converge_is_refused(fleet, reanalysis, power_curve, monkeypatch):
    def stalled(fun, x0, *args, **kwargs):
        return OptimizeResult(
            x=np.full(len(x0), 0.5), success=False, status=1, message="iteration limit"
        )

    monkeypatch.setattr(correction, "minimize", stalled)
    with pytest.warns(UserWarning, match="did not converge"):
        offsets = _fit(0.3, fleet, reanalysis, power_curve)
    assert all(np.isnan(v) for v in offsets.values())
