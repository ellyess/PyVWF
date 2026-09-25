"""The country-level joint offset fit closes a small error instead of stopping at zero.

The joint fit minimises the squared capacity-factor error, about 1e-4 at the
start of a typical fit. L-BFGS-B's ``ftol`` acts as an absolute threshold on a
value that small, and at the former 1e-6 a fit whose first step lowered it by
less stopped there: an error of 3e-3 was left in full with every offset still
at zero. The fits recorded on 2026-09-25 did this in 47 of 540 national
periods (45 stopped after one iteration, 2 after two), all in ES, NO and BE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyvwf import correction
from pyvwf.wind import train_simulate_wind


def _fleet(n_clusters):
    return pd.DataFrame(
        {
            "ID": ["a", "b", "c", "d"],
            "lat": [55.2, 55.4, 55.6, 55.8],
            "lon": [8.2, 8.4, 8.6, 8.8],
            "height": [100.0] * 4,
            "capacity": [1000.0, 3000.0, 2000.0, 2000.0],
            "model": ["GE.1.5sle"] * 4,
            "cluster": [0, 0, 0, 0] if n_clusters == 1 else [0, 0, 1, 1],
        }
    )


def _national_cf(fleet, offsets, reanalysis, power_curve):
    """The capacity-weighted national CF the objective compares with the observation."""
    period = reanalysis.sel(time=reanalysis.time.dt.year == 2020)
    clusters = sorted(fleet["cluster"].unique())
    cfs = [
        train_simulate_wind(period, fleet[fleet["cluster"] == c], power_curve, 1.0, offsets[c])
        for c in clusters
    ]
    weights = fleet.groupby("cluster")["capacity"].sum().loc[clusters].to_numpy()
    return float(np.dot(cfs, weights) / weights.sum())


@pytest.mark.parametrize("n_clusters", [1, 2])
@pytest.mark.parametrize("gap", [3e-3, 1e-3, 1e-4, -1e-3, 0.1])
def test_the_fit_closes_the_error(n_clusters, gap, reanalysis, power_curve):
    fleet = _fleet(n_clusters)
    zero = {c: 0.0 for c in range(n_clusters)}
    obs = _national_cf(fleet, zero, reanalysis, power_curve) + gap
    offsets = correction.find_offsets_country_level(
        year=2020,
        time_slice="1/1",
        obs_country_cf=obs,
        scalars_by_cluster={c: 1.0 for c in range(n_clusters)},
        turb_info=fleet,
        reanalysis=reanalysis,
        powerCurveFile=power_curve,
    )
    residual = _national_cf(fleet, offsets, reanalysis, power_curve) - obs
    # The same bound the bracketed per-cluster search holds its roots to.
    assert abs(residual) < correction.BRACKETED_MAX_RESIDUAL
    # The offsets moved in the direction of the error.
    assert all(np.sign(v) == np.sign(gap) for v in offsets.values())
