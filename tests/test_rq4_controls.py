"""Controls for the country-level identifiability question (RQ4).

Two things needed pinning before the RQ4 comparison means anything.

The **zero-offset control** (`scalar-only`) separates the two halves of an
N-cluster country fit. The scalars are identified; the offsets, against one
national observation, are under-determined by N-1. If the improvement over the
single-cluster baseline comes from the scalars, the joint offset fit is not
buying anything worth its ambiguity. The control only works if the scalars are
bit-identical to the affine model's, which is what these tests check.

**Per-zone scoring** exists because the national metric is the joint
optimiser's own objective. Scoring a zonal fit on the national aggregate
flatters the national fit by construction, so the comparison needed a metric on
the quantity a zonal fit actually targets.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vwf.harness.corrections import available_corrections, get_correction
from vwf.harness.driver import _country_skill, _zonal_skill


@pytest.fixture
def grid():
    return pd.DataFrame(
        {
            "ID": ["g0", "g1", "g2", "g3"],
            "lon": [8.1, 8.3, 9.2, 9.4],
            "lat": [55.2, 55.4, 55.6, 55.8],
            "height": [100.0] * 4,
            "capacity": [1000.0, 3000.0, 2000.0, 2000.0],
            "model": ["2019COE_Market_Average_2.6MW_121"] * 4,
            "cluster": [0, 0, 1, 1],
            "type": ["onshore"] * 4,
        }
    )


@pytest.fixture
def paired(grid):
    """Training pairs: two clusters, one national observation, two years."""
    rng = np.random.default_rng(0)
    frames = []
    for year in (2015, 2016):
        for month in range(1, 13):
            frames.append(
                pd.DataFrame(
                    {
                        "year": year,
                        "month": month,
                        "ID": grid["ID"],
                        "sim": 0.15 + 0.2 * rng.random(len(grid)),
                        "obs": 0.22,
                    }
                )
            )
    out = pd.concat(frames, ignore_index=True)
    out["fixed"] = "1/1"
    return out


# ---------------------------------------------------------------------------
# The zero-offset control
# ---------------------------------------------------------------------------

def test_scalar_only_is_registered():
    assert "scalar-only" in available_corrections()


def test_scalar_only_offsets_are_all_zero(paired, grid):
    factors, _ = get_correction("scalar-only").fit(
        paired, grid, None, pd.DataFrame(), num_clusters=2,
        time_res="fixed", obs_level="country",
    )
    assert (factors["offset"] == 0).all()


def test_scalar_only_needs_no_reanalysis(paired, grid):
    """It must not touch the offset solver, which is the expensive part and the
    part under test. Passing None for the reanalysis proves it never gets there."""
    factors, _ = get_correction("scalar-only").fit(
        paired, grid, None, pd.DataFrame(), num_clusters=2,
        time_res="fixed", obs_level="country",
    )
    assert len(factors) == 2


def test_scalar_only_scalars_match_the_affine_models(paired, grid, monkeypatch):
    """The control is only valid if the two models differ in exactly one term.
    The affine model's offset solver is stubbed out so the comparison can run
    without ERA5, leaving the scalars as the only thing being compared."""
    import vwf.correction as correction

    monkeypatch.setattr(
        correction, "find_offsets_country_level",
        lambda **kw: {c: 0.5 for c in kw["scalars_by_cluster"]},
    )

    affine, _ = get_correction("affine-wind").fit(
        paired, grid, None, pd.DataFrame(), num_clusters=2,
        time_res="fixed", obs_level="country",
    )
    scalar_only, _ = get_correction("scalar-only").fit(
        paired, grid, None, pd.DataFrame(), num_clusters=2,
        time_res="fixed", obs_level="country",
    )

    merged = affine.merge(scalar_only, on=["cluster", "fixed"], suffixes=("_a", "_s"))
    assert len(merged) == 2
    np.testing.assert_allclose(merged["scalar_a"], merged["scalar_s"])
    # The stub returned 0.5, so a nonzero affine offset confirms the comparison
    # would have detected a difference had one existed.
    assert (merged["offset_a"] != 0).all()
    assert (merged["offset_s"] == 0).all()


def test_scalar_only_inherits_the_turbine_path(paired, grid):
    factors, clus_info = get_correction("scalar-only").fit(
        paired, grid, None, pd.DataFrame(), num_clusters=2,
        time_res="fixed", obs_level="turbine",
    )
    assert (factors["offset"] == 0).all()
    assert "cluster" in clus_info.columns


# ---------------------------------------------------------------------------
# Per-zone scoring
# ---------------------------------------------------------------------------

def sim_wide(cluster_cfs, grid, periods=24):
    """Wide (time x ID) simulation where each cluster holds a constant CF."""
    times = pd.date_range("2023-01-01", periods=periods, freq="15D")
    data = {"time": times}
    for _, row in grid.iterrows():
        data[row["ID"]] = np.full(periods, cluster_cfs[row["cluster"]])
    return pd.DataFrame(data)


def zonal_obs(cluster_cfs, periods=24):
    times = pd.date_range("2023-01-01", periods=periods, freq="15D", tz="UTC")
    return pd.concat(
        [
            pd.DataFrame({"capacity_factor": cf, "cluster": c}, index=times)
            for c, cf in cluster_cfs.items()
        ]
    )


def test_per_zone_scoring_sees_error_the_national_metric_cancels(grid):
    """The decisive case: one zone over-predicted and the other under-predicted
    by the same capacity-weighted amount. The national aggregate is perfect and
    every zone is wrong, so a national-only metric reports success."""
    sim = sim_wide({0: 0.30, 1: 0.10}, grid)
    obs = zonal_obs({0: 0.10, 1: 0.30})

    per_zone = _zonal_skill(sim, obs, grid)
    assert per_zone["rmse"] == pytest.approx(0.20)
    assert per_zone["n_zones"] == 2

    # Capacity weights are 4000 and 4000, so the national means coincide.
    national_obs = pd.DataFrame({"time": obs.index.unique(), "obs": 0.20})
    national = _country_skill(sim, national_obs, grid)
    assert national["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_per_zone_and_national_agree_when_every_zone_is_right(grid):
    sim = sim_wide({0: 0.25, 1: 0.25}, grid)
    obs = zonal_obs({0: 0.25, 1: 0.25})
    assert _zonal_skill(sim, obs, grid)["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_per_zone_pools_every_zone_month(grid):
    sim = sim_wide({0: 0.30, 1: 0.10}, grid, periods=24)
    obs = zonal_obs({0: 0.10, 1: 0.30}, periods=24)
    result = _zonal_skill(sim, obs, grid)
    # 24 fortnightly steps span 12 months; two zones pooled.
    assert result["n_months"] == 24
    assert result["n_zones"] == 2


def test_per_zone_skips_a_zone_with_no_observation(grid):
    sim = sim_wide({0: 0.30, 1: 0.10}, grid)
    obs = zonal_obs({0: 0.30})
    result = _zonal_skill(sim, obs, grid)
    assert result["n_zones"] == 1
    assert result["rmse"] == pytest.approx(0.0, abs=1e-9)


def test_per_zone_returns_empty_metrics_when_nothing_lines_up(grid):
    sim = sim_wide({0: 0.30, 1: 0.10}, grid)
    result = _zonal_skill(sim, zonal_obs({7: 0.30}), grid)
    assert result["n_zones"] == 0
    assert np.isnan(result["rmse"])


def test_per_zone_weights_within_a_zone_by_capacity(grid):
    """Within a zone the aggregate is capacity-weighted, matching how the
    national metric and the training path both aggregate."""
    times = pd.date_range("2023-01-01", periods=4, freq="15D")
    sim = pd.DataFrame(
        {"time": times, "g0": 0.10, "g1": 0.30, "g2": 0.20, "g3": 0.20}
    )
    # Zone 0 is 1000 MW at 0.10 and 3000 MW at 0.30, so 0.25.
    obs = zonal_obs({0: 0.25, 1: 0.20}, periods=4)
    assert _zonal_skill(sim, obs, grid)["rmse"] == pytest.approx(0.0, abs=1e-9)
