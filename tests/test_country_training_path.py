"""The country-level training path: fleet prep, monthly aggregation, clusters.

These pin the three places the country path used to differ from the
turbine-level path in ways nothing detected:

- ``train_set`` filtered the fleet and ``val_set`` did not, so evaluation could
  score a larger grid than training fitted;
- the monthly observation was a mean of instantaneous ratios rather than energy
  over possible energy, which drifts whenever capacity moves inside a month;
- ``num_clu`` was ignored outright, so a file named ``factors_fixed_5.csv``
  could hold four clusters and a manifest could claim five.

See docs/findings/method-country-level.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vwf.data import (
    _country_cluster_means,
    assign_country_clusters,
    cluster_train_set,
    country_cf_to_monthly,
    prepare_country_fleet,
)


@pytest.fixture
def curves():
    """Minimal power-curve table: only the column names are consulted here."""
    return pd.DataFrame(
        {
            "data$speed": np.arange(0, 41, 1.0),
            "2019COE_Market_Average_2.6MW_121": np.linspace(0, 1, 41),
        }
    )


@pytest.fixture
def grid():
    return pd.DataFrame(
        {
            "ID": ["g0", "g1", "g2", "g3"],
            "lon": [5.0, 5.25, 6.0, 6.25],
            "lat": [52.0, 52.0, 53.0, 53.0],
            "height": [100.0, 100.0, 100.0, 100.0],
            "capacity": [100.0, 300.0, 50.0, 50.0],
            "model": ["M", "M", "M", "M"],
            "cluster": [0, 0, 1, 1],
        }
    )


# ---------------------------------------------------------------------------
# Fleet preparation
# ---------------------------------------------------------------------------

def test_unparseable_fields_are_dropped(grid, curves):
    # pandas 3 raises rather than upcasting a float column in place, so widen
    # it first. The point of the test is what prepare_country_fleet does with
    # an unparseable capacity, not how the bad value got there.
    grid["capacity"] = grid["capacity"].astype(object)
    grid.loc[1, "capacity"] = "not a number"
    out = prepare_country_fleet(grid, curves)
    assert out["ID"].tolist() == ["g0", "g2", "g3"]
    assert out["capacity"].dtype.kind == "f"


def test_a_grid_without_models_gets_the_default_curve(grid, curves):
    grid["model"] = np.nan
    out = prepare_country_fleet(grid, curves)
    assert out["model"].nunique() == 1
    assert out["model"].iloc[0] in curves.columns


def test_fix_turb_wins_over_the_default(grid, curves):
    grid["model"] = np.nan
    out = prepare_country_fleet(grid, curves, fix_turb="Chosen")
    assert (out["model"] == "Chosen").all()


def test_preparation_does_not_mutate_the_caller(grid, curves):
    before = grid.copy()
    prepare_country_fleet(grid, curves)
    pd.testing.assert_frame_equal(grid, before)


# ---------------------------------------------------------------------------
# Monthly aggregation
# ---------------------------------------------------------------------------

def month_of_hours(gen_mw, cap_mw):
    idx = pd.date_range("2015-01-01", periods=len(gen_mw), freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "generation_mw": gen_mw,
            "capacity_mw": cap_mw,
            "capacity_factor": np.asarray(gen_mw) / np.asarray(cap_mw),
        },
        index=idx,
    )


def test_monthly_cf_is_energy_over_possible_energy():
    """Capacity doubles mid-month. The mean of ratios over-weights the small
    early fleet; energy over possible energy is what the fleet actually did."""
    n = 31 * 24
    cap = np.r_[np.full(n // 2, 1000.0), np.full(n - n // 2, 2000.0)]
    gen = np.r_[np.full(n // 2, 400.0), np.full(n - n // 2, 200.0)]
    obs = month_of_hours(gen, cap)

    monthly = country_cf_to_monthly(obs)
    truth = gen.sum() / cap.sum()

    assert monthly["obs"].iloc[0] == pytest.approx(truth)
    assert obs["capacity_factor"].mean() != pytest.approx(truth)


def test_constant_capacity_leaves_the_two_definitions_equal():
    n = 31 * 24
    rng = np.random.default_rng(0)
    gen = rng.uniform(0, 900, n)
    obs = month_of_hours(gen, np.full(n, 1000.0))
    assert country_cf_to_monthly(obs)["obs"].iloc[0] == pytest.approx(
        obs["capacity_factor"].mean()
    )


def test_falls_back_to_the_ratio_when_generation_is_absent():
    n = 31 * 24
    obs = month_of_hours(np.full(n, 250.0), np.full(n, 1000.0))[["capacity_factor"]]
    assert country_cf_to_monthly(obs)["obs"].iloc[0] == pytest.approx(0.25)


def test_a_gap_deflates_neither_side():
    """A row missing generation must drop out of the denominator too, or the
    month reads as if the fleet produced nothing during the gap."""
    n = 31 * 24
    gen = np.full(n, 250.0)
    gen[:24] = np.nan
    obs = month_of_hours(gen, np.full(n, 1000.0))
    assert country_cf_to_monthly(obs)["obs"].iloc[0] == pytest.approx(0.25)


def test_mixed_offset_index_is_handled():
    n = 31 * 24
    obs = month_of_hours(np.full(n, 250.0), np.full(n, 1000.0))
    obs.index = pd.Index(obs.index.astype(str), dtype=object)
    out = country_cf_to_monthly(obs)
    assert out["obs"].iloc[0] == pytest.approx(0.25)
    assert out["year"].iloc[0] == 2015


# ---------------------------------------------------------------------------
# Cluster resolution
# ---------------------------------------------------------------------------

def test_one_cluster_collapses_the_country(grid):
    out = assign_country_clusters(grid, 1)
    assert out["cluster"].nunique() == 1


def test_matching_count_keeps_the_grids_own_clusters(grid):
    out = assign_country_clusters(grid, 2)
    assert out["cluster"].tolist() == [0, 0, 1, 1]


def test_a_mismatched_count_is_refused(grid):
    """The old behaviour silently ignored num_clu, so every ENTSO-E config
    asking for 5 clusters produced a factors_*_5.csv holding however many the
    grid happened to define."""
    with pytest.raises(ValueError, match="asked for 5 clusters"):
        assign_country_clusters(grid, 5)


def test_the_error_names_both_usable_values(grid):
    with pytest.raises(ValueError) as excinfo:
        assign_country_clusters(grid, 5)
    message = str(excinfo.value)
    assert "[1]" in message and "[2]" in message


def test_metadata_without_clusters_is_refused(grid):
    with pytest.raises(ValueError, match="'cluster' column"):
        assign_country_clusters(grid.drop(columns=["cluster"]), 2)


# ---------------------------------------------------------------------------
# Weighted cluster means
# ---------------------------------------------------------------------------

def paired(grid, sims, obs=0.22):
    """The frame _country_cluster_means sees: simulations already joined to
    their cluster and capacity, with the one national observation broadcast."""
    return pd.DataFrame(
        {
            "year": 2015,
            "fixed": "1/1",
            "ID": grid["ID"],
            "sim": sims,
            "obs": obs,
            "cluster": grid["cluster"],
            "capacity": grid["capacity"],
        }
    )


def test_cluster_mean_is_capacity_weighted(grid):
    frame = paired(grid, [0.10, 0.30, 0.20, 0.20])
    out = _country_cluster_means(frame, "fixed")
    # Cluster 0 is 100 MW at 0.10 and 300 MW at 0.30.
    assert out.loc[out["cluster"] == 0, "sim"].iloc[0] == pytest.approx(0.25)
    assert out.loc[out["cluster"] == 1, "sim"].iloc[0] == pytest.approx(0.20)


def test_uniform_capacity_matches_an_unweighted_mean(grid):
    grid["capacity"] = 3.0
    frame = paired(grid, [0.10, 0.30, 0.20, 0.40])
    out = _country_cluster_means(frame, "fixed")
    assert out.loc[out["cluster"] == 0, "sim"].iloc[0] == pytest.approx(0.20)


def test_a_cluster_with_no_fleet_is_dropped(grid):
    """Year-specific weights routinely zero out a cluster. Falling back to an
    equal-weight mean there would fit a correction for a region that has no
    turbines in that year."""
    grid.loc[grid["cluster"] == 1, "capacity"] = 0.0
    out = _country_cluster_means(paired(grid, [0.1, 0.3, 0.2, 0.2]), "fixed")
    assert out["cluster"].tolist() == [0]


def test_a_grid_with_no_capacities_at_all_still_works(grid):
    frame = paired(grid, [0.10, 0.30, 0.20, 0.40]).drop(columns=["capacity"])
    out = _country_cluster_means(frame, "fixed")
    assert out.loc[out["cluster"] == 0, "sim"].iloc[0] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# End to end through cluster_train_set
# ---------------------------------------------------------------------------

def test_single_cluster_fit_has_one_row_per_period(grid):
    # cluster_train_set joins the cluster and capacity itself, so it receives
    # only the simulation paired with the national observation.
    frame = paired(grid, [0.10, 0.30, 0.20, 0.20]).drop(columns=["cluster", "capacity"])
    factors, clus_info = cluster_train_set(frame, "fixed", 1, grid, obs_level="country")
    assert len(factors) == 1
    assert clus_info["cluster"].nunique() == 1
    # One observation against one offset: exactly determined, unlike the
    # N-offsets-against-one-number joint fit. The scalar is the national
    # observation over the capacity-weighted national simulation.
    national_sim = np.average([0.10, 0.30, 0.20, 0.20], weights=grid["capacity"])
    assert factors["scalar"].iloc[0] == pytest.approx(0.22 / national_sim)
