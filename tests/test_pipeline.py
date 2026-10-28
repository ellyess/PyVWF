"""The data pipeline on a synthetic fleet, and the fixture the harness tests share.

These drive the real, unmocked data functions the harness delegates to:

    ERA5-shaped winds -> hub-height extrapolation -> power curve -> paired
    simulated and observed capacity factors -> per-cluster scalars

The only thing synthetic is the *data*: a small fleet with a deliberately
planted bias (the reanalysis blows harder than the turbines actually generate),
so we know which way a correct correction must move. The full train and
evaluate loop runs through the harness in ``test_harness_driver.py``.

The fixture writes the on-disk layout the loaders expect, rather than
monkeypatching the loaders, so the schema contract is exercised too: if the
expected filename or column set changes, these fail. Until 2026-09-24 this file
also drove the legacy ``PyVWF`` class end to end; that class was removed.
"""

from __future__ import annotations


import pytest

from pyvwf.data import cluster_train_set, train_set


from support import TRUE_CF, YEARS
from support import DK_BBOX


# ---------------------------------------------------------------------------
# train_set / cluster_train_set: the data-preparation layer
# ---------------------------------------------------------------------------


def test_train_set_pairs_observations_with_simulations(synthetic_dk):
    gen_cf, turb_info, reanalysis, power_curves = train_set(
        "DK", calc_z0=True, mode="onshore", era5_dir=synthetic_dk["era5"], bbox=DK_BBOX
    )

    assert {"ID", "year", "month", "obs", "sim"} <= set(gen_cf.columns)
    assert len(turb_info) == len(synthetic_dk["fleet"])
    assert "model" in turb_info.columns  # add_models resolved a power curve
    assert turb_info["model"].isin(power_curves.columns).all()

    # Capacity factors must be physical, and the observations must come back as
    # the CF we planted (generation was written as TRUE_CF * hours * capacity).
    assert gen_cf["sim"].between(0.0, 1.0).all()
    assert gen_cf["obs"].to_numpy() == pytest.approx(TRUE_CF, abs=1e-6)

    # The planted bias: the reanalysis is windier than reality, so the
    # uncorrected simulation must over-predict.
    assert gen_cf["sim"].mean() > TRUE_CF

    # prep_era5 resamples to daily
    assert reanalysis.sizes["time"] == 731  # 2015 + 2016 (leap)


def test_cluster_train_set_fits_one_scalar_per_cluster(synthetic_dk):
    gen_cf, turb_info, _, _ = train_set(
        "DK", calc_z0=True, mode="onshore", era5_dir=synthetic_dk["era5"], bbox=DK_BBOX
    )
    bias_df, clus_info = cluster_train_set(gen_cf, "fixed", 2, turb_info)

    assert set(bias_df["cluster"].unique()) == {0, 1}
    assert "scalar" in bias_df.columns
    assert set(clus_info["cluster"].unique()) == {0, 1}

    # Simulation over-predicts, so the fitted scalar must pull it down.
    assert (bias_df["scalar"] < 1.0).all()


def test_cluster_train_set_respects_temporal_resolution(synthetic_dk):
    """`month` fits a correction per calendar month, `fixed` a single one.

    The bias table keeps the year dimension (one row per year x cluster x
    slice); the per-year rows are collapsed into a single factor per cluster
    when `train` writes the correction-factor file.
    """
    gen_cf, turb_info, _, _ = train_set(
        "DK", calc_z0=True, mode="onshore", era5_dir=synthetic_dk["era5"], bbox=DK_BBOX
    )
    n_years, n_clusters = len(YEARS), 2

    fixed, _ = cluster_train_set(gen_cf, "fixed", n_clusters, turb_info)
    monthly, _ = cluster_train_set(gen_cf, "month", n_clusters, turb_info)
    seasonal, _ = cluster_train_set(gen_cf, "season", n_clusters, turb_info)

    assert len(fixed) == n_clusters * n_years
    assert len(monthly) == n_clusters * n_years * 12
    assert len(seasonal) == n_clusters * n_years * 4

    assert set(fixed["time_slice"].unique()) == {"1/1"}
    assert set(int(m) for m in monthly["time_slice"].unique()) == set(range(1, 13))
    assert set(seasonal["time_slice"].unique()) == {"winter", "spring", "summer", "autumn"}
