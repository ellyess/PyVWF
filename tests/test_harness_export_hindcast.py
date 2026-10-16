"""The gridded export and the hindcast, on the synthetic Denmark fixture (#86).

``pyvwf.harness.export`` writes the correction field other pipelines consume,
and ``pyvwf.harness.hindcast`` places a period in the record; until 2026-10-16
no test imported either, so a regression in them would have been silent. Each
case trains the synthetic fleet with the real harness and checks a property a
consumer relies on.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from test_harness_driver import make_spec
from test_pipeline import TRUE_CF, YEARS

from pyvwf.harness.driver import run_train
from pyvwf.harness.export import export_correction_field
from pyvwf.harness.hindcast import _national_monthly, rank_in_context, run_hindcast


@pytest.fixture
def trained(synthetic_dk):
    spec = make_spec(time_slices=("fixed", "season"))
    root = synthetic_dk["root"]
    train_dir = run_train(spec, root / "validation", mode="onshore", run_name="t")
    return spec, train_dir, root / "era5"


def _export(trained, tmp_path, time_res="season"):
    spec, train_dir, era5 = trained
    path = export_correction_field(
        spec, train_dir, num_clu=2, time_res=time_res, out_path=tmp_path / "f.nc", era5_dir=era5
    )
    return xr.open_dataset(path), pd.read_csv(train_dir / f"factors_{time_res}_2.csv")


def test_each_cell_carries_its_clusters_factors_in_season_order(trained, tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        field, factors = _export(trained, tmp_path)
    # A clean fit carries no degenerate-cluster warning (the synthetic ERA5's
    # short extent warns separately, and that is the fixture, not the export).
    assert not [w for w in caught if "implausible scalar" in str(w.message)]
    assert list(field["slice"].values) == ["winter", "spring", "summer", "autumn"]
    for label in field["slice"].values:
        sub = factors[factors["season"] == label].set_index("cluster")
        for cluster in np.unique(field["cluster"].values):
            cells = field["cluster"].values == cluster
            got = field["scalar"].sel(slice=label).values[cells]
            assert np.allclose(got, sub.loc[cluster, "scalar"], rtol=1e-6)
            got = field["offset"].sel(slice=label).values[cells]
            assert np.allclose(got, sub.loc[cluster, "offset"], rtol=1e-6, atol=1e-6)
    assert int(field["degenerate"].values.sum()) == 0


def test_each_cell_is_assigned_its_nearest_centroid(trained, tmp_path):
    spec, train_dir, _ = trained
    field, _ = _export(trained, tmp_path, "fixed")
    fleet = pd.read_csv(train_dir / "train_turb_info_2.csv")
    centroids = fleet.groupby("cluster")[["lon", "lat"]].mean()
    lon2d, lat2d = np.meshgrid(field["lon"].values, field["lat"].values)
    coslat = np.cos(np.radians(lat2d))
    km = {
        c: np.hypot((lon2d - row.lon) * 111.0 * coslat, (lat2d - row.lat) * 111.0)
        for c, row in centroids.iterrows()
    }
    nearest = np.where(km[0] <= km[1], 0, 1)
    assert (field["cluster"].values == centroids.index.values[nearest]).all()
    assert np.allclose(field["km_to_centroid"].values, np.minimum(km[0], km[1]), rtol=1e-5)
    assert set(np.unique(field["cluster_n_train"].values)) <= set(fleet.groupby("cluster").size())


def test_a_degenerate_cluster_is_flagged_and_warned_about(trained, tmp_path):
    """Known positive: a scalar of 80 lands in the 'degenerate' layer, not silently."""
    spec, train_dir, _ = trained
    path = train_dir / "factors_fixed_2.csv"
    factors = pd.read_csv(path)
    factors.loc[factors["cluster"] == 0, "scalar"] = 80.0
    factors.to_csv(path, index=False)
    with pytest.warns(UserWarning, match="implausible scalar"):
        field, _ = _export(trained, tmp_path, "fixed")
    flagged = field["degenerate"].values == 1
    assert flagged.any()
    assert (field["cluster"].values[flagged] == 0).all()
    assert not (field["cluster"].values[~flagged] == 0).any()


def test_a_missing_factor_is_the_identity(trained, tmp_path):
    spec, train_dir, _ = trained
    path = train_dir / "factors_fixed_2.csv"
    factors = pd.read_csv(path)
    factors.loc[factors["cluster"] == 1, ["scalar", "offset"]] = np.nan
    factors.to_csv(path, index=False)
    field, _ = _export(trained, tmp_path, "fixed")
    cells = field["cluster"].values == 1
    assert (field["scalar"].values[0][cells] == 1.0).all()
    assert (field["offset"].values[0][cells] == 0.0).all()


def test_the_hindcast_covers_the_archive_and_the_correction_moves_toward_truth(trained):
    spec, train_dir, era5 = trained
    monthly = run_hindcast(
        spec,
        train_dir,
        num_clu=2,
        time_res="fixed",
        fleet_year=YEARS[-1],
        mode="onshore",
        era5_dir=era5,
    )
    assert list(monthly.columns) == ["year", "month", "cf_uncorrected", "cf_corrected"]
    assert sorted(monthly["year"].unique()) == list(YEARS)
    assert len(monthly) == 12 * len(YEARS)
    # The fixture plants a bias against a true capacity factor of TRUE_CF.
    unc = (monthly["cf_uncorrected"] - TRUE_CF).abs().mean()
    cor = (monthly["cf_corrected"] - TRUE_CF).abs().mean()
    assert cor < unc

    ranked = rank_in_context(monthly)
    assert ranked["pct"].between(0, 1, inclusive="right").all()
    assert (ranked["n_years"] == len(YEARS)).all()
    assert np.allclose(ranked.groupby("month")["anomaly"].sum(), 0.0, atol=1e-12)


def test_the_national_mean_leaves_a_missing_point_out_of_both_sums():
    cf = pd.DataFrame(
        {
            "time": pd.to_datetime(["2016-01-01", "2016-01-02"]),
            "a": [0.2, np.nan],
            "b": [0.4, 0.4],
        }
    )
    fleet = pd.DataFrame({"ID": ["a", "b"], "capacity": [1.0, 3.0]})
    out = _national_monthly(cf, fleet)
    # Day one (0.2 + 1.2) / 4 = 0.35; day two only b counts: 0.4. Month mean 0.375.
    assert out.iloc[0] == pytest.approx(0.375)
