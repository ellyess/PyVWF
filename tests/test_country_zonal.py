"""Per-zone country observations: RQ4 option 1.

With one national observation, N cluster offsets are under-determined by N-1
and `find_offsets_country_level` returns wherever L-BFGS-B stopped. With one
observation per cluster the fit is exactly determined and takes the same solver
the turbine-level path uses. These tests pin the routing between the two, since
picking the wrong one is silent: both produce a plausible factors table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vwf.data import (
    country_obs_is_per_cluster,
    country_zonal_cf_to_monthly,
    country_zonal_to_national,
)
from vwf.sources import available_sources
from vwf.sources.entsoe_zonal import EntsoeZonalFileSource


@pytest.fixture
def zonal_layout(tmp_path):
    """Two-zone country 'ZZ' with per-zone observation files."""
    base = tmp_path / "observations/country"
    (base / "grid_points" / "zz").mkdir(parents=True)

    pd.DataFrame(
        {
            "ID": ["g0", "g1"],
            "lon": [5.0, 6.0],
            "lat": [52.0, 53.0],
            "height": [100.0, 100.0],
            "capacity": [1000.0, 3000.0],
            "model": ["M", "M"],
            "cluster": [0, 1],
        }
    ).to_csv(base / "grid_points" / "zz" / "zz_grid_points.csv", index=False)

    idx = pd.date_range("2015-01-01", periods=8, freq="h", tz="UTC")
    for zone, cf in ((1, 0.20), (2, 0.30)):
        directory = base / "observations" / f"zz_{zone}"
        directory.mkdir(parents=True)
        frame = pd.DataFrame({"capacity_factor": [cf] * 7 + [0.9]}, index=idx)
        frame.to_csv(directory / f"zz_{zone}_train_2015_2019.csv")
        frame.to_csv(directory / f"zz_{zone}_test_2023.csv")
    return base


def source(layout, split="train"):
    return EntsoeZonalFileSource("ZZ", split, (2015, 2019), 2023, cl_data_dir=layout)


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

def test_registered():
    assert "entsoe-zonal" in available_sources()


def test_every_zone_is_loaded_and_tagged_with_its_cluster(zonal_layout):
    obs = source(zonal_layout).load_observations()
    # generate_country_level_training_data numbers clusters as zone - 1; if the
    # two partitions disagree the merge in train_set drops every row.
    assert sorted(obs["cluster"].unique()) == [0, 1]
    assert obs.loc[obs["cluster"] == 0, "capacity_factor"].iloc[0] == pytest.approx(0.20)
    assert obs.loc[obs["cluster"] == 1, "capacity_factor"].iloc[0] == pytest.approx(0.30)


def test_a_missing_zone_is_refused_not_silently_dropped(zonal_layout):
    """A partial zone set shifts the national aggregate the metrics are scored
    against, so it must not pass quietly."""
    (zonal_layout / "observations" / "zz_2" / "zz_2_train_2015_2019.csv").unlink()
    with pytest.raises(FileNotFoundError, match="missing zones"):
        source(zonal_layout).load_observations()


def test_no_zone_directories_at_all_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="per-zone observation directories"):
        source(tmp_path / "empty").load_observations()


def test_grid_point_loading_is_inherited(zonal_layout):
    assert source(zonal_layout).load_metadata()["cluster"].tolist() == [0, 1]


def test_aggregated_suffix_works_for_zones_too(zonal_layout):
    directory = zonal_layout / "observations" / "zz_1"
    plain = directory / "zz_1_train_2015_2019.csv"
    plain.rename(directory / "zz_1_train_2015_2019_aggregated.csv")
    assert len(source(zonal_layout).load_observations()) == 16


# ---------------------------------------------------------------------------
# Routing between the determined and under-determined fits
# ---------------------------------------------------------------------------

def bias_frame(obs_by_cluster):
    return pd.DataFrame(
        {
            "year": 2015,
            "fixed": "1/1",
            "cluster": list(obs_by_cluster),
            "obs": list(obs_by_cluster.values()),
            "sim": 0.3,
        }
    )


def test_distinct_per_cluster_observations_are_detected():
    assert country_obs_is_per_cluster(bias_frame({0: 0.20, 1: 0.30}), "fixed")


def test_one_national_observation_is_not_per_cluster():
    assert not country_obs_is_per_cluster(bias_frame({0: 0.22, 1: 0.22}), "fixed")


def test_a_frame_without_clusters_is_not_per_cluster():
    frame = bias_frame({0: 0.2, 1: 0.3}).drop(columns=["cluster"])
    assert not country_obs_is_per_cluster(frame, "fixed")


# ---------------------------------------------------------------------------
# Aggregation back to a comparable national series
# ---------------------------------------------------------------------------

def test_zonal_monthly_keeps_one_row_per_cluster(zonal_layout):
    monthly = country_zonal_cf_to_monthly(source(zonal_layout).load_observations())
    assert len(monthly) == 2
    assert set(monthly["cluster"]) == {0, 1}


def test_zonal_collapses_to_a_capacity_weighted_national_series(zonal_layout):
    """Evaluation scores the national aggregate, so a zonal run has to reduce
    to the same quantity or its metrics are not comparable with a national run's."""
    src = source(zonal_layout)
    national = country_zonal_to_national(src.load_observations(), src.load_metadata())
    # 1000 MW at 0.20 and 3000 MW at 0.30.
    assert national["capacity_factor"].iloc[0] == pytest.approx(0.275)
    assert "cluster" not in national.columns


def test_national_collapse_ignores_a_zone_with_no_capacity(zonal_layout):
    src = source(zonal_layout)
    grid = src.load_metadata()
    grid.loc[grid["cluster"] == 0, "capacity"] = 0.0
    national = country_zonal_to_national(src.load_observations(), grid)
    assert national["capacity_factor"].iloc[0] == pytest.approx(0.30)


def test_national_collapse_survives_a_missing_observation(zonal_layout):
    src = source(zonal_layout)
    obs = src.load_observations()
    obs.loc[obs["cluster"] == 0, "capacity_factor"] = np.nan
    national = country_zonal_to_national(obs, src.load_metadata())
    assert national["capacity_factor"].iloc[0] == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# The fit actually takes the other solver
# ---------------------------------------------------------------------------

@pytest.fixture
def country_fixture(tmp_path, monkeypatch):
    """Synthetic ERA5 plus a four-point, two-cluster grid."""
    from vwf.config import PyVWFPaths
    from tests import test_pipeline as tp

    tp._write_era5(tmp_path / "era5")
    monkeypatch.setattr(PyVWFPaths, "INPUT_ROOT", tmp_path)
    monkeypatch.setattr(PyVWFPaths, "ERA5_DATA", tmp_path / "era5")

    grid = pd.DataFrame(
        {
            "ID": ["g0", "g1", "g2", "g3"],
            "lon": [8.1, 8.3, 9.2, 9.4],
            "lat": [55.2, 55.4, 55.6, 55.8],
            "height": [100.0] * 4,
            "capacity": [2000.0] * 4,
            "model": ["2019COE_Market_Average_2.6MW_121"] * 4,
            "cluster": [0, 0, 1, 1],
            "type": ["onshore"] * 4,
        }
    )
    times = pd.date_range("2015-01-01", "2015-12-31", freq="D", tz="UTC")
    return grid, times


def fit_with(grid, obs, monkeypatch, forbid):
    """Fit the country path, with one of the two solvers wired to explode."""
    import vwf.correction as correction
    from vwf.data import train_set
    from vwf.harness.corrections import get_correction
    from vwf.sources.in_memory import InMemoryCountrySource

    def boom(*args, **kwargs):
        raise AssertionError(f"{forbid} must not be called for this observation shape")

    monkeypatch.setattr(correction, forbid, boom)

    gen_cf, turb_info, reanalysis, power_curves = train_set(
        "ZZ",
        calc_z0=True,
        mode="all",
        obs_level="country",
        source=InMemoryCountrySource(grid, obs),
    )
    return get_correction("affine-wind").fit(
        gen_cf,
        turb_info,
        reanalysis,
        power_curves,
        num_clusters=2,
        time_res="fixed",
        obs_level="country",
    )


def test_per_cluster_observations_do_not_use_the_joint_optimiser(
    country_fixture, monkeypatch
):
    """The whole point of a zonal source: each cluster's offset is determined by
    its own constraint, so the under-determined joint fit must not run."""
    grid, times = country_fixture
    obs = pd.concat(
        [
            pd.DataFrame({"capacity_factor": 0.18, "cluster": 0}, index=times),
            pd.DataFrame({"capacity_factor": 0.28, "cluster": 1}, index=times),
        ]
    )
    factors, _ = fit_with(grid, obs, monkeypatch, "find_offsets_country_level")
    assert sorted(factors["cluster"]) == [0, 1]
    # Distinct observations must give distinct scalars, or the zonal detail was
    # averaged away somewhere on the way in.
    assert factors["scalar"].nunique() == 2


def test_one_national_observation_still_uses_the_joint_optimiser(
    country_fixture, monkeypatch
):
    grid, times = country_fixture
    obs = pd.DataFrame({"capacity_factor": 0.22}, index=times)
    factors, _ = fit_with(grid, obs, monkeypatch, "find_offset")
    assert sorted(factors["cluster"]) == [0, 1]
