"""Skill metrics and pseudo-replicate handling (docs/HARNESS_DESIGN.md §5, §2)."""
import numpy as np
import pandas as pd
import pytest

from vwf.harness.regions import RegionSpec
from vwf.harness.skill import (
    collapse_pseudo_replicates,
    seasonal_cycle_rmse,
    skill_metrics,
    station_ids,
)


def make_spec(**overrides) -> RegionSpec:
    base = dict(
        code="ZZ",
        name="Testland",
        source="test-source",
        obs_level="turbine",
        obs_unit="farm",
        train_years=(2015, 2018),
        test_years=(2019,),
        era5_path="era5/ZZ",
        bbox=(0.0, 10.0, 40.0, 50.0),
        file_tag="ZZ",
        correction_model="affine-wind",
        cluster_list=(5,),
        time_slices=("fixed",),
        seasons={
            "winter": (12, 1, 2),
            "spring": (3, 4, 5),
            "summer": (6, 7, 8),
            "autumn": (9, 10, 11),
        },
    )
    base.update(overrides)
    return RegionSpec(**base)


def test_skill_metrics_known_answers():
    df = pd.DataFrame(
        {
            "ID": ["a", "b", "c", "d"],
            "cf_obs": [0.2, 0.3, 0.4, 0.5],
            "cf_sim": [0.3, 0.4, 0.5, 0.6],  # uniform +0.1 bias
            "capacity": [1.0, 1.0, 1.0, 1.0],
        }
    )
    m = skill_metrics(df)
    assert m["mbe"] == pytest.approx(0.1)
    assert m["mae"] == pytest.approx(0.1)
    assert m["rmse"] == pytest.approx(0.1)
    assert m["pearson_r"] == pytest.approx(1.0)
    assert m["emd"] == pytest.approx(0.1)
    assert m["n_units"] == 4
    assert m["n_samples"] == 4


def test_skill_metrics_capacity_weighting_distinguishes():
    # One big unit with +0.1 bias, one tiny unit with -0.1 bias: the weighted
    # and unweighted MBE must differ, and the weighted one must sit near the
    # big unit's bias. (Must-distinguish, not just pass-on-correct.)
    df = pd.DataFrame(
        {
            "ID": ["big", "small"],
            "cf_obs": [0.3, 0.3],
            "cf_sim": [0.4, 0.2],
            "capacity": [99.0, 1.0],
        }
    )
    weighted = skill_metrics(df, weighted=True)
    unweighted = skill_metrics(df, weighted=False)
    assert unweighted["mbe"] == pytest.approx(0.0)
    assert weighted["mbe"] == pytest.approx(0.098)
    assert weighted["mbe"] != unweighted["mbe"]


def test_skill_metrics_drops_nan_and_raises_on_empty():
    df = pd.DataFrame(
        {
            "ID": ["a", "b"],
            "cf_obs": [0.2, np.nan],
            "cf_sim": [0.25, 0.3],
            "capacity": [1.0, 1.0],
        }
    )
    assert skill_metrics(df)["n_samples"] == 1
    with pytest.raises(ValueError, match="no complete"):
        skill_metrics(df[df["ID"] == "b"])


def test_skill_metrics_requires_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        skill_metrics(pd.DataFrame({"ID": ["a"], "cf_sim": [0.1]}))


def test_station_ids_regex_and_fallback():
    spec = make_spec(pseudo_replicated_rows=True, station_id_regex=r"^(.+)-\d+$")
    ids = pd.Series(["R0001SQSC-1", "R0001SQSC-12", "NOSUFFIX"])
    stations = station_ids(ids, spec)
    assert list(stations) == ["R0001SQSC", "R0001SQSC", "NOSUFFIX"]
    # No spec / no regex → identity.
    assert list(station_ids(ids)) == list(ids)


def test_collapse_pseudo_replicates_uk_shape():
    """Three turbine rows of one station (identical obs, split capacity) plus
    an independent single-unit station: collapse must yield 2 independent
    units, sum capacity, and capacity-weight the simulated CF."""
    spec = make_spec(pseudo_replicated_rows=True, station_id_regex=r"^(.+)-\d+$")
    df = pd.DataFrame(
        {
            "ID": ["S1-1", "S1-2", "S1-3", "S2-1"],
            "year": [2019] * 4,
            "month": [1] * 4,
            "cf_obs": [0.30, 0.30, 0.30, 0.50],  # S1 rows share the farm obs
            "cf_sim": [0.20, 0.40, 0.40, 0.55],
            "capacity": [2.0, 1.0, 1.0, 3.0],
        }
    )
    collapsed = collapse_pseudo_replicates(df, spec)
    assert len(collapsed) == 2
    s1 = collapsed[collapsed["ID"] == "S1"].iloc[0]
    assert s1["capacity"] == pytest.approx(4.0)
    assert s1["cf_obs"] == pytest.approx(0.30)
    # capacity-weighted sim: (0.2*2 + 0.4*1 + 0.4*1) / 4 = 0.3
    assert s1["cf_sim"] == pytest.approx(0.30)
    # ...whereas an unweighted mean would give 1/3·(0.2+0.4+0.4) ≈ 0.333:
    # the weighting must be doing something on this fixture.
    assert s1["cf_sim"] != pytest.approx((0.20 + 0.40 + 0.40) / 3)

    metrics = skill_metrics(collapsed)
    assert metrics["n_units"] == 2


def test_collapse_refuses_divergent_obs_within_station():
    """Identity guard: rows of one station-time group with DIFFERENT obs are
    not pseudo-replicates; collapsing them would fabricate an observation."""
    spec = make_spec(pseudo_replicated_rows=True, station_id_regex=r"^(.+)-\d+$")
    df = pd.DataFrame(
        {
            "ID": ["S1-1", "S1-2"],
            "year": [2019, 2019],
            "month": [1, 1],
            "cf_obs": [0.30, 0.31],  # divergent: NOT an equal split
            "cf_sim": [0.20, 0.40],
            "capacity": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="divergent cf_obs"):
        collapse_pseudo_replicates(df, spec)


def test_collapse_is_noop_without_pseudo_replication():
    spec = make_spec()  # pseudo_replicated_rows=False
    df = pd.DataFrame(
        {
            "ID": ["a-1", "a-2"],
            "cf_obs": [0.2, 0.4],
            "cf_sim": [0.3, 0.5],
            "capacity": [1.0, 2.0],
        }
    )
    out = collapse_pseudo_replicates(df, spec)
    pd.testing.assert_frame_equal(out, df)
    assert out is not df  # a copy, not the caller's frame


def test_seasonal_cycle_rmse_known_answer():
    months = list(range(1, 13))
    df = pd.DataFrame(
        {
            "ID": ["a"] * 12,
            "month": months,
            "cf_obs": [0.3] * 12,
            "cf_sim": [0.35] * 12,  # constant +0.05 climatology error
            "capacity": [1.0] * 12,
        }
    )
    assert seasonal_cycle_rmse(df) == pytest.approx(0.05)


def test_seasonal_cycle_rmse_derives_month_from_time():
    times = pd.date_range("2019-01-01", periods=12, freq="MS")
    df = pd.DataFrame(
        {
            "ID": ["a"] * 12,
            "time": times,
            "cf_obs": [0.3] * 12,
            "cf_sim": [0.35] * 12,
            "capacity": [1.0] * 12,
        }
    )
    assert seasonal_cycle_rmse(df) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="month"):
        seasonal_cycle_rmse(df.drop(columns=["time"]))
