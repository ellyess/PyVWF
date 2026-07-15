"""Driver end-to-end on synthetic data: train, evaluate, country-level fit."""
import json

import pandas as pd
import pytest

import test_pipeline as tp
from vwf.config import PyVWFPaths
from vwf.data import train_set
from vwf.harness import get_correction
from vwf.harness.driver import run_evaluate, run_train
from vwf.harness.regions import RegionSpec
from vwf.sources import InMemoryCountrySource

NH = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}


def make_spec(**overrides) -> RegionSpec:
    base = dict(
        code="DK",
        name="Denmark (synthetic)",
        source="european-turbine",
        obs_level="turbine",
        obs_unit="turbine",
        train_years=(2015, 2015),
        test_years=(2016,),
        era5_path="era5",
        bbox=(7.5, 13.5, 54.0, 58.2),
        file_tag="SYN",
        correction_model="affine-wind",
        cluster_list=(2,),
        time_slices=("fixed",),
        seasons=NH,
    )
    base.update(overrides)
    return RegionSpec(**base)


@pytest.fixture
def synthetic_dk(tmp_path, monkeypatch):
    tp._write_era5(tmp_path / "era5")
    fleet = tp._write_fleet(tmp_path / "turbine_level_data" / "DK")
    monkeypatch.setattr(PyVWFPaths, "INPUT_ROOT", tmp_path)
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path / "turbine_level_data")
    monkeypatch.setattr(PyVWFPaths, "ERA5_DATA", tmp_path / "era5")
    return {"root": tmp_path, "fleet": fleet}


def test_driver_train_then_evaluate(synthetic_dk):
    """The full harness loop: train writes factors + fleet + manifest;
    evaluate writes metrics with the correction beating the uncorrected
    baseline on the planted-bias fixture."""
    spec = make_spec()
    out = synthetic_dk["root"] / "validation"

    train_dir = run_train(spec, out, mode="onshore", run_name="t1")
    assert (train_dir / "factors_fixed_2.csv").is_file()
    assert (train_dir / "train_turb_info_2.csv").is_file()

    manifest = json.loads((train_dir / "run_manifest.json").read_text())
    assert manifest["run_mode"] == "train"
    assert manifest["observations"]["obs_unit"] == "turbine"
    assert manifest["curve_library"]["library"] == "synthetic-bundled"

    factors = pd.read_csv(train_dir / "factors_fixed_2.csv")
    assert (factors["scalar"] < 1.0).all()  # the planted over-prediction

    eval_dir = run_evaluate(spec, train_dir, out, mode="onshore", run_name="e1")
    metrics = pd.read_csv(eval_dir / "metrics.csv")
    assert set(metrics["variant"]) == {"uncorrected", "affine-wind"}

    unc = metrics[metrics["variant"] == "uncorrected"].iloc[0]
    cor = metrics[metrics["variant"] == "affine-wind"].iloc[0]
    assert cor["rmse"] < unc["rmse"]
    assert abs(cor["mbe"]) < abs(unc["mbe"])
    assert unc["n_units"] == len(synthetic_dk["fleet"])

    eval_manifest = json.loads((eval_dir / "run_manifest.json").read_text())
    assert eval_manifest["run_mode"] == "evaluate"
    assert eval_manifest["evaluation_year"] == 2016


def test_country_level_fit_is_a_delegation_wrapper(synthetic_dk):
    """AffineWindCorrection.fit(obs_level='country') delegates to the legacy
    joint-offset optimiser and returns a factors table of the same shape as
    the turbine path."""
    grid_points = pd.DataFrame(
        {
            "ID": ["g0", "g1", "g2", "g3"],
            "lon": [8.1, 8.3, 9.2, 9.4],
            "lat": [55.2, 55.4, 55.6, 55.8],
            "height": [100.0] * 4,
            "capacity": [2000.0] * 4,
            "cluster": [0, 0, 1, 1],
            "type": ["onshore"] * 4,
        }
    )
    times = pd.date_range("2015-01-01", "2015-12-31", freq="D", tz="UTC")
    obs = pd.DataFrame({"capacity_factor": 0.2}, index=times)

    source = InMemoryCountrySource(grid_points, obs)
    gen_cf, turb_info, reanalysis, power_curves = train_set(
        "ZZ", calc_z0=True, mode="all", obs_level="country", source=source
    )

    model = get_correction("affine-wind")
    factors, clus_info = model.fit(
        gen_cf,
        turb_info,
        reanalysis,
        power_curves,
        num_clusters=2,
        time_res="fixed",
        obs_level="country",
    )

    assert set(factors.columns) == {"cluster", "fixed", "scalar", "offset"}
    assert sorted(factors["cluster"]) == [0, 1]
    # Planted bias: synthetic winds over-predict a 0.2 CF, so the fitted
    # correction must pull the simulation down.
    assert (factors["scalar"] < 1.0).all()
    assert factors["offset"].notna().all()
