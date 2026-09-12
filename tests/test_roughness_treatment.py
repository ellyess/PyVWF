"""Which temporal treatment of the roughness a run applies, and what it records.

Every region derives the roughness length from the 10 m to 100 m shear, but the
European files carry one annual mean of it per year while every other region
derives it per timestep. Which is better is under test
(``docs/findings/method-roughness-treatment-prereg.md``); until that reports,
``stored`` is the default because it is what every existing run did. These
tests pin the switch, the fallback when no stored field exists, and the record
of what was actually applied.
"""
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from test_harness_driver import make_spec, synthetic_dk  # noqa: F401  (fixture)
from vwf.datasets.era5 import prep_era5
from vwf.harness.driver import run_train
from vwf.harness.regions import load_region


def _combined_file(path, with_z0=True, with_10m=True):
    """A file shaped like the European combined ones: hourly winds, static z0."""
    path.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2015-01-01", periods=48, freq="h")
    lat, lon = np.array([55.0, 55.5]), np.array([8.0, 8.5])
    shape = (len(times), len(lat), len(lon))
    data = {
        "u100": (("time", "lat", "lon"), np.full(shape, 7.0)),
        "v100": (("time", "lat", "lon"), np.zeros(shape)),
    }
    if with_10m:
        # A 10 m speed below the 100 m one, so the shear inversion is defined.
        data["u10"] = (("time", "lat", "lon"), np.full(shape, 5.0))
        data["v10"] = (("time", "lat", "lon"), np.zeros(shape))
    if with_z0:
        data["z0"] = (("lat", "lon"), np.full((len(lat), len(lon)), 0.25))
    xr.Dataset(data, coords={"time": times, "lat": lat, "lon": lon}).to_netcdf(
        path / "era5_combined_2015_TEST.nc"
    )
    return path


def test_stored_is_the_default_and_is_static(tmp_path):
    ds = prep_era5("ZZ", False, True, era5_dir=_combined_file(tmp_path / "e"))
    assert ds.attrs["pyvwf_roughness_treatment"] == "stored"
    # The stored field is one value per cell for the whole year. The daily
    # resample broadcasts it over time, so the test is that it does not vary.
    assert float(ds["roughness"].std("time").max()) == pytest.approx(0.0)
    assert float(ds["roughness"].max()) == pytest.approx(0.25)


def test_derived_ignores_the_stored_field_and_varies_in_time(tmp_path):
    ds = prep_era5("ZZ", False, True, era5_dir=_combined_file(tmp_path / "e"),
                   roughness="derived")
    assert ds.attrs["pyvwf_roughness_treatment"] == "derived"
    # z0 = exp((w100 ln10 - w10 ln100) / (w100 - w10)), with w100 = 7, w10 = 5.
    expected = np.exp((7 * np.log(10) - 5 * np.log(100)) / (7 - 5))
    assert float(ds["roughness"].mean()) == pytest.approx(expected, rel=1e-6)
    assert float(ds["roughness"].max()) != pytest.approx(0.25)


def test_a_file_with_no_stored_field_is_derived_either_way(tmp_path):
    era5 = _combined_file(tmp_path / "e", with_z0=False)
    for asked in ("stored", "derived"):
        ds = prep_era5("ZZ", False, True, era5_dir=era5, roughness=asked)
        assert ds.attrs["pyvwf_roughness_treatment"] == "derived"


def test_derived_without_the_10m_winds_says_what_is_missing(tmp_path):
    era5 = _combined_file(tmp_path / "e", with_10m=False)
    with pytest.raises(ValueError, match="10 m wind components"):
        prep_era5("ZZ", False, True, era5_dir=era5, roughness="derived")


def test_an_unknown_treatment_is_refused(tmp_path):
    with pytest.raises(ValueError, match="roughness must be one of"):
        prep_era5("ZZ", False, True, era5_dir=_combined_file(tmp_path / "e"),
                  roughness="annual-mean")


def test_region_config_parses_the_treatment(tmp_path):
    base = open("configs/regions/scorecard/dk_k100.toml").read()
    (tmp_path / "derived.toml").write_text(
        base.replace("[era5]\n", '[era5]\nroughness = "derived"\n', 1))
    (tmp_path / "bad.toml").write_text(
        base.replace("[era5]\n", '[era5]\nroughness = "annual"\n', 1))
    assert load_region("configs/regions/scorecard/dk_k100.toml").roughness == "stored"
    assert load_region(tmp_path / "derived.toml").roughness == "derived"
    with pytest.raises(ValueError, match="roughness"):
        load_region(tmp_path / "bad.toml")


def test_a_run_records_what_it_asked_for_and_what_it_applied(synthetic_dk):  # noqa: F811
    """The synthetic ERA5 carries no stored field, so a run that asks for one
    is derived, and the record has to show both."""
    spec = make_spec()
    train_dir = run_train(spec, synthetic_dk["root"] / "validation", mode="onshore", run_name="t")
    record = json.loads((train_dir / "run_manifest.json").read_text())["era5_roughness"]
    assert record == {"requested": "stored", "applied": "derived"}
