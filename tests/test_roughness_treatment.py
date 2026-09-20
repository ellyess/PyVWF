"""Which temporal treatment of the roughness a run applies, and what it records.

Every region derives the roughness length from the 10 m to 100 m shear. The
question is whether the result varies in time, and the per-timestep derivation
is the method, adopted on 2026-09-12
(``docs/findings/method-roughness-treatment.md``). So ``derived`` is the
default, and ``stored`` has to be asked for: the ``era5/EU`` archive stores an
annual mean, the superseded treatment, and the daily pre-combined files store a
per-timestep roughness they also cannot re-derive. These tests pin the default,
the switch, the fallback when no stored field exists, and the record of what
was actually applied.
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


def test_derived_is_the_default_even_where_a_field_is_stored(tmp_path):
    """The file carries a static z0 of 0.25 and the default ignores it.

    This is the case the default change is about: before it, a file with a
    stored field silently applied the annual mean.
    """
    ds = prep_era5("ZZ", False, True, era5_dir=_combined_file(tmp_path / "e"))
    assert ds.attrs["pyvwf_roughness_treatment"] == "derived"
    assert float(ds["roughness"].max()) != pytest.approx(0.25)


def test_stored_is_used_when_it_is_asked_for_and_is_static(tmp_path):
    ds = prep_era5("ZZ", False, True, era5_dir=_combined_file(tmp_path / "e"), roughness="stored")
    assert ds.attrs["pyvwf_roughness_treatment"] == "stored"
    # The stored field is one value per cell for the whole year. The daily
    # resample broadcasts it over time, so the test is that it does not vary.
    assert float(ds["roughness"].std("time").max()) == pytest.approx(0.0)
    assert float(ds["roughness"].max()) == pytest.approx(0.25)


def test_derived_ignores_the_stored_field_and_varies_in_time(tmp_path):
    ds = prep_era5("ZZ", False, True, era5_dir=_combined_file(tmp_path / "e"), roughness="derived")
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
        prep_era5(
            "ZZ", False, True, era5_dir=_combined_file(tmp_path / "e"), roughness="annual-mean"
        )


def test_region_config_parses_the_treatment(tmp_path):
    """The three cases the setting has: absent, named, and named wrongly.

    The configurations are built here rather than taken from a shipped row,
    whose setting is a research decision that changes: this test asserted that
    the Denmark row read "stored" and broke the day the European rows moved to
    the per-timestep treatment. What a row is set to belongs in the tests that
    check the shipped tree, not in one that checks a parser.
    """
    base = open("configs/regions/scorecard/dk_k100.toml").read()
    without = base.replace('roughness = "derived"\n', "")
    (tmp_path / "absent.toml").write_text(without)
    (tmp_path / "derived.toml").write_text(
        without.replace("[era5]\n", '[era5]\nroughness = "derived"\n', 1)
    )
    (tmp_path / "stored.toml").write_text(
        without.replace("[era5]\n", '[era5]\nroughness = "stored"\n', 1)
    )
    (tmp_path / "bad.toml").write_text(
        without.replace("[era5]\n", '[era5]\nroughness = "annual"\n', 1)
    )
    assert load_region(tmp_path / "absent.toml").roughness == "derived"  # default
    assert load_region(tmp_path / "derived.toml").roughness == "derived"
    assert load_region(tmp_path / "stored.toml").roughness == "stored"
    with pytest.raises(ValueError, match="roughness"):
        load_region(tmp_path / "bad.toml")


def test_every_european_row_asks_for_the_per_timestep_treatment():
    """What the shipped rows are set to, asserted where it belongs: the
    European rows were re-run on the derived treatment on 2026-09-13
    (docs/findings/method-eu-rerun.md), and a configuration that quietly went
    back to the stored annual mean would be a silent method change."""
    european = [
        "de_k100",
        "dk_k100",
        "uk_k50",
        "fr_country",
        "be_country",
        "ie_country",
        "se_country",
        "no_country",
        "es_country",
        "it_country",
        "pt_country",
    ]
    for stem in european:
        spec = load_region(f"configs/regions/scorecard/{stem}.toml")
        assert spec.roughness == "derived", stem
        assert spec.era5_path == "era5/EU_2026-09", stem


def test_a_run_records_what_it_asked_for_and_what_it_applied(synthetic_dk):  # noqa: F811
    """The synthetic ERA5 carries no stored field, so a run that asks for one
    is derived, and the record has to show both. The request has to be explicit
    now that "derived" is the default, or the two could never differ here."""
    spec = make_spec(roughness="stored")
    train_dir = run_train(spec, synthetic_dk["root"] / "validation", mode="onshore", run_name="t")
    record = json.loads((train_dir / "run_manifest.json").read_text())["era5_roughness"]
    assert record == {"requested": "stored", "applied": "derived"}


def test_a_default_run_records_the_derived_treatment(synthetic_dk):  # noqa: F811
    """A run that asks for nothing now asks for, and applies, the method."""
    train_dir = run_train(
        make_spec(), synthetic_dk["root"] / "validation", mode="onshore", run_name="d"
    )
    record = json.loads((train_dir / "run_manifest.json").read_text())["era5_roughness"]
    assert record == {"requested": "derived", "applied": "derived"}
