"""Deprecated public names still work, and say they are deprecated.

Each name here has no caller in PyVWF, but it is documented API, so it is
deprecated before it is removed rather than deleted outright.
"""
import pandas as pd
import pytest

import vwf.data as data
from vwf import PyVWF
from vwf.config import PyVWFPaths
from vwf.loaders.country_level_loaders import country_gen_to_cf


@pytest.mark.parametrize(
    "name, target",
    [
        ("COUNTRY_DIR", "COUNTRY_DATA"),
        ("TURBINE_DIR", "TURBINE_DATA"),
        ("COUNTRY_LEVEL_DIR", "COUNTRY_LEVEL_DATA"),
    ],
)
def test_data_path_constants_warn_and_read_pyvwfpaths(name, target):
    with pytest.warns(DeprecationWarning, match=f"vwf.data.{name}"):
        value = getattr(data, name)
    assert value == getattr(PyVWFPaths, target)


def test_unknown_data_attribute_still_raises():
    with pytest.raises(AttributeError):
        data.NO_SUCH_NAME  # noqa: B018


def test_sim_turbines_to_country_cf_warns_and_still_aggregates():
    sim = pd.DataFrame(
        {"year": [2020, 2020], "month": [1, 1], "ID": [1, 2], "sim": [0.2, 0.4]}
    )
    turb = pd.DataFrame({"ID": [1, 2], "capacity": [1000.0, 3000.0]})
    with pytest.warns(DeprecationWarning, match="sim_turbines_to_country_cf"):
        out = data.sim_turbines_to_country_cf(sim, turb)
    # Capacity-weighted: (0.2 * 1000 + 0.4 * 3000) / 4000.
    assert out["sim"].iloc[0] == pytest.approx(0.35)


def test_country_gen_to_cf_warns():
    obs = pd.DataFrame({"year": [2018], "month": [1], "output_kwh": [2.0e8]})
    turb = pd.DataFrame({"capacity": [500.0, 300.0]})
    with pytest.warns(DeprecationWarning, match="country_gen_to_cf"):
        cf = country_gen_to_cf(obs, turb, capacity_unit="MW")
    # January 2018 has 744 hours; 800 MW is 800,000 kW.
    assert cf["obs"].iloc[0] == pytest.approx(2.0e8 / (800_000.0 * 744))


def test_from_config_warns_before_touching_the_filesystem(tmp_path):
    missing = tmp_path / "no_config_here"
    with pytest.warns(DeprecationWarning, match="from_config"):
        with pytest.raises(FileNotFoundError):
            PyVWF.from_config("NL", config_dir=str(missing))
