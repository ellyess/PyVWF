"""Turbine-level training loads the config's train_years, not the adapter's default.

Before 2026-09-24 turbine-level training asked the adapter for observations
without years, so each adapter's hard-coded default window was fitted and the
config's ``train_years`` reached only the manifest and the accepted-years
count. Every shipped config matched its adapter, so nothing had moved; a config
naming another window would have been recorded as trained on years it never
saw.
"""

from __future__ import annotations

import pandas as pd
import pytest

import vwf.harness.driver as driver
from vwf.data import prep_country
from vwf.harness.regions import load_region

CONFIG = """
[region]
code = "ZZ"
name = "Testland"

[observations]
source = "test-source"
obs_level = "turbine"
obs_unit = "farm"
train_years = [2016, 2017]
test_years = [2019]

[era5]
path = "era5/ZZ"
bbox = [0.0, 10.0, 40.0, 50.0]
file_tag = "ZZ"

[correction]
model = "affine-wind"
cluster_list = [1]
time_slices = ["fixed"]

[seasons]
summer = [6, 7, 8]
autumn = [9, 10, 11]
winter = [12, 1, 2]
spring = [3, 4, 5]
"""


class RecordingSource:
    """A turbine-level adapter whose own default window differs from the config."""

    obs_level = "turbine"
    default_train_years = (2010, 2020)

    def __init__(self):
        self.calls = []

    def load_metadata(self):
        return pd.DataFrame({"ID": ["a"]})

    def load_observations(self, year_start=None, year_end=None):
        self.calls.append((year_start, year_end))
        return pd.DataFrame()


def test_prep_country_forwards_the_training_window():
    source = RecordingSource()
    prep_country("ZZ", source=source, train_years=(2016, 2017))
    assert source.calls == [(2016, 2017)]


def test_prep_country_without_a_window_keeps_the_adapter_default():
    source = RecordingSource()
    prep_country("ZZ", source=source)
    assert source.calls == [(None, None)]


def test_run_train_passes_the_configs_years(tmp_path, monkeypatch):
    config = tmp_path / "zz.toml"
    config.write_text(CONFIG)
    spec = load_region(config)
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after capturing")

    monkeypatch.setattr(driver, "train_set", capture)
    with pytest.raises(RuntimeError, match="stop after capturing"):
        driver.run_train(spec, tmp_path, source=RecordingSource())
    assert seen["train_years"] == (2016, 2017)
