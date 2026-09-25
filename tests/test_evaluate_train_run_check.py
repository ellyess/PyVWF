"""run_evaluate refuses a training run that does not belong to its config.

Evaluation scores every factors file in the training directory, on the rows
every variant can score. A stray file from another configuration (a reused
run name, a sweep evaluated with a single-row config) therefore adds a variant
and can move the rows the reported variant is scored on. A run trained under
another region, correction model or season mapping would be applied to the
wrong clusters or months. Both are now refused before any data is loaded.
"""

from __future__ import annotations

import json

import pytest

from pyvwf.harness.driver import _check_train_run
from pyvwf.harness.regions import load_region

CONFIG = """
[region]
code = "ZZ"
name = "Testland"

[observations]
source = "test-source"
obs_level = "turbine"
obs_unit = "farm"
train_years = [2015, 2018]
test_years = [2019]

[era5]
path = "era5/ZZ"
bbox = [0.0, 10.0, 40.0, 50.0]
file_tag = "ZZ"

[correction]
model = "affine-wind"
cluster_list = [1, 5]
time_slices = ["fixed", "season"]

[seasons]
summer = [6, 7, 8]
autumn = [9, 10, 11]
winter = [12, 1, 2]
spring = [3, 4, 5]
"""

SEASONS = {"summer": [6, 7, 8], "autumn": [9, 10, 11], "winter": [12, 1, 2], "spring": [3, 4, 5]}


@pytest.fixture
def spec(tmp_path):
    path = tmp_path / "zz.toml"
    path.write_text(CONFIG)
    return load_region(path)


def train_run(tmp_path, factors, **manifest):
    run = tmp_path / "train-run"
    run.mkdir()
    for name in factors:
        (run / f"factors_{name}.csv").write_text("cluster,scalar,offset\n")
    if manifest is not None:
        record = {
            "region": {"code": "ZZ"},
            "correction": {"model": "affine-wind"},
            "seasons": SEASONS,
        }
        for key, value in manifest.items():
            record[key] = value
        (run / "run_manifest.json").write_text(json.dumps(record))
    return run


def test_a_run_matching_its_config_passes(spec, tmp_path):
    _check_train_run(spec, train_run(tmp_path, ["fixed_1", "season_5"]))


def test_a_subset_of_the_config_passes(spec, tmp_path):
    _check_train_run(spec, train_run(tmp_path, ["fixed_5"]))


def test_a_factors_file_outside_the_config_is_refused(spec, tmp_path):
    run = train_run(tmp_path, ["fixed_1", "fixed_7"])
    with pytest.raises(ValueError, match="fixed_7"):
        _check_train_run(spec, run)


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("region", {"code": "YY"}, "region code"),
        ("correction", {"model": "scalar-only"}, "correction model"),
        ("seasons", {**SEASONS, "winter": [6, 7, 8], "summer": [12, 1, 2]}, "seasons"),
    ],
)
def test_a_run_trained_under_another_config_is_refused(spec, tmp_path, field, value, match):
    run = train_run(tmp_path, ["fixed_1"], **{field: value})
    with pytest.raises(ValueError, match=match):
        _check_train_run(spec, run)


def test_a_run_without_a_manifest_warns_and_passes(spec, tmp_path):
    run = train_run(tmp_path, ["fixed_1"])
    (run / "run_manifest.json").unlink()
    with pytest.warns(UserWarning, match="no run_manifest.json"):
        _check_train_run(spec, run)
