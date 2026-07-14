"""Tests for the `pyvwf-train` console script.

The CLI is the public entry point most users meet first, so its argument
contract is worth pinning: a typo in a default or a renamed choice silently
changes what a published run actually did.

The full run is exercised by tests/test_pipeline.py; here we check the parser
and that `main` wires its arguments through to the model unchanged.
"""
from __future__ import annotations

import pytest

from vwf.cli.train import _build_parser, main


def test_defaults_match_the_documented_workflow():
    args = _build_parser().parse_args(["--outdir", "out"])

    assert args.country == "DK"
    assert args.year_test == 2020
    assert args.cluster_mode == "onshore"
    assert args.cluster_list == [1, 2, 5, 7, 10]
    assert args.time_res_list == ["fixed", "season", "bimonth", "month"]
    # Roughness is opt-in: it changes the hub-height extrapolation, so it must
    # not switch itself on behind the user's back.
    assert args.calc_z0 is False
    assert args.add_nan is None
    assert args.interp_nan is None
    assert args.fix_turb is None


def test_outdir_is_required():
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])


def test_cluster_mode_rejects_an_unknown_choice():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--outdir", "out", "--cluster-mode", "sideways"])


def test_lists_and_flags_parse():
    args = _build_parser().parse_args(
        [
            "--outdir", "out",
            "--country", "DE",
            "--year-test", "2019",
            "--calc-z0",
            "--cluster-list", "10", "100",
            "--time-res-list", "fixed", "month",
            "--fix-turb", "Synthetic.Onshore2000",
        ]
    )

    assert args.country == "DE"
    assert args.year_test == 2019
    assert args.calc_z0 is True
    assert args.cluster_list == [10, 100]
    assert args.time_res_list == ["fixed", "month"]
    assert args.fix_turb == "Synthetic.Onshore2000"


def test_main_passes_arguments_through_to_the_model(tmp_path, monkeypatch):
    """The CLI must not quietly rewrite what it was asked to run."""
    captured = {}

    class FakeModel:
        def __init__(self, path, country, correct, **kwargs):
            captured["path"] = path
            captured["country"] = country
            captured["correct"] = correct
            captured.update(kwargs)

        def train(self, check):
            captured["trained"] = check

        def simulate_cf(self, year_test):
            captured["year_test"] = year_test

    monkeypatch.setattr("vwf.cli.train.PyVWF", FakeModel)

    main(
        [
            "--outdir", str(tmp_path / "run"),
            "--country", "DE",
            "--year-test", "2018",
            "--calc-z0",
            "--cluster-list", "7",
            "--time-res-list", "season",
        ]
    )

    assert captured["country"] == "DE"
    assert captured["correct"] is True          # the CLI always trains a correction
    assert captured["calc_z0"] is True
    assert captured["cluster_mode"] == "onshore"
    assert captured["cluster_list"] == [7]
    assert captured["time_res_list"] == ["season"]
    assert captured["year_test"] == 2018
    assert captured["trained"] is False         # --train-plots not passed
    # The output directory is created up front, so a long run doesn't die at the
    # first write.
    assert (tmp_path / "run").is_dir()
