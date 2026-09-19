"""The shared CLI helpers (vwf.cli.common) and the pyvwf-validate entry point."""

from pathlib import Path

import pytest

from vwf.cli import common, validate
from vwf.config import PyVWFPaths


def test_input_path_follows_the_input_root_when_called(monkeypatch):
    monkeypatch.setattr(PyVWFPaths, "INPUT_ROOT", Path("input/combined"))
    assert common.input_path("raw", "cen") == Path("input/combined/raw/cen")
    monkeypatch.setattr(PyVWFPaths, "INPUT_ROOT", Path("input"))
    assert common.input_path("raw", "cen") == Path("input/raw/cen")


def test_add_input_path_default_and_override(monkeypatch):
    monkeypatch.setattr(PyVWFPaths, "INPUT_ROOT", Path("elsewhere"))
    parser = common.make_parser("doc")
    common.add_input_path(parser, "--raw", "raw", "emi", help="EMI downloads")
    assert parser.parse_args([]).raw == Path("elsewhere/raw/emi")
    assert parser.parse_args(["--raw", "x"]).raw == Path("x")
    assert "EMI downloads (default: elsewhere/raw/emi)" in parser.format_help()


def test_run_returns_the_exit_code():
    parser = common.make_parser(None)
    parser.add_argument("n", type=int)
    assert common.run(lambda a: None, parser, ["3"]) == 0
    assert common.run(lambda a: a.n, parser, ["3"]) == 3


@pytest.fixture
def calls(monkeypatch):
    """Replace the harness driver so the dispatch can be checked without data."""
    import vwf.harness.driver as driver
    import vwf.harness.regions as regions

    seen = []
    monkeypatch.setattr(regions, "load_region", lambda p: f"spec:{Path(p).name}")
    for name in ("run_train", "run_evaluate", "run_transfer"):
        monkeypatch.setattr(
            driver, name, lambda *a, _n=name, **k: seen.append((_n, a, k)) or "run-dir"
        )
    return seen


def test_validate_train(calls, capsys):
    assert validate.main(["train", "--region", "configs/regions/nz.toml"]) == 0
    assert calls == [
        ("run_train", ("spec:nz.toml", "output/validation"), {"mode": "all", "run_name": None})
    ]
    assert "Run complete: run-dir" in capsys.readouterr().out


def test_validate_evaluate_and_transfer(calls):
    validate.main(["evaluate", "--region", "r.toml", "--train-run", "t", "--year", "2020"])
    validate.main(
        [
            "transfer",
            "--region",
            "uk.toml",
            "--source-region",
            "au.toml",
            "--source-run",
            "s",
            "--mode",
            "onshore",
        ]
    )
    assert calls[0] == (
        "run_evaluate",
        ("spec:r.toml", "t", "output/validation"),
        {"year": 2020, "mode": "all", "run_name": None},
    )
    assert calls[1] == (
        "run_transfer",
        ("spec:au.toml", "s", "spec:uk.toml", "output/validation"),
        {"year": None, "mode": "onshore", "run_name": None},
    )


def test_validate_requires_a_command():
    with pytest.raises(SystemExit):
        validate.main([])
