"""The shared CLI helpers (pyvwf.cli.common) and the pyvwf-validate entry point."""

from pathlib import Path

import pytest

from pyvwf.cli import common, validate
from pyvwf.config import PyVWFPaths


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
    import pyvwf.harness.driver as driver
    import pyvwf.harness.regions as regions

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


def _announced(monkeypatch, capsys, *, env=None, root="input"):
    """The banner one run writes, with the announce-once latch reset."""
    monkeypatch.setattr(common, "_announced", False)
    monkeypatch.setattr(PyVWFPaths, "INPUT_ROOT", Path(root))
    monkeypatch.delenv("PYVWF_INPUT", raising=False)
    monkeypatch.delenv("PYVWF_QUIET_ROOT", raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    common.announce_input_root()
    return capsys.readouterr()


def test_the_root_and_its_provenance_are_announced(monkeypatch, capsys):
    """Which root a run read decides its numbers, so every run says it."""
    out = _announced(monkeypatch, capsys)
    assert out.err.strip() == "input root: input (default)"
    # stderr, not stdout: entry points whose stdout a caller parses are
    # unaffected (tests/test_pin_offset_fits.py reads the last stdout line).
    assert out.out == ""

    out = _announced(
        monkeypatch, capsys, env={"PYVWF_INPUT": "input/combined"}, root="input/combined"
    )
    assert out.err.strip() == "input root: input/combined (PYVWF_INPUT)"


def test_the_announcement_is_once_per_process_and_can_be_silenced(monkeypatch, capsys):
    _announced(monkeypatch, capsys)
    common.announce_input_root()  # an entry point that also goes through run()
    assert capsys.readouterr().err == ""

    monkeypatch.setattr(common, "_announced", False)
    monkeypatch.setenv("PYVWF_QUIET_ROOT", "1")
    common.announce_input_root()
    assert capsys.readouterr().err == ""


def test_only_an_input_flag_that_leaves_the_root_is_reported(monkeypatch, capsys):
    """A path still under the root is what the root already said."""
    monkeypatch.setattr(PyVWFPaths, "INPUT_ROOT", Path("input"))
    monkeypatch.delenv("PYVWF_QUIET_ROOT", raising=False)
    parser = common.make_parser("doc")
    common.add_input_path(parser, "--obs", "observations", "turbine")
    parser.add_argument("--out", type=Path, default=Path("output/x"))

    for argv in ([], ["--obs", "input/observations/country"]):
        common.announce_path_overrides(parser, parser.parse_args(argv))
        assert capsys.readouterr().err == "", argv

    # An output path is not the root's business, even well outside it.
    common.announce_path_overrides(parser, parser.parse_args(["--out", "/tmp/e"]))
    assert capsys.readouterr().err == ""

    common.announce_path_overrides(parser, parser.parse_args(["--obs", "/tmp/elsewhere"]))
    assert "outside the input root: --obs=/tmp/elsewhere" in capsys.readouterr().err
