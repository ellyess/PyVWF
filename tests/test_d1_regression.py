"""The D1 frame comparator (scripts/d1_regression.py)."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "d1_regression",
    Path(__file__).resolve().parents[1] / "scripts" / "d1_regression.py",
)
d1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(d1)


def _factors(path, scalar):
    pd.DataFrame(
        {"cluster": [0, 1], "fixed": ["1/1", "1/1"], "scalar": scalar, "offset": [0.0, 0.0]}
    ).to_csv(path, index=False)


def test_identical_dirs_pass(tmp_path):
    a, b = tmp_path / "ref", tmp_path / "run"
    a.mkdir()
    b.mkdir()
    _factors(a / "factors_fixed_2.csv", [0.8, 0.9])
    _factors(b / "factors_fixed_2.csv", [0.8, 0.9])
    rows, any_fail = d1.compare_dirs(a, b, atol=1e-12, label="x")
    assert not any_fail
    assert rows[0]["status"] == "ok"
    assert rows[0]["max_abs_diff"] == 0.0


def test_small_diff_within_tol_passes_but_is_reported(tmp_path):
    a, b = tmp_path / "ref", tmp_path / "run"
    a.mkdir()
    b.mkdir()
    _factors(a / "factors_fixed_2.csv", [0.80000000, 0.90000000])
    _factors(b / "factors_fixed_2.csv", [0.80000005, 0.90000000])
    rows, any_fail = d1.compare_dirs(a, b, atol=1e-6, label="x")
    assert not any_fail
    assert rows[0]["max_abs_diff"] == pytest.approx(5e-8)
    # ...and the SAME diff fails a tighter tolerance (real pass/fail, not cosmetic).
    _, fail_tight = d1.compare_dirs(a, b, atol=1e-12, label="x")
    assert fail_tight


def test_nan_pattern_difference_is_infinite_and_fails(tmp_path):
    a, b = tmp_path / "ref", tmp_path / "run"
    a.mkdir()
    b.mkdir()
    _factors(a / "factors_fixed_2.csv", [0.8, float("nan")])
    _factors(b / "factors_fixed_2.csv", [0.8, 0.9])
    rows, any_fail = d1.compare_dirs(a, b, atol=1e-6, label="x")
    assert any_fail
    assert rows[0]["max_abs_diff"] == float("inf")
    assert "NaN pattern" in rows[0]["note"]


def test_missing_frame_fails(tmp_path):
    a, b = tmp_path / "ref", tmp_path / "run"
    a.mkdir()
    b.mkdir()
    _factors(a / "factors_fixed_2.csv", [0.8, 0.9])
    rows, any_fail = d1.compare_dirs(a, b, atol=1e-6, label="x")
    assert any_fail
    assert rows[0]["status"] == "MISSING"


def test_shape_mismatch_is_structural_fail(tmp_path):
    a, b = tmp_path / "ref", tmp_path / "run"
    a.mkdir()
    b.mkdir()
    _factors(a / "factors_fixed_2.csv", [0.8, 0.9])
    pd.DataFrame(
        {"cluster": [0], "fixed": ["1/1"], "scalar": [0.8], "offset": [0.0]}
    ).to_csv(b / "factors_fixed_2.csv", index=False)
    rows, any_fail = d1.compare_dirs(a, b, atol=1e-6, label="x")
    assert any_fail
    assert rows[0]["status"] == "STRUCT"
