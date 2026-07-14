"""Tests for path resolution — the difference between "works in my checkout"
and "works when installed".

`load_power_curves` and `add_models` used to read the literal relative paths
``input/power_curves.csv`` and ``input/models.csv``, so a `pip install pyvwf`
user working anywhere other than a repository checkout got a FileNotFoundError
before they could simulate anything. They now resolve through
``PyVWFPaths.reference_file``, which prefers the user's own tables and falls
back to the synthetic ones bundled with the package.

The fallback must *warn*. The bundled power curves are invented; simulating with
them yields capacity factors that look entirely plausible and mean nothing,
which is the worst failure mode research software can have.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from vwf.config import PyVWFPaths
from vwf.data import add_models, load_power_curves


@pytest.fixture
def no_input_root(tmp_path, monkeypatch):
    """Simulate an installed PyVWF with no input/ directory in sight."""
    monkeypatch.setattr(PyVWFPaths, "INPUT_ROOT", tmp_path / "nonexistent")
    monkeypatch.chdir(tmp_path)


# ---------------------------------------------------------------------------
# reference_file resolution
# ---------------------------------------------------------------------------

def test_reference_file_prefers_the_users_own_table(tmp_path, monkeypatch):
    """A local (possibly licensed, non-redistributable) power-curve library must
    win over the bundled synthetic placeholder."""
    monkeypatch.setattr(PyVWFPaths, "INPUT_ROOT", tmp_path)
    local = tmp_path / "power_curves.csv"
    local.write_text("data$speed,MyTurbine.5000\n0,0\n10,0.8\n")

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # must NOT warn: this is the real thing
        resolved = PyVWFPaths.reference_file("power_curves.csv")

    assert resolved == local
    assert "MyTurbine.5000" in pd.read_csv(resolved).columns


def test_reference_file_falls_back_to_the_bundled_table(no_input_root):
    with pytest.warns(UserWarning, match="SYNTHETIC"):
        resolved = PyVWFPaths.reference_file("power_curves.csv")

    assert resolved.is_file()
    assert resolved.name == "power_curves.csv"
    # It came from inside the installed package, not the working directory.
    assert "vwf" in resolved.parts and "resources" in resolved.parts


def test_reference_file_fallback_warning_names_the_risk(no_input_root):
    """The warning has to say the numbers are meaningless, not merely that a
    default was used."""
    with pytest.warns(UserWarning) as record:
        PyVWFPaths.reference_file("models.csv")

    message = str(record[0].message)
    assert "SYNTHETIC" in message
    assert "not physically meaningful" in message
    assert "PYVWF_INPUT" in message


def test_reference_file_raises_for_an_unknown_table(no_input_root):
    with pytest.raises(FileNotFoundError, match="PYVWF_INPUT"):
        PyVWFPaths.reference_file("not_a_real_table.csv")


# ---------------------------------------------------------------------------
# The loaders that used to be cwd-dependent
# ---------------------------------------------------------------------------

def test_load_power_curves_works_outside_a_checkout(no_input_root):
    """Regression: this raised FileNotFoundError for every pip-installed user
    whose working directory was not a repository checkout."""
    with pytest.warns(UserWarning, match="SYNTHETIC"):
        curves = load_power_curves()

    assert "data$speed" in curves.columns
    assert any(c.startswith("Synthetic.") for c in curves.columns)
    assert len(curves) > 1


def test_add_models_works_outside_a_checkout(no_input_root):
    fleet = pd.DataFrame(
        {
            "ID": ["a", "b"],
            "manufacturer": ["Synthetic", "Synthetic"],
            "capacity": [2000.0, 3600.0],
            "diameter": [80.0, 112.0],
            "height": [100.0, 120.0],
            "lon": [8.5, 9.0],
            "lat": [55.5, 56.0],
        }
    )
    with pytest.warns(UserWarning, match="SYNTHETIC"):
        out = add_models(fleet)

    assert out["model"].notna().all()
    # Models must resolve to columns that actually exist in the power curves,
    # or the simulation blows up later with a KeyError.
    with pytest.warns(UserWarning):
        curves = load_power_curves()
    assert out["model"].isin(curves.columns).all()


def test_repo_checkout_still_uses_its_own_input_dir():
    """In a checkout (the default INPUT_ROOT), the repo's own tables are used and
    nothing warns — the existing workflow is untouched."""
    if not Path("input/power_curves.csv").is_file():
        pytest.skip("not running from a repository checkout")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        curves = load_power_curves()

    assert "data$speed" in curves.columns


def test_pyvwf_input_env_var_redirects_the_root(tmp_path, monkeypatch):
    """PYVWF_INPUT is what lets an installed copy find input data anywhere."""
    monkeypatch.setenv("PYVWF_INPUT", str(tmp_path))

    import importlib

    import vwf.config

    reloaded = importlib.reload(vwf.config)
    try:
        assert reloaded.PyVWFPaths.INPUT_ROOT == tmp_path
        assert reloaded.PyVWFPaths.ERA5_DATA == tmp_path / "era5" / "EU"
        assert reloaded.PyVWFPaths.TURBINE_DATA == tmp_path / "turbine_level_data"
    finally:
        # Restore the module for every other test in the session.
        monkeypatch.delenv("PYVWF_INPUT")
        importlib.reload(vwf.config)
