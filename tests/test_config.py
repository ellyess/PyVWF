"""Tests for path resolution: the difference between "works in my checkout"
and "works when installed".

`load_power_curves` and `add_models` used to read the literal relative paths
``input/reference/power_curves.csv`` and ``input/reference/models.csv``, so a
`pip install pyvwf`
user working anywhere other than a repository checkout got a FileNotFoundError
before they could simulate anything. They now resolve through
``PyVWFPaths.reference_file``, which prefers the user's own tables and falls
back to the open library bundled with the package, silently: since 2026-10-28
it is the only copy and the default, and a harness run records which library
and which curves it used.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd
import pytest

from pyvwf.config import PyVWFPaths
from pyvwf.curves import add_models, load_power_curves


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
    win over the bundled open library."""
    monkeypatch.setattr(PyVWFPaths, "INPUT_ROOT", tmp_path)
    local = tmp_path / "reference" / "power_curves.csv"
    local.parent.mkdir()
    local.write_text("data$speed,MyTurbine.5000\n0,0\n10,0.8\n")

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # must NOT warn: this is the real thing
        resolved = PyVWFPaths.reference_file("power_curves.csv")

    assert resolved == local
    assert "MyTurbine.5000" in pd.read_csv(resolved).columns


def test_reference_file_falls_back_to_the_bundled_table_silently(no_input_root):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolved = PyVWFPaths.reference_file("power_curves.csv")

    assert resolved.is_file()
    assert resolved.name == "power_curves.csv"
    # It came from inside the installed package, not the working directory.
    assert "pyvwf" in resolved.parts and "resources" in resolved.parts


def test_reference_file_raises_for_an_unknown_table(no_input_root):
    with pytest.raises(FileNotFoundError, match="PYVWF_INPUT"):
        PyVWFPaths.reference_file("not_a_real_table.csv")


# ---------------------------------------------------------------------------
# The loaders that used to be cwd-dependent
# ---------------------------------------------------------------------------


def test_load_power_curves_works_outside_a_checkout(no_input_root):
    """Regression: this raised FileNotFoundError for every pip-installed user
    whose working directory was not a repository checkout."""
    curves = load_power_curves()

    assert "data$speed" in curves.columns
    # The bundled open library carries the market-average composite that the
    # sampling-point default relies on.
    assert "2019COE_Market_Average_2.6MW_121" in curves.columns
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
    out = add_models(fleet)

    assert out["model"].notna().all()
    # Models must resolve to columns that actually exist in the power curves,
    # or the simulation blows up later with a KeyError.
    assert out["model"].isin(load_power_curves().columns).all()


def test_pyvwf_input_env_var_redirects_the_root(tmp_path):
    """PYVWF_INPUT is what lets an installed copy find input data anywhere.

    Run in a subprocess rather than with importlib.reload. Reloading
    ``pyvwf.config`` rebinds ``PyVWFPaths`` to a NEW class object, and every
    module that did ``from pyvwf.config import PyVWFPaths`` at import time keeps
    the old one. Reloading a second time restores the values but not those
    stale references, so a later test that monkeypatches ``PyVWFPaths`` patches
    a class the code under test is no longer reading. That surfaced as
    test_country_zonal failing with an empty ERA5 glob, but only when the whole
    suite shares one process, which is how CI runs it and how a per-file local
    run never does. A subprocess is also the more honest test: a fresh
    interpreter is exactly what an installed copy gets.
    """
    import json
    import subprocess
    import sys

    env = {**os.environ, "PYVWF_INPUT": str(tmp_path)}
    probe = (
        "import json;from pyvwf.config import PyVWFPaths as P;"
        "print(json.dumps([str(P.INPUT_ROOT), str(P.TURBINE_DATA)]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
    )
    root, turbine = (Path(p) for p in json.loads(out.stdout))
    assert root == tmp_path
    assert turbine == tmp_path / "observations" / "turbine"
