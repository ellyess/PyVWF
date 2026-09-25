"""Scripts reach pyvwf through public names only.

Analyses in ``scripts/`` used to import the harness's private helpers
(``driver._tidy_eval_frame`` and others), so a refactor of a private function
could break a study that no test ran. Each now has a public wrapper that calls the private function, and this
file checks two things: that every wrapper behaves as the function it wraps,
and that no script under ``scripts/`` or ``examples/`` names a private ``pyvwf``
attribute again.

``scripts/pinn/`` is exempt: it is deferred until the turbine-only study runs
(issue #12), and is tidied then.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXEMPT = ("scripts/pinn/",)


def private_vwf_references(source: str) -> list[str]:
    """Every ``_name`` a module imports from pyvwf or reads off a pyvwf-bound name."""
    tree = ast.parse(source)
    bound: set[str] = set()
    found: list[str] = []

    def private(name: str) -> bool:
        return name.startswith("_") and not name.startswith("__")

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "pyvwf":
            for alias in node.names:
                if private(alias.name):
                    found.append(f"{node.module}.{alias.name}")
                bound.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "pyvwf":
                    bound.add(alias.asname or alias.name.split(".")[0])
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and private(node.attr):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in bound:
                found.append(f"{root.id}...{node.attr}")
    return found


def test_the_detector_finds_what_it_should_and_nothing_else():
    bad = (
        "from pyvwf.harness.driver import _tidy_eval_frame, load_obs_and_fleet\n"
        "from pyvwf.harness import driver\n"
        "import pyvwf.correction\n"
        "import pyvwf.wind as w\n"
        "driver._SCOPE_KEYS\n"
        "pyvwf.correction._find_offset_iterative\n"
        "w._get_power_curve_cache\n"
    )
    assert private_vwf_references(bad) == [
        "pyvwf.harness.driver._tidy_eval_frame",
        "driver..._SCOPE_KEYS",
        "pyvwf..._find_offset_iterative",
        "w..._get_power_curve_cache",
    ]
    good = (
        "from pyvwf.harness import driver\n"
        "import other as o\n"
        "driver.SCOPE_KEYS\n"
        "driver.__name__\n"
        "o._private\n"
        "self._x\n"
    )
    assert private_vwf_references(good) == []


def _scripts():
    for p in sorted([*(ROOT / "scripts").rglob("*.py"), *(ROOT / "examples").rglob("*.py")]):
        rel = p.relative_to(ROOT).as_posix()
        if not rel.startswith(EXEMPT):
            yield rel


@pytest.mark.parametrize("rel", list(_scripts()))
def test_scripts_use_public_vwf_names(rel):
    assert private_vwf_references((ROOT / rel).read_text()) == []


def test_the_driver_wrappers_are_the_harness_functions():
    from pyvwf.harness import driver

    assert driver.SCOPE_KEYS is driver._SCOPE_KEYS
    for public, private in [
        (driver.era5_dir, driver._era5_dir),
        (driver.tidy_eval_frame, driver._tidy_eval_frame),
        (driver.country_pairs, driver._country_pairs),
        (driver.country_skill, driver._country_skill),
        (driver.error_metrics, driver._error_metrics),
        (driver.score_on_common_rows, driver._score_on_common_rows),
    ]:
        assert inspect.signature(public) == inspect.signature(private), public.__name__
    merged = pd.DataFrame({"cf_sim": [0.31, 0.27, 0.40], "cf_obs": [0.30, 0.25, 0.44]})
    assert driver.error_metrics(merged) == driver._error_metrics(merged)


def test_the_other_wrappers_are_the_functions_they_wrap():
    from pyvwf import wind
    from pyvwf.datasets import cen_cl

    curves = pd.DataFrame({"data$speed": np.arange(0.0, 5.0), "M1": np.linspace(0, 1, 5)})
    got, want = wind.power_curve_arrays(curves), wind._get_power_curve_cache(curves)
    np.testing.assert_array_equal(got[0], want[0])
    assert got[1].keys() == want[1].keys()
    stamps = pd.Series(pd.to_datetime(["2024-01-01 00:00", "2024-07-01 12:30"]))
    pd.testing.assert_series_equal(cen_cl.local_to_utc(stamps), cen_cl._local_to_utc(stamps))
