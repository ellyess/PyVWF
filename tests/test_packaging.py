"""Packaging invariants: one version, semantically formatted, no drift.

``vwf.__version__`` is the single source of truth — ``pyproject.toml`` reads it
dynamically. ``CITATION.cff`` cannot, so it is the one place a stale version can
hide; these tests fail loudly when it drifts.

Optional dependencies get the same treatment from the other side: the data
-acquisition extras must stay out of the import path of the simulation code, or
a plain ``pip install pyvwf`` silently stops working.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

import vwf

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10: tomllib is not in the standard library yet.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent

# Major.minor.patch, with an optional pre-release/build suffix (semver.org).
SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


def test_version_is_semver():
    assert SEMVER_RE.match(vwf.__version__), (
        f"vwf.__version__ = {vwf.__version__!r} is not semantic versioning"
    )


def test_pyproject_reads_version_from_package(pyproject):
    """The version must not be duplicated into pyproject — it must be derived."""
    project = pyproject["project"]
    assert "version" not in project, (
        "pyproject.toml hard-codes a version; it should stay dynamic so "
        "vwf.__version__ remains the single source of truth"
    )
    assert "version" in project["dynamic"]
    assert (
        pyproject["tool"]["setuptools"]["dynamic"]["version"]["attr"]
        == "vwf.__version__"
    )


def test_citation_version_matches_package():
    """CITATION.cff can't read the package, so it's the one drift risk."""
    citation = (ROOT / "CITATION.cff").read_text()
    match = re.search(r"^version:\s*(\S+)\s*$", citation, re.MULTILINE)
    assert match, "CITATION.cff has no version field"
    cff_version = match.group(1).strip("\"'")
    assert cff_version == vwf.__version__, (
        f"CITATION.cff version {cff_version!r} != vwf.__version__ "
        f"{vwf.__version__!r}; bump both when releasing"
    )


def test_data_acquisition_deps_are_optional(pyproject):
    """entsoe-py/openpyxl/pyarrow are needed only to *fetch* data, never to
    simulate, so they must not be core dependencies."""
    core = " ".join(pyproject["project"]["dependencies"])
    extra = " ".join(pyproject["project"]["optional-dependencies"]["data"])
    for pkg in ("entsoe-py", "openpyxl", "pyarrow"):
        assert pkg not in core, f"{pkg} should be in the 'data' extra, not core"
        assert pkg in extra, f"{pkg} missing from the 'data' extra"


def test_simulation_path_imports_without_data_extras():
    """The modules a plain `pip install pyvwf` user touches must import with
    only the core dependencies present."""
    import importlib

    for module in [
        "vwf",
        "vwf.wind",
        "vwf.correction",
        "vwf.metrics",
        "vwf.clustering",
        "vwf.datasets.era5",
        "vwf.viz",
    ]:
        importlib.import_module(module)


def test_package_ships_py_typed():
    """The 'Typing :: Typed' classifier is a lie without this marker file."""
    assert (ROOT / "src" / "vwf" / "py.typed").is_file()
    package_data = pyproject_package_data()
    assert "py.typed" in package_data, "py.typed must be declared as package-data"


def pyproject_package_data() -> list[str]:
    with open(ROOT / "pyproject.toml", "rb") as fh:
        data = tomllib.load(fh)
    return data["tool"]["setuptools"]["package-data"]["vwf"]
