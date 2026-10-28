"""CI's ruff, the pre-commit hook's ruff and environment.yaml's are one version.

CI installs ruff from the ``dev`` extra and runs ``ruff format --check``;
the pre-commit hook runs the version its rev names. Two ruff releases can
format the same file differently (0.12 and 0.16 do), so if the two drift, a
file the hook formatted can fail CI with no code change.
"""

from __future__ import annotations

import re
from support import ROOT, tomllib


def test_dev_ruff_pin_matches_the_pre_commit_rev():
    dev = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"][
        "dev"
    ]
    pins = [d for d in dev if re.match(r"ruff\b", d)]
    assert len(pins) == 1 and "==" in pins[0], f"dev extra must pin ruff exactly: {pins}"
    dev_version = pins[0].split("==")[1].strip()

    config = (ROOT / ".pre-commit-config.yaml").read_text()
    rev = re.search(r"astral-sh/ruff-pre-commit\s*\n\s*rev:\s*v?([\d.]+)", config)
    assert rev, "no ruff-pre-commit rev found"
    assert dev_version == rev.group(1)


def _dev_ruff_version() -> str:
    dev = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"][
        "dev"
    ]
    return next(d for d in dev if re.match(r"ruff\b", d)).split("==")[1].strip()


def test_environment_yaml_pins_the_same_ruff():
    """A conda env built from the file formats like CI, not like an older ruff."""
    env = (ROOT / "environment.yaml").read_text()
    pins = re.findall(r"^\s*-\s*ruff\s*(\S*)\s*$", env, flags=re.M)
    assert pins == [f"=={_dev_ruff_version()}"], pins


def test_environment_yaml_floors_are_no_lower_than_the_package_s():
    """environment.yaml may not promise a version the code cannot run on."""
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    floors = {}
    for dep in deps:
        m = re.match(r"([A-Za-z0-9_.-]+)(?:\[[^]]*\])?>=([\d.]+)", dep)
        if m:
            floors[m.group(1).lower()] = tuple(int(x) for x in m.group(2).split("."))
    env = (ROOT / "environment.yaml").read_text()
    checked = 0
    for name, version in re.findall(
        r"^\s*-\s*([A-Za-z0-9_.-]+)\s*(?:>=|==|=)\s*([\d.]+)", env, re.M
    ):
        floor = floors.get(name.lower())
        if floor is None:
            continue
        checked += 1
        got = tuple(int(x) for x in version.split("."))
        assert got[: len(floor)] >= floor, f"{name} {version} is below pyproject's floor {floor}"
    assert checked >= 10, f"only {checked} packages compared; the parse has drifted"
