"""The CI constraints files, the workflow that uses them and the script that writes them agree.

`scripts/dev/lock.py` writes one constraints file per pinned Python version,
and `.github/workflows/ci.yml` installs each pinned test cell through its file.
A version added to one and not the others would run unpinned without anyone
noticing, and a dependency added to `pyproject.toml` without a refresh would
be resolved freely inside an otherwise pinned install.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from support import load_script

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def _lock():
    return load_script(ROOT / "scripts" / "dev" / "lock.py", "lock")


def _workflow_constraints() -> dict[str, str]:
    """Python version to constraints path, from the test matrix's include list."""
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    pairs = re.findall(
        r'- python-version: "(\d\.\d+)"\n\s+constraints: (\S+)', text, flags=re.MULTILINE
    )
    return dict(pairs)


def _name(requirement: str) -> str:
    return re.split(r"[<>=!~\[; ]", requirement, maxsplit=1)[0].lower().replace("_", "-")


def _applies(requirement: str, version: str) -> bool:
    """False for a requirement whose marker excludes this Python version."""
    marker = re.search(r"python_version\s*([<>=]+)\s*['\"](\d\.\d+)['\"]", requirement)
    if not marker:
        return True
    op, bound = marker.groups()
    have = tuple(int(x) for x in version.split("."))
    want = tuple(int(x) for x in bound.split("."))
    return {"<": have < want, "<=": have <= want, ">": have > want, ">=": have >= want}[op]


def test_the_workflow_pins_exactly_the_versions_the_script_writes():
    lock = _lock()
    pinned = _workflow_constraints()
    assert sorted(pinned) == sorted(lock.PINNED)
    for version, path in pinned.items():
        assert (ROOT / path) == lock.path_for(version)
        assert (ROOT / path).is_file(), path


@pytest.mark.parametrize("version", _lock().PINNED)
def test_every_direct_dependency_is_pinned(version):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    wanted = project["dependencies"] + project["optional-dependencies"]["dev"]
    text = _lock().path_for(version).read_text()
    assert text.startswith("# Constraints for the CI test matrix on Python " + version)
    pinned = {m.lower() for m in re.findall(r"^([A-Za-z0-9_.-]+)==", text, flags=re.MULTILINE)}
    pinned = {p.replace("_", "-") for p in pinned}
    missing = sorted(_name(r) for r in wanted if _applies(r, version) and _name(r) not in pinned)
    assert not missing, f"not in {version}'s constraints; run python scripts/dev/lock.py"


@pytest.mark.parametrize("stem", list(_lock().JOBS))
def test_every_pinned_job_file_exists_is_used_and_pins_its_extras(stem):
    """The extras and docs jobs install through their own files (#87)."""
    lock = _lock()
    version, extras = lock.JOBS[stem]
    path = lock.job_path(stem)
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert f"-c constraints/{path.name}" in workflow, f"{path.name} is not used by ci.yml"
    text = path.read_text()
    assert text.startswith(f"# Constraints for the CI job installing `.[{','.join(extras)}]`")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    wanted = list(project["dependencies"])
    for extra in extras:
        wanted += project["optional-dependencies"][extra]
    pinned = {m.lower().replace("_", "-") for m in re.findall(r"^([A-Za-z0-9_.-]+)==", text, re.M)}
    missing = sorted(_name(r) for r in wanted if _applies(r, version) and _name(r) not in pinned)
    assert not missing, f"not in {path.name}; run python scripts/dev/lock.py"


def test_a_job_file_shares_its_test_cells_versions():
    """Resolved against the test matrix's file, so shared packages do not move."""
    lock = _lock()

    def pins(path):
        return dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", path.read_text(), re.M))

    for stem, (version, _extras) in lock.JOBS.items():
        cell, job = pins(lock.path_for(version)), pins(lock.job_path(stem))
        moved = {k: (cell[k], job[k]) for k in cell.keys() & job.keys() if cell[k] != job[k]}
        assert not moved, f"{stem} moves {moved} from {lock.path_for(version).name}"
