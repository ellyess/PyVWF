"""The development tooling agrees with the CI workflow it stands in for.

Three checks, each of a copy that can drift from what CI runs without any test
failing:

- **The local CI pre-check runs what the workflow runs.**
  `scripts/dev/ci_local.py` copies the workflow's commands and install lines. A
  check renamed or retargeted in `.github/workflows/ci.yml` and not there would
  leave the pre-check passing on something CI no longer runs.
- **The constraints files, the workflow and lock.py agree.**
  `scripts/dev/lock.py` writes one constraints file per pinned Python version,
  and `.github/workflows/ci.yml` installs each pinned test cell through its
  file. A version added to one and not the others would run unpinned without
  anyone noticing, and a dependency added to `pyproject.toml` without a refresh
  would be resolved freely inside an otherwise pinned install.
- **CI's ruff, the pre-commit hook's and environment.yaml's are one version.**
  CI installs ruff from the ``dev`` extra and runs ``ruff format --check``; the
  pre-commit hook runs the version its rev names. Two ruff releases can format
  the same file differently (0.12 and 0.16 do), so if the two drift, a file the
  hook formatted can fail CI with no code change.
"""

from __future__ import annotations

import re

import pytest
from support import ROOT, load_script, tomllib


# The local CI pre-check runs what the workflow runs.

WORKFLOW = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")


def _script():
    return load_script(ROOT / "scripts" / "dev" / "ci_local.py", "ci_local")


@pytest.mark.parametrize(
    "name", ["ruff check", "ruff format", "mypy", "lint-imports", "deptry", "vulture"]
)
def test_lint_commands_are_the_workflows(name):
    _, command = _script().CHECKS[name]
    assert " ".join(command) in WORKFLOW


def test_pytest_selects_the_workflows_fast_set():
    _, command = _script().CHECKS["pytest"]
    marker = command[command.index("-m") + 1]
    assert f'-m "{marker}"' in WORKFLOW
    assert command[:2] == ["pytest", "tests"]


def test_docs_build_is_the_workflows():
    _, command = _script().CHECKS["docs"]
    assert command[-1] == "-W"
    assert " ".join(command[:4]) + " docs/_build/html -W" in WORKFLOW


@pytest.mark.parametrize("env", ["dev", "docs"])
def test_environments_install_as_the_workflow_does(env):
    module = _script()
    spec = module.ENVS[env]
    assert f'pip install -e "{spec.extras}" -c {spec.constraints}' in WORKFLOW
    assert (ROOT / spec.constraints).is_file()
    assert spec.constraints.startswith(f"constraints/py{module.PYTHON.replace('.', '')}")


def test_environments_live_under_a_git_ignored_path():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert _script().BASE.relative_to(ROOT).parts[0] == "output"
    assert "/output" in gitignore


# The constraints files, the workflow and lock.py agree.


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


# CI's ruff, the pre-commit hook's and environment.yaml's are one version.


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
