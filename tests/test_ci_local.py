"""The local CI pre-check runs what the workflow runs, in the environments it installs.

`scripts/dev/ci_local.py` copies the workflow's commands and install lines. A
check renamed or retargeted in `.github/workflows/ci.yml` and not there would
leave the pre-check passing on something CI no longer runs.
"""

from __future__ import annotations


import pytest
from support import ROOT, load_script

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
