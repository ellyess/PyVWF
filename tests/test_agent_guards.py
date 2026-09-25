"""The Claude Code guards fire on a known positive and stay quiet on a known negative.

A guard that has never been seen to fire is an assumption (AGENTS.md). Each
hook in `.claude/hooks/` is run here as Claude Code runs it, a JSON event on
stdin, against a scratch repository, and its exit code is checked both ways:
2 blocks the tool call, 0 lets it through. `docs/design/agent-guards.md` says
what each guard is for.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN_LOCK = ".claude/hooks/guard_run_lock.py"
SUITES = ".claude/hooks/guard_local_suites.py"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for rel in (".claude/hooks", "scripts/dev"):
        shutil.copytree(ROOT / rel, repo / rel)
    (repo / "src/vwf").mkdir(parents=True)
    (repo / "src/vwf/metrics.py").write_text("x = 1\n")
    (repo / "src/vwf/pinn").mkdir()
    (repo / "src/vwf/pinn/model.py").write_text("y = 1\n")
    (repo / "README.md").write_text("readme\n")
    (repo / ".gitignore").write_text("/output\n__pycache__/\n")
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return repo


def _hook(repo: Path, hook: str, tool: str, tool_input: dict) -> int:
    event = {"tool_name": tool, "tool_input": tool_input, "cwd": str(repo)}
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(repo)}
    return subprocess.run(
        [sys.executable, str(repo / hook)],
        input=json.dumps(event),
        text=True,
        capture_output=True,
        env=env,
    ).returncode


def _lock(repo: Path, pid: int) -> None:
    (repo / "output").mkdir(exist_ok=True)
    (repo / "output/.run-lock").write_text(json.dumps({"pid": pid, "command": "test run"}))


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


# guard_run_lock


def test_run_lock_quiet_without_lock(repo):
    assert _hook(repo, RUN_LOCK, "Write", {"file_path": str(repo / "new.py")}) == 0


def test_run_lock_blocks_tree_write_while_held(repo):
    _lock(repo, os.getpid())
    assert _hook(repo, RUN_LOCK, "Write", {"file_path": str(repo / "new.py")}) == 2
    assert _hook(repo, RUN_LOCK, "Edit", {"file_path": "README.md"}) == 2


def test_run_lock_allows_ignored_and_outside_paths(repo, tmp_path):
    _lock(repo, os.getpid())
    assert _hook(repo, RUN_LOCK, "Write", {"file_path": str(repo / "output/notes.md")}) == 0
    assert _hook(repo, RUN_LOCK, "Write", {"file_path": str(tmp_path / "elsewhere.py")}) == 0


def test_run_lock_ignores_stale_lock(repo):
    _lock(repo, _dead_pid())
    assert _hook(repo, RUN_LOCK, "Write", {"file_path": str(repo / "new.py")}) == 0


def test_run_lock_blocks_tree_changing_git_only(repo):
    _lock(repo, os.getpid())
    assert _hook(repo, RUN_LOCK, "Bash", {"command": "git add README.md"}) == 2
    assert _hook(repo, RUN_LOCK, "Bash", {"command": "git status && ls"}) == 0


# guard_local_suites


def _stamp(repo: Path, suite: str, exit_code: int) -> None:
    sys.path.insert(0, str(repo / "scripts/dev"))
    try:
        from stamp import SUITES as defined
        from stamp import worktree_fingerprint

        fingerprint = worktree_fingerprint(repo, defined[suite].covered)
    finally:
        sys.path.pop(0)
        sys.modules.pop("stamp", None)
    (repo / "output").mkdir(exist_ok=True)
    (repo / f"output/.stamp-{suite}.json").write_text(
        json.dumps({"fingerprint": fingerprint, "exit_code": exit_code, "summary": "s"})
    )


def _commit(repo: Path) -> int:
    return _hook(repo, SUITES, "Bash", {"command": "git commit -m 'm'"})


def _change(repo: Path, rel: str, text: str) -> None:
    (repo / rel).write_text(text)
    _git(repo, "add", rel)


def test_suites_quiet_on_other_commands_and_uncovered_commits(repo):
    assert _hook(repo, SUITES, "Bash", {"command": "git status"}) == 0
    _change(repo, "README.md", "changed\n")
    assert _commit(repo) == 0


@pytest.mark.parametrize(
    "command",
    [
        "git commit -am 'x'",
        "git commit -a -m x",
        "git -C . commit --all -m x",
        "cd . && git commit -am x",
        "/usr/bin/git commit -a",
    ],
)
def test_suites_block_commit_all(repo, command):
    assert _hook(repo, SUITES, "Bash", {"command": command}) == 2


# Each of these once blocked, or would have: the -a check read the whole
# command, so a message quoting `git commit -a`, or `ls -la` chained before a
# commit, looked like -a.
@pytest.mark.parametrize(
    "command",
    [
        "git commit -m 'refuses git commit -a'",
        "git commit -F - <<'EOF'\nBlocks `git commit -a` now\nEOF",
        "git commit -m \"$(cat <<'EOF'\nRefuses git commit -a\nEOF\n)\"",
        "ls -la && git commit -m x",
        "git commit -m -a",
        "echo 'git commit -a'",
    ],
)
def test_suites_do_not_read_messages_or_other_commands_as_commit_all(repo, command):
    assert _hook(repo, SUITES, "Bash", {"command": command}) == 0


@pytest.mark.parametrize(
    ("suite", "rel"), [("realdata", "src/vwf/metrics.py"), ("pinn", "src/vwf/pinn/model.py")]
)
def test_suite_blocks_covered_commit_without_stamp(repo, suite, rel):
    _change(repo, rel, "changed = 2\n")
    assert _commit(repo) == 2


@pytest.mark.parametrize(
    ("suite", "rel"), [("realdata", "src/vwf/metrics.py"), ("pinn", "src/vwf/pinn/model.py")]
)
def test_suite_allows_matching_passing_stamp(repo, suite, rel):
    (repo / rel).write_text("changed = 2\n")
    _stamp(repo, suite, exit_code=0)
    _git(repo, "add", rel)
    assert _commit(repo) == 0


def test_suite_checks_a_commit_the_parser_cannot_see(repo):
    _change(repo, "src/vwf/metrics.py", "x = 2\n")
    assert _hook(repo, SUITES, "Bash", {"command": "sh -c 'git commit -m m'"}) == 2


def test_suite_blocks_failed_or_stale_stamp(repo):
    (repo / "src/vwf/pinn/model.py").write_text("y = 2\n")
    _stamp(repo, "pinn", exit_code=1)
    _git(repo, "add", "src/vwf/pinn/model.py")
    assert _commit(repo) == 2
    _stamp(repo, "pinn", exit_code=0)
    _change(repo, "src/vwf/pinn/model.py", "y = 3\n")
    assert _commit(repo) == 2


# The shape that let a commit through on 2026-09-25 with a failing realdata
# stamp: the paths were staged by the same command as the commit, after the
# hook had read an index with nothing covered in it.
@pytest.mark.parametrize(
    "command",
    [
        "git add src/vwf/metrics.py && git commit -F - <<'EOF'\nmsg\nEOF",
        "git add src/vwf/pinn/model.py; git commit -m x",
        "git add -A && git commit -m x",
        "git add . && git commit -m x",
        "git add src && git commit -m x",
        "git add 'src/vwf/*.py' && git commit -m x",
        "git rm src/vwf/metrics.py && git commit -m x",
        "git commit src/vwf/metrics.py -m x",
        "git commit -m x -- src/vwf/pinn/model.py",
    ],
)
def test_suite_blocks_covered_code_staged_by_the_commit_command(repo, command):
    (repo / "src/vwf/metrics.py").write_text("x = 2\n")
    (repo / "src/vwf/pinn/model.py").write_text("y = 2\n")
    _stamp(repo, "realdata", exit_code=1)
    assert _hook(repo, SUITES, "Bash", {"command": command}) == 2


@pytest.mark.parametrize(
    "command",
    [
        "git add README.md && git commit -m x",
        "git add docs/x.md tests/test_x.py && git commit -m x",
        "git commit README.md -m x",
        "git add src/vwf/metrics.py",
        "git commit -m 'stage with git add src/vwf/metrics.py first'",
        "git commit -F - <<'EOF'\ngit add -A && git commit\nEOF",
    ],
)
def test_suite_allows_staging_that_reaches_no_covered_code(repo, command):
    assert _hook(repo, SUITES, "Bash", {"command": command}) == 0


def test_one_suite_stamp_does_not_cover_the_other(repo):
    (repo / "src/vwf/pinn/model.py").write_text("y = 2\n")
    _stamp(repo, "pinn", exit_code=0)
    _git(repo, "add", "src/vwf/pinn/model.py")
    _change(repo, "src/vwf/metrics.py", "x = 2\n")
    assert _commit(repo) == 2


# The suite definitions against the real repository


def _defined_suites():
    sys.path.insert(0, str(ROOT / "scripts/dev"))
    try:
        from stamp import SUITES as defined
    finally:
        sys.path.pop(0)
        sys.modules.pop("stamp", None)
    return defined


def test_every_pinn_test_file_is_in_the_pinn_suite():
    listed = {Path(a).name for a in _defined_suites()["pinn"].pytest_args}
    on_disk = {p.name for p in (ROOT / "tests").glob("test_pinn_*.py")}
    assert listed == on_disk


def test_covered_paths_exist_and_agree_with_the_written_rules():
    suites = _defined_suites()
    for suite in suites.values():
        for rel in suite.covered:
            assert (ROOT / rel).exists(), rel
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    code_rules = (ROOT / "src/vwf/AGENTS.md").read_text(encoding="utf-8")
    for rel in suites["realdata"].covered:
        short = rel.removeprefix("src/vwf/")
        assert f"vwf/{short}" in contributing, rel
        assert f"`{short}`" in code_rules, rel
