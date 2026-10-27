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
import re
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
    (repo / "src/pyvwf").mkdir(parents=True)
    (repo / "src/pyvwf/metrics.py").write_text("x = 1\n")
    (repo / "src/pyvwf/pinn").mkdir()
    (repo / "src/pyvwf/pinn/model.py").write_text("y = 1\n")
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


@pytest.mark.parametrize(
    "command",
    [
        "echo x > new.py",
        "printf x >> README.md",
        "cat <<'EOF' > src/pyvwf/x.py\nbody\nEOF",
        "echo y | tee -a README.md",
        "sed -i '' s/a/b/ README.md",
        "sed -i -e s/a/b/ README.md",
        "ls &> notes.md",
    ],
)
def test_run_lock_blocks_shell_writes_to_the_tree(repo, command):
    _lock(repo, os.getpid())
    assert _hook(repo, RUN_LOCK, "Bash", {"command": command}) == 2


@pytest.mark.parametrize(
    "command",
    [
        "echo x > output/log.txt",
        "ls > /dev/null 2>&1",
        "cat <<'EOF'\na > not_a_target.py\nEOF",
        "sed s/a/b/ README.md",
        "sed -i -e s/a/b/ output/x.md",
        "grep -n x README.md | tee output/hits.txt",
        "echo x > $SCRATCH/x.md",
    ],
)
def test_run_lock_allows_shell_writes_outside_the_tree(repo, command):
    _lock(repo, os.getpid())
    assert _hook(repo, RUN_LOCK, "Bash", {"command": command}) == 0


def test_run_lock_allows_shell_writes_outside_the_repository(repo, tmp_path):
    _lock(repo, os.getpid())
    assert _hook(repo, RUN_LOCK, "Bash", {"command": f"echo x > {tmp_path / 'x.py'}"}) == 0


def test_run_lock_lets_shell_writes_through_without_a_lock(repo):
    assert _hook(repo, RUN_LOCK, "Bash", {"command": "echo x > new.py"}) == 0


# guard_docs_critic: the docs-critic agent's Bash is read-only git and wc

DOCS_CRITIC = ".claude/hooks/guard_docs_critic.py"


@pytest.mark.parametrize(
    "command",
    [
        "git diff",
        "git diff --stat HEAD~1",
        "git show 8105277:README.md",
        "git log --oneline -5",
        "git diff | wc -l",
        "git diff && git log -1",
    ],
)
def test_docs_critic_allows_read_only_git_and_wc(repo, command):
    assert _hook(repo, DOCS_CRITIC, "Bash", {"command": command}) == 0


@pytest.mark.parametrize(
    "command",
    [
        "git diff --output=x.md",
        "git show HEAD --output /tmp/a",
        "git -C . diff",
        "git commit -m x",
        "git diff > x.md",
        "cat README.md",
        "git diff; touch y",
        "echo $(git diff)",
        "git diff 'unterminated",
    ],
)
def test_docs_critic_blocks_everything_else(repo, command):
    assert _hook(repo, DOCS_CRITIC, "Bash", {"command": command}) == 2


def _deny_matches(rule: str, command: str) -> bool:
    """A Bash deny rule against a command, by the documented wildcard rule.

    https://code.claude.com/docs/en/permissions: a `*` in a Bash rule "matches
    any text, including spaces". This checks the rules as written against that
    rule. It does not run Claude Code's own matcher.
    """
    inner = rule.removeprefix("Bash(").removesuffix(")")
    return re.fullmatch(".*".join(map(re.escape, inner.split("*"))), command) is not None


def test_output_deny_rules_fire_on_output_and_not_on_a_plain_read():
    deny = json.loads((ROOT / ".claude/settings.json").read_text())["permissions"]["deny"]
    rules = [r for r in deny if "--output" in r]
    assert len(rules) == 3
    for command in (
        "git diff --output=x.md",
        "git diff HEAD~1 --output x.md",
        "git show HEAD --output=/tmp/a",
        "git log -p --output=/tmp/b",
    ):
        assert any(_deny_matches(r, command) for r in rules), command
    for command in ("git diff", "git diff --stat", "git show HEAD", "git log --oneline -5"):
        assert not any(_deny_matches(r, command) for r in deny), command
    # The matcher reads the existing rules the way they were meant, both ways.
    assert any(_deny_matches(r, "git push origin main") for r in deny)
    assert not any(_deny_matches(r, "git push -u origin topic") for r in deny)


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
    ("suite", "rel"), [("realdata", "src/pyvwf/metrics.py"), ("pinn", "src/pyvwf/pinn/model.py")]
)
def test_suite_blocks_covered_commit_without_stamp(repo, suite, rel):
    _change(repo, rel, "changed = 2\n")
    assert _commit(repo) == 2


@pytest.mark.parametrize(
    ("suite", "rel"), [("realdata", "src/pyvwf/metrics.py"), ("pinn", "src/pyvwf/pinn/model.py")]
)
def test_suite_allows_matching_passing_stamp(repo, suite, rel):
    (repo / rel).write_text("changed = 2\n")
    _stamp(repo, suite, exit_code=0)
    _git(repo, "add", rel)
    assert _commit(repo) == 0


def test_suite_checks_a_commit_the_parser_cannot_see(repo):
    _change(repo, "src/pyvwf/metrics.py", "x = 2\n")
    assert _hook(repo, SUITES, "Bash", {"command": "sh -c 'git commit -m m'"}) == 2


def test_suite_blocks_failed_or_stale_stamp(repo):
    (repo / "src/pyvwf/pinn/model.py").write_text("y = 2\n")
    _stamp(repo, "pinn", exit_code=1)
    _git(repo, "add", "src/pyvwf/pinn/model.py")
    assert _commit(repo) == 2
    _stamp(repo, "pinn", exit_code=0)
    _change(repo, "src/pyvwf/pinn/model.py", "y = 3\n")
    assert _commit(repo) == 2


# The shape that let a commit through on 2026-09-25 with a failing realdata
# stamp: the paths were staged by the same command as the commit, after the
# hook had read an index with nothing covered in it.
@pytest.mark.parametrize(
    "command",
    [
        "git add src/pyvwf/metrics.py && git commit -F - <<'EOF'\nmsg\nEOF",
        "git add src/pyvwf/pinn/model.py; git commit -m x",
        "git add -A && git commit -m x",
        "git add . && git commit -m x",
        "git add src && git commit -m x",
        "git add 'src/pyvwf/*.py' && git commit -m x",
        "git rm src/pyvwf/metrics.py && git commit -m x",
        "git commit src/pyvwf/metrics.py -m x",
        "git commit -m x -- src/pyvwf/pinn/model.py",
    ],
)
def test_suite_blocks_covered_code_staged_by_the_commit_command(repo, command):
    (repo / "src/pyvwf/metrics.py").write_text("x = 2\n")
    (repo / "src/pyvwf/pinn/model.py").write_text("y = 2\n")
    _stamp(repo, "realdata", exit_code=1)
    assert _hook(repo, SUITES, "Bash", {"command": command}) == 2


@pytest.mark.parametrize(
    "command",
    [
        "git add README.md && git commit -m x",
        "git add docs/x.md tests/test_x.py && git commit -m x",
        "git commit README.md -m x",
        "git add src/pyvwf/metrics.py",
        "git commit -m 'stage with git add src/pyvwf/metrics.py first'",
        "git commit -F - <<'EOF'\ngit add -A && git commit\nEOF",
    ],
)
def test_suite_allows_staging_that_reaches_no_covered_code(repo, command):
    assert _hook(repo, SUITES, "Bash", {"command": command}) == 0


def test_one_suite_stamp_does_not_cover_the_other(repo):
    (repo / "src/pyvwf/pinn/model.py").write_text("y = 2\n")
    _stamp(repo, "pinn", exit_code=0)
    _git(repo, "add", "src/pyvwf/pinn/model.py")
    _change(repo, "src/pyvwf/metrics.py", "x = 2\n")
    assert _commit(repo) == 2


# The fingerprint ignores comments and docstrings, and nothing else

CODE = '''"""Module docstring."""
import numpy as np  # type: ignore[import]


class Fit:
    """Class docstring."""

    def scale(self, x: float, k: float = 1.5) -> float:
        """Function docstring."""
        return np.sqrt(x) * k  # a comment
'''

#: Each edit leaves the code as the interpreter runs it unchanged.
SAME_CODE = [
    CODE.replace('"""Module docstring."""', '"""Edited, with a |sub| substitution."""'),
    CODE.replace('    """Class docstring."""\n', ""),
    CODE.replace("Function docstring.", "Function docstring.\n\n        More.\n        "),
    CODE.replace("# type: ignore[import]", "# type: ignore[import, unused-ignore]"),
    CODE.replace("  # a comment", "").replace("class Fit:", "#: comment\n\n\nclass Fit:"),
]

#: Each edit can change what the code computes.
OTHER_CODE = [
    CODE.replace("1.5", "1.6"),
    CODE.replace("* k", "/ k"),
    CODE.replace("k: float = 1.5", "k: float = 1.5, j: float = 0.0"),
    CODE.replace('"""Function docstring."""', '"""Function docstring."""\n        x = x + 0'),
]


def _fingerprints(repo: Path, rel: str = "src/pyvwf/metrics.py") -> tuple[str, str]:
    """(worktree, index) fingerprints of the realdata suite, after staging ``rel``."""
    _git(repo, "add", rel)
    sys.path.insert(0, str(repo / "scripts/dev"))
    try:
        from stamp import SUITES as defined
        from stamp import index_fingerprint, worktree_fingerprint

        covered = defined["realdata"].covered
        return worktree_fingerprint(repo, covered), index_fingerprint(repo, covered)
    finally:
        sys.path.pop(0)
        sys.modules.pop("stamp", None)


@pytest.fixture
def code_repo(repo):
    (repo / "src/pyvwf/metrics.py").write_text(CODE)
    _git(repo, "add", "src/pyvwf/metrics.py")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "code")
    return repo


@pytest.mark.parametrize(
    "edited", SAME_CODE, ids=["module-doc", "class-doc", "function-doc", "type-ignore", "comments"]
)
def test_fingerprint_ignores_comments_and_docstrings(code_repo, edited):
    worktree, index = _fingerprints(code_repo)
    assert worktree == index
    (code_repo / "src/pyvwf/metrics.py").write_text(edited)
    assert _fingerprints(code_repo) == (worktree, worktree)


@pytest.mark.parametrize(
    "edited", OTHER_CODE, ids=["constant", "operator", "argument", "statement"]
)
def test_fingerprint_moves_on_any_code_change(code_repo, edited):
    before, _ = _fingerprints(code_repo)
    (code_repo / "src/pyvwf/metrics.py").write_text(edited)
    worktree, index = _fingerprints(code_repo)
    assert worktree == index != before


def test_fingerprint_ignores_whitespace_inside_a_line(code_repo):
    before, _ = _fingerprints(code_repo)
    (code_repo / "src/pyvwf/metrics.py").write_text(
        CODE.replace("np.sqrt(x) * k", "np.sqrt( x )*k")
    )
    assert _fingerprints(code_repo) == (before, before)


def test_fingerprint_moves_on_a_file_that_is_not_python(code_repo):
    (code_repo / "src/pyvwf/pinn/table.csv").write_text("a,b\n1,2\n")
    _git(code_repo, "add", "src/pyvwf/pinn/table.csv")
    sys.path.insert(0, str(code_repo / "scripts/dev"))
    try:
        from stamp import SUITES as defined
        from stamp import index_fingerprint, worktree_fingerprint

        covered = defined["pinn"].covered
        before = worktree_fingerprint(code_repo, covered)
        # A comment character in a CSV is data, not a comment.
        (code_repo / "src/pyvwf/pinn/table.csv").write_text("a,b\n1,2\n# 3\n")
        _git(code_repo, "add", "src/pyvwf/pinn/table.csv")
        after = worktree_fingerprint(code_repo, covered)
        assert after == index_fingerprint(code_repo, covered) != before
    finally:
        sys.path.pop(0)
        sys.modules.pop("stamp", None)


def test_fingerprint_of_a_file_that_does_not_parse_is_its_bytes(code_repo):
    (code_repo / "src/pyvwf/metrics.py").write_text("x = (\n")
    before, index = _fingerprints(code_repo)
    assert before == index
    # Would be a no-op edit to code that parsed; here every byte counts.
    (code_repo / "src/pyvwf/metrics.py").write_text("x = (  # comment\n")
    worktree, index = _fingerprints(code_repo)
    assert worktree == index != before


def test_comment_only_commit_keeps_the_stamp_valid(code_repo):
    _stamp(code_repo, "realdata", exit_code=0)
    _change(code_repo, "src/pyvwf/metrics.py", SAME_CODE[3])
    assert _commit(code_repo) == 0
    _change(code_repo, "src/pyvwf/metrics.py", OTHER_CODE[0])
    assert _commit(code_repo) == 2


def test_code_digest_is_the_same_on_every_python():
    """The hook runs under whatever `python3` is first on PATH, and the stamp
    under the project's environment, so the digest cannot depend on the version.
    `ast.dump` does: its text changed in 3.13. This value was computed under
    3.9 to 3.14, and the test matrix checks it under each of its versions."""
    sys.path.insert(0, str(ROOT / "scripts/dev"))
    try:
        from stamp import _code_digest
    finally:
        sys.path.pop(0)
        sys.modules.pop("stamp", None)
    assert _code_digest(EVERY_CONSTRUCT.encode()) == EVERY_CONSTRUCT_DIGEST


#: Syntax from 3.9 onward, so the digest can be checked on every supported version.
EVERY_CONSTRUCT = (
    CODE
    + """
@dataclass(frozen=True)
class Row(Base, metaclass=Meta):
    name: str = "x"
    values: list[int] = field(default_factory=list)

    async def fetch(self, *args, key=None, **kwargs):
        async with session() as s:
            async for item in s:
                yield await item


def body(a, /, b, *, c=-1.0, d=1e-12, e=None, f=b"x", g=..., h=3j):
    global TOTAL
    try:
        with open(a) as fh, open(b) as gh:
            rows = [r for r in fh if r if not r.startswith("#")]
    except (OSError, ValueError) as err:
        raise RuntimeError(f"{a!r:>10} {b=}") from err
    else:
        rows = {k: v for k, v in zip(rows, rows[1:])}
    finally:
        del a
    lam = lambda x, y=2: x ** y if x else -x
    s = rows[1:-1:2], rows[..., None], {1, 2}, (yield), not b, a is not b
    while (n := len(rows)) > 0 and n in rows:
        n -= 1
        continue
    for i, (j, *k) in enumerate(rows):
        break
    assert c < d <= e, "message"
    return sum(x async for x in s) if c else ~1 << 2 >> 3 & 4 | 5 ^ 6 @ m
"""
)

EVERY_CONSTRUCT_DIGEST = "b8b56366da08ee755ee83bb6ca878caf64590becc7359f6bd390995e6d4fddec"


# stamp.py --workers: opt-in, capped at what each suite was measured at


def _stamp_module():
    from support import load_script

    return load_script(ROOT / "scripts/dev/stamp.py", "stamp_under_test")


@pytest.mark.parametrize(("suite", "workers"), [("realdata", 0), ("realdata", 4), ("pinn", 2)])
def test_stamp_refuses_an_unmeasured_worker_count(suite, workers, monkeypatch):
    module = _stamp_module()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("ran pytest"))
    assert module.main(suite, workers) == 2


@pytest.mark.parametrize(("workers", "expected"), [(1, []), (2, ["-n", "2", "--dist", "loadfile"])])
def test_stamp_passes_workers_to_pytest_and_records_them(workers, expected, tmp_path, monkeypatch):
    module = _stamp_module()
    commands = []

    class FakeRun:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.stdout = iter(["===== 43 passed in 1.00s =====\n"])

        def wait(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", FakeRun)
    monkeypatch.setattr(module, "_git", lambda root, *args: f"{tmp_path}\n")
    monkeypatch.setattr(module, "worktree_fingerprint", lambda root, covered: "f")
    assert module.main("realdata", workers) == 0
    (command,) = commands
    assert command[3:-2] == ["-rs", *expected]
    stamp = json.loads((tmp_path / "output/.stamp-realdata.json").read_text())
    assert stamp["workers"] == workers


#: The tail of a realdata run under `-q`: the result line has no `=` around it,
#: so the last `=` line is the warnings header. The 2026-10-16 stamp recorded it.
QUIET_TAIL = [
    "........................................... [100%]\n",
    "=============================== warnings summary ===============================\n",
    "tests/test_x.py::test_y\n",
    "  FutureWarning: 'H' is deprecated\n",
    "\n",
    "-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html\n",
    "43 passed, 1232 deselected, 33 warnings in 727.67s (0:12:07)\n",
]


@pytest.mark.parametrize(
    ("output", "summary"),
    [
        (QUIET_TAIL, "43 passed, 1232 deselected, 33 warnings in 727.67s (0:12:07)"),
        (
            ["4 workers [43 items]\n", "===== 1 failed, 42 passed in 9.12s =====\n"],
            "1 failed, 42 passed in 9.12s",
        ),
        (["===== no tests ran in 0.01s =====\n"], "no tests ran in 0.01s"),
        (["ERROR: usage: pytest [options]\n", "= warnings summary =\n"], ""),
    ],
    ids=["quiet-after-warnings", "decorated", "no-tests", "no-result"],
)
def test_stamp_records_pytests_result_line(output, summary, tmp_path, monkeypatch):
    module = _stamp_module()

    class FakeRun:
        def __init__(self, command, **kwargs):
            self.stdout = iter(output)

        def wait(self):
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", FakeRun)
    monkeypatch.setattr(module, "_git", lambda root, *args: f"{tmp_path}\n")
    monkeypatch.setattr(module, "worktree_fingerprint", lambda root, covered: "f")
    assert module.main("realdata") == 0
    stamp = json.loads((tmp_path / "output/.stamp-realdata.json").read_text())
    assert stamp["summary"] == summary


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


#: What the realdata suite leaves out of src/pyvwf, by the written rule
#: ("everything except pinn/, viz/ and cli/"), plus files that are not code
#: the pins can reach.
NOT_REALDATA = {"pinn", "viz", "cli", "__init__.py", "_version.py", "resources", "py.typed"}


def test_realdata_covers_every_module_the_rule_says_it_covers():
    """A module split out of a covered one is covered too.

    pyvwf.country_level and pyvwf.sampling were split out of data.py and
    clustering.py on 2026-09-25 and not added, so a change to either needed no
    stamp until this test.
    """
    covered = {
        rel.removeprefix("src/pyvwf/").rstrip("/") for rel in _defined_suites()["realdata"].covered
    }
    modules = {
        p.name
        for p in (ROOT / "src/pyvwf").iterdir()
        if (p.suffix == ".py" or (p.is_dir() and (p / "__init__.py").exists()))
    }
    assert sorted(modules - covered - NOT_REALDATA) == []


def test_covered_paths_exist_and_agree_with_the_written_rules():
    suites = _defined_suites()
    for suite in suites.values():
        for rel in suite.covered:
            assert (ROOT / rel).exists(), rel
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    code_rules = (ROOT / "src/pyvwf/AGENTS.md").read_text(encoding="utf-8")
    # src/pyvwf/AGENTS.md states the rule NOT_REALDATA encodes, not the list.
    assert "every module except `pinn/`, `viz/` and `cli/`" in code_rules
    for rel in suites["realdata"].covered:
        short = rel.removeprefix("src/pyvwf/")
        assert f"pyvwf/{short}" in contributing, rel
