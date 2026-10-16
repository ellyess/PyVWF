"""`scripts/dev/repo_stats.py` counts what it says, and only that.

The counts it prints are quoted in `docs/design/ai-native-development.md`, so
each one is checked here on a repository built with a known number of each
thing, and on one with none of them, so a count that silently returns zero
fails as surely as one that overcounts.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from support import load_script

ROOT = Path(__file__).resolve().parents[1]
stats = load_script(ROOT / "scripts" / "dev" / "repo_stats.py", "repo_stats")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _write(repo: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)


def _counts(repo: Path) -> dict[str, int]:
    return {label: value for label, value, _ in stats.count(repo, "HEAD")}


def test_each_count_matches_a_repository_built_to_known_counts(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(
        repo,
        {
            "tests/test_a.py": "def test_one():\n    pass\n\n\ndef test_two():\n    pass\n",
            "tests/test_agent_guards.py": "def test_guard():\n    pass\n",
            "tests/support.py": "def test_not_in_a_test_file():\n    pass\n",
            "docs/findings/method-x.md": (
                "**Correction notice, 2026-01-02: one.**\n\n"
                "The correction notice of that date is not a second one.\n\n"
                "**Correction notice, 2026-01-03: two.**\n"
            ),
            "docs/findings/method-x-prereg.md": "gates\n",
            "docs/findings/method-y-prespecification.md": "gates\n",
            "CHANGELOG.md": "- a claim. *[Correction, 2026-01-04: not so.]*\n",
            "AGENTS.md": "rules\n",
            "src/AGENTS.md": "rules\n",
            ".claude/skills/s/SKILL.md": "skill\n",
            ".claude/agents/critic.md": "agent\n",
            ".claude/hooks/guard.py": "hook\n",
        },
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")
    _git(repo, "checkout", "-qb", "topic")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "work")
    _git(repo, "checkout", "-q", "-")
    _git(repo, "merge", "-q", "--no-ff", "topic", "-m", "Merge pull request #7 from o/topic")
    # PR #10 was merged with a colon after its number.
    _git(repo, "commit", "-q", "--allow-empty", "-m", "Merge pull request #10: a harness")
    # An untracked file is not counted: the counts read the commit, not the disk.
    _write(repo, {"tests/test_untracked.py": "def test_x():\n    pass\n"})

    assert _counts(repo) == {
        "commits": 4,
        "pull requests merged": 2,
        "test files": 2,
        "test functions": 3,
        "agent guard tests": 1,
        "findings documents": 3,
        "pre-registrations": 2,
        "correction notices in findings": 2,
        "correction notices in the CHANGELOG": 1,
        "AGENTS.md files": 2,
        "skills": 1,
        "subagents": 1,
        "hooks": 1,
    }


def test_a_repository_with_none_of_them_counts_zero(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(
        repo, {"README.md": "Merge pull request #3 from o/x\n**Correction notice, 2026-01-02**\n"}
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")

    counts = _counts(repo)
    assert counts.pop("commits") == 1
    assert set(counts.values()) == {0}


def test_the_table_names_the_commit_it_counted(tmp_path, capsys):
    out = stats.main("HEAD", ROOT)
    sha = subprocess.run(
        ["git", "log", "-1", "--format=%h"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out.startswith(f"Counts at `{sha}` (")
    assert f"--rev {sha}" in out
