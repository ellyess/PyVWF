"""Count the repository's development record, as of one commit.

`docs/design/ai-native-development.md` reports these counts. They come from
this script, run against a stated commit, so that no figure there is recalled.
Every count reads git or the tracked tree at that commit, never the working
tree, so two runs at one commit agree.

    python scripts/dev/repo_stats.py            # at HEAD
    python scripts/dev/repo_stats.py --rev main

What each count is, and what it leaves out, is printed beside it.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MERGED_PR = re.compile(r"^Merge pull request #(\d+)\b", re.MULTILINE)
TEST_FUNCTION = re.compile(r"^\s*(?:async\s+)?def test_\w*\(", re.MULTILINE)
NOTICE = re.compile(r"\*\*Correction notice, \d{4}-\d{2}-\d{2}")
BRACKET = re.compile(r"\*\[Correction, \d{4}-\d{2}-\d{2}")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def _tracked(root: Path, rev: str) -> list[str]:
    return _git(root, "ls-tree", "-r", "--name-only", rev).splitlines()


def _show(root: Path, rev: str, path: str) -> str:
    return _git(root, "show", f"{rev}:{path}")


def count(root: Path, rev: str) -> list[tuple[str, int, str]]:
    """Return (label, count, what it counts) rows for the tree at `rev`."""
    files = _tracked(root, rev)
    subjects = _git(root, "log", "--format=%B%x00", rev)
    tests = [f for f in files if re.fullmatch(r"tests/(.+/)?test_[^/]*\.py", f)]
    findings = [f for f in files if re.fullmatch(r"docs/findings/[^/]+\.md", f)]
    prereg = [f for f in findings if f.endswith("-prereg.md") or f.endswith("-prespecification.md")]
    text = {f: _show(root, rev, f) for f in tests + findings + ["CHANGELOG.md"] if f in files}
    guards = "tests/test_agent_guards.py"
    return [
        (
            "commits",
            int(_git(root, "rev-list", "--count", rev)),
            "reachable from the commit, merges included",
        ),
        (
            "pull requests merged",
            len(set(MERGED_PR.findall(subjects))),
            "distinct numbers in 'Merge pull request #N' messages; a squashed or rebased PR is not counted",
        ),
        ("test files", len(tests), "tests/**/test_*.py"),
        (
            "test functions",
            sum(len(TEST_FUNCTION.findall(text[f])) for f in tests),
            "def test_ in those files, before parametrisation",
        ),
        (
            "agent guard tests",
            len(TEST_FUNCTION.findall(text.get(guards, ""))),
            f"test functions in {guards}",
        ),
        ("findings documents", len(findings), "docs/findings/*.md, the scorecard included"),
        (
            "pre-registrations",
            len(prereg),
            "docs/findings/*-prereg.md and the one *-prespecification.md",
        ),
        (
            "correction notices in findings",
            sum(len(NOTICE.findall(text[f])) for f in findings),
            "'**Correction notice, <date>' in docs/findings",
        ),
        (
            "correction notices in the CHANGELOG",
            len(BRACKET.findall(text.get("CHANGELOG.md", ""))),
            "'*[Correction, <date>' in CHANGELOG.md",
        ),
        (
            "AGENTS.md files",
            sum(Path(f).name == "AGENTS.md" for f in files),
            "agent instruction files",
        ),
        (
            "skills",
            sum(bool(re.fullmatch(r"\.claude/skills/[^/]+/SKILL\.md", f)) for f in files),
            ".claude/skills/*/SKILL.md",
        ),
        (
            "subagents",
            sum(bool(re.fullmatch(r"\.claude/agents/[^/]+\.md", f)) for f in files),
            ".claude/agents/*.md",
        ),
        (
            "hooks",
            sum(bool(re.fullmatch(r"\.claude/hooks/[^/]+\.py", f)) for f in files),
            ".claude/hooks/*.py",
        ),
    ]


def main(rev: str, root: Path = ROOT) -> str:
    """Render the counts at `rev` as a Markdown table headed by the commit and its date."""
    sha, date = _git(root, "log", "-1", "--format=%h %cs", rev).split()
    lines = [
        f"Counts at `{sha}` ({date}), from `python scripts/dev/repo_stats.py --rev {sha}`.",
        "",
        "| Count | Value | What it counts |",
        "|---|---|---|",
    ]
    lines += [f"| {label} | {value} | {what} |" for label, value, what in count(root, rev)]
    out = "\n".join(lines)
    print(out)
    return out


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rev", default="HEAD", help="the commit to count at (default HEAD)")
    args = parser.parse_args(argv)
    main(args.rev)


if __name__ == "__main__":
    cli()
