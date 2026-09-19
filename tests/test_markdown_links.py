"""Every relative link in a tracked Markdown file resolves, anchor included.

Sphinx builds only the published pages and does not check anchors in
Markdown links, and GitHub renders the rest, so a renamed heading or a moved
guide breaks links that nothing reports. This test resolves each relative
link in every tracked ``.md`` file: the target file must exist, and a
``#anchor`` must match a heading of the target, slugged as GitHub slugs it.
External links are not checked.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"```.*?```", re.S)
LINK = re.compile(r"\]\(([^)\s]+)\)")
HEADING = re.compile(r"^#+\s+(.*)$", re.M)


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces to hyphens."""
    text = heading.strip().lower().replace("`", "")
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def headings(path: Path) -> set[str]:
    return {slug(h) for h in HEADING.findall(FENCE.sub("", path.read_text(errors="ignore")))}


def broken_links(files: list[Path]) -> list[str]:
    """Each relative link in ``files`` whose target file or heading is missing."""
    problems = []
    for source in files:
        text = FENCE.sub("", source.read_text(errors="ignore"))
        for target in LINK.findall(text):
            if re.match(r"[a-z][a-z0-9+.-]*:", target):
                continue  # http:, https:, mailto: and the like
            path, _, anchor = target.partition("#")
            dest = (source.parent / path) if path else source
            if not dest.exists():
                problems.append(f"{source}: no file {target}")
            elif anchor and dest.suffix == ".md" and anchor not in headings(dest):
                problems.append(f"{source}: no heading #{anchor} in {dest.name}")
    return problems


def tracked_markdown() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "*.md"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not running from a git checkout")
    return [ROOT / f for f in out.split()]


def test_every_tracked_markdown_link_resolves():
    assert broken_links(tracked_markdown()) == []


def test_the_checker_finds_a_missing_file_and_a_missing_heading(tmp_path):
    (tmp_path / "target.md").write_text("# A heading\n\n## Choose the `input` root\n")
    source = tmp_path / "source.md"
    source.write_text(
        "[ok](target.md#a-heading) [ok](target.md#choose-the-input-root) "
        "[ok](https://example.org/x#y) [self](#local)\n\n# Local\n"
        "[bad file](missing.md) [bad heading](target.md#no-such-heading)\n"
        "```\n[inside a fence](ignored.md)\n```\n"
    )
    assert broken_links([source]) == [
        f"{source}: no file missing.md",
        f"{source}: no heading #no-such-heading in target.md",
    ]
