"""What may be committed: nothing the repository's own ignore rules exclude.

``.gitignore`` is where the never-commit rules live, with the reasons in its
comment blocks: input data, the licensed curve library, local working notes.
But ignore rules only stop ``git add``; a forced add, or a file tracked before
its rule existed, bypasses them silently. This checks the tracked tree against
the rules instead.

A small allowlist is force-added on purpose: the open library and the region
shapes under ``input/reference/``, which ``/input`` would otherwise exclude. A
new tracked file that the rules exclude fails here until it is either removed
or added to the allowlist with a reason.
"""

from __future__ import annotations

import hashlib
import subprocess

import pytest
from support import ROOT


#: Tracked files the ignore rules exclude, deliberately. Each is redistributable:
#: the region shapes, force-added under the otherwise git-ignored input/.
ALLOWED_DESPITE_IGNORE = {
    "input/README.md",
    "input/reference/shapes/country_shapes.geojson",
    "input/reference/shapes/offshore_shapes.geojson",
}


#: sha256 of the open library, as committed in pyvwf/resources/, its one copy. A
#: filename check cannot see a licensed library copied over power_curves.csv. A
#: deliberate library update therefore means editing these hashes, in the same
#: commit, which is the kind of change that should be explicit.
OPEN_LIBRARY_SHA256 = {
    "power_curves.csv": "56314f390d9bc2135125ea1b0e7a4d400503a6e3911d8513eafbc68f1fa8bb50",
    "models.csv": "a33a6c25af42fcf9d087353596b2b4b452e91272419b8ae466eb3d2a2a75f7da",
    "power_curves_provenance.csv": "3fccb9ec32ac0b01cb9f9f96cc0aa866d69538afadf80d830937f372957ccadb",
}


def _git(*args: str) -> list[str]:
    """Run git in the repository, with any user-level ignore file disabled,
    so the answer is the same on every machine and in CI."""
    try:
        out = subprocess.run(
            ["git", "-c", "core.excludesFile=", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pytest.skip("not running from a git checkout")
    return [line for line in out.splitlines() if line]


def test_no_tracked_file_is_excluded_by_the_ignore_rules():
    excluded = set(_git("ls-files", "--cached", "--ignored", "--exclude-standard"))
    unexpected = sorted(excluded - ALLOWED_DESPITE_IGNORE)
    assert not unexpected, (
        "tracked files that .gitignore excludes; remove them from the index, "
        f"or allowlist them here with a reason: {unexpected}"
    )


def test_no_licensed_curve_file_is_tracked():
    """The licensed library committed under its own file name. Copying it over
    the open library's file name is caught by the sha256 test below instead."""
    tracked = _git("ls-files")
    assert not [p for p in tracked if p.endswith(".real.csv")]


@pytest.mark.parametrize("name", sorted(OPEN_LIBRARY_SHA256))
def test_open_library_content_is_the_recorded_one(name):
    path = ROOT / "src/pyvwf/resources" / name
    if not path.is_file():
        pytest.skip(f"{name} not present outside a repository checkout")
    # Normalise line endings, so a Windows checkout with autocrlf does not fail
    # a file whose content is unchanged.
    digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert digest == OPEN_LIBRARY_SHA256[name], (
        f"{name} is not the recorded open library. If it was replaced by "
        "another library, restore it. If the open library was deliberately "
        "updated, record the new sha256 here in the same commit."
    )


def test_the_default_root_holds_no_curve_table():
    """A curve table in input/reference/ would silently replace the open library
    for the default root, which the test suite and the open-library scorecard
    rows read. Another library goes in its own input root. Local only: input/
    is git-ignored, so in CI this passes on an absent directory."""
    found = [n for n in OPEN_LIBRARY_SHA256 if (ROOT / "input/reference" / n).exists()]
    assert not found, (
        f"input/reference/ holds {found}; move it to its own input root "
        "(docs/guides/training.md, Choose the input root)"
    )
