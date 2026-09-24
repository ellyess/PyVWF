"""CI's ruff and the pre-commit hook's ruff are the same version.

CI installs ruff from the ``dev`` extra and runs ``ruff format --check``;
the pre-commit hook runs the version its rev names. Two ruff releases can
format the same file differently (0.12 and 0.16 do), so if the two drift, a
file the hook formatted can fail CI with no code change.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10: tomllib is not in the standard library yet.
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


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
