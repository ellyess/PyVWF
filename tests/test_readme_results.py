"""The README's results table is a copy of the scorecard's, and must agree with it.

``docs/scorecard.md`` is the home of every per-region result
(``docs/README.md``, "One home per fact"). The README carries a short copy so a
reader meets the numbers before the caveats, and a copy drifts: a re-run
updates the scorecard and leaves the README behind. This reads both tables and
requires the same regions, the same uncorrected and corrected RMSE, the same
train and test years for the turbine-level rows, and the same markers.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = re.compile(r"\(([A-Z]{2}(?:-[A-Z]+)?)\)")
NUMBER = re.compile(r"\d+\.\d+")


def _tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Every Markdown table in ``text``, as (header cells, row cells)."""
    tables, block = [], []
    for line in [*text.splitlines(), ""]:
        if line.startswith("|"):
            block.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif block:
            if len(block) > 2:
                tables.append((block[0], block[2:]))
            block = []
    return tables


def _one(tables, wanted: set[str], where: str):
    """The single table whose header contains every column in ``wanted``."""
    found = [t for t in tables if wanted <= set(t[0])]
    assert len(found) == 1, f"{where}: expected one table with {sorted(wanted)}, found {len(found)}"
    header, rows = found[0]
    return [dict(zip(header, row)) for row in rows]


def _number(cell: str) -> str:
    match = NUMBER.search(cell)
    assert match, f"no number in {cell!r}"
    return match.group(0)


def _code(cell: str) -> str:
    match = CODE.search(cell)
    assert match, f"no region code in {cell!r}"
    return match.group(1)


def _scorecard() -> dict[str, dict]:
    tables = _tables((ROOT / "docs/scorecard.md").read_text())
    rows = {}
    turbine = _one(tables, {"Region", "Fleet (test)", "Uncorr RMSE", "Corr RMSE"}, "scorecard")
    for r in turbine:
        rows[_code(r["Region"])] = {
            "level": "turbine",
            "train_test": r["Train → test"],
            "uncorrected": _number(r["Uncorr RMSE"]),
            "corrected": _number(r["Corr RMSE"]),
            "markers": {
                m
                for m, cell in (("†", r["Best cfg"]), ("‡", r["Corr RMSE"]), ("§", r["Region"]))
                if m in cell
            },
        }
    country = _one(tables, {"Region", "Uncorr RMSE", "Corr RMSE", "Substituted"}, "scorecard")
    for r in country:
        rows[_code(r["Region"])] = {
            "level": "country",
            "train_test": None,
            "uncorrected": _number(r["Uncorr RMSE"]),
            "corrected": _number(r["Corr RMSE"]),
            "markers": {m for m in "†‡§" if m in r["Best cfg"] + r["Corr RMSE"] + r["Region"]},
        }
    return rows


def _readme() -> dict[str, dict]:
    tables = _tables((ROOT / "README.md").read_text())
    rows = {}
    for r in _one(tables, {"Region", "Level", "Uncorrected", "Corrected"}, "README"):
        rows[_code(r["Region"])] = {
            "level": r["Level"],
            "train_test": r["Train → test"],
            "uncorrected": _number(r["Uncorrected"]),
            "corrected": _number(r["Corrected"]),
            "markers": {m for m in "†‡§" if m in r["Corrected"]},
        }
    return rows


SCORECARD = _scorecard()
README = _readme()


def test_the_readme_lists_exactly_the_scorecard_regions():
    assert set(README) == set(SCORECARD)


@pytest.mark.parametrize("code", sorted(SCORECARD))
def test_each_readme_row_matches_the_scorecard(code):
    want, got = SCORECARD[code], README[code]
    assert got["level"] == want["level"]
    assert (got["uncorrected"], got["corrected"]) == (want["uncorrected"], want["corrected"])
    assert got["markers"] == want["markers"]
    if want["train_test"] is not None:
        assert got["train_test"] == want["train_test"]


def test_a_changed_scorecard_number_is_caught(tmp_path, monkeypatch):
    """The check fires: move one scorecard figure and the README no longer agrees."""
    text = (ROOT / "docs/scorecard.md").read_text()
    moved = text.replace("| 0.086 | **0.057** |", "| 0.086 | **0.058** |", 1)
    assert moved != text, "the Germany row was not found to move"
    (tmp_path / "docs").mkdir(parents=True)
    (tmp_path / "docs/scorecard.md").write_text(moved)
    monkeypatch.setattr(sys.modules[__name__], "ROOT", tmp_path)
    assert _scorecard()["DE"]["corrected"] != README["DE"]["corrected"]
