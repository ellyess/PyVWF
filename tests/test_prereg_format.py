"""Every preregistration names its years, its gates and when it was registered.

``AGENTS.md`` requires every result to state its training years and its single
test year, and a preregistration is where a result's years are fixed. This
checks each ``docs/findings/*-prereg.md`` for four fields, in the forms those
documents use today:

- **registration**: a ``**Date:**`` line with an ISO date, whose paragraph says
  the design was fixed *before* something (a run, a fold, a download);
- **training years**: "Training years 2015-21", "Training years: DK 2015 to
  2019", "trained 2015 to 2019", or a table with a "Training years" column
  holding a span in every row, or "No training years are involved" for a study
  that fits nothing;
- **test year**: "test year 2023", "tested on 2020", a table with a "Test year"
  column holding one year in every row, or "No test year is involved" for a
  pass that reads training fits only;
- **gates**: a non-empty ``## Gates`` (or ``## Gate``) section.

Eight documents state their years by reference ("its training years and its
test year", "the same test years") rather than naming them, and one registers
readings instead of gates. The evaluation of the review agents
(``agent-critics-prereg.md``) fits no data, so it has no years, and calls its
gate "the bar". They are listed in ``PREDATES_FORM`` with the
reason, field by field. A registered design is amended only by a dated
amendment, so they are listed rather than edited. The list is exact: a new
document missing a field fails, and so does an entry whose field has since been
added, so the list can only shrink.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FINDINGS = ROOT / "docs" / "findings"
PREREGS = sorted(FINDINGS.glob("*-prereg.md"))

FIELDS = ("registration", "training_years", "test_year", "gates")

_BY_REFERENCE = "states its years by reference to {}, without naming them"

#: Documents missing a field, with the reason, per field.
PREDATES_FORM: dict[str, dict[str, str]] = {
    "method-cluster-selection-prereg.md": {
        "training_years": _BY_REFERENCE.format("each configuration"),
        "test_year": _BY_REFERENCE.format("each configuration"),
    },
    "method-curve-library-prereg.md": {
        "training_years": _BY_REFERENCE.format("each row's scorecard config"),
        "test_year": _BY_REFERENCE.format("each row's scorecard config"),
    },
    "method-domain-split-prereg.md": {
        "training_years": _BY_REFERENCE.format("the chapter-era control points"),
        "test_year": _BY_REFERENCE.format("the chapter-era observations"),
    },
    "method-eu-rerun-prereg.md": {
        "training_years": _BY_REFERENCE.format("each row's scorecard config"),
        "test_year": _BY_REFERENCE.format("each row's scorecard config"),
    },
    "method-grid-nl-holdout-prereg.md": {
        "training_years": _BY_REFERENCE.format("the chapter-era control points"),
    },
    "method-loco-interpolation-prereg.md": {
        "training_years": _BY_REFERENCE.format("the chapter-era control points"),
        "test_year": _BY_REFERENCE.format("the chapter-era control points"),
    },
    "method-national-single-cluster-prereg.md": {
        "training_years": _BY_REFERENCE.format("the parent registration"),
        "test_year": _BY_REFERENCE.format("the parent registration"),
    },
    "method-offshore-pool-prereg.md": {
        "training_years": _BY_REFERENCE.format("the chapter-era control points"),
        "test_year": _BY_REFERENCE.format("the chapter's evaluation"),
    },
    "agent-critics-prereg.md": {
        "training_years": "evaluates the review agents on written proposals; no data is fitted",
        "test_year": "evaluates the review agents on written proposals; no data is fitted",
        "gates": "registers its pass criterion under 'The bar', fixed before the run (1bd9607)",
    },
    "method-joint-fit-reachability-prereg.md": {
        "gates": "a classification pass that decides nothing about the method; "
        "its fixed-in-advance readings are under 'Readings, fixed now'",
    },
}

_YEAR = r"(?:19|20)\d{2}"
_SPAN = rf"{_YEAR}(?:\s*-\s*\d{{2}}(?:\d{{2}})?|\s+to\s+{_YEAR})"

_TRAINING_PROSE = (
    rf"\btraining years:?\s+(?:[A-Z]{{2}}\s+)?{_SPAN}",
    rf"\btrained\s+{_SPAN}",
    r"\bno training years are involved\b",
)
_TEST_PROSE = (
    rf"\btest year\s+{_YEAR}\b(?!\s*-|\s+to\b)",
    rf"\btested on\s+{_YEAR}\b",
    r"\bno test year is involved\b",
)


def _flat(text: str) -> str:
    """The text with emphasis and code marks removed and whitespace collapsed,
    so a phrase broken across lines or half in bold still reads as one."""
    return " ".join(re.sub(r"[*`]", "", text).split())


def _tables(text: str) -> list[list[list[str]]]:
    """Each Markdown table as rows of cells, header first, separator dropped."""
    tables: list[list[list[str]]] = []
    block: list[str] = []
    for line in [*text.splitlines(), ""]:
        if line.lstrip().startswith("|"):
            block.append(line.strip())
            continue
        if len(block) >= 3:
            rows = [[_flat(c) for c in row.strip("|").split("|")] for row in block]
            tables.append([rows[0], *rows[2:]])
        block = []
    return tables


def _table_states(text: str, column: str, cell: str) -> bool:
    """Whether some table has ``column`` with every data row matching ``cell``."""
    for header, *rows in _tables(text):
        names = [h.lower() for h in header]
        if column not in names or not rows:
            continue
        i = names.index(column)
        if all(i < len(r) and re.fullmatch(cell, r[i]) for r in rows):
            return True
    return False


def _registration(text: str) -> bool:
    lines = text.splitlines()
    for start, line in enumerate(lines):
        if line.startswith("**Date:**"):
            para = [line]
            for nxt in lines[start + 1 :]:
                if not nxt.strip() or nxt.startswith("**Scope:**"):
                    break
                para.append(nxt)
            flat = _flat(" ".join(para))
            return bool(re.search(rf"{_YEAR}-\d{{2}}-\d{{2}}", flat)) and bool(
                re.search(r"\bbefore\b", flat, re.IGNORECASE)
            )
    return False


def _gates(text: str) -> bool:
    lines = text.splitlines()
    for start, line in enumerate(lines):
        if re.match(r"#{2,3}\s+Gates?\b", line):
            for nxt in lines[start + 1 :]:
                if nxt.startswith("#"):
                    break
                if nxt.strip():
                    return True
    return False


def missing_fields(text: str) -> set[str]:
    """The fields of ``FIELDS`` that ``text`` does not state."""
    flat = _flat(text)
    found = {
        "registration": _registration(text),
        "training_years": any(re.search(p, flat, re.IGNORECASE) for p in _TRAINING_PROSE)
        or _table_states(text, "training years", _SPAN),
        "test_year": any(re.search(p, flat, re.IGNORECASE) for p in _TEST_PROSE)
        or _table_states(text, "test year", _YEAR),
        "gates": _gates(text),
    }
    return {f for f in FIELDS if not found[f]}


def test_the_preregistrations_are_found():
    assert len(PREREGS) >= 16, [p.name for p in PREREGS]


@pytest.mark.parametrize("path", PREREGS, ids=lambda p: p.name)
def test_each_preregistration_states_its_fields(path: Path):
    missing = missing_fields(path.read_text(encoding="utf-8"))
    excused = set(PREDATES_FORM.get(path.name, {}))
    assert not missing - excused, (
        f"{path.name} does not state {sorted(missing - excused)}. Add it in one "
        "of the forms this module's docstring lists, or, for a document that "
        "predates the form, list it in PREDATES_FORM with the reason."
    )
    assert not excused - missing, (
        f"{path.name} now states {sorted(excused - missing)}; remove "
        "those entries from PREDATES_FORM."
    )


def test_every_allowlisted_document_exists():
    names = {p.name for p in PREREGS}
    assert set(PREDATES_FORM) <= names, sorted(set(PREDATES_FORM) - names)
    for name, reasons in PREDATES_FORM.items():
        assert set(reasons) <= set(FIELDS), name
        assert all(r.strip() for r in reasons.values()), name


# The check, both ways: a complete document passes, and a copy with one field
# removed fails on that field and no other.

COMPLETE = FINDINGS / "method-wake-unit-prereg.md"

REMOVALS = {
    "registration": lambda t: t.replace("**Date:**", "**Note:**"),
    "training_years": lambda t: re.sub("Training years", "Years", t),
    "test_year": lambda t: re.sub(r"(?i)test year", "scored year", t),
    "gates": lambda t: re.sub(r"(?m)^## Gates$", "## Criteria", t),
}


def test_a_complete_document_passes():
    assert missing_fields(COMPLETE.read_text(encoding="utf-8")) == set()


@pytest.mark.parametrize("field", FIELDS)
def test_a_copy_with_one_field_removed_fails_on_it(field: str, tmp_path: Path):
    copy = tmp_path / COMPLETE.name
    original = COMPLETE.read_text(encoding="utf-8")
    copy.write_text(REMOVALS[field](original), encoding="utf-8")
    assert copy.read_text(encoding="utf-8") != original
    assert missing_fields(copy.read_text(encoding="utf-8")) == {field}


BOTH = {"training_years", "test_year"}


@pytest.mark.parametrize(
    "text, missing",
    [
        ("Training years 2015-21 (IE 2017-21), test\nyear 2023.", set()),
        ("**Training years 2019 to 2021** ... **Test year 2022** only", set()),
        ("`fixed`, trained 2015 to 2021, tested on 2023.", set()),
        ("Training years\n2015-21. No test year is involved: training fits.", set()),
        ("No training years are involved, and no test year is involved.", set()),
        ("No training data is read; no test year is involved.", {"training_years"}),
        ("held at its training years and its test year", BOTH),
        ("Training years 2015-21, with test years 2019 and 2023", {"test_year"}),
        ("Training years 2015-21, test year 2019-2023", {"test_year"}),
    ],
)
def test_the_prose_forms(text: str, missing: set[str]):
    assert missing_fields(text) & BOTH == missing


def test_a_table_with_two_test_years_in_a_row_fails():
    table = (
        "| Region | Training years | Test year |\n"
        "|---|---|---|\n"
        "| DK | 2015 to 2019 | 2020 |\n"
        "| DE | 2015 to 2018 | 2019 and 2023 |\n"
    )
    assert missing_fields(table) & BOTH == {"test_year"}
    single = table.replace("2019 and 2023", "2019")
    assert missing_fields(single) & BOTH == set()
