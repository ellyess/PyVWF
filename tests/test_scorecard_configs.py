"""Every scorecard row's configuration is committed, and every one of them loads.

The scorecard's claim to be reproducible against a commit rests on
``configs/regions/scorecard/`` holding the exact configuration behind each row.
Nothing checked that. A row could be added without its configuration, a
configuration could stop loading after a schema change, or a file could be
renamed out from under the document, and the suite would stay green.

These tests close that. They read the region codes out of the scorecard's own
tables rather than from a list kept beside them, so a new row is covered the
day it is published.
"""

import re
from pathlib import Path

import pytest

from pyvwf.harness.regions import load_region

ROOT = Path(__file__).resolve().parents[1]
SCORECARD = ROOT / "docs" / "findings" / "scorecard.md"
CONFIG_DIR = ROOT / "configs" / "regions" / "scorecard"
# "| Germany (DE) | ...", "| Denmark (DK) § 0.6% | ...", "| Australia (AU-NEM) |"
ROW = re.compile(r"^\|\s*[A-Z][A-Za-z .]*\(([A-Z][A-Z-]*)\)")


def scorecard_codes() -> set[str]:
    return {m.group(1) for line in SCORECARD.read_text().splitlines() if (m := ROW.match(line))}


def committed_configs() -> list[Path]:
    """The configurations of the rows standing today, not the superseded ones."""
    return sorted(CONFIG_DIR.glob("*.toml"))


def superseded_configs() -> list[Path]:
    return sorted(CONFIG_DIR.glob("superseded/*/*.toml"))


def test_the_scorecard_tables_are_readable():
    """A parser that silently matched nothing would make the rest vacuous."""
    codes = scorecard_codes()
    assert len(codes) >= 17
    assert {"DE", "DK", "UK", "US", "BR", "CL", "AR", "NZ", "AU-NEM"} <= codes


def test_every_committed_scorecard_config_loads():
    paths = committed_configs()
    assert paths, "no scorecard configurations are committed"
    for path in paths:
        load_region(path)  # raises with the file named if it does not


def test_every_superseded_config_still_loads():
    """The Superseded section of the scorecard cites these by path, so they
    have to keep resolving. They are records: never edited, only moved."""
    for path in superseded_configs():
        load_region(path)


def test_one_current_configuration_per_region():
    """The canonical name points at the row standing today. Two configurations
    for one code in this directory is the trap this layout exists to avoid: a
    reader reaching for a region gets whichever file they guess."""
    seen: dict[str, list[str]] = {}
    for path in committed_configs():
        seen.setdefault(load_region(path).code, []).append(path.name)
    duplicated = {code: names for code, names in seen.items() if len(names) > 1}
    assert not duplicated, f"more than one current configuration: {duplicated}"


def test_every_scorecard_row_has_a_committed_configuration():
    have = {load_region(p).code for p in committed_configs()}
    missing = sorted(scorecard_codes() - have)
    assert not missing, f"scorecard rows with no committed configuration: {missing}"


def test_no_committed_configuration_is_orphaned():
    """A configuration here names a row. One that names nothing is a leftover."""
    codes = scorecard_codes()
    orphans = sorted({load_region(p).code for p in committed_configs()} - codes)
    assert not orphans, f"configurations for no scorecard row: {orphans}"


CFG = re.compile(r"\|\s*(?:k|N=)(\d+)\s+(fixed|season|bimonth|month)\b")


def rows_with_a_configuration():
    """(code, cluster count, time slice) for every row naming one.

    Norway's cell reads "correction does not help" rather than naming a
    configuration, so it contributes nothing here and is covered by the tests
    above. The check is by region code, so a superseded row is checked against
    the configuration standing today; that holds while a supersession changes
    the input rather than the cluster count or the time slice.
    """
    out = []
    for line in SCORECARD.read_text().splitlines():
        row, cfg = ROW.match(line), CFG.search(line)
        if row and cfg:
            out.append((row.group(1), int(cfg.group(1)), cfg.group(2)))
    return sorted(set(out))


def test_the_best_cfg_column_is_readable():
    found = rows_with_a_configuration()
    assert len(found) >= 17
    assert ("DE", 100, "fixed") in found and ("FR", 10, "fixed") in found


@pytest.mark.parametrize(
    "code, clusters, slice_", rows_with_a_configuration(), ids=lambda v: str(v)
)
def test_the_configuration_a_row_reports_is_one_its_config_can_produce(code, clusters, slice_):
    """The Best cfg column names a variant. The committed configuration has to
    be able to produce it, or the row cites a run that configuration cannot
    make.

    Note what this does NOT say: that the configuration produces only that
    variant. Eleven of the seventeen carry several, and since every variant of
    a run is scored on the rows common to all of them, the ones a row does not
    report still move the one it does (`docs/guides/output-structure.md`). The
    variant set is part of a row's design, not incidental to it.
    """
    specs = [load_region(p) for p in committed_configs()]
    matching = [s for s in specs if s.code == code]
    assert matching, f"no committed configuration for {code}"
    assert any(clusters in s.cluster_list and slice_ in s.time_slices for s in matching), (
        f"{code} reports {slice_}_{clusters}, which none of its committed "
        f"configurations can produce"
    )
