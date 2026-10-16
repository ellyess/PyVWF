"""Bidding-zone grids (pyvwf.datasets.country_grid) read the input root in use.

Until 2026-10-16 the Norwegian and Swedish zone builders opened
``input/reference/shapes/...`` literally, ignoring ``PYVWF_INPUT`` (#84).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyvwf.config import PyVWFPaths
from pyvwf.datasets import country_grid as cg


def test_the_shapes_path_follows_the_input_root(tmp_path, monkeypatch):
    monkeypatch.setattr(PyVWFPaths, "INPUT_ROOT", tmp_path)
    assert cg.bidding_zone_shapes("SE") == tmp_path / "reference/shapes/se_bidding_zones.geojson"


def test_sweden_without_shapes_under_the_root_in_use_is_refused_by_that_path(tmp_path, monkeypatch):
    """Known positive: the repository's own input/ has the file; the root in use does not."""
    monkeypatch.setattr(PyVWFPaths, "INPUT_ROOT", tmp_path)
    config = {"height": 100.0, "model": "M", "capacity": 3.0}
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        cg.generate_sweden_zone_grids(config, tmp_path / "out", save_geojson=False)
    assert not Path(tmp_path / "out").exists()
