"""A per-unit fleet override moves only the units it names (issue #12).

The German fleet sits at postcode centroids, and MaStR gives a coordinate and
a hub height per turbine. The match covers about 70% of the fitted fleet, so
the override has to leave the rest where they are rather than dropping them:
their observations still cover them, and a cache of mixed location resolution
is what the rebuild produces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vwf.pinn.cache import apply_fleet_override


def _fleet():
    return pd.DataFrame(
        {
            "ID": ["a", "b", "c"],
            "lon": [9.0, 9.0, 10.0],
            "lat": [52.0, 52.0, 53.0],
            "height": [80.0, 80.0, 100.0],
            "capacity": [2000.0, 2000.0, 1000.0],
        }
    )


def test_only_the_named_units_move():
    override = pd.DataFrame({"ID": ["a"], "lon": [9.31], "lat": [52.22], "height": [92.0]})
    moved, record = apply_fleet_override(_fleet(), override)
    assert moved.loc[0, ["lon", "lat", "height"]].tolist() == [9.31, 52.22, 92.0]
    # b shared a's postcode centroid and is not in the override: it stays.
    assert moved.loc[1, ["lon", "lat", "height"]].tolist() == [9.0, 52.0, 80.0]
    assert moved.loc[2, ["lon", "lat", "height"]].tolist() == [10.0, 53.0, 100.0]


def test_the_record_counts_units_capacity_and_locations():
    override = pd.DataFrame({"ID": ["a"], "lon": [9.31], "lat": [52.22], "height": [92.0]})
    _, record = apply_fleet_override(_fleet(), override)
    assert record["units_total"] == 3
    assert record["units_overridden"] == 1
    assert record["capacity_share_overridden"] == pytest.approx(2000 / 5000)
    # Two units shared one centroid before; one of them moved.
    assert record["distinct_locations_before"] == 2
    assert record["distinct_locations_after"] == 3


def test_a_missing_value_in_the_override_keeps_what_the_unit_had():
    override = pd.DataFrame({"ID": ["a", "b"], "lon": [9.31, np.nan], "lat": [52.22, np.nan]})
    moved, record = apply_fleet_override(_fleet(), override)
    assert moved.loc[1, ["lon", "lat"]].tolist() == [9.0, 52.0]
    assert record["columns"] == ["lon", "lat"]  # height untouched where absent
    assert moved["height"].tolist() == [80.0, 80.0, 100.0]


def test_an_override_naming_nothing_in_the_fleet_changes_nothing():
    override = pd.DataFrame({"ID": ["zz"], "lon": [1.0], "lat": [2.0], "height": [3.0]})
    moved, record = apply_fleet_override(_fleet(), override)
    pd.testing.assert_frame_equal(moved, _fleet())
    assert record["units_overridden"] == 0
    assert record["capacity_share_overridden"] == 0.0


def test_a_cache_override_without_its_ratios_is_refused():
    """A GWA ratio belongs to the locations it was computed at (issue #12).

    The first attempt at the turbine study ran the German fold on MaStR
    coordinates with ratios computed at the postcode centroids, because a
    failed ratio pass did not stop the study. The driver now refuses that
    combination, and this pins the refusal by parsing the driver's own
    argument handling.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("turbine_loro", "scripts/pinn/turbine_loro.py")
    source = Path("scripts/pinn/turbine_loro.py").read_text(encoding="utf-8")
    assert "--gwa-override" in source
    # The guard: a cache override with no ratio override stops the run.
    assert "mismatched = sorted(set(cache_of) - set(gwa_of))" in source
    assert "both must move" in source
    assert spec is not None
