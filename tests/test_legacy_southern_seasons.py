"""The legacy path refuses the ``season`` slice for a southern fleet.

`PyVWF` has no season mapping and labels months with Northern-Hemisphere
seasons, so a southern fleet's "winter" factors were fitted on its summer.
`pyvwf-train` resolves southern regions (BR, AU, CL, AR, NZ) and defaults to
fitting ``season``. The harness takes each region's own months; the legacy
path now stops instead.
"""

from __future__ import annotations

import pandas as pd
import pytest

import vwf.vwf as legacy
from vwf.vwf import PyVWF, _refuse_northern_seasons_south


def fleet(lats):
    return pd.DataFrame({"ID": [f"t{i}" for i in range(len(lats))], "lat": lats})


def test_a_southern_fleet_with_season_is_refused():
    with pytest.raises(ValueError, match="south of the equator"):
        _refuse_northern_seasons_south(fleet([-35.0, -34.0, -38.0]), ["fixed", "season"])


@pytest.mark.parametrize(
    "lats, slices",
    [
        ([55.0, 56.0], ["fixed", "season"]),  # northern fleet: the mapping is right
        ([-35.0, -34.0], ["fixed", "month", "bimonth"]),  # no season slice requested
        ([-35.0, -34.0], []),
    ],
)
def test_other_cases_pass(lats, slices):
    _refuse_northern_seasons_south(fleet(lats), slices)


def test_train_refuses_before_fitting(tmp_path, monkeypatch):
    def southern_train_set(*args, **kwargs):
        return pd.DataFrame(), fleet([-35.0, -34.0]), None, None

    monkeypatch.setattr(legacy, "train_set", southern_train_set)
    model = PyVWF(str(tmp_path), "AU", True, True, "onshore", [1], ["season"])
    with pytest.raises(ValueError, match="south of the equator"):
        model.train()
