"""Multi-region validation harness.

Additive layer over the PyVWF core: declarative region configs, run
provenance, and standard skill metrics. Nothing in here changes the validated
correction or simulation code paths.

See ``docs/design/harness.md`` for the design this implements.
"""

from pyvwf.harness.corrections import (
    AffineWindCorrection,
    CorrectionModel,
    available_corrections,
    get_correction,
    register_correction,
)
from pyvwf.provenance import (
    build_manifest,
    curve_library_identity,
    write_manifest,
    write_manifest_safe,
)
from pyvwf.harness.regions import RegionSpec, load_region, season_of_month
from pyvwf.harness.skill import (
    collapse_pseudo_replicates,
    seasonal_cycle_rmse,
    skill_metrics,
    station_ids,
)

__all__ = [
    "AffineWindCorrection",
    "CorrectionModel",
    "available_corrections",
    "get_correction",
    "register_correction",
    "RegionSpec",
    "load_region",
    "season_of_month",
    "build_manifest",
    "curve_library_identity",
    "write_manifest",
    "write_manifest_safe",
    "skill_metrics",
    "seasonal_cycle_rmse",
    "collapse_pseudo_replicates",
    "station_ids",
]
