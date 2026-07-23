"""Multi-region validation harness.

Additive layer over the PyVWF core: declarative region configs, run
provenance, and standard skill metrics. Nothing in here changes the validated
correction or simulation code paths.

See ``docs/design/HARNESS_DESIGN.md`` for the design this implements.
"""
from vwf.harness.corrections import (
    AffineWindCorrection,
    CorrectionModel,
    available_corrections,
    get_correction,
    register_correction,
)
from vwf.harness.provenance import (
    build_manifest,
    curve_library_identity,
    write_manifest,
    write_manifest_safe,
)
from vwf.harness.regions import RegionSpec, load_region, season_of_month
from vwf.harness.skill import (
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
