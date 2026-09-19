"""Run provenance: make every harness output self-describing.

Every run directory gets a ``run_manifest.json`` recording the package
version, git state, Python and library versions, region config, observation
granularity caveats, and
(the reason this module exists) the identity of the curve library that
produced the numbers. Any result that turns on real rather than bundled
curves is only reportable from a manifest whose ``curve_library.library`` is
``"external"``.

Provenance is diagnostic, not load-bearing: :func:`write_manifest_safe` never
raises, so a manifest failure can never abort a run.
"""

from __future__ import annotations

import functools
import hashlib
import json
import platform
import sys
from importlib import metadata
import subprocess
import warnings
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import vwf
from vwf.config import PyVWFPaths
from vwf.harness.regions import RegionSpec
from vwf.wind import default_curve_key

MANIFEST_NAME = "run_manifest.json"

#: Distributions whose versions every manifest records. The numbers depend on
#: them: a random forest's scores moved in the third decimal between
#: scikit-learn 1.7.2 and 1.9.1, and pandas 3 changed the last digit of
#: unit-level sums. Optional extras are recorded when installed.
ENVIRONMENT_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xarray",
    "dask",
    "netCDF4",
    "bottleneck",
    "geopandas",
    "shapely",
    "pyproj",
    "matplotlib",
    "torch",
    "pykrige",
    "rasterio",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundled_hash(filename: str) -> str | None:
    """SHA-256 of the synthetic reference table bundled in vwf.resources."""
    packaged = Path(str(resources.files("vwf.resources") / filename))
    if not packaged.is_file():
        return None
    return _sha256(packaged)


def curve_library_identity() -> dict[str, Any]:
    """Identify the curve library the current configuration resolves to.

    Returns a dict with the resolved paths and SHA-256 hashes of
    ``power_curves.csv`` and ``models.csv``, plus ``library``:

    - ``"synthetic-bundled"`` if BOTH files hash-match the synthetic tables
      bundled in ``vwf.resources``;
    - ``"external"`` otherwise.

    A locally *edited* copy of the synthetic files therefore also labels as
    ``"external"``. This is intentional and fail-safe in the underclaiming
    direction: nothing that differs from the audited bundled tables can
    masquerade as them, at the cost of occasionally labelling a still-
    synthetic file as external. Contents are never copied into the manifest,
    only hashes and counts, so licensed curve libraries stay out of every
    output artifact.
    """
    with warnings.catch_warnings():
        # reference_file warns when falling back to the bundled synthetic
        # tables; here that fallback is exactly what we are recording, not a
        # condition to warn about.
        warnings.simplefilter("ignore", UserWarning)
        curves_path = PyVWFPaths.reference_file("power_curves.csv")
        models_path = PyVWFPaths.reference_file("models.csv")

    curves_sha = _sha256(curves_path)
    models_sha = _sha256(models_path)
    synthetic = curves_sha == _bundled_hash("power_curves.csv") and models_sha == _bundled_hash(
        "models.csv"
    )

    n_curves = max(len(pd.read_csv(curves_path, nrows=0).columns) - 1, 0)
    n_models = len(pd.read_csv(models_path))

    return {
        "power_curves_path": str(curves_path),
        "power_curves_sha256": curves_sha,
        "models_path": str(models_path),
        "models_sha256": models_sha,
        "library": "synthetic-bundled" if synthetic else "external",
        "n_curves": n_curves,
        "n_models": n_models,
    }


#: How a unit's model key got onto the fleet, read from the first of these
#: columns the fleet carries: ``model_source`` is written into the metadata by
#: the process scripts that call ``assign_curves_from_library``, and
#: ``model_match`` by :func:`vwf.data.add_models` at load time.
_ASSIGNMENT_COLUMNS = ("model_source", "model_match")

CURVE_RESOLUTION_NAME = "curve_resolution.csv"


def _curve_hash(values: Any) -> str:
    """SHA-256 of one curve's capacity-factor values, as float64 bytes."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


@functools.lru_cache(maxsize=1)
def _bundled_curve_hashes() -> frozenset[str]:
    packaged = Path(str(resources.files("vwf.resources") / "power_curves.csv"))
    if not packaged.is_file():
        return frozenset()
    table = pd.read_csv(packaged)
    return frozenset(_curve_hash(table[c]) for c in table.columns if c != "data$speed")


def curve_resolution(fleet: pd.DataFrame, power_curves: pd.DataFrame) -> pd.DataFrame:
    """Record which curve each model key in a fleet actually resolves to.

    The simulation looks each unit's ``model`` up in the curve table and, for a
    key the table lacks, silently uses :func:`vwf.wind.default_curve_key`'s
    curve after a one-off warning. That is how every country-level run on the
    bundled library came to simulate a 100 kW distributed-wind turbine without
    any artefact saying so. This reconstructs the lookup from the same fleet and
    table the simulation receives, using the same fallback helper, so it records
    what the simulation did without touching the simulation.

    Args:
        fleet: The units a run simulates, with ``model`` and ideally
            ``capacity``, and optionally ``model_source`` or ``model_match``.
        power_curves: The curve table the run simulates with.

    Returns:
        One row per requested key: ``requested``, ``n_units``, ``capacity``
        (in the fleet's own units), ``capacity_share``, ``assigned_by`` (how the
        key got onto the fleet, ``"as-given"`` when nothing records it),
        ``status`` (``"resolved"`` or ``"substituted"``), ``curve_used``,
        ``curve_sha256`` and ``origin``. ``origin`` is ``"open"`` when the
        curve's values hash-match a column of the bundled open library and
        ``"external"`` otherwise, so under ``input/combined`` external means the
        licensed library. Matching on values rather than names means a key
        present in both libraries is only "open" if it carries the open curve,
        and an edited open curve labels as external, failing towards
        underclaiming as :func:`curve_library_identity` does.
    """
    columns = [
        "requested",
        "n_units",
        "capacity",
        "capacity_share",
        "assigned_by",
        "status",
        "curve_used",
        "curve_sha256",
        "origin",
    ]
    present = set(power_curves.columns) - {"data$speed"}
    fallback = default_curve_key(power_curves)

    if "model" in fleet.columns:
        requested = fleet["model"].astype(object)
        requested = requested.where(requested.notna(), "<none>").astype(str)
    else:
        requested = pd.Series("<none>", index=fleet.index)
    if "capacity" in fleet.columns:
        capacity = pd.to_numeric(fleet["capacity"], errors="coerce").fillna(0.0)
    else:
        capacity = pd.Series(1.0, index=fleet.index)
    source_col = next((c for c in _ASSIGNMENT_COLUMNS if c in fleet.columns), None)
    assigned = (
        fleet[source_col].astype(object).where(fleet[source_col].notna(), "unrecorded").astype(str)
        if source_col is not None
        else pd.Series("as-given", index=fleet.index)
    )

    frame = pd.DataFrame(
        {
            "requested": requested.to_numpy(),
            "capacity": capacity.to_numpy(dtype=float),
            "assigned_by": assigned.to_numpy(),
        }
    )
    total = float(frame["capacity"].sum())
    bundled = _bundled_curve_hashes()

    rows = []
    for key, group in frame.groupby("requested", sort=True):
        resolved = key in present
        used = key if resolved else fallback
        sha = _curve_hash(power_curves[used]) if used is not None else None
        cap = float(group["capacity"].sum())
        rows.append(
            {
                "requested": key,
                "n_units": int(len(group)),
                "capacity": cap,
                "capacity_share": cap / total if total > 0 else float("nan"),
                "assigned_by": ";".join(sorted(set(group["assigned_by"]))),
                "status": "resolved" if resolved else "substituted",
                "curve_used": used,
                "curve_sha256": sha,
                "origin": None if sha is None else ("open" if sha in bundled else "external"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarise_curve_resolution(resolution: pd.DataFrame) -> dict[str, Any]:
    """Reduce a :func:`curve_resolution` table to the manifest's summary.

    ``substituted_capacity_share`` is the headline: the share of the fleet's
    capacity simulated on a curve other than the one its model key names.
    """
    share = resolution["capacity_share"].fillna(0.0)
    substituted = resolution["status"] == "substituted"
    by_assignment = share.groupby(resolution["assigned_by"]).sum()
    return {
        "n_models_requested": int(len(resolution)),
        "n_models_substituted": int(substituted.sum()),
        "substituted_capacity_share": float(share[substituted].sum()),
        "substitutions": {
            str(k): str(v)
            for k, v in zip(
                resolution.loc[substituted, "requested"], resolution.loc[substituted, "curve_used"]
            )
        },
        "open_capacity_share": float(share[resolution["origin"] == "open"].sum()),
        "external_capacity_share": float(share[resolution["origin"] == "external"].sum()),
        "assigned_by_capacity_share": {str(k): float(v) for k, v in by_assignment.items()},
    }


def _git_state() -> dict[str, Any]:
    """Best-effort git commit/dirty state; nulls outside a checkout."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


def environment() -> dict[str, Any]:
    """The Python version and the installed versions of the scientific stack.

    A distribution that is not installed is recorded as None, so a manifest
    says what was absent as well as what was present.
    """
    packages: dict[str, str | None] = {}
    for name in ENVIRONMENT_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version.split()[0], "packages": packages}


def build_manifest(
    spec: RegionSpec | None = None,
    *,
    correction: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest dict.

    Args:
        spec: Region config for this run, if the run is harness-driven.
            Legacy ``PyVWF`` runs pass ``None`` and describe themselves via
            ``extra``.
        correction: Correction settings; defaults to the spec's when present.
        extra: Additional top-level entries (legacy run parameters, notes).
    """
    manifest: dict[str, Any] = {
        "pyvwf_version": vwf.__version__,
        **_git_state(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "environment": environment(),
        "curve_library": curve_library_identity(),
    }

    if spec is not None:
        manifest["region"] = {"code": spec.code, "name": spec.name}
        # Granularity caveats live HERE, not only in the config: the manifest
        # is the self-describing record of what the numbers mean.
        manifest["observations"] = {
            "source": spec.source,
            "obs_level": spec.obs_level,
            "obs_unit": spec.obs_unit,
            "location_resolution": spec.location_resolution,
            "pseudo_replicated_rows": spec.pseudo_replicated_rows,
            "train_years": list(spec.train_years),
            "test_years": list(spec.test_years),
            "time_convention": spec.time_convention,
        }
        manifest["correction"] = correction or {
            "model": spec.correction_model,
            "cluster_list": list(spec.cluster_list),
            "time_slices": list(spec.time_slices),
        }
        manifest["seasons"] = {name: list(months) for name, months in spec.seasons.items()}
    elif correction is not None:
        manifest["correction"] = correction

    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Write ``run_manifest.json`` into ``run_dir`` and return its path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / MANIFEST_NAME
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return out


def write_manifest_safe(
    run_dir: str | Path,
    spec: RegionSpec | None = None,
    *,
    correction: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path | None:
    """Build and write a manifest; on ANY failure warn and return None.

    Provenance must never abort a run (design §6): a full disk, a read-only
    output directory, or a broken git binary degrades the run's
    self-description, not the run.
    """
    try:
        return write_manifest(run_dir, build_manifest(spec, correction=correction, extra=extra))
    except Exception as exc:  # noqa: BLE001 - never-abort is the contract
        warnings.warn(
            f"Could not write {MANIFEST_NAME} to {run_dir}: {exc}. "
            "The run continues, but its outputs are not self-describing.",
            UserWarning,
            stacklevel=2,
        )
        return None
