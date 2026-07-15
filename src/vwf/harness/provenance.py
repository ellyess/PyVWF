"""Run provenance: make every harness output self-describing.

Every run directory gets a ``run_manifest.json`` recording the package
version, git state, region config, observation granularity caveats, and —
the reason this module exists — the identity of the curve library that
produced the numbers. RQ6 results are only reportable from a manifest whose
``curve_library.library`` is ``"external"``.

Provenance is diagnostic, not load-bearing: :func:`write_manifest_safe` never
raises, so a manifest failure can never abort a run.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import warnings
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Any

import pandas as pd

import vwf
from vwf.config import PyVWFPaths
from vwf.harness.regions import RegionSpec

MANIFEST_NAME = "run_manifest.json"


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
    synthetic file as external. Contents are never copied into the manifest —
    only hashes and counts — so licensed curve libraries stay out of every
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
    synthetic = (
        curves_sha == _bundled_hash("power_curves.csv")
        and models_sha == _bundled_hash("models.csv")
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


def _git_state() -> dict[str, Any]:
    """Best-effort git commit/dirty state; nulls outside a checkout."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": None, "git_dirty": None}


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
