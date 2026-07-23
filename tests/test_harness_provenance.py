"""Run-manifest provenance (docs/HARNESS_DESIGN.md §6)."""
import json
import shutil
from importlib import resources
from pathlib import Path

import pytest

from vwf.config import PyVWFPaths
from vwf.harness.provenance import (
    MANIFEST_NAME,
    build_manifest,
    curve_library_identity,
    write_manifest,
    write_manifest_safe,
)
from vwf.harness.regions import load_region

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "regions"


@pytest.fixture
def empty_input_root(tmp_path, monkeypatch):
    """An INPUT_ROOT with no reference tables → resolution falls back to bundled."""
    root = tmp_path / "input"
    root.mkdir()
    monkeypatch.setattr(PyVWFPaths, "INPUT_ROOT", root)
    return root


def bundled(filename: str) -> Path:
    return Path(str(resources.files("vwf.resources") / filename))


def test_bundled_tables_label_synthetic(empty_input_root):
    identity = curve_library_identity()
    assert identity["library"] == "synthetic-bundled"
    assert identity["power_curves_sha256"]
    assert identity["n_curves"] > 0
    assert identity["n_models"] > 0


def test_edited_synthetic_labels_external(empty_input_root):
    # Copy the bundled synthetic table into INPUT_ROOT and change one byte.
    # Design §6: an edited synthetic file labels "external" ON PURPOSE —
    # fail-safe in the underclaiming direction.
    local = empty_input_root / "reference" / "power_curves.csv"
    local.parent.mkdir(exist_ok=True)
    shutil.copy(bundled("power_curves.csv"), local)
    with open(local, "a", encoding="utf-8") as fh:
        fh.write("\n")
    identity = curve_library_identity()
    assert identity["library"] == "external"
    assert identity["power_curves_path"] == str(local)


def test_identical_local_copy_labels_synthetic(empty_input_root):
    # A byte-identical local copy IS the bundled library; only difference
    # triggers "external".
    ref = empty_input_root / "reference"
    ref.mkdir(exist_ok=True)
    shutil.copy(bundled("power_curves.csv"), ref / "power_curves.csv")
    shutil.copy(bundled("models.csv"), ref / "models.csv")
    assert curve_library_identity()["library"] == "synthetic-bundled"


def test_manifest_observations_block_carries_granularity(empty_input_root):
    spec = load_region(CONFIG_DIR / "uk.toml")
    manifest = build_manifest(spec)
    obs = manifest["observations"]
    # The manifest is the self-describing record: granularity caveats must be
    # here, not only in the region config.
    assert obs["obs_unit"] == "farm"
    assert obs["pseudo_replicated_rows"] is True
    assert "location_resolution" in obs
    assert obs["time_convention"] == "utc-monthly-bins"
    assert manifest["seasons"]["winter"] == [12, 1, 2]
    assert manifest["curve_library"]["library"] == "synthetic-bundled"
    assert manifest["pyvwf_version"]


def test_manifest_de_location_resolution(empty_input_root):
    spec = load_region(CONFIG_DIR / "de.toml")
    obs = build_manifest(spec)["observations"]
    assert obs["obs_unit"] == "turbine"
    assert obs["location_resolution"] == "postcode"


def test_write_and_reload_round_trip(empty_input_root, tmp_path):
    spec = load_region(CONFIG_DIR / "au_nem.toml")
    out = write_manifest(tmp_path / "run", build_manifest(spec))
    assert out.name == MANIFEST_NAME
    loaded = json.loads(out.read_text())
    assert loaded["region"]["code"] == "AU-NEM"
    assert loaded["observations"]["time_convention"] == "utc-monthly-bins"
    assert loaded["seasons"]["winter"] == [6, 7, 8]


def test_write_manifest_safe_never_raises(empty_input_root, tmp_path):
    # Point the run dir at an existing FILE so mkdir/open must fail.
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("occupied")
    with pytest.warns(UserWarning, match="not self-describing"):
        result = write_manifest_safe(blocker / "run")
    assert result is None
