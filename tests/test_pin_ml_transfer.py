"""Pin the machine-learning transfer library before it moves into the package.

``scripts/analysis/ml_transfer_retest.py`` is both the round-one driver of
``docs/findings/method-ml-transfer.md`` and, in practice, a library: seven
``scripts/pinn/`` drivers and three analysis scripts import its
``build_centroids``, ``terrain_features``, ``rf_eval``, ``loro``,
``random_cv`` and ``variance_decomposition``. Those six moved to
``pyvwf.extensions.ml.transfer``; the run list and the gate stayed in the
script, which re-exports the six. Their output was pinned before the move, and
the tests call them through the script as the importers do.

Two layers:

- the six functions on synthetic inputs (two train directories and a small
  elevation grid written to a temporary root), against
  ``tests/data/pins/ml_transfer/``. These run in CI. The forest outputs were
  checked to agree to 15 significant figures under scikit-learn 1.7.2 and
  1.9.1 before being pinned.
- the whole script on the real July factor files and the ETOPO grid: the
  sha256 of everything it prints and of the centroid table it writes. Those
  inputs are local, so this layer skips in CI.

The real-data forest scores depend on the scikit-learn version, although the
synthetic ones do not: under 1.9.1 the leave-one-region-out table moves in the
third decimal from 1.7.2's (US scalar R2 -1.059 against -1.049), while the
centroid table is identical. The real-data layer is recorded under the version
CI's current Python jobs resolve, and skips under any other, because its
digest cannot distinguish a version change from a code change. It is a change
detector, not a guard (see CONTRIBUTING.md): when CI moves to a new
scikit-learn, re-record it under that version. It was first recorded under
1.7.2, the version the published table in ``method-ml-transfer.md`` matches;
that table is superseded.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analysis" / "ml_transfer_retest.py"
PINS = Path(__file__).resolve().parent / "data" / "pins" / "ml_transfer"

# Recorded at 3d1fe5a from a run on the real inputs (see the local test),
# under scikit-learn 1.9.1 and pandas 3.0.6, the versions CI's Python 3.11 and
# 3.12 jobs resolve.
REAL_SKLEARN = "1.9.1"
REAL_STDOUT_SHA256 = "08a27e8202d3d2abc78e1d6b28eb71dbc6e09cde44e8f4fefce25b8e382a06db"
REAL_CENTROIDS_SHA256 = "d2c897729da1d59ad597883de95c7bed14f77426e5aef30af59983cfd66c2bea"


def _load():
    spec = importlib.util.spec_from_file_location("ml_transfer_retest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ml = _load()


# --------------------------------------------------------------------------
# Synthetic inputs
# --------------------------------------------------------------------------


def write_synthetic_root(root: Path) -> dict:
    """Two train directories and an elevation grid under a fake repo root."""
    rng = np.random.default_rng(20260918)
    runs = {}
    for region, (lon0, lat0) in {"AA": (8.0, 55.0), "BB": (-2.0, 52.0)}.items():
        d = root / "output" / "validation" / region / "train-synthetic"
        d.mkdir(parents=True)
        k = 6
        n = 30
        turb = pd.DataFrame(
            {
                "ID": [f"{region}{i}" for i in range(n)],
                "cluster": np.arange(n) % k,
                "lon": lon0 + rng.uniform(-1.5, 1.5, n),
                "lat": lat0 + rng.uniform(-1.0, 1.0, n),
                "height": rng.uniform(60, 120, n).round(1),
                "capacity": rng.uniform(500, 4000, n).round(0),
            }
        )
        fac = pd.DataFrame(
            {
                "cluster": np.arange(k),
                "scalar": rng.uniform(0.7, 1.4, k),
                "offset": rng.uniform(-2.0, 1.0, k),
            }
        )
        turb.to_csv(d / f"train_turb_info_{k}.csv", index=False)
        fac.to_csv(d / f"factors_fixed_{k}.csv", index=False)
        runs[region] = (str(d.relative_to(root)), k)

    lon = np.arange(-5.0, 11.0, 1.0 / 120.0)
    lat = np.arange(49.0, 58.0, 1.0 / 120.0)
    lo, la = np.meshgrid(lon, lat)
    z = 300 * np.sin(lo / 2.0) * np.cos(la / 3.0) + 50 * (lo - 3) + 20 * (la - 53)
    terrain = root / "input" / "reference" / "terrain"
    terrain.mkdir(parents=True)
    xr.Dataset(
        {"z": (("lat", "lon"), z.astype("float32"))}, coords={"lat": lat, "lon": lon}
    ).to_netcdf(terrain / "etopo_global.nc")
    return runs


def synthetic_frame(root: Path) -> pd.DataFrame:
    """build_centroids then terrain_features, with the module rooted at ``root``."""
    runs = write_synthetic_root(root)
    old = ml.ROOT
    ml.ROOT = root
    try:
        df = ml.build_centroids(runs)
        for c in ("lon", "lat"):
            df[f"{c}_norm"] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
        df = ml.terrain_features(df)
    finally:
        ml.ROOT = old
    return df


def model_results(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for target in ("scalar", "offset"):
        out[f"loro_{target}"] = ml.loro(df, ml.SET_A, target)
        m, sd, mae = ml.random_cv(df, ml.SET_B, target)
        out[f"summary_{target}"] = pd.DataFrame(
            {
                "quantity": [
                    "random_cv_r2_mean",
                    "random_cv_r2_std",
                    "random_cv_mae",
                    "between_region_share",
                ],
                "value": [m, sd, mae, ml.variance_decomposition(df, target)],
            }
        )
    return out


# --------------------------------------------------------------------------
# CI layer
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frame(tmp_path_factory):
    return synthetic_frame(tmp_path_factory.mktemp("ml_root"))


def test_centroids_and_terrain_features(frame):
    want = pd.read_csv(PINS / "synthetic_frame.csv")
    pd.testing.assert_frame_equal(
        frame.reset_index(drop=True), want, check_dtype=False, rtol=0, atol=1e-12
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", ["loro_scalar", "loro_offset", "summary_scalar", "summary_offset"])
def test_forest_and_variance_results(frame, name):
    got = model_results(frame)[name]
    want = pd.read_csv(PINS / f"{name}.csv")
    pd.testing.assert_frame_equal(
        got.reset_index(drop=True), want, check_dtype=False, rtol=0, atol=1e-12
    )


# --------------------------------------------------------------------------
# Local layer: the whole script on the real inputs
# --------------------------------------------------------------------------


def _real_inputs_present() -> bool:
    runs = [ROOT / rel for rel, _ in ml.RUNS.values()]
    return (ROOT / "input/reference/terrain/etopo_global.nc").is_file() and all(
        p.is_dir() for p in runs
    )


def run_real(tmp_path: Path) -> tuple[str, str]:
    """Run main() with its output directory redirected; return both digests."""
    old = ml.OUT
    ml.OUT = tmp_path
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            ml.main()
    finally:
        ml.OUT = old
    stdout = hashlib.sha256(buffer.getvalue().encode()).hexdigest()
    centroids = hashlib.sha256(
        (tmp_path / "ml_retest_centroids_primary.csv").read_bytes()
    ).hexdigest()
    return stdout, centroids


@pytest.mark.realdata
@pytest.mark.skipif(
    not _real_inputs_present(), reason="the July factor files and the ETOPO grid are local only"
)
def test_script_on_the_real_inputs(tmp_path):
    import sklearn

    if sklearn.__version__ != REAL_SKLEARN:
        pytest.skip(f"pinned under scikit-learn {REAL_SKLEARN}, not {sklearn.__version__}")
    stdout, centroids = run_real(tmp_path)
    assert stdout == REAL_STDOUT_SHA256
    assert centroids == REAL_CENTROIDS_SHA256


if __name__ == "__main__":  # records the fixtures; run once, by hand
    import tempfile

    PINS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        df = synthetic_frame(Path(d))
    df.to_csv(PINS / "synthetic_frame.csv", index=False)
    for name, table in model_results(df).items():
        table.to_csv(PINS / f"{name}.csv", index=False)
    if _real_inputs_present():
        with tempfile.TemporaryDirectory() as d:
            print(run_real(Path(d)), file=sys.stderr)
