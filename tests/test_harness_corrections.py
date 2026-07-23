"""CorrectionModel interface and the affine baseline delegate (design §4).

The load-bearing test here is the golden regression: AffineWindCorrection
with seasons=None must reproduce the legacy turbine-level pipeline
bit for bit (exact frame equality, no tolerances). If that pin ever breaks,
the harness is no longer measuring the validated method.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import test_pipeline as tp
import vwf.correction as correction
import vwf.wind as wind
from vwf.config import PyVWFPaths
from vwf.data import cluster_train_set, format_bc_factors, train_set
from vwf.harness import available_corrections, get_correction
from vwf.harness.corrections import CorrectionModel, register_correction
from vwf.wind import correct_wind_speed

SH_SEASONS = {
    "summer": [12, 1, 2],
    "autumn": [3, 4, 5],
    "winter": [6, 7, 8],
    "spring": [9, 10, 11],
}


@pytest.fixture
def synthetic_dk(tmp_path, monkeypatch):
    """The synthetic-DK on-disk layout from the pipeline tests, reused."""
    tp._write_era5(tmp_path / "era5")
    fleet = tp._write_fleet(tmp_path / "observations/turbine" / "DK")
    monkeypatch.setattr(PyVWFPaths, "TURBINE_DATA", tmp_path / "observations/turbine")
    monkeypatch.setattr(PyVWFPaths, "ERA5_DATA", tmp_path / "era5")
    return {"root": tmp_path, "fleet": fleet}


def test_registry_has_the_affine_baseline():
    assert "affine-wind" in available_corrections()
    model = get_correction("affine-wind")
    assert isinstance(model, CorrectionModel)
    with pytest.raises(KeyError, match="Unknown correction model"):
        get_correction("no-such-model")


def test_register_rejects_duplicates_and_non_models():
    with pytest.raises(TypeError):
        register_correction(dict)  # type: ignore[arg-type]

    class Nameless(CorrectionModel):
        def fit(self, *a, **k):
            raise NotImplementedError

        def apply(self, *a, **k):
            raise NotImplementedError

    with pytest.raises(ValueError, match="non-empty 'name'"):
        register_correction(Nameless)


def test_affine_golden_regression_bit_for_bit(synthetic_dk):
    """AffineWindCorrection(seasons=None) == the legacy path, exactly."""
    gen_cf, turb_info, reanalysis, power_curves = train_set(
        "DK", calc_z0=True, mode="onshore"
    )

    # --- legacy path, as PyVWF.train(dask_n_workers=0) runs it ---
    bias_df, clus_info = cluster_train_set(gen_cf, "fixed", 2, turb_info)
    valid = bias_df[bias_df["obs"].notna() & (bias_df["obs"] > 0)].copy()
    valid["offset"] = valid.apply(
        correction.find_offset, args=(clus_info, reanalysis, power_curves), axis=1
    )
    zero_obs = bias_df[bias_df["obs"] == 0].copy()
    zero_obs["offset"] = 0.0
    nan_obs = bias_df[bias_df["obs"].isna()].copy()
    nan_obs["offset"] = np.nan
    legacy_bias = pd.concat([valid, zero_obs, nan_obs], ignore_index=True).sort_index()
    legacy_factors = format_bc_factors(legacy_bias, "fixed")
    legacy_ws, legacy_cf = wind.simulate_wind(
        reanalysis, clus_info, power_curves, legacy_factors, "fixed"
    )

    # --- harness path ---
    model = get_correction("affine-wind")
    factors, clus_info_h = model.fit(
        gen_cf, turb_info, reanalysis, power_curves, num_clusters=2, time_res="fixed"
    )
    ws, cf = model.apply(reanalysis, clus_info_h, power_curves, factors, "fixed")

    # Exact equality: no tolerances. Same code, same numbers.
    pd.testing.assert_frame_equal(clus_info_h, clus_info)
    pd.testing.assert_frame_equal(factors, legacy_factors)
    pd.testing.assert_frame_equal(ws, legacy_ws)
    pd.testing.assert_frame_equal(cf, legacy_cf)


def test_affine_fit_rejects_unknown_obs_level():
    model = get_correction("affine-wind")
    with pytest.raises(ValueError, match="obs_level"):
        model.fit(
            pd.DataFrame(),
            pd.DataFrame(),
            None,
            pd.DataFrame(),
            num_clusters=1,
            time_res="fixed",
            obs_level="farm",  # obs_unit value, not a pipeline branch
        )


# ---------------------------------------------------------------------------
# Season injection: the seam that lets SH regions reuse the validated apply
# path. Must-distinguish fixtures — the SH and default results are asserted
# to DIFFER, not merely to be individually plausible.
# ---------------------------------------------------------------------------

def _july_wind_and_factors():
    """Constant 10 m/s winds in July, with a factors table whose winter and
    summer scalars differ by 4x — so hemisphere mix-ups are unmissable."""
    times = pd.date_range("2019-07-01", periods=3, freq="D")
    ws = xr.DataArray(
        np.full((3, 2), 10.0),
        coords={"time": times, "turbine": [0, 1]},
        dims=["time", "turbine"],
    )
    turb_info = pd.DataFrame(
        {
            "ID": ["a", "b"],
            "cluster": [0, 0],
            "model": ["M", "M"],
            "capacity": [1.0, 1.0],
        }
    )
    factors = pd.DataFrame(
        {
            "cluster": [0, 0, 0, 0],
            "season": ["winter", "spring", "summer", "autumn"],
            "scalar": [2.0, 1.0, 0.5, 1.0],
            "offset": [0.0, 0.0, 0.0, 0.0],
        }
    )
    return ws, turb_info, factors


def test_correct_wind_speed_uses_explicit_seasons():
    ws, turb_info, factors = _july_wind_and_factors()

    # Default (NH): July is summer -> scalar 0.5 -> 5 m/s.
    cor_default = correct_wind_speed(ws, "season", factors, turb_info)
    assert float(cor_default.min()) == pytest.approx(5.0)
    assert float(cor_default.max()) == pytest.approx(5.0)

    # SH definitions: July is WINTER -> scalar 2.0 -> 20 m/s.
    cor_sh = correct_wind_speed(ws, "season", factors, turb_info, seasons=SH_SEASONS)
    assert float(cor_sh.min()) == pytest.approx(20.0)
    assert float(cor_sh.max()) == pytest.approx(20.0)

    # The two modes must produce different output on this fixture: a
    # season-handling regression that silently falls back to the NH map
    # cannot pass this test.
    assert float(cor_sh.max()) != float(cor_default.max())


def test_simulate_wind_forwards_seasons(synthetic_dk):
    """seasons= must flow through simulate_wind to the correction merge."""
    gen_cf, turb_info, reanalysis, power_curves = train_set(
        "DK", calc_z0=True, mode="onshore"
    )
    _, clus_info = cluster_train_set(gen_cf, "fixed", 2, turb_info)
    factors = pd.DataFrame(
        {
            "cluster": [0, 0, 0, 0, 1, 1, 1, 1],
            "season": ["winter", "spring", "summer", "autumn"] * 2,
            "scalar": [2.0, 1.0, 0.5, 1.0] * 2,
            "offset": [0.0] * 8,
        }
    )
    ws_default, _ = wind.simulate_wind(
        reanalysis, clus_info, power_curves, factors, "season"
    )
    ws_sh, _ = wind.simulate_wind(
        reanalysis, clus_info, power_curves, factors, "season", seasons=SH_SEASONS
    )
    ws_cols = [c for c in ws_default.columns if c != "time"]
    july_default = ws_default[pd.to_datetime(ws_default["time"]).dt.month == 7]
    july_sh = ws_sh[pd.to_datetime(ws_sh["time"]).dt.month == 7]
    # NH July scalar 0.5 vs SH July (winter) scalar 2.0: a 4x ratio.
    ratio = july_sh[ws_cols].to_numpy() / july_default[ws_cols].to_numpy()
    assert ratio == pytest.approx(4.0)
