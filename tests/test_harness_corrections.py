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
# path. Must-distinguish fixtures: the SH and default results are asserted
# to DIFFER, not merely to be individually plausible.
# ---------------------------------------------------------------------------

def _july_wind_and_factors():
    """Constant 10 m/s winds in July, with a factors table whose winter and
    summer scalars differ by 4x, so hemisphere mix-ups are unmissable."""
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


# ---------------------------------------------------------------------------
# scaled-affine: affine plus a per-cluster availability (loss) factor
# ---------------------------------------------------------------------------

def test_scaled_affine_is_registered():
    assert "scaled-affine" in available_corrections()


def test_scaled_affine_reduces_to_affine_when_no_loss(synthetic_dk):
    """On a fleet the affine correction already levels, every availability is
    1.0, so scaled-affine must match affine. The synthetic-DK fixture is such a
    fleet: it is planted with a wind-speed bias and nothing else."""
    gen_cf, turb_info, reanalysis, power_curves = train_set(
        "DK", calc_z0=True, mode="onshore"
    )
    affine = get_correction("affine-wind")
    scaled = get_correction("scaled-affine")

    a_fac, a_ci = affine.fit(gen_cf, turb_info, reanalysis, power_curves,
                             num_clusters=2, time_res="fixed")
    s_fac, s_ci = scaled.fit(gen_cf, turb_info, reanalysis, power_curves,
                             num_clusters=2, time_res="fixed")

    assert "avail" in s_fac.columns
    # No planted loss, so availability sits at its ceiling for every cluster.
    assert (s_fac["avail"] > 0.98).all()
    # The scalar and offset are the affine ones, untouched.
    merged = a_fac.merge(s_fac, on=["cluster", "fixed"], suffixes=("_a", "_s"))
    np.testing.assert_allclose(merged["scalar_a"], merged["scalar_s"])
    np.testing.assert_allclose(merged["offset_a"], merged["offset_s"])

    _, a_cf = affine.apply(reanalysis, a_ci, power_curves, a_fac, "fixed")
    _, s_cf = scaled.apply(reanalysis, s_ci, power_curves, s_fac, "fixed")
    # avail ~ 1 means the corrected CF is within rounding of the affine one.
    common = [c for c in a_cf.columns if c != "time"]
    np.testing.assert_allclose(
        a_cf[common].to_numpy(), s_cf[common].to_numpy(), rtol=0.02, atol=0.01
    )


def test_scaled_affine_availability_is_redundant_with_the_offset(synthetic_dk):
    """Planting a uniform level loss does NOT drive availability below 1,
    because the affine offset re-fits to absorb the level. This is the
    documented redundancy (docs/findings/region-south-america.md): the
    availability term only bites when the affine fit cannot reach the level, not
    in the ordinary case. Halving every observation is exactly such an ordinary
    level shift, and availability stays at its ceiling."""
    gen_cf, turb_info, reanalysis, power_curves = train_set(
        "DK", calc_z0=True, mode="onshore"
    )
    scaled = get_correction("scaled-affine")
    lossy = gen_cf.copy()
    lossy["obs"] = lossy["obs"] * 0.5
    fac, _ = scaled.fit(lossy, turb_info, reanalysis, power_curves,
                        num_clusters=2, time_res="fixed")
    assert (fac["avail"] > 0.9).all()


def _wide_cf(times, per_id):
    """A (time x ID) corrected-CF frame from {ID: constant cf}."""
    data = {"time": times}
    data.update({i: [v] * len(times) for i, v in per_id.items()})
    return pd.DataFrame(data)


def test_fit_availability_is_obs_over_corrected_level_clipped():
    """Unit test of the availability computation, bypassing the affine re-fit
    that hides it. A cluster the corrected sim over-predicts gets a<1; one it
    under-predicts is clipped to 1, because boosting output is not a loss."""
    from vwf.harness.corrections import ScaledAffineWindCorrection

    times = pd.date_range("2020-01-01", periods=6, freq="MS")
    clus_info = pd.DataFrame({
        "ID": ["a", "b"], "cluster": [0, 1], "capacity": [1000.0, 1000.0],
    })
    # Cluster 0 sim 0.40, cluster 1 sim 0.20.
    cor_cf = _wide_cf(times, {"a": 0.40, "b": 0.20})
    # Cluster 0 observed 0.20 (sim over-predicts 2x -> a = 0.5);
    # cluster 1 observed 0.40 (sim under-predicts -> a clipped to 1).
    gen_cf = pd.DataFrame({
        "ID": ["a"] * 6 + ["b"] * 6,
        "obs": [0.20] * 6 + [0.40] * 6,
    })
    avail = ScaledAffineWindCorrection._fit_availability(
        cor_cf, gen_cf, clus_info, obs_level="turbine"
    ).set_index("cluster")["avail"]
    assert avail[0] == pytest.approx(0.5)
    assert avail[1] == pytest.approx(1.0)


def test_scaled_affine_apply_scales_by_cluster_availability():
    """apply multiplies each grid point's corrected CF by its cluster's a."""
    from vwf.harness.corrections import ScaledAffineWindCorrection

    model = ScaledAffineWindCorrection()
    clus_info = pd.DataFrame({
        "ID": ["a", "b"], "cluster": [0, 1], "capacity": [1.0, 1.0],
        "lat": [55.0, 56.0], "lon": [8.0, 9.0], "height": [100.0, 100.0],
        "model": ["m", "m"],
    })
    times = pd.date_range("2020-01-01", periods=3, freq="MS")

    class _Affine:
        def apply(self, *a, **k):
            return None, _wide_cf(times, {"a": 0.4, "b": 0.4})

    # Patch the parent apply to a known corrected CF, then check scaling.
    import vwf.harness.corrections as c
    orig = c.AffineWindCorrection.apply
    c.AffineWindCorrection.apply = lambda self, *a, **k: _Affine().apply()
    try:
        factors = pd.DataFrame({"cluster": [0, 1], "fixed": ["1/1", "1/1"],
                                "scalar": [1.0, 1.0], "offset": [0.0, 0.0],
                                "avail": [0.5, 1.0]})
        _, cf = model.apply(None, clus_info, None, factors, "fixed")
    finally:
        c.AffineWindCorrection.apply = orig
    assert cf["a"].iloc[0] == pytest.approx(0.2)  # 0.4 * 0.5
    assert cf["b"].iloc[0] == pytest.approx(0.4)  # 0.4 * 1.0
