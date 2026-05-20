"""Tests for distribution-aware quantile-mapping correction.

The headline test (`test_qm_fixes_distribution_where_linear_fails`) encodes the
scientific motivation for the module: a linear mean-matching correction can make
the means agree while leaving the variance badly biased, whereas quantile
mapping repairs the full distribution.
"""
import numpy as np
import pandas as pd
import pytest

from vwf.quantile_correction import (
    QuantileMapper,
    empirical_quantile_mapping,
    quantile_delta_mapping,
    fit_quantile_correction_table,
    apply_quantile_correction,
    fit_quantile_factor_frame,
    apply_quantile_factor_frame,
)
from vwf.distribution_metrics import variance_ratio, wasserstein


@pytest.fixture
def biased_pair():
    """Model over-disperses and is shifted relative to obs (same RNG draw)."""
    rng = np.random.default_rng(42)
    obs = rng.normal(8.0, 1.0, size=5000)
    # Model: inflated spread (x1.8) and a +1.5 m/s mean shift.
    model = 8.0 + 1.5 + (obs - 8.0) * 1.8 + rng.normal(0, 0.2, size=5000)
    return model, obs


def test_eqm_matches_observed_distribution(biased_pair):
    model, obs = biased_pair
    corrected = empirical_quantile_mapping(model, obs, model)
    assert np.mean(corrected) == pytest.approx(np.mean(obs), abs=0.1)
    assert np.std(corrected) == pytest.approx(np.std(obs), rel=0.05)
    assert wasserstein(corrected, obs) < wasserstein(model, obs)


def test_qm_fixes_distribution_where_linear_fails(biased_pair):
    """Linear mean-matching fixes the mean but not the variance; QM fixes both."""
    model, obs = biased_pair

    # Linear correction that exactly matches the means (alpha=1, beta = shift).
    beta = np.mean(obs) - np.mean(model)
    linear = model + beta

    qm = empirical_quantile_mapping(model, obs, model)

    # Both correct the mean.
    assert abs(np.mean(linear) - np.mean(obs)) < 0.1
    assert abs(np.mean(qm) - np.mean(obs)) < 0.1

    # Linear leaves variance badly inflated; QM repairs it.
    assert variance_ratio(linear, obs) > 2.0
    assert variance_ratio(qm, obs) == pytest.approx(1.0, abs=0.1)


def test_mapper_is_monotonic(biased_pair):
    model, obs = biased_pair
    mapper = QuantileMapper(n_quantiles=50).fit(model, obs)
    xs = np.linspace(model.min(), model.max(), 200)
    ys = mapper.transform(xs)
    assert np.all(np.diff(ys) >= -1e-9)


def test_multiplicative_kind_positive(biased_pair):
    model, obs = biased_pair
    corrected = empirical_quantile_mapping(
        np.abs(model), np.abs(obs), np.abs(model), kind="multiplicative"
    )
    assert np.all(corrected >= 0)


def test_transform_preserves_nan():
    model = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    obs = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    mapper = QuantileMapper(n_quantiles=5).fit(model, obs)
    out = mapper.transform([2.0, np.nan, 4.0])
    assert np.isnan(out[1])
    assert np.all(np.isfinite([out[0], out[2]]))


def test_extrapolation_constant_vs_clip(biased_pair):
    model, obs = biased_pair
    far = np.array([model.max() + 10.0])
    const = QuantileMapper(extrapolate="constant").fit(model, obs).transform(far)
    clip = QuantileMapper(extrapolate="clip").fit(model, obs).transform(far)
    # Constant extrapolation keeps the edge delta -> value stays above range;
    # clip pulls the input back to the fitted max before mapping.
    assert const[0] > clip[0]


def test_qdm_reduces_to_eqm_when_target_equals_reference(biased_pair):
    """With target == reference sample, QDM and EQM should closely agree."""
    model, obs = biased_pair
    eqm = empirical_quantile_mapping(model, obs, model, n_quantiles=200)
    qdm = quantile_delta_mapping(model, obs, model, n_quantiles=200)
    # Compare sorted distributions (QDM reorders by rank, EQM by value).
    assert np.mean(np.sort(qdm)) == pytest.approx(np.mean(np.sort(eqm)), abs=0.05)
    assert np.std(qdm) == pytest.approx(np.std(eqm), rel=0.05)


def test_qdm_preserves_model_trend(biased_pair):
    """QDM should retain a mean shift the model introduces between periods."""
    model_ref, obs_ref = biased_pair
    # Target period: model warms by +3 m/s relative to reference.
    model_target = model_ref + 3.0
    corrected = quantile_delta_mapping(model_ref, obs_ref, model_target)
    # The +3 model trend survives bias correction (within tolerance).
    trend_retained = np.mean(corrected) - empirical_quantile_mapping(
        model_ref, obs_ref, model_ref
    ).mean()
    assert trend_retained == pytest.approx(3.0, abs=0.3)


def test_fit_and_apply_correction_table():
    rng = np.random.default_rng(0)
    frames = []
    for cluster in (0, 1):
        for tslice in ("winter", "summer"):
            n = 500
            obs = rng.normal(0.4 + 0.1 * cluster, 0.1, n)
            sim = obs * 1.5 + 0.05  # biased
            frames.append(pd.DataFrame({
                "cluster": cluster, "time_slice": tslice,
                "obs": obs, "sim": sim,
            }))
    train = pd.concat(frames, ignore_index=True)

    mappers = fit_quantile_correction_table(train)
    assert len(mappers) == 4  # 2 clusters x 2 slices

    out = apply_quantile_correction(train, mappers, value_col="sim", out_col="cor")
    # Per group, corrected variance ratio should be near 1.
    for (_, _), grp in out.groupby(["cluster", "time_slice"]):
        assert variance_ratio(grp["cor"], grp["obs"]) == pytest.approx(1.0, abs=0.2)


def test_mapper_from_quantiles_roundtrip(biased_pair):
    model, obs = biased_pair
    original = QuantileMapper(n_quantiles=40).fit(model, obs)
    p, mq, oq = original.quantiles()
    rebuilt = QuantileMapper.from_quantiles(p, mq, oq, kind=original.kind)
    xs = np.linspace(model.min(), model.max(), 50)
    np.testing.assert_allclose(original.transform(xs), rebuilt.transform(xs))


def _synthetic_gen_cf(seed=0):
    """Monthly-resolution gen_cf-like frame: cluster x time slice, biased sim.

    Means/spreads are chosen to stay clear of the 0/1 capacity-factor ceilings
    so the synthetic case mirrors realistic monthly CF (which rarely saturates).
    """
    rng = np.random.default_rng(seed)
    frames = []
    for cluster in (0, 1):
        for tslice in ("winter", "summer"):
            n = 400
            obs = np.clip(rng.normal(0.28 + 0.05 * cluster, 0.08, n), 0.01, 0.95)
            sim = np.clip(obs * 1.4 + 0.04, 0.01, 0.99)  # over-dispersed + offset
            frames.append(pd.DataFrame({
                "cluster": cluster, "season": tslice,
                "obs": obs, "sim": sim,
            }))
    return pd.concat(frames, ignore_index=True)


def test_fit_quantile_factor_frame_schema():
    gen_cf = _synthetic_gen_cf()
    frame = fit_quantile_factor_frame(gen_cf, time_res="season", n_quantiles=20)
    assert set(["cluster", "season", "p", "model_q", "obs_q", "kind"]).issubset(frame.columns)
    # 2 clusters x 2 slices x 20 knots
    assert len(frame) == 2 * 2 * 20


def test_factor_frame_csv_roundtrip_and_correction(tmp_path):
    gen_cf = _synthetic_gen_cf(seed=1)
    frame = fit_quantile_factor_frame(gen_cf, time_res="season", n_quantiles=50)

    # Persist and reload, mimicking PyVWF writing factors to disk.
    path = tmp_path / "qm_factors_season_2.csv"
    frame.to_csv(path, index=False)
    loaded = pd.read_csv(path)

    corrected = apply_quantile_factor_frame(
        gen_cf, loaded, time_res="season", value_col="sim", out_col="cor"
    )
    # The corrected distribution should match obs per group.
    for (_, _), grp in corrected.groupby(["cluster", "season"]):
        assert variance_ratio(grp["cor"], grp["obs"]) == pytest.approx(1.0, abs=0.25)
        assert abs(grp["cor"].mean() - grp["obs"].mean()) < 0.02


def test_apply_factor_frame_passthrough_unknown_group():
    gen_cf = _synthetic_gen_cf()
    frame = fit_quantile_factor_frame(gen_cf, time_res="season", n_quantiles=10)
    new = pd.DataFrame({"cluster": [9], "season": ["spring"], "sim": [0.5]})
    out = apply_quantile_factor_frame(new, frame, time_res="season")
    assert out["cor"].iloc[0] == 0.5


def test_apply_passthrough_for_unknown_group():
    train = pd.DataFrame({
        "cluster": [0] * 100, "time_slice": ["winter"] * 100,
        "obs": np.linspace(0, 1, 100), "sim": np.linspace(0, 2, 100),
    })
    mappers = fit_quantile_correction_table(train)
    test = pd.DataFrame({
        "cluster": [9], "time_slice": ["spring"], "sim": [1.23],
    })
    out = apply_quantile_correction(test, mappers, value_col="sim", out_col="cor")
    assert out["cor"].iloc[0] == 1.23  # unchanged
