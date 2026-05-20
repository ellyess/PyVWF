"""Tests for distributional skill metrics."""
import numpy as np
import pytest

from vwf.distribution_metrics import (
    variance_ratio,
    std_ratio,
    quantile_bias,
    wasserstein,
    ks_statistic,
    ramp_metrics,
    distribution_report,
)


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def test_variance_ratio_identity(rng):
    x = rng.normal(0, 1, 2000)
    assert variance_ratio(x, x) == pytest.approx(1.0)
    assert std_ratio(x, x) == pytest.approx(1.0)


def test_variance_ratio_detects_overdispersion(rng):
    obs = rng.normal(0, 1, 5000)
    sim = rng.normal(0, 2, 5000)
    assert variance_ratio(sim, obs) == pytest.approx(4.0, rel=0.1)


def test_wasserstein_zero_for_identical():
    x = np.linspace(0, 1, 100)
    assert wasserstein(x, x) == pytest.approx(0.0, abs=1e-9)


def test_wasserstein_increases_with_shift(rng):
    obs = rng.normal(0, 1, 3000)
    near = obs + 0.5
    far = obs + 3.0
    assert wasserstein(near, obs) < wasserstein(far, obs)


def test_ks_statistic_bounds(rng):
    obs = rng.normal(0, 1, 1000)
    assert 0.0 <= ks_statistic(obs, obs) <= 1.0
    shifted = obs + 10.0
    assert ks_statistic(shifted, obs) == pytest.approx(1.0, abs=0.05)


def test_quantile_bias_signs(rng):
    obs = rng.normal(5, 1, 5000)
    sim = obs + 2.0
    qb = quantile_bias(sim, obs, quantiles=(0.1, 0.9))
    assert qb["q10_bias"] == pytest.approx(2.0, abs=0.2)
    assert qb["q90_bias"] == pytest.approx(2.0, abs=0.2)


def test_ramp_metrics_detect_smoothing():
    t = np.linspace(0, 20 * np.pi, 1000)
    obs = np.sin(t)
    smooth = 0.3 * np.sin(t)  # damped ramps
    rm = ramp_metrics(smooth, obs)
    assert rm["ramp_std_ratio"] < 1.0


def test_distribution_report_columns(rng):
    obs = rng.normal(0.4, 0.1, 2000)
    report = distribution_report(
        {"raw": obs * 1.5, "good": obs}, obs
    )
    assert "variance_ratio" in report.columns
    assert "wasserstein" in report.columns
    assert report.loc["good", "variance_ratio"] == pytest.approx(1.0, abs=0.05)
    assert report.loc["good", "wasserstein"] < report.loc["raw", "wasserstein"]
