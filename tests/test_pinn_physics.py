"""Tests for the differentiable forward operator and the learned corrections.

The operator is only useful if it is the SAME physics the incumbent pipeline
runs, plus terms that switch off cleanly. So the tests here fall into three
groups: parity with ``vwf.wind`` where the two should agree exactly, the
behaviour of the additions when they are switched off, and the structural
guarantees the model relies on to extrapolate -- above all that the terrain
speed-up is exactly zero on flat ground whatever the network has learned.
"""
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.interpolate import Akima1DInterpolator

from vwf import wind
from vwf.pinn.model import (
    DELTA_BOUNDS, ETA_BOUNDS, GAMMA_BOUNDS, PhysicsCorrection,
)
from vwf.pinn.physics import (
    PowerCurveBank, expected_cf, gauss_hermite, hub_wind_ratio, monthly_mean,
)


@pytest.fixture
def fine_curve():
    """A production-shaped curve: 0 to 40 m/s in 0.01 m/s steps."""
    speed = np.arange(0.0, 40.0 + 1e-9, 0.01)
    cut_in, rated, cut_out = 3.0, 13.0, 25.0
    cf = np.zeros_like(speed)
    ramp = (speed >= cut_in) & (speed < rated)
    cf[ramp] = ((speed[ramp] - cut_in) / (rated - cut_in)) ** 3
    cf[(speed >= rated) & (speed <= cut_out)] = 1.0
    return pd.DataFrame({"data$speed": speed, "GE.1.5sle": cf})


@pytest.fixture
def bank(fine_curve):
    return PowerCurveBank(
        fine_curve["data$speed"].to_numpy(),
        fine_curve["GE.1.5sle"].to_numpy()[None, :],
    )


# --------------------------------------------------------------- the curve ---
def test_bank_matches_akima_on_a_production_grid(fine_curve, bank):
    """Linear interpolation on the 0.01 m/s grid agrees with the incumbent Akima."""
    speeds = fine_curve["data$speed"].to_numpy()
    akima = Akima1DInterpolator(speeds, fine_curve["GE.1.5sle"].to_numpy())
    probe = np.arange(0.05, 39.9, 0.037)
    ours = bank(torch.tensor(probe, dtype=torch.float32),
                torch.zeros(len(probe), dtype=torch.long)).numpy()
    assert np.max(np.abs(ours - akima(probe))) < 1e-4


def test_bank_clamps_outside_the_grid(bank):
    """Speeds off either end of the table evaluate at the table's ends."""
    probe = torch.tensor([-10.0, 0.0, 40.0, 1e6])
    out = bank(probe, torch.zeros(4, dtype=torch.long))
    assert torch.isfinite(out).all()
    assert out[0] == out[1]           # below the grid clamps to the first sample
    assert out[2] == out[3]           # above it clamps to the last


def test_bank_rejects_a_non_uniform_grid():
    with pytest.raises(ValueError, match="uniform"):
        PowerCurveBank(np.array([0.0, 1.0, 3.0]), np.zeros((1, 3)))


def test_bank_is_differentiable_in_wind_speed(bank):
    """The gradient that makes end-to-end fitting possible exists and is right."""
    u = torch.tensor([8.0], requires_grad=True)
    bank(u, torch.zeros(1, dtype=torch.long)).backward()
    # On the cubic ramp between cut-in and rated the curve is strictly rising.
    assert u.grad.item() > 0


# ------------------------------------------------------------- the profile ---
def test_log_profile_matches_the_incumbent_formula():
    h, z0 = torch.tensor([40.0, 100.0, 140.0]), torch.tensor([0.03, 0.03, 0.03])
    ours = hub_wind_ratio(h, z0=z0, profile="log").numpy()
    expected = np.log(h.numpy() / 0.03) / np.log(100.0 / 0.03)
    assert np.allclose(ours, expected, atol=1e-6)


def test_log_profile_reproduces_vwf_wind(reanalysis, turbines, power_curve):
    """Against the incumbent simulation itself, with roughness constant.

    ``vwf.wind.interpolate_wind`` applies the profile on the grid and then
    interpolates to turbines; the cache interpolates the fields first and
    applies the profile per turbine. With a spatially constant roughness the two
    orders are algebraically identical, so this test pins the physics exactly.
    On real, varying roughness the orders differ slightly; that difference is
    measured against the published scorecard, not asserted here.
    """
    ws = wind.interpolate_wind(reanalysis, turbines)
    ratio = hub_wind_ratio(torch.tensor(turbines["height"].to_numpy(dtype=float)),
                           z0=torch.tensor(0.03), profile="log")
    per_turbine = np.stack([
        reanalysis["wnd100m"].interp(lon=lo, lat=la).values
        for lo, la in zip(turbines["lon"], turbines["lat"])
    ], axis=1)
    assert np.allclose(ws.values, per_turbine * ratio.numpy(), atol=1e-6)


def test_power_profile_is_unity_at_the_reference_height():
    r = hub_wind_ratio(torch.tensor(100.0), shear=torch.tensor([0.0, 0.2, 0.5]),
                       profile="power")
    assert torch.allclose(r, torch.ones(3))


def test_power_profile_needs_its_input():
    with pytest.raises(ValueError, match="shear"):
        hub_wind_ratio(torch.tensor(80.0), profile="power")
    with pytest.raises(ValueError, match="z0"):
        hub_wind_ratio(torch.tensor(80.0), profile="log")
    with pytest.raises(ValueError, match="unknown profile"):
        hub_wind_ratio(torch.tensor(80.0), shear=torch.tensor(0.1), profile="nope")


# ----------------------------------------------------- the sub-daily spread ---
def test_zero_spread_reduces_to_point_evaluation(bank):
    u = torch.tensor([4.0, 8.0, 12.0])
    idx = torch.zeros(3, dtype=torch.long)
    quad = gauss_hermite(5)
    at_mean = bank(u, idx)
    integrated = expected_cf(u, torch.zeros(3), idx, bank, quad)
    assert torch.allclose(at_mean, integrated, atol=1e-6)


def test_no_quadrature_reduces_to_point_evaluation(bank):
    u = torch.tensor([7.0])
    idx = torch.zeros(1, dtype=torch.long)
    assert torch.allclose(expected_cf(u, None, idx, bank, None), bank(u, idx))


def test_spread_raises_output_where_the_curve_is_convex(bank):
    """Jensen's inequality, which is exactly the daily-averaging bias.

    Just above cut-in the curve is convex, so averaging power over a day's wind
    gives MORE than the power at the day's mean wind. The incumbent evaluates at
    the mean and has no way to represent this.
    """
    u = torch.tensor([5.0])
    idx = torch.zeros(1, dtype=torch.long)
    quad = gauss_hermite(9)
    assert (expected_cf(u, torch.tensor([1.5]), idx, bank, quad)
            > expected_cf(u, torch.tensor([0.0]), idx, bank, quad))


def test_gauss_hermite_weights_are_a_probability_measure():
    for n in (3, 5, 7, 9):
        _, w = gauss_hermite(n)
        assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------- the aggregation ---
def test_monthly_mean_averages_within_months():
    daily = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    month_id = torch.tensor([0, 0, 0, 1, 1, 1])
    out = monthly_mean(daily, month_id, 2)
    assert torch.allclose(out, torch.tensor([[2.0, 3.0], [8.0, 9.0]]))


# --------------------------------------------------------------- the model ---
def _inputs(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, 14, generator=g), torch.randn(n, 4, generator=g))


def test_model_starts_at_the_identity_state():
    """No speed-up, no shear correction: the reanalysis is right until proven wrong."""
    m = PhysicsCorrection(14, 4)
    t, f = _inputs()
    gamma, delta, _, _ = m(t, f, torch.full((8,), 500.0))
    assert torch.allclose(gamma, torch.zeros(8), atol=1e-6)
    assert torch.allclose(delta, torch.zeros(8), atol=1e-6)


@pytest.mark.parametrize("hidden", [None, 16])
def test_speedup_is_pinned_to_zero_on_flat_ground(hidden):
    """The structural guarantee the extrapolation rests on.

    Whatever the amplitude network has learned -- here, deliberately randomised
    away from its initialisation -- a site with no sub-grid relief receives
    exactly no terrain correction.
    """
    m = PhysicsCorrection(14, 4, hidden=hidden)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 2.0)
    t, f = _inputs(seed=3)
    relief = torch.tensor([0.0, 0.0, 1e-6, 10.0, 100.0, 500.0, 1500.0, 3000.0])
    gamma, _, _, _ = m(t, f, relief)
    assert gamma[0] == 0.0 and gamma[1] == 0.0
    assert abs(float(gamma[2].detach())) < 1e-6
    assert float(gamma[7].abs().detach()) > 0.0        # and it is not simply dead


def test_bounds_hold_under_extreme_inputs():
    m = PhysicsCorrection(14, 4)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 50.0)
    t = torch.randn(64, 14) * 100.0
    f = torch.randn(64, 4) * 100.0
    gamma, delta, eta, kappa = m(t, f, torch.rand(64) * 3000.0)
    assert gamma.min() >= GAMMA_BOUNDS[0] - 1e-6
    assert gamma.max() <= GAMMA_BOUNDS[1] + 1e-6
    assert delta.min() >= DELTA_BOUNDS[0] - 1e-6
    assert delta.max() <= DELTA_BOUNDS[1] + 1e-6
    assert eta.min() >= ETA_BOUNDS[0] - 1e-6
    assert eta.max() <= ETA_BOUNDS[1] + 1e-6
    assert 0.0 <= float(kappa.detach()) <= 1.5 + 1e-6


def test_ablation_matches_the_model_in_capacity():
    """Gate P3 is only a fair test if the two arms differ ONLY in the physics."""
    a = PhysicsCorrection(14, 4, physics=True)
    b = PhysicsCorrection(14, 4, physics=False)
    assert (sum(p.numel() for p in a.parameters())
            == sum(p.numel() for p in b.parameters()))
    t, f = _inputs()
    # Both start at the same physical state; they diverge only once fitted.
    for x, y in zip(a(t, f, torch.full((8,), 400.0)), b(t, f, torch.full((8,), 400.0))):
        assert torch.allclose(torch.as_tensor(x), torch.as_tensor(y), atol=1e-6)


def test_ablation_has_no_relief_pin():
    b = PhysicsCorrection(14, 4, physics=False)
    with torch.no_grad():
        for p in b.parameters():
            p.add_(torch.randn_like(p) * 2.0)
    t, f = _inputs(seed=5)
    gamma, _, _, _ = b(t, f, torch.zeros(8))
    assert float(gamma.abs().max().detach()) > 0.0


# ---------------------------------------------------------------- density ---
def test_density_is_unity_at_sea_level_and_below():
    from vwf.pinn.physics import air_density_ratio, density_speed_factor
    z = torch.tensor([-50.0, -1.0, 0.0])
    assert torch.allclose(air_density_ratio(z), torch.ones(3), atol=1e-6)
    assert torch.allclose(density_speed_factor(z), torch.ones(3), atol=1e-6)


def test_density_matches_the_standard_atmosphere():
    """Spot values from the ISA, which the formula must reproduce."""
    from vwf.pinn.physics import air_density_ratio
    z = torch.tensor([1000.0, 2000.0, 3000.0])
    # ISA density 1.1117, 1.0065, 0.9093 kg/m3 against 1.225 at sea level.
    expected = torch.tensor([1.1117, 1.0065, 0.9093]) / 1.225
    assert torch.allclose(air_density_ratio(z), expected, atol=2e-3)


def test_density_speed_factor_is_the_cube_root():
    from vwf.pinn.physics import air_density_ratio, density_speed_factor
    z = torch.tensor([0.0, 500.0, 1500.0, 2500.0])
    assert torch.allclose(density_speed_factor(z), air_density_ratio(z) ** (1 / 3),
                          atol=1e-6)


def test_density_correction_recovers_the_cube_law_exactly():
    """On a pure v-cubed curve the speed correction reproduces power ~ density.

    This is the identity the IEC equivalent-speed form is built to satisfy, and
    it is the sharpest available check that the exponent is right.
    """
    from vwf.pinn.physics import air_density_ratio, density_speed_factor
    speeds = np.arange(0.0, 40.0 + 1e-9, 0.01)
    cubic = PowerCurveBank(speeds, ((speeds / 40.0) ** 3)[None, :])
    u = torch.tensor([8.0, 12.0, 20.0])
    idx = torch.zeros(3, dtype=torch.long)
    z = torch.tensor([1500.0, 1500.0, 1500.0])
    ratio = cubic(u * density_speed_factor(z), idx) / cubic(u, idx)
    assert torch.allclose(ratio, air_density_ratio(z), atol=1e-4)


def test_density_reduces_output_at_altitude(bank):
    """A real curve, with a cut-in offset, is MORE density-sensitive than v-cubed.

    Below rated the curve rises as (v - v_cut_in) cubed, so a given fractional
    loss of wind speed costs proportionally more power than the ideal cube law
    predicts. The reduction must therefore exceed the density deficit itself.
    """
    from vwf.pinn.physics import air_density_ratio, density_speed_factor
    u = torch.tensor([8.0])
    idx = torch.zeros(1, dtype=torch.long)
    z = torch.tensor([1200.0])
    ratio = float(bank(u * density_speed_factor(z), idx) / bank(u, idx))
    assert ratio < float(air_density_ratio(z))     # steeper than the cube law
    assert 0.75 < ratio < 1.0                       # but not pathological
