"""Tests for the differentiable forward operator and the learned corrections.

The operator is only useful if it is the SAME physics the incumbent pipeline
runs, plus terms that switch off cleanly. So the tests here fall into three
groups: parity with ``pyvwf.wind`` where the two should agree exactly, the
behaviour of the additions when they are switched off, and the structural
guarantees the model relies on to extrapolate -- above all that the terrain
speed-up is exactly zero on flat ground whatever the network has learned.
"""

import numpy as np
import pandas as pd
import pytest

# torch lives in the optional [pinn] extra, so that installing PyVWF to run the
# affine pipeline does not pull a deep-learning stack. CI's test matrix does not
# install it; CI's extras job does, and runs this file.
pytest.importorskip("torch")

import torch  # noqa: E402
from scipy.interpolate import Akima1DInterpolator  # noqa: E402

from pyvwf import wind  # noqa: E402
from pyvwf.pinn.model import (  # noqa: E402
    DELTA_BOUNDS,
    ETA_BOUNDS,
    GAMMA_BOUNDS,
    PhysicsCorrection,
)
from pyvwf.pinn.physics import (  # noqa: E402
    PowerCurveBank,
    expected_cf,
    gauss_hermite,
    hub_wind_ratio,
    monthly_mean,
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
    ours = bank(
        torch.tensor(probe, dtype=torch.float32), torch.zeros(len(probe), dtype=torch.long)
    ).numpy()
    assert np.max(np.abs(ours - akima(probe))) < 1e-4


def test_bank_clamps_outside_the_grid(bank):
    """Speeds off either end of the table evaluate at the table's ends."""
    probe = torch.tensor([-10.0, 0.0, 40.0, 1e6])
    out = bank(probe, torch.zeros(4, dtype=torch.long))
    assert torch.isfinite(out).all()
    assert out[0] == out[1]  # below the grid clamps to the first sample
    assert out[2] == out[3]  # above it clamps to the last


def test_bank_rejects_a_non_uniform_grid():
    with pytest.raises(ValueError, match="uniform"):
        PowerCurveBank(np.array([0.0, 1.0, 3.0]), np.zeros((1, 3)))


@pytest.mark.parametrize("speed", [8.0, 8.005])
def test_bank_is_differentiable_in_wind_speed(bank, speed):
    """The gradient that makes end-to-end fitting possible exists and is right.

    8.0 m/s sits on a grid knot and 8.005 between two. Under torch 2.14 the
    knot's gradient was zero until the fraction stopped being clamped inside
    the table; on 2.13 both passed.
    """
    u = torch.tensor([speed], requires_grad=True)
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

    ``pyvwf.wind.interpolate_wind`` applies the profile on the grid and then
    interpolates to turbines; the cache interpolates the fields first and
    applies the profile per turbine. With a spatially constant roughness the two
    orders are algebraically identical, so this test pins the physics exactly.
    On real, varying roughness the orders differ slightly; that difference is
    measured against the published scorecard, not asserted here.
    """
    ws = wind.interpolate_wind(reanalysis, turbines)
    ratio = hub_wind_ratio(
        torch.tensor(turbines["height"].to_numpy(dtype=float)), z0=torch.tensor(0.03), profile="log"
    )
    per_turbine = np.stack(
        [
            reanalysis["wnd100m"].interp(lon=lo, lat=la).values
            for lo, la in zip(turbines["lon"], turbines["lat"])
        ],
        axis=1,
    )
    assert np.allclose(ws.values, per_turbine * ratio.numpy(), atol=1e-6)


def test_power_profile_is_unity_at_the_reference_height():
    r = hub_wind_ratio(torch.tensor(100.0), shear=torch.tensor([0.0, 0.2, 0.5]), profile="power")
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
    assert expected_cf(u, torch.tensor([1.5]), idx, bank, quad) > expected_cf(
        u, torch.tensor([0.0]), idx, bank, quad
    )


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
    m = PhysicsCorrection(14, 4, init_scale=0.0)
    t, f = _inputs()
    gamma, delta, _, _ = m(t, f, torch.full((8,), 500.0))
    assert torch.allclose(gamma, torch.zeros(8), atol=1e-6)
    assert torch.allclose(delta, torch.zeros(8), atol=1e-6)


def test_default_init_is_perturbed_but_still_near_identity():
    """Seeds must actually vary the fit, without moving the physical start.

    With zero weights the optimisation is fully deterministic and seeds measure
    nothing -- a spread of 0.0000 across seeds that looks like robustness and is
    not. The default perturbs the weights just enough to break that, while
    leaving the model's starting physics where it was stated to be.
    """
    torch.manual_seed(0)
    a = PhysicsCorrection(14, 4)
    torch.manual_seed(1)
    b = PhysicsCorrection(14, 4)
    t, f = _inputs()
    relief = torch.full((8,), 500.0)
    ga, da, _, _ = a(t, f, relief)
    gb, db, _, _ = b(t, f, relief)
    assert not torch.allclose(ga, gb)  # seeds differ
    # Still identity-ish: a log speed-up of 0.08 is a 8% departure, which is
    # small against the 0.67x-2.46x range the term is allowed to reach.
    assert ga.abs().max() < 0.08
    assert da.abs().max() < 0.08


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
    assert float(gamma[7].abs().detach()) > 0.0  # and it is not simply dead


@pytest.mark.parametrize("seed", range(8))
def test_bounds_hold_under_extreme_inputs(seed):
    """Bounds, and finiteness, under parameters and inputs far outside training.

    Seeded and swept rather than drawn once: the relief scale is learned in log
    space, and a perturbation large enough to underflow its exponential used to
    turn relief/scale into an infinity and gamma into a NaN in a few per cent of
    draws -- exactly the kind of failure a single random test misses.
    """
    g = torch.Generator().manual_seed(seed)
    m = PhysicsCorrection(14, 4)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn(p.shape, generator=g) * 50.0)
    t = torch.randn(64, 14, generator=g) * 100.0
    f = torch.randn(64, 4, generator=g) * 100.0
    gamma, delta, eta, kappa = m(t, f, torch.rand(64, generator=g) * 3000.0)
    for name, v in (("gamma", gamma), ("delta", delta), ("eta", eta)):
        assert torch.isfinite(v).all(), f"{name} is not finite"
    assert gamma.min() >= GAMMA_BOUNDS[0] - 1e-6
    assert gamma.max() <= GAMMA_BOUNDS[1] + 1e-6
    assert delta.min() >= DELTA_BOUNDS[0] - 1e-6
    assert delta.max() <= DELTA_BOUNDS[1] + 1e-6
    assert eta.min() >= ETA_BOUNDS[0] - 1e-6
    assert eta.max() <= ETA_BOUNDS[1] + 1e-6
    assert 0.0 <= float(kappa.detach()) <= 1.5 + 1e-6


def test_ablation_matches_the_model_in_capacity():
    """Gate P3 is only a fair test if the two arms differ ONLY in the physics."""
    a = PhysicsCorrection(14, 4, physics=True, init_scale=0.0)
    b = PhysicsCorrection(14, 4, physics=False, init_scale=0.0)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
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
    from pyvwf.pinn.physics import air_density_ratio, density_speed_factor

    z = torch.tensor([-50.0, -1.0, 0.0])
    assert torch.allclose(air_density_ratio(z), torch.ones(3), atol=1e-6)
    assert torch.allclose(density_speed_factor(z), torch.ones(3), atol=1e-6)


def test_density_matches_the_standard_atmosphere():
    """Spot values from the ISA, which the formula must reproduce."""
    from pyvwf.pinn.physics import air_density_ratio

    z = torch.tensor([1000.0, 2000.0, 3000.0])
    # ISA density 1.1117, 1.0065, 0.9093 kg/m3 against 1.225 at sea level.
    expected = torch.tensor([1.1117, 1.0065, 0.9093]) / 1.225
    assert torch.allclose(air_density_ratio(z), expected, atol=2e-3)


def test_density_speed_factor_is_the_cube_root():
    from pyvwf.pinn.physics import air_density_ratio, density_speed_factor

    z = torch.tensor([0.0, 500.0, 1500.0, 2500.0])
    assert torch.allclose(density_speed_factor(z), air_density_ratio(z) ** (1 / 3), atol=1e-6)


def test_density_correction_recovers_the_cube_law_exactly():
    """On a pure v-cubed curve the speed correction reproduces power ~ density.

    This is the identity the IEC equivalent-speed form is built to satisfy, and
    it is the sharpest available check that the exponent is right.
    """
    from pyvwf.pinn.physics import air_density_ratio, density_speed_factor

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
    from pyvwf.pinn.physics import air_density_ratio, density_speed_factor

    u = torch.tensor([8.0])
    idx = torch.zeros(1, dtype=torch.long)
    z = torch.tensor([1200.0])
    ratio = float(bank(u * density_speed_factor(z), idx) / bank(u, idx))
    assert ratio < float(air_density_ratio(z))  # steeper than the cube law
    assert 0.75 < ratio < 1.0  # but not pathological


# ------------------------------------------------------------ array losses ---
def test_array_efficiency_is_one_at_zero_density():
    """An isolated turbine wakes nobody, so the term must be exactly inert."""
    m = PhysicsCorrection(14, 4, wake=True, init_scale=0.0)
    assert float(m.array_efficiency(torch.zeros(1))) == pytest.approx(1.0)


def test_array_efficiency_falls_monotonically_with_density():
    m = PhysicsCorrection(14, 4, wake=True, init_scale=0.0)
    with torch.no_grad():
        m.raw_wake.fill_(0.0)  # a sizeable coefficient
    d = torch.tensor([0.0, 0.05, 0.14, 0.5, 2.0, 9.0])
    eff = m.array_efficiency(d)
    assert torch.all(eff[1:] < eff[:-1])
    assert float(eff.max()) <= 1.0 and float(eff.min()) > 0.0


def test_array_efficiency_saturates_rather_than_collapsing():
    """Deep-array losses approach an asymptote; they do not decay to nothing.

    This is why the term is hyperbolic and not exponential: observed densities
    reach 9 MW/km2, where an exponential steep enough to matter at the median
    (0.14) would leave essentially no output at all.
    """
    m = PhysicsCorrection(14, 4, wake=True, init_scale=0.0)
    with torch.no_grad():
        m.raw_wake.fill_(0.0)
    c = float(m.wake_coefficient())
    hyper = float(m.array_efficiency(torch.tensor([9.0])))
    exponential = float(torch.exp(-torch.tensor(c * 9.0)))
    assert hyper > exponential * 2


def test_wake_coefficient_cannot_go_negative():
    """A negative coefficient would mean crowding turbines RAISES output."""
    m = PhysicsCorrection(14, 4, wake=True, init_scale=0.0)
    with torch.no_grad():
        m.raw_wake.fill_(-50.0)
    assert float(m.wake_coefficient()) >= 0.0


def test_wake_off_reproduces_the_previous_efficiency_exactly():
    """Existing results must reproduce, so the default has to be a no-op."""
    t, f = _inputs()
    relief = torch.full((8,), 400.0)
    torch.manual_seed(0)
    off = PhysicsCorrection(14, 4, wake=False, init_scale=0.0)
    torch.manual_seed(0)
    on = PhysicsCorrection(14, 4, wake=True, init_scale=0.0)
    assert torch.allclose(off(t, f, relief)[2], on(t, f, relief, torch.zeros(8))[2], atol=1e-6)


def test_wake_needs_a_density_to_apply():
    m = PhysicsCorrection(14, 4, wake=True)
    t, f = _inputs()
    with pytest.raises(ValueError, match="capdens"):
        m(t, f, torch.full((8,), 400.0))


def test_efficiency_respects_its_floor_under_extreme_density():
    from pyvwf.pinn.model import ETA_FLOOR

    m = PhysicsCorrection(14, 4, wake=True, init_scale=0.0)
    with torch.no_grad():
        m.raw_wake.fill_(3.0)
    t, f = _inputs()
    eta = m(t, f, torch.full((8,), 400.0), torch.full((8,), 500.0))[2]
    assert float(eta.min()) >= ETA_FLOOR - 1e-6


# ------------------------------------------------------- profile curvature ---
def test_roughness_inversion_round_trips():
    """The shear-to-roughness inversion must be exact, not approximate.

    It is derived in closed form from the neutral log profile, so a measured
    exponent of 0.145 has to come back as z0 = 0.03 m to full precision; any
    drift here would silently bias every hub height.
    """
    from pyvwf.pinn.physics import roughness_from_shear

    for z0_true in (0.0002, 0.01, 0.03, 0.1, 0.5):
        r = np.log(100 / z0_true) / np.log(10 / z0_true)  # w100/w10
        shear = np.log(r) / np.log(10.0)
        z0 = float(roughness_from_shear(torch.tensor([shear])))
        assert z0 == pytest.approx(z0_true, rel=1e-4)


def test_shear_log_profile_reproduces_the_log_law_exactly():
    """The point of the change: right curvature away from the measured band.

    A power law fitted on 10-100 m and extrapolated errs by about 1% at 30 m and
    1% the other way at 150 m. The shear-log form inverts the same measurement
    and must land on the log law it came from, at every height.
    """
    z0_true = 0.03
    r = np.log(100 / z0_true) / np.log(10 / z0_true)
    shear = torch.tensor(float(np.log(r) / np.log(10.0)))
    for h in (20.0, 30.0, 45.0, 80.0, 120.0, 150.0):
        hh = torch.tensor(h)
        got = float(hub_wind_ratio(hh, shear=shear, profile="shear-log"))
        want = float(np.log(h / z0_true) / np.log(100 / z0_true))
        assert got == pytest.approx(want, abs=1e-5)


def test_power_law_errs_in_opposite_directions_either_side_of_100m():
    """The fact that corrected the reasoning behind this change.

    A power law extrapolated out of its fitting range does NOT over-predict at
    both ends: it gives less wind below 100 m and more above. So curvature can
    explain a tall-turbine over-prediction and cannot explain a short-turbine
    one.
    """
    z0_true = 0.03
    r = np.log(100 / z0_true) / np.log(10 / z0_true)
    shear = torch.tensor(float(np.log(r) / np.log(10.0)))

    def err(h):
        pw = float(hub_wind_ratio(torch.tensor(h), shear=shear, profile="power"))
        return pw / float(np.log(h / z0_true) / np.log(100 / z0_true)) - 1

    assert err(30.0) < -0.005  # under-predicts well below the band
    assert err(150.0) > +0.005  # over-predicts well above it


def test_shear_log_is_unity_at_the_reference_height():
    shear = torch.tensor([0.10, 0.145, 0.25])
    r = hub_wind_ratio(torch.tensor(100.0), shear=shear, profile="shear-log")
    assert torch.allclose(r, torch.ones(3), atol=1e-5)


def test_log_z0_offset_moves_the_profile_monotonically():
    """The learned quantity in this mode: rougher ground, more shear."""
    shear = torch.tensor(0.145)
    h = torch.tensor(45.0)
    ratios = [
        float(hub_wind_ratio(h, shear=shear, profile="shear-log", log_z0_offset=torch.tensor(d)))
        for d in (-2.0, -1.0, 0.0, 1.0, 2.0)
    ]
    assert all(a > b for a, b in zip(ratios, ratios[1:]))  # rougher -> less wind at 45 m


def test_shear_log_survives_a_degenerate_profile():
    """A uniform or reversed 10-100 m profile has no roughness that explains it."""
    from pyvwf.pinn.physics import roughness_from_shear

    out = roughness_from_shear(torch.tensor([-0.05, 0.0, 1e-9, 0.6]))
    assert torch.isfinite(out).all()
    assert (out > 0).all()


# --------------------------------------------------------- joint constraint ---
@pytest.mark.parametrize("scale", [1.0, 0.6, 0.35, 0.0])
def test_bound_scale_shrinks_every_term_toward_its_start(scale):
    """The joint knob must narrow the range without moving the starting state.

    Shrinking toward the midpoint instead would move where the model begins as
    well as how far it can go, and the experiment needs those separable.
    """
    m = PhysicsCorrection(14, 4, bound_scale=scale, init_scale=0.0)
    t, f = _inputs()
    gamma, delta, eta, kappa = m(t, f, torch.full((8,), 500.0))
    assert torch.allclose(gamma, torch.zeros(8), atol=1e-3)
    assert torch.allclose(delta, torch.zeros(8), atol=1e-3)
    assert float(eta[0]) == pytest.approx(0.90, abs=1e-3)
    assert float(kappa.detach()) == pytest.approx(0.50, abs=1e-3)


def test_bound_scale_actually_narrows_the_reachable_range():
    from pyvwf.pinn.model import GAMMA_BOUNDS

    wide = PhysicsCorrection(14, 4, bound_scale=1.0)._bounds(GAMMA_BOUNDS, 0.0)
    tight = PhysicsCorrection(14, 4, bound_scale=0.35)._bounds(GAMMA_BOUNDS, 0.0)
    assert (tight[1] - tight[0]) < (wide[1] - wide[0])
    assert tight[0] > wide[0] and tight[1] < wide[1]


def test_bound_scale_one_is_exactly_the_unconstrained_model():
    """The sweep's control arm must be bit-identical to the current model."""
    from pyvwf.pinn.model import GAMMA_BOUNDS, ETA_BOUNDS

    m = PhysicsCorrection(14, 4, bound_scale=1.0)
    assert m._bounds(GAMMA_BOUNDS, 0.0) == GAMMA_BOUNDS
    assert m._bounds(ETA_BOUNDS, 0.90) == ETA_BOUNDS


# ------------------------------------------------------------- run records ---
def test_off_curve_tally_counts_what_the_bank_clamps(bank):
    """The bank returns an end value outside its table where the harness
    returns nothing, so the tally is the only record that it happened."""
    from pyvwf.pinn.train import count_off_curve, off_curve_shares

    u = torch.tensor([[-1.0, 5.0], [41.0, 12.0], [8.0, 45.0]])  # (days, units)
    capacity = torch.tensor([1000.0, 3000.0])
    tally: dict = {}
    count_off_curve(u, capacity, bank, tally)
    shares = off_curve_shares(tally)
    assert shares["unit_days"] == 6
    assert shares["off_curve_below_days"] == 1
    assert shares["off_curve_above_days"] == 2
    # Capacity-weighted over unit-days: 3 x (1000 + 3000) = 12000 in all.
    assert shares["off_curve_below_share"] == pytest.approx(1000.0 / 12000.0)
    assert shares["off_curve_above_share"] == pytest.approx(4000.0 / 12000.0)

    inside: dict = {}
    count_off_curve(torch.full((3, 2), 7.0), capacity, bank, inside)
    assert off_curve_shares(inside)["off_curve_below_share"] == 0.0
    assert off_curve_shares(inside)["off_curve_above_share"] == 0.0


def test_a_unit_with_no_wind_is_dropped_and_recorded(fine_curve):
    """A unit outside the loaded extent has no wind at all. It is dropped, and
    the tensors say which unit and how much capacity went with it."""
    from pyvwf.pinn.cache import RegionCache
    from pyvwf.pinn.terrain import FEATURES
    from pyvwf.pinn.train import RegionTensors

    days = pd.date_range("2015-01-01", periods=31, freq="D")
    meta = pd.DataFrame(
        {
            "ID": ["a", "b", "c"],
            "lon": [8.0, 8.1, 14.9],
            "lat": [55.0, 55.1, 55.1],
            "capacity": [2000.0, 1000.0, 1000.0],
            "height": [80.0, 80.0, 80.0],
            "type": ["onshore"] * 3,
            "model": ["GE.1.5sle"] * 3,
            **{f: [1.0, 2.0, 3.0] for f in FEATURES},
        }
    )
    w = np.full((31, 3), 8.0, dtype="float32")
    w[:, 2] = np.nan
    obs = pd.DataFrame({"ID": ["a", "b", "c"], "year": 2015, "month": 1, "obs": [0.3, 0.3, 0.3]})
    cache = RegionCache(
        code="ZZ",
        split="test",
        dates=days,
        meta=meta,
        obs=obs,
        w_mean=w,
        w_std=np.ones_like(w),
        z0=np.full_like(w, 0.05),
        shear=np.full_like(w, 0.14),
        curve_speeds=fine_curve["data$speed"].to_numpy(),
        curve_cf=fine_curve["GE.1.5sle"].to_numpy()[None, :],
        curve_names=["GE.1.5sle"],
        turbine_curve=np.zeros(3, dtype="int64"),
        era5_record={"era5_roughness": {"requested": "derived", "applied": "derived"}},
    )
    r = RegionTensors.from_cache(cache, quiet=True)
    assert list(r.ids) == ["a", "b"]
    assert r.dropped_ids == ["c"]
    assert r.dropped_capacity_share == pytest.approx(0.25)
    assert r.era5_record["era5_roughness"]["applied"] == "derived"

    kept = RegionTensors.from_cache(
        RegionCache(**{**cache.__dict__, "w_mean": np.full((31, 3), 8.0, dtype="float32")}),
        quiet=True,
    )
    assert kept.dropped_ids == [] and kept.dropped_capacity_share == 0.0


# ------------------------------------------------------- country-level tier ---
def _country_cache(fine_curve, *, years=(2015, 2016), capacity_years=None):
    """Two grid points over two Januaries, with capacity changing by year."""
    from pyvwf.pinn.cache import NATIONAL_ID, RegionCache
    from pyvwf.pinn.terrain import FEATURES

    days = pd.DatetimeIndex(
        [d for y in years for d in pd.date_range(f"{y}-01-01", periods=31, freq="D")]
    )
    meta = pd.DataFrame(
        {
            "ID": ["p1", "p2"],
            "lon": [4.0, 5.0],
            "lat": [50.0, 51.0],
            "capacity": [1000.0, 3000.0],
            "height": [100.0, 100.0],
            "type": ["onshore"] * 2,
            "model": ["GE.1.5sle"] * 2,
            **{f: [0.0, 500.0] if f == "relief_28km" else [1.0, 2.0] for f in FEATURES},
        }
    )
    w = np.empty((len(days), 2), dtype="float32")
    w[:, 0], w[:, 1] = 6.0, 10.0
    obs = pd.DataFrame(
        {"ID": NATIONAL_ID, "year": list(years), "month": 1, "obs": [0.30, 0.40][: len(years)]}
    )
    capacity_years = list(years) if capacity_years is None else capacity_years
    capacity = np.array([[1000.0, 3000.0], [3000.0, 1000.0]])[: len(capacity_years)]
    return RegionCache(
        code="ZZ",
        split="train",
        dates=days,
        meta=meta,
        obs=obs,
        w_mean=w,
        w_std=np.ones_like(w),
        z0=np.full_like(w, 0.05),
        shear=np.full_like(w, 0.14),
        curve_speeds=fine_curve["data$speed"].to_numpy(),
        curve_cf=fine_curve["GE.1.5sle"].to_numpy()[None, :],
        curve_names=["GE.1.5sle"],
        turbine_curve=np.zeros(2, dtype="int64"),
        level="country",
        capacity_years=np.array(capacity_years),
        capacity_by_year=capacity,
    )


def test_country_months_are_weighted_by_their_own_years_capacity(fine_curve):
    from pyvwf.pinn.train import RegionTensors

    r = RegionTensors.from_cache(_country_cache(fine_curve), quiet=True)
    assert r.level == "country"
    assert r.months == [(2015, 1), (2016, 1)]
    assert r.cap_weights.tolist() == [[1000.0, 3000.0], [3000.0, 1000.0]]
    assert r.obs_national.tolist() == pytest.approx([0.30, 0.40])
    assert bool(torch.isnan(r.obs).all())


def test_a_country_month_without_capacities_is_refused(fine_curve):
    from pyvwf.pinn.train import RegionTensors

    with pytest.raises(ValueError, match="no grid capacities"):
        RegionTensors.from_cache(_country_cache(fine_curve, capacity_years=[2015]), quiet=True)


def test_the_country_loss_is_the_national_series_error(fine_curve):
    """Checked by hand: aggregate each month with that year's capacities."""
    from pyvwf.pinn.physics import gauss_hermite
    from pyvwf.pinn.train import (
        N_QUAD,
        RegionTensors,
        Standardiser,
        predict_national,
        region_loss,
        simulate_monthly,
    )

    r = RegionTensors.from_cache(_country_cache(fine_curve), quiet=True)
    model = PhysicsCorrection(14, 4, init_scale=0.0)
    std = Standardiser.fit([r])
    with torch.no_grad():
        per_unit = simulate_monthly(r, model, std, slice(None), quad=gauss_hermite(N_QUAD)).numpy()
        loss = float(region_loss(r, model, std, profile="power", quad=gauss_hermite(N_QUAD)))
    w = np.array([[1000.0, 3000.0], [3000.0, 1000.0]])
    national = (per_unit * w).sum(1) / w.sum(1)
    assert loss == pytest.approx(float(np.mean((national - [0.30, 0.40]) ** 2)), rel=1e-5)

    frame = predict_national(r, model, std)
    assert frame["cf_sim"].to_numpy() == pytest.approx(national, rel=1e-5)
    assert frame["cf_obs"].to_numpy() == pytest.approx([0.30, 0.40])


def test_a_turbine_national_series_uses_only_the_units_observed(fine_curve):
    from pyvwf.pinn.cache import RegionCache
    from pyvwf.pinn.terrain import FEATURES
    from pyvwf.pinn.train import RegionTensors, _predict_matrix, predict_national

    days = pd.date_range("2015-01-01", periods=59, freq="D")  # January and February
    meta = pd.DataFrame(
        {
            "ID": ["a", "b"],
            "lon": [8.0, 9.0],
            "lat": [55.0, 56.0],
            "capacity": [1000.0, 3000.0],
            "height": [80.0, 80.0],
            "type": ["onshore"] * 2,
            "model": ["GE.1.5sle"] * 2,
            **{f: [1.0, 2.0] for f in FEATURES},
        }
    )
    w = np.full((59, 2), 8.0, dtype="float32")
    w[:, 1] = 11.0
    # Unit b is unobserved in February.
    obs = pd.DataFrame(
        {"ID": ["a", "b", "a"], "year": 2015, "month": [1, 1, 2], "obs": [0.2, 0.6, 0.3]}
    )
    cache = RegionCache(
        code="ZZ",
        split="test",
        dates=days,
        meta=meta,
        obs=obs,
        w_mean=w,
        w_std=np.ones_like(w),
        z0=np.full_like(w, 0.05),
        shear=np.full_like(w, 0.14),
        curve_speeds=fine_curve["data$speed"].to_numpy(),
        curve_cf=fine_curve["GE.1.5sle"].to_numpy()[None, :],
        curve_names=["GE.1.5sle"],
        turbine_curve=np.zeros(2, dtype="int64"),
    )
    r = RegionTensors.from_cache(cache, quiet=True)
    frame = predict_national(r, None, None)
    sim = _predict_matrix(r, None, None, profile="power", density=False, damp=None, off_curve=None)
    assert frame["cf_obs"].tolist() == pytest.approx([(0.2 * 1000 + 0.6 * 3000) / 4000, 0.3])
    assert frame["cf_sim"].tolist() == pytest.approx(
        [(sim[0, 0] * 1000 + sim[0, 1] * 3000) / 4000, sim[1, 0]], rel=1e-6
    )


def test_predict_frame_refuses_a_country_region(fine_curve):
    from pyvwf.pinn.train import RegionTensors, predict_frame

    r = RegionTensors.from_cache(_country_cache(fine_curve), quiet=True)
    with pytest.raises(ValueError, match="predict_national"):
        predict_frame(r, None, None)


def test_features_off_makes_heads_global_and_keeps_the_relief_pin(fine_curve):
    from pyvwf.pinn.train import RegionTensors, Standardiser

    r = RegionTensors.from_cache(_country_cache(fine_curve), quiet=True)
    std = Standardiser.fit([r], features_off=True)
    assert bool((std.terrain(r) == 0).all()) and bool((std.fleet(r) == 0).all())
    torch.manual_seed(0)
    model = PhysicsCorrection(14, 4)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
        gamma, delta, eta, _ = model(std.terrain(r), std.fleet(r), r.relief, r.capdens)
    # Point p1 sits on flat ground and p2 on 500 m of relief.
    assert float(gamma[0]) == 0.0 and float(gamma[1]) != 0.0
    assert float(delta[0]) == pytest.approx(float(delta[1]))
    assert float(eta[0]) == pytest.approx(float(eta[1]))


def test_fleet_columns_select_the_efficiency_heads_inputs(fine_curve):
    from pyvwf.pinn.train import FLEET_FEATURES, RegionTensors, Standardiser, fit

    r = RegionTensors.from_cache(_country_cache(fine_curve), quiet=True)
    default = Standardiser.fit([r])
    explicit = Standardiser.fit([r], fleet_columns=FLEET_FEATURES)
    assert torch.equal(default.fleet(r), explicit.fleet(r))
    three = ("log_capdens_50km", "is_offshore", "log_height")
    selected = Standardiser.fit([r], fleet_columns=three)
    assert selected.fleet(r).shape == (2, 3)
    assert torch.equal(selected.fleet(r), default.fleet(r)[:, [1, 2, 3]])
    model, std, _ = fit([r], fleet_columns=three, epochs=1, verbose=False)
    assert model.eta.net.in_features == 3
    with pytest.raises(ValueError, match="subset"):
        Standardiser.fit([r], fleet_columns=("capacity",))
    with pytest.raises(ValueError, match="wake"):
        fit([r], fleet_columns=three, wake=True, epochs=1, verbose=False)


# ------------------------------------------------------- terrain switches ---
def test_relief_off_fixes_the_speedup_at_zero(fine_curve):
    from pyvwf.pinn.train import RegionTensors, Standardiser

    r = RegionTensors.from_cache(_country_cache(fine_curve), quiet=True)
    std = Standardiser.fit([r], relief_off=True)
    assert bool((std.relief(r) == 0).all())
    torch.manual_seed(0)
    model = PhysicsCorrection(14, 4)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
        gamma, *_ = model(std.terrain(r), std.fleet(r), std.relief(r), r.capdens)
    assert torch.equal(gamma, torch.zeros_like(gamma))


def test_terrain_off_leaves_the_fleet_head_working(fine_curve):
    """The pin still sees relief; only the strength and shear go global."""
    from pyvwf.pinn.train import RegionTensors, Standardiser

    r = RegionTensors.from_cache(_country_cache(fine_curve), quiet=True)
    r.fleet_raw = torch.tensor([[0.0, 0.0, 0.0, 1.0], [2.0, 2.0, 1.0, 2.0]])
    std = Standardiser.fit([r], terrain_off=True)
    assert bool((std.terrain(r) == 0).all())
    assert not bool((std.fleet(r) == 0).all())
    torch.manual_seed(0)
    model = PhysicsCorrection(14, 4)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
        gamma, delta, eta, _ = model(std.terrain(r), std.fleet(r), std.relief(r), r.capdens)
    assert float(delta[0]) == pytest.approx(float(delta[1]))
    assert float(eta[0]) != pytest.approx(float(eta[1]))
    # Point p1 is flat and p2 has relief, so the pin still separates them.
    assert float(gamma[0]) == 0.0 and float(gamma[1]) != 0.0


def test_a_fixed_speedup_replaces_the_learned_one(fine_curve):
    from pyvwf.pinn.physics import gauss_hermite
    from pyvwf.pinn.train import (
        N_QUAD,
        RegionTensors,
        Standardiser,
        attach_fixed_speedup,
        simulate_monthly,
    )

    r = RegionTensors.from_cache(_country_cache(fine_curve), quiet=True)
    model = PhysicsCorrection(14, 4, init_scale=0.0)
    quad = gauss_hermite(N_QUAD)
    zero = Standardiser.fit([r], terrain_off=True, relief_off=True)
    fixed = Standardiser.fit([r], terrain_off=True, relief_off=True, fixed_speedup=True)
    with pytest.raises(ValueError, match="attach_fixed_speedup"):
        simulate_monthly(r, model, fixed, slice(None), quad=quad)
    with pytest.raises(ValueError, match="no fixed speed-up"):
        attach_fixed_speedup(r, pd.Series({"p1": 1.0}))
    attach_fixed_speedup(r, pd.Series({"p1": 1.0, "p2": 1.0}))
    with torch.no_grad():
        base = simulate_monthly(r, model, zero, slice(None), quad=quad)
        same = simulate_monthly(r, model, fixed, slice(None), quad=quad)
        attach_fixed_speedup(r, pd.Series({"p1": 1.3, "p2": 1.0}))
        moved = simulate_monthly(r, model, fixed, slice(None), quad=quad)
    assert torch.equal(base, same)
    assert not torch.allclose(moved[:, 0], base[:, 0])
    assert torch.equal(moved[:, 1], base[:, 1])
