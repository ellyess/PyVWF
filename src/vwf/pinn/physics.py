"""The differentiable forward operator: ERA5 wind to monthly capacity factor.

Everything the incumbent pipeline does between the reanalysis and the observed
capacity factor, written so that gradients flow back through it. That is what
lets the correction be fitted directly to observed generation instead of to
per-cluster affine factors estimated in a separate, noisy first stage.

The chain is

    hub wind      ln u = ln w + shear_eff * ln(h/100) + gamma
    sub-daily     u ~ N(u, (kappa * s)^2), integrated by Gauss-Hermite
    power         cf = eta * E[P(u)]
    aggregate     monthly mean over days

with ``gamma`` (log speed-up), the shear correction inside ``shear_eff``,
``kappa`` (how much of the within-day spread the pre-smoothed curves have not
already absorbed) and ``eta`` (conversion efficiency) supplied by the model in
:mod:`vwf.pinn.model`. With gamma = 0, no shear correction, kappa = 0 and
eta = 1 the operator reduces to the incumbent simulation, which is what
``tests/test_pinn_physics.py`` checks.
"""
from __future__ import annotations

import numpy as np
import torch

# Reference height of the ERA5 wind field the correction starts from.
REF_HEIGHT = 100.0


class PowerCurveBank:
    """Differentiable evaluation of capacity-factor curves on a uniform grid.

    The curve tables ship on a uniform speed grid (0 to 40 m/s in 0.01 m/s
    steps), so the bracketing index is arithmetic rather than a search, and the
    linear interpolation between neighbouring samples is differentiable with
    respect to wind speed -- which is what the whole model needs, since wind
    speed is where the learned correction acts.

    The curves are already Gaussian-smoothed by the VWF method
    (sigma = 0.6 + 0.2w), representing turbulence and within-farm spread at
    hourly resolution. They do NOT represent within-day spread, which is why the
    model carries a separate, learned ``kappa``.
    """

    def __init__(self, speeds: np.ndarray, curves: np.ndarray,
                 device=None, dtype=torch.float32):
        speeds = np.asarray(speeds, dtype="float64")
        step = np.diff(speeds)
        if not np.allclose(step, step[0], rtol=1e-9, atol=1e-12):
            raise ValueError("power-curve speed grid must be uniform")
        self.v0 = float(speeds[0])
        self.dv = float(step[0])
        self.n = len(speeds)
        self.curves = torch.as_tensor(np.asarray(curves), dtype=dtype, device=device)

    def __call__(self, u: torch.Tensor, curve_idx: torch.Tensor) -> torch.Tensor:
        """Capacity factor at wind speed ``u`` for each unit's own curve.

        Args:
            u: Wind speeds, any shape broadcastable against ``curve_idx``.
            curve_idx: Integer curve row per unit, broadcastable to ``u``.

        Returns:
            Capacity factors, shaped like the broadcast of the two inputs.
        """
        x = (u - self.v0) / self.dv
        # Clamp the INDEX rather than the coordinate: in float32 the largest
        # grid coordinate minus an epsilon rounds back up to the grid size, and
        # the upper neighbour i0 + 1 then walks off the end of the table.
        i0 = x.floor().clamp(0.0, float(self.n - 2))
        frac = (x - i0).clamp(0.0, 1.0)
        i0 = i0.long()
        idx = curve_idx.expand_as(i0)
        y0 = self.curves[idx, i0]
        y1 = self.curves[idx, i0 + 1]
        return y0 + (y1 - y0) * frac


def hub_wind_ratio(
    height: torch.Tensor,
    z0: torch.Tensor | None = None,
    shear: torch.Tensor | None = None,
    *,
    profile: str = "power",
) -> torch.Tensor:
    """Ratio of hub-height wind to the 100 m reanalysis wind.

    Args:
        height: Hub height per unit, metres.
        z0: Roughness length, for ``profile="log"``.
        shear: Power-law exponent between 10 m and 100 m, for ``profile="power"``.
        profile: ``"log"`` reproduces the incumbent neutral log profile
            ``ln(h/z0)/ln(100/z0)``; ``"power"`` uses ``(h/100)**shear``, whose
            exponent is measured hourly and therefore responds to atmospheric
            stability, which a static roughness cannot.

    Returns:
        Multiplicative ratio, broadcast over whatever shape the inputs carry.

    Raises:
        ValueError: If the profile name is unknown or its input is missing.
    """
    if profile == "log":
        if z0 is None:
            raise ValueError("profile='log' needs z0")
        z0 = z0.clamp(min=1e-6, max=2.0)
        denom = torch.log(torch.tensor(REF_HEIGHT, dtype=z0.dtype, device=z0.device) / z0)
        # Guard the pathological z0 -> 100 m case exactly as vwf.wind does.
        denom = torch.where(denom.abs() > 1e-12, denom, torch.full_like(denom, float("nan")))
        return torch.log(height / z0) / denom
    if profile == "power":
        if shear is None:
            raise ValueError("profile='power' needs shear")
        return (height / REF_HEIGHT) ** shear
    raise ValueError(f"unknown profile {profile!r}")


def air_density_ratio(elevation: torch.Tensor) -> torch.Tensor:
    """Air density at a site's elevation, relative to the ISO standard 1.225.

    Power curves are published at standard sea-level density. A turbine at
    1,200 m stands in air about 11% thinner and makes about 11% less power at
    the same wind speed, and nothing in the incumbent pipeline represents that:
    it is absorbed by the fitted scalar, which is one more reason the scalar
    does not mean anything transferable.

    The International Standard Atmosphere below the tropopause, with no free
    parameters to fit:

        rho/rho0 = (1 - L z / T0) ** (g / (R L) - 1)

    with L = 0.0065 K/m, T0 = 288.15 K, g = 9.80665 m/s2, R = 287.05 J/(kg K).

    Args:
        elevation: Site elevation in metres above sea level. Negative values
            (below-sea-level sites, and offshore points whose ETOPO sample is
            bathymetry) are clamped to zero.

    Returns:
        Density ratio, dimensionless and close to 1 near sea level.
    """
    z = elevation.clamp(min=0.0)
    return (1.0 - 0.0065 * z / 288.15).clamp(min=1e-3) ** 4.2559


def density_speed_factor(elevation: torch.Tensor) -> torch.Tensor:
    """IEC 61400-12 equivalent-speed factor ``(rho/rho0) ** (1/3)``.

    The standard way to score a turbine at non-standard density: instead of
    rescaling the curve, rescale the wind speed entering it, because power goes
    as density times speed cubed.
    """
    return air_density_ratio(elevation) ** (1.0 / 3.0)


def gauss_hermite(n: int, device=None, dtype=torch.float32):
    """Nodes and weights for E[f(mu + sigma Z)], Z standard normal."""
    x, w = np.polynomial.hermite.hermgauss(n)
    nodes = torch.as_tensor(x * np.sqrt(2.0), dtype=dtype, device=device)
    weights = torch.as_tensor(w / np.sqrt(np.pi), dtype=dtype, device=device)
    return nodes, weights


def expected_cf(
    u: torch.Tensor,
    sigma: torch.Tensor | None,
    curve_idx: torch.Tensor,
    bank: PowerCurveBank,
    quad: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Capacity factor averaged over the within-day wind distribution.

    Evaluating a power curve at the daily MEAN wind is not the daily mean of the
    power, because the curve is strongly non-linear across the range a day's
    wind actually spans. The difference is systematic, and in the incumbent
    pipeline it is silently absorbed by the correction factors.

    Args:
        u: Daily mean hub-height wind speed.
        sigma: Within-day standard deviation, or None to evaluate at the mean
            (which reproduces the incumbent).
        curve_idx: Curve row per unit.
        bank: The curve bank.
        quad: Pre-built Gauss-Hermite ``(nodes, weights)``.

    Returns:
        Expected capacity factor, shaped like ``u``.
    """
    if sigma is None or quad is None:
        return bank(u, curve_idx)
    nodes, weights = quad
    # (..., K): one shifted evaluation per quadrature node.
    shifted = (u.unsqueeze(-1) + sigma.unsqueeze(-1) * nodes).clamp(min=0.0)
    cf = bank(shifted, curve_idx.unsqueeze(-1))
    return (cf * weights).sum(-1)


def monthly_mean(
    daily: torch.Tensor,
    month_id: torch.Tensor,
    n_months: int,
) -> torch.Tensor:
    """Average a (days, units) field into (months, units).

    Args:
        daily: Daily values, shape ``(T, N)``.
        month_id: For each day, the index of the month it belongs to, ``(T,)``.
        n_months: Number of distinct months.

    Returns:
        Monthly means, shape ``(n_months, N)``. Months with no days are NaN.
    """
    n_units = daily.shape[1]
    total = torch.zeros(n_months, n_units, dtype=daily.dtype, device=daily.device)
    count = torch.zeros(n_months, 1, dtype=daily.dtype, device=daily.device)
    total.index_add_(0, month_id, daily)
    count.index_add_(0, month_id, torch.ones(len(month_id), 1,
                                             dtype=daily.dtype, device=daily.device))
    return total / count
