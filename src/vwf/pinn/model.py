"""The learned part: four bounded physical quantities, as functions of place.

Nothing here predicts a correction factor. Each output is a physical quantity
with the same meaning in every region, which is the whole point: a fitted
scalar of 1.37 means "Brazil" and does not transfer, whereas a log speed-up of
0.12 means "this site sits in terrain the reanalysis smooths away" and means the
same thing in Bahia and in the Tehachapi pass.

    gamma(x)   log terrain speed-up, bounded, and PINNED TO ZERO on flat ground
    delta(x)   correction to the 10-100 m shear exponent, bounded
    eta(f)     conversion efficiency: wake, availability, curtailment
    kappa      one global number: how much within-day wind spread the
               pre-smoothed power curves have not already absorbed

The pin on gamma is structural, not a penalty. Speed-up is written as an
amplitude times a saturating function of the ERA5-cell relief, so a site in
genuinely flat terrain receives no terrain correction however the amplitude
network is fitted. Most training data is flat, most of the world is flat, and
without the pin the term is free to drift there and take the extrapolation with
it.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

# Bounds. Each is a physical statement, not a tuning knob.
#   gamma  exp(-0.4)=0.67x to exp(0.9)=2.46x speed-up. The largest fitted
#          scalars (US 4.25) sit outside this on purpose: an affine scalar also
#          carries losses and curve error, which here are separate terms.
#   delta  shear exponents run about 0.05 (unstable, offshore) to 0.4 (stable
#          nocturnal over land); this is the room to move the measured value.
#   eta    wake, availability and electrical losses; 0.55 is worse than any
#          credible fleet, 1.0 is lossless.
GAMMA_BOUNDS = (-0.4, 0.9)
DELTA_BOUNDS = (-0.15, 0.25)
ETA_BOUNDS = (0.55, 1.0)
KAPPA_BOUNDS = (0.0, 1.5)


def _scaled_tanh(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Map the reals onto ``(lo, hi)``, centred so that x = 0 is the midpoint."""
    return lo + (hi - lo) * 0.5 * (torch.tanh(x) + 1.0)


def _preactivation_for(value: float, lo: float, hi: float) -> float:
    """The pre-activation that makes :func:`_scaled_tanh` return ``value``.

    Used to start every head at a stated physical value rather than at the
    midpoint of its bounds. Training that begins at the identity correction is
    both easier to optimise and honest about its prior: the model has to earn
    every departure from "the reanalysis is right".
    """
    frac = (value - lo) / (hi - lo)
    frac = min(max(frac, 1e-4), 1 - 1e-4)
    return float(np.arctanh(2.0 * frac - 1.0))


class _Head(nn.Module):
    """Linear, or a small MLP when ``hidden`` is given."""

    def __init__(self, n_in: int, hidden: int | None, bias0: float, n_out: int = 1,
                 init_scale: float = 0.0):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(
                nn.Linear(n_in, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, n_out),
            )
        else:
            self.net = nn.Linear(n_in, n_out)
        # The final layer starts at ``bias0`` for every input, so the model
        # begins at a stated physical state and moves away from it only where
        # the observations require it. With init_scale = 0 that start is exactly
        # deterministic, which makes a run reproducible but also makes seeds
        # meaningless: the gradient is full-batch, so nothing else in the fit is
        # random and every seed returns the identical model. A small weight
        # perturbation restores what seeds are for -- probing whether the
        # optimum is unique, which D6 showed is a live question.
        last = self.net[-1] if hidden else self.net
        if init_scale > 0:
            nn.init.normal_(last.weight, std=init_scale)
        else:
            nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, bias0)

    def forward(self, x):
        return self.net(x)


class PhysicsCorrection(nn.Module):
    """Bounded physical corrections as smooth functions of local physiography.

    Args:
        n_terrain: Number of standardised terrain features.
        n_fleet: Number of standardised fleet features.
        hidden: Width of the hidden layers, or None for linear heads.
        init_scale: Standard deviation of the initial head weights. Zero gives
            an exactly deterministic fit, which is reproducible but makes seeds
            degenerate; the default perturbs the start so that seeds measure
            whether the optimum is unique.
        physics: When False, every physical constraint is removed -- the relief
            pin on the speed-up disappears and the bounds are widened tenfold --
            while the parameter count and the features stay the same. This is
            the pre-specified P3 ablation: if it matches the constrained model,
            the physics was decorative and the gain came from feature scale.
    """

    def __init__(self, n_terrain: int, n_fleet: int, hidden: int | None = None,
                 physics: bool = True, init_scale: float = 0.02):
        super().__init__()
        self.physics = physics
        gb, db, eb, kb = (self._bounds(GAMMA_BOUNDS), self._bounds(DELTA_BOUNDS),
                          self._bounds(ETA_BOUNDS), self._bounds(KAPPA_BOUNDS))
        # Initial state: no speed-up, no shear correction, a 10% conversion
        # loss (a fleet always loses something), and half the within-day spread
        # not yet absorbed by the pre-smoothed curves.
        self.amp = _Head(n_terrain, hidden, _preactivation_for(0.0, *gb),
                         init_scale=init_scale)
        self.delta = _Head(n_terrain, hidden, _preactivation_for(0.0, *db),
                           init_scale=init_scale)
        self.eta = _Head(n_fleet, hidden, _preactivation_for(0.90, *eb),
                         init_scale=init_scale)
        # Relief scale at which the speed-up term saturates, in metres. Learned
        # in log space so it stays positive; 300 m is the median ERA5-cell
        # relief across the five training fleets.
        self.log_relief_scale = nn.Parameter(torch.tensor(np.log(300.0), dtype=torch.float32))
        self.raw_kappa = nn.Parameter(torch.tensor(_preactivation_for(0.5, *kb)))

    def relief_scale(self) -> torch.Tensor:
        """Relief length scale in metres, bounded away from underflow."""
        return torch.exp(self.log_relief_scale.clamp(np.log(1.0), np.log(1e4)))

    def _bounds(self, b):
        if self.physics:
            return b
        lo, hi = b
        mid = 0.5 * (lo + hi)
        return (mid + (lo - mid) * 10.0, mid + (hi - mid) * 10.0)

    def forward(self, terrain: torch.Tensor, fleet: torch.Tensor,
                relief: torch.Tensor):
        """Compute the four physical quantities for a batch of units.

        Args:
            terrain: Standardised terrain features, ``(N, n_terrain)``.
            fleet: Standardised fleet features, ``(N, n_fleet)``.
            relief: RAW ERA5-cell relief in metres, ``(N,)``. Unstandardised on
                purpose: the pin needs a quantity that is genuinely zero on flat
                ground, which a standardised feature is not.

        Returns:
            ``(gamma, delta, eta, kappa)``; the first three are ``(N,)``.
        """
        amp = _scaled_tanh(self.amp(terrain).squeeze(-1), *self._bounds(GAMMA_BOUNDS))
        if self.physics:
            # Saturating in relief and exactly zero at zero relief. The scale is
            # clamped to 1 m - 10 km, which is generous for a terrain length
            # scale and, more to the point, keeps the exponential away from the
            # underflow that would turn relief/scale into an infinity and gamma
            # into a silent NaN.
            r = relief / self.relief_scale()
            gamma = amp * (r / (1.0 + r))
        else:
            gamma = amp
        delta = _scaled_tanh(self.delta(terrain).squeeze(-1), *self._bounds(DELTA_BOUNDS))
        eta = _scaled_tanh(self.eta(fleet).squeeze(-1), *self._bounds(ETA_BOUNDS))
        kappa = _scaled_tanh(self.raw_kappa, *self._bounds(KAPPA_BOUNDS))
        return gamma, delta, eta, kappa

    @torch.no_grad()
    def report(self, terrain, fleet, relief) -> dict[str, float]:
        """Summary of the fitted physical quantities, for the run record."""
        g, d, e, k = self.forward(terrain, fleet, relief)
        k = k.detach()
        return {
            "gamma_mean": float(g.mean()), "gamma_std": float(g.std()),
            "gamma_max": float(g.max()), "gamma_min": float(g.min()),
            "speedup_mean": float(torch.exp(g).mean()),
            "speedup_max": float(torch.exp(g).max()),
            "delta_mean": float(d.mean()), "delta_std": float(d.std()),
            "eta_mean": float(e.mean()), "eta_std": float(e.std()),
            "eta_min": float(e.min()), "eta_max": float(e.max()),
            "kappa": float(k),
            "relief_scale_m": float(self.relief_scale()),
        }
