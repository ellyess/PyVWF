"""Fit the physics-informed correction directly to observed generation.

There is no intermediate target. The model proposes four physical quantities
per site, the forward operator turns them into a monthly capacity factor, and
the loss compares that to what the fleet actually generated. The per-cluster
affine factors -- and the estimation noise, partition dependence and rank-1
degeneracy the diagnostics found in them -- never enter.

Regions are weighted equally in the loss regardless of fleet size. Denmark
brings 3,707 units and Brazil 125; pooling rows would make the fitted physics
mostly Danish, which is the opposite of what a transferable model needs. Within
a region, rows are capacity-weighted, matching how the harness scores skill.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import BallTree

from vwf.pinn.cache import RegionCache, load_cache
from vwf.pinn.model import PhysicsCorrection
from vwf.pinn.physics import (
    PowerCurveBank, density_speed_factor, expected_cf, gauss_hermite,
    hub_wind_ratio, monthly_mean,
)
from vwf.pinn.terrain import FEATURES as TERRAIN_FEATURES

FLEET_FEATURES = ("log_capdens_10km", "log_capdens_50km", "is_offshore", "log_height")
N_QUAD = 5          # Gauss-Hermite nodes; the curves are already smooth
UNIT_BATCH = 384    # units per step, chosen to bound peak memory
EARTH_R_KM = 6371.0


def _capacity_density(meta: pd.DataFrame, radius_km: float) -> np.ndarray:
    """Installed capacity per unit area within ``radius_km``, in MW/km2.

    The conversion-efficiency head needs a fleet descriptor that means the same
    thing in every region, and raw capacity does not: a row is one turbine in
    Denmark and Germany but a whole plant in the United States and a whole
    complex in Brazil, so median capacity runs 750 kW against 136,400 kW and
    separates the continents perfectly on data convention rather than physics.
    A model given that feature can label the region and call it a fleet
    property.

    Capacity density is invariant to how rows are aggregated -- the megawatts
    inside a 10 km circle are the same number whether they arrive as one plant
    row or forty turbine rows -- and it is also the quantity wake losses
    actually depend on. It is computed from the observed fleet only, so it
    understates the true density wherever coverage is partial; that
    understatement is a property of the observations, not of the region's
    convention.
    """
    lon = np.radians(meta["lon"].to_numpy(dtype=float))
    lat = np.radians(meta["lat"].to_numpy(dtype=float))
    cap_mw = meta["capacity"].to_numpy(dtype=float) / 1000.0
    pts = np.column_stack([lat, lon])
    tree = BallTree(pts, metric="haversine")
    neighbours = tree.query_radius(pts, r=radius_km / EARTH_R_KM)
    total = np.array([cap_mw[ix].sum() for ix in neighbours])
    return total / (np.pi * radius_km ** 2)


def _fleet_frame(meta: pd.DataFrame) -> np.ndarray:
    """Fleet descriptors chosen to be comparable across observation units."""
    offshore = (meta.get("type", pd.Series(["onshore"] * len(meta)))
                .astype(str).str.lower().eq("offshore").to_numpy(dtype=float))
    height = meta["height"].to_numpy(dtype=float)
    return np.column_stack([
        np.log10(np.clip(_capacity_density(meta, 10.0), 1e-4, None)),
        np.log10(np.clip(_capacity_density(meta, 50.0), 1e-4, None)),
        offshore,
        np.log10(np.clip(height, 5.0, None)),
    ])


def _drop_unsimulable(cache: RegionCache, *, quiet: bool = False):
    """Remove units the reanalysis cannot cover, and repair isolated gaps.

    Two distinct problems, handled differently because they mean different
    things. A unit whose whole wind series is missing sits OUTSIDE the region's
    configured ERA5 bounding box -- 47 Danish turbines on Bornholm, at 15 deg E
    against a box that stops at 13.5 -- and cannot be simulated at all. The
    incumbent pipeline drops these silently at metric time; they are dropped
    here too, but counted and reported, because a silently shrinking fleet is
    exactly the kind of thing that should never be quiet.

    Isolated missing cells are different: they come from the roughness
    inversion failing on a near-zero-shear hour and are a few dozen cells in
    the American fleets. Those are filled with the unit's own median, which for
    a slowly varying quantity like roughness changes nothing material.
    """
    names = ("w_mean", "w_std", "z0", "shear")
    arrs = {n: np.array(getattr(cache, n), dtype="float32", copy=True)
            for n in names}
    n_days = arrs["w_mean"].shape[0]
    all_bad = np.zeros(arrs["w_mean"].shape[1], dtype=bool)
    for a in arrs.values():
        all_bad |= (~np.isfinite(a)).sum(0) == n_days
    keep = ~all_bad
    if all_bad.any() and not quiet:
        print(f"  [{cache.code}/{cache.split}] dropping {int(all_bad.sum())} unit(s) "
              f"with no ERA5 coverage (outside the configured bbox)")

    filled = 0
    for n, a in arrs.items():
        a = a[:, keep]
        bad = ~np.isfinite(a)
        if bad.any():
            filled += int(bad.sum())
            med = np.nanmedian(np.where(bad, np.nan, a), axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            a = np.where(bad, np.broadcast_to(med, a.shape), a)
        arrs[n] = a
    if filled and not quiet:
        print(f"  [{cache.code}/{cache.split}] filled {filled} isolated missing "
              f"cell(s) with the unit's median")

    arrs["keep"] = keep
    return cache.meta.reset_index(drop=True).loc[keep].reset_index(drop=True), arrs


@dataclass
class RegionTensors:
    """One region's cache as aligned tensors on a common (day, unit) grid."""

    code: str
    split: str
    ids: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    w: torch.Tensor            # (T, N) daily mean 100 m wind
    s: torch.Tensor            # (T, N) within-day wind spread
    z0: torch.Tensor           # (T, N) roughness, incumbent definition
    shear: torch.Tensor        # (T, N) measured 10-100 m exponent
    height: torch.Tensor       # (N,)
    capacity: torch.Tensor     # (N,)
    curve_idx: torch.Tensor    # (N,)
    terrain_raw: torch.Tensor  # (N, F)
    fleet_raw: torch.Tensor    # (N, G)
    relief: torch.Tensor       # (N,) raw metres, for the speed-up pin
    elevation: torch.Tensor    # (N,) raw metres, for the air-density factor
    month_id: torch.Tensor     # (T,)
    obs: torch.Tensor          # (M, N), NaN where unobserved
    months: list[tuple[int, int]]
    bank: PowerCurveBank

    @property
    def n_units(self) -> int:
        return len(self.ids)

    @classmethod
    def from_cache(cls, cache: RegionCache, *, quiet: bool = False) -> "RegionTensors":
        meta, fields = _drop_unsimulable(cache, quiet=quiet)
        ids = meta["ID"].astype(str).to_numpy()
        id_pos = {i: k for k, i in enumerate(ids)}

        ym = pd.MultiIndex.from_arrays(
            [cache.dates.year, cache.dates.month]).unique().sort_values()
        months = [(int(y), int(m)) for y, m in ym]
        month_pos = {k: i for i, k in enumerate(months)}
        month_id = np.array(
            [month_pos[(int(d.year), int(d.month))] for d in cache.dates], dtype="int64")

        obs = np.full((len(months), len(ids)), np.nan, dtype="float32")
        o = cache.obs.dropna(subset=["obs"])
        rows = [month_pos.get((int(y), int(m)), -1) for y, m in zip(o.year, o.month)]
        cols = [id_pos.get(str(i), -1) for i in o.ID]
        rows, cols = np.array(rows), np.array(cols)
        ok = (rows >= 0) & (cols >= 0)
        obs[rows[ok], cols[ok]] = o["obs"].to_numpy(dtype="float32")[ok]

        t = lambda a, d=torch.float32: torch.as_tensor(np.asarray(a), dtype=d)  # noqa: E731
        return cls(
            code=cache.code, split=cache.split, ids=ids,
            lon=meta["lon"].to_numpy(dtype=float),
            lat=meta["lat"].to_numpy(dtype=float),
            w=t(fields["w_mean"]), s=t(fields["w_std"]),
            z0=t(fields["z0"]), shear=t(fields["shear"]),
            height=t(meta["height"].to_numpy(dtype=float)),
            capacity=t(meta["capacity"].to_numpy(dtype=float)),
            curve_idx=t(cache.turbine_curve[fields["keep"]], torch.long),
            terrain_raw=t(meta[list(TERRAIN_FEATURES)].to_numpy(dtype=float)),
            fleet_raw=t(_fleet_frame(meta)),
            relief=t(meta["relief_28km"].to_numpy(dtype=float)),
            elevation=t(meta["z_site"].to_numpy(dtype=float)),
            month_id=t(month_id, torch.long),
            obs=t(obs), months=months,
            bank=PowerCurveBank(cache.curve_speeds, cache.curve_cf),
        )


@dataclass
class Standardiser:
    """Feature centring and scaling, fitted on training regions only."""

    t_mean: torch.Tensor
    t_std: torch.Tensor
    f_mean: torch.Tensor
    f_std: torch.Tensor

    @classmethod
    def fit(cls, regions: list[RegionTensors]) -> "Standardiser":
        T = torch.cat([r.terrain_raw for r in regions])
        F = torch.cat([r.fleet_raw for r in regions])
        return cls(T.mean(0), T.std(0).clamp(min=1e-6),
                   F.mean(0), F.std(0).clamp(min=1e-6))

    def terrain(self, r: RegionTensors, sl=slice(None)) -> torch.Tensor:
        return (r.terrain_raw[sl] - self.t_mean) / self.t_std

    def fleet(self, r: RegionTensors, sl=slice(None)) -> torch.Tensor:
        return (r.fleet_raw[sl] - self.f_mean) / self.f_std


def _nn_distance(query: torch.Tensor, reference: torch.Tensor,
                  exclude_self: bool = False, chunk: int = 256) -> torch.Tensor:
    """Distance from each query row to its nearest reference row."""
    out = torch.empty(len(query))
    for i in range(0, len(query), chunk):
        d = torch.cdist(query[i:i + chunk], reference)
        if exclude_self:
            n = d.shape[0]
            idx = torch.arange(i, i + n)
            d[torch.arange(n), idx] = float("inf")
        out[i:i + chunk] = d.min(dim=1).values
    return out


def coverage_weight(
    test: "RegionTensors",
    train_regions: list["RegionTensors"],
    std: "Standardiser",
) -> torch.Tensor:
    """How far inside the training physiography each test unit sits, in [0, 1].

    D5 measured that half the British and American test units lie outside the
    training regions' terrain envelope, and that the American fleet reaches 22
    standard deviations from its nearest training analogue. A correction fitted
    on one envelope and evaluated far outside it is extrapolation whatever its
    functional form, so the terrain terms are damped with distance:

        w = exp(-max(0, d - d0)^2 / d0^2)

    where ``d`` is the distance to the nearest training unit in standardised
    terrain-feature space and ``d0`` is the 95th percentile of the training
    fleet's own within-training nearest-neighbour distance. The threshold is
    therefore calibrated by the training set against itself, with nothing tuned
    on the region being predicted: ``w`` is 1 anywhere the training data is as
    dense as it is internally, and decays beyond that.

    Returns:
        Weight per test unit, shape ``(N,)``.
    """
    # Deduplicated first, and this matters. German rows are postcode centroids
    # and British rows are farm generation split across turbines, so thousands
    # of rows describe a few hundred distinct sites: 52% of the raw
    # within-training nearest-neighbour distances are exactly zero, which drags
    # the 95th percentile from 2.38 down to 0.97 and damps the correction more
    # than twice as hard as the training data's real spacing warrants. The
    # threshold has to be the spacing between distinct SITES.
    ref = torch.unique(torch.cat([std.terrain(r) for r in train_regions]), dim=0)
    d0 = torch.quantile(_nn_distance(ref, ref, exclude_self=True), 0.95)
    d = _nn_distance(std.terrain(test), ref)
    excess = (d - d0).clamp(min=0.0)
    return torch.exp(-(excess ** 2) / (d0 ** 2).clamp(min=1e-6))


def simulate_monthly(
    r: RegionTensors,
    model: PhysicsCorrection | None,
    std: Standardiser | None,
    sl: slice,
    *,
    profile: str = "power",
    density: bool = False,
    damp: torch.Tensor | None = None,
    quad=None,
) -> torch.Tensor:
    """Monthly capacity factor for a slice of units.

    With ``model=None`` this is the incumbent simulation: neutral log profile on
    the pipeline roughness, power curve at the daily mean wind, no losses.
    """
    if model is None:
        ratio = hub_wind_ratio(r.height[sl], z0=r.z0[:, sl], profile="log")
        cf = expected_cf(r.w[:, sl] * ratio, None, r.curve_idx[sl], r.bank, None)
        return monthly_mean(cf, r.month_id, len(r.months))

    gamma, delta, eta, kappa = model(
        std.terrain(r, sl), std.fleet(r, sl), r.relief[sl]
    )
    if damp is not None:
        # Only the TERRAIN terms are damped outside the training envelope.
        # Conversion losses and thin air do not stop existing because the
        # terrain is unfamiliar, and the fleet features are covered everywhere.
        w = damp[sl]
        gamma = gamma * w
        delta = delta * w
    if profile == "power":
        ratio = hub_wind_ratio(r.height[sl], shear=r.shear[:, sl] + delta,
                               profile="power")
    else:
        ratio = hub_wind_ratio(r.height[sl], z0=r.z0[:, sl], profile="log")
    scale = torch.exp(gamma) * ratio
    if density:
        # Applied to the speed entering the curve, not to the wind itself: the
        # air is thinner, the wind is not slower.
        scale = scale * density_speed_factor(r.elevation[sl])
    u = r.w[:, sl] * scale
    sigma = (kappa * r.s[:, sl] * scale).clamp(min=1e-3)
    cf = expected_cf(u, sigma, r.curve_idx[sl], r.bank, quad)
    return monthly_mean(cf * eta, r.month_id, len(r.months))


def region_loss(r, model, std, *, profile, quad, density=False, generator=None):
    """Capacity-weighted mean squared monthly CF error, over unit minibatches."""
    n = r.n_units
    order = (torch.randperm(n, generator=generator) if generator is not None
             else torch.arange(n))
    total = torch.zeros((), dtype=torch.float32)
    wsum = 0.0
    for start in range(0, n, UNIT_BATCH):
        idx = order[start:start + UNIT_BATCH]
        sl = idx
        pred = simulate_monthly(r, model, std, sl, profile=profile,
                                density=density, quad=quad)
        obs = r.obs[:, sl]
        mask = torch.isfinite(obs)
        if not mask.any():
            continue
        wcap = r.capacity[sl].unsqueeze(0).expand_as(obs)[mask]
        err = (pred[mask] - obs[mask]) ** 2
        total = total + (err * wcap).sum()
        wsum += float(wcap.sum())
    return total / max(wsum, 1e-9)


def fit(
    regions: list[RegionTensors],
    *,
    hidden: int | None = None,
    physics: bool = True,
    profile: str = "power",
    density: bool = False,
    epochs: int = 120,
    lr: float = 0.05,
    weight_decay: float = 1e-3,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[PhysicsCorrection, Standardiser, list[float]]:
    """Fit one model on a list of training regions, weighting regions equally."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    std = Standardiser.fit(regions)
    model = PhysicsCorrection(len(TERRAIN_FEATURES), len(FLEET_FEATURES),
                              hidden=hidden, physics=physics)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    quad = gauss_hermite(N_QUAD)

    history = []
    for ep in range(epochs):
        opt.zero_grad()
        # Equal weight per region: the mean of per-region mean errors, not the
        # mean over pooled rows.
        loss = torch.stack([
            region_loss(r, model, std, profile=profile, quad=quad,
                        density=density, generator=gen)
            for r in regions
        ]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        history.append(float(loss.detach()))
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"    epoch {ep:3d}  loss {float(loss):.6f}  "
                  f"rmse {np.sqrt(float(loss)):.4f}")
    return model, std, history


@torch.no_grad()
def predict_frame(
    r: RegionTensors,
    model: PhysicsCorrection | None,
    std: Standardiser | None,
    *,
    profile: str = "power",
    density: bool = False,
    damp: torch.Tensor | None = None,
) -> pd.DataFrame:
    """Tidy (ID, year, month, cf_sim, cf_obs, capacity) frame for the harness."""
    quad = gauss_hermite(N_QUAD)
    preds = []
    for start in range(0, r.n_units, UNIT_BATCH):
        sl = torch.arange(start, min(start + UNIT_BATCH, r.n_units))
        preds.append(simulate_monthly(r, model, std, sl, profile=profile,
                                      density=density, damp=damp, quad=quad))
    pred = torch.cat(preds, dim=1).numpy()

    years = np.array([y for y, _ in r.months])
    months = np.array([m for _, m in r.months])
    M, N = pred.shape
    frame = pd.DataFrame({
        "ID": np.repeat(r.ids[None, :], M, axis=0).ravel(),
        "year": np.repeat(years[:, None], N, axis=1).ravel(),
        "month": np.repeat(months[:, None], N, axis=1).ravel(),
        "cf_sim": pred.ravel(),
        "cf_obs": r.obs.numpy().ravel(),
        "capacity": np.repeat(r.capacity.numpy()[None, :], M, axis=0).ravel(),
    })
    return frame.dropna(subset=["cf_obs"]).reset_index(drop=True)


def load_regions(codes, split, root: str | Path, *, quiet: bool = False) -> list[RegionTensors]:
    """Load and convert several region caches."""
    return [RegionTensors.from_cache(load_cache(c, split, root), quiet=quiet)
            for c in codes]
