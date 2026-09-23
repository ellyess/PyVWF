"""Stage 0 of the per-unit wake coefficient test: in-region density slopes.

Registered in ``docs/findings/method-wake-unit-prereg.md`` (gate K0, with the
deviation of 2026-09-23), committed before this driver. For each of DK, DE,
UK, US and BR it fits the physics-informed correction on the region's own
training years with the "clean control" settings (wake off, the 10 km
capacity density withheld from the efficiency head) and asks whether the test
year's residuals rise with 10 km capacity density, and whether that slope
differs by observation unit.

Everything the registration fixes is fixed here, not on the command line:

- The seeds, epochs, heads and bootstrap draws are module constants.
  ``--seeds``, ``--epochs``, ``--hidden`` and ``--draws`` exist only so the
  registered command line parses, and the run refuses any value other than the
  registered one.
- The residual is ``cf_sim - cf_obs`` per test-year unit-month, with
  ``cf_sim`` the mean of the five seeds' predictions. The slope is the
  capacity-weighted least-squares slope on D, the 10 km capacity density in
  MW/km2 the model's tensors carry (``RegionTensors.capdens``, computed on the
  units that have winds). Rows are not collapsed.
- The interval is a pigeonhole bootstrap: blocks of units and months are
  resampled independently with replacement, and a row's weight in a draw is
  its capacity times its block's count times its month's count. One generator,
  seeded 0, draws the counts for each region in the fixed order DK, DE, UK, US,
  BR, so draw ``d`` of every region is combined into draw ``d`` of K0b. K0c
  and the DK-restricted contrast reuse DK's counts.
- A block is a connected component of units linked by an exactly equal,
  non-missing, non-zero observed CF in some test-year month, formed over the
  units observed in the test year. A shared-target
  unit is one whose observed CF exactly equals another unit's in more than
  half of its observed test-year months.
- K0c's interaction coefficient is computed as the difference of the two
  groups' slopes. With a separate intercept and slope per group the
  interaction model is saturated, so the two are the same number.
- Gate V's planted half is exact by the linearity of least squares, so it
  checks the weights and the bookkeeping, not the estimator's power. Its
  permuted half assigns each block the capacity-weighted mean D of another
  block, drawn by a permutation seeded 1.

``--sensitivity CODE=PATH`` fits a region a second time from another cache
directory, writes under ``sensitivity/<CODE>/``, and reuses that region's
bootstrap counts by giving each unit its block in the gated cache. A unit the
other cache cannot simulate is recorded with its capacity share; a unit or a
month only the other cache has is refused. The sensitivity changes no gate.

Read-only with respect to the tree. Writes a manifest into ``--out`` before
any fit, and refuses a dirty tree unless ``--allow-dirty`` is given.

Usage, from the repository root:

    python scripts/dev/run_locked.py -- env PYVWF_INPUT=input/combined PYTHONPATH=src \\
      /opt/anaconda3/bin/python -u scripts/studies/method-wake-unit/k0_density_slopes.py \\
      --cache output/pinn_rerun_2026-09-16/cache \\
      --config DK=configs/regions/scorecard/dk_k100.toml \\
      --config DE=configs/regions/scorecard/de_k100.toml \\
      --config UK=configs/regions/scorecard/uk_k50.toml \\
      --config US=configs/regions/scorecard/us_k250.toml \\
      --config BR=configs/regions/scorecard/br_k60.toml \\
      --sensitivity DE=output/pinn_turbine_2026-09-18/cache_mastr \\
      --seeds 0 1 2 3 42 --epochs 60 --hidden 0 --draws 1000 \\
      --out output/method_wake_unit_<run date>/k0 \\
      --registration docs/findings/method-wake-unit-prereg.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from vwf.cli.common import make_parser

CACHE = Path("output/pinn_rerun_2026-09-16/cache")
E1_RAW = Path("output/pinn_rerun_2026-09-16/e1/e1_primary_raw.csv")
REGISTRATION = "docs/findings/method-wake-unit-prereg.md"

#: Registered in the pre-registration; not flags.
REGIONS = ("DK", "DE", "UK", "US", "BR")
TURBINE_UNITS = ("turbine",)
AGGREGATE_UNITS = ("farm", "plant", "complex")
SEEDS = (0, 1, 2, 3, 42)
EPOCHS = 60
HIDDEN = 0
PROFILE = "power"
DRAWS = 1000
BOOTSTRAP_SEED = 0
PERMUTATION_SEED = 1
PLANTED_SLOPE = 0.05
#: The clean control: the efficiency head without ``log_capdens_10km``.
CLEAN_FLEET = ("log_capdens_50km", "is_offshore", "log_height")
#: E9's six density bins, MW/km2, for the descriptive table only.
E9_BINS = (0.0, 0.045, 0.080, 0.138, 0.244, 0.747, np.inf)
#: Draws per chunk when computing bootstrap slopes, to bound memory.
CHUNK = 100


# ------------------------------------------------------------- statistics ---


def target_blocks(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resampling block and shared-target flag per unit, from observed CFs.

    Args:
        obs: (months, units) observed capacity factors, NaN where unobserved.

    Returns:
        ``(block, shared)``. ``block`` labels each unit's connected component
        of units linked by an exactly equal, non-missing, non-zero CF in some
        month, numbered in order of each component's first unit. ``shared``
        is True where a unit's CF equals another unit's in more than half of
        its observed months.
    """
    n_months, n_units = obs.shape
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    shared_months = np.zeros(n_units, dtype=int)
    for m in range(n_months):
        v = obs[m]
        seen = np.flatnonzero(np.isfinite(v))
        if seen.size < 2:
            continue
        _, inverse, counts = np.unique(v[seen], return_inverse=True, return_counts=True)
        repeated = counts[inverse] > 1
        shared_months[seen[repeated]] += 1
        linkable = repeated & (v[seen] != 0)
        members = seen[linkable]
        groups = inverse[linkable]
        order = np.argsort(groups, kind="stable")
        members, groups = members[order], groups[order]
        same = groups[1:] == groups[:-1]
        rows.append(members[:-1][same])
        cols.append(members[1:][same])
    r = np.concatenate(rows) if rows else np.array([], dtype=int)
    c = np.concatenate(cols) if cols else np.array([], dtype=int)
    graph = coo_matrix((np.ones(r.size), (r, c)), shape=(n_units, n_units))
    _, labels = connected_components(graph, directed=False)
    _, first = np.unique(labels, return_index=True)
    # Number components by their first unit, so the labels do not depend on
    # how the graph library orders them.
    renumber = np.argsort(np.argsort(first))
    observed = np.isfinite(obs).sum(axis=0)
    shared = shared_months > 0.5 * observed
    return renumber[labels], shared


def weighted_slope(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted least-squares slope of y on x with an intercept.

    ``w`` may be one weight vector or a (draws, rows) array, one slope per row.
    """
    w = np.atleast_2d(w)
    sw = w.sum(axis=1)
    sx = w @ x
    sy = w @ y
    sxx = w @ (x * x)
    sxy = w @ (x * y)
    return (sw * sxy - sx * sy) / (sw * sxx - sx * sx)


def draw_counts(
    rng: np.random.Generator, n_blocks: int, n_months: int, draws: int
) -> tuple[np.ndarray, np.ndarray]:
    """Resampling counts for one region: (draws, blocks) and (draws, months)."""
    cb = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=draws)
    cm = rng.multinomial(n_months, np.full(n_months, 1.0 / n_months), size=draws)
    return cb, cm


def bootstrap_slopes(
    x: np.ndarray,
    y: np.ndarray,
    capacity: np.ndarray,
    block: np.ndarray,
    month: np.ndarray,
    cb: np.ndarray,
    cm: np.ndarray,
) -> np.ndarray:
    """One slope per draw under the pigeonhole weights."""
    out = np.empty(cb.shape[0])
    for start in range(0, cb.shape[0], CHUNK):
        sl = slice(start, start + CHUNK)
        w = capacity[None, :] * cb[sl][:, block] * cm[sl][:, month]
        out[sl] = weighted_slope(x, y, w)
    return out


def interval(values: np.ndarray) -> tuple[float, float]:
    """The registered 95% percentile interval."""
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def permuted_density(
    d_unit: np.ndarray, capacity_unit: np.ndarray, block_unit: np.ndarray, seed: int
) -> np.ndarray:
    """Each unit gets the capacity-weighted mean D of another block."""
    # Permute among the blocks present, so no unit can draw an empty block.
    _, compact = np.unique(block_unit, return_inverse=True)
    n_blocks = int(compact.max()) + 1
    num = np.bincount(compact, weights=d_unit * capacity_unit, minlength=n_blocks)
    den = np.bincount(compact, weights=capacity_unit, minlength=n_blocks)
    block_mean = num / den
    perm = np.random.default_rng(seed).permutation(n_blocks)
    return block_mean[perm][compact]


# ---------------------------------------------------------------- the run ---


class RegionResult(NamedTuple):
    """One region's residual frame and what the gates need from it."""

    code: str
    rows: pd.DataFrame
    block_ids: list[str]
    months: list[tuple[int, int]]
    record: dict


def _rows_for_region(r_train, r_test, spec, e1_rmse: dict[int, float]) -> RegionResult:
    """Fit the clean control per seed and build the residual frame."""
    from vwf.pinn.runs import score_on_common_rows
    from vwf.pinn.train import FLEET_FEATURES, fit, predict_frame

    keys = ["ID", "year", "month"]
    frame: pd.DataFrame | None = None
    seed_rmse: dict[int, float] = {}
    head_inputs: set[tuple[int, ...]] = set()
    for seed in SEEDS:
        model, std, _ = fit(
            [r_train],
            hidden=HIDDEN or None,
            physics=True,
            profile=PROFILE,
            density=False,
            wake=False,
            epochs=EPOCHS,
            seed=seed,
            verbose=False,
            fleet_columns=CLEAN_FLEET,
        )
        head_inputs.add(tuple(FLEET_FEATURES[i] for i in std.fleet_idx))
        f = predict_frame(r_test, model, std, profile=PROFILE)
        metrics, _, _ = score_on_common_rows({"clean": ("clean", seed, f)}, spec)
        seed_rmse[seed] = float(metrics["clean"]["rmse"])
        f = f.rename(columns={"cf_sim": f"cf_sim_seed{seed}"})
        frame = f if frame is None else frame.merge(f[[*keys, f"cf_sim_seed{seed}"]], on=keys)
        print(f"  {r_test.code} seed {seed}: RMSE {seed_rmse[seed]:.5f}", flush=True)
    assert frame is not None

    sims = [f"cf_sim_seed{s}" for s in SEEDS]
    frame["cf_sim"] = frame[sims].mean(axis=1)
    frame["residual"] = frame["cf_sim"] - frame["cf_obs"]

    # Blocks are formed over the units observed in the test year: a unit with
    # no observation has no residual row, and a block of its own would only
    # dilute the resampling.
    obs_all = r_test.obs.numpy().astype(float)
    observed = np.isfinite(obs_all).any(axis=0)
    block, shared = target_blocks(obs_all[:, observed])
    capacity = r_test.capacity.numpy().astype(float)[observed]
    unit = pd.DataFrame(
        {
            "ID": np.asarray(r_test.ids).astype(str)[observed],
            "D": r_test.capdens.numpy().astype(float)[observed],
            "offshore": r_test.fleet_raw.numpy()[observed, 2] > 0.5,
            "block": block,
            "shared": shared,
        }
    )
    frame["ID"] = frame["ID"].astype(str)
    frame = frame.merge(unit, on="ID", how="left", validate="many_to_one")
    month_pos = {ym: i for i, ym in enumerate(r_test.months)}
    frame["month_idx"] = [month_pos[(int(y), int(m))] for y, m in zip(frame.year, frame.month)]

    # A block is named by its lowest unit ID, so a second cache of the same
    # units can be checked for the same partition.
    names = unit.groupby("block")["ID"].min().sort_index().tolist()
    train_years = sorted({y for y, _ in r_train.months})
    test_years = sorted({y for y, _ in r_test.months})
    record = {
        "obs_unit": spec.obs_unit,
        "training_years": train_years,
        "test_years": test_years,
        "rows": int(len(frame)),
        "units_observed": int(unit.shape[0]),
        "units_unobserved": int((~observed).sum()),
        "blocks": len(names),
        "multi_unit_blocks": int((np.bincount(block) > 1).sum()),
        "months": len(r_test.months),
        "capacity_mw": float(capacity.sum()) / 1000.0,
        "shared_capacity_share": float(capacity[shared].sum() / capacity.sum()),
        # The check the registration makes before any slope is read: every
        # seed's efficiency head saw these inputs, and not the 10 km density.
        "efficiency_head_inputs": "|".join(sorted({c for h in head_inputs for c in h})),
        "efficiency_head_has_10km_density": any("log_capdens_10km" in h for h in head_inputs),
        "rmse_by_seed": seed_rmse,
        "e1_in_region_rmse_by_seed": e1_rmse,
    }
    return RegionResult(r_test.code, frame, names, list(r_test.months), record)


def _slope_summary(rows: pd.DataFrame, cb: np.ndarray, cm: np.ndarray) -> dict:
    """Point slope, its draws and interval, and each seed's slope."""
    x = rows["D"].to_numpy(float)
    y = rows["residual"].to_numpy(float)
    cap = rows["capacity"].to_numpy(float)
    b = float(weighted_slope(x, y, cap)[0])
    draws = bootstrap_slopes(
        x, y, cap, rows["block"].to_numpy(), rows["month_idx"].to_numpy(), cb, cm
    )
    lo, hi = interval(draws)
    per_seed = {
        s: float(
            weighted_slope(x, rows[f"cf_sim_seed{s}"].to_numpy(float) - rows["cf_obs"], cap)[0]
        )
        for s in SEEDS
    }
    return {"b": b, "lo": lo, "hi": hi, "draws": draws, "per_seed": per_seed}


def _validation(rows: pd.DataFrame, cb: np.ndarray, cm: np.ndarray) -> dict:
    """Gate V: a planted slope is recovered and a permuted D gives none."""
    x = rows["D"].to_numpy(float)
    y = rows["residual"].to_numpy(float)
    cap = rows["capacity"].to_numpy(float)
    block = rows["block"].to_numpy()
    month = rows["month_idx"].to_numpy()
    base = bootstrap_slopes(x, y, cap, block, month, cb, cm)
    planted = bootstrap_slopes(x, y + PLANTED_SLOPE * (x - x.mean()), cap, block, month, cb, cm)
    plo, phi = interval(planted - base)

    units = rows.drop_duplicates("ID")
    d_perm = permuted_density(
        units["D"].to_numpy(float),
        units["capacity"].to_numpy(float),
        units["block"].to_numpy(),
        PERMUTATION_SEED,
    )
    x_perm = rows["ID"].map(dict(zip(units["ID"], d_perm))).to_numpy(float)
    q = bootstrap_slopes(x_perm, y, cap, block, month, cb, cm)
    qlo, qhi = interval(q)
    return {
        "planted_diff_lo": plo,
        "planted_diff_hi": phi,
        "planted_pass": bool(plo <= PLANTED_SLOPE <= phi),
        "permuted_b": float(weighted_slope(x_perm, y, cap)[0]),
        "permuted_lo": qlo,
        "permuted_hi": qhi,
        "permuted_pass": bool(qlo <= 0.0 <= qhi),
    }


def _density_bins(rows: pd.DataFrame, code: str) -> pd.DataFrame:
    """Capacity-weighted mean residual in E9's six bins, descriptive only."""
    cut = pd.cut(rows["D"], list(E9_BINS), right=False)
    g = rows.assign(bin=cut, wr=rows["residual"] * rows["capacity"]).groupby("bin", observed=False)
    out = pd.DataFrame(
        {"capacity": g["capacity"].sum(), "weighted": g["wr"].sum(), "rows": g.size()}
    )
    out["mean_residual"] = out["weighted"] / out["capacity"]
    return out.drop(columns="weighted").reset_index().assign(region=code)


def main(
    *,
    cache: Path,
    configs: list[str],
    sensitivity: list[str],
    out: Path,
    registration: str,
    e1_raw: Path,
    allow_dirty: bool,
) -> None:
    """Fit the clean control per region, and write residuals, slopes and gates."""
    import torch

    from vwf.harness.regions import load_region
    from vwf.pinn.runs import config_record, region_record, resolve_configs
    from vwf.pinn.train import load_regions
    from vwf.provenance import build_manifest, write_manifest

    launch = build_manifest()
    if launch["git_dirty"] and not allow_dirty:
        raise SystemExit(
            "refusing to run on a dirty tree: every result would be "
            "unattributable. Commit first, or pass --allow-dirty."
        )

    config_paths = resolve_configs(list(REGIONS), configs)
    specs = {c: load_region(config_paths[c]) for c in REGIONS}
    for c, spec in specs.items():
        if spec.code != c:
            raise SystemExit(f"{config_paths[c]} is region {spec.code}, not {c}")
    units = {c: specs[c].obs_unit for c in REGIONS}
    turbine = [c for c in REGIONS if units[c] in TURBINE_UNITS]
    aggregate = [c for c in REGIONS if units[c] in AGGREGATE_UNITS]
    if turbine != ["DK", "DE"] or aggregate != ["UK", "US", "BR"]:
        raise SystemExit(f"obs_unit grouping differs from the registration: {units}")

    extra_caches: dict[str, Path] = {}
    for item in sensitivity:
        code, sep, path = item.partition("=")
        if not sep or not path or code not in REGIONS:
            raise SystemExit(
                f"--sensitivity expects CODE=PATH with CODE in {REGIONS}, got {item!r}"
            )
        extra_caches[code] = Path(path)
    sensitivity_commits = {}
    for code, path in extra_caches.items():
        manifest_path = path / "run_manifest.json"
        m = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        sensitivity_commits[code] = {
            "path": str(path),
            "git_commit": m.get("git_commit"),
            "git_dirty": m.get("git_dirty"),
            "curve_library": m.get("curve_library"),
        }

    e1 = pd.read_csv(e1_raw)
    e1 = e1[e1["arm"] == "pinn-in-region"]
    e1_rmse = {
        c: {int(s): float(v) for s, v in zip(e1[e1.holdout == c].seed, e1[e1.holdout == c].rmse)}
        for c in REGIONS
    }

    out.mkdir(parents=True, exist_ok=True)
    train = {c: load_regions([c], "train", cache, quiet=True)[0] for c in REGIONS}
    test = {c: load_regions([c], "test", cache, quiet=True)[0] for c in REGIONS}
    write_manifest(
        out,
        build_manifest(
            extra={
                "run_mode": "wake-unit-k0",
                "registration": registration,
                "torch_version": torch.__version__,
                "cache": str(cache),
                "e1_raw": str(e1_raw),
                "configs": config_record(config_paths),
                "obs_unit": units,
                "sensitivity": sensitivity_commits,
                "settings": {
                    "seeds": list(SEEDS),
                    "epochs": EPOCHS,
                    "hidden": HIDDEN,
                    "profile": PROFILE,
                    "density": False,
                    "wake": False,
                    "fleet_columns": list(CLEAN_FLEET),
                    "draws": DRAWS,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "permutation_seed": PERMUTATION_SEED,
                    "planted_slope": PLANTED_SLOPE,
                },
                "regions": {
                    c: {"train": region_record(train[c]), "test": region_record(test[c])}
                    for c in REGIONS
                },
            }
        ),
    )

    results: dict[str, RegionResult] = {}
    for c in REGIONS:
        print(f"=== {c} ({units[c]}) ===", flush=True)
        results[c] = _rows_for_region(train[c], test[c], specs[c], e1_rmse[c])
        results[c].rows.to_csv(out / f"rows_{c}.csv.gz", index=False)
        del train[c], test[c]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    counts = {
        c: draw_counts(rng, len(results[c].block_ids), len(results[c].months), DRAWS)
        for c in REGIONS
    }
    slopes = {c: _slope_summary(results[c].rows, *counts[c]) for c in REGIONS}
    validation = {c: _validation(results[c].rows, *counts[c]) for c in REGIONS}

    def contrast(turb: dict[str, np.ndarray], agg: dict[str, np.ndarray]) -> np.ndarray:
        return np.mean([agg[c] for c in aggregate], axis=0) - np.mean(
            [turb[c] for c in turbine], axis=0
        )

    point = {c: np.array([slopes[c]["b"]]) for c in REGIONS}
    draws = {c: slopes[c]["draws"] for c in REGIONS}
    delta = float(contrast(point, point)[0])
    delta_lo, delta_hi = interval(contrast(draws, draws))

    dk = results["DK"].rows
    dk_own = dk[~dk["shared"]]
    own = _slope_summary(dk_own, *counts["DK"])
    delta_dk_own = float(contrast({**point, "DK": np.array([own["b"]])}, point)[0])
    ddo_lo, ddo_hi = interval(contrast({**draws, "DK": own["draws"]}, draws))

    onshore = dk[~dk["offshore"]]
    s_yes = _slope_summary(onshore[onshore["shared"]], *counts["DK"])
    s_no = _slope_summary(onshore[~onshore["shared"]], *counts["DK"])
    k0c = s_yes["b"] - s_no["b"]
    k0c_lo, k0c_hi = interval(s_yes["draws"] - s_no["draws"])

    slope_rows = []
    for c in REGIONS:
        s, rec = slopes[c], results[c].record
        slope_rows.append(
            {
                "region": c,
                "source": "gated",
                **{k: v for k, v in rec.items() if not isinstance(v, (dict, list))},
                "training_years": f"{rec['training_years'][0]}-{rec['training_years'][-1]}",
                "test_year": ",".join(map(str, rec["test_years"])),
                "b": s["b"],
                "b_lo": s["lo"],
                "b_hi": s["hi"],
                **{f"b_seed{k}": v for k, v in s["per_seed"].items()},
                **{f"rmse_seed{k}": v for k, v in rec["rmse_by_seed"].items()},
                **{f"e1_rmse_seed{k}": v for k, v in rec["e1_in_region_rmse_by_seed"].items()},
                "rmse_identical_to_e1": all(
                    abs(rec["rmse_by_seed"][k] - rec["e1_in_region_rmse_by_seed"].get(k, np.nan))
                    < 5e-6
                    for k in SEEDS
                ),
            }
        )
    gates = [
        *[
            {
                "gate": f"K0a_{c}",
                "value": slopes[c]["b"],
                "lo": slopes[c]["lo"],
                "hi": slopes[c]["hi"],
                "pass": bool(slopes[c]["b"] > 0 and slopes[c]["lo"] > 0),
            }
            for c in aggregate
        ],
        {
            "gate": "K0b",
            "value": delta,
            "lo": delta_lo,
            "hi": delta_hi,
            "pass": bool(delta > 0 and delta_lo > 0),
        },
        {
            "gate": "K0c",
            "value": k0c,
            "lo": k0c_lo,
            "hi": k0c_hi,
            "pass": bool(k0c > 0 and k0c_lo > 0),
        },
        {
            "gate": "reported_K0b_dk_own_target_only",
            "value": delta_dk_own,
            "lo": ddo_lo,
            "hi": ddo_hi,
            "pass": None,
        },
    ]
    bins = [_density_bins(results[c].rows, c) for c in REGIONS]

    for code, path in extra_caches.items():
        print(f"=== sensitivity {code} from {path} ===", flush=True)
        sub = out / "sensitivity" / code
        sub.mkdir(parents=True, exist_ok=True)
        r_train = load_regions([code], "train", path, quiet=True)[0]
        r_test = load_regions([code], "test", path, quiet=True)[0]
        res = _rows_for_region(r_train, r_test, specs[code], e1_rmse[code])
        if res.months != results[code].months:
            raise SystemExit(f"sensitivity {code}: months differ from the gated cache")
        # The registered draws are reused by giving every unit the block it has
        # in the gated cache. A unit the other cache cannot simulate has no row
        # there and is recorded; a unit only the other cache has is refused.
        gated = results[code].rows.drop_duplicates("ID").set_index("ID")["block"]
        extra_ids = sorted(set(res.rows["ID"]) - set(gated.index))
        if extra_ids:
            raise SystemExit(
                f"sensitivity {code}: units absent from the gated cache: {extra_ids[:5]}"
            )
        rows = res.rows.assign(block=res.rows["ID"].map(gated).astype(int))
        missing = sorted(set(gated.index) - set(rows["ID"]))
        gated_cap = results[code].rows.drop_duplicates("ID").set_index("ID")["capacity"]
        rows.to_csv(sub / f"rows_{code}.csv.gz", index=False)
        s = _slope_summary(rows, *counts[code])
        rec = {
            **res.record,
            "units_missing_vs_gated": len(missing),
            "capacity_share_missing_vs_gated": float(gated_cap[missing].sum() / gated_cap.sum()),
        }
        slope_rows.append(
            {
                "region": code,
                "source": f"sensitivity:{path}",
                **{k: v for k, v in rec.items() if not isinstance(v, (dict, list))},
                "training_years": f"{rec['training_years'][0]}-{rec['training_years'][-1]}",
                "test_year": ",".join(map(str, rec["test_years"])),
                "b": s["b"],
                "b_lo": s["lo"],
                "b_hi": s["hi"],
                **{f"b_seed{k}": v for k, v in s["per_seed"].items()},
                **{f"rmse_seed{k}": v for k, v in rec["rmse_by_seed"].items()},
            }
        )
        d_point = float(contrast({**point, code: np.array([s["b"]])}, point)[0])
        d_lo, d_hi = interval(contrast({**draws, code: s["draws"]}, draws))
        gates.append(
            {
                "gate": f"reported_K0b_with_{code}_sensitivity",
                "value": d_point,
                "lo": d_lo,
                "hi": d_hi,
                "pass": None,
            }
        )
        bins.append(_density_bins(rows, f"{code}:sensitivity"))

    pd.DataFrame(slope_rows).to_csv(out / "slopes.csv", index=False)
    pd.DataFrame([{"region": c, **validation[c]} for c in REGIONS]).to_csv(
        out / "validation.csv", index=False
    )
    pd.DataFrame(gates).to_csv(out / "gates.csv", index=False)
    pd.concat(bins, ignore_index=True).to_csv(out / "density_bins.csv", index=False)
    print(pd.DataFrame(slope_rows)[["region", "source", "b", "b_lo", "b_hi"]].to_string())
    print(pd.DataFrame(gates).to_string())


def cli(argv: list[str] | None = None) -> None:
    """Parse the registered command line and run."""
    ap = make_parser(__doc__)
    ap.add_argument("--cache", type=Path, default=CACHE)
    ap.add_argument("--config", action="append", default=[], metavar="CODE=PATH")
    ap.add_argument("--sensitivity", action="append", default=[], metavar="CODE=PATH")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--registration", default=REGISTRATION)
    ap.add_argument("--e1-raw", type=Path, default=E1_RAW)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args(argv)
    registered = {"seeds": list(SEEDS), "epochs": EPOCHS, "hidden": HIDDEN, "draws": DRAWS}
    given = {"seeds": args.seeds, "epochs": args.epochs, "hidden": args.hidden, "draws": args.draws}
    if given != registered:
        ap.error(f"these are registered and cannot change: expected {registered}, got {given}")
    main(
        cache=args.cache,
        configs=args.config,
        sensitivity=args.sensitivity,
        out=args.out,
        registration=args.registration,
        e1_raw=args.e1_raw,
        allow_dirty=args.allow_dirty,
    )


if __name__ == "__main__":
    cli()
