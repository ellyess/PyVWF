"""Does the US simulated capacity-factor deficit grow with sub-grid terrain?

The driver of `docs/findings/method-terrain-wind-deficit-prereg.md`, registered
on 2026-09-19 before this file existed. Everything it gates on is fixed there
and is a module constant here: the training years, the single confirmation
year, the nine clusters named in advance, the draw count and seed, and each
gate's threshold. Only paths are flags.

Per plant it computes the deficit `Y = ln(sum observed CF / sum uncorrected
simulated CF)` over the months with both values, the sub-grid relief `R` (the
standard deviation of 30 arc-second ETOPO elevation over the plant's ERA5
grid cell), the site elevation above that cell `H`, and the curve-match class
rebuilt with `scripts/analysis/curve_match_audit.py`'s own functions. It then
reads the five registered gates.

Issue #26 has the first look this study tests, and says why the 2022 figures
there cannot serve as its confirmation.

    PYVWF_INPUT=input/combined PYTHONPATH=src python \\
        scripts/studies/method-terrain-wind-deficit/terrain_deficit.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYVWF_INPUT", "input/combined")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from pyvwf.cli.common import make_parser  # noqa: E402
from pyvwf.data import train_set  # noqa: E402
from pyvwf.harness.driver import era5_dir, load_obs_and_fleet, resolve_source  # noqa: E402
from pyvwf.harness.regions import load_region  # noqa: E402

warnings.simplefilter("ignore")

CONFIG = Path("configs/regions/scorecard/us_k250.toml")
TRAIN_RUN = Path("output/validation/bracketed_2026-09-19/US/train-bracketed")
EVALUATE_RUN = Path("output/validation/bracketed_2026-09-19/US/evaluate-2022-bracketed")
ETOPO = Path("input/reference/terrain/etopo_global.nc")
AUDIT = Path("scripts/analysis/curve_match_audit.py")
OUT = Path("output/terrain_wind_deficit_2026-09-20_corrected")

#: Registered in the pre-registration; none of these is a flag.
TRAIN_YEARS = (2019, 2020, 2021)
TEST_YEAR = 2022
MIN_MONTHS = 12
CELL_DEGREES = 0.25
TAIL_CLUSTERS = (15, 27, 38, 102, 109, 156, 183, 232, 236)
DRAWS = 1000
SEED = 20260919
G1_MIN_RHO = 0.20
G2_MIN_PERCENTILE = 75.0
G3_MIN_PLANTS_PER_SIDE = 3
G3_MIN_KEYS = 5
G3_MIN_SHARE = 2 / 3
#: The scorecard's US other-brand share of capacity. The per-plant classes are
#: rebuilt here, so they are checked against the published aggregate before G4
#: is read; a mismatch makes G4 unassessable rather than wrong. The share is
#: taken over the FITTED FLEET, as the curve-match audit takes it. The first
#: run took it over the plants with an outcome instead, a third of the fleet,
#: which reads 0.476 and made G4 unassessable on a miscomputed precondition.
PUBLISHED_OTHER_BRAND_SHARE = 0.483
#: The recommendation boundary of the pre-registration's consequences section.
REGIME_MIN_RHO = 0.40
REGIME_MIN_TOP_DECILE_Y = math.log(2)


def _audit_functions(audit_path: Path):
    """The curve-match audit's own classifier, loaded by path.

    The audit script writes aggregates only, so the per-plant classes it
    computes internally are rebuilt here with the same three functions rather
    than a second implementation of the rule.
    """
    spec = importlib.util.spec_from_file_location("curve_match_audit", audit_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deficit_per_plant(gen_cf: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    """``Y`` and its month count per plant, over ``years``."""
    frame = gen_cf[gen_cf["year"].isin(years)].dropna(subset=["obs", "sim"])
    frame = frame[frame["sim"] > 0]
    grouped = frame.groupby("ID").agg(
        obs=("obs", "sum"), sim=("sim", "sum"), months=("obs", "size")
    )
    grouped = grouped[grouped["months"] >= MIN_MONTHS]
    grouped["Y"] = np.log(grouped["obs"] / grouped["sim"])
    return grouped.reset_index()


def terrain_per_plant(fleet: pd.DataFrame, grid_lon, grid_lat, etopo_path: Path) -> pd.DataFrame:
    """Sub-grid relief ``R`` and site elevation above the cell ``H``, per plant.

    The cell is the ERA5 grid box centred on the grid point nearest the plant,
    ``CELL_DEGREES`` on a side, and the elevation is ETOPO's 30 arc-second
    grid within it.
    """
    half = CELL_DEGREES / 2
    lon_centres, lat_centres = np.asarray(grid_lon), np.asarray(grid_lat)
    with xr.open_dataset(etopo_path) as etopo:
        elevation = etopo["z"]
        # ETOPO's latitude may run either way; slice in the file's own order.
        descending = float(elevation["lat"][0]) > float(elevation["lat"][-1])
        rows = []
        for unit, lon, lat in zip(fleet["ID"], fleet["lon"], fleet["lat"]):
            cell_lon = float(lon_centres[np.argmin(np.abs(lon_centres - lon))])
            cell_lat = float(lat_centres[np.argmin(np.abs(lat_centres - lat))])
            lats = (
                (cell_lat + half, cell_lat - half)
                if descending
                else (cell_lat - half, cell_lat + half)
            )
            box = elevation.sel(
                lon=slice(cell_lon - half, cell_lon + half), lat=slice(*lats)
            ).to_numpy()
            site = float(elevation.sel(lon=lon, lat=lat, method="nearest"))
            rows.append(
                {
                    "ID": unit,
                    "cell_lon": cell_lon,
                    "cell_lat": cell_lat,
                    "n_etopo": int(box.size),
                    "R": float(np.std(box)),
                    "H": site - float(np.mean(box)),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_rho(x: np.ndarray, y: np.ndarray, draws: int = DRAWS, seed: int = SEED):
    """Spearman's rho and its percentile interval over plant resamples."""
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    n = len(x)
    draws_rho = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            continue
        draws_rho.append(spearmanr(x[idx], y[idx]).statistic)
    low, high = np.percentile(draws_rho, [2.5, 97.5])
    return rho, float(low), float(high)


def curve_classes(fleet: pd.DataFrame, train_run: Path, audit_path: Path) -> pd.Series:
    """Each plant's curve-match class, from the audit script's own functions."""
    import hashlib
    import json

    audit = _audit_functions(audit_path)
    library = json.loads((train_run / "run_manifest.json").read_text())["curve_library"]
    models_file = audit._models_file(library)
    models = pd.read_csv(models_file)
    lut = models.drop_duplicates("model").set_index("model")["manufacturer"]
    assert hashlib.sha256(Path(models_file).read_bytes()).hexdigest() == library["models_sha256"]
    curve_side = audit.curve_side_manufacturer(fleet["model"], lut)
    own, _ = audit.own_manufacturer("US", fleet)
    return pd.Series(
        [audit.classify(o, m) for o, m in zip(own, curve_side)],
        index=fleet.index,
        name="curve_class",
    )


def within_key_control(table: pd.DataFrame) -> tuple[int, int, pd.DataFrame]:
    """G3: the same model key at high and low relief.

    Returns the number of keys where high-relief plants fall further short,
    the number of qualifying keys, and the per-key table.
    """
    median_r = table["R"].median()
    rows = []
    for key, group in table.groupby("model"):
        high = group[group["R"] > median_r]
        low = group[group["R"] <= median_r]
        if len(high) < G3_MIN_PLANTS_PER_SIDE or len(low) < G3_MIN_PLANTS_PER_SIDE:
            continue
        rows.append(
            {
                "model": key,
                "n_high": len(high),
                "n_low": len(low),
                "median_Y_high": high["Y"].median(),
                "median_Y_low": low["Y"].median(),
                "high_falls_further": bool(high["Y"].median() > low["Y"].median()),
            }
        )
    keys = pd.DataFrame(rows)
    if keys.empty:
        return 0, 0, keys
    return int(keys["high_falls_further"].sum()), len(keys), keys


def confirmation_deficit(evaluate_run: Path, spec) -> pd.DataFrame:
    """``Y`` per plant in the test year, from the evaluation run's frames."""
    observed, _ = load_obs_and_fleet(spec, TEST_YEAR)
    simulated = pd.read_csv(evaluate_run / "unc_cf.csv", parse_dates=["time"]).set_index("time")
    monthly = simulated.resample("MS").mean()
    obs_long = observed.melt(id_vars="time", var_name="ID", value_name="obs").assign(
        month=lambda d: pd.to_datetime(d["time"]).dt.month
    )
    sim_long = (
        monthly.stack()
        .rename("sim")
        .reset_index()
        .rename(columns={"level_1": "ID"})
        .assign(month=lambda d: d["time"].dt.month)
    )
    obs_long["ID"] = obs_long["ID"].astype(str)
    sim_long["ID"] = sim_long["ID"].astype(str)
    both = obs_long.merge(sim_long[["ID", "month", "sim"]], on=["ID", "month"]).dropna(
        subset=["obs", "sim"]
    )
    both = both[both["sim"] > 0]
    grouped = both.groupby("ID").agg(obs=("obs", "sum"), sim=("sim", "sum"), months=("obs", "size"))
    grouped = grouped[grouped["months"] >= MIN_MONTHS]
    grouped["Y"] = np.log(grouped["obs"] / grouped["sim"])
    return grouped.reset_index()


def main(
    config: Path = CONFIG,
    train_run: Path = TRAIN_RUN,
    evaluate_run: Path = EVALUATE_RUN,
    etopo: Path = ETOPO,
    audit: Path = AUDIT,
    out: Path = OUT,
) -> int:
    """Build the per-plant table, then read the five registered gates."""
    spec = load_region(config)
    fleet = pd.read_csv(train_run / "train_turb_info_250.csv", low_memory=False)
    fleet["ID"] = fleet["ID"].astype(str)

    gen_cf, turb_info, reanalysis, power_curves = train_set(
        spec.code,
        True,
        "all",
        obs_level=spec.obs_level,
        source=resolve_source(spec),
        era5_dir=era5_dir(spec),
        bbox=spec.bbox,
        allow_extrapolation=spec.allow_extrapolation,
        roughness=spec.roughness,
    )
    gen_cf["ID"] = gen_cf["ID"].astype(str)

    deficit = deficit_per_plant(gen_cf, TRAIN_YEARS)
    terrain = terrain_per_plant(
        fleet, reanalysis["lon"].to_numpy(), reanalysis["lat"].to_numpy(), etopo
    )
    classes = curve_classes(fleet, train_run, audit)
    table = (
        fleet[["ID", "cluster", "capacity", "model", "site_name", "lon", "lat"]]
        .assign(curve_class=classes.to_numpy())
        .merge(terrain, on="ID")
        .merge(deficit[["ID", "Y", "months"]], on="ID")
    )

    fleet_capacity = pd.to_numeric(fleet["capacity"], errors="coerce").fillna(0.0)
    share = float(
        fleet_capacity[classes.to_numpy() == "different-brand"].sum() / fleet_capacity.sum()
    )
    g4_assessable = abs(round(share, 3) - PUBLISHED_OTHER_BRAND_SHARE) < 0.0005

    print(f"plants with an outcome: {len(table)} of {len(fleet)} in the fitted fleet")
    print(
        f"other-brand share of capacity, fitted fleet: {share:.4f} "
        f"(published {PUBLISHED_OTHER_BRAND_SHARE})"
    )
    print(table[["R", "H", "Y"]].describe().to_string())

    # Reported, not gated: the plants with an outcome are the ones that report
    # monthly, and nothing says they sit in the same terrain as the fleet. The
    # direction matters for reading the relation: relief higher among them
    # would mean the relation is measured on rougher ground than the fleet's.
    fleet_relief = terrain.merge(fleet[["ID", "capacity"]], on="ID")
    scored = fleet_relief[fleet_relief["ID"].isin(table["ID"])]
    coverage = pd.DataFrame(
        {
            "population": ["fitted fleet", "with an outcome"],
            "plants": [len(fleet_relief), len(scored)],
            "R_median": [fleet_relief["R"].median(), scored["R"].median()],
            "R_mean": [fleet_relief["R"].mean(), scored["R"].mean()],
            "R_p90": [fleet_relief["R"].quantile(0.9), scored["R"].quantile(0.9)],
        }
    )
    direction = "higher" if scored["R"].median() > fleet_relief["R"].median() else "lower or equal"
    print("\nRelief of the plants with an outcome against the fitted fleet (reported, not gated):")
    print(coverage.to_string(index=False))
    print(f"  median relief among the scored plants is {direction} than the fleet's")

    x, y = table["R"].to_numpy(), table["Y"].to_numpy()
    rho, low, high = bootstrap_rho(x, y)
    rho_h, low_h, high_h = bootstrap_rho(table["H"].to_numpy(), y)
    g1 = rho >= G1_MIN_RHO and low > 0

    table["R_percentile"] = table["R"].rank(pct=True) * 100
    tail = table[table["cluster"].isin(TAIL_CLUSTERS)]
    tail_median = float(tail["R_percentile"].median()) if len(tail) else float("nan")
    g2 = tail_median >= G2_MIN_PERCENTILE

    agreeing, qualifying, keys = within_key_control(table)
    g3_assessable = qualifying >= G3_MIN_KEYS
    g3 = g3_assessable and agreeing / qualifying >= G3_MIN_SHARE

    kept = table[~table["curve_class"].isin(["different-brand", "unverifiable"])]
    rho_kept, low_kept, _ = bootstrap_rho(kept["R"].to_numpy(), kept["Y"].to_numpy())
    g4 = g4_assessable and rho_kept >= G1_MIN_RHO and low_kept > 0

    confirm = confirmation_deficit(evaluate_run, spec).merge(table[["ID", "R"]], on="ID")
    rho_test, low_test, high_test = bootstrap_rho(confirm["R"].to_numpy(), confirm["Y"].to_numpy())
    g5 = rho_test > 0 and low_test > 0

    top_decile_y = float(table.loc[table["R"] >= table["R"].quantile(0.9), "Y"].median())
    gates = pd.DataFrame(
        [
            {"gate": "G1 relation", "value": rho, "low": low, "high": high, "pass": g1},
            {"gate": "G2 tail", "value": tail_median, "low": np.nan, "high": np.nan, "pass": g2},
            {
                "gate": "G3 curve control",
                "value": agreeing / qualifying if qualifying else np.nan,
                "low": qualifying,
                "high": np.nan,
                "pass": g3 if g3_assessable else "not assessable",
            },
            {
                "gate": "G4 matching control",
                "value": rho_kept,
                "low": low_kept,
                "high": np.nan,
                "pass": g4 if g4_assessable else "not assessable",
            },
            {
                "gate": "G5 confirmation",
                "value": rho_test,
                "low": low_test,
                "high": high_test,
                "pass": g5,
            },
        ]
    )
    print(f"\nSecondary covariate H (gates nothing): rho {rho_h:.3f} [{low_h:.3f}, {high_h:.3f}]")
    print(f"Top-decile relief median Y: {top_decile_y:.3f} (ln 2 = {REGIME_MIN_TOP_DECILE_Y:.3f})")
    print("\n" + gates.to_string(index=False))
    print(
        "\nRecommendation boundary: a separate correction regime needs "
        f"rho >= {REGIME_MIN_RHO} and top-decile median Y >= ln 2; otherwise a terrain flag."
    )

    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "us_plant_deficit.csv", index=False)
    coverage.to_csv(out / "us_relief_coverage.csv", index=False)
    keys.to_csv(out / "us_within_key_control.csv", index=False)
    confirm.to_csv(out / "us_confirmation_2022.csv", index=False)
    gates.to_csv(out / "us_gates.csv", index=False)
    print(f"\nwrote {out}/us_plant_deficit.csv and four more")
    return 0


def cli(argv: list[str] | None = None) -> int:
    """Parse the recorded command line, which has no arguments, and run :func:`main`.

    Every path is a flag whose default is the path the registered run reads or
    writes, so a re-run can read another run's frames or write beside the
    record. The years, the named clusters, the seed and the gates are
    constants: a flag would let a run differ from its registration silently.
    """
    parser = make_parser(__doc__)
    parser.add_argument(
        "--config", type=Path, default=CONFIG, help=f"Region config (default: {CONFIG})"
    )
    parser.add_argument(
        "--train-run", type=Path, default=TRAIN_RUN, help=f"Training run (default: {TRAIN_RUN})"
    )
    parser.add_argument(
        "--evaluate-run",
        type=Path,
        default=EVALUATE_RUN,
        help=f"Evaluation run for the confirmation year (default: {EVALUATE_RUN})",
    )
    parser.add_argument("--etopo", type=Path, default=ETOPO, help=f"ETOPO grid (default: {ETOPO})")
    parser.add_argument(
        "--audit", type=Path, default=AUDIT, help=f"Curve-match audit script (default: {AUDIT})"
    )
    parser.add_argument("--out", type=Path, default=OUT, help=f"Output directory (default: {OUT})")
    args = parser.parse_args(argv)
    return main(
        config=args.config,
        train_run=args.train_run,
        evaluate_run=args.evaluate_run,
        etopo=args.etopo,
        audit=args.audit,
        out=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(cli())
