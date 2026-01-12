"""
PyVWF module.

Summary
-------
Creates virtual wind farm models and simulations.

Data conventions
----------------
Tabular inputs are assumed to be tidy (one observation per row) unless stated otherwise.
Datetime columns are assumed to be timezone-naive UTC unless specified.

Units
-----
Wind speed: [m s^-1]; Hub height: [m]; Power: [MW]; Energy: [MWh]; Capacity factor: [-] (unless stated otherwise).

Assumptions
-----------
- ERA5/reanalysis fields are treated as representative at the chosen spatial/temporal resolution.
- Wake effects, curtailment, availability losses are not modelled unless explicitly implemented in this module.

References
----------
Add dataset and methodological references relevant to this module.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Dict, Tuple, Any
import logging
import itertools
import time

import pandas as pd
import dask.dataframe as dd

from vwf.data import (
    train_set,
    val_set,
    format_bc_factors,
    cluster_train_set,
) 
import vwf.wind as wind
import vwf.plots as plots
import vwf.metrics as metrics

import vwf.correction as correction

from vwf.clustering import (
    cluster_turbines
)


log = logging.getLogger(__name__)

@dataclass(frozen=True)
class RunConfig:
    country: str
    correct: bool
    calc_z0: bool
    cluster_mode: str
    cluster_list: Tuple[int, ...] = ()
    time_res_list: Tuple[str, ...] = ()
    add_nan: Optional[float] = None
    interp_nan: Optional[float] = None
    fix_turb: Optional[str] = None

class PyVWF:
    def __init__(self, path: str | Path, country: str, correct: bool, calc_z0: bool,
                cluster_mode: str, cluster_list=None, time_res_list=None,
                add_nan=None, interp_nan=None, fix_turb=None):
        self.base_path = Path(path)
        self.cfg = RunConfig(
            country=country,
            correct=bool(correct),
            calc_z0=bool(calc_z0),
            cluster_mode=str(cluster_mode),
            cluster_list=tuple(cluster_list or ()),
            time_res_list=tuple(time_res_list or ()),
            add_nan=add_nan,
            interp_nan=interp_nan,
            fix_turb=fix_turb,
        )
        self.directory_path = self.base_path / "run" / self._run_id()

        # Validate early
        if self.cfg.correct:
            if not self.cfg.cluster_list or not self.cfg.time_res_list:
                raise ValueError("When correct=True, cluster_list and time_res_list must be provided (non-empty).")

        # Keep your existing behavior but make it explicit:
        self.setup_dirs()
        self._discover_training_status()

    def _run_id(self) -> str:
        c = self.cfg
        run = c.country
        if c.correct:
            run += f"-{c.cluster_mode}"
            if c.add_nan is None and c.interp_nan is None and c.fix_turb is None:
                run += "-corrected"
            else:
                if c.add_nan is not None:
                    run += f"-r{c.add_nan}"
                if c.interp_nan is not None:
                    run += f"-i{c.interp_nan}"
                if c.fix_turb is not None:
                    run += f"-{c.fix_turb}"
        else:
            run += "-uncorrected"
        if c.calc_z0:
            run += "-calc_z0"
        return run

    def setup_dirs(self) -> None:
        folders = [
            "training/correction-factors",
            "training/simulated-turbines",
            "results/capacity-factor",
            "results/wind-speed",
            "plots",
        ]
        for f in folders:
            (self.directory_path / f).mkdir(parents=True, exist_ok=True)

    def _factor_path(self, time_res: str, num_clu: int) -> Path:
        return self.directory_path / "training/correction-factors" / f"{self.cfg.country}_factors_{time_res}_{num_clu}.csv"

    def _discover_training_status(self) -> None:
        if not self.cfg.correct:
            self.trained = {}
            self.untrained = {}
            return

        trained, untrained = [], []
        for num_clu in self.cfg.cluster_list:
            for time_res in self.cfg.time_res_list:
                if self._factor_path(time_res, num_clu).is_file():
                    trained.append((num_clu, time_res))
                else:
                    untrained.append((num_clu, time_res))

        self.trained = set(trained)
        self.untrained = set(untrained)

        log.info("Already trained: %s", trained)
        log.info("To train: %s", untrained)

    def train(self, check: bool = False, npartitions: int = 40, save: bool = True):
        if not self.cfg.correct:
            raise ValueError("train() only applies when correct=True.")

        if not self.untrained:
            log.info("All correction factors are trained; skipping.")
            return {}

        # existing call
        gen_cf, turb_info_train, reanalysis, power_curves = train_set(
            self.cfg.country, self.cfg.calc_z0, self.cfg.cluster_mode,
            add_nan=self.cfg.add_nan, interp_nan=self.cfg.interp_nan, fix_turb=self.cfg.fix_turb
        )

        if save:
            turb_info_train.to_csv(self.directory_path / "training/simulated-turbines" / f"{self.cfg.country}_train_turb_info.csv", index=None)

        out: Dict[Tuple[int, str], pd.DataFrame] = {}
        for (num_clu, time_res) in sorted(self.untrained):
            start = time.time()
            train_bias_df, clus_info = cluster_train_set(gen_cf, time_res, num_clu, turb_info_train)

            def part_apply(df: pd.DataFrame) -> pd.Series:
                return df.apply(correction.find_offset, args=(clus_info, reanalysis, power_curves), axis=1)

            ddf = dd.from_pandas(train_bias_df, npartitions=npartitions)
            offsets = ddf.map_partitions(part_apply, meta=("offset", "float64")).compute(scheduler="processes")
            train_bias_df = train_bias_df.assign(offset=offsets)

            bc_factors = format_bc_factors(train_bias_df, time_res)
            out[(num_clu, time_res)] = bc_factors

            if save:
                bc_factors.to_csv(self._factor_path(time_res, num_clu), index=False)

            log.info("Trained (%s, %s) in %.2fs", num_clu, time_res, time.time() - start)

        # refresh status
        self._discover_training_status()
        return out


def simulate_cf(
    self,
    year_test: int,
    fix_turb_test: str | None = None,
    *,
    save: bool = True,
    overwrite: bool = False,
    return_data: bool = True,
):
    """
    Simulate capacity factor time series for a given validation year.

    This method:
        1) Loads validation inputs (obs_cf, turb_info, reanalysis, power_curves).
        2) Simulates uncorrected CF if not already cached (or overwrite=True).
        3) If self.correct=True, simulates corrected CF for each (cluster, time_res).

    Parameters
    ----------
    year_test : int
        Year to simulate.
    fix_turb_test : str, optional
        Turbine model name to use as power curve for validation. Default None.
    save : bool, keyword-only
        If True, write CSV outputs to the run directory (same filenames as before).
    overwrite : bool, keyword-only
        If True, recompute and overwrite cached CSVs.
    return_data : bool, keyword-only
        If True, return a dict of outputs as pandas DataFrames.

    Returns
    -------
    dict or PyVWF
        If return_data=True, returns a dict:
            {
            "obs_cf": DataFrame,
            "turb_info": DataFrame,
            "unc_cf": DataFrame,
            "corrected": {(num_clu, time_res): DataFrame, ...}
            }
        Otherwise returns self.
    """
    from pathlib import Path
    import itertools
    import time
    import pandas as pd

    # --- paths ---
    base = Path(self.directory_path)
    cf_dir = base / "results" / "capacity-factor"
    train_turb_dir = base / "training" / "simulated-turbines"
    factors_dir = base / "training" / "correction-factors"

    obs_cf_path = cf_dir / f"{self.country}_{year_test}_obs_cf.csv"
    unc_cf_path = cf_dir / f"{self.country}_{year_test}_unc_cf.csv"
    year_turb_info_path = train_turb_dir / f"{self.country}_{year_test}_turb_info.csv"
    train_turb_info_path = train_turb_dir / f"{self.country}_train_turb_info.csv"

    # --- load and preprocess validation data ---
    obs_cf, turb_info, reanalysis, power_curves = val_set(
        self.country, self.calc_z0, self.cluster_mode, year_test, fix_turb_test
    )

    # Persist validation inputs for research reproducibility (optional)
    if save:
        if overwrite or not obs_cf_path.exists():
            obs_cf.to_csv(obs_cf_path, index=None)
        if overwrite or not year_turb_info_path.exists():
            turb_info.to_csv(year_turb_info_path, index=None)

    print(
        "Simulating ",
        len(turb_info),
        " turbines/farms | ",
        len(turb_info[turb_info["type"] == "onshore"]),
        " onshore | ",
        len(turb_info[turb_info["type"] == "offshore"]),
        " offshore",
    )
    print(" ")

    # --- uncorrected simulation (cached) ---
    unc_cf = None
    if unc_cf_path.is_file() and not overwrite:
        print("Uncorrected CF was previously simulated.\n")
        if return_data:
            unc_cf = pd.read_csv(unc_cf_path)
    else:
        print("Simulating uncorrected CF ... ")
        _unc_ws, unc_cf_calc = wind.simulate_wind(reanalysis, turb_info, power_curves)
        unc_cf = unc_cf_calc
        if save:
            unc_cf.to_csv(unc_cf_path, index=None)

    # --- corrected simulations (if enabled) ---
    corrected: dict[tuple[int, str], pd.DataFrame] = {}

    if self.correct:
        # Ensure we have the training turbine info used to form clusters
        if not train_turb_info_path.is_file():
            raise FileNotFoundError(
                f"Missing training turbine info: {train_turb_info_path}\n"
                f"Run `train()` first (or ensure training outputs exist in this run directory)."
            )

        turb_info_train = pd.read_csv(train_turb_info_path)

        # Prefer full lists if they exist (your code sets these in __init__ when correct=True)
        full_clus_list = getattr(self, "full_clus_list", None) or self.cluster_list
        full_time_list = getattr(self, "full_time_list", None) or self.time_res_list

        for num_clu, time_res in itertools.product(full_clus_list, full_time_list):
            cor_cf_path = cf_dir / f"{self.country}_{year_test}_{time_res}_{num_clu}_cor_cf.csv"
            factors_path = factors_dir / f"{self.country}_factors_{time_res}_{num_clu}.csv"

            if cor_cf_path.is_file() and not overwrite:
                print(f"PyVWF({num_clu}--{time_res}) was previously simulated.\n")
                if return_data:
                    corrected[(num_clu, time_res)] = pd.read_csv(cor_cf_path)
                continue

            if not factors_path.is_file():
                raise FileNotFoundError(
                    f"Missing correction factors for PyVWF({num_clu}, {time_res}): {factors_path}\n"
                    f"Run `train()` for this (cluster, time_res) or check your run directory."
                )

            print(f"Simulating CF using PyVWF({num_clu}, {time_res}) ...")
            start_time = time.time()

            # Build clusters for this validation set against training turbine info
            clus_info = cluster_turbines(num_clu, turb_info_train, False, turb_info)

            # Load bias correction factors
            bc_factors = pd.read_csv(factors_path)

            # Simulate corrected wind/CF
            _cor_ws, cor_cf = wind.simulate_wind(
                reanalysis, clus_info, power_curves, bc_factors, time_res
            )

            if save:
                cor_cf.to_csv(cor_cf_path, index=None)

            if return_data:
                corrected[(num_clu, time_res)] = cor_cf

            elapsed_time = time.time() - start_time
            print(f"Completed and saved. Elapsed time: {elapsed_time:.2f} seconds")
            print(" ")

    # Keep these for backwards compatibility with your research_error() method
    self.turb_info = turb_info
    self.year_test = year_test

    if not return_data:
        return self

    return {
        "obs_cf": obs_cf,
        "turb_info": turb_info,
        "unc_cf": unc_cf,
        "corrected": corrected,
    }
            
    def research_error(self):
        """
        Plots the overall error of the bias correction
        """
        temporal_metrics = metrics.overall_error('temporal-focus', self.directory_path, self.country, self.turb_info, self.full_clus_list, self.full_time_list, False, self.year_test)
        spatial_metrics = metrics.overall_error('spatial-focus', self.directory_path, self.country, self.turb_info, self.full_clus_list, self.full_time_list, False, self.year_test)
        total_metrics = metrics.overall_error('total', self.directory_path, self.country, self.turb_info, self.full_clus_list, self.full_time_list, False, self.year_test)
        # plot_overall_error(run, country, train_metrics, 'train')
        plots.plot_overall_error(self.directory_path, self.country, total_metrics, 'full')
        plots.plot_overall_error(self.directory_path, self.country, temporal_metrics, 'temporal_focus')
        plots.plot_overall_error(self.directory_path, self.country, spatial_metrics, 'spatial_focus')

