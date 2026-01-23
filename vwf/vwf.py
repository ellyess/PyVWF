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
import os
import time
from pathlib import Path
import itertools

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

pd.options.mode.chained_assignment = None  # default='warn'


class PyVWF:
    """
    This class trains and creates the virtual wind farm model.

    Attributes:
        path (str): path to where you want the outputs to be
        country (str): country code e.g. Denmark "DK"
        correct (bool):
            True - apply bias correction
            False - simulate uncorrected reanalysis
        calc_z0 (bool):
            True - calculate surface roughness using log wind profile
            False - use surface roughness from ERA-5, fsr variable
        cluster_mode (str):
            'all' forms clusters mixing onshore and offshore,
            'onshore' forms clusters with onshore ignoring offshore,
            'offshore' forms clusters with onshore ignoring onshore.
        obs_level (str):
            "turbine" - train using per-turbine observations (existing behaviour)
            "country" - train using country-wide observation series (no pseudo turbine splitting)
        cluster_list (list[int], optional): list of the spatial resolutions for the model. Defaults to None.
        time_res_list (list[str], optional): list of the temporal resolutions for the model. Defaults to None.
        add_nan (float, optional): percentage of data to randomly remove from training data 0 < add_nan < 1. Defaults to None.
        interp_nan (float, optional): set limit on simultaneous missing data points when interpolating nan. Defaults to None.
        fix_turb (str, optional): turbine model name as seen in models file, fixes to this turbine. Defaults to None.
    """

    def __init__(
        self,
        path,
        country,
        correct,
        calc_z0,
        cluster_mode,
        cluster_list=None,
        time_res_list=None,
        add_nan=None,
        interp_nan=None,
        fix_turb=None,
        *,
        obs_level: str = "turbine",  # NEW
    ):
        """
        Initialising the PyVWF object and creating necessary folders.
        """
        # validate obs_level early
        if obs_level not in ("turbine", "country"):
            raise ValueError("obs_level must be one of: 'turbine', 'country'")

        # creating folders
        directory_path = os.path.join(path, "run")
        run = country

        if correct:
            run += "-" + cluster_mode

            # include obs_level in run name so you don't overwrite factors
            run += f"-obs_{obs_level}"

            if (add_nan is None) & (interp_nan is None) & (fix_turb is None):
                run += "-corrected"
            else:
                if add_nan is not None:
                    run += "-r" + str(add_nan)
                if interp_nan is not None:
                    run += "-i" + str(interp_nan)
                if fix_turb is not None:
                    run += "-" + fix_turb
        else:
            run += "-uncorrected"

        # for calculated FSR
        if calc_z0:
            run += "-calc_z0"

        directory_path = os.path.join(directory_path, run)

        folder_names = [
            "training/correction-factors",
            "training/simulated-turbines",
            "results/capacity-factor",
            "results/wind-speed",
            "plots",
        ]
        print(f"Creating new directories in '{directory_path}':")

        for folder_name in folder_names:
            new_path = os.path.join(directory_path, folder_name)
            try:
                os.makedirs(new_path)
            except OSError:
                pass
            else:
                print(f"Created {new_path}")

        if correct:
            trained_res = []
            untrained_res = []
            untrained_cluster_list = []
            untrained_time_res_list = []

            for num_clu in cluster_list:
                for time_res in time_res_list:
                    my_file = Path(
                        directory_path
                        + "/training/correction-factors/"
                        + country
                        + "_factors_"
                        + time_res
                        + "_"
                        + str(num_clu)
                        + ".csv"
                    )
                    if my_file.is_file():
                        trained_res.append(str(num_clu) + "-" + time_res)
                    else:
                        untrained_res.append(str(num_clu) + "-" + time_res)
                        untrained_cluster_list.append(num_clu)
                        untrained_time_res_list.append(time_res)

            print("PyVWF is already trained for the following [Clusters-Temporal Resolution]:")
            print(trained_res)
            print("")
            print("PyVWF will be trained for:")
            print(untrained_res)
            print("--------------------------------")

            self.full_clus_list = cluster_list
            self.full_time_list = time_res_list
            self.untrained_time_res_list = untrained_time_res_list
            self.cluster_list = untrained_cluster_list
            self.time_res_list = untrained_time_res_list

            self.add_nan = add_nan
            self.interp_nan = interp_nan
            self.fix_turb = fix_turb

        self.cluster_mode = cluster_mode

        # NEW: store obs_level
        self.obs_level = obs_level

        self.country = country
        self.directory_path = directory_path
        self.correct = correct
        self.calc_z0 = calc_z0

    def train(self, check=False):
        """
        Derives bias correction factors at the desired spatiotemporal resolutions.
        """
        if len(self.cluster_list) < 1:
            print("All correction factors are trained ... Ending train.")
            print("--------------------------------")
            return self

        # NOTE: obs_level forwarded
        gen_cf, turb_info_train, reanalysis, power_curves = train_set(
            self.country,
            self.calc_z0,
            self.cluster_mode,
            add_nan=self.add_nan,
            interp_nan=self.interp_nan,
            fix_turb=self.fix_turb,
            obs_level=self.obs_level,  # NEW
        )

        turb_info_train.to_csv(
            self.directory_path
            + "/training/simulated-turbines/"
            + self.country
            + "_train_turb_info.csv",
            index=None,
        )

        print(
            "Training on ",
            len(turb_info_train),
            " turbines/farms | ",
            len(turb_info_train[turb_info_train["type"] == "onshore"]),
            " onshore | ",
            len(turb_info_train[turb_info_train["type"] == "offshore"]),
            " offshore",
        )

        for num_clu, time_res in zip(self.cluster_list, self.time_res_list):
            print("Deriving correction factors for PyVWF(", num_clu, ",", time_res, ") ...")
            start_time = time.time()

            # NOTE: obs_level forwarded
            train_bias_df, clus_info = cluster_train_set(
                gen_cf,
                time_res,
                num_clu,
                turb_info_train,
                obs_level=self.obs_level,  # NEW
            )

            # NEW: With country-wide observations, offsets are not identifiable per cluster
            # (and we only have a single cluster=0 anyway). Keep offset=0.
            if self.obs_level == "turbine":
                # parallelisation to find offset
                def find_offset_parallel(df):
                    """
                    Find offset parallel.
                    """
                    return df.apply(correction.find_offset, args=(clus_info, reanalysis, power_curves), axis=1)

                ddf = dd.from_pandas(train_bias_df, npartitions=40)
                ddf["offset"] = ddf.map_partitions(find_offset_parallel, meta=("offset", "float"))
                train_bias_df = ddf.compute(scheduler="processes")
            else:
                # ensure the column exists for format_bc_factors
                if "offset" not in train_bias_df.columns:
                    train_bias_df["offset"] = 0.0

            bc_factors = format_bc_factors(train_bias_df, time_res)
            bc_factors.to_csv(
                self.directory_path
                + "/training/correction-factors/"
                + self.country
                + "_factors_"
                + time_res
                + "_"
                + str(num_clu)
                + ".csv",
                index=False,
            )

            end_time = time.time()
            elapsed_time = end_time - start_time
            print("Completed and saved. Elapsed time: {:.2f} seconds\n".format(elapsed_time))
            print("--------------------------------")

        return self

    def simulate_cf(self, year_test, fix_turb_test=None):
        """
        Simulating capacity factor using the defined model.
        """
        # NOTE: obs_level forwarded
        obs_cf, turb_info, reanalysis, power_curves = val_set(
            self.country,
            self.calc_z0,
            self.cluster_mode,
            year_test,
            fix_turb_test,
            obs_level=self.obs_level,  # NEW
        )

        obs_cf.to_csv(
            self.directory_path
            + "/results/capacity-factor/"
            + self.country
            + "_"
            + str(year_test)
            + "_obs_cf.csv",
            index=None,
        )
        turb_info.to_csv(
            self.directory_path
            + "/training/simulated-turbines/"
            + self.country
            + "_"
            + str(year_test)
            + "_turb_info.csv",
            index=None,
        )

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

        my_file = Path(
            self.directory_path
            + "/results/capacity-factor/"
            + self.country
            + "_"
            + str(year_test)
            + "_unc_cf.csv"
        )
        if my_file.is_file():
            print("Uncorrected CF was previously simulated.\n")
        else:
            print("Simulating uncorrected CF ... ")
            unc_ws, unc_cf = wind.simulate_wind(reanalysis, turb_info, power_curves)
            unc_cf.to_csv(
                self.directory_path
                + "/results/capacity-factor/"
                + self.country
                + "_"
                + str(year_test)
                + "_unc_cf.csv",
                index=None,
            )

        if self.correct:
            turb_info_train = pd.read_csv(
                self.directory_path
                + "/training/simulated-turbines/"
                + self.country
                + "_train_turb_info.csv"
            )

            for num_clu, time_res in itertools.product(self.full_clus_list, self.full_time_list):
                my_file = Path(
                    self.directory_path
                    + "/results/capacity-factor/"
                    + self.country
                    + "_"
                    + str(year_test)
                    + "_"
                    + time_res
                    + "_"
                    + str(num_clu)
                    + "_cor_cf.csv"
                )
                if my_file.is_file():
                    print("PyVWF(", num_clu, "--", time_res, ") was previously simulated.\n")
                else:
                    print("Simulating CF using PyVWF(", num_clu, ", ", time_res, ") ...")
                    start_time = time.time()

                    # For obs_level="country" training, you end up with a single cluster (0).
                    # cluster_turbines will still work fine; if you want to force all=0 you can
                    # override here, but it's not necessary as long as bc_factors only has cluster=0.
                    if self.obs_level == "country":
                        # Country-level training only produces factors for a single cluster (0),
                        # so force every turbine into that cluster.
                        clus_info = turb_info.copy()
                        clus_info["cluster"] = 0
                    else:
                        clus_info = cluster_turbines(num_clu, turb_info_train, False, turb_info)

                    bc_factors = pd.read_csv(
                        self.directory_path
                        + "/training/correction-factors/"
                        + self.country
                        + "_factors_"
                        + time_res
                        + "_"
                        + str(num_clu)
                        + ".csv"
                    )

                    cor_ws, cor_cf = wind.simulate_wind(reanalysis, clus_info, power_curves, bc_factors, time_res)
                    cor_cf.to_csv(
                        self.directory_path
                        + "/results/capacity-factor/"
                        + self.country
                        + "_"
                        + str(year_test)
                        + "_"
                        + time_res
                        + "_"
                        + str(num_clu)
                        + "_cor_cf.csv",
                        index=None,
                    )

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("Completed and saved. Elapsed time: {:.2f} seconds".format(elapsed_time))
                    print(" ")

        self.turb_info = turb_info
        self.year_test = year_test

        return self

    def research_error(self):
        """
        Plots the overall error of the bias correction
        """
        temporal_metrics = metrics.overall_error(
            "temporal-focus",
            self.directory_path,
            self.country,
            self.turb_info,
            self.full_clus_list,
            self.full_time_list,
            False,
            self.year_test,
        )
        spatial_metrics = metrics.overall_error(
            "spatial-focus",
            self.directory_path,
            self.country,
            self.turb_info,
            self.full_clus_list,
            self.full_time_list,
            False,
            self.year_test,
        )
        total_metrics = metrics.overall_error(
            "total",
            self.directory_path,
            self.country,
            self.turb_info,
            self.full_clus_list,
            self.full_time_list,
            False,
            self.year_test,
        )

        plots.plot_overall_error(self.directory_path, self.country, total_metrics, "full")
        plots.plot_overall_error(self.directory_path, self.country, temporal_metrics, "temporal_focus")
        plots.plot_overall_error(self.directory_path, self.country, spatial_metrics, "spatial_focus")