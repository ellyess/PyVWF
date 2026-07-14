"""Core PyVWF training and simulation workflow."""
import os

# Prevent OpenBLAS/MKL thread contention when using Dask process workers.
# Must be set before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
from pathlib import Path
import itertools

import pandas as pd
import numpy as np
import dask.dataframe as dd

from vwf.data import (
    train_set,
    val_set,
    format_bc_factors,
    cluster_train_set,
)
import vwf.wind as wind

import vwf.correction as correction

from vwf.clustering import (
    cluster_turbines
)

from vwf.sources import InMemoryCountrySource, ObservationSource

pd.options.mode.chained_assignment = None  # default='warn'


class PyVWF:
    """Train and run the virtual wind farm model.

    Supports two workflows:
    1. **Turbine-level** (obs_level="turbine"): Uses actual turbine locations and observations
    2. **Country-level** (obs_level="country"): Uses grid-based sampling with representative metadata

    Attributes:
        path: Output path for model artifacts.
        country: Country code (e.g., "DK").
        correct: If True, apply bias correction; otherwise simulate uncorrected reanalysis.
        calc_z0: If True, calculate surface roughness from wind profiles; otherwise
            use ERA5 roughness (fsr).
        cluster_mode: Clustering mode ("all", "onshore", "offshore").
        obs_level: Observation level ("turbine" or "country").
        cluster_list: List of spatial resolutions for the model.
        time_res_list: List of temporal resolutions for the model.
        add_nan: Fraction of data to randomly remove from training data.
        interp_nan: Limit on simultaneous missing data points when interpolating.
        fix_turb: Turbine model name to fix to a single model.
        grid_points: Optional grid points DataFrame (for country-level workflows).
        obs_data_train: Optional training observations DataFrame (for country-level workflows).
        obs_data_test: Optional test observations DataFrame (for country-level workflows).

    Country-Level Workflow Example:
        >>> # Generate data first using scripts/generate_country_level_training_data.py
        >>> from vwf.vwf import PyVWF
        >>> import sys, pandas as pd
        >>> sys.path.insert(0, "input/country_level_data")
        >>> from pyvwf_config import get_config
        >>>
        >>> config = get_config("NL")
        >>> grid_points = pd.read_csv(config["grid_points_path"])
        >>> obs_train = pd.read_csv(config["train_obs_path"], index_col=0, parse_dates=True)
        >>> obs_test = pd.read_csv(config["test_obs_path"], index_col=0, parse_dates=True)
        >>>
        >>> model = PyVWF(
        ...     path="",
        ...     country="NL",
        ...     correct=True,
        ...     calc_z0=True,
        ...     cluster_mode="all",
        ...     cluster_list=[5],
        ...     time_res_list=["fixed", "season"],
        ...     obs_level="country"
        ... )
        >>> model.load_country_data(grid_points, obs_train, obs_test)
        >>> model.train(check=False)
        >>> model.simulate_cf(2021)
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
        obs_level: str = "turbine",
    ):
        """Initialize the PyVWF object and create output folders."""
        if obs_level not in ("turbine", "country"):
            raise ValueError("obs_level must be one of: 'turbine', 'country'")

        # creating folders
        # If path is empty or ".", use default output/run structure
        # Otherwise use provided path directly (for organized multi-run training)
        if not path or path == ".":
            directory_path = os.path.join("output", "run")
        else:
            directory_path = path
        
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

        # Country-level data: unset until load_country_data(),
        # load_country_data_with_year_specific() or from_config() populates it.
        self.grid_points: pd.DataFrame | None = None
        self.obs_data_train: pd.DataFrame | None = None
        self.obs_data_test: pd.DataFrame | None = None
        self.grid_points_by_year: dict[int, pd.DataFrame] | None = None

        # Observation sources built from the country-level data above. Left as
        # None for turbine-level runs, where the source is resolved by country.
        self.source_train: ObservationSource | None = None
        self.source_test: ObservationSource | None = None

        self.country = country
        self.directory_path = directory_path
        self.correct = correct
        self.calc_z0 = calc_z0

    def _build_country_sources(self) -> None:
        """Wrap the loaded country-level frames in observation sources.

        Raises:
            ValueError: If the country-level frames have not been loaded yet.
        """
        if (
            self.grid_points is None
            or self.obs_data_train is None
            or self.obs_data_test is None
        ):
            raise ValueError(
                "Country-level data has not been loaded. Call load_country_data(), "
                "load_country_data_with_year_specific(), or from_config() first."
            )
        self.source_train = InMemoryCountrySource(self.grid_points, self.obs_data_train)
        self.source_test = InMemoryCountrySource(self.grid_points, self.obs_data_test)

    def load_country_data(
        self,
        grid_points: pd.DataFrame,
        obs_train: pd.DataFrame,
        obs_test: pd.DataFrame,
    ):
        """Load grid points and observations for country-level workflow.

        This method allows you to provide externally generated grid points and
        observations (e.g., from generate_country_level_training_data.py) instead
        of using the built-in loaders.

        Args:
            grid_points: Grid points with columns: lat, lon, ID, height, model, capacity, cluster.
            obs_train: Training observations with index=timestamp and columns: capacity_factor, generation_mw, capacity_mw.
            obs_test: Test observations with same format as obs_train.

        Returns:
            Self for method chaining.

        Example:
            >>> model = PyVWF("", "NL", True, True, "all", [5], ["fixed"], obs_level="country")
            >>> model.load_country_data(grid_points, obs_train, obs_test)
            >>> model.train(False)
        """
        if self.obs_level != "country":
            print("Warning: load_country_data() is intended for obs_level='country'")

        self.grid_points = grid_points.copy()
        self.obs_data_train = obs_train.copy()
        self.obs_data_test = obs_test.copy()
        self._build_country_sources()

        print("✓ Loaded country-level data:")
        print(f"  Grid points: {len(self.grid_points)} points")
        if 'cluster' in self.grid_points.columns:
            print(f"  Clusters: {self.grid_points['cluster'].nunique()}")
        if 'zone' in self.grid_points.columns:
            print(f"  Zones: {list(self.grid_points['zone'].unique())}")
        print(f"  Training observations: {len(self.obs_data_train)} timesteps")
        print(f"  Test observations: {len(self.obs_data_test)} timesteps")

        return self

    def load_country_data_with_year_specific(
        self,
        obs_train: pd.DataFrame,
        obs_test: pd.DataFrame,
        grid_points_dir: Path | None = None,
    ):
        """Load year-specific grid points and observations for country-level workflow.

        This advanced method automatically loads year-specific grid points that reflect
        the installed capacity for each year in the training data. This ensures that
        bias corrections account for the actual turbine fleet at the time.

        Args:
            obs_train: Training observations with index=timestamp and columns: capacity_factor, generation_mw, capacity_mw.
                      Index must be DatetimeIndex with year information.
            obs_test: Test observations with same format as obs_train.
            grid_points_dir: Directory containing year-specific grid point files
                           (default: input/country_level_data/grid_points/{country}/)

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If training data spans multiple years and year-specific files are missing.

        Example:
            >>> model = PyVWF("", "NL", True, True, "all", [5], ["fixed"], obs_level="country")
            >>> obs_train = pd.read_csv("obs_train.csv", index_col=0, parse_dates=True)
            >>> obs_test = pd.read_csv("obs_test.csv", index_col=0, parse_dates=True)
            >>> model.load_country_data_with_year_specific(obs_train, obs_test)
            >>> model.train(False)
        """
        from vwf.loaders import load_year_specific_grid_points
        
        if self.obs_level != "country":
            raise ValueError("load_country_data_with_year_specific() requires obs_level='country'")

        # Ensure index is DatetimeIndex with UTC timezone
        if not isinstance(obs_train.index, pd.DatetimeIndex):
            obs_train.index = pd.to_datetime(obs_train.index, utc=True)
        else:
            obs_train.index = obs_train.index.tz_convert('UTC') if obs_train.index.tz is not None else obs_train.index
            
        if not isinstance(obs_test.index, pd.DatetimeIndex):
            obs_test.index = pd.to_datetime(obs_test.index, utc=True)
        else:
            obs_test.index = obs_test.index.tz_convert('UTC') if obs_test.index.tz is not None else obs_test.index

        # Get unique years from training data
        train_years = sorted(obs_train.index.year.unique())
        print(f"Training years: {train_years}")

        # Load year-specific grid points using the loader
        # base_dir should be input/country_level_data/, not input/country_level_data/grid_points/
        # grid_points_dir is input/country_level_data/grid_points/{country}/
        # So we need .parent.parent to get base_dir
        self.grid_points, self.grid_points_by_year = load_year_specific_grid_points(
            self.country, 
            train_years,
            base_dir=grid_points_dir.parent.parent if grid_points_dir else None
        )

        # Store observations
        self.obs_data_train = obs_train.copy()
        self.obs_data_test = obs_test.copy()
        self._build_country_sources()

        print("\n✓ Loaded country-level data with year-specific grid points:")
        print(f"  Grid points: {len(self.grid_points)} unique points")
        print(f"  Year-specific variants: {len(self.grid_points_by_year)} years")
        if 'cluster' in self.grid_points.columns:
            print(f"  Clusters: {self.grid_points['cluster'].nunique()}")
        print(f"  Training observations: {len(self.obs_data_train)} timesteps")
        print(f"  Test observations: {len(self.obs_data_test)} timesteps")

        return self

    @classmethod
    def from_config(
        cls,
        country_code: str,
        path: str = "",
        config_dir: str = "input/country_level_data",
        **kwargs
    ):
        """Create PyVWF model from generated country-level configuration.

        This is a convenience method that loads configuration and data from
        files generated by scripts/generate_country_level_training_data.py.

        Args:
            country_code: Country code (NL, FR, BE, NO).
            path: Output path for model artifacts (default: "").
            config_dir: Directory containing pyvwf_config.py (default: "input/country_level_data").
            **kwargs: Additional arguments to override config (e.g., calc_z0=False).

        Returns:
            Initialized PyVWF model with loaded grid points and observations.

        Example:
            >>> # Assumes you've run scripts/generate_country_level_training_data.py
            >>> model = PyVWF.from_config("NL")
            >>> model.train(False)
            >>> model.simulate_cf(2021)

        Example with overrides:
            >>> model = PyVWF.from_config("FR", time_res_list=["fixed", "season", "month"])
            >>> model.train(False)
        """
        import sys
        from pathlib import Path as PathLib

        # Add config directory to path
        config_path = PathLib(config_dir)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config directory not found: {config_dir}\n"
                f"Run: python scripts/generate_country_level_training_data.py"
            )

        sys.path.insert(0, str(config_path))

        try:
            from pyvwf_config import get_config
        except ImportError:
            raise ImportError(
                f"Could not import pyvwf_config from {config_dir}\n"
                f"Run: python scripts/generate_country_level_training_data.py"
            )

        # Load configuration
        config = get_config(country_code.upper())

        # Load data files
        grid_points = pd.read_csv(config["grid_points_path"])
        obs_train = pd.read_csv(config["train_obs_path"], index_col=0, parse_dates=True)
        obs_test = pd.read_csv(config["test_obs_path"], index_col=0, parse_dates=True)

        # Create model with config defaults, allowing kwargs to override
        model_kwargs = {
            "path": path,
            "country": config["country"],
            "correct": True,
            "calc_z0": config.get("calc_z0", True),
            "cluster_mode": config.get("cluster_mode", "all"),
            "cluster_list": config.get("cluster_list", [5]),
            "time_res_list": config.get("time_res_list", ["fixed"]),
            "obs_level": "country",
        }
        model_kwargs.update(kwargs)

        # Create model
        model = cls(**model_kwargs)

        # Load country data
        model.load_country_data(grid_points, obs_train, obs_test)

        print(f"\n✓ Initialized {config['name']} model from config")
        print(f"  Using generated data from: {config_dir}/")

        return model

    def train(
        self,
        check=False,
        dask_n_workers=3,
        dask_threads_per_worker=1,
        dask_use_processes=True,
        dask_use_distributed=True,
        dask_npartitions=0,
    ):
        """Derive bias correction factors at desired spatiotemporal resolutions.

        Args:
            check: Unused compatibility flag.
            dask_n_workers: Number of worker processes for distributed offsets.
            dask_threads_per_worker: Threads per worker (use 1 for CPU-bound).
            dask_use_processes: Use processes instead of threads.
            dask_use_distributed: If True, use dask.distributed LocalCluster.
            dask_npartitions: Override Dask partition count (0 to auto).
        """
        # obs_level selects the pipeline branch; source supplies the observations.
        # A None source means "resolve from the country code" (turbine-level).
        gen_cf, turb_info_train, reanalysis, power_curves = train_set(
            self.country,
            self.calc_z0,
            self.cluster_mode,
            add_nan=self.add_nan,
            interp_nan=self.interp_nan,
            fix_turb=self.fix_turb,
            obs_level=self.obs_level,
            source=self.source_train,
        )

        # Store training data for downstream access
        self.gen_cf = gen_cf

        if len(self.cluster_list) < 1:
            print("All correction factors are trained ... Ending train.")
            print("--------------------------------")
            return self

        # For country-level with year-specific grid points: merge year-specific capacity
        if self.obs_level == "country" and hasattr(self, 'grid_points_by_year') and self.grid_points_by_year:
            print("  Merging year-specific grid point capacities...")
            
            # For each year in gen_cf, merge the corresponding year-specific capacity
            year_capacities = []
            for year, grid_pts_year in self.grid_points_by_year.items():
                year_caps = grid_pts_year[['ID', 'capacity']].copy()
                year_caps['year'] = year
                year_capacities.append(year_caps)
            
            if year_capacities:
                year_capacity_df = pd.concat(year_capacities, ignore_index=True)
                
                # Drop the old capacity column and merge year-specific capacity
                gen_cf = gen_cf.drop(columns=['capacity'], errors='ignore')
                gen_cf = gen_cf.merge(year_capacity_df, on=['ID', 'year'], how='left')
                print(f"  ✓ Merged year-specific capacities for {len(self.grid_points_by_year)} years")

        turb_info_train.to_csv(
            self.directory_path
            + "/training/simulated-turbines/"
            + self.country
            + "_train_turb_info.csv",
            index=None,
        )

        # Print training statistics (type column may not exist for country-level)
        if "type" in turb_info_train.columns:
            print(
                "Training on ",
                len(turb_info_train),
                " turbines/farms | ",
                len(turb_info_train[turb_info_train["type"] == "onshore"]),
                " onshore | ",
                len(turb_info_train[turb_info_train["type"] == "offshore"]),
                " offshore",
            )
        else:
            print("Training on ", len(turb_info_train), " turbines/farms")

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

            # Calculate offsets
            if self.obs_level == "turbine":
                # Skip rows with zero or missing observations (can't optimize)
                valid_obs = train_bias_df[train_bias_df['obs'].notna() & (train_bias_df['obs'] > 0)].copy()

                if len(valid_obs) == 0:
                    print("  Warning: No valid observations for offset optimization (all obs=0 or NaN)")
                    train_bias_df['offset'] = 0.0
                else:
                    # parallelisation to find offset
                    def find_offset_parallel(df, clus_info_arg, reanalysis_arg, power_curves_arg):
                        """Compute offsets in parallel for a partition."""
                        return df.apply(
                            correction.find_offset,
                            args=(clus_info_arg, reanalysis_arg, power_curves_arg),
                            axis=1,
                        )

                    if not dask_n_workers or dask_n_workers == 0:
                        # Direct pandas apply (no Dask overhead, avoids serialization issues)
                        print(f"  Computing offsets sequentially for {len(valid_obs)} rows...")
                        valid_obs["offset"] = valid_obs.apply(
                            correction.find_offset,
                            args=(clus_info, reanalysis, power_curves),
                            axis=1,
                        )
                    elif dask_use_distributed:
                        if dask_npartitions and dask_npartitions > 0:
                            npartitions = dask_npartitions
                        else:
                            npartitions = max(dask_n_workers * 4, 1)
                        ddf = dd.from_pandas(valid_obs, npartitions=npartitions)

                        try:
                            from dask.distributed import Client, LocalCluster

                            cluster = LocalCluster(
                                n_workers=dask_n_workers,
                                threads_per_worker=dask_threads_per_worker,
                                processes=dask_use_processes,
                            )
                            client = Client(cluster)
                            try:
                                clus_info_arg = client.scatter(clus_info, broadcast=True)
                                reanalysis_arg = client.scatter(reanalysis, broadcast=True)
                                power_curves_arg = client.scatter(power_curves, broadcast=True)
                                ddf["offset"] = ddf.map_partitions(
                                    find_offset_parallel,
                                    clus_info_arg,
                                    reanalysis_arg,
                                    power_curves_arg,
                                    meta=("offset", "float"),
                                )
                                valid_obs = ddf.compute()
                            finally:
                                client.close()
                                cluster.close()
                        except Exception as exc:
                            print(
                                "Warning: Dask distributed setup failed; falling back to sequential. "
                                f"Details: {exc}"
                            )
                            valid_obs["offset"] = valid_obs.apply(
                                correction.find_offset,
                                args=(clus_info, reanalysis, power_curves),
                                axis=1,
                            )
                    else:
                        # Dask with processes scheduler (no distributed)
                        if dask_npartitions and dask_npartitions > 0:
                            npartitions = dask_npartitions
                        else:
                            npartitions = max(dask_n_workers * 4, 1)
                        ddf = dd.from_pandas(valid_obs, npartitions=npartitions)
                        ddf["offset"] = ddf.map_partitions(
                            find_offset_parallel,
                            clus_info,
                            reanalysis,
                            power_curves,
                            meta=("offset", "float"),
                        )
                        valid_obs = ddf.compute(scheduler="processes")

                    # Merge back with zero-obs and NaN-obs rows
                    zero_obs = train_bias_df[(train_bias_df['obs'] == 0)].copy()
                    zero_obs['offset'] = 0.0
                    nan_obs = train_bias_df[train_bias_df['obs'].isna()].copy()
                    nan_obs['offset'] = np.nan
                    train_bias_df = pd.concat([valid_obs, zero_obs, nan_obs], ignore_index=True).sort_index()
            else:
                # Country-level: optimize offsets for all clusters simultaneously
                print("  Optimizing offsets for country-level data...")

                # Group by year and time_slice
                unique_periods = train_bias_df[['year', time_res]].drop_duplicates()

                offsets_list = []

                for _, period_row in unique_periods.iterrows():
                    year = period_row['year']
                    time_slice = period_row[time_res]

                    # Get observed country CF for this period
                    period_data = train_bias_df[
                        (train_bias_df['year'] == year) &
                        (train_bias_df[time_res] == time_slice)
                    ]

                    obs_country_cf = period_data['obs'].iloc[0]  # Same for all clusters

                    # Get scalars for each cluster
                    scalars_by_cluster = dict(zip(period_data['cluster'], period_data['scalar']))

                    # Optimize offsets for all clusters
                    offsets_dict = correction.find_offsets_country_level(
                        year=year,
                        time_slice=time_slice,
                        obs_country_cf=obs_country_cf,
                        scalars_by_cluster=scalars_by_cluster,
                        turb_info=clus_info,
                        reanalysis=reanalysis,
                        powerCurveFile=power_curves
                    )

                    # Store offsets
                    for cluster_id, offset in offsets_dict.items():
                        offsets_list.append({
                            'year': year,
                            time_res: time_slice,
                            'cluster': cluster_id,
                            'offset': offset
                        })

                # Merge offsets back into train_bias_df
                offsets_df = pd.DataFrame(offsets_list)
                train_bias_df = train_bias_df.drop(columns=['offset'], errors='ignore')
                train_bias_df = train_bias_df.merge(
                    offsets_df,
                    on=['year', time_res, 'cluster'],
                    how='left'
                )

                print(f"  ✓ Optimized offsets for {len(unique_periods)} periods")

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
        """Simulate capacity factor for a test year."""
        # obs_level selects the pipeline branch; source supplies the observations.
        obs_cf, turb_info, reanalysis, power_curves = val_set(
            self.country,
            self.calc_z0,
            self.cluster_mode,
            year_test,
            fix_turb_test,
            obs_level=self.obs_level,
            source=self.source_test,
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

        # Print simulation statistics (type column may not exist for country-level)
        if "type" in turb_info.columns:
            print(
                "Simulating ",
                len(turb_info),
                " turbines/farms | ",
                len(turb_info[turb_info["type"] == "onshore"]),
                " onshore | ",
                len(turb_info[turb_info["type"] == "offshore"]),
                " offshore",
            )
        else:
            print("Simulating ", len(turb_info), " turbines/farms")
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
                out_path = (
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
                if Path(out_path).is_file():
                    print("PyVWF(", num_clu, "--", time_res, ") was previously simulated.\n")
                else:
                    print("Simulating CF using PyVWF(", num_clu, ", ", time_res, ") ...")
                    start_time = time.time()

                    # For country-level data, use existing cluster assignments from training
                    if self.obs_level == "country":
                        # Use the same cluster assignments as training
                        clus_info = turb_info.copy()
                        # Cluster assignments already exist in turb_info from training
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
                    cor_cf.to_csv(out_path, index=None)

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("Completed and saved. Elapsed time: {:.2f} seconds".format(elapsed_time))
                    print(" ")

        self.turb_info = turb_info
        self.year_test = year_test

        return self