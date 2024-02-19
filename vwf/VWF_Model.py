import pandas as pd
import numpy as np
import xarray as xr
import dask.dataframe as dd

from pathlib import Path
import time

from vwf.preprocessing import (
    prep_era5,
    prep_obs,
    merge_gen_cf,
)
from vwf.bias_correction import (
    cluster_turbines,
    train_data,
    find_offset,
    closest_cluster,
    format_bc_factors
)

from vwf.simulation import (
    simulate_wind,
)

pd.options.mode.chained_assignment = None  # default='warn'

class VWF():
    """
    This class trains and creates the virtual wind farm model.
    
    Attributes:
        country (str): country code e.g. Denmark "DK"
    """
    def __init__(self, country, cluster_list, time_res_list):
        self.country = country
        self.cluster_list = cluster_list
        self.time_res_list = time_res_list
      
        
    def train(self):
        """
        Training of the VWF model.
        
        Saves the turbine metadata used in training, the clustered turbine
        metadata and the correction factors.

        Args:
            cluster_list (list): list of number of clusters
            time_res_listlist (list): list of time resolutions
        """
        powerCurveFileLoc = 'data/input/power_curves.csv'
        powerCurveFile = pd.read_csv(powerCurveFileLoc)
        era5 = prep_era5(True)
        obs_cf, turb_info = prep_obs(self.country, True)
        gen_cf = merge_gen_cf(era5, obs_cf, turb_info, powerCurveFile)
        
        # for research purpose
        gen_cf.to_csv('data/training/'+self.country+'_train_gen_cf.csv', index = None)
        
        turb_info.to_csv('data/training/simulated-turbines/'+self.country+'_train_turb_info.csv', index = None)
        obs_cf.to_csv('data/results/capacity-factor/'+self.country+'_train_obs_cf.csv', index = None)
        
        for num_clu in self.cluster_list:
                        # sim_cf, xr_obs_cf, clus_info = prep_clusters(num_clu, turb_info, obs_cf, era5, powerCurveFile)
            clus_info = cluster_turbines(num_clu, turb_info)
            clus_info.to_csv('data/training/simulated-turbines/'+self.country+'_clus_info_'+str(num_clu)+'.csv', index = None)
        
            for time_res in self.time_res_list:
                # checking if correction factors already exist
                my_file = Path('data/training/correction-factors/'+self.country+'_factors_'+time_res+'_'+str(num_clu)+'.csv')
                if my_file.is_file():
                    print("Training for ", num_clu, " clusters with time resolution: ", time_res, " is already trained!")
                    print(" ")
                
                else:
                    print("Training for ", num_clu, " clusters with time resolution: ", time_res, " is taking place.")
                    start_time = time.time()
                    
                    bias_data = train_data(time_res, gen_cf, clus_info)
                    # bias_data = prep_training(num_clu, time_res, clus_info, obs_cf, era5, powerCurveFile)
                    # bias_data = prep_time_res(time_res, sim_cf, xr_obs_cf)
                    
                    ddf = dd.from_pandas(bias_data, npartitions=40)
                    
                    def find_offset_parallel(df):
                        return df.apply(find_offset, args=(clus_info, era5, powerCurveFile), axis=1)
                        
                    ddf["offset"] = ddf.map_partitions(find_offset_parallel, meta=('offset', 'float'))
                    ddf.to_csv(
                        'data/training/correction-factors/'+self.country+'_factors_'+time_res+'_'+str(num_clu)+'.csv', 
                        single_file=True, 
                        compute_kwargs={'scheduler':'processes'}
                    )
                    
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("Training completed and saved. Elapsed time: {:.2f} seconds".format(elapsed_time))
                    print(" ")
        return self

    
    def test(self, year_test):
        """
        Simulating capacity factor using the trained correction factors.
        
        Saves the simulated wind speeds and capacity factors, both corrected
        and uncorrected.

        Args:
            year_test (int): year for the model to simulate
            cluster_list (list): list of number of clusters
            time_res_listlist (list): list of time resolutions
        """
        powerCurveFileLoc = 'data/input/power_curves.csv'
        powerCurveFile = pd.read_csv(powerCurveFileLoc)
        era5 = prep_era5()
        obs_cf, turb_info = prep_obs(self.country, False, year_test)
        obs_cf.to_csv('data/results/capacity-factor/'+self.country+"_"+str(year_test)+'_obs_cf.csv', index = None)
        turb_info.to_csv('data/training/simulated-turbines/'+self.country+'_'+str(year_test)+'_turb_info.csv', index = None)
        
        # simulate uncorrected wind
        unc_ws, unc_cf = simulate_wind(era5, turb_info, powerCurveFile)
        # unc_ws.to_csv('data/results/wind-speed/'+self.country+"_"+str(year_test)+'_unc_ws.csv', index = None)
        unc_cf.to_csv('data/results/capacity-factor/'+self.country+"_"+str(year_test)+'_unc_cf.csv', index = None)
            
        for num_clu in self.cluster_list:
            clus_info = pd.read_csv('data/training/simulated-turbines/'+self.country+'_clus_info_'+str(num_clu)+'.csv')
            clus_info = closest_cluster(clus_info, turb_info)
        
            for time_res in self.time_res_list:
                print("Test for ", num_clu, " clusters with time resolution: ", time_res, " is taking place.")
                start_time = time.time()
                
                bc_factors = format_bc_factors(time_res, num_clu, self.country)
        
                # simulate corrected wind
                cor_ws, cor_cf = simulate_wind(era5, clus_info, powerCurveFile, bc_factors, time_res)
                # cor_ws.to_csv('data/results/wind-speed/'+self.country+"_"+str(year_test)+'_'+time_res+'_'+str(num_clu)+'_cor_ws.csv', index = None)
                cor_cf.to_csv('data/results/capacity-factor/'+self.country+"_"+str(year_test)+'_'+time_res+'_'+str(num_clu)+'_cor_cf.csv', index = None)
        
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("Results completed and saved. Elapsed time: {:.2f} seconds".format(elapsed_time))
                print(" ")
        
        return self
        
    def train_check(self):
        """
        Simulating capacity factor using the trained correction factors.
        
        Saves the simulated wind speeds and capacity factors, both corrected
        and uncorrected.

        Args:
            year_test (int): year for the model to simulate
            cluster_list (list): list of number of clusters
            time_res_listlist (list): list of time resolutions
        """
        powerCurveFileLoc = 'data/input/power_curves.csv'
        powerCurveFile = pd.read_csv(powerCurveFileLoc)
        era5 = prep_era5(True)
        obs_cf, turb_info = prep_obs(self.country, True)
        
        # turb_info.to_csv('data/training/simulated-turbines/'+self.country+'_'+str(year_test)+'_turb_info.csv', index = None)
        
        # simulate uncorrected wind
        unc_ws, unc_cf = simulate_wind(era5, turb_info, powerCurveFile)
        unc_cf.to_csv('data/results/capacity-factor/'+self.country+'_train_unc_cf.csv', index = None)
            
        for num_clu in self.cluster_list:
            clus_info = pd.read_csv('data/training/simulated-turbines/'+self.country+'_clus_info_'+str(num_clu)+'.csv')
            clus_info = closest_cluster(clus_info, turb_info)
        
            for time_res in self.time_res_list:
                print("Test for ", num_clu, " clusters with time resolution: ", time_res, " is taking place.")
                start_time = time.time()
                
                bc_factors = format_bc_factors(time_res, num_clu, self.country)
        
                # simulate corrected wind
                cor_ws, cor_cf = simulate_wind(era5, clus_info, powerCurveFile, bc_factors, time_res)
                cor_cf.to_csv('data/results/capacity-factor/'+self.country+'_train_'+time_res+'_'+str(num_clu)+'_cor_cf.csv', index = None)
        
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("Results completed and saved. Elapsed time: {:.2f} seconds".format(elapsed_time))
                print(" ")
        
        return self
    