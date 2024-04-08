import pandas as pd
import numpy as np
import xarray as xr
import dask.dataframe as dd

from pathlib import Path
import time
import os

from vwf.preprocessing import (
    prep_era5,
    prep_merra2,
    prep_obs,
    merge_gen_cf,
)
from vwf.bias_correction import (
    cluster_turbines,
    train_data,
    find_offset,
    format_bc_factors
)

from vwf.simulation import simulate_wind

pd.options.mode.chained_assignment = None  # default='warn'

class VWF():
    """
    This class trains and creates the virtual wind farm model.
    
    Attributes:
        country (str): country code e.g. Denmark "DK"
        cluster_list (int): list of the spatial resolutions for the model
        time_res_list (str): list of the temporal resolutions for the model
        cluster_mode (str): 
            'all' forms clusters mixing onshore and offshore,
            'onshore' forms clusters with onshore while fixing all offshore to 1 cluster,
            'offshore' forms clusters with offshore while fixing all onshore to 1 cluster,          
        add_nan (float): percentage of data to randomly remove from training data 0 < add_nan < 1
        interp_nan (float): set limit on simultaneous missing data points when interpolating nan 
        directory_path (string): where files will be saved
    """
    def __init__(self, country, cluster_list, time_res_list, cluster_mode, add_nan=None, interp_nan=None, fix_turb=None):
        # creating folders
        run = country+'-'+cluster_mode
        if (add_nan == None) & (interp_nan == None) & (fix_turb == None):
            # run += '-standard-scalar-equation'
            run += '-standard'
        else:
            if add_nan != None:
                run += '-r'+str(add_nan)
            if interp_nan != None:
                run += '-i'+str(interp_nan)
            if fix_turb != None:
                run += '-'+fix_turb
        
        # Specify where to make the new directory path
        directory_path = os.path.join('run',run)
        # Define a list of folders to make in that directory
        folder_names = [
            'training/correction-factors',
            'training/simulated-turbines',
            'results/capacity-factor',
            'results/wind-speed',
            'plots'
        ]
        print(f"Creating new directories in '{directory_path}':")
        # Go through the folder names and attempt to make each. Ignore any errors
        # due to existing directories or troublesome paths.
        for folder_name in folder_names:
            path = os.path.join(directory_path, folder_name)
            try:
                os.makedirs(path)
            except OSError:
                pass  # We suppress any error and instead continue with the list
            else:
                print(f"Created {path}")
        
        # setting attributes
        self.country = country
        self.cluster_list = cluster_list
        self.time_res_list = time_res_list
        self.add_nan = add_nan
        self.interp_nan = interp_nan
        self.fix_turb = fix_turb
        self.directory_path = directory_path
        self.cluster_mode = cluster_mode
      
        
    def train(self, check=False): 
        """
        Training of the VWF model.
        
        Saves the turbine metadata used in training, the clustered turbine
        metadata and the correction factors.
        """
        # load and preprocess input data
        powerCurveFileLoc = 'input/power_curves.csv'
        powerCurveFile = pd.read_csv(powerCurveFileLoc)
        era5 = prep_era5(self.country, True)
        obs_cf, turb_info = prep_obs(self.country, self.cluster_mode, add_nan=self.add_nan, interp_nan=self.interp_nan, fix_turb=self.fix_turb)
        print("Training on ", len(turb_info), " turbines/farms | ", len(turb_info[turb_info['type'] == 'onshore']), " onshore | ", len(turb_info[turb_info['type'] == 'offshore']), " offshore")
        gen_cf = merge_gen_cf(era5, obs_cf, turb_info, powerCurveFile)
        
        # for research purpose
        gen_cf.to_csv(self.directory_path+'/training/'+self.country+'_train_gen_cf.csv', index = None)
        turb_info.to_csv(self.directory_path+'/training/simulated-turbines/'+self.country+'_train_turb_info.csv', index = None)
        obs_cf.to_csv(self.directory_path+'/results/capacity-factor/'+self.country+'_train_obs_cf.csv', index = None)
        
        for num_clu in self.cluster_list:
            clus_info = cluster_turbines(num_clu, turb_info, True)
            clus_info.to_csv(self.directory_path+'/training/simulated-turbines/'+self.country+'_clus_info_'+str(num_clu)+'.csv', index = None)
        
            for time_res in self.time_res_list:
                # checking if bc factors have been derived already
                my_file = Path(self.directory_path+'/training/correction-factors/'+self.country+'_factors_'+time_res+'_'+str(num_clu)+'.csv')
                if my_file.is_file():
                    print("PyVWF(",num_clu,"--",time_res,") was previously trained.\n")
                    bc_factors = pd.read_csv(self.directory_path+'/training/correction-factors/'+self.country+'_factors_'+time_res+'_'+str(num_clu)+'.csv')
                else:
                    print("Deriving correction factors for PyVWF(",num_clu,",",time_res,") ...")
                    start_time = time.time()
                    # creating the dataframe that will contain the resulted bc factors
                    bias_data = train_data(time_res, gen_cf, clus_info)
                    
                    # parellisation to find offset
                    def find_offset_parallel(df):
                        return df.apply(find_offset, args=(clus_info, era5, powerCurveFile), axis=1)
                    ddf = dd.from_pandas(bias_data, npartitions=40)
                    ddf["offset"] = ddf.map_partitions(find_offset_parallel, meta=('offset', 'float'))

                    bias_data = ddf.compute(scheduler='processes')
                    bc_factors = format_bc_factors(bias_data, time_res)
                    bc_factors.to_csv(self.directory_path+'/training/correction-factors/'+self.country+'_factors_'+time_res+'_'+str(num_clu)+'.csv')
                    
                    if check == True:
                        # simulate corrected wind
                        cor_ws, cor_cf = simulate_wind(era5, clus_info, powerCurveFile, bc_factors, time_res)
                        cor_cf.to_csv(self.directory_path+'/results/capacity-factor/'+self.country+'_train_'+time_res+'_'+str(num_clu)+'_cor_cf.csv', index = None)
                    
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("Completed and saved. Elapsed time: {:.2f} seconds\n".format(elapsed_time))
                    
        #         if check == True:
        #             # simulate corrected wind
        #             cor_ws, cor_cf = simulate_wind(era5, clus_info, powerCurveFile, bc_factors, time_res)
        #             cor_cf.to_csv(self.directory_path+'/results/capacity-factor/'+self.country+'_train_'+time_res+'_'+str(num_clu)+'_cor_cf.csv', index = None)
                    
        if check == True:
            unc_ws, unc_cf = simulate_wind(era5, turb_info, powerCurveFile)
            unc_cf.to_csv(self.directory_path+'/results/capacity-factor/'+self.country+'_train_unc_cf.csv', index = None)
            
        return self
        
    
    def simulate_cf(self, year_test, mode='all', fix_turb_test=None):
        """
        Simulating capacity factor using the defined model.
        
        Saves the simulated wind speeds and capacity factors, both corrected
        and uncorrected.

        Args:
            year_test (int): year for the model to simulate
            fix_turb_test (list): fixing a single turbine model to be simulated
        """
        # load and preprocess input data
        powerCurveFileLoc = 'input/power_curves.csv'
        powerCurveFile = pd.read_csv(powerCurveFileLoc)
        era5 = prep_era5(self.country)
        obs_cf, turb_info = prep_obs(self.country, mode, year_test, fix_turb=fix_turb_test)
        turb_info_train = pd.read_csv(self.directory_path+'/training/simulated-turbines/'+self.country+'_train_turb_info.csv')
        print("Simulating ", len(turb_info), " turbines/farms | ", len(turb_info[turb_info['type'] == 'onshore']), " onshore | ", len(turb_info[turb_info['type'] == 'offshore']), " offshore")
        
        # for research purpose
        obs_cf.to_csv(self.directory_path+'/results/capacity-factor/'+self.country+"_"+str(year_test)+'_obs_cf.csv', index = None)
        turb_info.to_csv(self.directory_path+'/training/simulated-turbines/'+self.country+'_'+str(year_test)+'_turb_info.csv', index = None)
        
        # simulate uncorrected wind
        unc_ws, unc_cf = simulate_wind(era5, turb_info, powerCurveFile)
        # unc_ws.to_csv(self.directory_path+'results/wind-speed/'+self.country+"_"+str(year_test)+'_unc_ws.csv', index = None)
        unc_cf.to_csv(self.directory_path+'/results/capacity-factor/'+self.country+"_"+str(year_test)+'_unc_cf.csv', index = None)
            
        for num_clu in self.cluster_list:
            clus_info = cluster_turbines(num_clu, turb_info_train, False, turb_info)
        
            for time_res in self.time_res_list:
                print("Simulating CF using PyVWF(", num_clu, ", ", time_res, ") ...")
                start_time = time.time()
                
                # loading and formatting bc factors
                bc_factors = pd.read_csv(self.directory_path+'/training/correction-factors/'+self.country+'_factors_'+time_res+'_'+str(num_clu)+'.csv')
                
                # simulate corrected wind
                cor_ws, cor_cf = simulate_wind(era5, clus_info, powerCurveFile, bc_factors, time_res)
                # cor_ws.to_csv(self.directory_path+'results/wind-speed/'+self.country+"_"+str(year_test)+'_'+time_res+'_'+str(num_clu)+'_cor_ws.csv', index = None)
                cor_cf.to_csv(self.directory_path+'/results/capacity-factor/'+self.country+"_"+str(year_test)+'_'+time_res+'_'+str(num_clu)+'_cor_cf.csv', index = None)
        
                end_time = time.time()
                elapsed_time = end_time - start_time
                print("Completed and saved. Elapsed time: {:.2f} seconds".format(elapsed_time))
                print(" ")
        
        return self
        
    def simulate_cf_merra2(self, year_test):
        """
        Simulating capacity factor using pretrained merra-2 factors
        
        Saves the simulated wind speeds and capacity factors, both corrected
        and uncorrected.

        Args:
            year_test (int): year for the model to simulate
            fix_turb_test (list): fixing a single turbine model to be simulated
        """
        # load and preprocess input data
        powerCurveFileLoc = 'input/power_curves.csv'
        powerCurveFile = pd.read_csv(powerCurveFileLoc)
        merra2 = prep_merra2(self.country)
        obs_cf, turb_info = prep_obs(self.country, year_test)
        
        # for research purpose
        obs_cf.to_csv(self.directory_path+'/results/capacity-factor/'+self.country+"_"+str(year_test)+'_obs_cf.csv', index = None)
        turb_info.to_csv(self.directory_path+'/training/simulated-turbines/'+self.country+'_'+str(year_test)+'_turb_info.csv', index = None)
        
        # merra-2 factors were trained on country-scale and no time dependency
        turb_info['cluster'] = 0
        bias_fact_dict = {'0': [0 , '1/1', 0.597, 2.836]}
        bc_factors = pd.DataFrame.from_dict(bias_fact_dict, orient='index',
                                    columns=['cluster','fixed', 'scalar', 'offset'])

        
        print("Simulating CF using Merra-2 ...")
        start_time = time.time()
        
        # simulate uncorrected wind
        unc_ws, unc_cf = simulate_wind(merra2, turb_info, powerCurveFile)
        unc_cf.to_csv(self.directory_path+'/results/capacity-factor/'+self.country+"_"+str(year_test)+'_merra2_unc_cf.csv', index = None)
        # simulate corrected wind
        cor_ws, cor_cf = simulate_wind(merra2, turb_info, powerCurveFile, bc_factors, 'fixed')
        cor_cf.to_csv(self.directory_path+'/results/capacity-factor/'+self.country+"_"+str(year_test)+'_merra2_cor_cf.csv', index = None)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Completed and saved. Elapsed time: {:.2f} seconds".format(elapsed_time))
        print(" ")
        return self